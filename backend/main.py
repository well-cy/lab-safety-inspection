# -*- coding: utf-8 -*-
"""
Web 后端：FastAPI。

API 概览：
  GET  /                      前端页面
  GET  /api/model/info        模型信息
  GET  /api/dashboard         今日统计
  GET  /api/statistics        历史统计
  GET  /api/violations        违规记录查询（支持筛选）
  POST /api/detect/image      上传图片 → 检测 + 违规判断
  POST /api/detect/video      上传视频 → 检测 + 违规判断
  GET  /api/settings          实验室/区域/规则配置
  POST /api/settings/lab      新建实验室
  POST /api/settings/area     新建区域（含 PPE 规则）
  DELETE /api/settings/area   删除区域
"""
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend import database as db
from rule_engine.engine import SEVERITY_NONE

ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = ROOT / "data" / "test" / "uploads"
OUT_IMG_DIR = ROOT / "outputs" / "annotated"
OUT_VID_DIR = ROOT / "outputs" / "videos"
for d in (UPLOAD_DIR, OUT_IMG_DIR, OUT_VID_DIR):
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="实验室安全智能巡检与违规预警系统", version="0.1.0-MVP")

_detector = None
_pipeline = None


def get_pipeline():
    """懒加载模型管线（首次请求时加载，避免服务启动卡顿）"""
    global _detector, _pipeline
    if _pipeline is None:
        from ai.detector import PPEDetector
        from video.processor import InspectionPipeline
        _detector = PPEDetector(ROOT / "ai" / "model" / "sh17_yolov8s.pt")
        from rule_engine.engine import SafetyRuleEngine
        _pipeline = InspectionPipeline(
            detector=_detector, engine=SafetyRuleEngine())
    return _pipeline


@app.on_event("startup")
def startup():
    db.init_db(seed=True)


# ---------- 静态资源 ----------
app.mount("/static", StaticFiles(directory=ROOT / "backend" / "static"), name="static")
app.mount("/media", StaticFiles(directory=ROOT / "outputs"), name="media")


@app.get("/")
def index():
    return FileResponse(ROOT / "backend" / "static" / "index.html")


# ---------- 模型信息 ----------
@app.get("/api/model/info")
def model_info():
    p = get_pipeline()
    det = p.detector
    return {
        "model_file": Path(det.weights_path).name,
        "base": "SH17 dataset (official YOLOv8s weights, ultralytics)",
        "device": det.device_name,
        "conf_threshold": det.conf_threshold,
        "raw_classes": det.raw_names,
        "system_classes": ["person", "mask", "gloves", "lab_coat", "goggles", "helmet"],
    }


# ---------- 统计 ----------
@app.get("/api/dashboard")
def dashboard():
    return db.dashboard_stats()


@app.get("/api/statistics")
def statistics():
    return db.statistics()


@app.get("/api/violations")
def violations(vtype: str | None = None, severity: str | None = None,
               date: str | None = None, limit: int = 200, offset: int = 0):
    rows = db.query_violations(vtype=vtype, severity=severity, date_str=date,
                               limit=limit, offset=offset)
    for r in rows:
        r["screenshot_url"] = f"/media/screenshots/{r['screenshot_path']}" if r.get("screenshot_path") else ""
    return {"total": len(rows), "items": rows}


# ---------- 检测：图片 ----------
@app.post("/api/detect/image")
async def detect_image(file: UploadFile = File(...), lab_id: int = Form(1)):
    pipeline = get_pipeline()
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    save_path = UPLOAD_DIR / f"img_{uuid.uuid4().hex[:8]}{suffix}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    areas = db.load_areas(lab_id)
    from video.processor import process_image
    out_name = f"annotated_{save_path.stem}.jpg"
    result = process_image(pipeline, save_path, areas,
                           save_path=OUT_IMG_DIR / out_name)

    persons = result["person_states"]
    events = result["events"]
    viol = [e for e in events if e["severity"] != SEVERITY_NONE]
    normal = [e for e in events if e["severity"] == SEVERITY_NONE]
    record_id = db.save_detection_record(
        lab_id, "image", file.filename, len(persons), len(viol), len(normal),
        result["elapsed_ms"])
    # 图片模式：正常评估也入库（供 dashboard normal_count 使用）
    from rule_engine.engine import ViolationEvent
    ev_objs = []
    for e in events:
        ev_objs.append(ViolationEvent(
            person_id=e["person_id"], area_name=e["area"],
            violation_types=e["violation_types"], missing_ppe=[],
            severity=e["severity"], timestamp=e["timestamp"],
            frame_time=0, screenshot=e.get("screenshot", "")))
    db.save_violation_events(record_id, lab_id, ev_objs, include_normal=True)

    return {
        "record_id": record_id,
        "elapsed_ms": result["elapsed_ms"],
        "person_count": len(persons),
        "persons": persons,
        "events": events,
        "violation_count": len(viol),
        "annotated_url": f"/media/annotated/{out_name}",
        "detections": result["detections"],
    }


# ---------- 检测：视频 ----------
@app.post("/api/detect/video")
async def detect_video(file: UploadFile = File(...), lab_id: int = Form(1),
                       stride: int = Form(2)):
    pipeline = get_pipeline()
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    save_path = UPLOAD_DIR / f"vid_{uuid.uuid4().hex[:8]}{suffix}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    areas = db.load_areas(lab_id)
    from video.processor import process_video
    out_name = f"annotated_{save_path.stem}.mp4"
    result = process_video(pipeline, save_path, areas,
                           OUT_VID_DIR / out_name, stride=stride)

    events = result["events"]
    record_id = db.save_detection_record(
        lab_id, "video", file.filename, 0, len(events), 0,
        result["elapsed_s"] * 1000)
    db.save_event_dicts(record_id, lab_id, events)

    return {
        "record_id": record_id,
        "total_frames": result["total_frames"],
        "inferred_frames": result["inferred_frames"],
        "video_fps": result["video_fps"],
        "elapsed_s": result["elapsed_s"],
        "violation_count": len(events),
        "events": events,
        "output_video_url": f"/media/videos/{out_name}",
    }


# ---------- 设置 ----------
@app.get("/api/settings")
def get_settings():
    conn = db.get_conn()
    labs = [dict(r) for r in conn.execute("SELECT * FROM laboratories").fetchall()]
    for lab in labs:
        lab["areas"] = [dict(r) for r in conn.execute(
            "SELECT * FROM areas WHERE lab_id=?", (lab["id"],)).fetchall()]
        for a in lab["areas"]:
            a["required_ppe"] = [r["ppe_type"] for r in conn.execute(
                "SELECT ppe_type FROM safety_rules WHERE area_id=? AND required=1",
                (a["id"],)).fetchall()]
    conn.close()
    return {"labs": labs}


@app.post("/api/settings/lab")
async def create_lab(name: str = Form(...), description: str = Form("")):
    conn = db.get_conn()
    try:
        conn.execute("INSERT INTO laboratories(name, description) VALUES(?,?)",
                     (name, description))
        conn.commit()
    except Exception:
        return JSONResponse({"error": "实验室名称已存在"}, status_code=400)
    finally:
        conn.close()
    return {"ok": True}


@app.post("/api/settings/area")
async def create_area(lab_id: int = Form(...), name: str = Form(...),
                      x1: float = Form(...), y1: float = Form(...),
                      x2: float = Form(...), y2: float = Form(...),
                      required_ppe: str = Form("mask,gloves,lab_coat")):
    conn = db.get_conn()
    cur = conn.execute(
        "INSERT INTO areas(lab_id, name, x1, y1, x2, y2) VALUES(?,?,?,?,?,?)",
        (lab_id, name, x1, y1, x2, y2))
    area_id = cur.lastrowid
    for ppe in [s.strip() for s in required_ppe.split(",") if s.strip()]:
        conn.execute("INSERT INTO safety_rules(area_id, ppe_type, required)"
                     " VALUES(?,?,1)", (area_id, ppe))
    conn.commit()
    conn.close()
    return {"ok": True, "area_id": area_id}


@app.delete("/api/settings/area/{area_id}")
def delete_area(area_id: int):
    conn = db.get_conn()
    conn.execute("DELETE FROM safety_rules WHERE area_id=?", (area_id,))
    conn.execute("DELETE FROM areas WHERE id=?", (area_id,))
    conn.commit()
    conn.close()
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
