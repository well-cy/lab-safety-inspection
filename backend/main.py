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

# 上传与配置校验白名单（防止无效输入触发 500）
ALLOWED_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VID_EXT = {".mp4", ".avi", ".mov", ".mkv"}
ALLOWED_PPE = {"mask", "gloves", "lab_coat", "goggles", "helmet"}

app = FastAPI(title="实验室安全智能巡检与违规预警系统", version="0.1.0-MVP")


@app.exception_handler(Exception)
async def _unhandled_exception(request, exc):
    """兜底：任何未预期异常返回结构化 JSON，避免前端拿到非 JSON 的 500 文本"""
    import traceback
    traceback.print_exc()
    return JSONResponse(
        {"error": f"服务器内部错误（{type(exc).__name__}）：{exc}"}, status_code=500)


# ---------- 输入校验辅助 ----------
def _lab_exists(lab_id: int) -> bool:
    conn = db.get_conn()
    try:
        return conn.execute("SELECT 1 FROM laboratories WHERE id=?",
                            (lab_id,)).fetchone() is not None
    finally:
        conn.close()


def _save_upload(file: UploadFile, save_path: Path) -> int:
    """流式落盘，返回字节数"""
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return save_path.stat().st_size


def _safe_unlink(path: Path):
    try:
        path.unlink()
    except OSError:
        pass


def _readable_image(path: Path) -> bool:
    """能否作为图片解码（拒绝伪装扩展名的文本/空文件）"""
    try:
        import cv2
        import numpy as np
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return False
        return cv2.imdecode(data, cv2.IMREAD_COLOR) is not None
    except Exception:
        return False


def _readable_video(path: Path) -> bool:
    """能否作为视频打开且有可读帧"""
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        ok = cap.isOpened()
        if ok:
            ret, _ = cap.read()
            ok = bool(ret)
        cap.release()
        return ok
    except Exception:
        return False

_detector = None
_pipeline = None        # 视频管线（快速，无切片复检）
_pipeline_img = None    # 图片管线（开启 2×2 切片复检，提升小目标召回）


def get_pipeline():
    """懒加载模型管线（首次请求时加载，避免服务启动卡顿）"""
    global _detector, _pipeline, _pipeline_img
    if _pipeline is None:
        from ai.detector import PPEDetector, PersonDetector
        from video.processor import InspectionPipeline
        # 模型选择：优先加载自制 4 类模型（labppe_v8s.pt），没有则回退 SH17 官方权重
        _custom = ROOT / "ai" / "model" / "labppe_v8s.pt"
        _fallback = ROOT / "ai" / "model" / "sh17_yolov8s.pt"
        _detector = PPEDetector(_custom if _custom.exists() else _fallback, imgsz=1280)
        # 人员锚点：COCO 预训练模型（自带 person 类），用于多人分别判定；
        # 缺文件时自动降级为虚拟聚合人员（原行为），不影响服务可用性
        _person_w = ROOT / "ai" / "model" / "yolov8n_coco.pt"
        _person_det = PersonDetector(_person_w) if _person_w.exists() else None
        from rule_engine.engine import SafetyRuleEngine
        _pipeline = InspectionPipeline(
            detector=_detector, engine=SafetyRuleEngine(), tile_recheck=False,
            person_detector=_person_det)
        _pipeline_img = InspectionPipeline(
            detector=_detector, engine=SafetyRuleEngine(), tile_recheck=True,
            person_detector=_person_det)
    return _pipeline


def get_image_pipeline():
    """图片检测专用管线（含切片复检）"""
    get_pipeline()
    return _pipeline_img


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
        "base": ("Lab-PPE 自制模型 (ultralytics)"
                 if Path(det.weights_path).name == "labppe_v8s.pt"
                 else "SH17 dataset (official YOLOv8s weights, ultralytics)"),
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
    suffix = (Path(file.filename or "upload.jpg").suffix or ".jpg").lower()
    if suffix not in ALLOWED_IMG_EXT:
        return JSONResponse(
            {"error": f"不支持的图片格式 {suffix}，请上传 jpg / jpeg / png / bmp / webp"},
            status_code=400)
    if not _lab_exists(lab_id):
        return JSONResponse(
            {"error": f"实验室 id={lab_id} 不存在，请先在“系统设置”中创建实验室"},
            status_code=400)

    pipeline = get_image_pipeline()
    save_path = UPLOAD_DIR / f"img_{uuid.uuid4().hex[:8]}{suffix}"
    size = _save_upload(file, save_path)
    if size == 0:
        _safe_unlink(save_path)
        return JSONResponse({"error": "上传文件为空，请选择有效的图片文件"}, status_code=400)
    if not _readable_image(save_path):
        _safe_unlink(save_path)
        return JSONResponse({"error": "无法解析该文件为图片，请确认上传的是有效图片"}, status_code=400)

    areas = db.load_areas(lab_id)
    from video.processor import process_image
    out_name = f"annotated_{save_path.stem}.jpg"
    try:
        result = process_image(pipeline, save_path, areas,
                               save_path=OUT_IMG_DIR / out_name)
    except ValueError as e:
        _safe_unlink(save_path)
        return JSONResponse({"error": f"图片处理失败：{e}"}, status_code=400)

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
    suffix = (Path(file.filename or "upload.mp4").suffix or ".mp4").lower()
    if suffix not in ALLOWED_VID_EXT:
        return JSONResponse(
            {"error": f"不支持的视频格式 {suffix}，请上传 mp4 / avi / mov / mkv"},
            status_code=400)
    if stride < 1:
        return JSONResponse({"error": "抽帧步长 stride 必须 ≥ 1"}, status_code=400)
    if not _lab_exists(lab_id):
        return JSONResponse(
            {"error": f"实验室 id={lab_id} 不存在，请先在“系统设置”中创建实验室"},
            status_code=400)

    pipeline = get_pipeline()
    save_path = UPLOAD_DIR / f"vid_{uuid.uuid4().hex[:8]}{suffix}"
    size = _save_upload(file, save_path)
    if size == 0:
        _safe_unlink(save_path)
        return JSONResponse({"error": "上传文件为空，请选择有效的视频文件"}, status_code=400)
    if not _readable_video(save_path):
        _safe_unlink(save_path)
        return JSONResponse({"error": "无法打开该视频（无有效视频帧），请确认文件未损坏"}, status_code=400)

    areas = db.load_areas(lab_id)
    from video.processor import process_video
    out_name = f"annotated_{save_path.stem}.mp4"
    try:
        result = process_video(pipeline, save_path, areas,
                               OUT_VID_DIR / out_name, stride=stride)
    except ValueError as e:
        _safe_unlink(save_path)
        return JSONResponse({"error": f"视频处理失败：{e}"}, status_code=400)

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
    if not name.strip():
        return JSONResponse({"error": "实验室名称不能为空"}, status_code=400)
    conn = db.get_conn()
    try:
        conn.execute("INSERT INTO laboratories(name, description) VALUES(?,?)",
                     (name.strip(), description))
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
    # ---- 输入校验（ROI 越界 / 反序 / 非法 PPE 类别 / 实验室不存在）----
    if not name.strip():
        return JSONResponse({"error": "区域名称不能为空"}, status_code=400)
    for nm, v in (("x1", x1), ("y1", y1), ("x2", x2), ("y2", y2)):
        if not (0.0 <= v <= 1.0):
            return JSONResponse(
                {"error": f"坐标 {nm}={v} 超出 0~1 归一化范围"}, status_code=400)
    if x2 <= x1 or y2 <= y1:
        return JSONResponse(
            {"error": "ROI 需满足 x2>x1 且 y2>y1（按左上角→右下角顺序填写）"},
            status_code=400)
    ppe_list = [s.strip() for s in required_ppe.split(",") if s.strip()]
    bad = [p for p in ppe_list if p not in ALLOWED_PPE]
    if bad:
        return JSONResponse(
            {"error": f"不支持的 PPE 类别：{', '.join(bad)}"
                      f"（可选：mask / gloves / lab_coat / goggles / helmet）"},
            status_code=400)
    if not _lab_exists(lab_id):
        return JSONResponse({"error": f"实验室 id={lab_id} 不存在"}, status_code=400)

    conn = db.get_conn()
    try:
        cur = conn.execute(
            "INSERT INTO areas(lab_id, name, x1, y1, x2, y2) VALUES(?,?,?,?,?,?)",
            (lab_id, name.strip(), x1, y1, x2, y2))
        area_id = cur.lastrowid
        for ppe in ppe_list:
            conn.execute("INSERT INTO safety_rules(area_id, ppe_type, required)"
                         " VALUES(?,?,1)", (area_id, ppe))
        conn.commit()
    finally:
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
