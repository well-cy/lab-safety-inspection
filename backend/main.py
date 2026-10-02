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
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend import database as db
from backend import export as export_api
from backend.config import config
from backend.validators import find_invalid_date
from rule_engine.engine import SEVERITY_NONE

# 路径与阈值统一来自 backend/config.py（支持 LABSAFETY_* 环境变量覆盖）
ROOT = config.ROOT
UPLOAD_DIR = config.UPLOAD_DIR
OUT_IMG_DIR = config.ANNOTATED_DIR
OUT_VID_DIR = config.VIDEO_DIR
config.ensure_dirs()

app = FastAPI(title="实验室安全智能巡检与违规预警系统", version="0.1.0-MVP")

# 可作为区域 PPE 要求的业务类别（person 是检测主体，不是 PPE）
ALLOWED_PPE = config.ALLOWED_PPE


def bad_request(msg: str) -> JSONResponse:
    """统一 4xx 响应：{"error": "..."}（与前端 apiFetch 的错误解析格式一致）"""
    return JSONResponse({"error": msg}, status_code=400)

class ModelUnavailableError(RuntimeError):
    """模型权重缺失或加载失败（模型不入 git 仓库，新成员首次运行常遇到）"""


_detector = None
_pipeline = None

MODEL_PATH = config.MODEL_PATH
MODEL_DOWNLOAD_HINT = (
    "模型权重不随 git 仓库分发，获取方式见 README「如何运行」第 3 节：\n"
    "  curl -L -o ai/model/sh17_yolov8s.pt "
    "https://github.com/ahmadmughees/SH17dataset/releases/download/v1/yolo8s.pt\n"
    "（国内网络需加代理：curl 追加 --proxy http://127.0.0.1:7890）"
)


@app.exception_handler(ModelUnavailableError)
def _model_unavailable_handler(request, exc):
    """模型不可用时返回 503 + 可读提示，而不是 500 + 原始堆栈"""
    return JSONResponse({"error": str(exc)}, status_code=503)


def get_pipeline():
    """懒加载模型管线（首次请求时加载，避免服务启动卡顿）"""
    global _detector, _pipeline
    if _pipeline is None:
        if not MODEL_PATH.exists():
            raise ModelUnavailableError(
                f"模型文件不存在：{MODEL_PATH}\n{MODEL_DOWNLOAD_HINT}")
        try:
            from ai.detector import PPEDetector
            from video.processor import InspectionPipeline
            from rule_engine.engine import SafetyRuleEngine
            # 阈值 / 设备 / 截图目录 / 冷却时间均来自集中配置
            _detector = PPEDetector(MODEL_PATH,
                                    conf_threshold=config.CONF_THRESHOLD,
                                    device=config.DEVICE)
            _pipeline = InspectionPipeline(
                detector=_detector,
                engine=SafetyRuleEngine(config.MINOR_THRESHOLD),
                screenshot_dir=config.SCREENSHOT_DIR,
                cooldown_s=config.VIOLATION_COOLDOWN_S)
        except OSError as exc:
            raise ModelUnavailableError(
                f"模型加载失败：{exc}\n{MODEL_DOWNLOAD_HINT}") from exc
    return _pipeline


@app.on_event("startup")
def startup():
    db.init_db(seed=True)


# ---------- 导出功能（独立模块 backend/export.py） ----------
app.include_router(export_api.router)


# ---------- 静态资源 ----------
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")
app.mount("/media", StaticFiles(directory=config.OUTPUTS_DIR), name="media")


@app.get("/")
def index():
    return FileResponse(config.STATIC_DIR / "index.html")


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
               date: str | None = None,
               date_from: str | None = None, date_to: str | None = None,
               sort: str = Query("id",
                                 pattern="^(id|created_at|severity|area_name)$"),
               order: str = Query("desc", pattern="^(asc|desc)$"),
               limit: int = Query(200, ge=1, le=1000),
               offset: int = Query(0, ge=0)):
    # 参数校验：与 /api/export/violations 保持同一口径。
    # 原实现完全不校验日期，date_to=2026-99-99 这类 typo 会被静默忽略并返回
    # 全表数据，用户会误以为"筛选生效了"，实际拿到的是全部记录。
    invalid = find_invalid_date(("date", date), ("date_from", date_from),
                                ("date_to", date_to))
    if invalid:
        return bad_request(f"{invalid[0]} 不是有效日期：{invalid[1]}"
                           f"（应为 YYYY-MM-DD）")
    if date_from and date_to and date_from > date_to:
        return bad_request(f"日期区间无效：起始 {date_from} 晚于结束 {date_to}")

    rows = db.query_violations(vtype=vtype, severity=severity, date_str=date,
                               date_from=date_from, date_to=date_to,
                               sort=sort, order=order,
                               limit=limit, offset=offset)
    total_count = db.count_violations(vtype=vtype, severity=severity,
                                      date_str=date, date_from=date_from,
                                      date_to=date_to)
    for r in rows:
        r["screenshot_url"] = f"/media/screenshots/{r['screenshot_path']}" if r.get("screenshot_path") else ""
    # total 保持契约原语义「本次返回条数」；total_count 为符合筛选条件的总条数（新增字段）
    return {"total": len(rows), "total_count": total_count, "items": rows}


# ---------- 检测：图片 ----------
@app.post("/api/detect/image")
async def detect_image(file: UploadFile = File(...),
                       lab_id: int = Form(config.DEFAULT_LAB_ID)):
    # 参数校验：非法输入返回 4xx，而不是 500
    if not db.lab_exists(lab_id):
        return bad_request(f"实验室不存在: lab_id={lab_id}")

    pipeline = get_pipeline()
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    save_path = UPLOAD_DIR / f"img_{uuid.uuid4().hex[:8]}{suffix}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    if save_path.stat().st_size == 0:
        save_path.unlink(missing_ok=True)
        return bad_request("上传内容为空，请选择有效的图片文件")

    areas = db.load_areas(lab_id)
    from video.processor import process_image
    out_name = f"annotated_{save_path.stem}.jpg"
    try:
        result = process_image(pipeline, save_path, areas,
                               save_path=OUT_IMG_DIR / out_name)
    except ValueError as exc:
        # 非图片 / 损坏文件等 → 400（原先未捕获会变成 500）
        return bad_request(f"图片无法读取或格式不支持：{exc}")

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
async def detect_video(file: UploadFile = File(...),
                       lab_id: int = Form(config.DEFAULT_LAB_ID),
                       stride: int = Form(config.VIDEO_STRIDE)):
    # 参数校验：非法输入返回 4xx，而不是 500
    if not db.lab_exists(lab_id):
        return bad_request(f"实验室不存在: lab_id={lab_id}")
    if stride < 1:
        # stride=0 会让 processor 里的 n_frames % stride 抛 ZeroDivisionError
        return bad_request(f"stride 必须为 ≥1 的整数，当前为 {stride}")

    pipeline = get_pipeline()
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    save_path = UPLOAD_DIR / f"vid_{uuid.uuid4().hex[:8]}{suffix}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    if save_path.stat().st_size == 0:
        save_path.unlink(missing_ok=True)
        return bad_request("上传内容为空，请选择有效的视频文件")

    areas = db.load_areas(lab_id)
    from video.processor import process_video
    out_name = f"annotated_{save_path.stem}.mp4"
    try:
        result = process_video(pipeline, save_path, areas,
                               OUT_VID_DIR / out_name, stride=stride)
    except ValueError as exc:
        # 非视频 / 损坏文件等 → 400（原先未捕获会变成 500）
        return bad_request(f"视频无法打开或格式不支持：{exc}")

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
    # SQL 已收敛到 database 层；其中含 ORDER BY，保证区域内 required_ppe 顺序稳定
    # （修复原实现顺序不固定、与 INTERFACE_CONTRACT 6.8 示例不一致的问题）
    return {"labs": db.list_labs_with_areas()}


@app.post("/api/settings/lab")
async def create_lab(name: str = Form(...), description: str = Form("")):
    try:
        db.create_lab(name, description)
    except sqlite3.IntegrityError:
        # 只捕获唯一约束冲突；其他异常（如磁盘故障）应保持 5xx 以暴露问题
        return bad_request("实验室名称已存在")
    return {"ok": True}


@app.post("/api/settings/area")
async def create_area(lab_id: int = Form(...), name: str = Form(...),
                      x1: float = Form(...), y1: float = Form(...),
                      x2: float = Form(...), y2: float = Form(...),
                      required_ppe: str = Form(config.DEFAULT_REQUIRED_PPE)):
    # ---- 参数校验（原先缺失，非法输入会触发外键约束 500）----
    if not db.lab_exists(lab_id):
        return bad_request(f"实验室不存在: lab_id={lab_id}")
    for nm, v in (("x1", x1), ("y1", y1), ("x2", x2), ("y2", y2)):
        if not 0.0 <= v <= 1.0:
            return bad_request(f"区域坐标 {nm}={v} 越界，必须为 0~1 的归一化值")
    if x1 >= x2 or y1 >= y2:
        return bad_request("区域坐标无效：要求 x1 < x2 且 y1 < y2，"
                           f"当前 x1={x1}, y1={y1}, x2={x2}, y2={y2}")
    ppe_list = [s.strip() for s in required_ppe.split(",") if s.strip()]
    invalid = [p for p in ppe_list if p not in ALLOWED_PPE]
    if invalid:
        return bad_request(f"不支持的 PPE 类型: {','.join(invalid)}；"
                           f"可选值: {','.join(ALLOWED_PPE)}")

    area_id = db.create_area(lab_id, name, x1, y1, x2, y2, ppe_list)
    return {"ok": True, "area_id": area_id}


@app.delete("/api/settings/area/{area_id}")
def delete_area(area_id: int):
    db.delete_area(area_id)
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
