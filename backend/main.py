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
import threading
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend import database as db
from backend import export as export_api
from backend import maintenance
from backend.config import config
from backend.tasks import QueueFull, TaskManager
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


# 视频后台任务管理器：用线程池执行阻塞的视频处理，避免占死事件循环。
# worker 数由 LABSAFETY_VIDEO_MAX_WORKERS 控制（默认 1，多余任务排队等待）。
task_manager = TaskManager(max_workers=config.VIDEO_MAX_WORKERS,
                           max_pending=config.MAX_PENDING_TASKS)


class ModelUnavailableError(RuntimeError):
    """模型权重缺失或加载失败（模型不入 git 仓库，新成员首次运行常遇到）"""


_detector = None
_pipeline = None
_pipeline_lock = threading.Lock()

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
    with _pipeline_lock:
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
# 特定媒体路径必须先于 /media 挂载，保持外置目录下的 URL 契约。
app.mount("/media/annotated", StaticFiles(directory=config.ANNOTATED_DIR), name="annotated")
app.mount("/media/videos", StaticFiles(directory=config.VIDEO_DIR), name="videos")
app.mount("/media/screenshots", StaticFiles(directory=config.SCREENSHOT_DIR), name="screenshots")
config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
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
# 注意：这里刻意用同步 def 而非 async def。
# FastAPI 会把同步端点自动放到线程池执行，因此 YOLO 推理（阻塞式 CPU/GPU）
# 不会占死事件循环 —— 检测进行中，其他接口仍能正常响应。
@app.post("/api/detect/image")
def detect_image(file: UploadFile = File(...),
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
def _save_upload(file: UploadFile, prefix: str, default_suffix: str) -> Path:
    """把上传文件落盘到临时上传目录，返回保存路径"""
    suffix = Path(file.filename or f"upload{default_suffix}").suffix or default_suffix
    save_path = UPLOAD_DIR / f"{prefix}_{uuid.uuid4().hex[:8]}{suffix}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return save_path


def run_video_job(video_path: Path, lab_id: int, source_name: str, stride: int,
                  progress_cb=None) -> dict:
    """
    处理视频并落库，返回响应体所需字段。

    同步端点与后台任务**共用**本函数，保证两条路径的处理与落库逻辑完全一致；
    progress_cb 为 None 时行为与旧版完全相同。
    """
    pipeline = get_pipeline()
    areas = db.load_areas(lab_id)
    from video.processor import process_video
    out_name = f"annotated_{video_path.stem}.mp4"
    result = process_video(pipeline, video_path, areas,
                           OUT_VID_DIR / out_name, stride=stride,
                           progress_cb=progress_cb)
    events = result["events"]
    # 人数取"单帧最大人数"：原实现恒填 0，导致 detection_records 中
    # 所有视频记录的 person_count 都是 0，库内数据不准确
    person_count = int(result.get("max_person_count", 0))
    record_id = db.save_detection_record(
        lab_id, "video", source_name, person_count, len(events), 0,
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


def _validate_video_params(lab_id: int, stride: int):
    """视频端点共用的参数校验；非法时返回 4xx 响应，合法时返回 None"""
    if not db.lab_exists(lab_id):
        return bad_request(f"实验室不存在: lab_id={lab_id}")
    if stride < 1:
        # stride=0 会让 processor 里的 n_frames % stride 抛 ZeroDivisionError
        return bad_request(f"stride 必须为 ≥1 的整数，当前为 {stride}")
    return None


# 注意：同步 def 让 FastAPI 把这段阻塞处理放到线程池执行，因此长视频处理期间
# **其他接口仍然可用**。原先写成 async def 却做阻塞调用，会把事件循环占死，
# 连 /api/dashboard 都打不开。
@app.post("/api/detect/video")
def detect_video(file: UploadFile = File(...),
                 lab_id: int = Form(config.DEFAULT_LAB_ID),
                 stride: int = Form(config.VIDEO_STRIDE)):
    invalid = _validate_video_params(lab_id, stride)
    if invalid is not None:
        return invalid

    # 先确认模型可用（缺失时由异常处理器返回 503），避免白存一个临时文件
    get_pipeline()

    save_path = _save_upload(file, "vid", ".mp4")
    if save_path.stat().st_size == 0:
        save_path.unlink(missing_ok=True)
        return bad_request("上传内容为空，请选择有效的视频文件")

    try:
        return run_video_job(save_path, lab_id,
                             file.filename or "upload.mp4", stride)
    except ValueError as exc:
        # 非视频 / 损坏文件等 → 400（原先未捕获会变成 500）
        return bad_request(f"视频无法打开或格式不支持：{exc}")


@app.post("/api/detect/video/async", status_code=202)
def detect_video_async(file: UploadFile = File(...),
                       lab_id: int = Form(config.DEFAULT_LAB_ID),
                       stride: int = Form(config.VIDEO_STRIDE)):
    """
    提交视频检测**后台任务**，立即返回 task_id（HTTP 202）。

    与同步端点 /api/detect/video 的差别：
      - 不等待处理完成，请求耗时只取决于上传耗时
      - 进度通过 GET /api/tasks/{task_id} 轮询（status / progress）
      - 处理在线程池中进行，服务全程保持可用

    同步端点保留不变，供短视频与脚本直接调用。
    """
    invalid = _validate_video_params(lab_id, stride)
    if invalid is not None:
        return invalid

    save_path = _save_upload(file, "vid", ".mp4")
    if save_path.stat().st_size == 0:
        save_path.unlink(missing_ok=True)
        return bad_request("上传内容为空，请选择有效的视频文件")

    # 闭包捕获本次请求的参数；模型加载放在 worker 内，
    # 这样提交请求能立刻返回，不被 5 秒的模型加载拖住
    video_path = save_path
    source_name = file.filename or "upload.mp4"

    def work(progress_cb):
        return run_video_job(video_path, lab_id, source_name, stride,
                             progress_cb=progress_cb)

    try:
        task_id = task_manager.submit("video_detect", work)
    except QueueFull as exc:
        # 队列已满：删掉刚落盘的临时文件，避免白白占用磁盘
        save_path.unlink(missing_ok=True)
        return JSONResponse({"error": str(exc)}, status_code=429)

    return {
        "task_id": task_id,
        "status": "pending",
        "status_url": f"/api/tasks/{task_id}",
        "message": "视频已提交后台处理，请轮询 status_url 获取进度",
    }


# ---------- 后台任务查询 ----------
@app.get("/api/tasks")
def list_tasks(limit: int = Query(20, ge=1, le=100)):
    """列出最近的后台任务概览（不含 result，避免响应体过大）"""
    items = []
    for task in task_manager.list_recent(limit):
        data = task.to_dict()
        data.pop("result", None)
        items.append(data)
    return {"stats": task_manager.stats(), "items": items}


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str):
    """查询单个后台任务的进度与结果"""
    task = task_manager.get(task_id)
    if task is None:
        return JSONResponse({"error": f"任务不存在: {task_id}"}, status_code=404)
    return task.to_dict()


# ---------- 存储维护（清理运行期文件） ----------
@app.get("/api/maintenance/storage")
def get_storage_report():
    """统计运行期目录占用：上传暂存 / 标注图 / 标注视频 / 违规截图"""
    return maintenance.storage_report()


@app.post("/api/maintenance/cleanup")
def cleanup_storage(
    days: int = Query(config.CLEANUP_KEEP_DAYS, ge=0, le=3650,
                      description="保留最近 N 天的文件"),
    dry_run: bool = Query(True, description="true 表示只统计不删除（默认）"),
    targets: str = Query("", description="逗号分隔；留空使用默认目标"
                                         "（uploads,annotated,videos）"),
):
    """
    清理运行期文件。

    安全设计：
      - 默认 dry_run=true，只统计不删除；确认无误后再传 dry_run=false
      - 只删除修改时间早于「今天 - days 天」的文件
      - 只在集中配置的那几个目录内操作，不会触碰其他路径
      - **不删除数据库记录**，历史违规记录始终保留

    注意：违规截图被 violation_events.screenshot_path 引用，默认**不清理**；
    如需清理请显式传 targets=screenshots，届时"违规记录"页中已清理的截图
    会显示为裂图。
    """
    target_list = [t.strip() for t in targets.split(",") if t.strip()] or None
    try:
        return maintenance.cleanup(keep_days=days, dry_run=dry_run,
                                   targets=target_list)
    except ValueError as exc:
        return bad_request(str(exc))


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
