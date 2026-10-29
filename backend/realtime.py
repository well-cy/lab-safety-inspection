# -*- coding: utf-8 -*-
"""
实时摄像头巡检模块（新增独立文件，不改动既有图片/视频检测链路）。

前端入口：backend/static/realtime.js（动态渲染到 index.html 的 #tab-realtime）

    POST /api/realtime/detect   接收摄像头单帧 → 检测 + 关联 + ROI + 多帧平滑
                                + 规则判定 + 违规截图 + 人员级去重 + 写既有违规表
    POST /api/realtime/stop     结束一次实时巡检会话（回写统计、清空全部平滑状态）

设计原则（本模块刻意"少做"）
----------------------------
1. **只复用，不重写**：检测走 video/processor.py 的 InspectionPipeline.
   analyze_frame，与图片/视频检测**完全相同**的 YOLO 推理、person-PPE 关联、
   ROI 归一化判断、安全规则引擎和违规等级判定；不新增第二套检测逻辑。
2. **不改动数据库结构**：沿用 detection_records（source_type='realtime'）与
   violation_events 两张既有表，实时巡检产生的违规记录自动出现在违规记录
   页、统计页和 CSV/Excel 导出中。
3. **实时模式事件语义 = "一次连续违规 = 一条事件"**（与视频检测不同）：
   去重键为 (session_id, track_id, 区域)。首次经多帧确认进入违规时写 1 条；
   之后只要该人员未恢复，无论持续 10 秒 / 30 秒 / 1 分钟，也无论缺失 PPE
   集合与违规等级如何变化，都**不再新增**记录；必须连续 RECOVER_FRAMES 帧
   稳定正常（active_violation 解除）后，才允许写下一条。
   **LABSAFETY_VIOLATION_COOLDOWN_S 只服务视频检测**：实时模式不再以
   "超过 cooldown_s" 作为重复写库条件（视频侧冷却逻辑未做任何改动）。
4. **会话模型**：浏览器开启摄像头 = 一个会话，对应一条 detection_records；
   停止时回写人数/违规数/耗时，避免每帧产生一条"检测记录"。

实时专属：多帧平滑（不影响图片/视频检测）
------------------------------------------
摄像头单帧 YOLO 结果存在偶发假阳性（例如裸手被误判 gloves），直接用单帧
判定会让"缺口罩 → 缺手套+口罩 → 缺手套+实验服+口罩"来回跳变。因此本模块
在**实时链路内部**加了一层短时平滑，图片/视频检测完全不经过这层逻辑：

  * 每个短时人员（track）维护最近 SMOOTH_WINDOW 帧的"各类 PPE 是否存在"
    滑动窗口，按类别做多数投票（满窗口需 ≥ VOTE_MIN 帧检出才算"已佩戴"）；
  * 业务判定（缺失 PPE / 违规等级 / 是否写库）一律走**平滑后**的稳定状态；
  * 前端画框仍用**单帧原始检测结果**（detections），不篡改模型输出；
  * 人员短时匹配只用 IoU / 中心点距离做最近邻（不引入 ByteTrack 等依赖），
    track_id 仅存活于当前 session，不写数据库。
"""
import logging
import os
import threading
import time
from datetime import datetime

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse

from ai.ppe_matcher import PersonState
from backend import database as db
from rule_engine.engine import SEVERITY_NONE

router = APIRouter(prefix="/api/realtime", tags=["实时摄像头巡检"])

# 前端默认抓帧间隔（毫秒）。真正生效值由前端控制，这里仅作为建议值回传。
DEFAULT_INTERVAL_MS = 400
# 单帧体积上限（前端 JPEG 质量 0.7，640x480 通常 < 100KB）
MAX_FRAME_BYTES = 20 * 1024 * 1024
# 会话闲置超时：超过该时长未收到帧则自动丢弃，防止异常退出导致状态泄漏
SESSION_IDLE_TIMEOUT_S = 3600

# ---------------- 实时多帧平滑参数（仅本模块使用） ----------------
# 滑动窗口帧数：最近 5 次有效检测结果
SMOOTH_WINDOW = 5
# 满窗口多数投票阈值：5 帧中至少 3 帧检出 → 认定"已佩戴"
VOTE_MIN = 3
# 参与投票的 PPE 类别（与正式模型 6 类中的 PPE 部分一致）
PPE_CLASSES = ("mask", "gloves", "lab_coat", "goggles", "coverall")
# 人员短时匹配：IoU 阈值
MATCH_IOU = 0.3
# 人员短时匹配：中心点距离 / person 框对角线 的上限（IoU 不足时的兜底）
MATCH_CENTER_RATIO = 0.5
# ---------------------------------------------------------------------------
# track 失效策略：**按时间而非帧数**
# 实时巡检实际只有 1~2 FPS，按"连续 5 帧未匹配"删除在低帧率下相当于 2.5~5
# 秒，且只要某帧框漂移就会立刻新建 track（旧 track 要等 5 帧才删），于是出现
# T1 → T2 → T3 → T4 连续换号。改为：未匹配到 person 后仍保留 TRACK_KEEP_S 秒，
# 期间再次检出优先复用原 track；只有超过该时长才认定人员真正离开。
TRACK_KEEP_S = 5.0
# 判失效所需的**最少连续未出现帧数**。单帧推理偶发很慢（首帧加载模型可能
# 5~6 秒）时，人员其实一直在画面里，只看墙钟时间会把 T1 误删并重建为 T2。
TRACK_MIN_MISS_FRAMES = 2
# 单人场景宽松回收窗口：当前帧只有 1 个 person 且内存里只有 1 个 track 时，
# 即使 IoU 因姿势/远近变化下降，也直接复用该 track（不新建）。
SOLO_RECLAIM_S = 8.0
# 框面积变化容忍：真人前后移动 / 坐下站起会让框面积成倍变化，IoU 随之骤降。
# 只要面积比落在该区间且中心点接近，仍视为同一个人。
AREA_MIN_RATIO = 0.25
AREA_MAX_RATIO = 4.0
# 面积变化落在上述区间时，中心距阈值放宽到该值（否则用 MATCH_CENTER_RATIO）
MATCH_CENTER_RATIO_LOOSE = 0.9
# 仅用于日志与兜底：连续未匹配帧数（不再作为删除依据）
TRACK_MAX_MISS = 5
# 同一帧内两个 person 框的 IoU 超过该值 → 视为**同一个人的重复框**，
# 不再为它新建 track（避免"一个 person 被拆成人员1+人员2"）。
DUP_BOX_IOU = 0.7
# IoU 兜底判据 —— **包含比例**（交集 / 较小框面积）。
# 坐得离摄像头近时 YOLO 常把同一个人同时检成"全身框 + 躯干/半身框"：小框
# 几乎完全落在大框内（包含比例接近 1.0），但两者 IoU 可能只有 0.4，单靠
# DUP_BOX_IOU 识别不出来，于是画面里出现"人员1 + 人员2"。
DUP_BOX_CONTAIN = 0.6
# 新 track 需**连续**确认的帧数：未确认前不算"正式人员"，不写违规事件。
# 用于挡住只出现 1~2 帧的假阳性框（镜像/衣物/屏幕反光等），避免它立刻变成
# "人员2"并写出第二条记录。已确认过的 track 不会再退回未确认状态。
TRACK_CONFIRM_FRAMES = 3
# track 断裂回收：被删除的 track 在多长时间内、多大重合度下可以被新 track
# "认领"回来（继承 episodes），用于防止"同一真人 track 断裂 → 重复写库"。
TRACK_RECLAIM_S = 15.0
TRACK_RECLAIM_IOU = 0.45
TRACK_RECLAIM_RATIO = 0.45
# 内存中保留多少个已删除 track 的快照用于回收判定
LOST_TRACK_KEEP = 8
# 违规解除确认：连续多少帧"稳定结果正常"才认定本次违规已恢复
RECOVER_FRAMES = 3
# 违规成立确认：连续多少帧"稳定结果违规"才允许写库
VIOLATION_CONFIRM = 2

# ---------------- track 调试日志（仅开发排查，不在页面上显示） ----------------
# 作用：当页面出现"人员1 → 人员2 → 人员1"这类编号跳变时，用日志判断到底是
#   A. 同一个真人的短时 track 发生断裂（IoU/中心距离骤降）；还是
#   B. 画面里确实多检出了另一个 person（手机/屏幕/照片中的人）。
# 关闭方式：设置环境变量 LABSAFETY_TRACK_DEBUG=0（默认开启，只写服务端日志）
TRACK_DEBUG = os.getenv("LABSAFETY_TRACK_DEBUG", "1").strip().lower() not in (
    "0", "false", "off", "no")
# 每个会话在内存中保留的日志条数上限（环形缓冲，避免长时间巡检占内存）
# 每帧会记一条 frame 条目，60 秒巡检约 60~150 条，400 足够覆盖一次排查
TRACK_LOG_MAX = 400

_logger = logging.getLogger("realtime.track")
if not _logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [track] %(message)s",
                                      "%H:%M:%S"))
    _logger.addHandler(_h)
_logger.setLevel(logging.DEBUG if TRACK_DEBUG else logging.WARNING)
# 不向 root 传播，避免污染 uvicorn 的访问日志格式
_logger.propagate = False


def _fmt_box(b) -> str:
    """bbox → 便于阅读的字符串（调试日志用）"""
    if not b:
        return "-"
    return "[%.0f,%.0f,%.0f,%.0f]" % (b[0], b[1], b[2], b[3])


def _log_track(session_id: int, session: dict, action: str, tid, bbox=None,
               extra=None) -> None:
    """
    记录一条 track 生命周期事件：创建 / 匹配 / 丢失 / 删除。

    同时写入两处：
      1. 服务端日志（logger realtime.track，可用 LABSAFETY_TRACK_DEBUG=0 关闭）；
      2. 会话内存环形缓冲，可通过 GET /api/realtime/tracklog 拉取，
         或随 /api/realtime/stop 返回，便于事后排查。
    页面本身不展示这些信息。
    """
    entry = {"t": round(time.time(), 3), "action": action,
             "track": f"T{tid}", "bbox": _fmt_box(bbox)}
    if extra:
        entry.update(extra)
    log = session.setdefault("track_log", [])
    log.append(entry)
    if len(log) > TRACK_LOG_MAX:
        del log[0]
    if TRACK_DEBUG:
        kv = " ".join(f"{k}={v}" for k, v in entry.items()
                      if k not in ("t", "action", "track"))
        _logger.debug("sid=%s %-6s %s %s", session_id, action, entry["track"],
                      kv)

_pipeline_getter = None
_lock = threading.Lock()
# session_id(=detection_records.id) -> 会话状态
_sessions: dict[int, dict] = {}


def init_realtime(pipeline_getter) -> None:
    """由 backend/main.py 注入模型管线获取函数（与图片/视频共用同一管线）"""
    global _pipeline_getter
    _pipeline_getter = pipeline_getter


def _get_pipeline():
    if _pipeline_getter is None:
        raise RuntimeError("实时巡检模块未初始化：请先调用 init_realtime()")
    return _pipeline_getter()


def _prune_sessions(now: float) -> None:
    """清理长时间未收到帧的会话（异常关闭浏览器时兜底）"""
    stale = [sid for sid, s in _sessions.items()
             if now - s["last_seen"] > SESSION_IDLE_TIMEOUT_S]
    for sid in stale:
        _sessions.pop(sid, None)


def _new_session(lab_id: int, now: float) -> int:
    """新建实时巡检会话：落一条 detection_records，返回 session_id"""
    record_id = db.save_detection_record(
        lab_id, "realtime", "摄像头实时巡检", 0, 0, 0, 0.0)
    _sessions[record_id] = {
        "id": record_id,
        "lab_id": lab_id,
        "t0": now,
        "last_seen": now,
        "frames": 0,
        "max_persons": 0,
        "violations": 0,
        "total_ms": 0.0,
        # 短时人员跟踪：tid -> track（仅内存，stop / 会话清理时整体丢弃）
        "tracks": {},
        "next_track": 1,
        # 已删除 track 的快照（用于断裂回收判定，仅内存）
        "lost": [],
        # track 生命周期调试日志（环形缓冲，仅排查用，不落库不显示）
        "track_log": [],
    }
    return record_id


def _decode_frame(raw: bytes):
    """JPEG/PNG 字节 → BGR 图像；解码失败返回 None"""
    buf = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# ---------------- 人员短时匹配（轻量最近邻，非完整跟踪器） ----------------

def _box_of(st) -> tuple:
    return (float(st.bbox.x1), float(st.bbox.y1),
            float(st.bbox.x2), float(st.bbox.y2))


def _iou(a: tuple, b: tuple) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _containment(a: tuple, b: tuple) -> float:
    """
    包含比例 = 交集面积 / **较小框面积**（1.0 表示小框完全落在大框内）。

    用来识别"全身框 + 躯干/半身框"这类**同一个人的重复框**：此时 IoU 可能
    只有 0.3~0.5（因为大框比小框大得多），但包含比例接近 1.0。两个真正
    不同的人即使并排站立，也很难出现这么高的包含比例。
    """
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    return inter / min(_area(a), _area(b))


def _is_dup_box(a: tuple, b: tuple) -> bool:
    """两个 person 框是否为同一个人的重复框（IoU 高 或 小框被大框包含）"""
    return _iou(a, b) >= DUP_BOX_IOU or _containment(a, b) >= DUP_BOX_CONTAIN


def _center_ratio(a: tuple, b: tuple) -> float:
    """两个框的中心点距离 / person 框对角线（越小越可能是同一人）"""
    ca = ((a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0)
    cb = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
    dist = ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5
    diag = ((a[2] - a[0]) ** 2 + (a[3] - a[1]) ** 2) ** 0.5
    return dist / diag if diag > 0 else 999.0


def _center_ratio_sym(a: tuple, b: tuple) -> float:
    """
    中心点距离 / **两框平均对角线**（对称版本，用于匹配判定）。

    为什么不用 _center_ratio：它以"上一帧框"的对角线做分母。真人从半身变
    全身（框变大）或走近走远时，分母会跟着变，同一个相对位移算出的比值会
    忽大忽小，容易在框尺寸突变那一帧误判为"不是同一个人"。改用两框平均
    对角线后，框大小变化对判定更稳定。
    """
    ca = ((a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0)
    cb = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
    dist = ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5
    diag = ((((a[2] - a[0]) ** 2 + (a[3] - a[1]) ** 2) ** 0.5)
            + (((b[2] - b[0]) ** 2 + (b[3] - b[1]) ** 2) ** 0.5)) / 2.0
    return dist / diag if diag > 0 else 999.0


def _area(b: tuple) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _area_ratio(a: tuple, b: tuple) -> float:
    """
    当前框面积 / 上一帧框面积。真人在摄像头前前后移动、坐下站起、从半身变
    全身时，person 框面积会成倍变化，IoU 随之骤降 —— 但仍是同一个人。
    """
    pa = _area(a)
    if pa <= 0:
        return 1.0
    return _area(b) / pa


def _shape_compatible(a: tuple, b: tuple) -> bool:
    """面积变化是否落在"同一人合理范围"内（配合中心距离使用）"""
    return AREA_MIN_RATIO <= _area_ratio(a, b) <= AREA_MAX_RATIO


def _match_tracks(tracks: dict, boxes: list) -> list:
    """
    上一帧 track 与当前帧 person 框做最近邻匹配。
    返回与 boxes 等长的列表，元素为匹配到的 tid（未匹配为 None）。
    """
    cands = []
    for tid, tr in tracks.items():
        if not tr.get("bbox"):
            continue
        for i, b in enumerate(boxes):
            iou = _iou(tr["bbox"], b)
            ratio = _center_ratio_sym(tr["bbox"], b)
            # 不能只依赖 IoU：真人前后移动 / 坐下站起会让框面积成倍变化，
            # IoU 随之骤降但仍是同一人。此时放宽中心距阈值（面积变化合理即可）。
            loose = (MATCH_CENTER_RATIO_LOOSE
                     if _shape_compatible(tr["bbox"], b) else MATCH_CENTER_RATIO)
            if iou >= MATCH_IOU or ratio <= loose:
                # IoU 优先；IoU 不足时用中心距离换算成分数
                score = iou if iou >= MATCH_IOU else (1.0 - min(ratio, 1.0))
                cands.append((score, tid, i))
    cands.sort(key=lambda x: -x[0])
    used_tid, assign = set(), [None] * len(boxes)
    for _score, tid, i in cands:
        if tid in used_tid or assign[i] is not None:
            continue
        used_tid.add(tid)
        assign[i] = tid
    return assign


def _dup_pairs(boxes: list) -> list:
    """
    找出同一帧里高度重合的 person 框（很可能是同一个人被拆成两框）。

    判定：IoU ≥ DUP_BOX_IOU **或** 包含比例 ≥ DUP_BOX_CONTAIN。
    返回 [(i, j, 重合度)]，i<j。仅用于开发日志，帮助区分：
      情况 2（画面里真的有第二个人，例如屏幕/照片中的人）→ 无高重合对；
      情况 3（同一个 person 被错误分配成两个 track）→ 存在高重合对。
    """
    out = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            iou = _iou(boxes[i], boxes[j])
            contain = _containment(boxes[i], boxes[j])
            if _is_dup_box(boxes[i], boxes[j]):
                out.append((i, j, round(max(iou, contain), 3)))
    return out


def _merge_dup_boxes(boxes: list):
    """
    同帧重复框合并：把"同一个人的多个框"压成**一个代表框**。

    代表框取面积最大的那个（通常是覆盖最全的全身框，PPE 关联最完整，
    业务判定也最准确）。合并发生在 track 匹配**之前**，这样：
      * 近距离时"全身框 + 躯干框"只算 1 个 person，单人连续性保护才能生效；
      * 同一个人不会因为重复框而在一帧里投两次票、或写出两条记录。

    返回 (保留下标列表, [(被合并下标, 保留下标)] )
    """
    if len(boxes) < 2:
        return list(range(len(boxes))), []
    order = sorted(range(len(boxes)), key=lambda i: -_area(boxes[i]))
    keep, merged = [], []
    for i in order:
        hit = None
        for j in keep:
            if _is_dup_box(boxes[j], boxes[i]):
                hit = j
                break
        if hit is None:
            keep.append(i)
        else:
            merged.append((i, hit))
    return sorted(keep), merged


def _best_candidate(tracks: dict, box: tuple, exclude=()):
    """
    在现有 track 中找与 box 最相近的一个，返回 (tid, iou, center_ratio, bbox)。
    仅用于日志与断裂回收判定，不参与实际分配。
    """
    best = (None, 0.0, None, None)
    for tid, tr in tracks.items():
        if tid in exclude or not tr.get("bbox"):
            continue
        iou = _iou(tr["bbox"], box)
        ratio = _center_ratio(tr["bbox"], box)
        if iou > best[1]:
            best = (tid, iou, ratio, tr["bbox"])
    return best


def _pop_lost_for(session: dict, tid: int, box: tuple, now: float):
    """
    track 断裂回收：新建 track 时，如果它与**刚失效的 track** 高度重合，
    就取出对方的完整快照（滑动窗口 / 稳定 PPE / active_violation / 已写事件数）。

    目的：摄像头抖动、短暂遮挡、检测器漏检会让同一个真人先被 delete、
    再被当成"新人"新建 track。若不做回收，新 track 的 episodes 是空的，
    active=False，于是**同一次连续违规会被再写一条**——这正是"持续违规
    仍然重复写库"的第二种成因（第一种是旧的 10 秒 cooldown 判据）。

    回收继承的是**全部业务状态**（滑动窗口 / 稳定 PPE / active_violation /
    已写事件数），保证"T1 漏检几秒后再次检出仍算同一次连续违规，不再写第 2 条"。
    """
    lost = session.get("lost") or []
    for item in reversed(lost):
        if now - item["t"] > TRACK_RECLAIM_S:
            continue
        iou = _iou(item["bbox"], box)
        ratio = _center_ratio(item["bbox"], box)
        if iou >= TRACK_RECLAIM_IOU or ratio <= TRACK_RECLAIM_RATIO:
            _log_track(session["id"], session, "reclaim", tid, box, {
                "from_track": f"T{item['tid']}",
                "iou": round(iou, 3),
                "center_ratio": round(ratio, 3),
                "area_ratio": round(_area_ratio(item["bbox"], box), 3),
                "prev_bbox": _fmt_box(item["bbox"]),
                "inherited_active": any(
                    e.get("active") for e in item["episodes"].values()),
                "inherited_events": sum(
                    int(e.get("events", 0)) for e in item["episodes"].values()),
                "inherited_window": len(item.get("win") or []),
            })
            lost.remove(item)
            return item
    return None


def _inherit_track_state(tr_new: dict, item: dict) -> None:
    """
    回收时把旧 track 的**全部业务状态**搬过来：滑动窗口、稳定 PPE 快照、
    active_violation、已写事件计数。

    这样即使真的发生断裂重建，同一次连续违规也不会被当成"新人员"再写一条
    —— 对应验收"T1 漏检 2 秒后再次检出必须仍算 T1，不能变成 T2、不能写第 2 条"。
    """
    tr_new["win"] = list(item.get("win") or [])
    tr_new["roi_win"] = list(item.get("roi_win") or [])
    tr_new["last_det"] = dict(item.get("last_det") or {})
    tr_new["episodes"] = {a: dict(e) for a, e in
                          (item.get("episodes") or {}).items()}


def _new_track(tid: int, now: float = 0.0) -> dict:
    return {
        "tid": tid,
        "bbox": None,
        "miss": 0,
        # 连续命中帧数：需达到 TRACK_CONFIRM_FRAMES 才算"正式人员"，
        # 之前不参与违规判定与写库（挡住 1~2 帧的假阳性框）
        "confirm": 0,
        # 最近一次匹配到 person 的时间（track 失效按该时间判断，不再按帧数）
        "last_seen_t": now,
        # 创建时间，便于排查"这个 track 是何时出现的"
        "created_t": now,
        # 每帧一条：{ppe类别: 是否检出}
        "win": [],
        # 每帧一条：(是否在ROI, 区域名, 区域必需PPE)
        "roi_win": [],
        # 各类最近一次真实 Detection（稳定判定为"已佩戴"时回放用）
        "last_det": {},
        # 区域 -> 违规事件管理（去重键 = session_id + track_id + 区域）
        "episodes": {},
    }


def _vote(values: list) -> bool:
    """
    多数投票：窗口内至少 need 帧为真才判 True。
      1 帧      → 直接用当前结果（页面标记"建立中"）
      2 帧      → 两帧一致才确认
      3~5 帧    → 严格多数（n//2+1，5 帧即 ≥3）
    """
    n = len(values)
    if n == 0:
        return False
    if n == 1:
        return bool(values[0])
    need = 2 if n == 2 else n // 2 + 1
    return sum(1 for v in values if v) >= need


def _stabilize(tr: dict):
    """
    对一个 track 的滑动窗口做多数投票，返回
    (稳定PPE字典, 是否在ROI, 区域名, 区域必需PPE, 窗口是否已满)
    """
    win, roi_win = tr["win"], tr["roi_win"]
    ppe = {}
    for cls in PPE_CLASSES:
        if _vote([bool(w.get(cls, False)) for w in win]):
            det = tr["last_det"].get(cls)
            if det is not None:
                ppe[cls] = det
    in_roi = _vote([bool(r[0]) for r in roi_win]) if roi_win else False
    area_name, required = "", []
    if in_roi:
        names = [r[1] for r in roi_win if r[0]]
        if names:
            area_name = max(set(names), key=names.count)
            for r in reversed(roi_win):
                if r[0] and r[1] == area_name:
                    required = list(r[2])
                    break
    return ppe, in_roi, area_name, required, len(win) >= SMOOTH_WINDOW


def _episode(tr: dict, area: str) -> dict:
    """取（或建）该人员在该区域的违规事件管理状态"""
    ep = tr["episodes"].get(area)
    if ep is None:
        ep = {"active": False, "last_t": 0.0, "severity": SEVERITY_NONE,
              "vio_streak": 0, "ok_streak": 0,
              # 本次连续违规的起始时间 / 最近一次仍违规的时间 / 已写事件条数
              "start_t": 0.0, "last_vio_t": 0.0, "events": 0}
        tr["episodes"][area] = ep
    return ep


def _update_episode(tr: dict, ev, now: float, cooldown_s: float = 0.0) -> bool:
    """
    实时模式人员级事件管理（去重键 = session_id + track_id + 区域）。
    返回 True 表示本次允许写入一条**新的**违规事件。

    语义：**一次连续违规 = 一条事件**
      * 违规需连续 VIOLATION_CONFIRM 帧"稳定违规"才确认；
      * 仅当该 (session, track, 区域) 当前**没有 active_violation** 时才写库，
        写完立即置 active=True；
      * active=True 期间：无论持续多久、缺失 PPE 集合如何变化、严重/一般
        如何切换，都**不写第二条**（页面显示照常跟随稳定状态更新）；
      * 必须连续 RECOVER_FRAMES 帧"稳定正常"才清除 active（本次事件结束），
        此后再次违规才允许写下一条。

    参数 cooldown_s 仅为兼容既有调用保留，**实时模式不再用它决定是否重复
    写库**（LABSAFETY_VIOLATION_COOLDOWN_S 仍服务于视频检测，未做任何改动）。
    """
    if ev is not None and ev.is_violation:
        ep = _episode(tr, ev.area_name or "默认区域")
        ep["ok_streak"] = 0
        ep["vio_streak"] += 1
        ep["severity"] = ev.severity          # 页面立即显示升级后的等级
        ep["last_vio_t"] = now
        if ep["vio_streak"] >= VIOLATION_CONFIRM and not ep["active"]:
            ep["active"] = True
            ep["start_t"] = now
            ep["last_t"] = now
            ep["events"] += 1
            return True
        return False
    # 稳定结果正常 / 不在 ROI：连续 RECOVER_FRAMES 帧才认定已恢复
    for ep in tr["episodes"].values():
        ep["vio_streak"] = 0
        ep["ok_streak"] += 1
        if ep["ok_streak"] >= RECOVER_FRAMES:
            ep["active"] = False
            ep["severity"] = SEVERITY_NONE
            ep["vio_streak"] = 0
    return False


@router.post("/detect")
def realtime_detect(frame: UploadFile = File(...),
                    lab_id: int = Form(1),
                    session_id: int = Form(0)):
    """
    接收摄像头单帧并返回检测结果（业务判定为多帧平滑后的稳定结果）。

    请求（multipart/form-data）：
      frame       必填，JPEG 图片（前端 canvas.toBlob 产出）
      lab_id      实验室 ID（决定使用哪套 ROI 与 PPE 规则）
      session_id  会话 ID；0 或无效值表示开启新会话（返回新的 session_id）

    响应关键字段：
      session_id      当前会话 ID（前端需回传，用于续接会话）
      person_count    本帧人数
      persons         每人状态（track_id / bbox / 区域 / 稳定缺失PPE / 违规等级）
      detections      单帧原始检测框（含 PPE 类别与置信度，未做平滑）
      areas           ROI 归一化坐标（前端叠加绘制）
      new_events      本次**真正写入数据库**的违规事件（一次连续违规仅 1 条；
                      同人员同区域未恢复正常前不重复写，与 10 秒冷却无关）
      smooth          平滑状态（窗口大小 / 投票阈值 / 是否已建立）
    """
    if not db.lab_exists(lab_id):
        return JSONResponse({"error": f"实验室不存在: lab_id={lab_id}"},
                            status_code=400)

    raw = frame.file.read()
    if not raw:
        return JSONResponse({"error": "收到空帧，未进行检测"}, status_code=400)
    if len(raw) > MAX_FRAME_BYTES:
        return JSONResponse(
            {"error": f"单帧过大（{len(raw)} 字节），上限 {MAX_FRAME_BYTES}"},
            status_code=413)
    img = _decode_frame(raw)
    if img is None:
        return JSONResponse({"error": "图片无法解码，请确认发送的是 JPEG/PNG"},
                            status_code=400)

    pipeline = _get_pipeline()
    areas = db.load_areas(lab_id)
    now = time.time()

    with _lock:
        _prune_sessions(now)
        session = _sessions.get(session_id)
        if session is None or session["lab_id"] != lab_id:
            # 首次调用 / 服务重启 / 切换实验室 → 开新会话
            session_id = _new_session(lab_id, now)
            session = _sessions[session_id]

    frame_time = round(now - session["t0"], 3)
    t0 = time.perf_counter()
    # ↓↓↓ 与视频检测完全相同的分析入口（YOLO → 关联 → ROI → 规则引擎）
    states, raw_events, dets = pipeline.analyze_frame(img, areas,
                                                      frame_time=frame_time)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    raw_map = {e.person_id: e for e in raw_events}

    # ---------- 短时人员跟踪 + 多帧平滑 + 人员级去重 ----------
    new_events = []
    with _lock:
        tracks = session["tracks"]
        boxes = [_box_of(st) for st in states]
        confs = [round(float(getattr(st.bbox, "conf", 0.0) or 0.0), 3)
                 for st in states]

        # 0) 先清掉"已经超过保留时长"的 track（按时间失效，不再按帧数）
        for tid in list(tracks.keys()):
            tr = tracks[tid]
            idle = now - (tr.get("last_seen_t") or tr.get("created_t") or now)
            # 必须**连续 2 帧以上没出现**才允许判失效：单帧推理偶发很慢
            # （例如首帧加载模型 5~6 秒）时，人员其实一直在画面里，不能因为
            # 那一帧的墙钟时间超过 TRACK_KEEP_S 就把 T1 删掉、再建 T2。
            missed = session["frames"] - int(tr.get("last_frame", 0) or 0)
            if idle > TRACK_KEEP_S and missed >= TRACK_MIN_MISS_FRAMES:
                session.setdefault("lost", []).append({
                    "tid": tid, "t": now, "bbox": tr["bbox"],
                    "win": list(tr["win"]), "roi_win": list(tr["roi_win"]),
                    "last_det": dict(tr["last_det"]),
                    "episodes": {a: dict(e) for a, e in tr["episodes"].items()},
                })
                if len(session["lost"]) > LOST_TRACK_KEEP:
                    del session["lost"][0]
                if TRACK_DEBUG:
                    _log_track(session_id, session, "delete", tid, tr["bbox"], {
                        "reason": "idle_exceeded_keep_seconds",
                        "idle_s": round(idle, 2),
                        "keep_s": TRACK_KEEP_S,
                        "miss": tr["miss"],
                        "missed_frames": missed,
                        "last_bbox": _fmt_box(tr["bbox"]),
                        "had_active": any(e.get("active")
                                          for e in tr["episodes"].values()),
                        "tracks_after": len(tracks) - 1})
                tracks.pop(tid, None)

        # 0.5) 同帧重复框合并（在匹配**之前**做）
        #
        # 坐得离摄像头近时 YOLO 常把同一个人同时检成"全身框 + 躯干/半身框"，
        # 两者 IoU 可能只有 0.4，但小框几乎完全落在大框内。若不先合并，画面
        # 里就是 2 个 person 框 → 匹配阶段容易给不出 track 而新建 T2 —— 这
        # 正是"坐近了容易出现人员1 + 人员2"的直接原因。合并后本帧只算 1 个
        # person，单人连续性保护才能正常生效。
        keep_idx, merged = _merge_dup_boxes(boxes)
        eff = keep_idx
        boxes_eff = [boxes[i] for i in eff]
        if TRACK_DEBUG and merged:
            for drop_i, keep_i in merged:
                _log_track(session_id, session, "dup_box", "-", boxes[drop_i], {
                    "merged_into_idx": keep_i,
                    "iou": round(_iou(boxes[keep_i], boxes[drop_i]), 3),
                    "contain": round(_containment(boxes[keep_i],
                                                  boxes[drop_i]), 3),
                    "n_boxes_raw": len(boxes),
                    "n_boxes_after": len(eff)})

        assign = _match_tracks(tracks, boxes_eff)

        if TRACK_DEBUG:
            # 帧级日志：本帧检出几个 person 框、合并后剩几个、各自 bbox 与
            # 置信度、是否存在高重合度重复框。这是判断"人员1 + 人员2 同时
            # 出现"到底是两个人、还是重复框 / 屏幕人像的第一手依据。
            dups = _dup_pairs(boxes)
            _log_track(session_id, session, "frame", "-", None, {
                "n_boxes": len(boxes),
                "n_eff": len(eff),
                "n_tracks": len(tracks),
                "boxes": " ".join(_fmt_box(b) for b in boxes[:4]),
                "confs": ",".join(str(c) for c in confs[:4]) or "-",
                "no_person_detected": (len(boxes) == 0),
                "dup_pairs": (",".join(f"{i}-{j}:{v}" for i, j, v in dups)
                              or "-")})

        # 1) 匹配/新建 track，并把本帧结果压入滑动窗口
        seen = set()
        for k, i in enumerate(eff):
            box = boxes[i]
            st = states[i]
            tid = assign[k]
            if tid is None:
                # 单人连续性保护：本帧合并后**只有 1 个 person**、内存里**只有
                # 1 个 track** → 画面里明确只有一个人。此时即使 IoU 因姿势变化、
                # 坐下站起、前后移动、半身变全身而骤降，也必须复用原 track，
                # 不允许新建 T2/T3/T4。多人场景（eff>1 或 tracks>1）绝不启用，
                # 避免把两个不同的人错误合并成一个 track。
                solo_tid = None
                if len(eff) == 1 and len(tracks) == 1:
                    _only_tid = next(iter(tracks))
                    _only_tr = tracks[_only_tid]
                    _idle = now - (_only_tr.get("last_seen_t")
                                   or _only_tr.get("created_t") or now)
                    if _only_tr.get("bbox") and _idle <= SOLO_RECLAIM_S:
                        solo_tid = _only_tid

                if solo_tid is not None:
                    _prev = tracks[solo_tid]["bbox"]
                    if TRACK_DEBUG:
                        _log_track(session_id, session, "solo_reclaim", solo_tid,
                                   box, {
                                       "iou": round(_iou(_prev, box), 3),
                                       "center_ratio": round(
                                           _center_ratio_sym(_prev, box), 3),
                                       "area_ratio": round(
                                           _area_ratio(_prev, box), 3),
                                       "idle_s": round(_idle, 2),
                                       "prev_bbox": _fmt_box(_prev),
                                       "conf": confs[i]})
                    tid = solo_tid
                    assign[k] = tid
                else:
                    # 跨帧重复框吸收：新框几乎完全落在某个已有 track 的框内
                    # （近距离"全身框 + 躯干框"，IoU 可能不达标）→ 直接归入
                    # 该 track，不为它新开人员编号。
                    dup_tid, dup_val = None, 0.0
                    for _tid, _tr in tracks.items():
                        if not _tr.get("bbox"):
                            continue
                        _v = _containment(_tr["bbox"], box)
                        if _v >= DUP_BOX_CONTAIN and _v > dup_val:
                            dup_tid, dup_val = _tid, _v
                    if dup_tid is not None:
                        if TRACK_DEBUG:
                            _log_track(session_id, session, "dup_prev", dup_tid,
                                       box, {
                                           "contain": round(dup_val, 3),
                                           "prev_bbox": _fmt_box(
                                               tracks[dup_tid]["bbox"]),
                                           "area_ratio": round(
                                               _area_ratio(
                                                   tracks[dup_tid]["bbox"],
                                                   box), 3),
                                           "conf": confs[i]})
                        tid = dup_tid
                        assign[k] = tid
                    else:
                        # 新建 track 前先记录"最接近的旧 track"：用于区分
                        #   A. 同一真人 track 断裂（best_iou 本该很高却没匹配上）
                        #   B. 画面里真的多了一个 person（手机/屏幕/照片中的人）
                        best_tid, best_iou, best_ratio, best_bbox = \
                            _best_candidate(tracks, box)
                        if not tracks:
                            reason = "no_track_yet"     # 会话首帧，还没有 track
                        elif best_tid is None:
                            reason = "no_track_with_box"  # 有 track 但都没位置
                        elif best_iou >= MATCH_IOU or (best_ratio is not None
                                                       and best_ratio <= MATCH_CENTER_RATIO):
                            reason = "best_candidate_taken"  # 最近 track 被占用
                        else:
                            reason = "no_match_iou_and_center"  # 疑似新出现的人
                        tid = session["next_track"]
                        session["next_track"] += 1
                        tracks[tid] = _new_track(tid, now)
                        assign[k] = tid
                        # 断裂回收：与刚失效的 track 高度重合 → 继承其全部业务
                        # 状态（滑动窗口 / 稳定 PPE / active_violation / 已写
                        # 事件数），保证同一次连续违规不被写成两条。
                        item = _pop_lost_for(session, tid, box, now)
                        if item is not None:
                            _inherit_track_state(tracks[tid], item)
                            # 继承即视为已确认：断裂前后是同一个人，不该重新
                            # 走一遍"新人员确认"，否则会多出一帧的空窗期
                            tracks[tid]["confirm"] = TRACK_CONFIRM_FRAMES
                        if TRACK_DEBUG:
                            _log_track(session_id, session, "create", tid, box, {
                                "reason": reason,
                                "best_iou": round(best_iou, 3),
                                "best_iou_track": (f"T{best_tid}" if best_tid
                                                   else "-"),
                                "best_center_ratio": (round(best_ratio, 3)
                                                      if best_ratio is not None
                                                      else "-"),
                                "inherited": item is not None,
                                "tracks_after": len(tracks),
                                "conf": confs[i]})
            else:
                # 命中已有 track：记录本次匹配的 IoU / 中心距离
                prev = tracks[tid].get("bbox")
                if TRACK_DEBUG:
                    _log_track(session_id, session, "match", tid, box, {
                        "iou": round(_iou(prev, box) if prev else 0.0, 3),
                        "center_ratio": (round(_center_ratio(prev, box), 3)
                                         if prev else "-"),
                        "prev_bbox": _fmt_box(prev)})
            seen.add(tid)
            tr = tracks[tid]
            tr["bbox"] = box
            tr["miss"] = 0
            # 连续命中计数：达到 TRACK_CONFIRM_FRAMES 才算正式人员
            tr["confirm"] = min(int(tr.get("confirm", 0)) + 1,
                                TRACK_CONFIRM_FRAMES)
            tr["last_seen_t"] = now      # track 失效按该时间戳（秒）判定
            tr["last_frame"] = session["frames"]   # 失效还需看连续未出现帧数
            tr["win"].append({c: (c in st.ppe) for c in PPE_CLASSES})
            tr["roi_win"].append(
                (bool(st.in_roi), getattr(st, "area_name", "") or "",
                 list(getattr(st, "area_required", []) or [])))
            for c, det in st.ppe.items():
                if c in PPE_CLASSES:
                    tr["last_det"][c] = det
            if len(tr["win"]) > SMOOTH_WINDOW:
                del tr["win"][0]
                del tr["roi_win"][0]

        # 1.5) 每个 track 只取**一个代表框**（面积最大）参与平滑与判定。
        #      重复框被合并后可能仍有多个框落到同一 track，取代表框可避免
        #      同一人一帧投多次票、或一帧产生两条违规。
        primary = {}
        for k, i in enumerate(eff):
            tid = assign[k]
            cur = primary.get(tid)
            if cur is None or _area(boxes[i]) > _area(boxes[cur]):
                primary[tid] = i

        # 本帧未匹配到 person 的 track：只累加 miss 并记日志，**不在这里删除**。
        # 删除统一在下一帧开头按"距上次出现的时间"判定（见步骤 0），
        # 这样 1~2 FPS 下偶尔 1~2 帧漏检不会立刻丢号。
        for tid in list(tracks.keys()):
            if tid not in seen:
                tracks[tid]["miss"] += 1
                # 尚未确认的新 track 若连续 2 帧没命中，确认进度清零 —— 保证
                # "连续 N 帧确认"名副其实，偶发假阳性框永远不会转正。
                if (tracks[tid]["miss"] >= 2
                        and tracks[tid].get("confirm", 0) < TRACK_CONFIRM_FRAMES):
                    tracks[tid]["confirm"] = 0
                if TRACK_DEBUG:
                    _tr = tracks[tid]
                    _log_track(session_id, session, "miss", tid, _tr["bbox"], {
                        "miss": _tr["miss"],
                        "idle_s": round(now - (_tr.get("last_seen_t")
                                               or _tr.get("created_t") or now), 2),
                        "keep_s": TRACK_KEEP_S,
                        "confirm": _tr.get("confirm", 0),
                        "last_bbox": _fmt_box(_tr["bbox"])})

        # 2) 按 track 做多数投票 → 稳定状态 → 复用既有规则引擎重新判定
        stable_states = []
        stable_by_tid = {}
        for tid in sorted(primary):
            st = states[primary[tid]]
            tr = tracks[tid]
            ppe, in_roi, area_name, required, _ready = _stabilize(tr)
            sst = PersonState(person_id=tid, bbox=st.bbox, ppe=ppe,
                              in_roi=in_roi, area_name=area_name,
                              virtual=st.virtual)
            sst.area_required = required   # 由 annotate_person_states 同款字段
            stable_states.append(sst)
            stable_by_tid[tid] = sst
        stable_events = pipeline.engine.evaluate(stable_states,
                                                 frame_time=frame_time)
        stable_map = {ev.person_id: ev for ev in stable_events}

        # 3) 人员级违规事件管理（去重键 = session_id + track_id + 区域）
        #    一次连续违规 = 一条事件；active=True 期间不重复写库（与冷却无关）。
        #    未确认的新 track（命中不足 TRACK_CONFIRM_FRAMES 帧）不参与判定，
        #    避免偶发假阳性框立刻变成"人员2"并写出记录。
        for tid in sorted(primary):
            tr = tracks[tid]
            if tr.get("confirm", 0) < TRACK_CONFIRM_FRAMES:
                continue
            ev = stable_map.get(tid)
            if _update_episode(tr, ev, now, pipeline.cooldown_s):
                # 页面照常显示违规（含等级升级），但只有这里返回 True 才写库
                ev.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                new_events.append(ev)

        # 4) 组装响应所需的每人信息（业务字段 = 平滑后稳定结果）
        #    只有**已确认**的 track 才算正式人员：未确认的候选框放进
        #    pending_persons（页面不列为人员、不参与判定、不写库）。
        persons = []
        pending_persons = []
        for tid in sorted(primary):
            st = states[primary[tid]]
            tr = tracks[tid]
            sev = stable_map.get(tid)
            raw_ev = raw_map.get(st.person_id)
            sst = stable_by_tid[tid]
            confirmed = tr.get("confirm", 0) >= TRACK_CONFIRM_FRAMES
            if not confirmed:
                # 还在确认中：只回传编号与进度，供页面提示"确认中 n/N"
                pending_persons.append({
                    "track_id": f"T{tid}",
                    "confirm_frames": int(tr.get("confirm", 0)),
                    "bbox": [round(st.bbox.x1, 1), round(st.bbox.y1, 1),
                             round(st.bbox.x2, 1), round(st.bbox.y2, 1)],
                })
                continue
            persons.append({
                "person_id": tid,
                "track_id": f"T{tid}",
                "bbox": [round(st.bbox.x1, 1), round(st.bbox.y1, 1),
                         round(st.bbox.x2, 1), round(st.bbox.y2, 1)],
                "in_roi": bool(sst.in_roi),
                "area": sst.area_name or "普通区域",
                # 区域外人员不参与规则判定（与视频/图片口径一致）
                "evaluated": bool(sst.in_roi),
                "confirmed": True,
                "confirm_frames": TRACK_CONFIRM_FRAMES,
                # 稳定结果（业务判断用）
                "ppe": {k: v.to_dict() for k, v in sst.ppe.items()},
                "missing_ppe": list(sev.missing_ppe) if sev is not None else [],
                "violation_types": (list(sev.violation_types)
                                    if sev is not None else []),
                "severity": sev.severity if sev is not None else SEVERITY_NONE,
                # 单帧原始结果（仅供画框/对照，不参与业务判定）
                "raw_ppe": {k: v.to_dict() for k, v in st.ppe.items()},
                "raw_missing_ppe": (list(raw_ev.missing_ppe)
                                    if raw_ev is not None else []),
                "raw_severity": (raw_ev.severity
                                 if raw_ev is not None else SEVERITY_NONE),
                "stable": len(tr["win"]) >= SMOOTH_WINDOW,
                "window": len(tr["win"]),
                "virtual": bool(getattr(st, "virtual", False)),
                # 本次连续违规状态（同一次违规只对应数据库里的 1 条记录）
                "continuous_violation": bool(
                    tr["episodes"].get(sst.area_name or "默认区域", {})
                    .get("active", False)),
                "episode_seconds": (round(
                    now - tr["episodes"][sst.area_name or "默认区域"]["start_t"], 1)
                    if tr["episodes"].get(sst.area_name or "默认区域", {})
                    .get("active", False) else 0.0),
                "episode_events": int(
                    tr["episodes"].get(sst.area_name or "默认区域", {})
                    .get("events", 0)),
            })

        need_shot = bool(new_events)
        if need_shot:
            # 截图只在确定要写库时才生成（未去重掉的帧不产生磁盘文件）
            from video.processor import draw_annotations
            annotated = draw_annotations(img.copy(), stable_states,
                                         stable_events, areas)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            rel = f"rt_{ts}.jpg"
            ok = cv2.imwrite(str(pipeline.screenshot_dir / rel), annotated)
            for ev in new_events:
                ev.screenshot = rel if ok else ""

        session["frames"] += 1
        session["total_ms"] += elapsed_ms
        # 人数按"合并重复框 + 连续确认后的正式人员"统计
        session["max_persons"] = max(session["max_persons"], len(persons))
        session["violations"] += len(new_events)
        session["last_seen"] = now
        if new_events:
            # 与视频检测共用落库函数：violation_events 表结构与查询口径不变
            db.save_event_dicts(session_id, lab_id,
                                [e.to_dict() for e in new_events])
        db.update_detection_record_stats(
            session_id,
            person_count=session["max_persons"],
            violation_count=session["violations"],
            process_time_ms=round(session["total_ms"], 1))
        snapshot = dict(session)
        n_tracks = len(tracks)

    return {
        "session_id": session_id,
        "record_id": session_id,
        "elapsed_ms": round(elapsed_ms, 1),
        # 人数 = 合并重复框后的独立人员数（同一个人的多个框只算 1 人）
        "person_count": len(persons),
        "persons": persons,
        # 尚未连续确认的候选人员（不列为正式人员、不写库，仅用于页面提示）
        "pending_persons": pending_persons,
        "detections": [d.to_dict() for d in dets],
        "areas": [{"area_id": a.area_id, "name": a.name,
                   "x1": a.x1, "y1": a.y1, "x2": a.x2, "y2": a.y2,
                   "required_ppe": list(a.required_ppe)} for a in areas],
        "violation_count": sum(1 for p in persons
                               if p["evaluated"] and p["severity"] != SEVERITY_NONE),
        "new_events": [e.to_dict() for e in new_events],
        # 仅供参考：视频检测仍在用该冷却值；实时模式不再据此重复写库
        "cooldown_s": pipeline.cooldown_s,
        "frame_time": frame_time,
        "smooth": {
            "window": SMOOTH_WINDOW,
            "vote": VOTE_MIN,
            # 画面内无人时不显示"建立中"（否则空场景会一直提示建立状态）
            "ready": (all(p["stable"] for p in persons) if persons
                      else snapshot["frames"] >= SMOOTH_WINDOW),
            "recover_frames": RECOVER_FRAMES,
            "confirm_frames": VIOLATION_CONFIRM,
            # 新 track 需连续确认的帧数（未确认不写库）
            "track_confirm_frames": TRACK_CONFIRM_FRAMES,
            # 本帧尚未确认、还在观察中的候选人员数
            "pending": len(pending_persons),
            "tracks": n_tracks,
        },
        "session": {
            "frames": snapshot["frames"],
            "duration_s": round(now - snapshot["t0"], 1),
            "violation_count": snapshot["violations"],
            "max_person_count": snapshot["max_persons"],
            "avg_elapsed_ms": (round(snapshot["total_ms"] / snapshot["frames"], 1)
                               if snapshot["frames"] else 0.0),
        },
        "interval_ms_hint": DEFAULT_INTERVAL_MS,
    }


@router.post("/stop")
def realtime_stop(session_id: int = Form(0)):
    """
    结束实时巡检会话：回写检测记录统计并释放会话状态（含全部 track、
    滑动窗口与人员级去重状态）。重新开启巡检时会重新建立。
    """
    with _lock:
        session = _sessions.pop(session_id, None)
    if session is None:
        return {"ok": True, "message": "会话不存在或已结束",
                "session_id": session_id}
    duration = round(time.time() - session["t0"], 1)
    db.update_detection_record_stats(
        session_id,
        person_count=session["max_persons"],
        violation_count=session["violations"],
        process_time_ms=round(session["total_ms"], 1))
    return {
        "ok": True,
        "session_id": session_id,
        "frames": session["frames"],
        "duration_s": duration,
        "violation_count": session["violations"],
        "max_person_count": session["max_persons"],
        "cleared_tracks": len(session.get("tracks", {})),
        # 排查用：本会话的 track 生命周期日志（页面不展示）
        "track_log": (session.get("track_log") or [])[-50:],
    }


@router.get("/tracklog")
def realtime_tracklog(session_id: int = 0, limit: int = 100):
    """
    开发排查用：拉取指定会话的 track 生命周期日志（创建 / 匹配 / 丢失 / 删除，
    含对应 bbox 与 IoU / 中心距离）。

    用途：页面出现"人员1 → 人员2 → 人员1"编号跳变时，据此判断是同一个真人的
    track 断裂，还是画面里确实多检出了另一个 person（手机/屏幕/照片中的人）。
    该接口与日志均不在页面上展示；设置 LABSAFETY_TRACK_DEBUG=0 可停止写日志。
    """
    limit = max(1, min(int(limit), TRACK_LOG_MAX))
    with _lock:
        session = _sessions.get(session_id)
        if session is None:
            return {"ok": False, "error": "会话不存在或已结束",
                    "session_id": session_id, "entries": []}
        log = list(session.get("track_log") or [])
        # 当前仍存在的 active_violation（键 = "T{track}|{区域}"）
        active = {}
        for tid, tr in session["tracks"].items():
            for area, ep in tr["episodes"].items():
                active[f"T{tid}|{area}"] = {"active": bool(ep["active"]),
                                            "events": int(ep["events"])}
        # 每帧 person 框数量分布：判断"画面里到底有几个 person"最直接的依据
        box_hist = {}
        for e in log:
            if e.get("action") == "frame":
                n = e.get("n_boxes")
                box_hist[str(n)] = box_hist.get(str(n), 0) + 1
        lost = [{"tid": f"T{i['tid']}", "age_s": round(time.time() - i["t"], 1),
                 "bbox": _fmt_box(i["bbox"])} for i in (session.get("lost") or [])]
    return {"ok": True, "session_id": session_id, "count": len(log),
            "entries": log[-limit:], "active_violations": active,
            "person_box_hist": box_hist, "lost_tracks": lost}
