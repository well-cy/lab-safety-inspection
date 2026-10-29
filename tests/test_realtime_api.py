# -*- coding: utf-8 -*-
"""
实时摄像头巡检接口测试（新增独立文件，不改动 T01~T14 与 run_tests.py）。

为什么用"桩管线"（stub pipeline）而不是真实 YOLO：
    本文件测的是**实时巡检这一层**的逻辑（会话、ROI 复用、规则判定、冷却
    去重、截图落盘、写入既有违规表），而不是模型精度。桩管线直接复用真实的
    match_person_ppe + annotate_person_states + SafetyRuleEngine，因此 ROI
    归一化判断与违规等级判定仍然是**真实代码路径**，只是把 YOLO 推理换成
    固定的假检测框 —— 测试不需要 GPU / 权重，几秒内跑完。

运行（必须使用临时数据库，禁止污染 backend/labsafety.db）：
    set LABSAFETY_DB_PATH=%TEMP%\\realtime_test.db
    venv\\Scripts\\python -m pytest tests/test_realtime_api.py -v
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from ai.detector import Detection  # noqa: E402
from ai.ppe_matcher import match_person_ppe  # noqa: E402
from backend import database as db  # noqa: E402
from backend import realtime as rt  # noqa: E402
from backend.main import app  # noqa: E402
from rule_engine.engine import (  # noqa: E402
    SafetyRuleEngine, annotate_person_states, SEVERITY_MAJOR,
)

FRAME_W, FRAME_H = 640, 480


class StubPipeline:
    """替身管线：固定检测框 + 真实关联/ROI/规则引擎"""

    def __init__(self, screenshot_dir, person_center, ppe=("lab_coat",),
                 cooldown_s=10.0):
        self.engine = SafetyRuleEngine(1)
        self.cooldown_s = cooldown_s
        self.screenshot_dir = Path(screenshot_dir)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.person_center = person_center
        self.ppe = list(ppe)
        self.calls = 0

    def analyze_frame(self, frame, areas, frame_time=0.0):
        self.calls += 1
        h, w = frame.shape[:2]
        cx, cy = self.person_center
        px1, py1 = cx * w - 50, cy * h - 100
        px2, py2 = cx * w + 50, cy * h + 100
        dets = [Detection("person", "person", 0.9, px1, py1, px2, py2)]
        for i, name in enumerate(self.ppe):
            dets.append(Detection(name, name, 0.9,
                                  px1 + 5, py1 + 20 + 30 * i,
                                  px1 + 45, py1 + 50 + 30 * i))
        states = match_person_ppe(dets)
        states = annotate_person_states(states, areas, w, h)
        events = self.engine.evaluate(states, frame_time=frame_time)
        return states, events, dets


def make_jpeg(w=FRAME_W, h=FRAME_H, color=(120, 120, 120)):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """每个用例独立：清空会话状态、把截图写到临时目录、注入桩管线"""
    rt._sessions.clear()
    pipeline = StubPipeline(tmp_path / "shots", person_center=(0.5, 0.5))
    rt.init_realtime(lambda: pipeline)
    with TestClient(app) as c:
        yield c, pipeline
    rt._sessions.clear()   # 不重置管线获取函数：下一个用例会重新注入


def post_frame(client, lab_id=1, session_id=0, raw=None):
    payload = make_jpeg() if raw is None else raw   # raw=b"" 要真发出去（测空帧）
    return client.post(
        "/api/realtime/detect",
        files={"frame": ("frame.jpg", payload, "image/jpeg")},
        data={"lab_id": str(lab_id), "session_id": str(session_id)},
    )


def count_events(source_type="realtime"):
    conn = db.get_conn()
    n = conn.execute(
        "SELECT COUNT(*) c FROM violation_events v"
        " JOIN detection_records d ON d.id = v.record_id"
        " WHERE d.source_type = ?", (source_type,)).fetchone()["c"]
    conn.close()
    return int(n)


def test_detect_creates_session_and_writes_violation(client):
    """
    开会话 → 检出 ROI 内人员 → 缺 PPE → 人员连续 3 帧确认 + 违规连续确认后
    写违规事件 + 存截图（前几帧仍在建立稳定状态，按设计不立即写库）
    """
    c, pipeline = client
    before = count_events()
    r = post_frame(c)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["session_id"] > 0
    sid = body["session_id"]
    # 新 track 需**连续 TRACK_CONFIRM_FRAMES 帧**确认后才算正式人员：
    # 前几帧只作为"候选人员"返回（pending），不列人员、不写违规记录
    assert body["person_count"] == 0
    assert body["smooth"]["pending"] == 1
    # 第 2 帧仍在确认中；第 3 帧成为正式人员（此时稳定违规尚未连续确认）
    for _ in range(rt.TRACK_CONFIRM_FRAMES - 2):
        assert post_frame(c, session_id=sid).json()["new_events"] == []
    body = post_frame(c, session_id=sid).json()          # 第 3 帧：已确认
    assert body["person_count"] == 1
    assert body["persons"][0]["in_roi"] is True
    assert body["persons"][0]["area"] == "实验操作区"
    # 只戴了实验服 → 缺 mask / gloves → 严重违规（稳定判定）
    assert body["persons"][0]["severity"] == SEVERITY_MAJOR
    assert body["persons"][0]["track_id"] == "T1"
    assert body["new_events"] == []          # 稳定违规尚未连续确认
    r2 = post_frame(c, session_id=sid).json()
    assert len(r2["new_events"]) == 1        # 违规连续确认后落库
    assert count_events() == before + 1
    # 截图确实落盘
    shot = r2["new_events"][0]["screenshot"]
    assert shot.startswith("rt_") and shot.endswith(".jpg")
    assert (pipeline.screenshot_dir / shot).stat().st_size > 0


def test_continuous_violation_prevents_duplicate_events(client):
    """连续多帧同类违规：同一次连续违规只写 1 条（实时模式不再按冷却补写）"""
    c, _ = client
    r1 = post_frame(c)
    sid = r1.json()["session_id"]
    assert r1.json()["new_events"] == []
    # 前 3 帧用于"新人员连续确认"，之后才进入违规确认
    for _ in range(rt.TRACK_CONFIRM_FRAMES - 1):
        assert post_frame(c, session_id=sid).json()["new_events"] == []
    r2 = post_frame(c, session_id=sid)
    assert len(r2.json()["new_events"]) == 1
    before = count_events()
    for _ in range(6):                       # 连续 6 帧，远快于冷却时间
        body = post_frame(c, session_id=sid).json()
        assert body["new_events"] == []
    assert count_events() == before          # 一条都没多写
    assert post_frame(c, session_id=sid).json()["session"]["violation_count"] == 1


def test_person_outside_roi_is_not_evaluated(client, monkeypatch, tmp_path):
    """人员在 ROI 之外：不套用实验操作区规则，不产生违规"""
    c, _ = client
    rt._sessions.clear()
    rt.init_realtime(lambda: StubPipeline(tmp_path / "shots2",
                                          person_center=(0.02, 0.02)))
    before = count_events()
    body = post_frame(c).json()
    sid = body["session_id"]
    # 新 track 需连续确认 N 帧才成为正式人员（确认前只算候选）
    for _ in range(rt.TRACK_CONFIRM_FRAMES - 1):
        post_frame(c, session_id=sid)
    body = post_frame(c, session_id=sid).json()
    assert body["person_count"] == 1
    assert body["persons"][0]["in_roi"] is False
    assert body["persons"][0]["area"] == "普通区域"
    assert body["violation_count"] == 0
    assert body["new_events"] == []
    assert count_events() == before


def test_session_reuse_and_stop(client):
    """会话复用：多帧共用一条 detection_records；stop 后回写统计"""
    c, _ = client
    sid = post_frame(c).json()["session_id"]
    for _ in range(3):
        post_frame(c, session_id=sid)
    r = c.post("/api/realtime/stop", data={"session_id": str(sid)})
    assert r.status_code == 200 and r.json()["ok"] is True
    assert r.json()["frames"] == 4
    conn = db.get_conn()
    rec = conn.execute("SELECT * FROM detection_records WHERE id=?",
                       (sid,)).fetchone()
    conn.close()
    assert rec["source_type"] == "realtime"
    assert rec["person_count"] == 1          # 会话内最大人数已回写
    assert rec["violation_count"] == 1       # 去重后只记 1 条


def test_invalid_inputs_return_4xx(client):
    """非法输入返回 4xx 而不是 500"""
    c, _ = client
    assert post_frame(c, lab_id=99999).status_code == 400      # 实验室不存在
    assert post_frame(c, raw=b"").status_code == 400           # 空帧
    assert post_frame(c, raw=b"not-an-image").status_code == 400  # 非图片
