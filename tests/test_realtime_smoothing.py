# -*- coding: utf-8 -*-
"""
实时巡检「多帧平滑 + 人员级违规去重」专项测试（新增独立文件）。

背景（Windows 实机复现的真实问题）：
    摄像头单帧 YOLO 存在偶发假阳性，直接用单帧判定会让"缺口罩 →
    缺手套+口罩 → 缺手套+实验服+口罩"来回跳变，并在几秒内写出多条记录。
    本文件验证实时链路新增的：滑动窗口多数投票、人员短时 track 匹配、
    违规恢复多帧确认、人员级去重（session_id + track_id + 区域）。

为什么用"脚本化桩管线"：
    用例要的是**可控的逐帧检测结果**（例如"5 帧里只有 1 帧误检 gloves"），
    真实 YOLO 无法稳定复现。桩管线仍复用真实的 match_person_ppe +
    annotate_person_states + SafetyRuleEngine，所以 ROI 归一化判断与
    违规等级判定走的是真实代码路径；只有 YOLO 推理被替换成脚本化检测框。

运行（必须使用临时数据库，禁止污染 backend/labsafety.db）：
    set LABSAFETY_DB_PATH=%TEMP%\\realtime_smooth_test.db
    venv\\Scripts\\python -m pytest tests/test_realtime_smoothing.py -v
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from ai.detector import Detection  # noqa: E402
from ai.ppe_matcher import PersonState, match_person_ppe  # noqa: E402
from backend import database as db  # noqa: E402
from backend import realtime as rt  # noqa: E402
from backend.main import app  # noqa: E402
from rule_engine.engine import (  # noqa: E402
    SafetyRuleEngine, ViolationEvent, annotate_person_states,
    SEVERITY_MAJOR, SEVERITY_MINOR, SEVERITY_NONE,
)

FRAME_W, FRAME_H = 640, 480
CENTER = (0.5, 0.5)          # 位于默认 ROI(0.22,0.18,0.88,0.92) 内
NO_PPE = ()                  # 什么都不戴
FULL_PPE = ("mask", "gloves", "lab_coat")


class ScriptedPipeline:
    """按脚本逐帧返回固定检测结果的替身管线"""

    def __init__(self, screenshot_dir, cooldown_s=10.0):
        self.engine = SafetyRuleEngine(1)
        self.cooldown_s = cooldown_s
        self.screenshot_dir = Path(screenshot_dir)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.script = []     # 每帧：[(中心点, (ppe...)), ...]
        self.i = 0

    def set_script(self, script):
        self.script = script
        self.i = 0

    def analyze_frame(self, frame, areas, frame_time=0.0):
        h, w = frame.shape[:2]
        spec = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        dets = []
        for (cx, cy), ppe in spec:
            px1, py1 = cx * w - 50, cy * h - 100
            px2, py2 = cx * w + 50, cy * h + 100
            dets.append(Detection("person", "person", 0.9, px1, py1, px2, py2))
            for i, name in enumerate(ppe):
                dets.append(Detection(name, name, 0.9,
                                      px1 + 5, py1 + 20 + 30 * i,
                                      px1 + 45, py1 + 50 + 30 * i))
        states = match_person_ppe(dets)
        states = annotate_person_states(states, areas, w, h)
        events = self.engine.evaluate(states, frame_time=frame_time)
        return states, events, dets


class BoxPipeline:
    """
    按脚本返回**指定 bbox** 的 person（不跑 YOLO）。

    用于验证与框几何相关的逻辑（重复框合并、尺寸剧变、单人连续性），
    这些用固定尺寸的 ScriptedPipeline 无法表达。仍复用真实的
    annotate_person_states + SafetyRuleEngine，ROI 与违规判定是真实口径。
    """

    def __init__(self, screenshot_dir, cooldown_s=10.0):
        self.engine = SafetyRuleEngine(1)
        self.cooldown_s = cooldown_s
        self.screenshot_dir = Path(screenshot_dir)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.script = []      # 每帧：[((x1,y1,x2,y2), (ppe...)), ...]
        self.i = 0

    def set_script(self, script):
        self.script = script
        self.i = 0

    def analyze_frame(self, frame, areas, frame_time=0.0):
        h, w = frame.shape[:2]
        spec = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        dets, states = [], []
        for k, (box, ppe) in enumerate(spec):
            d = Detection("person", "person", 0.9, *box)
            dets.append(d)
            st = PersonState(person_id=k + 1, bbox=d, ppe={})
            for i, name in enumerate(ppe):
                pd = Detection(name, name, 0.9, box[0] + 5, box[1] + 20 + 30 * i,
                               box[0] + 45, box[1] + 50 + 30 * i)
                dets.append(pd)
                st.ppe[name] = pd
            states.append(st)
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
def client(tmp_path):
    """每个用例独立：清空会话、注入脚本化桩管线、截图写临时目录"""
    rt._sessions.clear()
    # 清空上一用例留下的实时记录，使 count_events() 从 0 开始（隔离断言）
    _conn = db.get_conn()
    _conn.execute("DELETE FROM violation_events WHERE record_id IN"
                  " (SELECT id FROM detection_records WHERE source_type='realtime')")
    _conn.execute("DELETE FROM detection_records WHERE source_type='realtime'")
    _conn.commit()
    _conn.close()
    pipeline = ScriptedPipeline(tmp_path / "shots")
    rt.init_realtime(lambda: pipeline)
    with TestClient(app) as c:
        yield c, pipeline
    rt._sessions.clear()


def post_frame(c, lab_id=1, session_id=0):
    return c.post(
        "/api/realtime/detect",
        files={"frame": ("frame.jpg", make_jpeg(), "image/jpeg")},
        data={"lab_id": str(lab_id), "session_id": str(session_id)},
    )


def run_script(c, pipeline, script, sid=0, clock=None, step_s=1.5):
    """
    按顺序发帧，返回每帧响应 body 列表。

    clock 不为 None 时使用可控时钟（monkeypatch 过的 time.time 状态字典），
    每帧推进 step_s 秒 —— 这样"持续违规 30 秒"能在毫秒级测试里真实复现，
    用来验证实时模式不再按 10 秒冷却重复写库。
    """
    pipeline.set_script(script)
    bodies = []
    for _ in script:
        if clock is not None:
            clock["t"] += step_s
        body = post_frame(c, session_id=sid).json()
        sid = body["session_id"]
        bodies.append(body)
    return bodies


@pytest.fixture()
def clock(monkeypatch):
    """可控时钟：把 time.time 换成手动推进的假时钟（用例结束自动还原）"""
    state = {"t": time.time()}
    monkeypatch.setattr(rt.time, "time", lambda: state["t"])
    return state


def count_events():
    conn = db.get_conn()
    n = conn.execute(
        "SELECT COUNT(*) c FROM violation_events v"
        " JOIN detection_records d ON d.id = v.record_id"
        " WHERE d.source_type='realtime'").fetchone()["c"]
    conn.close()
    return int(n)


def one_person(ppe, center=CENTER):
    return [(center, ppe)]


# ---------------- 单元测试：投票阈值 / 短时匹配 / 事件管理 ----------------

def test_vote_thresholds():
    """1 帧直接用；2 帧需一致；3~5 帧取严格多数（5 帧需 ≥3）"""
    assert rt._vote([True]) is True
    assert rt._vote([False]) is False
    assert rt._vote([True, False]) is False
    assert rt._vote([True, True]) is True
    assert rt._vote([False, True, True]) is True        # 3 帧需 ≥2
    assert rt._vote([True, False, False]) is False
    assert rt._vote([True, False, False, True, True]) is True    # 5 帧 3 次
    assert rt._vote([True, False, False, False, True]) is False  # 5 帧 2 次


def test_track_matching_by_iou_and_center():
    """上一帧 person 框与当前框：IoU≥0.3 或中心距离≤0.5 对角线判为同一人"""
    tracks = {1: {"bbox": (100, 100, 200, 300)},
              2: {"bbox": (400, 100, 500, 300)}}
    # 轻微移动 → 仍匹配到原 track
    assert rt._match_tracks(tracks, [(105, 105, 205, 305),
                                     (410, 95, 510, 295)]) == [1, 2]
    # 远处出现的新框 → 不匹配（调用方会新建 track）
    assert rt._match_tracks(tracks, [(20, 20, 60, 60)]) == [None]
    # 位置几乎重合但被挤开的两人：不被互相抢走
    assert rt._match_tracks(tracks, [(108, 102, 208, 302)]) == [1]


def test_continuous_violation_writes_one_event_until_recovered():
    """
    一次连续违规 = 一条事件：
    确认后无论持续 1 秒 / 11 秒 / 60 秒 / 300 秒都不再写库（与 10 秒冷却无关），
    只有连续 RECOVER_FRAMES 帧正常解除 active 后，才允许写下一条。
    """
    tr = rt._new_track(1)
    now = time.time()
    ev = ViolationEvent(person_id=1, area_name="实验操作区",
                        violation_types=["未佩戴口罩"], missing_ppe=["mask"],
                        severity=SEVERITY_MINOR, timestamp="")
    assert rt._update_episode(tr, ev, now, 10.0) is False   # 第 1 帧：仅累积
    assert rt._update_episode(tr, ev, now, 10.0) is True    # 第 2 帧：确认写库
    ep = tr["episodes"]["实验操作区"]
    assert ep["active"] is True and ep["events"] == 1
    # 持续违规：远超 10 秒冷却也不再写（这是本次修复的核心）
    for dt in (0.4, 1.0, 5.0, 11.0, 30.0, 60.0, 300.0):
        assert rt._update_episode(tr, ev, now + dt, 10.0) is False
    assert ep["active"] is True and ep["events"] == 1
    # 恢复正常：需连续 3 帧稳定正常才解除
    assert rt._update_episode(tr, None, now + 301, 10.0) is False  # 正常第 1 帧
    assert ep["active"] is True
    assert rt._update_episode(tr, None, now + 302, 10.0) is False  # 正常第 2 帧
    assert ep["active"] is True
    assert rt._update_episode(tr, None, now + 303, 10.0) is False  # 第 3 帧 → 解除
    assert ep["active"] is False
    # 解除后再次违规：才允许新建第 2 条
    assert rt._update_episode(tr, ev, now + 304, 10.0) is False
    assert rt._update_episode(tr, ev, now + 305, 10.0) is True
    assert ep["active"] is True and ep["events"] == 2


def test_ppe_and_severity_change_within_one_episode_not_written():
    """
    同一次连续违规内 一般违规 ⇄ 严重违规 来回切换：
    页面显示等级立即跟随（内存态），但不新增数据库记录（即使超过 10 秒冷却）。
    """
    tr = rt._new_track(1)
    now = time.time()
    minor = ViolationEvent(person_id=1, area_name="实验操作区",
                           violation_types=["未佩戴口罩"], missing_ppe=["mask"],
                           severity=SEVERITY_MINOR, timestamp="")
    major = ViolationEvent(person_id=1, area_name="实验操作区",
                           violation_types=["未佩戴口罩", "未佩戴手套"],
                           missing_ppe=["mask", "gloves"],
                           severity=SEVERITY_MAJOR, timestamp="")
    assert rt._update_episode(tr, minor, now, 10.0) is False
    assert rt._update_episode(tr, minor, now, 10.0) is True        # 第 1 条
    assert rt._update_episode(tr, major, now + 1, 10.0) is False   # 升级
    assert tr["episodes"]["实验操作区"]["severity"] == SEVERITY_MAJOR  # 页面已升级
    assert rt._update_episode(tr, minor, now + 12, 10.0) is False  # 降级
    assert tr["episodes"]["实验操作区"]["severity"] == SEVERITY_MINOR  # 页面已降级
    assert rt._update_episode(tr, major, now + 25, 10.0) is False  # 再升级
    assert rt._update_episode(tr, major, now + 60, 10.0) is False  # 超冷却也不写
    assert tr["episodes"]["实验操作区"]["events"] == 1


# ---------------- 场景测试：单帧误检不改变稳定结果 ----------------

def test_single_frame_gloves_false_positive_ignored(client):
    """5 帧中只有 1 帧误检 gloves → 稳定结果仍为"未佩戴手套"（严重违规）"""
    c, pipeline = client
    bodies = run_script(c, pipeline, [
        one_person(("lab_coat",)),
        one_person(("lab_coat",)),
        one_person(("lab_coat", "gloves")),   # ← 单帧假阳性
        one_person(("lab_coat",)),
        one_person(("lab_coat",)),
    ])
    blip = bodies[2]
    assert "gloves" in blip["persons"][0]["raw_ppe"]      # 单帧原始确实检到了
    assert blip["persons"][0]["raw_severity"] == SEVERITY_MINOR
    assert blip["persons"][0]["severity"] == SEVERITY_MAJOR  # 稳定结果未受影响

    last = bodies[-1]
    assert "gloves" not in last["persons"][0]["ppe"]
    assert sorted(last["persons"][0]["missing_ppe"]) == ["gloves", "mask"]
    assert last["persons"][0]["severity"] == SEVERITY_MAJOR
    assert last["smooth"]["ready"] is True


def test_four_of_five_frames_gloves_confirmed(client):
    """5 帧中 4 帧检出 gloves → 稳定结果认定"已佩戴手套" """
    c, pipeline = client
    bodies = run_script(c, pipeline, [
        one_person(("lab_coat", "gloves")),
        one_person(("lab_coat", "gloves")),
        one_person(("lab_coat", "gloves")),
        one_person(("lab_coat",)),            # ← 单帧漏检
        one_person(("lab_coat", "gloves")),
    ])
    last = bodies[-1]
    assert "gloves" in last["persons"][0]["ppe"]
    assert last["persons"][0]["missing_ppe"] == ["mask"]
    assert last["persons"][0]["severity"] == SEVERITY_MINOR


def test_single_frame_lab_coat_false_positive_not_applied(client):
    """单帧 lab_coat 误检：不能立即改变稳定结果"""
    c, pipeline = client
    bodies = run_script(c, pipeline, [
        one_person(NO_PPE),
        one_person(NO_PPE),
        one_person(("lab_coat",)),            # ← 单帧误检实验服
        one_person(NO_PPE),
        one_person(NO_PPE),
    ])
    blip = bodies[2]
    assert "lab_coat" in blip["persons"][0]["raw_ppe"]
    assert blip["persons"][0]["raw_severity"] == SEVERITY_MAJOR
    assert "lab_coat" not in blip["persons"][0]["ppe"]      # 稳定结果仍缺实验服
    assert sorted(bodies[-1]["persons"][0]["missing_ppe"]) == \
        ["gloves", "lab_coat", "mask"]


# ---------------- 场景测试：抖动不产生多条数据库记录 ----------------

def test_severity_jitter_writes_single_event(client):
    """单帧在"缺口罩"与"缺手套+口罩"之间抖动 10 帧 → 数据库只写 1 条"""
    c, pipeline = client
    before = count_events()
    script = []
    for i in range(10):
        script.append(one_person(("lab_coat",) if i % 2 == 0
                                 else ("mask", "lab_coat")))
    bodies = run_script(c, pipeline, script)
    # 前 TRACK_CONFIRM_FRAMES-1 帧人员尚未确认（persons 为空），跳过
    raw = {b["persons"][0]["raw_severity"] for b in bodies if b["persons"]}
    assert len(raw) == 2                       # 单帧结果确实在抖
    assert count_events() == before + 1        # 但只写了一条
    assert bodies[-1]["session"]["violation_count"] == 1


def test_full_ppe_sequence_recovery_and_re_violation(client):
    """
    场景 4/5/6/7：依次戴上实验服 → 口罩 → 手套（违规解除并多帧确认）
    → 再脱掉手套 → 只有 2 条记录（首次 1 条 + 重新违规 1 条）
    """
    c, pipeline = client
    before = count_events()
    script = []
    script += [one_person(NO_PPE)] * 4                  # 缺 3 项 → 严重违规
    script += [one_person(("lab_coat",))] * 5           # 穿上实验服
    script += [one_person(("mask", "lab_coat"))] * 5    # 戴上口罩
    script += [one_person(FULL_PPE)] * 5                # 戴上手套 → 全部合规
    script += [one_person(("mask", "lab_coat"))] * 5    # 再脱掉手套
    bodies = run_script(c, pipeline, script)

    assert bodies[3]["persons"][0]["severity"] == SEVERITY_MAJOR
    assert sorted(bodies[3]["persons"][0]["missing_ppe"]) == \
        ["gloves", "lab_coat", "mask"]
    assert bodies[13]["persons"][0]["missing_ppe"] == ["gloves"]   # 只剩手套
    assert bodies[18]["persons"][0]["severity"] == SEVERITY_NONE   # 全部合规
    assert bodies[23]["persons"][0]["severity"] == SEVERITY_MINOR  # 重新违规
    assert count_events() == before + 2
    assert bodies[-1]["session"]["violation_count"] == 2


# -------- 场景测试：一次连续违规 = 一条事件（可控时钟：每帧 1.5 秒） --------

def test_thirty_seconds_continuous_violation_writes_one_event(client, clock):
    """
    验收 1/2/3：一人持续违规 36 秒，期间缺失项与等级反复变化
    （缺3种 → 缺2种 → 缺1种 → 缺3种），数据库只新增 1 条记录，
    且不因超过 10 秒冷却而补写第二条。
    """
    c, pipeline = client
    before = count_events()
    script = ([one_person(NO_PPE)] * 6                  # 缺 3 种 → 严重
              + [one_person(("lab_coat",))] * 6         # 缺 2 种 → 严重
              + [one_person(("mask", "lab_coat"))] * 6  # 缺 1 种 → 一般
              + [one_person(NO_PPE)] * 6)               # 又缺 3 种 → 严重
    bodies = run_script(c, pipeline, script, clock=clock, step_s=1.5)
    assert count_events() == before + 1                 # ← 核心断言
    # 页面状态确实跟随稳定结果变化（缺失项集合、违规等级都变过）
    shown = [b for b in bodies if b["persons"]]     # 人员确认后才进 persons
    miss_seen = {tuple(sorted(b["persons"][0]["missing_ppe"])) for b in shown}
    sev_seen = {b["persons"][0]["severity"] for b in shown}
    assert len(miss_seen) >= 3
    assert SEVERITY_MAJOR in sev_seen and SEVERITY_MINOR in sev_seen
    # 除首次确认那一帧外，全程不再产生新事件
    assert sum(len(b["new_events"]) for b in bodies) == 1
    last = bodies[-1]
    assert last["session"]["violation_count"] == 1
    assert last["persons"][0]["continuous_violation"] is True
    assert last["persons"][0]["episode_events"] == 1
    assert last["persons"][0]["episode_seconds"] >= 30


def test_recover_then_reviolate_writes_second_event(client, clock):
    """验收 4/5：连续 3 帧正常解除 active → 再次违规才写第 2 条"""
    c, pipeline = client
    before = count_events()
    script = ([one_person(NO_PPE)] * 5           # 违规 → 第 1 条
              + [one_person(FULL_PPE)] * 8       # 全部合规 → 解除
              + [one_person(NO_PPE)] * 5)        # 再次违规 → 第 2 条
    bodies = run_script(c, pipeline, script, clock=clock, step_s=1.5)
    assert count_events() == before + 2
    assert bodies[-1]["session"]["violation_count"] == 2
    assert any(b["persons"] and b["persons"][0]["severity"] == SEVERITY_NONE
               for b in bodies)
    assert bodies[-1]["persons"][0]["episode_events"] == 2


def test_restart_session_records_new_event_again(client, clock):
    """验收 8：停止后重新开启巡检 → 新会话不继承旧状态，重新开始记录"""
    c, pipeline = client
    before = count_events()
    b1 = run_script(c, pipeline, [one_person(NO_PPE)] * 4, clock=clock)
    sid = b1[-1]["session_id"]
    c.post("/api/realtime/stop", data={"session_id": str(sid)})
    b2 = run_script(c, pipeline, [one_person(NO_PPE)] * 4, clock=clock, sid=0)
    assert b2[-1]["session_id"] != sid
    assert count_events() == before + 2            # 旧会话 1 条 + 新会话 1 条
    assert b2[-1]["persons"][0]["track_id"] == "T1"
    assert b2[-1]["persons"][0]["episode_events"] == 1
    assert b2[-1]["session"]["violation_count"] == 1


def test_box_size_change_still_matches_same_track():
    """
    框尺寸剧变（半身 → 全身，面积 4 倍）导致 IoU 直接归零时，仍应匹配到
    同一个 track：不能只依赖 IoU，还要看中心距 + 面积变化是否合理。
    """
    prev = (300, 200, 400, 400)      # 100x200，中心 (350,300)
    cur = (420, 100, 620, 500)       # 200x400，中心 (520,300)，面积 4 倍
    assert rt._iou(prev, cur) < rt.MATCH_IOU          # IoU 确实不达标
    assert rt._shape_compatible(prev, cur)            # 但面积变化合理
    assert 0.5 < rt._center_ratio_sym(prev, cur) <= rt.MATCH_CENTER_RATIO_LOOSE
    tracks = {1: {"bbox": prev, "miss": 0}}
    assert rt._match_tracks(tracks, [cur]) == [1]     # 靠放宽的中心距阈值匹配上
    # 面积变化超出合理范围（0.1 倍）时不再放宽，避免把远处小人误并进来
    far = (420, 100, 460, 160)       # 40x60，面积仅 0.06 倍
    assert not rt._shape_compatible(prev, far)


def test_single_person_large_move_keeps_t1_via_solo_reclaim(client, clock):
    """
    单人场景 person 框位置大幅跳变（超出所有匹配阈值）时，单人安全回收策略
    必须复用 T1，不能新建 T2 —— 对应"人员移动/伸手后编号从 T1 跳到 T2"。
    """
    c, pipeline = client
    near, far = (0.5, 0.5), (0.15, 0.5)      # 帧间位移约 224px，超出阈值
    script = [[(near, NO_PPE)], [(near, NO_PPE)], [(far, NO_PPE)],
              [(near, NO_PPE)], [(near, NO_PPE)]]
    bodies = run_script(c, pipeline, script, clock=clock, step_s=1.0)
    tids = {p["track_id"] for body in bodies for p in body["persons"]}
    assert tids == {"T1"}, f"单人大幅移动后仍应只有 T1，实际 {tids}"

    sid = list(rt._sessions)[0]
    log = c.get(f"/api/realtime/tracklog?session_id={sid}&limit=200").json()
    solo = [e for e in log["entries"] if e["action"] == "solo_reclaim"]
    assert solo, "大幅跳变应触发单人回收并记录排查日志"
    assert solo[0]["prev_bbox"] != "-"
    # 回收后仍是同一个 track，违规状态随之延续，不会写成第 2 条
    assert sum(len(b["new_events"]) for b in bodies) <= 1
    assert count_events() <= 1


def test_single_person_gap_1_to_3_seconds_keeps_t1(client, clock):
    """
    单人短暂漏检 1~3 秒（未超过 TRACK_KEEP_S）后再次检出：继续用 T1，
    不能新建 T2，也不能再写第 2 条违规。
    """
    c, pipeline = client
    script = [one_person(NO_PPE)] * 4          # 已确认违规并写库
    bodies = run_script(c, pipeline, script, clock=clock, step_s=1.0)
    assert sum(len(b["new_events"]) for b in bodies) == 1
    sid = bodies[-1]["session_id"]

    # 漏检 3 秒（3 帧 × 1s），小于 TRACK_KEEP_S=5s
    bodies += run_script(c, pipeline, [[]] * 3, sid=sid, clock=clock, step_s=1.0)
    # 再次检出同一人
    bodies += run_script(c, pipeline, [one_person(NO_PPE)] * 3, sid=sid,
                         clock=clock, step_s=1.0)
    tids = {p["track_id"] for b in bodies for p in b["persons"]}
    assert tids == {"T1"}, f"漏检 3 秒后仍应复用 T1，实际 {tids}"
    assert sum(len(b["new_events"]) for b in bodies) == 1
    assert count_events() == 1


def test_person_leaving_longer_than_keep_seconds_creates_new_track(client,
                                                                   clock):
    """
    人员真正离开超过 TRACK_KEEP_S（5 秒）后才允许建立新 track；
    且若为同一次连续违规（未恢复正常），回收后仍不重复写库。
    """
    c, pipeline = client
    gap = int(rt.TRACK_KEEP_S) + 4          # 明显超过保留时长
    script = ([one_person(NO_PPE)] * 4 + [[]] * gap
              + [one_person(NO_PPE)] * 4)
    bodies = run_script(c, pipeline, script, clock=clock, step_s=1.0)
    sid = list(rt._sessions)[0]
    log = c.get(f"/api/realtime/tracklog?session_id={sid}&limit=400").json()
    deletes = [e for e in log["entries"] if e["action"] == "delete"]
    assert deletes, "超过保留时长后应删除 track"
    assert all(d["reason"] == "idle_exceeded_keep_seconds" for d in deletes)
    # 关键：即便换了 track 号，同一次连续违规也只写 1 条（状态被回收继承）
    assert sum(len(b["new_events"]) for b in bodies) == 1
    assert count_events() == 1


def test_two_persons_not_merged_by_solo_reclaim(client):
    """
    两人同时在场：单人宽松回收策略不得启用，不能把两个人合并成同一 track。
    """
    c, pipeline = client
    left, right = (0.35, 0.5), (0.65, 0.5)
    script = [[left and (left, NO_PPE), (right, NO_PPE)]] * 5
    bodies = run_script(c, pipeline, script)
    tids = {p["track_id"] for b in bodies for p in b["persons"]}
    assert len(tids) == 2, f"两人应分别为两个 track，实际 {tids}"
    sid = list(rt._sessions)[0]
    log = c.get(f"/api/realtime/tracklog?session_id={sid}&limit=200").json()
    assert not [e for e in log["entries"] if e["action"] == "solo_reclaim"], \
        "多人场景不得触发单人回收"


def test_track_break_does_not_duplicate_violation_event(client, clock):
    """
    track 断裂回收：同一个真人因漏检被 delete、再被当成新 track 建回来时，
    必须继承原 episode 的 active 状态 —— 否则同一次连续违规会被写成两条。

    脚本：违规 4 帧 → 完全漏检 7 帧（超过 TRACK_MAX_MISS，track 被删除）
          → 同一位置再次出现并持续违规 4 帧。
    """
    c, pipeline = client
    gap = rt.TRACK_MAX_MISS + 2
    script = ([one_person(NO_PPE)] * 4 + [[]] * gap
              + [one_person(NO_PPE)] * 4)
    bodies = run_script(c, pipeline, script, clock=clock)
    written = sum(len(b["new_events"]) for b in bodies)
    assert written == 1, f"断裂前后合计只应写 1 条，实际 {written}"
    assert count_events() == 1

    sid = list(rt._sessions)[0]
    log = c.get(f"/api/realtime/tracklog?session_id={sid}&limit=400").json()
    created = [e for e in log["entries"] if e["action"] == "create"]
    assert len(created) == 2                       # 断裂后确实新建了第二个 track
    assert created[1]["inherited"] is True         # 并继承了原 episode 状态
    reclaimed = [e for e in log["entries"] if e["action"] == "reclaim"]
    assert reclaimed and reclaimed[0]["from_track"] == "T1"


def test_duplicate_person_box_detection_and_frame_log(client):
    """
    同帧重复框防御：_dup_pairs 能识别高度重合的 person 框（同一个人的重复框），
    真实链路中 match_person_ppe 会先合并掉它们，帧日志如实记录框数量与重合对。
    用于排查"人员1 + 人员2 同时出现"到底是两个人还是重复框。
    """
    # 1) 重合框 → 判为重复框；分离框 → 判为两个人
    dup = rt._dup_pairs([(100, 100, 200, 300), (106, 100, 206, 300),
                         (400, 100, 500, 300)])
    assert [d[2] for d in dup] and dup[0][0] == 0 and dup[0][1] == 1
    assert all(d[2] >= rt.DUP_BOX_IOU for d in dup)
    assert rt._dup_pairs([(100, 100, 200, 300), (400, 100, 500, 300)]) == []

    # 2) 帧级日志记录本帧框数量（画面里到底检出几个 person 的第一手依据）
    c, pipeline = client
    run_script(c, pipeline, [one_person(NO_PPE)] * 3)
    sid = list(rt._sessions)[0]
    log = c.get(f"/api/realtime/tracklog?session_id={sid}&limit=200").json()
    frames = [e for e in log["entries"] if e["action"] == "frame"]
    assert len(frames) == 3 and all(f["n_boxes"] == 1 for f in frames)
    assert log["person_box_hist"] == {"1": 3}


def test_close_range_torso_box_is_merged_before_track_matching():
    """
    近距离"全身框 + 躯干框"：小框几乎完全落在大框内，但 IoU 只有 0.3 左右
    （大框远大于小框）→ 必须靠**包含比例**识别为重复框并在匹配前合并掉，
    否则画面里会凭空多出"人员1 + 人员2"。
    """
    full = (200, 60, 440, 460)        # 240x400 全身框
    torso = (250, 150, 390, 340)      # 140x190 躯干框，完全落在全身框内
    other = (500, 100, 600, 300)      # 画面里真正的第二个人
    assert rt._iou(full, torso) < rt.DUP_BOX_IOU          # IoU 判据确实失效
    assert rt._containment(full, torso) >= rt.DUP_BOX_CONTAIN   # 包含比例兜底
    assert rt._is_dup_box(full, torso) is True
    assert rt._is_dup_box(full, other) is False           # 真第二个人不合并
    keep, merged = rt._merge_dup_boxes([full, torso, other])
    assert keep == [0, 2] and (1, 0) in merged            # 躯干框被合并掉


def test_slow_first_frame_does_not_drop_t1(client, clock):
    """
    首帧加载模型可能耗时 5~6 秒：人员其实一直在画面里，不能因为**单帧**的
    墙钟间隔超过 TRACK_KEEP_S 就把 T1 删除并重建为 T2（会导致人员编号跳变）。
    判失效要求"连续 ≥2 帧未出现且超时"。
    """
    c, pipeline = client
    b1 = run_script(c, pipeline, [one_person(NO_PPE)], clock=clock, step_s=0.0)[0]
    sid = b1["session_id"]
    clock["t"] += rt.TRACK_KEEP_S + 1.5      # 第二帧"迟到" 6.5 秒（模型预热）
    rest = run_script(c, pipeline, [one_person(NO_PPE)] * 3, sid=sid,
                      clock=clock, step_s=0.0)
    bodies = [b1] + rest
    tids = {p["track_id"] for b in bodies for p in b["persons"]}
    assert tids == {"T1"}, f"单帧推理慢不应换 track，实际 {tids}"
    log = c.get(f"/api/realtime/tracklog?session_id={sid}&limit=200").json()
    assert not [e for e in log["entries"] if e["action"] == "delete"], \
        "人员一直在画面里，不应因单帧耗时被判失效"
    assert count_events() == 1


def test_close_range_duplicate_box_keeps_one_person_one_event(client, tmp_path):
    """
    近距离"全身框 + 躯干框"（小框完全落在大框内，IoU 只有 0.31）：
    合并后只算 **1 个正式人员 T1**，数据库只写 **1 条**；修复前会被当成
    两个人 → 人员1 + 人员2、写出 2 条记录。
    """
    c, _ = client
    pipeline = BoxPipeline(tmp_path / "boxshots")
    rt.init_realtime(lambda: pipeline)      # 必须是同一实例，脚本才生效
    full = ((200, 60, 440, 460), NO_PPE)      # 240x400 全身框
    torso = ((250, 150, 390, 340), NO_PPE)    # 140x190 躯干框（完全落在全身框内）
    assert rt._iou(full[0], torso[0]) < rt.DUP_BOX_IOU       # IoU 判据确实失效
    assert rt._containment(full[0], torso[0]) >= rt.DUP_BOX_CONTAIN

    bodies = run_script(c, pipeline, [[full, torso]] * 6)
    last = bodies[-1]
    assert last["person_count"] == 1, f"重复框应合并为 1 人，实际 {last['person_count']}"
    assert {p["track_id"] for p in last["persons"]} == {"T1"}
    assert count_events() == 1
    assert sum(len(b["new_events"]) for b in bodies) == 1

    sid = list(rt._sessions)[0]
    log = c.get(f"/api/realtime/tracklog?session_id={sid}&limit=200").json()
    dups = [e for e in log["entries"] if e["action"] == "dup_box"]
    assert dups, "重复框必须被记录到排查日志"
    assert all(float(d["contain"]) >= rt.DUP_BOX_CONTAIN for d in dups)


def test_spurious_box_never_confirmed_never_written(client, clock):
    """
    偶发假阳性框（只闪现 1 帧）：新 track 需连续 TRACK_CONFIRM_FRAMES 帧
    确认才成为正式人员 —— 确认前既不列入人员列表，也不写违规记录，
    避免凭空多出"人员2 + 第 2 条记录"。
    """
    c, pipeline = client
    before = count_events()
    far = (0.75, 0.5)                 # ROI 内、与本人不重叠的另一个位置
    script = ([[(CENTER, NO_PPE)]] * 2
              + [[(CENTER, NO_PPE), (far, NO_PPE)]]       # 假阳性框闪现 1 帧
              + [[(CENTER, NO_PPE)]] * 6)
    bodies = run_script(c, pipeline, script, clock=clock, step_s=1.0)

    blip = bodies[2]
    assert blip["person_count"] == 1                      # 假阳性不算正式人员
    assert [p["track_id"] for p in blip["pending_persons"]] == ["T2"]
    assert all(p["track_id"] == "T1" for b in bodies for p in b["persons"])
    assert sum(len(b["new_events"]) for b in bodies) == 1
    assert count_events() == before + 1


def test_sit_stand_box_size_change_keeps_same_track():
    """
    坐下 → 站起：person 框面积变化超过面积容忍上限（4.8 倍）、中心点下移，
    仍应靠中心距离匹配到同一个 track（不能只依赖 IoU / 面积比）。
    """
    sitting = (270, 140, 370, 340)     # 100x200，中心 (320,240)
    standing = (200, 60, 440, 460)     # 240x400，中心 (320,260)，面积 4.8 倍
    assert rt._area_ratio(sitting, standing) > rt.AREA_MAX_RATIO  # 超出容忍区间
    assert rt._iou(sitting, standing) < rt.MATCH_IOU              # IoU 也不达标
    assert rt._center_ratio_sym(sitting, standing) <= rt.MATCH_CENTER_RATIO
    tracks = {1: {"bbox": sitting, "miss": 0}}
    assert rt._match_tracks(tracks, [standing]) == [1]


# ---------------- 场景测试：多人 track / 停止清理 ----------------

def test_two_persons_get_separate_tracks(client):
    """两人同时进入画面 → 短时 track_id 能区分（T1 / T2）"""
    c, pipeline = client
    two = [((0.35, 0.5), NO_PPE), ((0.70, 0.5), NO_PPE)]
    bodies = run_script(c, pipeline, [two] * 3)
    last = bodies[-1]
    assert last["person_count"] == 2
    assert {p["track_id"] for p in last["persons"]} == {"T1", "T2"}
    assert all(p["in_roi"] for p in last["persons"])


def test_two_persons_each_write_one_continuous_event(client, clock):
    """验收 6：两人同时持续违规 12 秒 → T1 / T2 各 1 条，互不干扰"""
    c, pipeline = client
    before = count_events()
    two = [((0.35, 0.5), NO_PPE), ((0.70, 0.5), NO_PPE)]
    bodies = run_script(c, pipeline, [two] * 8, clock=clock, step_s=1.5)
    last = bodies[-1]
    assert last["person_count"] == 2
    assert {p["track_id"] for p in last["persons"]} == {"T1", "T2"}
    assert count_events() == before + 2
    assert all(p["episode_events"] == 1 for p in last["persons"])
    assert last["session"]["violation_count"] == 2


def test_track_lifecycle_logged_for_debug(client, clock):
    """
    排查日志：人员出现 → 离开画面，日志依次记录 create / match / miss / delete，
    含对应 bbox、IoU / 中心距离、置信度、空闲秒数（页面不展示，
    仅 /api/realtime/tracklog 可查）。track 失效按时间，故用可控时钟推进。
    """
    c, pipeline = client
    gap = int(rt.TRACK_KEEP_S) + 2          # 空闲时长需超过保留时长
    script = [one_person(NO_PPE)] * 2 + [[]] * gap
    run_script(c, pipeline, script, clock=clock, step_s=1.0)
    sid = list(rt._sessions)[0]
    log = c.get(f"/api/realtime/tracklog?session_id={sid}&limit=200").json()
    assert log["ok"] is True and log["entries"]
    # 每帧会先记一条 frame 条目（本帧 person 框数量），生命周期事件排除它
    actions = [e["action"] for e in log["entries"] if e["action"] != "frame"]
    assert actions[0] == "create"
    assert "match" in actions and "miss" in actions and "delete" in actions
    matched = [e for e in log["entries"] if e["action"] == "match"][0]
    assert "iou" in matched and "center_ratio" in matched and matched["bbox"] != "-"
    deleted = [e for e in log["entries"] if e["action"] == "delete"][0]
    assert deleted["track"] == "T1" and deleted["bbox"] != "-"
    assert deleted["reason"] == "idle_exceeded_keep_seconds"   # 按时间失效
    created = [e for e in log["entries"] if e["action"] == "create"][0]
    assert "best_iou" in created and "best_center_ratio" in created
    assert created["reason"] == "no_track_yet"      # 创建原因可追溯
    # 帧级日志：记录本帧 person 框数量，用于判断画面里到底有几个人
    frames = [e for e in log["entries"] if e["action"] == "frame"]
    assert frames and frames[0]["n_boxes"] == 1 and frames[-1]["n_boxes"] == 0
    assert frames[-1]["no_person_detected"] is True     # 本帧是否没检出 person
    assert frames[0]["confs"] != "-"                    # 记录检测置信度
    assert log["person_box_hist"] == {"1": 2, "0": gap}


def test_stop_clears_active_violation_and_track_log(client):
    """验收 7：停止会话 → active_violation / track / 窗口 / 日志全部清除"""
    c, pipeline = client
    run_script(c, pipeline, [one_person(NO_PPE)] * 4)
    sid = list(rt._sessions)[0]
    log = c.get(f"/api/realtime/tracklog?session_id={sid}").json()
    assert log["ok"] is True and log["entries"]
    r = c.post("/api/realtime/stop", data={"session_id": str(sid)}).json()
    assert r["ok"] is True and r["cleared_tracks"] >= 1
    assert r["track_log"]                      # 停止响应附带排查日志
    assert sid not in rt._sessions             # 状态整体丢弃
    gone = c.get(f"/api/realtime/tracklog?session_id={sid}").json()
    assert gone["ok"] is False and gone["entries"] == []


def test_stop_clears_all_session_state(client):
    """停止巡检后 track / 滑动窗口 / 去重状态全部清除，重新开启重新建立"""
    c, pipeline = client
    run_script(c, pipeline, [one_person(NO_PPE)] * 4)
    sid = rt._sessions and list(rt._sessions)[0]
    assert sid in rt._sessions
    r = c.post("/api/realtime/stop", data={"session_id": str(sid)}).json()
    assert r["ok"] is True and r["cleared_tracks"] >= 1
    assert sid not in rt._sessions            # 会话状态已整体丢弃

    # 复用旧 session_id 再发帧：应开新会话，track 重新从 T1 开始
    # （新 track 需连续 TRACK_CONFIRM_FRAMES 帧确认后才列为正式人员）
    pipeline.set_script([one_person(NO_PPE)])
    body = post_frame(c, session_id=sid).json()
    assert body["session_id"] != sid
    assert body["person_count"] == 0 and body["smooth"]["pending"] == 1
    for _ in range(rt.TRACK_CONFIRM_FRAMES - 1):
        body = post_frame(c, session_id=body["session_id"]).json()
    assert body["persons"][0]["track_id"] == "T1"
    assert body["persons"][0]["window"] == rt.TRACK_CONFIRM_FRAMES
    assert body["smooth"]["ready"] is False   # 稳定状态需重新建立
