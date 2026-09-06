# -*- coding: utf-8 -*-
"""
ROI 演示测试（需求：人员进入指定区域后才按规则判违规）

场景：
  合成一段 640x480 视频，1 名"人员"（person 检测框）从左向右运动，
  穿过系统真实配置的"实验操作区 ROI"（实验室A默认值 [0.25,0.35,0.75,0.9]）。
  这名人员始终【缺少口罩】（只戴手套 + 穿实验服 lab_coat）。

期望行为：
  1) 人员中心在 ROI 外  → 即使缺少 PPE，也不判违规（"ROI外：跳过评估"）
  2) 人员中心进入 ROI  → 开始按 mask+gloves+lab_coat 规则判断
  3) 进入 ROI 后缺少口罩 → 立即生成"未佩戴口罩"违规事件 + 保存截图

输出：
  outputs/roi_demo.mp4                  —— 标注视频（回放 observable）
  outputs/screenshots/roi_demo_*.jpg    —— 违规帧截图
  控制台逐帧关键点表格
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from ai.detector import Detection
from ai.ppe_matcher import match_person_ppe
from rule_engine.engine import SafetyRuleEngine, annotate_person_states
from backend.database import load_areas

W, H = 640, 480
FPS = 30
TOTAL = 240  # 8 秒

# ---- 系统真实 ROI（实验室A → 实验操作区，归一化坐标）----
areas = load_areas(lab_id=1)
roi = areas[0]  # 归一化坐标 x1,y1,x2,y2
rx1, ry1, rx2, ry2 = [int(roi.x1 * W), int(roi.y1 * H),
                      int(roi.x2 * W), int(roi.y2 * H)]
print(f"ROI「{roi.name}」像素=({rx1},{ry1},{rx2},{ry2})  "
      f"required_ppe={roi.required_ppe}")

engine = SafetyRuleEngine()
out = cv2.VideoWriter(str(ROOT / "outputs" / "roi_demo.mp4"),
                      cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

def make_person_dets(cx):
    """构造一帧检测结果：人员一直在动，始终缺口罩，有手套+实验服。"""
    cy = 250                       # 中心固定在画面中线
    pw, ph = 120, 260
    px1, py1, px2, py2 = cx - pw // 2, cy - ph // 2, cx + pw // 2, cy + ph // 2
    return [
        Detection("person", "person", 0.92, px1, py1, px2, py2),
        Detection("gloves", "gloves", 0.88, cx - 30, py1 + 210, cx + 12, py1 + 250),
        Detection("lab_coat", "medical-suit", 0.80, cx - 56, py1 + 12, cx + 56, py1 + 246),
        # 注意：刻意不构造 mask —— 模拟"缺口罩"
    ]

records = []     # (frame, cx, in_roi, severity, missing, screenshot)
shot_id = 0
prev_missing = None   # 上一帧的缺失PPE元组，用于"状态切换才截图"

def imwrite_cn(path: Path, img):
    """cv2.imwrite 不支持 Windows 中文路径，用 imencode+tofile 兜底"""
    ok, buf = cv2.imencode(".jpg", img)
    if ok:
        path.write_bytes(buf.tobytes())
    return ok

for f in range(TOTAL):
    # 轨迹：前50帧在ROI外(静止)，50-70帧穿过边界，70帧后停在ROI内
    if f < 50:
        cx = 120
    elif f < 70:
        cx = 120 + int((f - 50) / 20 * 60)   # 120 -> 180（穿越 rx1=160）
    else:
        cx = 200                             # ROI 内停留

    dets = make_person_dets(cx)
    states = match_person_ppe(dets)
    states = annotate_person_states(states, areas, W, H)
    events = engine.evaluate(states)

    img = np.zeros((H, W, 3), dtype=np.uint8) + 30
    cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (200, 160, 60), 2)
    cv2.putText(img, f"ROI: {roi.name}", (rx1 + 4, ry1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 160, 60), 1)

    # events 只包含 ROI 内人员，必须按 person_id 对齐，否则 ROI 外的帧绘制不到
    ev_map = {ev.person_id: ev for ev in events}
    for st in states:
        ev = ev_map.get(st.person_id)
        b = st.bbox
        x1, y1, x2, y2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)
        viol = ev is not None and ev.is_violation
        color = (60, 60, 230) if viol else (80, 200, 80)   # 红=违规 绿=合规
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        lbl = f"person(centered)" if not viol else (", ".join(ev.violation_types) or "违规")
        cv2.putText(img, lbl, (x1, max(28, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        # 匹配到的 PPE 框
        for k, p in st.ppe.items():
            cv2.rectangle(img, (int(p.x1), int(p.y1)), (int(p.x2), int(p.y2)),
                          (230, 200, 60), 1)
            cv2.putText(img, k, (int(p.x1), int(p.y1) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 200, 60), 1)

        state_txt = ("ROI外：跳过评估(缺PPE不违规)" if not st.in_roi
                     else (f"ROI内→违规: {', '.join(ev.violation_types)}" if ev else "ROI内→合规"))
        cv2.putText(img, state_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(img, f"frame {f}", (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

        missing = ev.violation_types if (ev and ev.is_violation) else []
        shot = ""
        # 只在状态切换（进入ROI首次违规 / 缺失集合变化）时保存截图，避免刷屏
        if ev and ev.is_violation and tuple(missing) != prev_missing:
            shot_id += 1
            sp = ROOT / "outputs" / "screenshots" / f"roi_demo_{f:03d}.jpg"
            imwrite_cn(sp, img)
            shot = str(sp.name)
        prev_missing = tuple(missing)
        records.append((f, cx, st.in_roi, ev.severity if ev else "跳过",
                        missing, shot))

    out.write(img)

out.release()
cv2.destroyAllWindows()

# ---- 输出关键点表格 ----
print("\n=== ROI 演示逐帧关键点 ===")
print(f"{'frame':>6} {'cx':>5} {'inROI':>5}  {'判定':<6} {'缺失PPE':<20} 截图")
seen = set()
for f, cx, inroi, sev, missing, shot in records:
    # 只在状态变化或每60帧打印，避免刷屏
    key = (inroi, sev, tuple(missing))
    if key not in seen:
        seen.add(key)
        print(f"{f:>6} {cx:>5} {str(inroi):>5}  {sev:<6} {','.join(missing) or '-':<20} {shot}")
print("\n输出视频: outputs/roi_demo.mp4")
print(f"违规截图: {shot_id} 张（outputs/screenshots/roi_demo_*.jpg）")
