# -*- coding: utf-8 -*-
"""E3-4 t01-t10 前后对比测试（同口径）。

口径与 tests/acceptance_check.py 完全一致：
  - conf=0.35, imgsz=640
  - PPEDetector + match_person_ppe + annotate_person_states + SafetyRuleEngine
  - 实验室A 实验操作区规则（mask+gloves+lab_coat）

两种配置：
  A) SH17-only：原系统行为（person + PPE 全部来自 SH17）——本次实际重跑作为基线
  B) DUAL：双模型方案（person 来自 SH17，mask/gloves/lab_coat/goggles 来自 LAB_PPE best.pt）

不修改任何生产代码；仅组合两路检测输出后走同一规则链路。
"""
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai.detector import PPEDetector
from ai.ppe_matcher import match_person_ppe
from rule_engine.engine import SafetyRuleEngine, annotate_person_states
from backend.database import load_areas

SH17 = ROOT / "ai" / "model" / "sh17_yolov8s.pt"
LABPPE = ROOT / "runs" / "e2" / "labppe_v8s" / "weights" / "best.pt"
CONF = 0.35
PPE_CLASSES = {"mask", "gloves", "lab_coat", "goggles"}


def eval_image(dets, areas, w, h):
    states = match_person_ppe(dets)
    states = annotate_person_states(states, areas, w, h)
    engine = SafetyRuleEngine()
    events = engine.evaluate(states)
    return states, {ev.person_id: ev for ev in events}


def summarize(states, ev_map):
    lines = []
    for st in states:
        ppe = sorted(st.ppe.keys())
        ev = ev_map.get(st.person_id)
        if ev is None:
            judge = "不评估"
        elif not ev.violation_types:
            judge = "正常"
        else:
            judge = f"{ev.severity}: " + "+".join(ev.violation_types)
        lines.append(f"p#{st.person_id}[{'/'.join(ppe) or '无PPE'}] -> {judge}")
    return lines


def main():
    sh17 = PPEDetector(str(SH17), conf_threshold=CONF)
    labppe = PPEDetector(str(LABPPE), conf_threshold=CONF)
    areas = load_areas(lab_id=1)
    test_dir = ROOT / "data" / "test"
    images = sorted(p for p in test_dir.iterdir()
                    if p.suffix.lower() in (".jpg", ".png", ".jpeg"))

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        dets_sh = sh17.detect(str(img_path))
        dets_lab = labppe.detect(str(img_path))

        # A) SH17-only（原系统）
        st_a, ev_a = eval_image(dets_sh, areas, w, h)
        # B) DUAL：person 用 SH17，PPE 用 LAB_PPE
        dets_dual = [d for d in dets_sh if d.cls_name == "person"] + \
                    [d for d in dets_lab if d.cls_name in PPE_CLASSES]
        st_b, ev_b = eval_image(dets_dual, areas, w, h)

        print("=" * 78)
        print(f"图片: {img_path.name} ({w}x{h})")
        print("-- A) SH17-only 原始检测 --")
        for d in sorted(dets_sh, key=lambda x: -x.conf):
            print(f"  {d.cls_name:<10} (raw={d.raw_cls:<18}) conf={d.conf:.2f}")
        if not dets_sh:
            print("  （无任何检测）")
        print("-- B) DUAL 原始检测 (person=SH17, PPE=LAB_PPE) --")
        for d in sorted(dets_dual, key=lambda x: -x.conf):
            print(f"  {d.cls_name:<10} (raw={d.raw_cls:<18}) conf={d.conf:.2f}")
        if not dets_dual:
            print("  （无任何检测）")
        print("-- A) SH17-only 判定 --")
        for line in summarize(st_a, ev_a):
            print(f"  {line}")
        if not st_a:
            print("  （无人员）")
        print("-- B) DUAL 判定 --")
        for line in summarize(st_b, ev_b):
            print(f"  {line}")
        if not st_b:
            print("  （无人员）")


if __name__ == "__main__":
    main()
