# -*- coding: utf-8 -*-
"""
MVP 验收阶段：10 张公开测试图片逐张详细检测报告。
对每张图输出：原始检测框（类别/置信度）、Person-PPE 匹配结果、ROI 判定、
规则评估结果（违规类型/严重程度）。
仅用于验收分析，不写入数据库。
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai.detector import PPEDetector
from ai.ppe_matcher import match_person_ppe
from rule_engine.engine import SafetyRuleEngine, annotate_person_states
from backend.database import load_areas


def main():
    det = PPEDetector(str(ROOT / "ai" / "model" / "sh17_yolov8s.pt"),
                      conf_threshold=0.35)
    areas = load_areas(lab_id=1)  # 实验室A → 实验操作区
    test_dir = ROOT / "data" / "test"
    images = sorted(p for p in test_dir.iterdir()
                    if p.suffix.lower() in (".jpg", ".png", ".jpeg"))

    for img_path in images:
        dets = det.detect(str(img_path))
        import cv2
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        states = match_person_ppe(dets)
        states = annotate_person_states(states, areas, w, h)

        print("=" * 78)
        print(f"图片: {img_path.name}  ({w}x{h})")
        print("-- 原始检测 --")
        for d in sorted(dets, key=lambda x: -x.conf):
            print(f"  {d.cls_name:<10} (raw={d.raw_cls:<18}) conf={d.conf:.2f}"
                  f"  box=({d.x1:.0f},{d.y1:.0f},{d.x2:.0f},{d.y2:.0f})")
        if not dets:
            print("  （无任何检测）")
        print("-- 人员状态与规则评估 --")
        engine = SafetyRuleEngine()
        events = engine.evaluate(states)
        ev_map = {ev.person_id: ev for ev in events}
        for st in states:
            ppe_desc = ", ".join(f"{k}({v.conf:.2f})" for k, v in sorted(st.ppe.items())) or "无"
            print(f"  person#{st.person_id} 区域={st.area_name or '普通区域'} "
                  f"PPE=[{ppe_desc}]")
            ev = ev_map.get(st.person_id)
            if ev is None:
                print("    → 判定: 不评估（普通区域/区域未配置规则）")
            else:
                print(f"    → 判定: {ev.severity}"
                      + (f" | 缺失: {', '.join(ev.violation_types)}" if ev.violation_types else " | 齐全"))
    print("=" * 78)


if __name__ == "__main__":
    main()
