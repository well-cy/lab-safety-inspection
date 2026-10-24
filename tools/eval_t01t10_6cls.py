# -*- coding: utf-8 -*-
"""t01~t10 场景评测（适配 6 类单模型架构）。

背景：旧的 tools/eval_e3_t01t10.py 用的是「SH17 出 person + LAB_PPE 出 PPE」
的 DUAL 方案，且权重路径指向 runs/e2/...（已废弃）。
本项目现已统一为「单个 6 类模型同时输出 person + 全部 PPE」，本脚本按新架构重写。

口径：
  - 单模型 ai/model/labppe_v8s.pt（6 类：person/mask/gloves/lab_coat/goggles/coverall）
  - conf=0.35, imgsz=640
  - PPEDetector + match_person_ppe + annotate_person_states + SafetyRuleEngine
  - 实验室A 实验操作区规则（mask+gloves+lab_coat）

用法：
  NO_PROXY='*' .venv/Scripts/python.exe tools/eval_t01t10_6cls.py
  NO_PROXY='*' .venv/Scripts/python.exe tools/eval_t01t10_6cls.py --conf 0.35 --imgsz 1280
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from ai.detector import PPEDetector
from ai.ppe_matcher import match_person_ppe
from rule_engine.engine import SafetyRuleEngine, annotate_person_states
from backend.database import load_areas

MODEL = ROOT / "ai" / "model" / "labppe_v8s.pt"


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--model", default=str(MODEL))
    args = ap.parse_args()

    det = PPEDetector(args.model, conf_threshold=args.conf, imgsz=args.imgsz)
    print("模型:", args.model)
    print("模型类别:", det.model.names)
    print("conf=%.2f  imgsz=%d" % (args.conf, args.imgsz))
    print()

    areas = load_areas(lab_id=1)
    print("已加载区域:", [(a.name, a.required_ppe) for a in areas])
    print()

    test_dir = ROOT / "data" / "test"
    images = sorted(p for p in test_dir.iterdir()
                    if p.suffix.lower() in (".jpg", ".png", ".jpeg"))

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        dets = det.detect(img)
        states, ev_map = eval_image(dets, areas, w, h)

        print("=" * 78)
        print("图片: %s (%dx%d)" % (img_path.name, w, h))
        print("-- 检测结果 --")
        for d in sorted(dets, key=lambda x: -x.conf):
            print("  %-12s (raw=%-12s) conf=%.2f" % (d.cls_name, d.raw_cls, d.conf))
        if not dets:
            print("  （无任何检测）")
        print("-- 规则判定 --")
        for line in summarize(states, ev_map):
            print("  " + line)
        if not states:
            print("  （无人员）")


if __name__ == "__main__":
    main()
