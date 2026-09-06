# -*- coding: utf-8 -*-
"""
模型快速验证脚本：对 data/test 下全部测试图片执行检测，
打印每张图的检测类别与置信度，用于评估 SH17 YOLOv8s 的实际效果。

运行：python tests/model_check.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from ai.detector import PPEDetector  # noqa: E402


def main():
    detector = PPEDetector(ROOT / "ai" / "model" / "sh17_yolov8s.pt")
    print(f"模型: {detector.weights_path}")
    print(f"设备: {detector.device_name}")
    print(f"模型原始类别: {list(detector.raw_names.values())}")
    print(f"置信度阈值: {detector.conf_threshold}")
    print("=" * 70)

    test_dir = ROOT / "data" / "test"
    images = sorted([p for p in test_dir.iterdir()
                     if p.suffix.lower() in (".jpg", ".png", ".jpeg")])
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"{img_path.name}: 读取失败")
            continue
        t0 = time.perf_counter()
        dets = detector.detect(img)
        ms = (time.perf_counter() - t0) * 1000
        print(f"\n{img_path.name}  ({img.shape[1]}x{img.shape[0]}, {ms:.0f}ms)")
        if not dets:
            print("  (无业务类别检测)")
        for d in dets:
            print(f"  {d.cls_name:<10} conf={d.conf:.2f}  "
                  f"bbox=({d.x1:.0f},{d.y1:.0f},{d.x2:.0f},{d.y2:.0f})  "
                  f"[raw: {d.raw_cls}]")


if __name__ == "__main__":
    main()
