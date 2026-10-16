"""CPPE-5 第二遍筛查：person 检测 + 背景均匀度联合判定。

第一遍（screen_cppe5.py）纯颜色/边缘特征误杀"白墙医院里的真实医护场景"。
第二遍加入强信号：
  - yolov8n_coco 检测人数（空防护服/人台棚拍检出不到人）
  - 背景均匀度（影棚背景 border 方差极小）

判定：
  product = (无人 & 背景均匀)                      ← 空防护服/人台/纯背景目录图
          | (恰 1 人 & 背景均匀 & 偏白/低饱和)      ← 单模特棚拍
          | (white_ratio>0.5 & border 极均匀)       ← 极端白底（即便有人）
  其他 → natural
"""
import csv
import io
import sys
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "cppe5_screen"
CPPE = ROOT / "data" / "raw" / "cppe5" / "extracted"
CSV_P = OUT / "features.csv"


def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def main():
    from ultralytics import YOLO
    model = YOLO(str(ROOT / "ai" / "model" / "yolov8n_coco.pt"))

    rows = list(csv.DictReader(CSV_P.open(encoding="utf-8")))
    print(f"载入 {len(rows)} 行，开始 person 检测（imgsz=640, conf=0.25, batch=16）…")

    paths = [CPPE / "images" / r["file"] for r in rows]
    counts = [0] * len(rows)
    maxarea = [0.0] * len(rows)
    B = 16
    for i in range(0, len(paths), B):
        chunk = [str(p) for p in paths[i:i + B]]
        res = model.predict(chunk, classes=[0], conf=0.25, imgsz=640,
                            verbose=False, device="cpu")
        for j, r in enumerate(res):
            k = i + j
            H, W = r.orig_shape
            boxes = r.boxes
            counts[k] = len(boxes)
            if len(boxes):
                a = max((float(b.xywh[0][2]) * float(b.xywh[0][3])) for b in boxes) / (W * H)
                maxarea[k] = round(a, 4)
        if (i // B) % 8 == 0:
            print(f"  … {i + len(chunk)}/{len(paths)}")

    for r, c, a in zip(rows, counts, maxarea):
        r["person_n"] = c
        r["person_max_area"] = a
        bv_std = float(r["border_v_std"])
        bs_mean = float(r["border_s_mean"])
        wr = float(r["white_ratio"])
        lr = float(r["light_ratio"])
        uniform = bv_std < 18 and bs_mean < 40
        whitish = wr > 0.20 or lr > 0.45
        prod = (c == 0 and uniform) or \
               (c == 1 and uniform and whitish) or \
               (wr > 0.50 and bv_std < 10)
        r["product"] = int(prod)

    cols = list(rows[0].keys())
    with CSV_P.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    prod = sum(int(r["product"]) for r in rows)
    print(f"\n重判完成：product {prod} / natural {len(rows)-prod}")
    for sp in ("train", "test"):
        sub = [r for r in rows if r["split"] == sp]
        c2 = Counter()
        for r in sub:
            if r["product"] == "0":
                for u in r["useful"].split("|"):
                    if u:
                        c2[u] += 1
        print(f"[{sp}] natural 中含可用类：{dict(c2)}")


if __name__ == "__main__":
    main()
