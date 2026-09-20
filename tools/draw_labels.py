#!/usr/bin/env python3
"""draw_labels.py — 通用 YOLO 标注画框工具（审计抽检用）

对指定图片目录里的每张图，读取同名 .txt 标注（支持类名文本和数字 id 两种格式），
画出所有框，输出到可视化目录。替代写死抽检逻辑的 visualize_b3.py，可对任意数据集复用
（B3 干净子集、新下载的公开数据集等）。

用法：
  python tools/draw_labels.py                          # 默认画 data/b3_clean 全部三个 split
  python tools/draw_labels.py --src data/raw/labcoat_si/train/images --out outputs/labcoat_si_check
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

# 类名 → BGR 颜色（与 visualize_b3.py 口径一致，便于对照）
CLS = {"Lab_coat": (0, 165, 255), "Gloves": (0, 200, 0), "Goggles": (160, 0, 255)}
NUM_COLORS = [(0, 165, 255), (0, 200, 0), (160, 0, 255), (200, 0, 0), (0, 200, 200)]


def color_of(cls: str):
    if cls in CLS:
        return CLS[cls]
    try:  # 数字 id
        return NUM_COLORS[int(cls) % len(NUM_COLORS)]
    except ValueError:
        return (255, 255, 255)


def draw(img_path: Path, lbl_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    if not lbl_path.exists():
        cv2.putText(img, "NO LABEL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return img
    for line in lbl_path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        cls, xc, yc, bw, bh = p[0], *map(float, p[1:])
        l, r = int((xc - bw / 2) * w), int((xc + bw / 2) * w)
        t, b = int((yc - bh / 2) * h), int((yc + bh / 2) * h)
        c = color_of(cls)
        cv2.rectangle(img, (l, t), (r, b), c, 3)
        cv2.putText(img, cls, (max(l, 5), max(t - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", default=[],
                    help="图片目录（可多次传）；默认为 data/b3_clean 的三个 split")
    ap.add_argument("--out", default="outputs/draw_labels_check", help="输出目录")
    ap.add_argument("--limit", type=int, default=0, help="每个目录最多画几张（0=全部）")
    args = ap.parse_args()

    srcs = args.src or [f"data/b3_clean/{s}/images" for s in ("train", "valid", "test")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    n = 0
    for src in srcs:
        src = Path(src)
        if not src.exists():
            print(f"[跳过] 目录不存在: {src}")
            continue
        imgs = sorted(p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if args.limit:
            imgs = imgs[: args.limit]
        for p in imgs:
            lbl = p.parent.parent / "labels" / (p.stem + ".txt")
            img = draw(p, lbl)
            if img is None:
                print(f"[跳过] 读图失败: {p}")
                continue
            cv2.imwrite(str(out / f"{p.stem[:60]}.jpg"), img)
            n += 1
    print(f"完成：共画 {n} 张 → {out}/")


if __name__ == "__main__":
    main()
