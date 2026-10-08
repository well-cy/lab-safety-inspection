# -*- coding: utf-8 -*-
"""网图预标注精读图：原图 + 现有框 + 坐标网格（E6 五类口径）。

给 tools/make_webscene_viewer.py 的编号体系用：
    index.txt (编号<tab>文件名) + data/_prelabel/webscene_r1/labels/<stem>.txt

输出每张一张 JPG：框按类别着色并标序号，叠加浅色坐标网格（标 norm 值），
便于逐框核对落点后精确修框。

用法:
    python tools/make_webscene_grid.py --out outputs/web_lab_scene_review/grid \
        --nos 1,3,24 --width 1200
    python tools/make_webscene_grid.py --out ... --all --width 1100
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "data" / "raw" / "web_lab_scene" / "images"
LBL = ROOT / "data" / "_prelabel" / "webscene_r1" / "labels"
INDEX = ROOT / "outputs" / "web_lab_scene_review" / "index.txt"

CLS = {0: "person", 1: "mask", 2: "gloves", 3: "lab_coat", 4: "goggles"}
COLOR = {0: (0, 200, 0), 1: (0, 0, 235), 2: (235, 140, 0), 3: (200, 0, 200), 4: (0, 200, 255)}


def imread_u(p: Path):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def read_index():
    m = {}
    if not INDEX.exists():
        raise SystemExit(f"index 不存在: {INDEX}")
    for line in INDEX.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            m[int(parts[0])] = parts[1]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--nos", default="", help="逗号分隔编号")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--no-grid", action="store_true", help="不叠加坐标网格（抽查看图用）")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    idx = read_index()
    if args.all:
        nos = sorted(idx)
    else:
        nos = [int(x) for x in args.nos.split(",") if x.strip()]

    for n in nos:
        name = idx.get(n)
        if not name:
            print(f"  [跳过] #{n} 无索引")
            continue
        img = imread_u(IMG / name)
        if img is None:
            print(f"  [跳过] #{n} 读图失败")
            continue
        h, w = img.shape[:2]
        z = args.width / w
        big = cv2.resize(img, (int(w * z), int(h * z)), interpolation=cv2.INTER_LINEAR)

        # 坐标网格（每 0.05 一条）——--no-grid 时跳过
        if not args.no_grid:
            for k in range(1, 20):
                v = k / 20
                x = int(v * w * z)
                y = int(v * h * z)
                cv2.line(big, (x, 0), (x, big.shape[0]), (90, 90, 90), 1)
                cv2.line(big, (0, y), (big.shape[1], y), (90, 90, 90), 1)
                cv2.putText(big, f"{v:.2f}", (x + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 220, 160), 1)
                cv2.putText(big, f"{v:.2f}", (2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 220, 160), 1)

        stem = name.rsplit(".", 1)[0]
        lp = LBL / (stem + ".txt")
        rows = []
        if lp.exists():
            rows = [l.split() for l in lp.read_text(encoding="utf-8").splitlines() if l.strip()]
        for i, r in enumerate(rows):
            c = int(r[0])
            cx, cy, bw, bh = (float(v) for v in r[1:5])
            x1 = int((cx - bw / 2) * w * z); y1 = int((cy - bh / 2) * h * z)
            x2 = int((cx + bw / 2) * w * z); y2 = int((cy + bh / 2) * h * z)
            cv2.rectangle(big, (x1, y1), (x2, y2), COLOR[c], 2)
            cv2.putText(big, f"{i}{CLS[c][:2]}", (x1 + 2, max(12, y1 + 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR[c], 2)
        hdr = f"#{n} {name} | orig {w}x{h} | boxes={len(rows)}"
        hb = np.zeros((30, big.shape[1], 3), np.uint8) + 30
        cv2.putText(hb, hdr[:150], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
        final = np.vstack([hb, big])
        dst = out / f"g{n:02d}.jpg"
        cv2.imencode(".jpg", final, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(str(dst))
        print(f"  #{n} {w}x{h} boxes={len(rows)} -> {dst.name}")

    print(f"完成 -> {out}")


if __name__ == "__main__":
    main()
