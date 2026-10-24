"""C1 续批「嫌疑筛查」（2026-10-07）

背景：C1（~穿戴演示-含白大褂 dataset1-C1）共 1,153 图，她已复核 240 张
（sheet_01 的 48 张 + c1_batch02~05 的 192 张），其余 900+ 张逐张看太费眼。
本脚本用**确定性规则**（几何 + 框内颜色）把最可疑的挑出来，只给她一版
~100 张的嫌疑图版，其余默认保留。

动机来自她今天报的错框类型：
- 洗手衣/隔离衣/刷手服 被标成 lab_coat（框内偏蓝/绿、白色占比低）
- 框贴图片边缘导致缩略图看不出框（#20 那种）
- 框只罩局部 / 框太大框住整图 / 重复框

规则（每命中一条加分）：
  几何：触碰 ≥3 条边、面积占比 >0.75 或 <0.002、宽高比 >6 或 <1/6、同类框 IoU>0.6
  颜色：lab_coat 框内"白系"占比 <0.30 且蓝/绿系 >0.25 → 疑似非白大褂

输出：
  outputs/phase2_provenance/c1_suspects/suspect_list.txt   （喂给 make_contact_sheet.py）
  outputs/phase2_provenance/c1_suspects/screen_report.csv  （全量筛查明细）

用法：
    .venv/Scripts/python.exe tools/screen_c1_suspects.py --top 100
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "lab_ppe"
SHEETS = ROOT / "outputs" / "phase2_provenance"
OUT = SHEETS / "c1_suspects"
NAMES = {0: "mask", 1: "gloves", 2: "lab_coat", 3: "goggles"}

C1_ALL = SHEETS / "~穿戴演示-含白大褂dataset1-C1.all.txt"
REVIEWED_LISTS = [
    SHEETS / "~穿戴演示-含白大褂dataset1-C1.list",
    SHEETS / "c1_batch02.list",
    SHEETS / "c1_batch03.list",
    SHEETS / "c1_batch04.list",
    SHEETS / "c1_batch05.list",
]


def read_list(p: Path):
    if not p.exists():
        return []
    out = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        rel = ln.strip().split("\t")[0].replace("\\", "/").lstrip("./")
        if rel and not rel.startswith("#") and "/" in rel:
            out.append(rel)
    return out


def load_label(p: Path):
    """返回 [(cls, cx, cy, w, h)]"""
    if not p.exists():
        return []
    boxes = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        parts = ln.split()
        if len(parts) >= 5 and parts[0].lstrip("-").isdigit():
            boxes.append((int(float(parts[0])), *[float(v) for v in parts[1:5]]))
    return boxes


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def color_stats(img_bgr, box):
    """返回 (白系占比, 蓝绿系占比, 深色占比)"""
    h, w = img_bgr.shape[:2]
    cls, cx, cy, bw, bh = box
    x1 = max(0, int((cx - bw / 2) * w))
    y1 = max(0, int((cy - bh / 2) * h))
    x2 = min(w, int((cx + bw / 2) * w))
    y2 = min(h, int((cy + bh / 2) * h))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return 0.0, 0.0, 0.0
    roi = img_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    tot = roi.shape[0] * roi.shape[1]
    white = ((V > 150) & (S < 65)).sum() / tot
    blue = (((H >= 90) & (H <= 140)) & (S > 65) & (V > 60)).sum() / tot
    green = (((H >= 40) & (H < 90)) & (S > 65) & (V > 60)).sum() / tot
    dark = (V < 80).sum() / tot
    return white, blue + green, dark


def screen_one(img_path: Path, lab_path: Path):
    flags, score = [], 0.0
    boxes = load_label(lab_path)
    if not boxes:
        return ["空标签"], 0.5
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return ["读图失败"], 9.0
    h, w = img.shape[:2]
    if max(h, w) > 800:  # 色彩统计用降采样图，提速
        s = 800 / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

    for (cls, cx, cy, bw, bh) in boxes:
        x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
        nm = NAMES.get(cls, str(cls))
        edges = sum([x1 < 0.004, y1 < 0.004, x2 > 0.996, y2 > 0.996])
        area = bw * bh
        ar = bw / bh if bh > 0 else 99
        if edges >= 3:
            flags.append(f"{nm}框贴{edges}边"); score += 1.0
        if area > 0.75:
            flags.append(f"{nm}框过大{area:.2f}"); score += 1.0
        if area < 0.002:
            flags.append(f"{nm}框过小{area:.4f}"); score += 0.7
        if ar > 6 or ar < 1 / 6:
            flags.append(f"{nm}框怪比例{ar:.1f}"); score += 0.7
        white, bluegreen, dark = color_stats(img, (cls, cx, cy, bw, bh))
        if cls == 2:  # lab_coat
            if white < 0.30 and bluegreen > 0.25:
                flags.append(f"lab_coat框偏蓝绿(白{white:.2f}/蓝绿{bluegreen:.2f})"); score += 2.0
            elif white < 0.20 and dark > 0.45:
                flags.append(f"lab_coat框偏深色(白{white:.2f}/深{dark:.2f})"); score += 1.2
            elif white < 0.25:
                flags.append(f"lab_coat框白色少({white:.2f})"); score += 0.6

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][0] != boxes[j][0]:
                continue
            a = to_xyxy(boxes[i]); b = to_xyxy(boxes[j])
            if iou(a, b) > 0.6:
                nm = NAMES.get(boxes[i][0], str(boxes[i][0]))
                flags.append(f"{nm}重复框"); score += 0.6
                break
        else:
            continue
        break
    return flags, score


def to_xyxy(b):
    _, cx, cy, bw, bh = b
    return cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=100)
    args = ap.parse_args()

    reviewed = set()
    for p in REVIEWED_LISTS:
        reviewed.update(read_list(p))
    all_rel = read_list(C1_ALL)
    todo = [r for r in all_rel
            if r not in reviewed and (DATASET / "images" / r).exists()]
    print(f"C1 全组 {len(all_rel)} | 已复核 {len(reviewed)} | 待筛查 {len(todo)}")

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, rel in enumerate(todo, 1):
        split, name = rel.split("/", 1)
        img = DATASET / "images" / split / name
        lab = DATASET / "labels" / split / (Path(name).stem + ".txt")
        flags, score = screen_one(img, lab)
        rows.append((rel, score, "; ".join(flags)))
        if k % 200 == 0:
            print(f"  ...已筛查 {k}/{len(todo)}", flush=True)

    rows.sort(key=lambda r: -r[1])
    with open(OUT / "screen_report.csv", "w", encoding="utf-8", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["file", "score", "flags"])
        for rel, score, fl in rows:
            wtr.writerow([rel, f"{score:.2f}", fl])

    hit = [r for r in rows if r[1] > 0]
    top = rows[:args.top]
    with open(OUT / "suspect_list.txt", "w", encoding="utf-8", newline="\n") as f:
        for rel, score, fl in top:
            f.write(f"{rel}\t[{score:.1f}] {fl}\n")

    print(f"\n筛查完成：疑似命中 {len(hit)}/{len(rows)}（{len(hit)/max(1,len(rows))*100:.1f}%）")
    print(f"输出 top{len(top)} 清单 → {OUT/'suspect_list.txt'}")
    print(f"全量明细 → {OUT/'screen_report.csv'}")
    print("\nTop10 预览：")
    for rel, score, fl in top[:10]:
        print(f"  {score:4.1f}  {rel.split('/')[-1][:58]}  {fl[:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
