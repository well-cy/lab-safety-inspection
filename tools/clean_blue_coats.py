"""蓝大褂清洗（2026-10-08，她定口径：蓝大褂不算 lab_coat）

判据（框内颜色，与 tools/screen_c1_suspects.py 同一套 HSV 规则）：
    白系   ：V>150 且 S<65
    蓝/绿系：H∈[90,140]（蓝）或 [40,90)（绿），且 S>65 且 V>60
    深色   ：V<80

一个 lab_coat 框被判为「蓝大褂」需同时满足：
    white < --white-thr(默认 0.35)
    bluegreen > --bg-thr(默认 0.30)
    bluegreen > white * 1.3      ← 蓝绿必须明显压过白色（防蓝光照片误伤白大褂）

动作：删掉该 lab_coat 框（图片保留 → 变成负样本，正是"蓝大褂≠白大褂"的监督信号）。
改前标签整份备份到 data/_label_backup/blue_coat_20261008/<split>/。

用法：
    .venv/Scripts/python.exe tools/clean_blue_coats.py                 # 只报告（默认）
    .venv/Scripts/python.exe tools/clean_blue_coats.py --apply         # 执行删框（分批）
    .venv/Scripts/python.exe tools/clean_blue_coats.py --restore       # 备份整体还原
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "lab_ppe"
OUT = ROOT / "outputs" / "blue_coat_20261008"
BACKUP = ROOT / "data" / "_label_backup" / "blue_coat_20261008"
SPLITS = ("train", "valid")
LAB_COAT = 2


def color_stats(img, box):
    """返回 (白系, 蓝系, 绿系, 深色) 占比。蓝/绿分开统计——绿色多为灯光色偏（白大褂被染绿），蓝色才是蓝大褂。"""
    h, w = img.shape[:2]
    _, cx, cy, bw, bh = box
    x1 = max(0, int((cx - bw / 2) * w)); y1 = max(0, int((cy - bh / 2) * h))
    x2 = min(w, int((cx + bw / 2) * w)); y2 = min(h, int((cy + bh / 2) * h))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return 0.0, 0.0, 0.0, 0.0
    roi = img[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    tot = roi.shape[0] * roi.shape[1]
    white = ((V > 150) & (S < 65)).sum() / tot
    blue = ((H >= 95) & (H <= 130) & (S > 90) & (V > 60)).sum() / tot
    green = ((H >= 40) & (H < 95) & (S > 90) & (V > 60)).sum() / tot
    dark = (V < 80).sum() / tot
    return white, blue, green, dark


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--white-thr", type=float, default=0.35)
    ap.add_argument("--bg-thr", type=float, default=0.30)
    ap.add_argument("--ratio", type=float, default=1.3)
    ap.add_argument("--limit", type=int, default=0, help="本次最多处理多少个文件（防环境回收）")
    ap.add_argument("--tier", default="all", choices=["A","B","all"], help="A=高置信自动删 B=待人工 all=全部")
    args = ap.parse_args()

    if args.restore:
        n = 0
        for split in SPLITS:
            src_dir = BACKUP / split
            if not src_dir.is_dir():
                continue
            for f in sorted(src_dir.iterdir()):
                dst = DATASET / "labels" / split / f.name
                shutil.copy2(f, dst)
                n += 1
        print(f"已从备份还原 {n} 个标签文件")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    hits, scanned, scanned_files = [], 0, 0
    t0 = time.time()
    for split in SPLITS:
        ldir = DATASET / "labels" / split
        for fn in sorted(os.listdir(ldir)):
            if not fn.endswith(".txt"):
                continue
            lp = ldir / fn
            lines = [l for l in lp.read_text(encoding="utf-8").splitlines() if l.strip()]
            boxes = []
            for i, ln in enumerate(lines):
                parts = ln.split()
                if len(parts) >= 5 and parts[0].lstrip("-").isdigit():
                    boxes.append((i, int(float(parts[0])), *[float(v) for v in parts[1:5]]))
            if not any(b[1] == LAB_COAT for b in boxes):
                continue
            img_p = DATASET / "images" / split / (fn[:-4] + ".jpg")
            if not img_p.exists():
                alt = list((DATASET / "images" / split).glob(fn[:-4] + ".*"))
                if not alt:
                    continue
                img_p = alt[0]
            img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
            if img is None:
                continue
            scanned_files += 1
            if max(img.shape[:2]) > 800:
                s = 800 / max(img.shape[:2])
                img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
            drop_idx = []
            for i, cls, cx, cy, bw, bh in boxes:
                scanned += 1
                if cls != LAB_COAT:
                    continue
                white, blue, green, dark = color_stats(img, (cls, cx, cy, bw, bh))
                dom = max(blue, green)
                if white < args.white_thr and dom > args.bg_thr and dom > white * args.ratio:
                    if blue >= green and white < 0.20 and blue > 0.55:
                        tier = "A"   # 极端蓝：白占比极低+高饱和蓝占绝对主导，几乎必是真蓝大褂
                    else:
                        tier = "B"   # 边缘（含绿光污染/蓝色灯光下的白大褂/浅蓝大褂）→ 人工判
                    drop_idx.append((i, tier))
                    hits.append((tier, split, fn, i, f"{white:.3f}", f"{blue:.3f}", f"{green:.3f}"))
            if drop_idx and args.apply:
                BACKUP.mkdir(parents=True, exist_ok=True)
                (BACKUP / split).mkdir(parents=True, exist_ok=True)
                bp = BACKUP / split / fn
                if not bp.exists():
                    shutil.copy2(lp, bp)
                drop_set = {i for i, _ in drop_idx if True} if args.tier == "all" else {
                    i for i, t in drop_idx if t == args.tier}
                keep = [l for i, l in enumerate(lines) if i not in drop_set]
                lp.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8", newline="\n")
                if len(hits) >= args.limit > 0:
                    print(f"达到 --limit {args.limit}，停止（重跑续做，已删的框不会再命中）")
                    break
        if args.limit and len(hits) >= args.limit:
            break

    with open(OUT / "blue_coat_report.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tier", "split", "label_file", "line_index", "white", "blue", "green"])
        w.writerows(hits)
    files = len({(h[1], h[2]) for h in hits})
    print(f"\n{'已删框' if args.apply else '报告'}：扫描 {scanned_files} 个含 lab_coat 的标签文件 / {scanned} 个 lab_coat 框")
    print(f"命中蓝大褂框 {len(hits)} 个，涉及 {files} 张图  | 用时 {time.time()-t0:.1f}s")
    print(f"明细 → {OUT/'blue_coat_report.csv'}")
    if args.apply:
        print(f"改前标签备份 → {BACKUP}（--restore 可整体还原）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
