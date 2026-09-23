# -*- coding: utf-8 -*-
"""
给「未标注候选池」出审查拼版（make_contact_sheet.py 的清单模式要求 split/name，
吃不下裸 images 目录，故单独实现）。

每格左上角写**全局序号**，同时把 序号->文件名 映射写进 <out>/index.txt，
人工圈选时只需报序号，例如「3、7、12 剔除」。

用法：
    python tools/make_raw_sheet.py --dir data/raw/web_lab/images --out outputs/web_review
    python tools/make_raw_sheet.py --dir ... --out ... --tsv data/raw/web_lab/stats.tsv   # 叠加筛查标记
    python tools/make_raw_sheet.py --dir ... --out ... --list keep.txt                    # 只出指定文件
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

# 亮底图用深色字，暗底图用亮色字（按格子中心亮度自适应）
TXT_DARK = (28, 28, 28)
TXT_LIGHT = (245, 245, 245)


def imread_unicode(p: Path):
    try:
        return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def fit_tile(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(size / w, size / h)
    r = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 20, np.uint8)
    y = (size - r.shape[0]) // 2
    x = (size - r.shape[1]) // 2
    canvas[y:y + r.shape[0], x:x + r.shape[1]] = r
    return canvas


def lum(tile: np.ndarray) -> float:
    g = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    return float(g.mean())


def main() -> int:
    ap = argparse.ArgumentParser(description="裸目录候选池审查拼版")
    ap.add_argument("--dir", required=True, help="图片目录")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--thumb", type=int, default=320)
    ap.add_argument("--list", default="", help="只出该清单里的文件名（每行一个）")
    ap.add_argument("--tsv", default="", help="screen_dirty_graphics --tsv 的明细，用于叠标记")
    ap.add_argument("--start-index", type=int, default=1)
    args = ap.parse_args()

    d = Path(args.dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    flags: dict[str, str] = {}
    if args.tsv:
        with Path(args.tsv).open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                fl = (row.get("flag") or "").strip()
                if fl and fl != "ok":
                    flags[row["name"]] = fl
        print(f"[标记] 载入 {len(flags)} 条筛查标记")

    keep: set[str] | None = None
    if args.list:
        keep = {ln.strip() for ln in Path(args.list).read_text(encoding="utf-8").splitlines()
                if ln.strip() and not ln.startswith("#")}
        print(f"[清单] 只出 {len(keep)} 张")

    files = sorted(p for p in d.iterdir()
                   if p.is_file() and p.suffix.lower() in {e.lower() for e in IMG_EXT})
    if keep is not None:
        files = [p for p in files if p.name in keep]
    if not files:
        print("[空] 没有图片")
        return 1

    # 序号 -> 文件名 映射（全局一致，跨页续编）
    idx_file = out / "index.txt"
    with idx_file.open("w", encoding="utf-8") as f:
        f.write("# 全局序号\t文件名\t筛查标记\n")
        for i, p in enumerate(files, start=args.start_index):
            f.write(f"{i}\t{p.name}\t{flags.get(p.name, '')}\n")
    print(f"[映射] {idx_file}")

    per = args.cols * args.rows
    n_sheets = (len(files) + per - 1) // per
    pad = 26
    cell = args.thumb
    W = args.cols * cell + (args.cols + 1) * 8
    H = args.rows * cell + (args.rows + 1) * 8 + pad

    bad_cnt = 0
    for s in range(n_sheets):
        canvas = np.full((H, W, 3), 34, np.uint8)
        chunk = files[s * per:(s + 1) * per]
        for k, p in enumerate(chunk):
            gi = args.start_index + s * per + k
            r, c = divmod(k, args.cols)
            x = 8 + c * (cell + 8)
            y = pad + 8 + r * (cell + 8)
            img = imread_unicode(p)
            if img is None:
                tile = np.full((cell, cell, 3), 60, np.uint8)
                cv2.putText(tile, "unreadable", (10, cell // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 255), 2)
            else:
                tile = fit_tile(img, cell)
            fl = flags.get(p.name, "")
            if fl:
                bad_cnt += 1
                cv2.rectangle(tile, (0, 0), (cell - 1, cell - 1), (60, 60, 235), 4)
            color = (200, 235, 255) if lum(tile) < 128 else (40, 40, 40)
            cv2.putText(tile, str(gi), (8, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (30, 30, 30), 6)          # 描边
            cv2.putText(tile, str(gi), (8, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (120, 255, 255), 2)
            canvas[y:y + cell, x:x + cell] = tile
            if fl:
                cv2.putText(canvas, fl[:24], (x + 4, y + cell - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 255), 1)
        for c in range(args.cols):   # 列间分隔线
            cv2.line(canvas, (8 + c * (cell + 8) - 4, pad), (8 + c * (cell + 8) - 4, H - 8),
                     (48, 48, 48), 1)
        dst = out / f"sheet_{s + 1:02d}.jpg"
        ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 86])
        buf.tofile(str(dst))
        print(f"  {dst.name}  ({len(chunk)} 张)")

    print(f"\n[完成] {n_sheets} 页 / {len(files)} 张（其中带筛查标记 {bad_cnt} 张，红框标出）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
