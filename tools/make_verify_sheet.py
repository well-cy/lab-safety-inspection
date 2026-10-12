"""为指定编号生成「逐条核实图版」：原图 + 已有标签框（彩色细线+类别标签） + 候选漏标框（红实线）。

用法:
    python tools/make_verify_sheet.py --index outputs/missing_labels_20261008/index_full.csv \
        --out outputs/missing_labels_20261008/verify --nos 4,5,11,17 --cell 700 --cols 4
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "lab_ppe"
CLS = {0: "mask", 1: "gloves", 2: "lab_coat", 3: "goggles"}
# 已有标签配色 (BGR)
TAG_COLOR = {0: (200, 120, 0), 1: (180, 160, 0), 2: (0, 160, 0), 3: (200, 0, 160)}
ABBR = {0: "M", 1: "G", 2: "C", 3: "K"}  # mask / gloves / lab_coat / goggles


def imread_u(p: Path):
    try:
        return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_u(p: Path, img) -> None:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if ok:
        buf.tofile(str(p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nos", required=True, help="逗号分隔编号")
    ap.add_argument("--cell", type=int, default=700)
    ap.add_argument("--cols", type=int, default=4)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = {int(r["编号"]): r for r in csv.DictReader(open(args.index, encoding="utf-8-sig"))}
    nos = [int(x) for x in args.nos.split(",") if x.strip()]

    items = []
    for n in nos:
        r = rows.get(n)
        if not r:
            print(f"[跳过] 无编号 {n}")
            continue
        img = imread_u(Path(r["原图绝对路径"]))
        if img is None:
            print(f"[跳过] 原图读不到 #{n}")
            continue
        h, w = img.shape[:2]
        # 已有标签
        lp = Path(r["标签绝对路径"])
        if lp.exists():
            for line in lp.read_text(encoding="utf-8").splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                c = int(float(p[0]))
                cx, cy, bw, bh = (float(v) for v in p[1:5])
                x1 = int((cx - bw / 2) * w); y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w); y2 = int((cy + bh / 2) * h)
                col = TAG_COLOR.get(c, (128, 128, 128))
                cv2.rectangle(img, (x1, y1), (x2, y2), col, max(3, w // 500))
                t = f"{ABBR.get(c, '?')}:{CLS.get(c, c)}"
                cv2.putText(img, t, (x1, max(26, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                            max(0.9, w / 1400), col, max(2, w // 700), cv2.LINE_AA)
        # 候选框（红实线）
        if r["候选框(x1,y1,x2,y2)"]:
            for seg in r["候选框(x1,y1,x2,y2)"].split(";"):
                seg = seg.strip()
                if not seg or ":" not in seg:
                    continue
                name, coords = seg.split(":", 1)
                try:
                    x1, y1, x2, y2 = (int(float(v)) for v in coords.strip("() ").split(","))
                except ValueError:
                    continue
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), max(4, w // 420))
                cv2.putText(img, f"cand {name}", (x1, min(h - 10, y2 + int(34 * max(1, w / 1400)))),
                            cv2.FONT_HERSHEY_SIMPLEX, max(0.9, w / 1400), (0, 0, 255),
                            max(2, w // 700), cv2.LINE_AA)
        # 左上角留编号条
        bar = np.full((64, w, 3), 255, np.uint8)
        cv2.putText(bar, f"#{n}  {r['split']}", (14, 46), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (0, 0, 0), 3, cv2.LINE_AA)
        img = np.vstack([bar, img])
        cell = args.cell
        ih, iw = img.shape[:2]
        s = cell / iw
        img = cv2.resize(img, (cell, max(1, int(ih * s))), interpolation=cv2.INTER_AREA)
        items.append((n, img))

    cols = args.cols
    gap = 10
    for s in range(0, len(items), cols * 3):
        chunk = items[s:s + cols * 3]
        rh = max(im.shape[0] for _, im in chunk)
        sheet = np.full((3 * (rh + gap) + gap, cols * (cell + gap) + gap, 3), 245, np.uint8)
        for k, (n, im) in enumerate(chunk):
            r, c = divmod(k, cols)
            y = gap + r * (rh + gap)
            x = gap + c * (cell + gap)
            sheet[y:y + im.shape[0], x:x + im.shape[1]] = im
        no = s // (cols * 3) + 1
        imwrite_u(out_dir / f"vsheet_{no:02d}.jpg", sheet)
        print(f"vsheet_{no:02d}.jpg  ← " + ",".join(str(n) for n, _ in chunk))
    print(f"[完成] {len(items)} 张 → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
