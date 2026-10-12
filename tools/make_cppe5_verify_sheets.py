"""CPPE-5 入库前核查拼版：图上直接画五类框（PPE 映射 + person 检测），并标记脏图 flag。

每版 6x5 / cell 300，红色大编号角标；可疑图右上角标黄底 flag（grayscale_like / graphic_color / graphic_white）。

用法：
    python tools/make_cppe5_verify_sheets.py                # 全 468 张
    python tools/make_cppe5_verify_sheets.py --only-dirty   # 只出可疑图
    python tools/make_cppe5_verify_sheets.py --nos 1,2,3    # 指定编号
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CPPE = ROOT / "data" / "raw" / "cppe5"
PICKED = CPPE / "picked"
LBL = CPPE / "prep" / "labels"
INDEX = ROOT / "outputs" / "cppe5_screen" / "picked_index.csv"
FLAGS = CPPE / "flags_dirty_raw.txt"
OUT = ROOT / "outputs" / "cppe5_screen" / "verify_sheets"

COLS, ROWS, CELL = 6, 5, 300

COLORS = {0: (255, 128, 0), 1: (60, 60, 255), 2: (60, 210, 60),
          3: (0, 210, 255), 4: (220, 60, 220)}
NAMES = {0: "person", 1: "mask", 2: "gloves", 3: "lab_coat", 4: "goggles"}


def imread_u(p: Path):
    try:
        return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def put(img, txt, org, color, sc=0.6, th=2):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, sc, color, th, cv2.LINE_AA)


def load_flags() -> dict[str, str]:
    out: dict[str, str] = {}
    if not FLAGS.exists():
        return out
    for ln in FLAGS.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split("\t")
        if len(parts) >= 3:
            out[parts[1]] = parts[2].replace("dirty:", "")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-dirty", action="store_true")
    ap.add_argument("--nos", default="")
    ap.add_argument("--tag", default="verify")
    args = ap.parse_args()

    rows = []
    with INDEX.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    flags = load_flags()

    if args.nos:
        keep = {int(x) for x in args.nos.split(",") if x.strip()}
        rows = [r for r in rows if int(r["no"]) in keep]
    if args.only_dirty:
        rows = [r for r in rows if r["file"] in flags]

    OUT.mkdir(parents=True, exist_ok=True)
    page = COLS * ROWS
    pages = max(1, (len(rows) + page - 1) // page)
    tag = args.tag + ("_dirty" if args.only_dirty else "")

    for pg in range(pages):
        chunk = rows[pg * page:(pg + 1) * page]
        sheet = np.full((ROWS * CELL, COLS * CELL, 3), 28, np.uint8)
        for k, r in enumerate(chunk):
            fname = r["file"]
            img = imread_u(PICKED / fname)
            if img is None:
                continue
            H, W = img.shape[:2]
            s = min(CELL / W, CELL / H)
            dw, dh = max(1, int(W * s)), max(1, int(H * s))
            small = cv2.resize(img, (dw, dh), interpolation=cv2.INTER_AREA)
            oy = (k // COLS) * CELL + (CELL - dh) // 2
            ox = (k % COLS) * CELL + (CELL - dw) // 2
            sheet[oy:oy + dh, ox:ox + dw] = small

            # 画标签框
            lp = LBL / (Path(fname).stem + ".txt")
            if lp.exists():
                for ln in lp.read_text(encoding="utf-8").splitlines():
                    ln = ln.strip()
                    if not ln:
                        continue
                    p = ln.split()
                    if len(p) != 5:
                        continue
                    c = int(p[0])
                    cx, cy, nw, nh = (float(v) for v in p[1:5])
                    x1 = ox + int((cx - nw / 2) * W * s)
                    y1 = oy + int((cy - nh / 2) * H * s)
                    x2 = ox + int((cx + nw / 2) * W * s)
                    y2 = oy + int((cy + nh / 2) * H * s)
                    cv2.rectangle(sheet, (x1, y1), (x2, y2), COLORS[c], 1, cv2.LINE_AA)
                    if c != 0:  # PPE 标名字，person 只画框避免太挤
                        put(sheet, NAMES[c][:1].upper(), (x1, max(y1 - 2, oy + 10)),
                            COLORS[c], 0.42, 1)

            # 编号角标
            cx0, cy0 = (k % COLS) * CELL, (k // COLS) * CELL
            tg = f"#{r['no']}"
            cv2.rectangle(sheet, (cx0 + 2, cy0 + 2), (cx0 + 12 + 16 * len(tg), cy0 + 32), (0, 0, 0), -1)
            put(sheet, tg, (cx0 + 8, cy0 + 26), (0, 0, 255), 0.9, 2)

            # 脏图 flag
            fl = flags.get(fname)
            if fl:
                put(sheet, fl, (cx0 + CELL - 6, cy0 + 26), (0, 230, 230), 0.40, 1)

        put(sheet, f"{tag}  {pg+1}/{pages}   no {chunk[0]['no']}-{chunk[-1]['no']}  n={len(chunk)}",
            (10, ROWS * CELL - 10), (255, 255, 255), 0.7, 2)
        op = OUT / f"{tag}_{pg+1:02d}.jpg"
        cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 86])[1].tofile(str(op))

    print(f"{len(rows)} 张 -> {pages} 版 -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    raise SystemExit(main())
