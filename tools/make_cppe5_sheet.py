"""CPPE-5 拼版工具：按 features.csv 过滤/抽样出编号拼版，供目检。

用法示例：
  python tools/make_cppe5_sheet.py --pred product --n 24 --seed 1 --out diag_product.jpg
  python tools/make_cppe5_sheet.py --pred natural --has Mask --n 24 --seed 7 --out diag_nat_mask.jpg
  python tools/make_cppe5_sheet.py --pred natural --has Mask --all --cols 6 --rows 8 --out sheet_mask_p1.jpg
"""
import argparse
import csv
import io
import sys
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
CPPE = ROOT / "data" / "raw" / "cppe5" / "extracted"
OUT = ROOT / "outputs" / "cppe5_screen"
NAMES = {1: "Coverall", 2: "Face_Shield", 3: "Gloves", 4: "Goggles", 5: "Mask"}
CCOL = {1: (200, 200, 0), 2: (0, 165, 255), 3: (0, 140, 235), 4: (255, 200, 0), 5: (0, 200, 0)}


def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def put(img, txt, org, color, sc=0.6, th=2):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, sc, (255, 255, 255), th + 2, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, sc, color, th, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--pred", choices=["product", "natural", "any"], default="natural")
    ap.add_argument("--has", default="", help="要求含某类，如 Mask / Gloves / Goggles")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--all", action="store_true", help="全量（不分页时慎用）")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--cell", type=int, default=340)
    ap.add_argument("--out", default="sheet.jpg")
    ap.add_argument("--draw-boxes", action="store_true", help="画可用类框")
    args = ap.parse_args()

    rows = list(csv.DictReader((OUT / "features.csv").open(encoding="utf-8")))
    rows = [r for r in rows if r["split"] == args.split]
    if args.pred == "product":
        rows = [r for r in rows if r["product"] == "1"]
    elif args.pred == "natural":
        rows = [r for r in rows if r["product"] == "0"]
    if args.has:
        rows = [r for r in rows if args.has in r["useful"].split("|")]

    rows.sort(key=lambda r: int(r["iid"]))
    if not args.all:
        rng = np.random.default_rng(args.seed)
        if len(rows) > args.n:
            pick = rng.choice(len(rows), size=args.n, replace=False)
            rows = [rows[i] for i in sorted(pick)]
    else:
        rows = rows[args.start:]

    cols, cw = args.cols, args.cell
    ch = args.cell
    page = cols * args.rows
    pages = max(1, (len(rows) + page - 1) // page)
    base = Path(args.out)
    if not base.is_absolute():
        base = OUT / base
    if base.suffix:
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    made = []
    for pg in range(pages):
        chunk = rows[pg * page:(pg + 1) * page]
        if not chunk:
            break
        sheet = np.full((args.rows * ch, cols * cw, 3), 30, np.uint8)
        for k, r in enumerate(chunk):
            img = imread_u(CPPE / "images" / r["file"])
            if img is None:
                continue
            H, W = img.shape[:2]
            s = min(cw / W, ch / H)
            img = cv2.resize(img, (max(1, int(W * s)), max(1, int(H * s))), interpolation=cv2.INTER_AREA)
            y0 = (k // cols) * ch + (ch - img.shape[0]) // 2
            x0 = (k % cols) * cw + (cw - img.shape[1]) // 2
            sheet[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
            cx, cy = (k % cols) * cw, (k // cols) * ch
            # 编号角标（醒目）
            no = args.start + pg * page + k + 1 if args.all else pg * page + k + 1
            tag = f"#{no}"
            cv2.rectangle(sheet, (cx + 2, cy + 2), (cx + 8 + 15 * len(tag), cy + 34), (0, 0, 0), -1)
            put(sheet, tag, (cx + 8, cy + 27), (0, 0, 255), 0.9, 2)
            # 类名右下
            cls_txt = "|".join(r["useful"].split("|")) or "-"
            put(sheet, cls_txt, (cx + 6, cy + ch - 8), (0, 255, 255), 0.5, 1)
            # 文件名左下（小）
            put(sheet, r["file"], (cx + 6, cy + 18), (180, 180, 180), 0.4, 1)
        op = base.parent / (base.name + (f"_p{pg+1}" if pages > 1 else "") + ".jpg")
        cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 86])[1].tofile(str(op))
        try:
            made.append(op.relative_to(ROOT))
        except ValueError:
            made.append(op)
    print(f"过滤后 {len(rows)} 图，出 {len(made)} 版")
    for m in made:
        print("  →", m)


if __name__ == "__main__":
    main()
