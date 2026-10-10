# -*- coding: utf-8 -*-
"""make_dup_conflict_sheet.py — 同源副本「标注冲突」对照图版

去重工具 dedup_aug_dups.py 对同源副本只做「标签被代表完全覆盖」的剔除外，
剩下的 review_extra 是"两版标注互相矛盾"（同一位置框不一样，或一方多标了真实物体）。
本脚本把它们并排画出来：左=代表（绿框，将被保留），右=副本（橙框，待定），
供她按编号拍板（留代表 / 留副本 / 两者都留）。

用法:
    python tools/make_dup_conflict_sheet.py            # 读 data/aug_dup_report.txt
    python tools/make_dup_conflict_sheet.py --report data/aug_dup_report.txt --out outputs/aug_dup_review/conflict
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import cv2
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "data" / "lab_ppe"
NAMES = ["person", "mask", "gloves", "lab_coat", "goggles"]
CELL = 520


def imread_u(p: Path):
    try:
        return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def read_rows(split: str, stem: str):
    f = LAB / "labels" / split / (stem + ".txt")
    if not f.exists():
        return []
    out = []
    for ln in f.read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) == 5:
            out.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return out


def fmt(v):
    return f"{v:g}"


def draw(img, rows, color, tag):
    h, w = img.shape[:2]
    for c, cx, cy, bw, bh in rows:
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        txt = NAMES[c] if c < len(NAMES) else str(c)
        cv2.putText(img, txt, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3)
        cv2.putText(img, txt, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
    cv2.putText(img, tag, (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3)
    cv2.putText(img, tag, (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return img


def fit(img, size=CELL):
    h, w = img.shape[:2]
    s = min(size / w, size / h)
    r = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
    c = np.full((size, size, 3), 245, np.uint8)
    y0 = (size - r.shape[0]) // 2
    x0 = (size - r.shape[1]) // 2
    c[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="data/aug_dup_report.txt")
    ap.add_argument("--out", default="outputs/aug_dup_review/conflict")
    ap.add_argument("--per-sheet", type=int, default=3)
    args = ap.parse_args()

    txt = (ROOT / args.report).read_text(encoding="utf-8").splitlines()
    idx = [i for i, l in enumerate(txt) if l.startswith("# 待人工复核清单")]
    items = []
    if idx:
        for ln in txt[idx[0] + 1:]:
            p = ln.split("\t")
            if len(p) >= 4 and p[0].startswith(("review_extra", "review_class_conflict", "review_pixel_diff")):
                items.append(p)
    print(f"冲突项 {len(items)}")
    if not items:
        return 0

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    per = max(1, args.per_sheet)
    sheets = (len(items) + per - 1) // per
    for si in range(sheets):
        chunk = items[si * per:(si + 1) * per]
        H = per * (CELL + 30)
        W = 2 * (CELL + 12)
        sheet = np.full((H, W, 3), 255, np.uint8)
        for ri, p in enumerate(chunk):
            no = si * per + ri + 1
            sp, name = p[1].split("/")
            rep = p[2].replace("代表=", "")
            a = read_rows(sp, Path(name).stem)
            b = read_rows(sp, Path(rep).stem)
            ia = imread_u(LAB / "images" / sp / name)   # 副本图
            ib = imread_u(LAB / "images" / sp / rep)    # 代表图
            # 左=代表图+代表框(绿,KEEP)；右=副本图+副本框(橙,待定)。图与框必须同源！
            for ci, (im, rows, col, tag) in enumerate((
                    (ib, b, (60, 150, 60), f"#{no} 代表 KEEP  n={len(b)}  {rep[:34]}"),
                    (ia, a, (0, 140, 255), f"#{no} 副本 ?  n={len(a)}  {name[:34]}"))):
                if im is None:
                    continue
                v = fit(draw(im.copy(), rows, col, tag))
                sheet[ri * (CELL + 30):ri * (CELL + 30) + CELL,
                      ci * (CELL + 12):ci * (CELL + 12) + CELL] = v
        ok, buf = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if ok:
            (out_dir / f"conflict_{si + 1:02d}.jpg").write_bytes(buf.tobytes())
    print(f"图版 {sheets} 版 → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
