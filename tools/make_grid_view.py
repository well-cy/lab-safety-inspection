"""生成带坐标网格的单图放大版（目测定坐标专用）。

用法: python tools/make_grid_view.py --index outputs/missing_labels_20261008/index_full.csv \
        --out outputs/missing_labels_20261008/grid --nos 17,29 --width 1100
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TAG_COLOR = {0: (200, 120, 0), 1: (180, 160, 0), 2: (0, 160, 0), 3: (200, 0, 160)}
ABBR = {0: "M", 1: "G", 2: "C", 3: "K"}


def imread_u(p: Path):
    try:
        return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_u(p: Path, img) -> None:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if ok:
        buf.tofile(str(p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nos", required=True)
    ap.add_argument("--width", type=int, default=1100)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = {int(r["编号"]): r for r in csv.DictReader(open(args.index, encoding="utf-8-sig"))}

    for n in [int(x) for x in args.nos.split(",") if x.strip()]:
        r = rows.get(n)
        if not r:
            continue
        img = imread_u(Path(r["原图绝对路径"]))
        if img is None:
            print(f"[跳过] #{n}")
            continue
        h, w = img.shape[:2]
        # 已有标签
        lp = Path(r["标签绝对路径"])
        boxes = []
        if lp.exists():
            for i, line in enumerate(lp.read_text(encoding="utf-8").splitlines()):
                p = line.split()
                if len(p) < 5:
                    continue
                c = int(float(p[0]))
                cx, cy, bw, bh = (float(v) for v in p[1:5])
                x1 = int((cx - bw / 2) * w); y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w); y2 = int((cy + bh / 2) * h)
                boxes.append(("old", c, i, x1, y1, x2, y2))
        # 候选
        if r["候选框(x1,y1,x2,y2)"]:
            for seg in r["候选框(x1,y1,x2,y2)"].split(";"):
                seg = seg.strip()
                if seg and ":" in seg:
                    name, coords = seg.split(":", 1)
                    try:
                        x1, y1, x2, y2 = (int(float(v)) for v in coords.strip("() ").split(","))
                    except ValueError:
                        continue
                    boxes.append(("cand", name, -1, x1, y1, x2, y2))

        scale = args.width / w
        img = cv2.resize(img, (args.width, int(h * scale)), interpolation=cv2.INTER_AREA)
        H = img.shape[0]
        # 网格：每 1/10，标原图坐标
        for k in range(1, 10):
            xg = int(args.width * k / 10)
            yg = int(H * k / 10)
            cv2.line(img, (xg, 0), (xg, H), (0, 200, 200), 1)
            cv2.line(img, (0, yg), (args.width, yg), (0, 200, 200), 1)
            cv2.putText(img, str(int(w * k / 10)), (xg + 2, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 160, 160), 1, cv2.LINE_AA)
            cv2.putText(img, str(int(h * k / 10)), (4, yg + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 160, 160), 1, cv2.LINE_AA)
        for kind, c, i, x1, y1, x2, y2 in boxes:
            X1, Y1, X2, Y2 = [int(v * scale) for v in (x1, y1, x2, y2)]
            if kind == "old":
                col = TAG_COLOR.get(c, (128, 128, 128))
                cv2.rectangle(img, (X1, Y1), (X2, Y2), col, 2)
                cv2.putText(img, f"old{ i } {ABBR.get(c, '?')}", (X1, max(14, Y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
            else:
                cv2.rectangle(img, (X1, Y1), (X2, Y2), (0, 0, 255), 3)
                cv2.putText(img, f"cand {c}", (X1, min(H - 6, Y2 + 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)
        header = np.full((34, args.width, 3), 30, np.uint8)
        cv2.putText(header, f"#{n} {r['split']}  orig {w}x{h}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        imwrite_u(out_dir / f"g_{n:03d}.jpg", np.vstack([header, img]))
        print(f"g_{n:03d}.jpg  old={sum(1 for b in boxes if b[0]=='old')} "
              f"cand={sum(1 for b in boxes if b[0]=='cand')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
