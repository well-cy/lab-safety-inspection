"""CPPE-5 评审拼版生成器：按 _pool.json 出编号拼版 + 索引 CSV。

每版 30 张（6×5，cell 280），红色大编号角标 + 文件名，另写 review_index.csv（no→file 映射）。
"""
import csv
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
CPPE = ROOT / "data" / "raw" / "cppe5" / "extracted"
OUT = ROOT / "outputs" / "cppe5_screen"
SHEET_DIR = OUT / "sheets"
SHEET_DIR.mkdir(parents=True, exist_ok=True)

COLS, ROWS, CELL = 6, 5, 280


def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def put(img, txt, org, color, sc=0.6, th=2):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, sc, (255, 255, 255), th + 2, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, sc, color, th, cv2.LINE_AA)


def main():
    pool = json.loads((OUT / "_pool.json").read_text(encoding="utf-8"))
    page = COLS * ROWS
    with (OUT / "review_index.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["no", "file", "split", "classes", "bg_ratio", "person_n"])
        for r in pool:
            w.writerow([r["no"], r["file"], r["split"], r["useful"], r["bg_ratio"], r["person_n"]])
    pages = (len(pool) + page - 1) // page
    for pg in range(pages):
        chunk = pool[pg * page:(pg + 1) * page]
        sheet = np.full((ROWS * CELL, COLS * CELL, 3), 28, np.uint8)
        for k, r in enumerate(chunk):
            img = imread_u(CPPE / "images" / r["file"])
            if img is None:
                continue
            H, W = img.shape[:2]
            s = min(CELL / W, CELL / H)
            img = cv2.resize(img, (max(1, int(W * s)), max(1, int(H * s))), interpolation=cv2.INTER_AREA)
            y0 = (k // COLS) * CELL + (CELL - img.shape[0]) // 2
            x0 = (k % COLS) * CELL + (CELL - img.shape[1]) // 2
            sheet[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
            cx, cy = (k % COLS) * CELL, (k // COLS) * CELL
            tag = f"#{r['no']}"
            cv2.rectangle(sheet, (cx + 2, cy + 2), (cx + 10 + 17 * len(tag), cy + 36), (0, 0, 0), -1)
            put(sheet, tag, (cx + 8, cy + 29), (0, 0, 255), 0.95, 2)
            put(sheet, r["file"], (cx + 8, cy + CELL - 8), (190, 190, 190), 0.42, 1)
            put(sheet, r["useful"].replace("|", "+"), (cx + CELL - 8, cy + CELL - 8),
                (0, 230, 230), 0.45, 1, right=True) if False else None
        # 页脚
        put(sheet, f"sheet {pg+1}/{pages}   no {chunk[0]['no']}-{chunk[-1]['no']}",
            (10, ROWS * CELL - 10), (255, 255, 255), 0.7, 2)
        op = SHEET_DIR / f"review_{pg+1:02d}.jpg"
        cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile(str(op))
    print(f"评审池 {len(pool)} → {pages} 版 → {SHEET_DIR}")
    print("索引 → outputs/cppe5_screen/review_index.csv")


if __name__ == "__main__":
    main()
