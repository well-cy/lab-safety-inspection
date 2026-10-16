import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
CPPE = ROOT / "data" / "raw" / "cppe5" / "extracted"
OUT = ROOT / "outputs" / "cppe5_review"
OUT.mkdir(parents=True, exist_ok=True)


def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def put(img, txt, org, color, sc=0.6):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, sc, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, sc, color, 1, cv2.LINE_AA)


TARGET = {"Mask", "Gloves", "Goggles", "Coverall"}  # 对她项目有参考价值的先看，Face_Shield 另出
COLS, ROWS, CELL = 4, 3, 330

splits = {}
for sp in ("train", "test"):
    jf = CPPE / "annotations" / f"{sp}.json"
    if not jf.exists():
        continue
    d = json.loads(jf.read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    by_cat = defaultdict(list)
    for a in d["annotations"]:
        by_cat[cats[a["category_id"]]].append(a)
    print(f"[{sp}] 图 {len(d['images'])} / 框 {len(d['annotations'])}")
    for name in sorted(by_cat, key=lambda n: -len(by_cat[n])):
        print(f"   {name:12} 框 {len(by_cat[name]):5}  覆盖图 {len({a['image_id'] for a in by_cat[name]}):4}")
    splits[sp] = d

    # 每类抽样拼版
    idx = {im["id"]: im for im in d["images"]}
    if sp == "train":
        for name in sorted(by_cat, key=lambda n: -len(by_cat[n])):
            im_ids = sorted({a["image_id"] for a in by_cat[name]})
            step = max(1, len(im_ids) // (COLS * ROWS))
            picked = im_ids[::step][: COLS * ROWS]
            sheet = np.full((ROWS * CELL, COLS * CELL, 3), 235, np.uint8)
            for k, iid in enumerate(picked):
                im = idx[iid]
                ip = CPPE / "images" / im["file_name"]
                img = imread_u(ip)
                if img is None:
                    continue
                for a in by_cat[name]:
                    if a["image_id"] != iid:
                        continue
                    x, y, w, h = a["bbox"]
                    cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 140, 255), 2)
                    put(img, name, (int(x), max(14, int(y) - 4)), (0, 140, 255), 0.55)
                H, W = img.shape[:2]
                s = min(CELL / W, CELL / H)
                r = cv2.resize(img, (max(1, int(W * s)), max(1, int(H * s))))
                y0 = (k // COLS) * CELL + (CELL - r.shape[0]) // 2
                x0 = (k % COLS) * CELL + (CELL - r.shape[1]) // 2
                sheet[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
                put(sheet, im["file_name"], ((k % COLS) * CELL + 6, (k // COLS) * CELL + 20), (90, 90, 90), 0.5)
            op = OUT / f"{sp}_{name}.jpg"
            cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tofile(str(op))
            print(f"   → {op.relative_to(ROOT)}（抽 {len(picked)} 张）")

# 宽高比/尺寸概况（域特征）
imgs = splits["train"]["images"]
ws = np.array([i["width"] for i in imgs])
hs = np.array([i["height"] for i in imgs])
print(f"\ntrain 尺寸中位 {int(np.median(ws))}x{int(np.median(hs))}  "
      f"范围 {ws.min()}x{hs.min()} ~ {ws.max()}x{hs.max()}")
