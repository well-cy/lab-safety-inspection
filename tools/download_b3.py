# -*- coding: utf-8 -*-
"""E4: 下载 B3 (ez-spn/lab-coat-v8qin, 479张, CC BY 4.0) 到 data/raw/b3/
该项目 versions=0 未发布，走 REST search API + 单图 API 逐张下载（同 dataset2 方案）。
保留原生类别（Goggles/Gloves/Lab_coat），映射到 4 类在集成阶段做。
"""
import time
import requests
from pathlib import Path
from collections import Counter

API_KEY = "nNQADL3zz9PMVKGnkYc1"
BASE = "https://api.roboflow.com"
WS, PROJ = "ez-spn", "lab-coat-v8qin"
DEST = Path("data/raw/b3")

for sp in ("train", "valid", "test"):
    (DEST / sp / "images").mkdir(parents=True, exist_ok=True)
    (DEST / sp / "labels").mkdir(parents=True, exist_ok=True)

sess = requests.Session()

# ---- 1. search 列出全部图片 ----
all_imgs, offset = [], 0
while True:
    r = sess.post(f"{BASE}/{WS}/{PROJ}/search?api_key={API_KEY}",
                  json={"offset": offset, "limit": 250, "in_dataset": True,
                        "fields": ["id", "name", "split"]}, timeout=60)
    r.raise_for_status()
    batch = r.json().get("results", []) or r.json().get("images", [])
    if not batch:
        break
    all_imgs += batch
    offset += len(batch)
    if len(batch) < 250:
        break
print(f"search 列出 {len(all_imgs)} 张")

# ---- 2. 逐张下载（断点续传：图片已存在则只补标注）----
stats = Counter()
rows = []
for i, meta in enumerate(all_imgs, 1):
    iid, name, split = meta["id"], meta["name"], meta.get("split") or "train"
    imgp = DEST / split / "images" / name
    labp = DEST / split / "labels" / (Path(name).stem + ".txt")

    need_img = not imgp.exists() or imgp.stat().st_size == 0
    need_lab = not labp.exists()

    if not (need_img or need_lab):
        stats["skip"] += 1
        continue

    for attempt in range(4):
        try:
            r = sess.get(f"{BASE}/{WS}/{PROJ}/images/{iid}?api_key={API_KEY}", timeout=60)
            r.raise_for_status()
            d = r.json().get("image", r.json())
            imgp.parent.mkdir(parents=True, exist_ok=True)
            if need_img:
                src = (d.get("urls") or {}).get("original")
                if not src:
                    raise ValueError("no urls.original")
                ir = sess.get(src, timeout=120)
                ir.raise_for_status()
                imgp.write_bytes(ir.content)
                stats["img"] += 1
            if need_lab:
                ann = d.get("annotation") or {}
                W = float(ann.get("width") or 0)
                H = float(ann.get("height") or 0)
                boxes = ann.get("boxes") or []
                lines = []
                for a in boxes:
                    if not (W and H):
                        W = W or float(a.get("width") or 0)
                        H = H or float(a.get("height") or 0)
                    if not (W and H):
                        continue
                    x, y = float(a.get("x", 0)), float(a.get("y", 0))
                    w, h = float(a.get("width", 0)), float(a.get("height", 0))
                    if w <= 0 or h <= 0:
                        continue
                    # polygon 的 x/y/w/h 即外接矩形；类别用原生 label
                    cls = (a.get("label") or a.get("class") or "").strip()
                    cx = (x + w / 2) / W
                    cy = (y + h / 2) / H
                    nw, nh = w / W, h / H
                    if 0 < cx < 1 and 0 < cy < 1 and 0 < nw <= 1 and 0 < nh <= 1:
                        lines.append(f"{cls}\t{cx:.6f}\t{cy:.6f}\t{nw:.6f}\t{nh:.6f}")
                labp.parent.mkdir(parents=True, exist_ok=True)
                labp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                stats["lab"] += 1
            rows.append((split, name, len(lines) if need_lab else "?"))
            break
        except Exception as e:
            if attempt == 3:
                print(f"  FAIL {name}: {e}")
                stats["fail"] += 1
            else:
                time.sleep(2 ** attempt)

# ---- 3. 汇总 ----
print("stats:", dict(stats))
for sp in ("train", "valid", "test"):
    ni = len(list((DEST / sp / "images").glob("*")))
    nl = len(list((DEST / sp / "labels").glob("*.txt")))
    cls = Counter()
    for f in (DEST / sp / "labels").glob("*.txt"):
        for line in f.read_text(encoding="utf-8").splitlines():
            p = line.split("\t")
            if p and p[0]:
                cls[p[0]] += 1
    print(f"{sp}: images {ni} / labels {nl} / classes {dict(cls)}")
