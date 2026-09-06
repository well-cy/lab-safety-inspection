# -*- coding: utf-8 -*-
"""E4: B3 lab_coat 标注可视化抽检（12张，覆盖各来源系列）。"""
import random
from pathlib import Path

import cv2
import numpy as np

B3 = Path("data/raw/b3")
OUT = Path("outputs/b3_label_check")
OUT.mkdir(parents=True, exist_ok=True)

CLS = {"Lab_coat": (0, 165, 255), "Gloves": (0, 200, 0), "Goggles": (160, 0, 255)}

random.seed(2026)
# 按来源系列各抽几张
picks = []
groups = {}
for sp in ("train", "valid", "test"):
    for p in sorted((B3 / sp / "images").iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lab = B3 / sp / "labels" / (p.stem + ".txt")
        if not lab.exists() or "Lab_coat" not in lab.read_text(encoding="utf-8"):
            continue
        n = p.name.lower()
        if "gettyimages" in n or "istockphoto" in n:
            g = "stock"
        elif n.startswith("detect_") or "白大褂" in p.name:
            g = "self"
        elif "crop_margin" in n:
            g = "crop"
        elif n.startswith("lc"):
            g = "lc"
        elif "frame" in n:
            g = "frame"
        else:
            g = "misc"
        groups.setdefault(g, []).append((sp, p))

for g, lst in groups.items():
    k = 3 if g in ("self", "crop", "misc") else 2
    picks += random.sample(lst, min(k, len(lst)))
picks = picks[:12]

COLORS = {"Lab_coat": (0, 165, 255), "Gloves": (0, 200, 0), "Goggles": (160, 0, 255)}
for i, (sp, p) in enumerate(picks, 1):
    img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("read fail:", p.name)
        continue
    H, W = img.shape[:2]
    lab = B3 / sp / "labels" / (p.stem + ".txt")
    for ln in lab.read_text(encoding="utf-8").splitlines():
        parts = ln.split("\t")
        if len(parts) != 5:
            continue
        cls, cx, cy, w, h = parts[0], *map(float, parts[1:])
        x1, y1 = int((cx - w / 2) * W), int((cy - h / 2) * H)
        x2, y2 = int((cx + w / 2) * W), int((cy + h / 2) * H)
        color = COLORS.get(cls, (128, 128, 128))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, cls, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    out = OUT / f"b3check_{i:02d}_{p.stem[:40]}.jpg"
    ok, buf = cv2.imencode(".jpg", img)
    if ok:
        out.write_bytes(buf.tobytes())
        print(f"saved {out.name}  ({sp}/{p.name}, {len(lab.read_text(encoding='utf-8').splitlines())} boxes)")
