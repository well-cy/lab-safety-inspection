# -*- coding: utf-8 -*-
"""E1-4 标注可视化抽检：从 data/lab_ppe 各 split 随机抽取共20张，画框输出到 outputs/label_check/。

用法:
    python tools/visualize_labels.py [数量，默认20]
"""
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / "data" / "lab_ppe"
OUT = ROOT / "outputs" / "label_check"

SPLITS = ["train", "valid", "test"]
CLASS_NAMES = {0: "mask", 1: "gloves", 2: "lab_coat", 3: "goggles"}
COLORS = {0: (60, 60, 230), 1: (60, 200, 60), 2: (230, 160, 60), 3: (200, 60, 200)}


def imwrite_safe(path: Path, img):
    ext = ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        path.write_bytes(buf.tobytes())
    return ok


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    random.seed(42)
    OUT.mkdir(parents=True, exist_ok=True)

    pool = []
    for sp in SPLITS:
        img_dir = DST / "images" / sp
        if not img_dir.exists():
            continue
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                pool.append((sp, p))
    if not pool:
        print("[error] data/lab_ppe 无图片，请先运行转换")
        return 1

    samples = random.sample(pool, min(n, len(pool)))
    # 保证尽量均衡：每 split 至少抽几张（若数量允许）
    print(f"抽样 {len(samples)} / {len(pool)} 张")
    per_class_boxes = {v: 0 for v in CLASS_NAMES.values()}
    dropped_check = []

    for i, (sp, img_path) in enumerate(samples, 1):
        img = cv2.imdecode(np.fromfile(str(img_path), dtype="uint8"), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        lbl = DST / "labels" / sp / (img_path.stem + ".txt")
        n_box = 0
        if lbl.exists():
            for line in lbl.read_text(encoding="utf-8").strip().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                cid = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
                x1, y1 = int((cx - w / 2) * W), int((cy - h / 2) * H)
                x2, y2 = int((cx + w / 2) * W), int((cy + h / 2) * H)
                color = COLORS[cid]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, CLASS_NAMES[cid], (x1, max(y1 - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                per_class_boxes[CLASS_NAMES[cid]] += 1
                n_box += 1
        cv2.putText(img, f"{sp}/{img_path.name} boxes={n_box}", (5, H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        imwrite_safe(OUT / f"check_{i:02d}_{img_path.stem[:40]}.jpg", img)
        if n_box == 0:
            dropped_check.append(f"{sp}/{img_path.name} (无保留标注)")
        print(f"  [{i:02d}] {sp}/{img_path.name}  框数={n_box}")

    print("\n抽检中各类别框数:", per_class_boxes)
    if dropped_check:
        print("无保留标注（原标注全被丢弃）的图片:")
        for d in dropped_check:
            print("  ", d)
    print(f"\n可视化已保存到: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
