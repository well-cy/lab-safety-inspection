"""CPPE-5 第三遍：背景连通域占比（bg_ratio）特征 + 重判。

原理：影棚/产品图的背景是连通的大块均匀色，从四边 flood fill 能吞掉画面 30%~80%；
自然场景（即使白墙）会被阴影、设备、门框、家具切断，占比低。

bg_ratio 计算方式：
  1) 边框像素聚成 ≤2 个主色（覆盖"上白墙+下灰地"双色调背景）
  2) 与任一主色距离 < 28 的像素记为"类背景"
  3) 取与边框连通的类背景连通域面积 / 全图

重判（在 person_pass 结果基础上追加）：
  product = 原判定
          | (person_n<=1 & bg_ratio>0.34)     ← 单人/无人 + 大块连通背景 = 棚拍/目录图
"""
import csv
import io
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "cppe5_screen"
CPPE = ROOT / "data" / "raw" / "cppe5" / "extracted"
CSV_P = OUT / "features.csv"


def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def bg_ratio(img):
    H0, W0 = img.shape[:2]
    s = 256.0 / max(H0, W0)
    if s < 1:
        img = cv2.resize(img, (max(1, int(W0 * s)), max(1, int(H0 * s))), interpolation=cv2.INTER_AREA)
    H, W = img.shape[:2]
    blur = cv2.GaussianBlur(img, (5, 5), 0).astype(np.float32)

    b = max(3, int(min(H, W) * 0.07))
    ring = np.concatenate([blur[:b].reshape(-1, 3), blur[-b:].reshape(-1, 3),
                           blur[:, :b].reshape(-1, 3), blur[:, -b:].reshape(-1, 3)])
    # 边框主色 ≤2（kmeans）
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(ring, 2, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    # 只保留占比 > 12% 的主色
    ref = []
    for k in range(2):
        if (labels == k).mean() > 0.12:
            ref.append(centers[k])
    if not ref:
        ref = [ring.mean(axis=0)]

    dist = np.min(np.stack([np.linalg.norm(blur - r, axis=2) for r in ref]), axis=0)
    mask = (dist < 28).astype(np.uint8)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    border_labels = set(lab[:b].ravel()) | set(lab[-b:].ravel()) | \
                    set(lab[:, :b].ravel()) | set(lab[:, -b:].ravel())
    border_labels.discard(0)
    tot = sum(int((lab == l).sum()) for l in border_labels)
    return tot / (H * W)


def main():
    rows = list(csv.DictReader(CSV_P.open(encoding="utf-8")))
    print(f"载入 {len(rows)} 行，计算 bg_ratio …")
    for i, r in enumerate(rows):
        img = imread_u(CPPE / "images" / r["file"])
        r["bg_ratio"] = round(float(bg_ratio(img)), 4) if img is not None else 0.0
        if i % 200 == 0:
            print(f"  … {i}/{len(rows)}")

    for r in rows:
        pn = int(r["person_n"])
        br = float(r["bg_ratio"])
        if int(r["product"]) == 0 and pn <= 1 and br > 0.34:
            r["product"] = 1

    cols = list(rows[0].keys())
    with CSV_P.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    prod = sum(1 for r in rows if r["product"] == "1")
    print(f"\n三遍后：product {prod} / natural {len(rows)-prod}")
    for sp in ("train", "test"):
        sub = [r for r in rows if r["split"] == sp]
        c2 = Counter()
        for r in sub:
            if r["product"] == "0":
                for u in r["useful"].split("|"):
                    if u:
                        c2[u] += 1
        print(f"[{sp}] natural 含可用类：{dict(c2)}")


if __name__ == "__main__":
    main()
