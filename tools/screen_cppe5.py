"""CPPE-5 可用图筛选（第一遍：自动预筛）。

红线（docs/E6_AI_优化与评测方案.md §2.1）：训练图须为"实验室/科研/医疗操作场景"；
电商商品图（白底影棚/套件图）、水印图不可入训练集。

类别口径：Mask✅ / Gloves✅ / Goggles⚠(需确认为护目镜本体) / Coverall❌ / Face_Shield❌。

本工具输出：
  outputs/cppe5_screen/features.csv      每图特征 + 自动判定（product / natural）
  outputs/cppe5_screen/natural_<类>.jpg  存活候选按类拼版（供目检）
  outputs/cppe5_screen/sheet_<N>.jpg     存活候选全量分页拼版（编号）
"""
import csv
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
CPPE = ROOT / "data" / "raw" / "cppe5" / "extracted"
OUT = ROOT / "outputs" / "cppe5_screen"
OUT.mkdir(parents=True, exist_ok=True)

NAMES = {1: "Coverall", 2: "Face_Shield", 3: "Gloves", 4: "Goggles", 5: "Mask"}
USEFUL = {3: "Gloves", 4: "Goggles", 5: "Mask"}  # 有映射价值的类


def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def feat(img):
    """返回域特征：白底程度 / 边框均匀度 / 饱和度 / 边缘密度。"""
    if img is None:
        return None
    H, W = img.shape[:2]
    # 缩小加速
    s = 512.0 / max(H, W)
    if s < 1:
        img = cv2.resize(img, (max(1, int(W * s)), max(1, int(H * s))), interpolation=cv2.INTER_AREA)
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1].astype(np.float32)
    V = hsv[:, :, 2].astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    white_ratio = float(((V > 210) & (S < 30)).mean())      # 近白像素占比
    light_ratio = float(((V > 190) & (S < 60)).mean())      # 浅色低饱和占比
    sat_mean = float(S.mean())

    b = max(3, int(min(H, W) * 0.07))
    ring = np.zeros((H, W), bool)
    ring[:b, :] = ring[-b:, :] = ring[:, :b] = ring[:, -b:] = True
    bv, bs = V[ring], S[ring]
    border_v_mean, border_v_std = float(bv.mean()), float(bv.std())
    border_s_mean = float(bs.mean())

    edges = cv2.Canny(gray, 60, 160)
    edge_density = float((edges > 0).mean())
    lapvar = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    return dict(W=W, H=H, ar=round(W / H, 3),
                white_ratio=round(white_ratio, 4),
                light_ratio=round(light_ratio, 4),
                sat_mean=round(sat_mean, 2),
                border_v_mean=round(border_v_mean, 2),
                border_v_std=round(border_v_std, 2),
                border_s_mean=round(border_s_mean, 2),
                edge_density=round(edge_density, 4),
                lapvar=round(lapvar, 1))


def classify_product(f):
    """白底/纯色影棚产品图判定（保守：宁可放过，不可错杀自然场景）。"""
    # 近白底或有大片浅色低饱和背景
    if f["white_ratio"] > 0.30:
        return True
    if f["light_ratio"] > 0.55 and f["border_s_mean"] < 40:
        return True
    # 边框极亮且极均匀 = 影棚背景
    if f["border_v_mean"] > 225 and f["border_v_std"] < 12:
        return True
    if f["border_v_mean"] > 205 and f["border_v_std"] < 8 and f["border_s_mean"] < 25:
        return True
    return False


def load_split(sp):
    jf = CPPE / "annotations" / f"{sp}.json"
    d = json.loads(jf.read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    by_img = defaultdict(list)
    for a in d["annotations"]:
        by_img[a["image_id"]].append(a)
    return d, cats, by_img


def main():
    rows = []
    for sp in ("train", "test"):
        d, cats, by_img = load_split(sp)
        idx = {im["id"]: im for im in d["images"]}
        for im in d["images"]:
            anns = by_img.get(im["id"], [])
            cls = sorted({cats[a["category_id"]] for a in anns})
            useful = sorted({cats[a["category_id"]] for a in anns if a["category_id"] in USEFUL})
            ip = CPPE / "images" / im["file_name"]
            img = imread_u(ip)
            f = feat(img)
            if f is None:
                print("  读取失败:", im["file_name"])
                continue
            is_prod = classify_product(f)
            rows.append(dict(split=sp, file=im["file_name"], iid=im["id"],
                             n_cls=len(cls), classes="|".join(cls),
                             useful="|".join(useful),
                             product=int(is_prod),
                             **{k: f[k] for k in ("W", "H", "ar", "white_ratio", "light_ratio",
                                                  "sat_mean", "border_v_mean", "border_v_std",
                                                  "border_s_mean", "edge_density", "lapvar")}))

    cols = list(rows[0].keys())
    with (OUT / "features.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    tot = len(rows)
    prod = sum(r["product"] for r in rows)
    print(f"总计 {tot} 图；自动判产品图 {prod}（{prod/tot:.1%}），自然场景候选 {tot-prod}")
    print()
    # 有映射价值类的分布（拆分 split）
    for sp in ("train", "test"):
        sub = [r for r in rows if r["split"] == sp]
        c = Counter()
        for r in sub:
            for u in r["useful"].split("|"):
                if u:
                    c[u] += 1
        c2 = Counter()
        for r in sub:
            if r["product"] == 0:
                for u in r["useful"].split("|"):
                    if u:
                        c2[u] += 1
        print(f"[{sp}] {len(sub)} 图")
        for k in ("Mask", "Gloves", "Goggles"):
            print(f"   {k:9} 含该类 {c[k]:4} 图 → 自动预筛后 {c2[k]:4}")
        only_prod = sum(1 for r in sub if r["useful"] and r["product"] == 1)
        print(f"   （有可用类的图中，被自动判产品图 {only_prod}）")
    print(f"\n特征表 → outputs/cppe5_screen/features.csv")


if __name__ == "__main__":
    main()
