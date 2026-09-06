# -*- coding: utf-8 -*-
"""E4-1: lab_coat 数据审计（只读，不修改任何数据/代码）
对比 dataset1(训练源) 与 dataset2(external_test) 中 lab_coat 的视觉域差异。
用 HSV 饱和度作为"白大褂/白色衣物"的代理指标：白色衣物饱和度低。
"""
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "data" / "lab_ppe"          # dataset1 转换后（训练用）
D2 = ROOT / "data" / "raw" / "dataset2"  # external_test（只读！）
D2_CLS = {0: "Mask", 1: "Glasses", 2: "Person", 3: "Gloves",
          4: "Lab Coat", 5: "No Gloves", 6: "No Glasses", 7: "No Lab Coat",
          8: "No Mask", 9: "Shoes"}
D2_LABCOAT = 4

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_image(stem_dir, stem):
    for e in IMG_EXTS:
        p = stem_dir / (stem + e)
        if p.exists():
            return p
    return None


def load_labels(label_dir):
    """返回 {stem: [(cls, cx, cy, w, h), ...]}"""
    out = {}
    for f in label_dir.glob("*.txt"):
        boxes = []
        for line in f.read_text().splitlines():
            parts = line.split()
            if len(parts) == 5:
                boxes.append(tuple(float(x) for x in parts))
        out[f.stem] = boxes
    return out


def imread_unicode(path):
    """中文路径安全读图"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def box_saturation_brightness(img, box):
    """box=(cls,cx,cy,w,h) 归一化；返回框内中心70%区域的 HSV 饱和度均值/亮度均值/V>200比例"""
    H, W = img.shape[:2]
    cx, cy, bw, bh = box[1:5]
    x1 = int((cx - bw * 0.35) * W); x2 = int((cx + bw * 0.35) * W)
    y1 = int((cy - bh * 0.35) * H); y2 = int((cy + bh * 0.35) * H)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    roi = img[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat = float(hsv[:, :, 1].mean())
    val = float(hsv[:, :, 2].mean())
    white_ratio = float((hsv[:, :, 2] > 200).mean())  # 高亮度像素比例
    return sat, val, white_ratio


def analyze(name, label_dir, image_dir, labcoat_id):
    labels = load_labels(label_dir)
    imgs_with = {s: b for s, b in labels.items() if any(x[0] == labcoat_id for x in b)}
    stats = {
        "name": name,
        "total_imgs": len(labels),
        "imgs_with_labcoat": len(imgs_with),
        "labcoat_boxes": sum(1 for b in labels.values() for x in b if x[0] == labcoat_id),
    }
    # 框尺度（相对面积）
    areas = [x[3] * x[4] for b in labels.values() for x in b if x[0] == labcoat_id]
    if areas:
        a = np.array(areas)
        stats["area_pct"] = (float(a.min()), float(np.percentile(a, 25)),
                             float(np.median(a)), float(np.percentile(a, 75)), float(a.max()))
    # 每图 lab_coat 框数
    per_img = [sum(1 for x in b if x[0] == labcoat_id) for b in imgs_with.values()]
    stats["boxes_per_img"] = Counter(per_img)
    # 白色度代理：框内中心区域饱和度/亮度
    sats, vals, wrs = [], [], []
    small_skip = 0
    for stem, boxes in imgs_with.items():
        imgp = find_image(image_dir, stem)
        if imgp is None:
            continue
        img = imread_unicode(imgp)
        if img is None:
            continue
        for x in boxes:
            if x[0] != labcoat_id:
                continue
            if x[3] * x[4] < 0.002:  # 太小的框跳过（ROI噪声大）
                small_skip += 1
                continue
            r = box_saturation_brightness(img, x)
            if r:
                sats.append(r[0]); vals.append(r[1]); wrs.append(r[2])
    if sats:
        s, v, w = map(np.array, (sats, vals, wrs))
        stats["sat"] = (float(np.median(s)), float(np.percentile(s, 25)), float(np.percentile(s, 75)))
        stats["val"] = float(np.median(v))
        stats["white_ratio"] = (float(np.median(w)), float(np.percentile(w, 75)))
        stats["n_measured"] = len(sats)
    stats["small_skipped"] = small_skip
    return stats, imgs_with


def filename_clusters(stems, top=15):
    """按 rf 前缀聚类：xxx_123_jpeg_jpg.rf.hash → xxx"""
    c = Counter()
    for s in stems:
        base = re.split(r"_jpeg_jpg\.rf\.|\.rf\.|_\d+_(?:jpg|png|jpeg)$", s)[0]
        base = re.sub(r"_?\d+$", "", base)
        c[base[:60]] += 1
    return c.most_common(top)


print("=" * 70)
print("Part 1: dataset1 (data/lab_ppe, lab_coat=2) 各 split 审计")
print("=" * 70)
d1_stems_all = []
for sp in ("train", "valid", "test"):
    st, imgs_with = analyze(f"dataset1/{sp}", LAB / "labels" / sp, LAB / "images" / sp, 2)
    print(f"\n--- {sp} ---")
    print(f"  总图片: {st['total_imgs']}  含lab_coat图片: {st['imgs_with_labcoat']}  lab_coat框: {st['labcoat_boxes']}")
    if "area_pct" in st:
        mn, p25, med, p75, mx = st["area_pct"]
        print(f"  框相对面积 min={mn:.4f} p25={p25:.3f} med={med:.3f} p75={p75:.3f} max={mx:.3f}")
    print(f"  每图框数分布: {dict(st['boxes_per_img'])}")
    if "sat" in st:
        med, p25, p75 = st["sat"]
        print(f"  框内饱和度 med={med:.0f} p25={p25:.0f} p75={p75:.0f} (低=偏白)")
        print(f"  框内亮度 med={st['val']:.0f}  高亮像素比例 med={st['white_ratio'][0]:.2f} p75={st['white_ratio'][1]:.2f}")
        print(f"  实测框数: {st['n_measured']}  跳过过小框: {st['small_skipped']}")
    if sp == "train":
        d1_stems_all = list(imgs_with.keys())

print("\n" + "=" * 70)
print("Part 2: dataset1 train 含 lab_coat 图片的文件名来源聚类（前缀）")
print("=" * 70)
for k, v in filename_clusters(d1_stems_all):
    print(f"  {v:4d}  {k}")

print("\n" + "=" * 70)
print("Part 3: dataset2 (external_test, 原生Lab Coat=4) 域特征")
print("=" * 70)
for sp in ("train", "valid", "test"):
    st, imgs_with = analyze(f"dataset2/{sp}", D2 / sp / "labels", D2 / sp / "images", D2_LABCOAT)
    print(f"\n--- {sp} ---")
    print(f"  总图片: {st['total_imgs']}  含Lab Coat图片: {st['imgs_with_labcoat']}  Lab Coat框: {st['labcoat_boxes']}")
    if "area_pct" in st:
        mn, p25, med, p75, mx = st["area_pct"]
        print(f"  框相对面积 min={mn:.4f} p25={p25:.3f} med={med:.3f} p75={p75:.3f} max={mx:.3f}")
    if "sat" in st:
        med, p25, p75 = st["sat"]
        print(f"  框内饱和度 med={med:.0f} p25={p25:.0f} p75={p75:.0f} (低=偏白)")
        print(f"  框内亮度 med={st['val']:.0f}  高亮像素比例 med={st['white_ratio'][0]:.2f} p75={st['white_ratio'][1]:.2f}")
        print(f"  实测框数: {st['n_measured']}  跳过过小框: {st['small_skipped']}")

print("\n" + "=" * 70)
print("Part 4: dataset2 全部类别框数（背景信息）")
print("=" * 70)
c = Counter()
for sp in ("train", "valid", "test"):
    for b in load_labels(D2 / sp / "labels").values():
        for x in b:
            c[D2_CLS.get(int(x[0]), x[0])] += 1
for k, v in c.most_common():
    print(f"  {k:12} {v}")

print("\n" + "=" * 70)
print("Part 5: dataset1 中 lab_coat 框与其他类框的平均尺寸对比（train）")
print("=" * 70)
CLS = {0: "mask", 1: "gloves", 2: "lab_coat", 3: "goggles"}
byc = defaultdict(list)
for b in load_labels(LAB / "labels" / "train").values():
    for x in b:
        byc[int(x[0])].append(x[3] * x[4])
for i in range(4):
    a = np.array(byc[i])
    print(f"  {CLS[i]:9} n={len(a):5d}  相对面积 med={np.median(a):.4f}  p25={np.percentile(a,25):.4f}  p75={np.percentile(a,75):.4f}")
