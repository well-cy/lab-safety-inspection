# -*- coding: utf-8 -*-
"""E1-5 数据集去重检查（防数据泄漏）。

检查范围:
    dataset1 train / dataset1 valid / dataset1 test / dataset2（全部图片）

检查项:
    1. 文件名重复（basename 相同）
    2. MD5 / SHA256 完全重复（逐字节相同）
    3. 感知哈希 pHash(64bit) 汉明距离 <= 阈值 的近重复

重点输出:
    - dataset1 各 split 与 dataset2 之间的重复 / 近重复数量
    - dataset1 内部跨 split 重复
    - 结论：dataset2 剩余可用独立测试图片数量
"""
import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
REPORT = ROOT / "data" / "dedup_report.txt"
PAIRS_CSV = ROOT / "data" / "dedup_pairs.csv"

GROUPS = {
    "ds1_train": RAW / "dataset1" / "train" / "images",
    "ds1_valid": RAW / "dataset1" / "valid" / "images",
    "ds1_test": RAW / "dataset1" / "test" / "images",
    "ds2_all": RAW / "dataset2",  # dataset2 若未分 split，则直接扫描根目录下所有图
}
PHASH_HAMMING_MAX = 10  # 64bit pHash, <=10 视为高度相似


def find_images(d: Path):
    if not d.exists():
        return []
    if (d / "train" / "images").exists():  # dataset2 有标准划分
        out = []
        for sp in ("train", "valid", "test"):
            out += [(f"{sp}/{p.name}", p) for p in sorted((d / sp / "images").iterdir())
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
        return out
    return [(p.name, p) for p in sorted(d.iterdir())
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]


def phash(img, hash_size=8):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (hash_size * 4, hash_size * 4))
    dct = cv2.dct(gray.astype("float32"))
    dct_low = dct[:hash_size, :hash_size]
    med = (dct_low.flatten()[1:] + dct_low.flat[0]) / 2  # 近似：跳过DC
    med = sorted(dct_low.flatten())[len(dct_low.flatten()) // 2]
    bits = (dct_low.flatten() > med).astype("uint8")
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def hamming(a, b):
    return bin(a ^ b).count("1")


def main() -> int:
    files = []  # (group, rel_name, path)
    for g, d in GROUPS.items():
        imgs = find_images(d)
        print(f"{g:10} {len(imgs):5} 张  ({d})")
        files += [(g, n, p) for n, p in imgs]

    if not files:
        print("[error] 未找到任何图片")
        return 1

    # ---- 1. 文件名重复 ----
    by_name = defaultdict(list)
    for g, n, p in files:
        by_name[Path(n).name.lower()].append((g, n))
    name_dup_groups = {k: v for k, v in by_name.items() if len(v) > 1}

    # ---- 2. MD5/SHA256 ----
    md5_map = {}
    sha_map = {}
    exact_dup_pairs = []
    for g, n, p in files:
        data = p.read_bytes()
        h1 = hashlib.md5(data).hexdigest()
        h2 = hashlib.sha256(data).hexdigest()
        if h1 in md5_map and h2 in sha_map.get(h1, set()):
            g2, n2 = md5_map[h1]
            exact_dup_pairs.append((g, n, g2, n2, "md5+sha256"))
        else:
            md5_map[h1] = (g, n)
            sha_map.setdefault(h1, set()).add(h2)

    # ---- 3. pHash 近重复 ----
    ph_list = []
    for g, n, p in files:
        img = cv2.imdecode(__import__("numpy").fromfile(str(p), dtype="uint8"), cv2.IMREAD_COLOR)
        if img is None:
            continue
        ph_list.append((g, n, phash(img)))

    near_pairs = []
    ds2_idx = [i for i, (g, _, _) in enumerate(ph_list) if g == "ds2_all"]
    other_idx = [i for i, (g, _, _) in enumerate(ph_list) if g != "ds2_all"]
    # dataset1 <-> dataset2 全量比对（重点）
    for i in ds2_idx:
        for j in other_idx:
            d = hamming(ph_list[i][2], ph_list[j][2])
            if d <= PHASH_HAMMING_MAX:
                near_pairs.append((ph_list[i][0], ph_list[i][1], ph_list[j][0], ph_list[j][1], d))
    # dataset1 内部跨 split（train/valid/test 之间）
    for a in range(len(other_idx)):
        for b in range(a + 1, len(other_idx)):
            i, j = other_idx[a], other_idx[b]
            if ph_list[i][0] == ph_list[j][0]:
                continue
            d = hamming(ph_list[i][2], ph_list[j][2])
            if d <= PHASH_HAMMING_MAX:
                near_pairs.append((ph_list[i][0], ph_list[i][1], ph_list[j][0], ph_list[j][1], d))

    # ---- 汇总 ----
    lines = []
    lines.append("=" * 70)
    lines.append("数据集去重检查报告（防数据泄漏）")
    lines.append("=" * 70)
    lines.append(f"\n检查图片总数: {len(files)}")
    lines.append(f"pHash 汉明距离阈值: <= {PHASH_HAMMING_MAX} (64bit)\n")

    lines.append("---- 1. 文件名重复 ----")
    lines.append(f"重复文件名组数: {len(name_dup_groups)}")
    for k, v in list(name_dup_groups.items())[:20]:
        lines.append(f"  {k}: {v}")

    lines.append("\n---- 2. MD5+SHA256 完全重复 ----")
    lines.append(f"完全重复图片对数: {len(exact_dup_pairs)}")
    for p_ in exact_dup_pairs[:20]:
        lines.append(f"  {p_}")

    lines.append("\n---- 3. pHash 近重复（重点：dataset1 vs dataset2）----")
    ds1_ds2_near = [p_ for p_ in near_pairs
                    if ("ds2_all" in (p_[0], p_[2])) and ("ds1_" in (p_[0], p_[2]))]
    ds1_internal = [p_ for p_ in near_pairs if p_ not in ds1_ds2_near]
    lines.append(f"dataset1 <-> dataset2 近重复对数: {len(ds1_ds2_near)}")
    for p_ in ds1_ds2_near[:30]:
        lines.append(f"  {p_}")
    lines.append(f"dataset1 内部跨 split 近重复对数: {len(ds1_internal)}")
    for p_ in ds1_internal[:30]:
        lines.append(f"  {p_}")

    # dataset2 被污染的图片集合
    leaked = set()
    for p_ in exact_dup_pairs:
        if "ds2_all" in (p_[0], p_[2]):
            leaked.add(p_[1] if p_[0] == "ds2_all" else p_[3])
    for p_ in ds1_ds2_near:
        leaked.add(p_[1] if p_[0] == "ds2_all" else p_[3])
    n_ds2 = sum(1 for g, _, _ in files if g == "ds2_all")
    lines.append("\n---- 结论 ----")
    lines.append(f"dataset2 图片总数: {n_ds2}")
    lines.append(f"与 dataset1 重复/近重复的 dataset2 图片数: {len(leaked)}")
    lines.append(f"dataset2 可保留的独立测试图片数: {n_ds2 - len(leaked)}")
    lines.append("(若泄漏数为0，dataset2 可正式锁定为 external_test；否则需剔除泄漏图片或换测试集)")

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    with open(PAIRS_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["group_a", "file_a", "group_b", "file_b", "hamming_or_hash"])
        for p_ in exact_dup_pairs:
            w.writerow(p_)
        for p_ in near_pairs:
            w.writerow(p_)
    print("\n".join(lines))
    print(f"\n报告已保存: {REPORT}")
    print(f"重复对明细: {PAIRS_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
