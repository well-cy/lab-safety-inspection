# -*- coding: utf-8 -*-
"""E4: B3 (ez-spn/lab-coat-v8qin) 去重审计。

检查项:
    1. B3 内部完整性（图片↔标注一一对应、空标注、类别统计）
    2. B3 <-> dataset1 (train/valid/test) 与 B3 <-> dataset2 (external_test):
       文件名重复 / MD5+SHA256 完全重复 / pHash(64bit) 汉明距离<=10 近重复
    3. 版权风险信号: 文件名含 getty/istock/shutterstock 等图库水印来源
    4. B3 内部重复

结论: B3 可安全使用的图片数量（剔除泄漏与重复后）。
"""
import csv
import hashlib
import sys
from collections import defaultdict, Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
REPORT = ROOT / "data" / "b3_dedup_report.txt"

PHASH_HAMMING_MAX = 10

B3 = RAW / "b3"
GROUPS = {
    "b3": B3,
    "ds1_train": RAW / "dataset1" / "train" / "images",
    "ds1_valid": RAW / "dataset1" / "valid" / "images",
    "ds1_test": RAW / "dataset1" / "test" / "images",
}

STOCK_PATTERNS = ("gettyimages", "istockphoto", "shutterstock", "alamy",
                  "dreamstime", "123rf", "depositphotos", "adobestock")


def find_images(d: Path, group: str):
    """收集 (group, rel_name, path)。B3 与 dataset2 都有 train/valid/test 结构。"""
    out = []
    if (d / "train" / "images").exists():
        for sp in ("train", "valid", "test"):
            sd = d / sp / "images"
            if sd.exists():
                out += [(group, f"{sp}/{p.name}", p) for p in sorted(sd.iterdir())
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    elif (d / "images").exists():
        out += [(group, p.name, p) for p in sorted((d / "images").iterdir())
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    return out


def phash(img, hash_size=8):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (hash_size * 4, hash_size * 4))
    dct = cv2.dct(gray.astype("float32"))
    dct_low = dct[:hash_size, :hash_size]
    flat = dct_low.flatten()
    med = sorted(flat)[len(flat) // 2]
    v = 0
    for b in (flat > med):
        v = (v << 1) | int(b)
    return v


def hamming(a, b):
    return bin(a ^ b).count("1")


def main() -> int:
    lines = []
    A = lines.append

    # ---- 0. B3 完整性 ----
    A("=" * 70)
    A("B3 (ez-spn/lab-coat-v8qin, 479张, CC BY 4.0) 去重审计报告")
    A("=" * 70)

    b3_files = find_images(B3, "b3")
    A(f"\nB3 图片落盘数: {len(b3_files)}（项目页标称 479）")

    # 图片↔标注对应
    missing_label, empty_label, orphan_label = [], [], []
    cls_counter = Counter()
    for _, rel, p in b3_files:
        sp, name = rel.split("/", 1)
        lab = B3 / sp / "labels" / (Path(name).stem + ".txt")
        if not lab.exists():
            missing_label.append(rel)
            continue
        txt = lab.read_text(encoding="utf-8").strip()
        if not txt:
            empty_label.append(rel)
        for ln in txt.splitlines():
            parts = ln.split("\t")
            if parts and parts[0]:
                cls_counter[parts[0]] += 1
    for sp in ("train", "valid", "test"):
        ld = B3 / sp / "labels"
        if ld.exists():
            for lf in ld.glob("*.txt"):
                if not any(rel == f"{sp}/{p.stem}{p.suffix}" for _, rel, p in b3_files if rel.startswith(sp + "/")):
                    img_exists = any((B3 / sp / "images" / (lf.stem + e)).exists()
                                     for e in (".jpg", ".jpeg", ".png", ".bmp", ".webp"))
                    if not img_exists:
                        orphan_label.append(f"{sp}/{lf.name}")
    A(f"B3 类别框统计: {dict(cls_counter)}")
    A(f"缺标注的图片: {len(missing_label)} {missing_label[:5]}")
    A(f"空标注图片(背景图): {len(empty_label)} {empty_label[:5]}")
    A(f"有标注无图片: {len(orphan_label)} {orphan_label[:5]}")

    # 同名不同 split（下载时可能相互覆盖）
    by_name = defaultdict(list)
    for _, rel, p in b3_files:
        by_name[Path(rel).name.lower()].append(rel)
    same_name = {k: v for k, v in by_name.items() if len(v) > 1}
    A(f"B3 内部同名文件(跨split): {len(same_name)} 组 {list(same_name.items())[:5]}")

    # ---- 1. 汇集全部图片 ----
    files = list(b3_files)
    for g, d in GROUPS.items():
        if g == "b3":
            continue
        imgs = find_images(d, g) if (d / "train" / "images").exists() else \
              [(g, p.name, p) for p in sorted(d.iterdir())
               if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
        A(f"{g:10} {len(imgs):5} 张")
        files += imgs
    # dataset2 (external_test)
    ds2 = find_images(RAW / "dataset2", "ds2")
    A(f"{'ds2':10} {len(ds2):5} 张")
    files += ds2

    # ---- 2. 文件名重复（B3 vs 其他）----
    name_map = defaultdict(list)
    for g, n, p in files:
        name_map[Path(n).name.lower()].append((g, n))
    b3_name_conflict = {}
    for k, v in name_map.items():
        groups = {g for g, _ in v}
        if len(v) > 1 and "b3" in groups and len(groups) > 1:
            b3_name_conflict[k] = v
    A(f"\n---- 文件名重复 ----")
    A(f"B3 与 dataset1/dataset2 同名: {len(b3_name_conflict)} 组")
    for k, v in list(b3_name_conflict.items())[:20]:
        A(f"  {k}: {v}")

    # ---- 3. MD5+SHA256 ----
    md5_map = {}
    exact_pairs = []
    for g, n, p in files:
        data = p.read_bytes()
        h = (hashlib.md5(data).hexdigest(), hashlib.sha256(data).hexdigest())
        if h in md5_map:
            exact_pairs.append((g, n, *md5_map[h]))
        else:
            md5_map[h] = (g, n)
    b3_exact = [p_ for p_ in exact_pairs if p_[0] == "b3" or p_[2] == "b3"]
    A(f"\n---- MD5+SHA256 完全重复 ----")
    A(f"涉及 B3 的完全重复对: {len(b3_exact)}")
    for p_ in b3_exact[:20]:
        A(f"  {p_}")

    # ---- 4. pHash 近重复 ----
    ph = []
    for g, n, p in files:
        img = cv2.imdecode(np.fromfile(str(p), dtype="uint8"), cv2.IMREAD_COLOR)
        if img is None:
            continue
        ph.append((g, n, phash(img)))
    A(f"\n---- pHash 近重复 (汉明距离<={PHASH_HAMMING_MAX}) ----")

    b3_idx = [i for i, (g, _, _) in enumerate(ph) if g == "b3"]
    other_idx = [i for i, (g, _, _) in enumerate(ph) if g != "b3"]
    near_pairs = []
    for i in b3_idx:
        for j in other_idx:
            d = hamming(ph[i][2], ph[j][2])
            if d <= PHASH_HAMMING_MAX:
                near_pairs.append((ph[i][0], ph[i][1], ph[j][0], ph[j][1], d))
    # B3 内部
    b3_internal = []
    for a in range(len(b3_idx)):
        for b in range(a + 1, len(b3_idx)):
            i, j = b3_idx[a], b3_idx[b]
            d = hamming(ph[i][2], ph[j][2])
            if d <= PHASH_HAMMING_MAX:
                b3_internal.append((ph[i][1], ph[j][1], d))

    near_ds1 = [p_ for p_ in near_pairs if p_[2].startswith("ds1")]
    near_ds2 = [p_ for p_ in near_pairs if p_[2] == "ds2"]
    A(f"B3 <-> dataset1 近重复对: {len(near_ds1)}")
    for p_ in near_ds1[:30]:
        A(f"  {p_}")
    A(f"B3 <-> dataset2(external_test) 近重复对: {len(near_ds2)}  <-- 泄漏红线")
    for p_ in near_ds2[:30]:
        A(f"  {p_}")
    A(f"B3 内部近重复对: {len(b3_internal)}")
    for p_ in b3_internal[:30]:
        A(f"  {p_}")

    # ---- 5. 图库水印风险 ----
    stock = [rel for _, rel, _ in b3_files
             if any(s in Path(rel).name.lower() for s in STOCK_PATTERNS)]
    A(f"\n---- 版权风险（图库水印文件名）----")
    A(f"getty/istock 等图库来源文件: {len(stock)} 张")
    for s in stock[:30]:
        A(f"  {s}")

    # ---- 6. 结论 ----
    leaked_b3 = set()
    for p_ in exact_pairs:
        if p_[0] == "b3":
            leaked_b3.add(p_[1])
        elif p_[2] == "b3":
            leaked_b3.add(p_[3])
    for p_ in near_pairs:
        leaked_b3.add(p_[1])
    A(f"\n---- 结论 ----")
    A(f"B3 落盘图片: {len(b3_files)}")
    A(f"与 dataset1/dataset2 重复或近重复: {len(leaked_b3)} 张")
    A(f"图库水印风险图: {len(stock)} 张")
    clean = len(b3_files) - len(leaked_b3) - len(stock)
    A(f"剔除泄漏+图库图后 B3 可用: {clean} 张")
    A("(B3 最终是否可用以本报告 + 用户决策为准)")

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    with open(ROOT / "data" / "b3_dedup_pairs.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["group_a", "file_a", "group_b", "file_b", "hamming"])
        for p_ in near_pairs:
            w.writerow(p_)
        for p_ in b3_internal:
            w.writerow(("b3", p_[0], "b3", p_[1], p_[2]))
    print("\n".join(lines))
    print(f"\n报告已保存: {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
