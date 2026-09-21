# -*- coding: utf-8 -*-
"""通用「新数据集」去重/泄漏审计（入库前最后一道关）。

用途：任何准备加入训练集的公开数据集，先跑这个脚本，确认：
    1. 自身完整性（图片↔标注对应、空标注、越界框、类别统计）
    2. 与 dataset1(3 splits) / dataset2(external_test) / b3 的
       文件名重复 / MD5+SHA256 完全重复 / pHash 近重复(汉明距离<=10)
    3. 自身内部重复
    4. 结论：可安全入库的图片数，以及必须剔除的图片清单

红线：与 dataset2(external_test) 有命中的图片必须整批剔除（测试集泄漏）。

用法:
    python tools/dedup_new_dataset.py --src data/raw/labcoat_si --name labcoat_si

输出:
    data/<name>_dedup_report.txt     完整报告
    data/<name>_leak_images.txt      必须剔除的图片清单（含原因）

说明:
    - 只读，不修改任何原始数据
    - 图片读取用 np.fromfile + imdecode，兼容中文/印地语等非 ASCII 文件名
"""
import argparse
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

PHASH_HAMMING_MAX = 10
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OOB_TOL = 0.05  # 边界容差：超出 [-0.05, 1.05] 视为越界


# ---------------------------------------------------------------- 基础工具
def imread_unicode(path: Path):
    """Windows 下 cv2.imread 读不了非 ASCII 路径，用 fromfile + imdecode 绕过。"""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def phash(img, hash_size=8):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (hash_size * 4, hash_size * 4))
    dct = cv2.dct(gray.astype("float32"))
    flat = dct[:hash_size, :hash_size].flatten()
    med = sorted(flat)[len(flat) // 2]
    v = 0
    for b in (flat > med):
        v = (v << 1) | int(b)
    return v


def hamming_matrix(src_hashes, other_hashes):
    """返回 shape=(len(src), len(other)) 的汉明距离矩阵。"""
    s = np.asarray(src_hashes, dtype=np.uint64)[:, None]
    o = np.asarray(other_hashes, dtype=np.uint64)[None, :]
    x = np.bitwise_xor(s, o)
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1).reshape(
        len(src_hashes), len(other_hashes)
    )


def collect(d: Path, group: str):
    """收集 (group, 相对名, 路径)。兼容 {split}/images 与 images 两种布局。"""
    out = []
    if (d / "train" / "images").exists():
        for sp in ("train", "valid", "test"):
            sd = d / sp / "images"
            if sd.exists():
                out += [(group, f"{sp}/{p.name}", p) for p in sorted(sd.iterdir())
                        if p.suffix.lower() in IMG_EXT]
    elif (d / "images").exists():
        out += [(group, p.name, p) for p in sorted((d / "images").iterdir())
                if p.suffix.lower() in IMG_EXT]
    return out


def scan_self(src: Path, name: str, lines, A):
    """自身完整性 + 标注质量，返回 (文件列表, 越界图片集合)。"""
    files = collect(src, name)
    A(f"\n---- 1. 自身完整性 ----")
    A(f"{name} 图片落盘数: {len(files)}")

    missing_label, empty_label, orphan_label = [], [], []
    cls_counter = Counter()
    oob_imgs = set()
    oob_boxes = 0
    box_total = 0
    cls_names = {}

    # data.yaml 类别名
    yml = src / "data.yaml"
    if yml.exists():
        try:
            import yaml
            meta = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
            nm = meta.get("names")
            if isinstance(nm, list):
                cls_names = {i: n for i, n in enumerate(nm)}
            elif isinstance(nm, dict):
                cls_names = {int(k): v for k, v in nm.items()}
            A(f"data.yaml 类别: {cls_names}")
            A(f"data.yaml 许可证: {meta.get('license', '(未标注)')}")
        except Exception as e:
            A(f"[warn] 读 data.yaml 失败: {e}")

    for _, rel, p in files:
        sp = rel.split("/", 1)[0] if "/" in rel else ""
        lab = src / sp / "labels" / (p.stem + ".txt")
        if not lab.exists():
            missing_label.append(rel)
            continue
        txt = lab.read_text(encoding="utf-8", errors="ignore").strip()
        if not txt:
            empty_label.append(rel)
            continue
        for ln in txt.splitlines():
            parts = ln.split()
            if len(parts) != 5:
                continue
            cls_raw = parts[0]
            try:
                xc, yc, w, h = map(float, parts[1:])
            except ValueError:
                continue
            box_total += 1
            try:
                cname = cls_names.get(int(float(cls_raw)), cls_raw)
            except ValueError:
                cname = cls_raw
            cls_counter[cname] += 1
            x1, x2, y1, y2 = xc - w / 2, xc + w / 2, yc - h / 2, yc + h / 2
            if x1 < -OOB_TOL or x2 > 1 + OOB_TOL or y1 < -OOB_TOL or y2 > 1 + OOB_TOL:
                oob_imgs.add(rel)
                oob_boxes += 1

    for sp in ("train", "valid", "test"):
        ld = src / sp / "labels"
        if not ld.exists():
            continue
        for lf in ld.glob("*.txt"):
            has_img = any((src / sp / "images" / (lf.stem + e)).exists() for e in IMG_EXT)
            if not has_img:
                orphan_label.append(f"{sp}/{lf.name}")

    A(f"标注框总数: {box_total}  类别分布: {dict(cls_counter)}")
    A(f"缺标注的图片: {len(missing_label)} {missing_label[:5]}")
    A(f"空标注图片(疑负样本): {len(empty_label)}")
    A(f"有标注无图片: {len(orphan_label)} {orphan_label[:5]}")
    A(f"越界框: {oob_boxes} 个，涉及 {len(oob_imgs)} 张图  {sorted(oob_imgs)[:5]}")

    # 跨 split 同名
    by_name = defaultdict(list)
    for _, rel, _ in files:
        by_name[Path(rel).name.lower()].append(rel)
    same = {k: v for k, v in by_name.items() if len(v) > 1}
    A(f"内部同名文件(跨split): {len(same)} 组 {list(same.items())[:3]}")

    return files, oob_imgs


def main() -> int:
    ap = argparse.ArgumentParser(description="新数据集去重/泄漏审计")
    ap.add_argument("--src", required=True, help="待审数据集目录，如 data/raw/labcoat_si")
    ap.add_argument("--name", default=None, help="数据集简称，默认取目录名")
    ap.add_argument("--refs", default="dataset1,dataset2,b3",
                    help="比对参照集（逗号分隔），默认 dataset1,dataset2,b3")
    args = ap.parse_args()

    src = (ROOT / args.src) if not Path(args.src).is_absolute() else Path(args.src)
    name = args.name or src.name
    refs = [r.strip() for r in args.refs.split(",") if r.strip()]

    lines = []
    A = lines.append
    A("=" * 72)
    A(f"{name} 去重 / 泄漏审计报告  (对照: {', '.join(refs)})")
    A("=" * 72)

    src_files, oob_imgs = scan_self(src, name, lines, A)

    # ---- 2. 汇集参照集 ----
    A(f"\n---- 2. 参照集规模 ----")
    other = []
    ref_set_names = set()
    for r in refs:
        d = RAW / r
        if not d.exists():
            A(f"{r:14} [不存在，跳过]")
            continue
        g = collect(d, r)
        A(f"{r:14} {len(g):5} 张")
        other += g
        ref_set_names.add(r)

    if not other:
        A("\n[error] 没有可比对的参照集，无法检查泄漏")
        Path(ROOT / "data" / f"{name}_dedup_report.txt").write_text("\n".join(lines), encoding="utf-8")
        return 1

    # ---- 3. 文件名重复 ----
    A(f"\n---- 3. 文件名重复 ----")
    ref_by_name = defaultdict(list)
    for g, n, _ in other:
        ref_by_name[Path(n).name.lower()].append((g, n))
    name_hits = {Path(n).name.lower(): v for _, n, _ in src_files
                 if Path(n).name.lower() in ref_by_name}
    A(f"{name} 与参照集同名: {len(name_hits)} 组")
    for k, v in list(name_hits.items())[:10]:
        A(f"  {k}: {ref_by_name[k][:3]}")

    # ---- 4. MD5 + SHA256 完全重复 ----
    A(f"\n---- 4. MD5+SHA256 完全重复 ----")
    ref_hash = {}
    for g, n, p in other:
        data = p.read_bytes()
        ref_hash[(hashlib.md5(data).hexdigest(), hashlib.sha256(data).hexdigest())] = (g, n)
    exact_hits = []
    src_md5 = {}
    for _, n, p in src_files:
        data = p.read_bytes()
        h = (hashlib.md5(data).hexdigest(), hashlib.sha256(data).hexdigest())
        if h in ref_hash:
            exact_hits.append((n, *ref_hash[h]))
        src_md5[h] = n
    A(f"完全重复: {len(exact_hits)} 张")
    for row in exact_hits[:20]:
        A(f"  {row[0]}  ==  [{row[1]}] {row[2]}")

    # ---- 5. pHash 近重复 ----
    A(f"\n---- 5. pHash 近重复 (汉明距离 <= {PHASH_HAMMING_MAX}) ----")
    src_ok, src_hashes, src_failed = [], [], []
    for g, n, p in src_files:
        img = imread_unicode(p)
        if img is None:
            src_failed.append(n)
            continue
        src_ok.append((g, n, p))
        src_hashes.append(phash(img))

    ref_ok, ref_hashes = [], []
    for g, n, p in other:
        img = imread_unicode(p)
        if img is None:
            continue
        ref_ok.append((g, n))
        ref_hashes.append(phash(img))

    if src_failed:
        A(f"[warn] {len(src_failed)} 张图无法解码（已跳过）: {src_failed[:5]}")

    near_hits = []
    if src_hashes and ref_hashes:
        dm = hamming_matrix(src_hashes, ref_hashes)
        for i in range(dm.shape[0]):
            row = dm[i]
            idx = np.where(row <= PHASH_HAMMING_MAX)[0]
            if len(idx):
                best = idx[np.argmin(row[idx])]
                near_hits.append((src_ok[i][1], ref_ok[best][0], ref_ok[best][1], int(row[best])))
    A(f"近重复: {len(near_hits)} 张")
    for row in near_hits[:20]:
        A(f"  {row[0]}  ~  [{row[1]}] {row[2]}  (距离 {row[3]})")

    # ---- 6. 自身内部重复 ----
    A(f"\n---- 6. 自身内部重复 ----")
    dup_name = defaultdict(list)
    for _, n, _ in src_files:
        dup_name[Path(n).name.lower()].append(n)
    inner_dup = {k: v for k, v in dup_name.items() if len(v) > 1}
    A(f"自身同名: {len(inner_dup)} 组")
    # 内容重复（同一 md5 出现多次）
    md5_count = Counter()
    for h in src_md5:
        md5_count[h] += 1
    inner_exact = {h: c for h, c in md5_count.items() if c > 1}
    A(f"自身内容完全重复: {len(inner_exact)} 组（{'、'.join(src_md5[h] for h in list(inner_exact)[:5])}）")

    # ---- 7. 结论与剔除清单 ----
    must_drop = {}          # rel_name -> 原因
    for n, g, rn in exact_hits:
        must_drop[n] = f"与 {g} 完全重复({rn})"
    for n, g, rn, d in near_hits:
        must_drop.setdefault(n, f"与 {g} 近重复({rn}, 距离{d})")
    for n in oob_imgs:
        must_drop.setdefault(n, "含越界框（标注不可用）")

    leak_to_ds2 = [n for n, g, _ in exact_hits if g == "dataset2"] + \
                  [n for n, g, _, _ in near_hits if g == "dataset2"]

    A(f"\n---- 7. 结论 ----")
    A(f"{name} 总图片数: {len(src_files)}")
    A(f"必须剔除: {len(must_drop)} 张")
    A(f"  - 与 dataset2(external_test) 命中【红线，必须整批剔除】: {len(set(leak_to_ds2))} 张")
    A(f"可安全入库: {len(src_files) - len(must_drop)} 张")
    if must_drop:
        cnt = Counter(v.split("（")[0].split("(")[0] for v in must_drop.values())
        A(f"剔除原因分布: {dict(cnt)}")

    report = ROOT / "data" / f"{name}_dedup_report.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    drop_file = ROOT / "data" / f"{name}_leak_images.txt"
    with open(drop_file, "w", encoding="utf-8") as f:
        f.write(f"# {name} 必须剔除的图片（供人工复核）\n")
        f.write("# 格式: 相对路径\t原因\n")
        for k in sorted(must_drop):
            f.write(f"{k}\t{must_drop[k]}\n")

    print(f"\n报告: {report}")
    print(f"剔除清单: {drop_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
