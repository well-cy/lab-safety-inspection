#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dedup_phash.py —— 全库近重复（pHash）检测与去重清单

背景：按文件名家族（.rf. 之前）去重只能处理"同一原图的增强副本"，
     而不同来源（dataset1/dataset2/labcoat_si/ppes…）里可能存在**视觉上几乎相同**的图，
      文件名完全不同，家族去重抓不到。这类重复会：
        1. 让训练集虚高、模型过拟合这几张图
        2. 若一张在 train、另一张在 valid/test → 评估结果虚高（数据泄漏）

方法：
  - 每图算 64 位 pHash（32x32 灰度 DCT 低频 8x8，去直流后中位数二值化）
  - 4 段 16 位 LSH 分桶取候选对，再精确算汉明距离
  - 并查集聚类，输出重复组报告 + 可选的剔除清单

用法:
    python tools/dedup_phash.py                                  # 全库，阈值 6
    python tools/dedup_phash.py --class mask                     # 只看含 mask 框的图
    python tools/dedup_phash.py --thr 4 --report data/xxx.txt    # 更严格阈值
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "data" / "lab_ppe"
NAMES = ["mask", "gloves", "lab_coat", "goggles"]
SPLITS = ["train", "valid", "test"]


def imread_unicode(p: Path):
    try:
        return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def phash(img, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    size = hash_size * highfreq_factor
    gray = cv2.cvtColor(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(np.float32(gray))
    low = dct[:hash_size, :hash_size].flatten()
    med = np.median(low[1:])  # 去掉直流分量
    bits = (low > med).astype(np.uint64)
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DS),
                    help="数据集根目录（默认 data/lab_ppe，项目布局 <root>/images/<split>；"
                         "也支持 raw 布局 <root>/<split>/images）")
    ap.add_argument("--names", default="",
                    help="类别名（逗号分隔），默认项目四类 mask,gloves,lab_coat,goggles")
    ap.add_argument("--exclude", default="",
                    help="剔除清单（每行 <split>/<文件名>，可带 tab 原因），扫描时跳过")
    ap.add_argument("--class", dest="cls", default="", help="只看含该类框的图（mask/gloves/lab_coat/goggles 或 id）")
    ap.add_argument("--thr", type=int, default=6, help="pHash 汉明距离阈值（越小越严格，默认 6）")
    ap.add_argument("--keep", type=int, default=1, help="每组最多保留几张代表（默认 1）")
    ap.add_argument("--report", default="", help="报告输出路径")
    ap.add_argument("--index", default="", help="输出剔除索引（给 remove_bad_images.py）")
    args = ap.parse_args()

    root = Path(args.root)
    names = [n.strip() for n in args.names.split(",") if n.strip()] or list(NAMES)

    def label_of(sp: str, name: str) -> Path:
        """兼容两种布局定位标注文件。"""
        stem = Path(name).stem
        raw = root / sp / "labels" / (stem + ".txt")
        if raw.exists():
            return raw
        return root / "labels" / sp / (stem + ".txt")

    want_cid = None
    if args.cls:
        tok = args.cls.strip().lower()
        want_cid = int(tok) if tok.isdigit() else names.index(tok)

    excluded: set[str] = set()
    if args.exclude:
        for ln in Path(args.exclude).read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                excluded.add(ln.split("\t")[0].strip())

    items: list[tuple[str, str, Path, list[int]]] = []  # (split, 文件名, path, class_ids)
    for sp in SPLITS:
        idir = root / "images" / sp
        if not idir.is_dir():                       # raw 数据集布局
            idir = root / sp / "images"
        if not idir.is_dir():
            continue
        for ip in sorted(idir.iterdir()):
            if ip.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            if f"{sp}/{ip.name}" in excluded:
                continue
            lp = label_of(sp, ip.name)
            cids: list[int] = []
            if lp.exists():
                for ln in lp.read_text(encoding="utf-8").splitlines():
                    if ln.strip():
                        cids.append(int(float(ln.split()[0])))
            if want_cid is not None and want_cid not in cids:
                continue
            items.append((sp, ip.name, ip, cids))

    print(f"[扫描] {len(items)} 张图（{'全部类别' if want_cid is None else names[want_cid]}），计算 pHash…")
    hashes = []
    for i, (sp, stem, ip, cids) in enumerate(items):
        img = imread_unicode(ip)
        hashes.append(phash(img) if img is not None else None)
        if (i + 1) % 4000 == 0:
            print(f"  {i+1}/{len(items)}")

    # LSH 分桶找候选
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, h in enumerate(hashes):
        if h is None:
            continue
        for band in range(4):
            buckets[(band, (h >> (band * 16)) & 0xFFFF)].append(i)

    # 严格聚类：每张图只与"已确认为代表"的图比较，杜绝 A~B~C 传递链把
    # 同一场景的不同帧误并成一组（union-find 会链式漂移）。
    # 用 numpy 向量化 popcount 加速。
    POP = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)

    def dist_to_reps(h: int, rep_arr: np.ndarray) -> np.ndarray:
        x = np.uint64(h) ^ rep_arr
        d = POP[(x & np.uint64(0xFFFF)).astype(np.uint64)].astype(np.uint8)
        d += POP[((x >> np.uint64(16)) & np.uint64(0xFFFF)).astype(np.uint64)]
        d += POP[((x >> np.uint64(32)) & np.uint64(0xFFFF)).astype(np.uint64)]
        d += POP[((x >> np.uint64(48)) & np.uint64(0xFFFF)).astype(np.uint64)]
        return d

    order = [i for i in range(len(items)) if hashes[i] is not None]
    rep_arr = np.empty(0, dtype=np.uint64)
    rep_idx: list[int] = []
    assign: dict[int, int] = {}  # img_idx -> rep_idx（重复图 -> 代表图）
    for i in order:
        h = hashes[i]
        if rep_arr.size == 0:
            rep_arr = np.array([np.uint64(h)], dtype=np.uint64)
            rep_idx.append(i)
            assign[i] = i
            continue
        d = dist_to_reps(h, rep_arr)
        j = int(d.argmin())
        if int(d[j]) <= args.thr:
            assign[i] = rep_idx[j]
        else:
            rep_arr = np.append(rep_arr, np.uint64(h))
            rep_idx.append(i)
            assign[i] = i

    groups: dict[int, list[int]] = defaultdict(list)
    for i in order:
        groups[assign[i]].append(i)

    # 标注感知二次分桶：pHash 只看画面整体，同机位不同人位的帧也会被并到一组；
    # 检测任务的"真冗余"要求框也基本重合。每组内部再按标注相似度细分。
    def boxes_of(idx: int) -> list[tuple[int, float, float, float, float]]:
        lp = label_of(items[idx][0], items[idx][1])
        outb = []
        if lp.exists():
            for ln in lp.read_text(encoding="utf-8").splitlines():
                p = ln.split()
                if len(p) >= 5:
                    c, xc, yc, w, h = int(float(p[0])), *map(float, p[1:5])
                    outb.append((c, xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2))
        return outb

    def ann_similar(a: list, b: list) -> bool:
        if abs(len(a) - len(b)) > max(1, 0.1 * max(len(a), len(b))):
            return False
        if not a and not b:
            return True
        used = [False] * len(b)
        ious = []
        matched = 0
        for ca, ax1, ay1, ax2, ay2 in a:
            best, bi = 0.0, -1
            for k, (cb, bx1, by1, bx2, by2) in enumerate(b):
                if used[k] or cb != ca:
                    continue
                ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
                iy = max(0.0, min(ay2, by2) - max(ay1, by1))
                inter = ix * iy
                union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
                iou = inter / union if union > 0 else 0.0
                if iou > best:
                    best, bi = iou, k
            if best >= 0.80:
                used[bi] = True
                matched += 1
                ious.append(best)
        if not a and not b:
            return True
        frac = matched / max(len(a), len(b))
        return frac >= 0.90 and (sum(ious) / len(ious) if ious else 1.0) >= 0.85

    final_groups: dict[int, list[int]] = defaultdict(list)
    n_split = 0
    for gkey, members in groups.items():
        subs: list[tuple[list, list[int]]] = []
        for i in members:
            bx = boxes_of(i)
            placed = False
            for sbx, sidx in subs:
                if ann_similar(bx, sbx):
                    sidx.append(i)
                    placed = True
                    break
            if not placed:
                subs.append((bx, [i]))
        if len(subs) > 1:
            n_split += len(subs) - 1
        for _, sidx in subs:
            final_groups[gkey if len(subs) == 1 else id(sidx)] = sidx
    dup_groups = {k: v for k, v in final_groups.items() if len(v) > 1}

    n_dup_imgs = sum(len(v) for v in dup_groups.values())
    n_extra = n_dup_imgs - len(dup_groups)

    # 跨 split 泄漏统计
    leak_groups = 0
    leak_imgs = 0
    for v in dup_groups.values():
        sps = {items[i][0] for i in v}
        if len(sps) > 1:
            leak_groups += 1
            leak_imgs += len(v)

    out: list[str] = []
    add = out.append
    add(f"# 近重复（pHash）检测报告 —— 阈值 汉明距离 <= {args.thr}")
    add(f"# 扫描图片: {len(items)}  重复组: {len(dup_groups)}  涉及图片: {n_dup_imgs}  可精简: {n_extra}")
    add(f"# 其中跨 split 组（泄漏风险）: {leak_groups} 组 / {leak_imgs} 张")
    add("")
    sizes = Counter(len(v) for v in dup_groups.values())
    add("## 组规模分布")
    for k in sorted(sizes, reverse=True):
        add(f"  组内 {k} 张: {sizes[k]} 组")
    add("")
    add("## 重复组明细（每组只列前 6 张）")
    idx_lines: list[str] = []
    n = 0
    for gi, (gkey, v) in enumerate(sorted(dup_groups.items(), key=lambda kv: -len(kv[1])), 1):
        sps = Counter(items[i][0] for i in v)
        tag = "  <<< 跨split泄漏" if len(sps) > 1 else ""
        add(f"[组{gi}] {len(v)} 张  splits={dict(sps)}{tag}")
        # 保留策略：优先 train（保证训练量）→ 框数最多 → 名字最短
        v_sorted = sorted(v, key=lambda i: (items[i][0] != "train", -len(items[i][3]), len(items[i][1])))
        keep_n = max(1, args.keep)
        keep = v_sorted[0]
        add(f"    保留: {items[keep][0]}/{items[keep][1]}  ({len(items[keep][3])} 框)")
        for extra in v_sorted[1:keep_n]:
            add(f"    保留: {items[extra][0]}/{items[extra][1]}  ({len(items[extra][3])} 框)")
        for i in v_sorted[keep_n:]:
            add(f"    剔除: {items[i][0]}/{items[i][1]}  ({len(items[i][3])} 框)")
            n += 1
            idx_lines.append(f"{n}\t{items[i][0]}/{items[i][1]}\tdup:phash")

    # 导出前 3 大组的前 40 张（带框拼版验证用）
    top3 = sorted(dup_groups.values(), key=len, reverse=True)[:3]
    for rank, g in enumerate(top3, 1):
        lines = [f"{items[i][0]}/{items[i][2].name}\tdup_g{rank}" for i in sorted(g)[:40]]
        Path(ROOT / "data" / f"_top{rank}_sheet.txt").write_text("# 验证用\n" + "\n".join(lines) + "\n", encoding="utf-8")

    report = Path(args.report) if args.report else (ROOT / "data" / "phash_dedup_report.txt")
    report.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out[:14]))
    print(f"\n[报告] {report}")

    if args.index and idx_lines:
        ip = Path(args.index)
        ip.write_text("# pHash 近重复剔除清单（每组保留 1 张）\n" + "\n".join(idx_lines) + "\n", encoding="utf-8")
        print(f"[剔除索引] {ip}  ({len(idx_lines)} 张)")
    print(f"  代表图数: {len(rep_idx)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
