#!/usr/bin/env python3
"""cluster_phash_list.py — 对清单内图片做**纯图像内容**pHash 聚类（忽略标注差异）

为什么要单独一个工具：dedup_phash.py 为了不误伤"同机位不同帧"的训练样本，
在 pHash 分组后又做了标注感知二次分桶——框坐标略有差异的近重复就会被拆开。
但外部公开库（尤其网图）里同一模特/同一场景的摆拍系列**本来就是冗余**，
必须按内容聚类才能看清重复率、按家族精简。

用法：
    python tools/cluster_phash_list.py --root data/raw/sglasses_proj \
        --list outputs/goggles_ext_review/sglasses_proj_cand_list.txt \
        --thr 10 --keep 1 --report data/sglasses_proj_family_report.txt \
        --keep-list data/sglasses_proj_keep.txt

输出：家族报告（组规模、成员、建议保留者）+ 保留清单（keep 张/家族）。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def imread_unicode(p: Path):
    try:
        return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def phash(img, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    size = hash_size * highfreq_factor
    gray = cv2.cvtColor(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(np.float32(gray))
    low = dct[:hash_size, :hash_size].flatten()
    med = np.median(low[1:])
    h = 0
    for b in (low > med).astype(np.uint64):
        h = (h << 1) | int(b)
    return h


POP = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)


def dist_to(h: int, rep: np.ndarray) -> np.ndarray:
    x = np.uint64(h) ^ rep
    d = POP[(x & np.uint64(0xFFFF)).astype(np.uint64)].astype(np.uint8)
    d += POP[((x >> np.uint64(16)) & np.uint64(0xFFFF)).astype(np.uint64)]
    d += POP[((x >> np.uint64(32)) & np.uint64(0xFFFF)).astype(np.uint64)]
    d += POP[((x >> np.uint64(48)) & np.uint64(0xFFFF)).astype(np.uint64)]
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--list", required=True, help="清单（每行 <split>/<文件名>）")
    ap.add_argument("--thr", type=int, default=10, help="pHash 汉明距离阈值")
    ap.add_argument("--keep", type=int, default=1, help="每家族保留几张")
    ap.add_argument("--report", default="")
    ap.add_argument("--keep-list", default="", help="保留清单输出路径")
    args = ap.parse_args()

    root = Path(args.root)
    rels = [ln.strip() for ln in Path(args.list).read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")]

    def resolve(rel: str) -> Path:
        """兼容 raw 布局 <root>/<split>/images/<name> 与项目布局 <root>/images/<split>/<name>。"""
        p = root / rel
        if p.exists():
            return p
        parts = rel.split("/")
        if len(parts) >= 2:
            sp, name = parts[0], parts[-1]
            for cand in (root / sp / "images" / name, root / "images" / sp / name):
                if cand.exists():
                    return cand
        return p

    hashes, names = [], []
    for rel in rels:
        ip = resolve(rel)
        if not ip.exists():
            print(f"[跳过] 找不到 {ip}")
            continue
        img = imread_unicode(ip)
        if img is None:
            print(f"[跳过] 读图失败 {ip}")
            continue
        hashes.append(phash(img))
        names.append(rel)
    print(f"[聚类] {len(names)} 张，阈值 {args.thr}（纯内容，忽略标注）")

    rep_arr = np.empty(0, dtype=np.uint64)
    rep_idx: list[int] = []
    assign: dict[int, int] = {}
    for i, h in enumerate(hashes):
        if rep_arr.size == 0:
            rep_arr, rep_idx = np.array([np.uint64(h)], dtype=np.uint64), [i]
            assign[i] = i
            continue
        d = dist_to(h, rep_arr)
        j = int(d.argmin())
        if int(d[j]) <= args.thr:
            assign[i] = rep_idx[j]
        else:
            rep_arr = np.append(rep_arr, np.uint64(h))
            rep_idx.append(i)
            assign[i] = i

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(names)):
        groups[assign[i]].append(i)

    dup = {k: v for k, v in groups.items() if len(v) > 1}
    n_dup = sum(len(v) for v in dup.values())
    print(f"  家族数 {len(groups)}  含重复家族 {len(dup)}  涉及图片 {n_dup}  "
          f"→ 保留 {args.keep} 张/家族后可降为 "
          f"{len(groups) + sum(max(0, len(v) - args.keep) * 0 for v in dup.values())} 张")

    keep: list[str] = []
    rep_all: list[str] = []
    for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        v_sorted = sorted(v, key=lambda i: names[i])
        rep_all.append(names[v_sorted[0]])
        keep.extend(names[i] for i in v_sorted[:args.keep])

    if args.report:
        out = [f"# 内容 pHash 家族报告 —— root={root}  阈值<={args.thr}  共 {len(names)} 张",
               f"# 家族 {len(groups)} 个，其中 {len(dup)} 个家族含 >=2 张重复",
               ""]
        for gi, (k, v) in enumerate(sorted(groups.items(), key=lambda kv: -len(kv[1])), 1):
            mark = "重复" if len(v) > 1 else "单张"
            out.append(f"[家族{gi}] {len(v)} 张 ({mark})")
            for i in sorted(v, key=lambda i: names[i]):
                tag = "保留" if names[i] in keep else "剔除"
                out.append(f"    {tag}: {names[i]}")
        Path(args.report).write_text("\n".join(out) + "\n", encoding="utf-8")
        print(f"[报告] {args.report}")

    if args.keep_list:
        Path(args.keep_list).write_text("\n".join(keep) + "\n", encoding="utf-8")
        print(f"[保留清单] {args.keep_list}  ({len(keep)} 张)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
