#!/usr/bin/env python3
"""dupcheck_orb_list.py — 用 ORB 特征匹配检测"同一张照片的裁剪/缩放/水印变体"

为什么需要它：pHash 是全局 64 位摘要，对**裁剪、缩放、加边框**很敏感——
同一张原图裁成两半，pHash 距离可能大于 10，被判成"不同图"，但它们其实是同源冗余。
ORB 关键点匹配（带 Lowe 比值检验 + RANSAC 单应估计）能给出：
    * 好匹配点数 good
    * 单应内点数 inliers —— 内点数高 = 同一平面物体（同一张照片的变体）
用于外部公开库的精简：同源冗余会虚增训练量、拉偏类别分布。

用法：
    python tools/dupcheck_orb_list.py --root data/raw/sglasses_proj \
        --list outputs/goggles_ext_review/sglasses_proj_cand_list.txt \
        --min-inliers 25 --report data/sglasses_proj_orb_report.txt \
        --drop-list data/sglasses_proj_orb_drop.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def imread_unicode(p: Path):
    try:
        return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def resolve(root: Path, rel: str) -> Path:
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--max-side", type=int, default=640, help="特征提取前缩放长边（提速）")
    ap.add_argument("--min-inliers", type=int, default=25,
                    help="判定同源的内点下限（越大越严格）")
    ap.add_argument("--report", default="")
    ap.add_argument("--drop-list", default="", help="建议剔除清单（每家族保留首张）")
    args = ap.parse_args()

    root = Path(args.root)
    rels = [ln.strip() for ln in Path(args.list).read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")]
    orb = cv2.ORB_create(nfeatures=1200)

    names, descs, kps = [], [], []
    for rel in rels:
        ip = resolve(root, rel)
        img = imread_unicode(ip)
        if img is None:
            print(f"[跳过] {ip}")
            continue
        h, w = img.shape[:2]
        s = args.max_side / max(h, w)
        if s < 1:
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, de = orb.detectAndCompute(g, None)
        names.append(rel)
        kps.append(kp)
        descs.append(de)
    print(f"[ORB] {len(names)} 张，两两匹配中…")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    n = len(names)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    pairs = []
    for i in range(n):
        if descs[i] is None or len(descs[i]) < 10 or kps[i] is None:
            continue
        for j in range(i + 1, n):
            if descs[j] is None or len(descs[j]) < 10:
                continue
            try:
                m = bf.knnMatch(descs[i], descs[j], k=2)
            except cv2.error:
                continue
            good = [p[0] for p in m if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
            if len(good) < 12:
                continue
            src = np.float32([kps[i][g.queryIdx].pt for g in good]).reshape(-1, 1, 2)
            dst = np.float32([kps[j][g.trainIdx].pt for g in good]).reshape(-1, 1, 2)
            Hm, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            inl = int(mask.sum()) if mask is not None else 0
            if inl >= args.min_inliers:
                pairs.append((i, j, len(good), inl))
                union(i, j)
        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{n}")

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    dup = {k: v for k, v in groups.items() if len(v) > 1}
    n_dup = sum(len(v) for v in dup.values())
    print(f"  同源家族 {len(dup)} 个，涉及 {n_dup} 张（总 {n}）")
    print(f"  可精简 {n_dup - len(dup)} 张 → 保留 {n - (n_dup - len(dup))} 张")

    if args.report:
        out = [f"# ORB 同源检测报告 —— root={root}  内点阈值>={args.min_inliers}",
               f"# 共 {n} 张，同源家族 {len(dup)} 个，涉及 {n_dup} 张",
               ""]
        for gi, (k, v) in enumerate(sorted(dup.items(), key=lambda kv: -len(kv[1])), 1):
            out.append(f"[家族{gi}] {len(v)} 张")
            for i in sorted(v):
                out.append(f"    {names[i]}")
        out.append("")
        out.append("## 高置信配对（内点最多的 40 对）")
        for i, j, good, inl in sorted(pairs, key=lambda x: -x[3])[:40]:
            out.append(f"  inliers={inl:<4} good={good:<4} {names[i]}  <->  {names[j]}")
        Path(args.report).write_text("\n".join(out) + "\n", encoding="utf-8")
        print(f"[报告] {args.report}")

    if args.drop_list:
        drops = []
        for k, v in sorted(dup.items()):
            vs = sorted(v)
            drops.extend(names[i] for i in vs[1:])
        if drops:
            Path(args.drop_list).write_text("\n".join(drops) + "\n", encoding="utf-8")
            print(f"[建议剔除] {args.drop_list}  ({len(drops)} 张)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
