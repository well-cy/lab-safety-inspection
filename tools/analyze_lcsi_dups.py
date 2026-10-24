#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analyze_lcsi_dups.py — 拆解 lcsi 子集内部近重复连通分量（只读）

输出：每组的分支构成（train/valid）、组内逐对 pHash 距离、是否跨 split。
用途：判断哪些组是"完全同图冗余"（可直接去一）、哪些是"同场景相邻帧"
（应整组同 split），以及哪些组已造成 train/valid 泄漏。
"""
from __future__ import annotations

import io
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from crosslib_phash import phash, hamming_matrix, imread_unicode  # noqa: E402

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
LAB = ROOT / "data" / "lab_ppe"
PREFIX = "lcsi_"
THR = 8

imgs = []
for sp in ["train", "valid"]:
    for p in sorted((LAB / "images" / sp).glob(PREFIX + "*")):
        imgs.append((sp, p))

H, ok = [], []
for sp, p in imgs:
    im = imread_unicode(p)
    if im is None:
        continue
    H.append(phash(im))
    ok.append((sp, p))
H = np.array([np.uint64(h) for h in H], dtype=np.uint64)

D = hamming_matrix(H, H)
np.fill_diagonal(D, 255)
parent = list(range(len(ok)))


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


ii, jj = np.where(D <= THR)
for a, b in zip(ii.tolist(), jj.tolist()):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra

comps = defaultdict(list)
for k in range(len(ok)):
    comps[find(k)].append(k)

lines = []
lines.append(f"lcsi 内部近重复分量（thr<={THR}）  共 {sum(1 for v in comps.values() if len(v)>1)} 组")
n_cross = 0
cross_valid = set()
for r, v in sorted(comps.items(), key=lambda x: -len(x[1])):
    if len(v) <= 1:
        continue
    sps = sorted({ok[k][0] for k in v})
    cross = "跨split!" if len(sps) > 1 else ""
    if cross:
        n_cross += 1
        for k in v:
            if ok[k][0] == "valid":
                cross_valid.add(ok[k][1].name)
    lines.append("")
    lines.append(f"### 组大小 {len(v)}  split={sps} {cross}")
    # 组内逐对距离
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            d = int(D[v[i], v[j]])
            if d <= THR:
                lines.append(f"  d={d:>3}  {ok[v[i]][0]}/{ok[v[i]][1].name}")
                lines.append(f"        {ok[v[j]][0]}/{ok[v[j]][1].name}")

lines.append("")
lines.append("=" * 70)
lines.append(f"跨 split 组数：{n_cross}")
lines.append(f"跨 split 组中落在 valid 侧的图（需处理，防 valid 泄漏）：{len(cross_valid)}")
for n in sorted(cross_valid):
    lines.append(f"  valid/{n}")

out = ROOT / "data" / "lcsi_dup_analysis.txt"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines[-40:]))
print(f"\n[报告] {out}")
