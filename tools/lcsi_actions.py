#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""lcsi_actions.py — 依据四关审计结果，生成 lcsi 子集处置清单（只出清单，不动数据）

产出：
  data/_lcsi_dedup_zero.txt        A 类：pHash 完全一致（d=0）冗余，每分量留 1 张，其余列剔除
  data/_lcsi_leak_validside.txt    B 类：跨 split 近重复组中落在 valid 侧的图（valid 泄漏源）
  data/_lcsi_family_summary.txt    C 类：子树来源族结构（供域偏置/降采样决策）
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
from audit_lcsi_subset import fam_of, read_label  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(sys.stdout, "reconfigure") else None
LAB = ROOT / "data" / "lab_ppe"
PREFIX = "lcsi_"
THR = 8

# 与 r3 入库第①关一致：优先保留标注更全/已在 train 的副本
PRIORITY_VALID_KEEP = {"lcsi_27_jpg.rf.f79da049a10250757a4052c01bd59458.jpg"}  # 已挂待剔，见 outputs/lcsi_audit_pending.txt

ok = []
H = []
for sp in ["train", "valid"]:
    for p in sorted((LAB / "images" / sp).glob(PREFIX + "*")):
        im = imread_unicode(p)
        if im is None:
            continue
        ok.append((sp, p))
        H.append(phash(im))
H = np.array([np.uint64(h) for h in H], dtype=np.uint64)

D = hamming_matrix(H, H)
np.fill_diagonal(D, 255)


def comps_from(edges_mask) -> dict[int, list[int]]:
    parent = list(range(len(ok)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    ii, jj = np.where(edges_mask)
    for a, b in zip(ii.tolist(), jj.tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    out = defaultdict(list)
    for k in range(len(ok)):
        out[find(k)].append(k)
    return out


# --- A 类：d=0 完全冗余
comps0 = comps_from(D == 0)
dup_groups = {r: v for r, v in comps0.items() if len(v) > 1}
drop_a, keep_a = [], []
for r, v in sorted(dup_groups.items(), key=lambda x: -len(x[1])):
    # 保留策略：优先 valid？不 —— 优先 train 中的、标签框多者；同则取排序第一个
    def score(k):
        sp, p = ok[k]
        rows, _ = read_label(sp, p.stem)
        return (0 if sp == "valid" else 1, len(rows))
    v_sorted = sorted(v, key=score, reverse=True)
    keep_a.append(v_sorted[0])
    for k in v_sorted[1:]:
        drop_a.append(k)

lines_a = [
    "# lcsi A 类剔除清单：pHash 完全一致（d=0）冗余副本",
    "# 每个分量保留 1 张（优先 train、框多者），其余剔除；共 %d 张" % len(drop_a),
    "# 字段：<split>/<文件名>\\t<同名分量代表（保留）>",
]
for k in sorted(drop_a, key=lambda k: ok[k][1].name):
    sp, p = ok[k]
    lines_a.append(f"{sp}/{p.name}\tredun_d0")
lines_a.append("")
lines_a.append("## 分量明细（每组：保留 → 剔除…）")
for r, v in sorted(dup_groups.items(), key=lambda x: -len(x[1])):
    def score(k):
        sp, p = ok[k]
        rows, _ = read_label(sp, p.stem)
        return (0 if sp == "valid" else 1, len(rows))
    vs = sorted(v, key=score, reverse=True)
    lines_a.append(f"- 组{len(v)}：保留 {ok[vs[0]][0]}/{ok[vs[0]][1].name}")
    for k in vs[1:]:
        lines_a.append(f"       剔 {ok[k][0]}/{ok[k][1].name}")

# --- B 类：跨 split 近重复组 valid 侧
comps_all = comps_from(D <= THR)
leak_groups = []
for r, v in comps_all.items():
    if len(v) <= 1:
        continue
    sps = {ok[k][0] for k in v}
    if len(sps) > 1:
        val = [k for k in v if ok[k][0] == "valid"]
        trn = [k for k in v if ok[k][0] == "train"]
        leak_groups.append((len(v), trn, val))

lines_b = [
    "# lcsi B 类：跨 split 近重复（valid 泄漏源）",
    "# 口径：近重复连通分量应整组同 split。labcoat_si 原方案只进 train，",
    "#       故建议把下列 valid 侧图移回 train（或整组剔除），消除 valid 泄漏。",
    f"# 涉及 {len(leak_groups)} 组 / valid 侧 {sum(len(v) for _,_,v in leak_groups)} 张",
]
for n, trn, val in sorted(leak_groups, key=lambda x: -len(x[2])):
    lines_b.append(f"- 组{n}（train {len(trn)} / valid {len(val)}）")
    for k in trn:
        lines_b.append(f"    train {ok[k][1].name}")
    for k in val:
        lines_b.append(f"    valid {ok[k][1].name}   <== 待处理")

# --- C 类：家族结构
fam = defaultdict(lambda: dict(imgs=0, boxes=0, empty=0))
for sp, p in ok:
    rows, _ = read_label(sp, p.stem)
    f = fam_of(p.name)
    fam[f]["imgs"] += 1
    fam[f]["boxes"] += len(rows)
    if not rows:
        fam[f]["empty"] += 1
lines_c = ["# lcsi 子树来源族结构（域偏置/降采样决策参考）",
           f"{'家族':<26}{'图数':>6}{'框数':>6}{'空标签':>8}{'框/图':>8}"]
for f, d in sorted(fam.items(), key=lambda x: -x[1]["imgs"]):
    lines_c.append(f"{f:<26}{d['imgs']:>6}{d['boxes']:>6}{d['empty']:>8}{d['boxes']/max(d['imgs'],1):>8.2f}")

for path, lines in [(ROOT / "data" / "_lcsi_dedup_zero.txt", lines_a),
                    (ROOT / "data" / "_lcsi_leak_validside.txt", lines_b),
                    (ROOT / "data" / "_lcsi_family_summary.txt", lines_c)]:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[写出] {path.name}  ({len(lines)} 行)")

print(f"\nA 类冗余剔除：{len(drop_a)} 张（{len(dup_groups)} 组）")
print(f"B 类 valid 泄漏：{len(leak_groups)} 组 / valid 侧 {sum(len(v) for _,_,v in leak_groups)} 张")
print("\n".join(lines_c))
