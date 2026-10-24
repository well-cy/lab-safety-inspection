#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""apply_webscene_fix_r3b.py —— r3 轮1 补充：#14 背景人物两框下移（2026-10-09）

渲染目检发现背景人物 person/lab_coat 纵向偏高（框在头顶上方、
大褂下摆没盖到），按重渲染实测整体下移。
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
LBL = ROOT / "data" / "_prelabel" / "webscene_r3" / "labels"
BAK = ROOT / "data" / "_label_backup" / "webscene_r3a_20261009"
TOL = 0.07

STEM = "r3s_14_130938"
OPS = [
    ("SET", 0, 0.595, 0.370, 0.595, 0.5225, 0.15, 0.445),   # 背景人物下移拉长
    ("SET", 3, 0.597, 0.4125, 0.597, 0.58, 0.13, 0.31),     # 背景大褂下移拉长
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    lp = LBL / f"{STEM}.txt"
    rows = [l.split() for l in lp.read_text(encoding="utf-8").splitlines() if l.strip()]
    used = [False] * len(rows)
    log = []
    for kind, c, tx, ty, ncx, ncy, nw, nh in OPS:
        best, bd = -1, 1e9
        for i, r in enumerate(rows):
            if used[i] or int(r[0]) != c:
                continue
            d = ((float(r[1]) - tx) ** 2 + (float(r[2]) - ty) ** 2) ** 0.5
            if d < bd:
                bd, best = d, i
        if best < 0 or bd > TOL:
            log.append(f"!! SET {c} @({tx},{ty}) 未匹配 ({bd:.3f})")
            continue
        used[best] = True
        rows[best] = [str(c), f"{ncx:.6f}", f"{ncy:.6f}", f"{nw:.6f}", f"{nh:.6f}"]
        log.append(f"SET {c} @({tx:.3f},{ty:.3f}) -> ({ncx:.3f},{ncy:.3f},{nw:.3f},{nh:.3f})")
    for l in log:
        print("  ", l)
    if not args.apply:
        print("[干跑]")
        return 0
    BAK.mkdir(parents=True, exist_ok=True)
    (BAK / lp.name).write_text(lp.read_text(encoding="utf-8"), encoding="utf-8")
    lp.write_text("\n".join(" ".join(r) for r in rows) + "\n", encoding="utf-8")
    print("[落盘]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
