# -*- coding: utf-8 -*-
"""stock 网图批次 r2 修框第三轮——她复核 fix_r2b 回报 3 条（2026-10-09）。

回报: 9男生口罩框选过大 / 12手套框还是不对 / 13右侧人实验服框偏右。
处理:
  9  mask#6 过大偏右 → HSV 目测收窄到真实口罩 (0.615,0.468,0.84,0.595)
  12 两只手套框 → HSV 蓝色阈值分割实测（outputs/_r2c_debug/seg12.py）:
     0gl 右缘 0.625→0.549(收掉镜筒空白) 上缘 0.13→0.176
     4gl 中心右移, 分割 x0.3395~0.5967 y0.620~0.9985(贴画面底缘)
  13 lab_coat#9 整体左移: x 0.235~0.415 → 0.21~0.40

用法: .venv/Scripts/python.exe tools/apply_webscene_fix_r2c.py [--apply]
动作坐标一律写 xyxy（脚本内部与标签 cxcywh 互转）。
"""
from __future__ import annotations

import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
LBL = ROOT / "data" / "_prelabel" / "webscene_r2" / "labels"
BAK = ROOT / "data" / "_label_backup" / "webscene_r2c_fix_20261009"
TOL = 0.07
CLS = {0: "person", 1: "mask", 2: "gloves", 3: "lab_coat", 4: "goggles"}

NAMES = {
    1: "stock_01_224903", 2: "stock_02_225611", 3: "stock_03_225630",
    4: "stock_04_225724", 5: "stock_05_225758", 6: "stock_06_225819",
    7: "stock_07_225832", 8: "stock_08_225855", 9: "stock_09_225921",
    10: "stock_10_225949", 11: "stock_11_230024", 12: "stock_12_230101",
    13: "stock_13_230120", 14: "stock_14_230146", 15: "stock_15_230221",
    16: "stock_16_230239", 17: "stock_17_230425", 18: "stock_18_230455",
    19: "stock_19_230636",
}

FIXES = {
    9: [  # 男生口罩框过大偏右 → 收窄到真实口罩
        ("SET", 1, (0.60, 0.465, 0.905, 0.625), (1, (0.615, 0.468, 0.84, 0.595))),
    ],
    12: [  # 两只手套框 HSV 分割实测收拢
        ("SET", 2, (0.285, 0.13, 0.625, 0.63), (2, (0.2964, 0.1762, 0.5485, 0.6192))),
        ("SET", 2, (0.33, 0.63, 0.56, 1.0), (2, (0.3395, 0.6200, 0.5967, 0.9985))),
    ],
    13: [  # 右侧人实验服框偏右 → 整体左移
        ("SET", 3, (0.235, 0.255, 0.415, 0.675), (3, (0.21, 0.255, 0.40, 0.675))),
    ],
}


def xyxy2cxcywh(b):
    x0, y0, x1, y1 = b
    return ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)


def cdist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def match_box(rows, used, cls, target_xyxy):
    tc = xyxy2cxcywh(target_xyxy)
    best, best_d = None, 1e9
    for i, (c, box) in enumerate(rows):
        if used[i] or (cls is not None and c != cls):
            continue
        d = cdist((box[0], box[1]), (tc[0], tc[1]))
        if d < best_d:
            best, best_d = i, d
    return best if (best is not None and best_d < TOL) else None


def main() -> int:
    apply = "--apply" in sys.argv
    stats = {"DEL": 0, "SET": 0, "SETCLS": 0, "ADD": 0}
    unmatched = []
    per_after = {}
    if apply:
        BAK.mkdir(parents=True, exist_ok=True)

    for n in sorted(FIXES):
        stem = NAMES[n]
        lp = LBL / f"{stem}.txt"
        rows = []
        if lp.exists():
            for ln in lp.read_text(encoding="utf-8").splitlines():
                ln = ln.strip()
                if ln:
                    p = ln.split()
                    rows.append([int(p[0]), tuple(float(v) for v in p[1:5])])
        n0 = len(rows)
        used = [False] * len(rows)
        deleted = [False] * len(rows)
        adds = []
        log = []

        for act in FIXES[n]:
            op = act[0]
            if op == "DEL":
                _, cls, old = act
                i = match_box(rows, used, cls, old)
                if i is None:
                    unmatched.append((n, act)); continue
                used[i] = True
                deleted[i] = True
                stats["DEL"] += 1
                log.append(f"  DEL [{i}] {CLS[rows[i][0]]}")
            elif op == "SET":
                _, cls, old, (nc, nb) = act
                i = match_box(rows, used, cls, old)
                if i is None:
                    unmatched.append((n, act)); continue
                rows[i] = [nc, xyxy2cxcywh(nb)]
                used[i] = True
                stats["SET"] += 1
                log.append(f"  SET [{i}] -> {CLS[nc]}")
            elif op == "SETCLS":
                _, cls, old, nc, nb = act
                i = match_box(rows, used, cls, old)
                if i is None:
                    unmatched.append((n, act)); continue
                rows[i] = [nc, xyxy2cxcywh(nb)]
                used[i] = True
                stats["SETCLS"] += 1
                log.append(f"  SETCLS [{i}] -> {CLS[nc]}")
            elif op == "ADD":
                _, cls, nb = act
                adds.append([cls, xyxy2cxcywh(nb)])
                stats["ADD"] += 1
                log.append(f"  ADD {CLS[cls]}")

        final = [tuple(r) for i, r in enumerate(rows) if not deleted[i]] + [tuple(a) for a in adds]
        for c, _ in final:
            per_after[c] = per_after.get(c, 0) + 1
        print(f"#{n} {stem}: {n0} -> {len(final)} 框")
        for l in log:
            print(l)

        if apply:
            if lp.exists():
                shutil.copyfile(lp, BAK / f"{stem}.txt")
            with open(lp, "w", encoding="utf-8", newline="\n") as f:
                for c, b in final:
                    f.write(f"{c} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n")

    print(f"\n统计: {stats}")
    print("修后按类别:", {CLS[k]: v for k, v in sorted(per_after.items())})
    if unmatched:
        print(f"\n!! {len(unmatched)} 条未匹配:")
        for n, act in unmatched:
            print(f"  #{n} {act}")
        return 1
    print("全部匹配 ✅" + ("  已写盘" if apply else "  (干跑)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
