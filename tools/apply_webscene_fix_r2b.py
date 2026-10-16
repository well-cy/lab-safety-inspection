# -*- coding: utf-8 -*-
"""stock 网图批次 r2 修框第二轮——她抽查 verify.html 回报 7 条（2026-10-09）。

回报: 4女生实验服漏 / 9男生护目镜漏 / 10男生护目镜+口罩漏(她拍板:戴着就要标,
     推翻 AI 上轮"未佩戴不标"判断) / 12手套框错位 / 13男生实验服漏 /
     14口罩漏(两人都戴) / 16男生实验服框偏右。

用法: .venv/Scripts/python.exe tools/apply_webscene_fix_r2b.py [--apply]
动作坐标一律写 xyxy（脚本内部与标签 cxcywh 互转）。
"""
from __future__ import annotations

import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
LBL = ROOT / "data" / "_prelabel" / "webscene_r2" / "labels"
BAK = ROOT / "data" / "_label_backup" / "webscene_r2b_fix_20261009"
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
    4: [  # 女生白大褂漏(被前排男生部分遮挡, 按可见范围补)
        ("ADD", 3, (0.48, 0.27, 0.78, 0.97)),
    ],
    9: [  # 男生护目镜漏; 6ma口罩框偏高套在护目镜上→下移到真实口罩
        ("ADD", 4, (0.52, 0.365, 0.935, 0.50)),
        ("SET", 1, (0.6408, 0.3732, 0.8992, 0.4897), (1, (0.60, 0.465, 0.905, 0.625))),
    ],
    10: [  # 她拍板: 男生护目镜戴在眼上+白口罩都要标(推翻上轮"未佩戴不标")
        ("ADD", 4, (0.405, 0.255, 0.555, 0.385)),
        ("ADD", 1, (0.425, 0.295, 0.545, 0.485)),
    ],
    12: [  # 0gl收全拇指与右缘; 4gl错位在白大褂上→左移到腕部蓝手套
        ("SET", 2, (0.28, 0.175, 0.575, 0.55), (2, (0.285, 0.13, 0.625, 0.63))),
        ("SET", 2, (0.51, 0.84, 0.63, 1.0), (2, (0.33, 0.63, 0.56, 1.0))),
    ],
    13: [  # 男生白大褂漏
        ("ADD", 3, (0.235, 0.255, 0.415, 0.675)),
    ],
    14: [  # 两人白口罩都漏
        ("ADD", 1, (0.665, 0.085, 0.775, 0.175)),
        ("ADD", 1, (0.815, 0.145, 0.90, 0.225)),
    ],
    16: [  # 3la偏右→左扩到衣领与躯干
        ("SET", 3, (0.7166, 0.5521, 0.9990, 0.9996), (3, (0.615, 0.52, 1.0, 1.0))),
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
