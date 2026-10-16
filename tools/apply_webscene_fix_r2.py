# -*- coding: utf-8 -*-
"""stock 网图批次 r2（19 张）预标注修框——AI 精读动作表。

背景: prelabel_generic.py 双模型预标注 → AI 逐张精读网格图 + 放大裁剪核对 → 本动作表。
口径: coverall/连体防护服不算 lab_coat（她 2026-10-08 拍板）；普通处方眼镜不算 goggles，
     包裹式安全眼镜/护目镜都收；#4 男生深色眼镜按护目镜暂收（待她复核）。
#9 stock_09_225921 已裁掉底部 22 行播放条并重缩放标签（备份 webscene_r2_20261008）。

用法: .venv/Scripts/python.exe tools/apply_webscene_fix_r2.py [--apply]
动作坐标一律写 xyxy（脚本内部与标签 cxcywh 互转）。
"""
from __future__ import annotations

import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
LBL = ROOT / "data" / "_prelabel" / "webscene_r2" / "labels"
BAK = ROOT / "data" / "_label_backup" / "webscene_r2_fix_20261008"
TOL = 0.07
CLS = {0: "person", 1: "mask", 2: "gloves", 3: "lab_coat", 4: "goggles"}

# 编号 → 文件 stem（= stock_NN_HHMMSS）
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
    1: [  # 手套已对(gloves误色虚惊); 补 person/mask/goggles
        ("SET", 2, (0.288, 0.281, 0.712, 0.440), (2, (0.275, 0.280, 0.735, 0.475))),
        ("ADD", 0, (0.0, 0.0, 0.75, 0.50)),
        ("ADD", 4, (0.0, 0.08, 0.19, 0.21)),
        ("ADD", 1, (0.0, 0.195, 0.12, 0.315)),
    ],
    2: [  # 耳朵误标mask删; 底部goggles FP删; 三person并一; 大褂延到右缘底部
        ("SET", 0, (0.028, 0.116, 0.664, 0.721), (0, (0.0, 0.11, 1.0, 1.0))),
        ("SET", 3, (0.007, 0.398, 0.516, 0.870), (3, (0.0, 0.40, 0.62, 1.0))),
        ("DEL", 0, (0.004, 0.496, 0.521, 0.979)),
        ("DEL", 0, (0.394, 0.331, 0.998, 0.908)),
        ("DEL", 1, (0.082, 0.238, 0.170, 0.327)),
        ("DEL", 2, (0.482, 0.872, 0.973, 1.000)),
    ],
    3: [  # 女生脸上的误标mask实为护目镜(改类); 补女生person
        ("SETCLS", 1, (0.272, 0.405, 0.470, 0.555), 4, (0.272, 0.405, 0.470, 0.555)),
        ("ADD", 0, (0.0, 0.28, 0.31, 1.0)),
    ],
    4: [  # 补女生 person/goggles/gloves(放大核对)
        ("ADD", 0, (0.585, 0.255, 0.785, 0.62)),
        ("ADD", 4, (0.625, 0.265, 0.755, 0.365)),
        ("ADD", 2, (0.465, 0.475, 0.545, 0.565)),
    ],
    5: [  # 上手套框含进移液枪身, 收紧
        ("SET", 2, (0.134, 0.381, 0.513, 0.569), (2, (0.145, 0.385, 0.365, 0.51))),
    ],
    6: [  # coverall不算lab_coat删; 男生mask框修正; 补男生person/goggles/gloves
        ("DEL", 3, (0.500, 0.496, 1.000, 0.998)),
        ("SET", 1, (0.188, 0.377, 0.340, 0.719), (1, (0.27, 0.395, 0.46, 0.63))),
        ("ADD", 0, (0.25, 0.02, 0.47, 1.0)),
        ("ADD", 4, (0.215, 0.275, 0.40, 0.425)),
        ("ADD", 2, (0.26, 0.875, 0.45, 1.0)),
    ],
    7: [  # coverall删
        ("DEL", 3, (0.504, 0.406, 0.897, 0.992)),
    ],
    8: [  # person/lab_coat 一框两人拆分; 小person FP删
        ("SET", 3, (0.229, 0.439, 1.000, 0.782), (3, (0.55, 0.34, 1.0, 0.99))),
        ("SET", 0, (0.166, 0.292, 0.993, 0.898), (0, (0.53, 0.235, 1.0, 0.99))),
        ("ADD", 0, (0.245, 0.30, 0.655, 0.99)),
        ("ADD", 3, (0.265, 0.45, 0.635, 0.99)),
        ("DEL", 0, (0.279, 0.673, 0.391, 0.756)),
    ],
    9: [  # 已裁播放条; 手套框含放大镜柄收紧; 补女生大褂
        ("SET", 2, (0.0, 0.580, 0.414, 0.711), (2, (0.0, 0.58, 0.215, 0.711))),
        ("ADD", 3, (0.0, 0.422, 0.54, 0.711)),
    ],
    10: [  # coverall删; 女生裸手gloves FP删; 男生person收紧
        # 男生护目镜推额头/口罩拉下巴=未佩戴(口径:推头顶护目镜删), 不标;
        # 补女生规范佩戴的黄护目镜
        ("DEL", 3, (0.648, 0.337, 0.999, 0.987)),
        ("DEL", 2, (0.544, 0.544, 0.685, 0.736)),
        ("SET", 0, (0.060, 0.107, 0.602, 0.996), (0, (0.24, 0.10, 0.63, 1.0))),
        ("ADD", 4, (0.695, 0.13, 0.86, 0.225)),
    ],
    11: [  # 补左手套; 补大褂
        ("ADD", 2, (0.495, 0.575, 0.725, 0.705)),
        ("ADD", 3, (0.03, 0.33, 0.95, 0.66)),
    ],
    12: [  # 手套框缩到镜上那只; 补画面底缘第二只手套; 补护目镜
        ("SET", 2, (0.268, 0.179, 0.578, 1.000), (2, (0.28, 0.175, 0.575, 0.55))),
        ("ADD", 2, (0.51, 0.84, 0.63, 1.0)),
        ("ADD", 4, (0.545, 0.0, 0.905, 0.135)),
    ],
    13: [  # 9la框在仪器上删; 7la重复大褂删; person收成女生; 补男生person
        ("DEL", 3, (0.371, 0.022, 0.814, 0.965)),
        ("DEL", 3, (0.176, 0.282, 0.398, 0.887)),
        ("SET", 0, (0.080, 0.158, 0.391, 0.984), (0, (0.08, 0.16, 0.395, 0.99))),
        ("ADD", 0, (0.19, 0.06, 0.43, 0.67)),
    ],
    14: [],  # 10 框全对
    15: [  # person延到拿笔的手; 补黄色护目镜
        ("SET", 0, (0.008, 0.152, 0.870, 0.718), (0, (0.0, 0.15, 0.88, 0.95))),
        ("ADD", 4, (0.325, 0.355, 0.735, 0.465)),
    ],
    16: [  # 补两人护目镜(各自放大核对)
        ("ADD", 4, (0.275, 0.145, 0.46, 0.245)),
        ("ADD", 4, (0.625, 0.315, 0.795, 0.425)),
    ],
    17: [  # coverall删; 键盘区手套FP删
        ("DEL", 3, (0.045, 0.341, 0.657, 0.827)),
        ("DEL", 2, (0.834, 0.565, 0.994, 0.647)),
    ],
    18: [  # coverall×3删; 头部手套FP删; 重复person删; 女生person收紧; 补男生person
        ("DEL", 3, (0.000, 0.263, 0.510, 0.754)),
        ("DEL", 3, (0.358, 0.513, 0.853, 0.822)),
        ("DEL", 3, (0.000, 0.504, 0.644, 0.999)),
        ("DEL", 2, (0.233, 0.097, 0.642, 0.295)),
        ("DEL", 0, (0.342, 0.351, 0.715, 0.717)),
        ("SET", 0, (0.276, 0.351, 0.869, 0.869), (0, (0.44, 0.36, 0.88, 0.87))),
        ("ADD", 0, (0.0, 0.10, 0.53, 1.0)),
    ],
    19: [  # 7pe一框两人→收成男生; 补两人大褂
        ("SET", 0, (0.271, 0.088, 0.969, 0.700), (0, (0.55, 0.07, 1.0, 0.72))),
        ("ADD", 3, (0.13, 0.38, 0.62, 0.72)),
        ("ADD", 3, (0.55, 0.30, 1.0, 0.72)),
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
            shutil.copyfile(lp, BAK / f"{stem}.txt") if lp.exists() else None
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
