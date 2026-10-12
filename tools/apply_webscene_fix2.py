# -*- coding: utf-8 -*-
"""网图预标注修框第二轮 —— 她抽查 verify.html 后的回报落地（2026-10-08 深夜）。

她的回报（按编号）:
  1 手套漏 / 3 左男生大褂漏 / 9 左女生护目镜漏 / 11 手套框歪 / 12 实验服没框出
  / 17 一只手套漏 / 18 手套框错 / 19 两只手套漏 / 21+22 没戴护目镜(22 戴口罩)
  / 23 戴口罩没框 / 33 是眼镜不是护目镜 / 34 没戴口罩 / 35 少框一人
  / 36 没戴护目镜 / 38 人物没框 / 39 手套框歪 / 41 剔除 / 42 人物没框 / 45 人物漏

标签格式: YOLO `cls cx cy w h`；动作坐标写 xyxy，脚本内换算。
匹配目标一律抄自标签真实框 xyxy（中心距≈0）。
#41 剔图不在本脚本（单独 move 到 _rejected + 更新 index.txt）。

用法:
    python tools/apply_webscene_fix2.py            # 干跑
    python tools/apply_webscene_fix2.py --apply    # 备份后写盘
"""
import argparse
import io
import shutil
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
LBL_DIR = ROOT / "data" / "_prelabel" / "webscene_r1" / "labels"
INDEX = ROOT / "outputs" / "web_lab_scene_review" / "index.txt"
BACKUP = ROOT / "data" / "_label_backup" / "webscene_fix2_20261008"

CLS = {0: "person", 1: "mask", 2: "gloves", 3: "lab_coat", 4: "goggles"}
TOL = 0.07

# ---------------------------------------------------------------------------
# 动作表: 编号 -> [(op, ...)]   (op 格式同 fix1)
# ---------------------------------------------------------------------------
FIXES = {
    1: [  # 滤芯_1: 上方持玻璃盖的白手套漏
        ("ADD", 2, (0.375, 0.035, 0.465, 0.200)),
    ],
    3: [  # 滤芯_2: 左男生(1ma)大褂漏
        ("ADD", 3, (0.205, 0.405, 0.355, 0.610)),
    ],
    9: [  # STEM_6: 左侧红发女生护目镜漏
        ("ADD", 4, (0.340, 0.350, 0.440, 0.410)),
    ],
    11: [  # 避风港_1: 3gl 框落在金属盒上 → 删, 补绿手套真实位置
        ("DEL", 2, (0.700, 0.400, 0.800, 0.530)),
        ("ADD", 2, (0.750, 0.375, 0.935, 0.550)),
    ],
    12: [  # 做实验_1: 2la 与 3pe 完全重合导致大褂"看不见" → 上边下移到肩、右缘含袖
        ("SET", 3, (0.0000, 0.0167, 0.4144, 0.7500), (3, (0.000, 0.075, 0.440, 0.755))),
    ],
    17: [  # 壹束_1: 近景右手(带镯)手套漏
        ("ADD", 2, (0.565, 0.598, 0.715, 0.685)),
    ],
    18: [  # 壹束_2: 2gl 框在纸上(错) → 删, 补真实白手套
        ("DEL", 2, (0.415, 0.605, 0.515, 0.675)),
        ("ADD", 2, (0.560, 0.668, 0.730, 0.725)),
    ],
    19: [  # 壹束_3: 上下两panel各漏一只手套
        ("ADD", 2, (0.505, 0.348, 0.592, 0.398)),   # 上: 调焦手
        ("ADD", 2, (0.465, 0.778, 0.535, 0.835)),   # 下: 载物台手
    ],
    21: [  # 朱师兄_1: 0go 实为普通眼镜(推发区) → 删
        ("DEL", 4, (0.245, 0.291, 0.360, 0.344)),
    ],
    22: [  # 朱师兄_2: 4go 是眼镜不是护目镜 → 删; 侧脸口罩漏 → 补
        ("DEL", 4, (0.000, 0.240, 0.060, 0.380)),
        ("ADD", 1, (0.000, 0.355, 0.045, 0.450)),
    ],
    23: [  # 朱师兄_3: 侧脸口罩漏
        ("ADD", 1, (0.265, 0.495, 0.385, 0.575)),
    ],
    33: [  # 青农大_3: 4go 是普通眼镜 → 删
        ("DEL", 4, (0.490, 0.360, 0.600, 0.470)),
    ],
    34: [  # 青农大_3b: 4ma 误标(没戴口罩) → 删
        ("DEL", 1, (0.450, 0.380, 0.550, 0.480)),
    ],
    35: [  # 青农大_4: 左弯腰女生(0ma/6la)缺 person
        ("ADD", 0, (0.190, 0.080, 0.460, 0.760)),
    ],
    36: [  # 青农大_4b: 5go 是普通眼镜 → 删
        ("DEL", 4, (0.500, 0.240, 0.680, 0.340)),
    ],
    38: [  # 青农大_6: 右侧操作者(0la)缺 person
        ("ADD", 0, (0.680, 0.550, 1.000, 1.000)),
    ],
    39: [  # 傲娇咪_1: 0gl 左移含指尖; 7gl 框在袖子上 → 删, 补顶部真实手套
        ("SET", 2, (0.5750, 0.1610, 0.7730, 0.3200), (2, (0.545, 0.158, 0.772, 0.322))),
        ("DEL", 2, (0.780, 0.040, 0.940, 0.160)),
        ("ADD", 2, (0.725, 0.005, 0.835, 0.078)),
    ],
    42: [  # 傲娇咪_4: 只有手+前臂入镜, 补 person
        ("ADD", 0, (0.600, 0.460, 1.000, 1.000)),
    ],
    45: [  # qPCR_1: 双手套+大褂同属一人, 补 person
        ("ADD", 0, (0.120, 0.050, 1.000, 1.000)),
    ],
}


def load_index():
    out = {}
    for line in INDEX.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("编号"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            out[int(parts[0])] = Path(parts[1]).stem
    return out


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    idx = load_index()
    stats = {"DEL": 0, "SET": 0, "ADD": 0}
    problems = []
    totals = {c: 0 for c in CLS}

    for num in sorted(FIXES):
        stem = idx.get(num)
        if stem is None:
            problems.append(f"#{num}: index 无此编号")
            continue
        lbl = LBL_DIR / f"{stem}.txt"
        if not lbl.exists():
            problems.append(f"#{num}: 缺标签 {lbl.name}")
            continue

        rows = []
        for ln in lbl.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            p = ln.split()
            rows.append([int(p[0]), tuple(float(v) for v in p[1:5])])
        n0 = len(rows)
        used = [False] * len(rows)
        deleted = [False] * len(rows)
        adds, log = [], []

        for op in FIXES[num]:
            kind = op[0]
            if kind == "DEL":
                _, cls, old = op
                i = match_box(rows, used, cls, old)
                if i is None:
                    problems.append(f"#{num}: DEL 未匹配 cls={cls} {old}")
                    continue
                c, b = rows[i]
                log.append(f"  DEL  [{i}] {CLS[c]} ({b[0]:.3f},{b[1]:.3f},{b[2]:.3f},{b[3]:.3f})")
                used[i] = True
                deleted[i] = True
                stats["DEL"] += 1
            elif kind == "SET":
                _, cls, old, (nc, nb) = op
                i = match_box(rows, used, cls, old)
                if i is None:
                    problems.append(f"#{num}: SET 未匹配 cls={cls} {old}")
                    continue
                c, b = rows[i]
                log.append(f"  SET  [{i}] {CLS[c]} -> {CLS[nc]} {tuple(round(v, 2) for v in xyxy2cxcywh(nb))}")
                rows[i] = [nc, xyxy2cxcywh(nb)]
                used[i] = True
                stats["SET"] += 1
            elif kind == "ADD":
                _, cls, nb = op
                adds.append((cls, xyxy2cxcywh(nb)))
                log.append(f"  ADD  {CLS[cls]} {tuple(round(v, 2) for v in xyxy2cxcywh(nb))}")
                stats["ADD"] += 1

        final = [tuple(r) for i, r in enumerate(rows) if not deleted[i]] + adds
        for c, b in final:
            assert c in CLS, f"#{num} 坏类别 {c}"
            assert all(-0.001 <= v <= 1.001 for v in b), f"#{num} 越界 {b}"
            assert b[2] > 0 and b[3] > 0, f"#{num} 零尺寸 {b}"
            totals[c] += 1

        print(f"#{num} {stem[:36]}  框 {n0} -> {len(final)}")
        for line in log:
            print(line)

        if args.apply:
            BACKUP.mkdir(parents=True, exist_ok=True)
            bak = BACKUP / f"{stem}.txt"
            if not bak.exists():
                shutil.copyfile(lbl, bak)
            lbl.write_text("".join(
                f"{c} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n" for c, b in final),
                encoding="utf-8")

    print(f"\n统计: {stats}")
    print("修后这些图按类别:", {CLS[c]: n for c, n in totals.items()})
    if problems:
        print("\n!! 问题:")
        for p in problems:
            print(" ", p)
        sys.exit(1)
    print("\n干跑全部匹配" if not args.apply else f"\n已写盘，备份 {BACKUP}")


if __name__ == "__main__":
    main()
