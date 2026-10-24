# -*- coding: utf-8 -*-
"""apply_webscene_fix_r3c.py —— r3 批次第 2 轮修框（2026-10-09，她回报 21 张）

意见映射：
  15 缺 person/lab_coat        16 缺 goggles            17 缺 lab_coat + 删蓝椅误框 gloves
  20 人物重复框（合并为1）      21 mask错框删/person偏/缺gloves
  22 人物合并重框 + 缺 goggles/lab_coat/gloves(左手)      23 删2错gloves/修1/缺 goggles
  24 person 微扩 + 缺 lab_coat  25 缺 mask/goggles/lab_coat
  26 lab_coat 错位重设 + 缺 goggles/gloves                27 缺 person/lab_coat×2/goggles/gloves
  28 全部重画（删大错框 gloves，补 person/lab_coat/gloves×2/mask）
  30 缺 goggles                31 修 gloves + 删2袖口误框 + 缺 goggles
  33 gloves 收窄 + 缺 goggles/lab_coat×2                  35 删错person+错gloves、主 person 重框
  36 person 合并重框 + 缺 goggles                          37 返工：person/lab_coat 重设 + 补 person/goggles
  38 person#4 重框 + 缺 lab_coat（goggles=普通眼镜待拍板，不标）
  39 goggles=普通眼镜待拍板，不动
  40 删裸手误框 gloves + 重叠手套合并 + 补左手手套

格式：("DEL", stem, cls, cx, cy) / ("SET", stem, cls, cx, cy, ncx,ncy,nw,nh) / ("ADD", stem, cls, cx,cy,w,h)
匹配：同类别内按 (cx,cy) 欧氏最近，TOL=0.07。备份 -> data/_label_backup/webscene_r3c_fix_20261009
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
LBL = ROOT / "data" / "_prelabel" / "webscene_r3" / "labels"
BAK = ROOT / "data" / "_label_backup" / "webscene_r3c_fix_20261009"
TOL = 0.07

OPS = [
    # --- 15 缺 person / lab_coat ---
    ("ADD", "r3s_15_131040", 0, 0.800, 0.500, 0.400, 1.000),
    ("ADD", "r3s_15_131040", 3, 0.800, 0.775, 0.400, 0.450),
    # --- 16 缺 goggles（红色半包式安全眼镜）---
    ("ADD", "r3s_16_131602", 4, 0.845, 0.333, 0.170, 0.065),
    # --- 17 缺 lab_coat；删右下蓝椅误框 gloves ---
    ("ADD", "r3s_17_131736", 3, 0.660, 0.690, 0.460, 0.620),
    ("DEL", "r3s_17_131736", 2, 0.865921, 0.834540),
    # --- 20 两个 person 合并成一个 ---
    ("SET", "r3s_20_132201", 0, 0.732843, 0.269661, 0.659, 0.446, 0.683, 0.883),
    ("DEL", "r3s_20_132201", 0, 0.658134, 0.583832),
    # --- 21 mask 错框在头发上 -> 删；person 偏到显示器 -> 重框；补手套 ---
    ("DEL", "r3s_21_132637", 1, 0.199659, 0.233988),
    ("SET", "r3s_21_132637", 0, 0.699110, 0.656957, 0.465, 0.500, 0.930, 1.000),
    ("ADD", "r3s_21_132637", 2, 0.888, 0.490, 0.085, 0.220),
    # --- 22 person 合并重框 + 缺 goggles/lab_coat/gloves ---
    ("SET", "r3s_22_132715", 0, 0.320429, 0.470446, 0.485, 0.525, 0.770, 0.950),
    ("DEL", "r3s_22_132715", 0, 0.249593, 0.624203),
    ("ADD", "r3s_22_132715", 4, 0.455, 0.365, 0.250, 0.110),
    ("ADD", "r3s_22_132715", 3, 0.490, 0.720, 0.760, 0.560),
    ("ADD", "r3s_22_132715", 2, 0.350, 0.510, 0.160, 0.200),
    # --- 23 删 2 个错 gloves、修 1 个、缺 goggles ---
    ("DEL", "r3s_23_132757", 2, 0.233785, 0.203575),
    ("DEL", "r3s_23_132757", 2, 0.298066, 0.681975),
    ("SET", "r3s_23_132757", 2, 0.420603, 0.821392, 0.425, 0.815, 0.310, 0.370),
    ("ADD", "r3s_23_132757", 4, 0.795, 0.215, 0.310, 0.330),
    # --- 24 person 微扩 + 缺 lab_coat ---
    ("SET", "r3u_01_hauan-mo", 0, 0.438420, 0.676708, 0.460, 0.677, 0.260, 0.645),
    ("ADD", "r3u_01_hauan-mo", 3, 0.455, 0.775, 0.250, 0.450),
    # --- 25 缺 mask / goggles / lab_coat ---
    ("ADD", "r3u_02_naidoo-j", 1, 0.430, 0.305, 0.145, 0.115),
    ("ADD", "r3u_02_naidoo-j", 4, 0.450, 0.200, 0.145, 0.095),
    ("ADD", "r3u_02_naidoo-j", 3, 0.345, 0.550, 0.415, 0.650),
    # --- 26 lab_coat 错在显微镜上 -> 重设；缺 goggles/gloves ---
    ("SET", "r3u_03_chiang-8", 3, 0.269646, 0.753313, 0.780, 0.740, 0.440, 0.520),
    ("ADD", "r3u_03_chiang-8", 4, 0.725, 0.320, 0.270, 0.100),
    ("ADD", "r3u_03_chiang-8", 2, 0.395, 0.510, 0.130, 0.300),
    # --- 27 缺 person / lab_coat×2 / goggles / gloves ---
    ("ADD", "r3u_04_chiang-f", 0, 0.610, 0.520, 0.700, 0.960),
    ("ADD", "r3u_04_chiang-f", 3, 0.780, 0.610, 0.360, 0.780),
    ("ADD", "r3u_04_chiang-f", 3, 0.475, 0.740, 0.290, 0.240),
    ("ADD", "r3u_04_chiang-f", 4, 0.500, 0.275, 0.200, 0.170),
    ("ADD", "r3u_04_chiang-f", 2, 0.295, 0.860, 0.080, 0.140),
    # --- 28 全部重画 ---
    ("DEL", "r3u_05_syafi-al", 2, 0.737783, 0.441218),
    ("ADD", "r3u_05_syafi-al", 0, 0.740, 0.500, 0.520, 1.000),
    ("ADD", "r3u_05_syafi-al", 3, 0.750, 0.550, 0.500, 0.900),
    ("ADD", "r3u_05_syafi-al", 2, 0.575, 0.390, 0.130, 0.160),
    ("ADD", "r3u_05_syafi-al", 2, 0.520, 0.475, 0.105, 0.110),
    ("ADD", "r3u_05_syafi-al", 1, 0.745, 0.160, 0.080, 0.140),
    # --- 30 缺 goggles（蓝框安全眼镜）---
    ("ADD", "r3u_07_cancer-i", 4, 0.685, 0.290, 0.135, 0.070),
    # --- 31 修 gloves、删 2 个袖口误框、缺 goggles ---
    ("SET", "r3u_08_cancer-i", 2, 0.406901, 0.686757, 0.415, 0.685, 0.120, 0.090),
    ("DEL", "r3u_08_cancer-i", 2, 0.331466, 0.702282),
    ("DEL", "r3u_08_cancer-i", 2, 0.265982, 0.726176),
    ("ADD", "r3u_08_cancer-i", 4, 0.262, 0.327, 0.080, 0.050),
    # --- 33 gloves 过大收窄、缺 goggles/lab_coat×2 ---
    ("SET", "r3u_10_cancer-i", 2, 0.196701, 0.232647, 0.260, 0.308, 0.150, 0.325),
    ("ADD", "r3u_10_cancer-i", 4, 0.7175, 0.4025, 0.125, 0.075),
    ("ADD", "r3u_10_cancer-i", 3, 0.880, 0.800, 0.240, 0.400),
    ("ADD", "r3u_10_cancer-i", 3, 0.525, 0.585, 0.390, 0.330),
    # --- 35 删错 person / 错 gloves，主 person 重框 ---
    ("DEL", "r3u_12_cancer-i", 0, 0.619819, 0.513345),
    ("SET", "r3u_12_cancer-i", 0, 0.703456, 0.498086, 0.630, 0.525, 0.540, 0.850),
    ("DEL", "r3u_12_cancer-i", 2, 0.437141, 0.493142),
    # --- 36 person 合并重框 + 缺 goggles ---
    ("SET", "r3u_13_cancer-i", 0, 0.647576, 0.600178, 0.630, 0.570, 0.540, 0.860),
    ("DEL", "r3u_13_cancer-i", 0, 0.702580, 0.537167),
    ("ADD", "r3u_13_cancer-i", 4, 0.620, 0.280, 0.120, 0.060),
    # --- 37 返工：person/lab_coat 重设 + 补第二人 person/goggles ---
    ("SET", "r3u_14_institut", 0, 0.531839, 0.571946, 0.430, 0.550, 0.460, 0.900),
    ("ADD", "r3u_14_institut", 0, 0.665, 0.580, 0.350, 0.680),
    ("SET", "r3u_14_institut", 3, 0.381709, 0.681612, 0.410, 0.680, 0.420, 0.640),
    ("SET", "r3u_14_institut", 3, 0.522904, 0.670172, 0.665, 0.650, 0.330, 0.540),
    ("ADD", "r3u_14_institut", 4, 0.550, 0.3025, 0.070, 0.055),
    # --- 38 person#4 只框了头 -> 重框；缺 lab_coat；goggles=普通眼镜待拍板 ---
    ("SET", "r3u_15_institut", 0, 0.440327, 0.434066, 0.4825, 0.615, 0.275, 0.630),
    ("ADD", "r3u_15_institut", 3, 0.470, 0.585, 0.255, 0.330),
    # --- 40 删裸手误框、重叠手套合并、补左手手套 ---
    ("DEL", "r3u_17_institut", 2, 0.342399, 0.877330),
    ("SET", "r3u_17_institut", 2, 0.725883, 0.395535, 0.7575, 0.405, 0.085, 0.120),
    ("DEL", "r3u_17_institut", 2, 0.756466, 0.416840),
    ("ADD", "r3u_17_institut", 2, 0.6925, 0.530, 0.060, 0.090),
]


def load(stem: str):
    p = LBL / f"{stem}.txt"
    return [l.split() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def save(stem: str, rows) -> None:
    p = LBL / f"{stem}.txt"
    p.write_text("\n".join(" ".join(r) for r in rows) + "\n", encoding="utf-8")


def main() -> int:
    apply = "--apply" in sys.argv
    per_stem: dict[str, list] = {}
    for op in OPS:
        per_stem.setdefault(op[1], []).append(op)

    total = {"DEL": 0, "SET": 0, "ADD": 0, "MISS": 0}
    for stem, ops in per_stem.items():
        rows = load(stem)
        n0 = len(rows)
        used = [False] * n0          # 任何命中的动作都占用（防重复匹配）
        deleted = [False] * n0       # 仅 DEL 置位
        log = []
        for op in ops:
            if op[0] == "ADD":
                _, _, c, cx, cy, w, h = op
                rows.append([str(c), f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"])
                total["ADD"] += 1
                log.append(f"ADD {c} @({cx:.3f},{cy:.3f})")
                continue
            kind, _, c, tx, ty = op[0], op[1], op[2], op[3], op[4]
            best, bd = -1, 1e9
            for i in range(n0):
                if used[i] or deleted[i] or int(rows[i][0]) != c:
                    continue
                d = ((float(rows[i][1]) - tx) ** 2 + (float(rows[i][2]) - ty) ** 2) ** 0.5
                if d < bd:
                    bd, best = d, i
            if best < 0 or bd > TOL:
                total["MISS"] += 1
                log.append(f"MISS {kind} cls={c} @({tx:.3f},{ty:.3f})")
                continue
            used[best] = True
            if kind == "DEL":
                deleted[best] = True
                total["DEL"] += 1
                log.append(f"DEL #{best} cls={c} d={bd:.3f}")
            else:
                _, _, _, _, _, ncx, ncy, nw, nh = op
                rows[best] = [str(c), f"{ncx:.6f}", f"{ncy:.6f}", f"{nw:.6f}", f"{nh:.6f}"]
                total["SET"] += 1
                log.append(f"SET #{best} cls={c} d={bd:.3f} -> ({ncx:.3f},{ncy:.3f},{nw:.3f},{nh:.3f})")

        keep = [r for i, r in enumerate(rows) if i >= n0 or not deleted[i]]
        print(f"== {stem}: {n0} -> {len(keep)} 框")
        for l in log:
            print("   ", l)
        if apply:
            BAK.mkdir(parents=True, exist_ok=True)
            src = LBL / f"{stem}.txt"
            dst = BAK / src.name
            if not dst.exists() and src.exists():
                os.replace(src, BAK / (src.stem + "_pre.txt"))
            save(stem, keep)

    print(f"\n统计: DEL={total['DEL']} SET={total['SET']} ADD={total['ADD']} MISS={total['MISS']}")
    print("[干跑] 加 --apply 落盘" if not apply else "[已落盘]")
    return 1 if total["MISS"] else 0


if __name__ == "__main__":
    sys.exit(main())
