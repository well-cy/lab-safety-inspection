#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""apply_webscene_fix_r3a.py —— r3 批次第一轮修框（2026-10-09，她回报 10 张意见）

她的意见：
  2 右边人物没框 | 3 护目镜没框 | 4 剔除(错误太多) | 5 手套框#5错 | 7 人物错+缺手套护目镜
  8 重画(人物错/护目镜错/实验服过大) | 11 人物错+缺护目镜 | 12 右侧人缺实验服手套
  13 人物错+缺口罩 | 14 人物错+缺护目镜

动作：DEL/SET 按类别+中心点最近匹配（TOL=0.07），ADD 直接追加。
#4 剔除 = 图+标签移动到 _rejected/web_lab_scene_r3_20261009/（永不删除）。

用法: python tools/apply_webscene_fix_r3a.py           # 干跑
      python tools/apply_webscene_fix_r3a.py --apply   # 落盘（先备份）
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
LBL = ROOT / "data" / "_prelabel" / "webscene_r3" / "labels"
IMG = ROOT / "data" / "raw" / "web_lab_scene" / "images_r3"
REJ = ROOT / "data" / "_rejected" / "web_lab_scene_r3_20261009"
BAK = ROOT / "data" / "_label_backup" / "webscene_r3a_20261009"
TOL = 0.07

# stem -> ops
# op: ("DEL", cls, cx, cy) | ("SET", cls, cx, cy, ncx,ncy,nw,nh) | ("ADD", cls, cx,cy,w,h)
OPS: dict[str, list[tuple]] = {
    "r3s_02_125836": [
        ("ADD", 0, 0.875, 0.395, 0.25, 0.71),      # 右侧人物整体（含手臂到手套）
    ],
    "r3s_03_130120": [
        ("ADD", 4, 0.4275, 0.155, 0.175, 0.12),    # 护目镜（放大实测）
    ],
    "r3s_05_130352": [
        ("DEL", 2, 0.2969, 0.6334),                # gloves#5 框在台面
    ],
    "r3s_07_130425": [
        ("DEL", 0, 0.4771, 0.4596),                # 两个半截 person 合并
        ("DEL", 0, 0.6972, 0.4986),
        ("ADD", 0, 0.635, 0.495, 0.73, 0.81),
        ("ADD", 4, 0.5235, 0.2905, 0.139, 0.119),  # 护目镜
        ("ADD", 2, 0.3055, 0.819, 0.0555, 0.092),  # 载物台上的蓝手套
        ("SET", 3, 0.7498, 0.4926, 0.64, 0.48, 0.72, 0.74),  # 大褂补左袖
    ],
    "r3s_08_130637": [
        ("SET", 0, 0.7759, 0.3379, 0.695, 0.4975, 0.61, 0.865),  # 男 person 拉到全身
        ("DEL", 0, 0.6963, 0.6007),                # 幽灵大 person
        ("SET", 3, 0.7582, 0.4574, 0.70, 0.565, 0.60, 0.73),     # 男大褂拉到全身
        ("SET", 3, 0.5415, 0.6149, 0.5435, 0.486, 0.2395, 0.176),# 女大褂收小
        ("SET", 4, 0.6844, 0.2157, 0.6635, 0.2175, 0.1115, 0.065),# 男眼镜归位
        ("DEL", 4, 0.4625, 0.3917),                # 女护目镜框错位（目视未见佩戴）
        ("ADD", 1, 0.522, 0.419, 0.074, 0.042),    # 女口罩补
    ],
    "r3s_11_130825": [
        ("DEL", 0, 0.3478, 0.3708),
        ("DEL", 0, 0.2166, 0.7582),
        ("DEL", 0, 0.4019, 0.6212),
        ("ADD", 0, 0.345, 0.50, 0.69, 0.94),       # 单人全身
        ("ADD", 4, 0.426, 0.3395, 0.277, 0.113),   # 大面罩式护目镜（放大实测）
        ("SET", 3, 0.3552, 0.5427, 0.355, 0.6295, 0.71, 0.671),  # 大褂拉到底
        ("SET", 1, 0.3733, 0.3922, 0.389, 0.428, 0.26, 0.126),   # 口罩收小
    ],
    "r3s_12_130901": [
        ("ADD", 3, 0.8265, 0.6955, 0.153, 0.199),  # 右侧人大褂
        ("ADD", 2, 0.68, 0.764, 0.052, 0.078),     # 右侧人手套（蓝块，待她复核）
    ],
    "r3s_13_130926": [
        ("DEL", 0, 0.7754, 0.4235),
        ("DEL", 0, 0.6164, 0.5820),
        ("ADD", 0, 0.77, 0.487, 0.46, 0.946),      # 单人全身
        ("ADD", 1, 0.725, 0.635, 0.162, 0.33),     # 白口罩
        ("ADD", 3, 0.787, 0.916, 0.426, 0.168),    # 底部大褂肩（顺手补，待确认）
    ],
    "r3s_14_130926".replace("130926", "130938"): [
        ("DEL", 0, 0.1683, 0.2199),                # 货架小鬼框
        ("SET", 0, 0.4299, 0.4606, 0.415, 0.465, 0.38, 0.87),   # 主人物
        ("SET", 0, 0.4154, 0.6043, 0.595, 0.37, 0.15, 0.31),    # 背景人物
        ("SET", 3, 0.4658, 0.6175, 0.4155, 0.608, 0.369, 0.754),# 主人物大褂
        ("ADD", 3, 0.597, 0.4125, 0.13, 0.213),    # 背景人物大褂
        ("ADD", 4, 0.4275, 0.16, 0.099, 0.07),     # 护目镜
    ],
}

REJECT_STEMS = ["r3s_04_130332"]  # 她拍板剔除


def load(stem: str):
    lp = LBL / f"{stem}.txt"
    if not lp.exists():
        return []
    return [l.split() for l in lp.read_text(encoding="utf-8").splitlines() if l.strip()]


def save(stem: str, rows):
    lp = LBL / f"{stem}.txt"
    lp.write_text("\n".join(" ".join(r) for r in rows) + ("\n" if rows else ""), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    total = {"DEL": 0, "SET": 0, "ADD": 0}
    miss = 0
    for stem, ops in OPS.items():
        rows = load(stem)
        n0 = len(rows)
        used = [False] * n0
        deleted = [False] * n0
        log = []
        for op in ops:
            if op[0] == "ADD":
                _, c, cx, cy, w, h = op
                rows.append([str(c), f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"])
                total["ADD"] += 1
                log.append(f"ADD {c} @({cx:.3f},{cy:.3f})")
                continue
            kind, c, tx, ty = op[0], op[1], op[2], op[3]
            best, bd = -1, 1e9
            for i in range(n0):
                r = rows[i]
                if used[i] or deleted[i] or int(r[0]) != c:
                    continue
                d = ((float(r[1]) - tx) ** 2 + (float(r[2]) - ty) ** 2) ** 0.5
                if d < bd:
                    bd, best = d, i
            if best < 0 or bd > TOL:
                miss += 1
                log.append(f"!! {kind} {c} @({tx:.3f},{ty:.3f}) 未匹配 (best {bd:.3f})")
                continue
            used[best] = True
            if kind == "DEL":
                deleted[best] = True
                log.append(f"DEL {c} @({tx:.3f},{ty:.3f})")
            else:
                _, _, _, _, ncx, ncy, nw, nh = op
                rows[best] = [str(c), f"{ncx:.6f}", f"{ncy:.6f}", f"{nw:.6f}", f"{nh:.6f}"]
                log.append(f"SET {c} @({tx:.3f},{ty:.3f}) -> ({ncx:.3f},{ncy:.3f},{nw:.3f},{nh:.3f})")
            total[kind] += 1
        rows = [r for i, r in enumerate(rows) if i >= len(deleted) or not deleted[i]]
        print(f"--- {stem} ---")
        for l in log:
            print("  ", l)
        if args.apply:
            BAK.mkdir(parents=True, exist_ok=True)
            lp = LBL / f"{stem}.txt"
            if lp.exists():
                (BAK / lp.name).write_text(lp.read_text(encoding="utf-8"), encoding="utf-8")
            save(stem, rows)

    # 剔除
    for stem in REJECT_STEMS:
        print(f"--- 剔除 {stem} ---")
        for base in (IMG, LBL):
            src = base / f"{stem}{'.jpg' if base == IMG else '.txt'}"
            if not src.exists():
                print(f"   跳过(不存在): {src.name}")
                continue
            dst = REJ / src.name
            if args.apply:
                REJ.mkdir(parents=True, exist_ok=True)
                os.replace(src, dst)
            print(f"   {src.name} -> _rejected/web_lab_scene_r3_20261009/")

    print(f"\n汇总: DEL {total['DEL']} / SET {total['SET']} / ADD {total['ADD']} / 未匹配 {miss}")
    if not args.apply:
        print("[干跑] 未写盘。确认后加 --apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
