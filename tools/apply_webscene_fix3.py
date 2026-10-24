# -*- coding: utf-8 -*-
"""网图预标注修框第三轮 —— 她复核 fix2 后的回报落地（2026-10-08 深夜）。

她的回报（按编号）:
  1 还有一只手套漏 / 2 只有最右边实验服正确(7la骑在两人中间) / 4 下面的手套框错
  / 9 左女生护目镜框过大 / 33 没戴手套 / 35 后排女生戴口罩漏 / 46 下面的手套框过小

标签格式: YOLO `cls cx cy w h`；动作坐标写 xyxy，脚本内换算。
匹配目标一律抄自标签真实框 xyxy（中心距≈0）。

用法:
    python tools/apply_webscene_fix3.py            # 干跑
    python tools/apply_webscene_fix3.py --apply    # 备份后写盘
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
BACKUP = ROOT / "data" / "_label_backup" / "webscene_fix3_20261008"

CLS = {0: "person", 1: "mask", 2: "gloves", 3: "lab_coat", 4: "goggles"}
TOL = 0.07

FIXES = {
    1: [  # 滤芯_1: 中上部持金属工具的白手套漏
        ("ADD", 2, (0.475, 0.395, 0.545, 0.485)),
    ],
    2: [  # 青春药发光_1: 7la 骑在男生与站姿女生中间 → SET 回男生大褂; 站姿女生大褂漏 → ADD
        ("SET", 3, (0.4043, 0.2642, 0.5872, 0.8121), (3, (0.335, 0.220, 0.480, 0.635))),
        ("ADD", 3, (0.455, 0.340, 0.640, 0.880)),
    ],
    4: [  # 青春药发光_2: 5gl 框住台面蓝色布/杂物, 非手套 → 删
        ("DEL", 2, (0.4720, 0.9042, 0.5304, 0.9796)),
    ],
    9: [  # STEM_6: 10go 框过大(罩半张脸) → 收紧到护目镜本体
        ("SET", 4, (0.3400, 0.3500, 0.4400, 0.4100), (4, (0.398, 0.330, 0.450, 0.370))),
    ],
    33: [  # 青农大_3: 4gl 误标(没戴手套) → 删
        ("DEL", 2, (0.4700, 0.6600, 0.5800, 0.7800)),
    ],
    35: [  # 青农大_4: 后排低头女生白口罩漏 (坐标经白色阈值分割实测)
        ("ADD", 1, (0.520, 0.093, 0.585, 0.160)),
    ],
    46: [  # qPCR_2: 1gl 只框手掌一小块 → 扩大到指尖+腕部
        ("SET", 2, (0.0500, 0.4400, 0.3000, 0.6200), (2, (0.045, 0.355, 0.460, 0.790))),
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
