# -*- coding: utf-8 -*-
"""把她在 editor.html 里手工改好的标签文本落盘回 labels/。

输入格式（编辑器「复制全部改动标签」的输出）:
    ### stock_09_225921
    1 0.727500 0.532500 0.225000 0.127000
    ...

用法:
    .venv/Scripts/python.exe tools/import_manual_boxes.py --in outputs/_manual_in.txt --batch r2
    （先干跑看对账；确认后加 --apply）
"""
from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
CLS = {0: "person", 1: "mask", 2: "gloves", 3: "lab_coat", 4: "goggles"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--batch", default="r2")
    ap.add_argument("--stamp", default="manual")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    _b = f"_{args.batch}" if args.batch else ""
    LBL = ROOT / "data" / "_prelabel" / f"webscene{_b}" / "labels"
    BAK = ROOT / "data" / "_label_backup" / f"webscene{_b}_{args.stamp}_20261009"

    text = Path(args.src).read_text(encoding="utf-8")
    blocks, cur = {}, None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            cur = re.sub(r"^#+\s*", "", line).replace(".jpg", "").replace(".txt", "").strip()
            blocks[cur] = []
            continue
        if cur is None:
            continue
        p = line.split()
        if len(p) != 5:
            print(f"  !! 跳过硬行: {line}")
            continue
        c = int(p[0])
        vals = [float(v) for v in p[1:5]]
        if not (0 <= c <= 4):
            print(f"  !! 类别越界: {line}")
            continue
        if not all(0 <= v <= 1 for v in vals) or vals[2] <= 0 or vals[3] <= 0:
            print(f"  !! 坐标越界: {line}")
            continue
        blocks[cur].append((c, vals))

    if not blocks:
        print("没解析到任何 ### 段落，检查输入格式")
        return 1

    bad = 0
    for stem, rows in blocks.items():
        lp = LBL / f"{stem}.txt"
        if not lp.exists():
            print(f"!! {stem}.txt 不在 {LBL}（文件名对不上？）")
            bad += 1
            continue
        old = sum(1 for l in lp.read_text(encoding="utf-8").splitlines() if l.strip())
        per = {}
        for c, _ in rows:
            per[c] = per.get(c, 0) + 1
        print(f"{stem}: {old} -> {len(rows)} 框  "
              f"{ {CLS[k]: v for k, v in sorted(per.items())} }")
        if args.apply:
            BAK.mkdir(parents=True, exist_ok=True)
            if lp.exists():
                import shutil
                shutil.copyfile(lp, BAK / lp.name)
            with open(lp, "w", encoding="utf-8", newline="\n") as f:
                for c, v in rows:
                    f.write(f"{c} {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {v[3]:.6f}\n")

    if bad:
        print(f"\n{bad} 个文件没找到，未写盘")
        return 1
    print("\n" + ("已写盘（备份 -> " + str(BAK) + "）" if args.apply else "干跑通过，加 --apply 写盘"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
