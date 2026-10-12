#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""reject_labcoat_imagenetcv.py —— 按筛查清单把可疑图从 raw 池**移动**到 data/_rejected/

项目铁律：剔除一律**移动**、**永不删除**（`os.replace` / `shutil.move`，
绝不用 os.unlink —— 本机沙箱把 unlink 劫持为回收站、中止时 SAFE_DELETE_FAIL_CLOSED 杀进程）。

用法:
    $PY tools/reject_labcoat_imagenetcv.py --list outputs/labcoat_imagenetcv_screen/suspect.txt --dry
    $PY tools/reject_labcoat_imagenetcv.py --list outputs/labcoat_imagenetcv_screen/suspect.txt --apply
"""
from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "raw" / "labcoat_imagenetcv" / "data"
DEST = ROOT / "data" / "_rejected" / "labcoat_imagenetcv_screen_20261011"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="筛查清单（每行：编号\\t相对路径\\tflags）")
    ap.add_argument("--dry", action="store_true", help="只报告，不移动")
    ap.add_argument("--apply", action="store_true", help="执行移动")
    args = ap.parse_args()
    if not (args.dry or args.apply):
        ap.error("需要 --dry 或 --apply")

    rows = []
    for ln in Path(args.list).read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split("\t")
        if len(parts) < 3:
            continue
        rows.append((parts[1].strip(), parts[2].strip()))

    print(f"[{'干跑' if args.dry else '执行'}] 清单 {args.list}  条目 {len(rows)}")

    tagcnt = Counter(t for _, t in rows)
    for t, n in tagcnt.most_common():
        print(f"  {t:16s} {n}")

    missing, moved, exist = [], [], []
    for rel, tag in rows:
        src = SRC / rel
        if not src.exists():
            missing.append(rel)
            continue
        dst = DEST / rel
        if dst.exists():
            exist.append(rel)
            continue
        if args.apply:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))   # 跨目录移动：先试 rename，失败回退 copy+删
            # 复核：源已消失、目标存在
            if not src.exists() and dst.exists():
                moved.append(rel)
            else:
                print(f"  ⚠ 移动未确认: {rel}")
        else:
            moved.append(rel)

    print(f"\n  将移动/已移动 {len(moved)}")
    if exist:
        print(f"  目标已存在（上次已移）{len(exist)}")
    if missing:
        print(f"  源缺失 {len(missing)}：{missing[:5]}")
    if args.dry:
        print(f"\n  目标目录：{DEST}")
        print("  加 --apply 执行")
        return 0

    # 移动后核对：源目录剩余、目标目录计数
    remain = sum(1 for _ in SRC.rglob("*") if _.is_file())
    destn = sum(1 for _ in DEST.rglob("*") if _.is_file())
    print(f"\n  源目录剩余 {remain} 张 / 目标目录 {destn} 张")
    return 0


if __name__ == "__main__":
    sys.exit(main())
