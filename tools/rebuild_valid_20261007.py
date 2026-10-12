"""E6 手术配套 · valid 重建补足（2026-10-07）

手术后保留池：train 1,698 图（1,189 家族）/ valid 122 图（122 家族），
train∩valid 家族交集 = 0（无泄漏）。valid 只有 122 张（6.7%），太少。

本脚本从 train 的家族里按**稳定哈希排序**挑「整家族」补进 valid，
目标 valid ≈ 全池 15%（同源整组划分，绝不拆家族，杜绝跨 split 泄漏）：
  1. 家族内所有成员必须同在 train（已是事实，交集为 0）
  2. 优先 size=1 的家族，其次 size=2（少动文件）
  3. md5(family) 排序保证可复现、与内容无关

回滚：把 data/_valid_rebuild_20261007_list.txt 里列的文件从 valid 移回 train
（本脚本 --restore 一键做）。

用法：
    .venv/Scripts/python.exe tools/rebuild_valid_20261007.py            # 执行
    .venv/Scripts/python.exe tools/rebuild_valid_20261007.py --restore  # 整体回滚
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "lab_ppe"
MAPFILE = ROOT / "data" / "_valid_rebuild_20261007_list.txt"
TARGET_RATIO = 0.15


def fam(name: str) -> str:
    stem = name.split(".rf.")[0]
    stem = re.sub(r"\.(jpg|jpeg|png)$", "", stem)
    stem = re.sub(r"-(\d{4,})$", "", stem)
    return stem


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--restore", action="store_true", help="按清单把补进 valid 的图整体移回 train")
    args = ap.parse_args()

    if args.restore:
        moved = 0
        for ln in MAPFILE.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            name = ln
            for src_dir, dst_dir in (("valid", "train"),):
                for sub in ("images", "labels"):
                    src = DATASET / sub / src_dir / (name if sub == "images" else Path(name).stem + ".txt")
                    dst = DATASET / sub / dst_dir / src.name
                    if src.exists():
                        os.replace(src, dst)
                        moved += 1
        print(f"回滚完成：移回 {moved} 个文件")
        return 0

    tr_files = os.listdir(DATASET / "images" / "train")
    va_files = os.listdir(DATASET / "images" / "valid")
    pool = len(tr_files) + len(va_files)
    target = round(pool * TARGET_RATIO)
    need = max(0, target - len(va_files))
    print(f"现状 train {len(tr_files)} / valid {len(va_files)}，池 {pool}，目标 valid≈{target}，需补 {need} 张")

    fams: dict[str, list[str]] = {}
    for f in tr_files:
        fams.setdefault(fam(f), []).append(f)

    # 优先 size=1，其次 size=2+；同 size 内按 md5 排序（稳定）
    order = sorted(fams.items(), key=lambda kv: (len(kv[1]), hashlib.md5(kv[0].encode()).hexdigest()))
    picked: list[str] = []
    n = 0
    for fname, members in order:
        if n >= need:
            break
        picked.extend(members)
        n += len(members)
    print(f"选中 {len(picked)} 张（{len({fam(p) for p in picked})} 个整家族）")

    MAPFILE.write_text(
        "# 2026-10-07 valid 重建：以下图片已从 train 移入 valid（--restore 可整体移回）\n"
        + "\n".join(sorted(picked)) + "\n",
        encoding="utf-8",
    )
    moved = failed = 0
    t0 = time.time()
    for k, name in enumerate(picked, 1):
        for sub, ext in (("images", name), ("labels", Path(name).stem + ".txt")):
            src = DATASET / sub / "train" / ext
            dst = DATASET / sub / "valid" / ext
            if not src.exists():
                continue
            try:
                os.replace(src, dst)
                moved += 1
            except OSError as e:
                failed += 1
                print(f"  [失败] {ext}: {e}")
        if k % 50 == 0:
            print(f"  ...{k}/{len(picked)} 组 | {moved} 文件 | {time.time()-t0:.0f}s", flush=True)
    print(f"\n完成：移动 {moved} 文件 | 失败 {failed} | 用时 {time.time()-t0:.1f}s")
    for s in ("train", "valid"):
        n_img = len(os.listdir(DATASET / "images" / s))
        print(f"  {s}: {n_img} 图")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
