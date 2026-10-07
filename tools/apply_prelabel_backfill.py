#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把人工确认过的预标注草稿合并进 data/lab_ppe（带备份、可回滚、幂等）。

背景：空标签图审计后挑出的「有目标但模型漏标」好图，经 prelabel_draft.py 预标注、
人工逐条判定修正后，用本工具写入数据集正式标签。

用法::

    # 干跑（默认，只打印计划）
    python tools/apply_prelabel_backfill.py --draft data/_prelabel/backfill_20261001/labels \
        --items outputs/_prelabel_confirm/items.txt --tag backfill_20261007

    # 执行；同时把编号 3 剔除（移入 _rejected，不删除）
    python tools/apply_prelabel_backfill.py ... --apply --reject 3

处置规则
--------
- 草稿有框       → 覆盖数据集标签（原标签备份到 data/_label_backup/<tag>/）
- 草稿为空标签   → 保持空标签（负样本），不做改动
- --reject 编号  → 图片 + 标签一起 **移动** 到 data/_rejected/<tag>_rejected/（永不删除）
- 幂等：目标已与草稿一致则跳过；源缺失则跳过
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "data" / "lab_ppe"


def norm(text: str) -> str:
    """规范化标签文本，便于比对（去空行、统一结尾换行）。"""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines) + ("\n" if lines else "")


def main() -> None:
    ap = argparse.ArgumentParser(description="合并人工确认过的预标注草稿进 data/lab_ppe")
    ap.add_argument("--draft", default="data/_prelabel/backfill_20261001/labels",
                    help="草稿标签根目录（其下 <split>/*.txt）")
    ap.add_argument("--items", default="outputs/_prelabel_confirm/items.txt",
                    help="清单：每行 <split>/<文件名>，行号即编号（从 1 开始）")
    ap.add_argument("--tag", default="backfill", help="备份/剔除目录标记名")
    ap.add_argument("--reject", default="", help="要剔除的编号，逗号分隔（如 3 或 3,7）")
    ap.add_argument("--apply", action="store_true", help="真正执行（默认只干跑）")
    args = ap.parse_args()

    draft_dir = ROOT / args.draft
    items = [ln.strip() for ln in (ROOT / args.items).read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.startswith("#")]
    reject_ids = {int(x) for x in args.reject.replace("，", ",").split(",") if x.strip()}

    bk_root = ROOT / "data" / "_label_backup" / args.tag
    rj_root = ROOT / "data" / "_rejected" / f"{args.tag}_rejected"

    n_write = n_same = n_empty = n_reject = 0
    for i, rel in enumerate(items, start=1):
        split, name = rel.split("/")
        stem = Path(name).stem
        src = draft_dir / split / f"{stem}.txt"
        dst = DS / "labels" / split / f"{stem}.txt"
        img = DS / "images" / split / name

        if i in reject_ids:
            n_reject += 1
            print(f"[剔除] #{i} {split}/{name}")
            if args.apply:
                rj_img = rj_root / split / name
                rj_lbl = rj_root / split / f"{stem}.txt"
                rj_img.parent.mkdir(parents=True, exist_ok=True)
                if img.exists():
                    os.replace(img, rj_img)          # 沙箱下必须用 os.replace
                if dst.exists():
                    os.replace(dst, rj_lbl)
                (rj_root / "剔除清单.txt").open("a", encoding="utf-8").write(
                    f"#{i}\t{split}/{name}\t人工判定：非实验室域\n")
            continue

        if not src.exists():
            print(f"[跳过] #{i} 草稿缺失 {src}")
            continue
        want = norm(src.read_text(encoding="utf-8"))
        cur = norm(dst.read_text(encoding="utf-8")) if dst.exists() else ""
        if want == cur:
            n_same += 1
            print(f"[一致] #{i} {split}/{stem[:34]}  ({len(want.splitlines())} 框)")
            continue
        if not want:
            n_empty += 1
            print(f"[空标签] #{i} {split}/{stem[:34]}  → 负样本，不改动")
            continue
        n_write += 1
        print(f"[写入] #{i} {split}/{stem[:34]}  {len(cur.splitlines())} → {len(want.splitlines())} 框")
        if args.apply:
            bk = bk_root / split
            bk.mkdir(parents=True, exist_ok=True)
            if dst.exists() and not (bk / dst.name).exists():
                shutil.copy2(dst, bk / dst.name)
            dst.write_text(want, encoding="utf-8")

    print(f"\n{'已执行' if args.apply else '干跑'}：写入 {n_write} / 一致 {n_same} / "
          f"空标签保留 {n_empty} / 剔除 {n_reject}")
    if args.apply:
        print(f"备份目录: {bk_root.relative_to(ROOT)}")
        if n_reject:
            print(f"剔除目录: {rj_root.relative_to(ROOT)}（可整体移回）")


if __name__ == "__main__":
    main()
