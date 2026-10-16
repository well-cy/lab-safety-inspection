"""E6 来源域手术 · 批量移出执行器（2026-10-07）

背景：tools/reject_by_list.py 单个调用内移动超过 ~1000 个文件时，
本机沙箱会以 SIGTERM 终止进程（历史红线见项目备忘）。本脚本把工作
切成小块（默认每块 200 文件）并在块间短暂退让，一次跑完所有剩余分组。

设计约束（与项目既有约定一致）：
- 只移动不删除：os.replace（同盘 rename，规避 unlink 被劫持）
- 幂等：源不存在即跳过，可随时中断重跑续做
- test 目录硬保护
- 清单用 *.all.txt（全量），不要用 *.list（48 张抽样）

用法：
    .venv/Scripts/python.exe tools/apply_surgery_20261007.py            # 执行
    .venv/Scripts/python.exe tools/apply_surgery_20261007.py --dry-run  # 干跑
    .venv/Scripts/python.exe tools/apply_surgery_20261007.py --group cctv  # 只跑一组

回滚：tools/reject_by_list.py --tag <tag> --restore
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "lab_ppe"
REJECTED = ROOT / "data" / "_rejected"
SHEETS = ROOT / "outputs" / "phase2_provenance"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# (短名, 清单文件, _rejected 子目录)
GROUPS = [
    ("cctv",   "X-工业CCTVppes_star.all.txt",      "domain_audit_20261007_cctv"),
    ("ds1a",   "X-手套商品图dataset1-A.all.txt",    "domain_audit_20261007_ds1a"),
    ("ds1b",   "X-护目镜网图dataset1-B.all.txt",    "domain_audit_20261007_ds1b"),
    ("misc",   "X-未归类_其他.all.txt",             "domain_audit_20261007_misc"),
    ("glasses", "X-眼镜网图外部.all.txt",           "domain_audit_20261007_glasses"),
]


def load(items_file: Path):
    out = []
    for line in items_file.read_text(encoding="utf-8").splitlines():
        rel = line.strip().split("\t")[0].replace("\\", "/").lstrip("./")
        if not rel or rel.startswith("#") or "/" not in rel:
            continue
        out.append(rel)
    return out


def move_chunk(rels, tag: str, apply: bool):
    moved = skipped = failed = 0
    for rel in rels:
        split, name = rel.split("/", 1)
        if split == "test":
            print(f"  [拒绝] 命中 test 红线: {rel}")
            failed += 1
            continue
        img = DATASET / "images" / split / name
        lab = DATASET / "labels" / split / (Path(name).stem + ".txt")
        di = REJECTED / tag / "images" / split
        dl = REJECTED / tag / "labels" / split
        if apply:
            di.mkdir(parents=True, exist_ok=True)
            dl.mkdir(parents=True, exist_ok=True)
        for src, dst_dir in ((img, di), (lab, dl)):
            if not src.exists():
                skipped += 1
                continue
            if not apply:
                moved += 1
                continue
            dst = dst_dir / src.name
            if dst.exists():
                dst = dst_dir / (src.stem + "__dup" + src.suffix)
            try:
                os.replace(src, dst)
                moved += 1
            except OSError as e:
                failed += 1
                print(f"  [失败] {src.name}: {e}")
    return moved, skipped, failed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="", help="只跑指定短名（cctv/ds1a/ds1b/misc/glasses）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--chunk", type=int, default=200, help="每块处理多少组（默认 200≈400 文件）")
    ap.add_argument("--max-files", type=int, default=0, help="单次调用最多移动多少文件（0=不限，建议分段防中断）")
    args = ap.parse_args()

    for short, listname, tag in GROUPS:
        if args.group and args.group != short:
            continue
        items_file = SHEETS / listname
        if not items_file.exists():
            print(f"[跳过] 清单不存在: {items_file}")
            continue
        rels = load(items_file)
        total_moved = total_skip = total_fail = 0
        t0 = time.time()
        for start in range(0, len(rels), args.chunk):
            moved, skipped, failed = move_chunk(rels[start:start + args.chunk], tag, not args.dry_run)
            total_moved += moved
            total_skip += skipped
            total_fail += failed
            if args.dry_run:
                continue
            print(f"  [{short}] {start + args.chunk}/{len(rels)} 组 | 累计移动 {total_moved} 文件 | "
                  f"{time.time()-t0:.0f}s", flush=True)
            if args.max_files and total_moved >= args.max_files:
                print(f"  [{short}] 达到 --max-files {args.max_files}，本轮停止（重跑续做，幂等）", flush=True)
                break
            time.sleep(0.05)
        print(f"[{short:8s}] 清单 {len(rels):5d} 组 → 移动 {total_moved:5d} 文件 | "
              f"跳过 {total_skip:5d} | 失败 {total_fail} | 用时 {time.time()-t0:.1f}s | tag={tag}")
    print("\n完成。校验：")
    remaining = 0
    for short, listname, tag in GROUPS:
        if args.group and args.group != short:
            continue
        left = sum(1 for rel in load(SHEETS / listname) if (DATASET / "images" / rel).exists())
        remaining += left
        print(f"  {short:8s} 清单内仍在数据集中的文件: {left}")
    print(f"REMAINING={remaining}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
