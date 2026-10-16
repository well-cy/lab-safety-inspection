"""按「文件清单」把图片（连同标签）移入 _rejected，可整体移回。

用途：自动审计（audit_empty_labels.py）判定出的「疑似漏标毒样本」等，
需要按清单剔除时使用。区别于 reduce_family_sample.py（按前缀家族抽样），
本工具接受任意文件清单。

设计约束（与项目既有约定一致）：
- 只**移动**不删除；目标 data/_rejected/<tag>/，可整体移回
- 用 os.replace（同盘 rename，不产生 unlink，规避沙箱 SAFE_DELETE_FAIL_CLOSED 杀进程）
- 幂等：源不存在即跳过，可中断重跑续做
- 默认干跑，加 --apply 才写文件
- test 目录受保护：清单里若出现 test/ 会被拒绝

清单格式（每行）：
    train/xxx.jpg            ← split/文件名
    train/xxx.jpg\t说明       ← tab 或空格后的说明会被忽略
    # 注释行会被忽略

用法：
    PY=.venv/Scripts/python.exe
    # 1) 干跑
    $PY tools/reject_by_list.py --list data/_reject_poison.txt --tag empty_poison_20261001
    # 2) 执行（建议 --limit 400 分批，规避沙箱单次文件操作量红线）
    $PY tools/reject_by_list.py --list data/_reject_poison.txt --tag empty_poison_20261001 --apply --limit 400
    # 3) 反悔：整体移回
    $PY tools/reject_by_list.py --tag empty_poison_20261001 --restore
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REJECTED = ROOT / "data" / "_rejected"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_list(path: Path, dataset: Path):
    """返回 [(split, 图片Path, 标签Path, 原名)]，跳过不存在的。"""
    items, missing, bad = [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rel = line.split("\t")[0].strip().split("  ")[0].strip()
        rel = rel.replace("\\", "/").lstrip("./")
        if "/" not in rel:
            bad.append(rel)
            continue
        split, name = rel.split("/", 1)
        if split == "test":
            bad.append(rel)
            continue
        img = dataset / "images" / split / name
        if not img.exists() or img.suffix.lower() not in IMG_EXT:
            missing.append(rel)
            continue
        lab = dataset / "labels" / split / (img.stem + ".txt")
        items.append((split, img, lab, rel))
    return items, missing, bad


def do_execute(items, tag: str, limit: int = 0) -> int:
    moved = skipped = failed = 0
    for split, img, lab, rel in items:
        if limit and moved >= limit:
            print(f"\n达到 --limit {limit}，本轮停止（剩余 {len(items) - moved} 组，重跑续做）")
            break
        di = REJECTED / tag / "images" / split
        dl = REJECTED / tag / "labels" / split
        di.mkdir(parents=True, exist_ok=True)
        dl.mkdir(parents=True, exist_ok=True)
        for src, dst_dir in ((img, di), (lab, dl)):
            if not src.exists():
                skipped += 1
                continue
            dst = dst_dir / src.name
            try:
                if dst.exists():
                    dst = dst_dir / (src.stem + "__dup" + src.suffix)
                os.replace(src, dst)          # 同盘 rename，绝不 unlink
                moved += 1
            except OSError as e:
                failed += 1
                print(f"  [失败] {src.name}: {e}")
    print(f"\n执行完成：移动 {moved} 个文件 | 跳过(源已不存在) {skipped} | 失败 {failed}")
    print(f"落盘位置: {REJECTED / tag}")
    return 0 if failed == 0 else 1


def do_restore(tag: str, dataset: Path) -> int:
    base = REJECTED / tag
    if not base.is_dir():
        print(f"[错误] 不存在: {base}")
        return 1
    moved = failed = 0
    for split_dir in sorted((base / "images").glob("*")):
        split = split_dir.name
        for src in sorted(split_dir.iterdir()):
            if not src.is_file():
                continue
            dst = dataset / "images" / split / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.replace(src, dst)
                moved += 1
            except OSError as e:
                failed += 1
                print(f"  [失败] {src.name}: {e}")
    for split_dir in sorted((base / "labels").glob("*")):
        split = split_dir.name
        for src in sorted(split_dir.iterdir()):
            if not src.is_file():
                continue
            dst = dataset / "labels" / split / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.replace(src, dst)
                moved += 1
            except OSError as e:
                failed += 1
                print(f"  [失败] {src.name}: {e}")
    print(f"\n还原完成：移回 {moved} 个文件 | 失败 {failed}")
    return 0 if failed == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="按文件清单移入 _rejected（可整体移回）")
    ap.add_argument("--list", default="", help="清单文件（每行 split/文件名）")
    ap.add_argument("--dataset", default=str(ROOT / "data" / "lab_ppe"))
    ap.add_argument("--tag", required=True, help="_rejected 下的子目录名")
    ap.add_argument("--limit", type=int, default=0, help="本轮最多移动多少组（0=不限，建议 ≤400 分批）")
    ap.add_argument("--apply", action="store_true", help="真正移动（默认只干跑）")
    ap.add_argument("--restore", action="store_true", help="把该 tag 目录整体移回原位")
    args = ap.parse_args()

    dataset = Path(args.dataset).resolve()

    if args.restore:
        return do_restore(args.tag, dataset)

    if not args.list:
        print("[错误] 需要 --list（或用 --restore 还原）")
        return 1
    lp = Path(args.list)
    if not lp.exists():
        print(f"[错误] 清单不存在: {lp}")
        return 1

    items, missing, bad = parse_list(lp, dataset)
    print(f"[清单] {lp} → 有效 {len(items)} 组")
    if missing:
        print(f"[警告] {len(missing)} 项在数据集中找不到（已跳过）: {missing[:3]}")
    if bad:
        print(f"[警告] {len(bad)} 项格式异常或属 test（已拒绝）: {bad[:3]}")

    by_split: dict[str, int] = {}
    for split, *_ in items:
        by_split[split] = by_split.get(split, 0) + 1
    print("[分布] " + " | ".join(f"{k}:{v}" for k, v in sorted(by_split.items())))

    if not args.apply:
        print("\n[干跑] 未写入任何文件。确认无误后加 --apply 执行")
        for split, img, lab, rel in items[:5]:
            print(f"    {rel}  (标签{'有' if lab.exists() else '无'})")
        if len(items) > 5:
            print(f"    ... 其余 {len(items) - 5} 组")
        return 0

    return do_execute(items, args.tag, args.limit)


if __name__ == "__main__":
    sys.exit(main())
