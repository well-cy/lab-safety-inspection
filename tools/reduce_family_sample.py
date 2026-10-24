"""按「文件名前缀家族」等间隔抽样，削减高度冗余的同源抽帧数据集。

背景（2026-09-30 家族级标签审计）：
  项目 train 的 47%（5096 张）来自 `ppes_*` 家族（源 data/raw/ppes_kaxsi），
  经抽检确认全部是同一间仓库的同一台 CCTV 机位，属连续视频抽帧 → 高度冗余；
  且该家族只标注了 gloves/goggles 两类，图里的白大褂/口罩一概未标（=背景），
  是 lab_coat(12.8%)/mask(8.8%) 类别短板与"眼镜→护目镜"毒标签的最大来源。
  处理方式：按家族等间隔抽样，保留少量样本维持场景多样性，其余移入 _rejected（可整体移回）。

设计约束：
- **只动 train**，valid/test 一律不碰
- 只**移动**不删除，目标目录 data/_rejected/<tag>_<日期>/，可整体移回原位
- 用 os.replace（同盘 rename，不产生 unlink，规避沙箱 SAFE_DELETE_FAIL_CLOSED 杀进程）
- 幂等：源不存在即跳过，可中断重跑续做
- 默认干跑（--dry-run），不加 --execute 不写任何文件

用法：
    PY=.venv/Scripts/python.exe
    # 1) 干跑：看保留/剔除计划
    $PY tools/reduce_family_sample.py --prefix ppes_ --keep 1400 --dry-run
    # 2) 实际执行（移动图片+标签到 _rejected）
    $PY tools/reduce_family_sample.py --prefix ppes_ --keep 1400 --execute
    # 3) 反悔：把整个 tag 目录移回
    $PY tools/reduce_family_sample.py --prefix ppes_ --restore _rejected/ppes_reduce_20260930
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "lab_ppe"
REJECTED = ROOT / "data" / "_rejected"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def group_and_key(stem: str, prefix: str) -> tuple[str, int, str]:
    """返回 (批次组名, 数字序列, 原名)。

    组名 = 去掉家族前缀后的前导字母段（如 ppes_packing748_jpg.rf.xxx -> "packing"），
    同一批次的连续抽帧归为一组，才能做等间隔抽样。
    数字序列用于还原抽帧时序。
    """
    base = stem.split(".rf.")[0]                      # 去掉 roboflow 哈希后缀
    rest = base[len(prefix):] if base.startswith(prefix) else base
    m = re.match(r"^([A-Za-z]+)", rest)
    gname = m.group(1) if m else rest
    digits = "".join(re.findall(r"\d+", base))
    return (gname, int(digits) if digits else 0, stem)


def order_key(stem: str) -> tuple[str, int, str]:
    """兼容旧调用：不带 prefix 时按首个下划线前段分组。"""
    return group_and_key(stem, "")


def plan(prefix: str, keep: int, splits: list[str]):
    """返回 [(split, 图片路径, 标签路径, 是否保留)]，按批次等比例分配保留名额。"""
    kept, dropped, missing_lab = [], [], []
    for split in splits:
        img_dir = DATA / "images" / split
        lab_dir = DATA / "labels" / split
        if not img_dir.is_dir():
            continue
        files = [p for p in sorted(img_dir.iterdir())
                 if p.suffix.lower() in IMG_EXT and p.name.startswith(prefix)]
        if not files:
            continue
        groups: dict[str, list[Path]] = defaultdict(list)
        for p in files:
            groups[group_and_key(p.stem, prefix)[0]].append(p)
        total = len(files)
        for gname, gfiles in sorted(groups.items(), key=lambda x: -len(x[1])):
            gfiles.sort(key=lambda p: group_and_key(p.stem, prefix)[1])
            n_keep = max(1, round(len(gfiles) * keep / total))
            if len(gfiles) <= n_keep:
                picked = set(range(len(gfiles)))
            else:
                # 等间隔取索引，覆盖整段时序（首尾都取到）
                step = len(gfiles) / n_keep
                picked = {min(len(gfiles) - 1, int(i * step)) for i in range(n_keep)}
            for i, p in enumerate(gfiles):
                lab = lab_dir / f"{p.stem}.txt"
                if not lab.exists():
                    missing_lab.append(p)
                    continue
                (kept if i in picked else dropped).append((split, p, lab))
    return kept, dropped, missing_lab


def do_execute(dropped, tag: str, limit: int = 0) -> int:
    moved = skipped = failed = 0
    for split, img, lab in dropped:
        if limit and moved >= limit:
            print(f"\n达到 --limit {limit}，本轮停止（剩余 {len(dropped) - moved} 对，重跑续做）")
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
    print(f"\n执行完成：移动 {moved} | 跳过(源已不存在) {skipped} | 失败 {failed}")
    print(f"落盘位置: {REJECTED / tag}")
    return 0 if failed == 0 else 1


def do_restore(tag_dir: str) -> int:
    src_root = REJECTED / tag_dir if not Path(tag_dir).is_absolute() else Path(tag_dir)
    if not (src_root / "images").is_dir():
        print(f"目录不含 images/：{src_root}")
        return 1
    moved = skipped = 0
    for split_dir in (src_root / "images").iterdir():
        if not split_dir.is_dir():
            continue
        split = split_dir.name
        for img in sorted(split_dir.iterdir()):
            lab = src_root / "labels" / split / f"{img.stem}.txt"
            for src, dst_dir in ((img, DATA / "images" / split), (lab, DATA / "labels" / split)):
                if not src.exists():
                    skipped += 1
                    continue
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst = dst_dir / src.name
                if dst.exists():
                    dst = dst_dir / (src.stem + "__restored" + src.suffix)
                os.replace(src, dst)
                moved += 1
    print(f"已移回 {moved} 个文件（跳过 {skipped}）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="按文件名前缀家族等间隔抽样，削减冗余抽帧数据")
    ap.add_argument("--prefix", default="ppes_", help="家族前缀，默认 ppes_")
    ap.add_argument("--keep", type=int, default=1400, help="整个家族保留多少张，默认 1400")
    ap.add_argument("--splits", nargs="+", default=["train"], help="处理的 split，默认仅 train")
    ap.add_argument("--tag", default="", help="_rejected 下的子目录名，默认 <prefix>reduce_<日期>")
    ap.add_argument("--execute", action="store_true", help="真正移动文件（默认只干跑）")
    ap.add_argument("--dry-run", action="store_true", help="只统计不写文件（默认即为干跑，此参数可显式声明）")
    ap.add_argument("--list-out", default="", help="把剔除清单写到该文本文件")
    ap.add_argument("--keep-list-out", default="", help="把保留清单写到该文本文件")
    ap.add_argument("--restore", default=None, help="把某个 _rejected 子目录整体移回")
    ap.add_argument("--limit", type=int, default=0, help="本轮最多移动多少对文件（0=不限）。沙箱环境建议 ≤800，分批续跑（脚本幂等）")
    args = ap.parse_args()

    if args.restore:
        return do_restore(args.restore)

    keep = args.keep
    tag = args.tag or f"{args.prefix.strip('_')}_reduce_20260930"
    kept, dropped, missing = plan(args.prefix, keep, args.splits)

    total = len(kept) + len(dropped)
    if total == 0:
        print(f"没找到前缀为 {args.prefix} 的图片（split={args.splits}）")
        return 1

    print(f"家族 {args.prefix}* 共 {total} 张（split={args.splits}）")
    print(f"  计划保留 {len(kept)}（{len(kept)/total*100:.1f}%）| 计划移入 _rejected {len(dropped)}（{len(dropped)/total*100:.1f}%）")
    if missing:
        print(f"  警告：{len(missing)} 张缺标签文件，已跳过不动")

    print("\n按批次分组的保留/剔除：")
    by = defaultdict(lambda: [0, 0])
    for split, p, lab in kept:
        by[group_and_key(p.stem, args.prefix)[0]][0] += 1
    for split, p, lab in dropped:
        by[group_and_key(p.stem, args.prefix)[0]][1] += 1
    print(f"  {'批次':<14}{'保留':>7}{'剔除':>7}{'合计':>7}")
    for g, (k, d) in sorted(by.items(), key=lambda x: -(x[1][0] + x[1][1]))[:15]:
        print(f"  {g:<14}{k:>7}{d:>7}{k + d:>7}")
    if len(by) > 15:
        print(f"  ...（共 {len(by)} 个批次，其余略）")

    if args.keep_list_out:
        Path(args.keep_list_out).write_text(
            "\n".join(f"{s}/{p.name}" for s, p, _ in kept) + "\n", encoding="utf-8")
        print(f"\n保留清单 → {args.keep_list_out}")
    if args.list_out:
        Path(args.list_out).write_text(
            "\n".join(f"{s}/{p.name}" for s, p, _ in dropped) + "\n", encoding="utf-8")
        print(f"剔除清单 → {args.list_out}")

    if args.execute:
        print(f"\n开始移动 → data/_rejected/{tag}/")
        return do_execute(dropped, tag, args.limit)
    print(f"\n[dry-run] 未移动任何文件。实际执行加 --execute")
    print(f"落盘位置将是: data/_rejected/{tag}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
