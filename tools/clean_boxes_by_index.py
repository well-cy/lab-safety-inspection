#!/usr/bin/env python3
"""clean_boxes_by_index.py — 按复查编号删除指定标注框（毒标签清洗）

背景：
  人工在图版上按编号判定"这个框其实画在普通眼镜上"，需要把该框从标签里删掉。
  手工改 270 个 txt 既慢又易错，本工具把"编号 → 具体哪个框"自动对应起来。

口径（与协作约定一致）：
  - **删框不删图**：眼镜/误标物不是目标类，仅删该框；图与图内其它正确的框全部保留。
    删完该图在目标类上就是"无目标"样本 —— 对护目镜而言这正是我们要的**负样本**
    （治"眼镜→护目镜误判"）。
  - **永不丢数据**：改动前把原标签备份到 `data/_label_backup/<tag>/`，可 `--restore` 整体还原。
  - **默认干跑**：不给 --execute 只打印计划，不写任何文件。

用法：
  # 1) 干跑：--keep 列出"人工判定为正确"的编号，其余目标框全部删除
  python tools/clean_boxes_by_index.py --index outputs/_goggles_clean_batch01_zoom/index.txt \
      --class goggles --keep 5,7,9,10,12,13,16,23,28,29 --tag goggles_clean_20260930

  # 2) 确认后执行
  python tools/clean_boxes_by_index.py ... --execute

  # 3) 反悔：整体还原该 tag 下所有改动
  python tools/clean_boxes_by_index.py --restore goggles_clean_20260930

输出：
  data/_label_backup/<tag>/labels/<split>/<stem>.txt   原标签备份（首次改动时写）
  data/_label_backup/<tag>/clean_manifest.csv          改动流水（编号/图片/删除的框内容/时间）
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "lab_ppe"
BACKUP = ROOT / "data" / "_label_backup"
CLASS_ID = {"mask": 0, "gloves": 1, "lab_coat": 2, "goggles": 3}


def parse_index(path: Path):
    """index.txt → {编号: (rel_path, box_no)}；box_no 从 1 开始。"""
    out = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        parts = ln.split("\t")
        if len(parts) < 2:
            continue
        try:
            num = int(parts[0].strip())
        except ValueError:
            continue
        rel = parts[1].strip()
        box_no = 1
        if len(parts) > 2:
            tok = parts[2].strip()
            if "box" in tok:
                try:
                    box_no = int(tok.split("#")[-1])
                except ValueError:
                    box_no = 1
        out[num] = (rel, box_no)
    return out


def resolve_label(rel: str) -> Path:
    sp, name = rel.split("/", 1)
    return DATA / "labels" / sp / (Path(name).stem + ".txt")


def plan_changes(index: dict, nums_remove: list, cls_ids: set):
    """返回 [(编号, rel, box_no, 原行内容)]，按图片分组校验序号是否越界。"""
    changes, missing = [], []
    for n in nums_remove:
        if n not in index:
            missing.append(n)
            continue
        rel, box_no = index[n]
        lp = resolve_label(rel)
        if not lp.exists():
            missing.append(n)
            continue
        target_lines = [ln for ln in lp.read_text(encoding="utf-8").splitlines()
                        if ln.strip() and int(float(ln.split()[0])) in cls_ids]
        if box_no > len(target_lines):
            missing.append(n)
            continue
        changes.append((n, rel, box_no, target_lines[box_no - 1]))
    return changes, missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="", help="复查对照表（index.txt）")
    ap.add_argument("--class", dest="cls", required=False, default="goggles",
                    help="目标类别（本工具只删该类的框），如 goggles")
    ap.add_argument("--keep", default="", help="人工判定为【正确】的编号（逗号分隔）；其余目标框删除")
    ap.add_argument("--drop", default="", help="人工判定为【错误】的编号（逗号分隔），与 --keep 二选一")
    ap.add_argument("--tag", default="", help="备份目录名（默认 label_clean_<日期>）")
    ap.add_argument("--execute", action="store_true", help="真正写入（默认只干跑）")
    ap.add_argument("--restore", default="", help="还原某个 tag 下的全部标签改动")
    # 区域批量模式（无需人工逐编号判定，配合 cluster_boxes.py 的簇表使用）
    ap.add_argument("--region", default="",
                    help="区域批量清理：x1,y1,x2,y2（归一化，按框中心点判定）。"
                         "与 --prefix/--class 组合使用，删除该区域内全部目标框；无需 --index/--keep")
    ap.add_argument("--prefix", default="", help="区域模式下只处理文件名以此开头的图（如 ppes_）")
    ap.add_argument("--split", default="train", help="区域模式处理的 split（默认 train）")
    ap.add_argument("--exclude-list", default="",
                    help="区域模式下跳过的图片清单（每行 split/文件名 或 文件名），"
                         "如人工已判定为正确的那张")
    args = ap.parse_args()

    tag = args.tag or f"label_clean_{datetime.now():%Y%m%d}"

    # --- 还原模式 ---
    if args.restore:
        src = BACKUP / args.restore / "labels"
        if not src.is_dir():
            print(f"[错误] 备份目录不存在: {src}")
            return 1
        n = 0
        for p in src.rglob("*.txt"):
            sp = p.parent.name
            dst = DATA / "labels" / sp / p.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(p.read_bytes())
            n += 1
        print(f"[还原] {n} 个标签已还原回 data/lab_ppe/labels/")
        return 0

    cls_ids = {CLASS_ID.get(args.cls.strip().lower(), -1)} if not args.cls.strip().isdigit() \
        else {int(args.cls)}

    index, nums_remove = {}, []
    if args.region:
        # ---- 区域批量模式：无需 --index/--keep，直接按框中心点选框 ----
        try:
            x1, y1, x2, y2 = (float(v) for v in args.region.split(","))
        except ValueError:
            print(f"[错误] --region 格式应为 x1,y1,x2,y2: {args.region}")
            return 1
        excludes = set()
        if args.exclude_list:
            for ln in Path(args.exclude_list).read_text(encoding="utf-8").splitlines():
                ln = ln.split("\t")[0].strip()
                if ln and not ln.startswith("#"):
                    excludes.add(Path(ln.split("/")[-1]).stem)
        lab_dir = DATA / "labels" / args.split
        img_dir = DATA / "images" / args.split
        img_stems = {p.stem for p in img_dir.iterdir()} if img_dir.is_dir() else set()
        num = 1
        for p in sorted(lab_dir.glob("*.txt")):
            if args.prefix and not p.name.startswith(args.prefix):
                continue
            if p.stem not in img_stems or p.stem in excludes:
                continue
            seen = 0
            for ln in p.read_text(encoding="utf-8").splitlines():
                s = ln.split()
                if len(s) != 5 or int(float(s[0])) not in cls_ids:
                    continue
                seen += 1
                xc, yc = float(s[1]), float(s[2])
                if x1 <= xc <= x2 and y1 <= yc <= y2:
                    index[num] = (f"{args.split}/{p.stem}.jpg", seen)
                    num += 1
        nums_remove = sorted(index)
        print(f"[区域模式] region={args.region} prefix='{args.prefix or '*'}' split={args.split} "
              f"排除 {len(excludes)} 张 → 命中 {len(index)} 个目标框")
    else:
        if not args.index or not (args.keep or args.drop):
            print("[错误] 需要 --index 以及 --keep 或 --drop（或用 --region 区域模式）")
            return 1
        index = parse_index(Path(args.index))
        print(f"[对照表] {args.index} → {len(index)} 个编号")
        if args.keep:
            keep = {int(x) for x in args.keep.replace(" ", "").split(",") if x.strip()}
            nums_remove = sorted(set(index) - keep)
            print(f"[口径] 判定为正确(保留) {sorted(keep)} → 其余 {len(nums_remove)} 个目标框删除")
        else:
            nums_remove = sorted({int(x) for x in args.drop.replace(" ", "").split(",") if x.strip()})
            print(f"[口径] 指定删除 {nums_remove}")

    changes, missing = plan_changes(index, nums_remove, cls_ids)
    if missing:
        print(f"[警告] {len(missing)} 个编号无法定位（跳过）: {missing}")

    # 按图片汇总
    by_img: dict[str, list] = {}
    for n, rel, box_no, line in changes:
        by_img.setdefault(rel, []).append((n, box_no, line))
    print(f"\n将删除 {len(changes)} 个框，涉及 {len(by_img)} 张图：")
    for rel, items in sorted(by_img.items()):
        lp = resolve_label(rel)
        total = len([l for l in lp.read_text(encoding='utf-8').splitlines() if l.strip()])
        tgt = len([l for l in lp.read_text(encoding='utf-8').splitlines()
                   if l.strip() and int(float(l.split()[0])) in cls_ids])
        print(f"  {rel}")
        for n, box_no, line in items:
            print(f"    编号 {n:>3}  删 box#{box_no}  ({args.cls})  原行: {line.strip()}")
        print(f"    → 该图共 {total} 行标注，其中 {args.cls} {tgt} 个")

    if not args.execute:
        print(f"\n[干跑] 未写入任何文件。确认无误后加 --execute 执行（备份到 data/_label_backup/{tag}/）")
        return 0

    # --- 执行 ---
    bdir = BACKUP / tag / "labels"
    rows = []
    for rel, items in sorted(by_img.items()):
        lp = resolve_label(rel)
        sp = lp.parent.name
        bdir_sp = bdir / sp
        bdir_sp.mkdir(parents=True, exist_ok=True)
        if not (bdir_sp / lp.name).exists():      # 只备份首个版本，保证可还原到最初状态
            (bdir_sp / lp.name).write_bytes(lp.read_bytes())
        lines = lp.read_text(encoding="utf-8").splitlines()
        drop_idx = set()
        seen = 0
        for i, ln in enumerate(lines):
            if not ln.strip():
                continue
            if int(float(ln.split()[0])) in cls_ids:
                seen += 1
                for _, box_no, _ in items:
                    if seen == box_no:
                        drop_idx.add(i)
        new_lines = [ln for i, ln in enumerate(lines) if i not in drop_idx]
        lp.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")
        for n, box_no, line in items:
            rows.append([n, rel, box_no, args.cls, line.strip(),
                         f"{datetime.now():%Y-%m-%d %H:%M:%S}"])

    man = BACKUP / tag / "clean_manifest.csv"
    man.parent.mkdir(parents=True, exist_ok=True)
    with man.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["编号", "图片", "框序号", "类别", "删除的原标注行", "时间"])
        w.writerows(rows)

    print(f"\n[完成] 删除 {len(rows)} 个框 / {len(by_img)} 张图")
    print(f"  备份: {bdir}  (还原: --restore {tag})")
    print(f"  流水: {man}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
