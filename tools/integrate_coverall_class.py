#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""integrate_coverall_class.py · 把 ppedetect 的 Coverall 数据接入训练集，新增 coverall 类

背景（E5.7 决策）：
    旧口径（Route A）把连体防护服（Coverall/medical-suit/safety-suit）视作满足 lab_coat。
    新口径（Route B'）要求二者分开：连体服不满足实验服要求，穿连体服者判「未穿实验服」。

现状核对结论（重要）：
    - 当前训练集 data/lab_ppe/labels 共 10936 框，其中 lab_coat 2463 框，
      **全部来自白大褂类数据，不含连体防护服**（无标签污染）。
    - data/raw/ppedetect 是一份「已下载但未接入」的现成数据集：
        Coverall 402 / Face_Shield 158 / Gloves 473 / Goggles 169 / Mask 394
      其图片与当前训练集**交集为 0**，可直接整体并入。
    - 因此新增 coverall 类的做法是「**增量接入**」而非「从 lab_coat 剥离」。

本脚本做两件事：
    1) 把 data/raw/ppedetect 的图片 + 标签（按类映射）复制/转换进 data/lab_ppe/
    2) 更新 data/lab_ppe/data.yaml：nc 5 -> 6，新增 5: coverall

类别映射（ppedetect 原始 -> 新体系）：
    Coverall(0)    -> 5 coverall
    Face_Shield(1) -> 丢弃
    Gloves(2)      -> 2 gloves
    Goggles(3)     -> 4 goggles
    Mask(4)        -> 1 mask

用法：
    python tools/integrate_coverall_class.py --dry-run          # 只报告
    python tools/integrate_coverall_class.py --apply            # 实际写入
    python tools/integrate_coverall_class.py --apply --split-ratio 0.85 0.15
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "ppedetect"
DST = ROOT / "data" / "lab_ppe"
DST_IMG = DST / "images"
DST_LBL = DST / "labels"
DATA_YAML = DST / "data.yaml"
OUT_REPORT = ROOT / "outputs" / "coverall_integrate_report.json"

# ppedetect 原始类别顺序（见 data/raw/ppedetect/data.yaml）
RAW_NAMES = ["Coverall", "Face_Shield", "Gloves", "Goggles", "Mask"]
# 原始类别 -> 新类别 id（None 表示丢弃）
RAW_TO_NEW = {
    0: 5,     # Coverall  -> coverall
    1: None,  # Face_Shield 丢弃
    2: 2,     # Gloves    -> gloves
    3: 4,     # Goggles   -> goggles
    4: 1,     # Mask      -> mask
}
NEW_NAMES = ["person", "mask", "gloves", "lab_coat", "goggles", "coverall"]


def find_split_dirs():
    """返回 [(split_name, images_dir, labels_dir), ...]

    兼容两种布局：
        A) <split>/images/*.jpg  +  <split>/labels/*.txt     （本仓库现状）
        B) <split>/*.jpg         +  <split>/*.txt
    """
    out = []
    for sp in ("train", "valid", "val", "test"):
        sp_dir = RAW / sp
        if not sp_dir.is_dir():
            continue
        # 布局 A
        if (sp_dir / "images").is_dir() and (sp_dir / "labels").is_dir():
            out.append((sp, sp_dir / "images", sp_dir / "labels"))
        # 布局 B
        elif list(sp_dir.glob("*.txt")):
            out.append((sp, sp_dir, sp_dir))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--val-ratio", type=float, default=0.12,
                    help="从 ppedetect 数据里划给 valid 的比例（默认 0.12）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    do_apply = args.apply and not args.dry_run

    print(f"[模式] {'APPLY 实际写入' if do_apply else 'DRY-RUN 只报告'}")
    print(f"[源] {RAW}")
    if not RAW.exists():
        print("[错误] ppedetect 原始数据不存在")
        return

    splits = find_split_dirs()
    print(f"[布局] 发现 {len(splits)} 个 split 目录：")
    for sp, img, lbl in splits:
        n_img = len(list(img.glob('*')))
        n_lbl = len(list(lbl.glob('*.txt')))
        print(f"    {sp:6s} images={n_img:4d} labels={n_lbl:4d}  ({img})")

    # 收集所有样本，按类统计
    samples = []          # (img_path, lbl_path)
    cls_counter = Counter()
    for sp, img_dir, lbl_dir in splits:
        for lp in lbl_dir.glob("*.txt"):
            # 找同名图片
            ip = None
            for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
                cand = img_dir / (lp.stem + ext)
                if cand.exists():
                    ip = cand
                    break
            if ip is None:
                continue
            classes = []
            for line in lp.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    c = int(parts[0])
                    classes.append(c)
                    cls_counter[RAW_NAMES[c] if c < len(RAW_NAMES) else str(c)] += 1
            if classes:
                samples.append((ip, lp, classes))

    print(f"\n[统计] 可用样本 {len(samples)} 张，原始框分布：")
    for k, v in cls_counter.most_common():
        tgt = RAW_TO_NEW.get(RAW_NAMES.index(k)) if k in RAW_NAMES else None
        print(f"    {k:12s} {v:4d}  -> {'丢弃' if tgt is None else NEW_NAMES[tgt]}")

    # 划分 train / valid
    rng = random.Random(args.seed)
    samples.sort(key=lambda x: str(x[0]))
    rng.shuffle(samples)
    n_val = int(len(samples) * args.val_ratio)
    val_set = set(str(s[0]) for s in samples[:n_val])

    report = {
        "mode": "apply" if do_apply else "dry-run",
        "val_ratio": args.val_ratio,
        "samples_total": len(samples),
        "raw_class_counts": dict(cls_counter),
        "written": Counter(),
        "dropped_files": [],   # 转换后无有效框的图
    }

    for ip, lp, _ in samples:
        target_split = "valid" if str(ip) in val_set else "train"
        new_lines = []
        for line in lp.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            c = int(parts[0])
            nc = RAW_TO_NEW.get(c)
            if nc is None:
                continue
            new_lines.append(f"{nc} {parts[1]} {parts[2]} {parts[3]} {parts[4]}")

        if not new_lines:
            report["dropped_files"].append(ip.name)
            continue

        # 目标文件名保持原名（已确认与现有训练集无重名冲突）
        dst_img = DST_IMG / target_split / ip.name
        dst_lbl = DST_LBL / target_split / (ip.stem + ".txt")

        if dst_img.exists() or dst_lbl.exists():
            report.setdefault("name_conflicts", []).append(ip.name)
            continue

        if do_apply:
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ip, dst_img)
            dst_lbl.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

        for l in new_lines:
            report["written"][NEW_NAMES[int(l.split()[0])]] += 1
        report["written"][f"__file_{target_split}"] += 1

    report["written"] = dict(report["written"])

    # 更新 data.yaml
    if do_apply:
        if "coverall" not in DATA_YAML.read_text(encoding="utf-8"):
            DATA_YAML.write_text(
                "path: " + str(DST).replace("\\", "/") + "\n"
                "train: images/train\n"
                "val: images/valid\n"
                "test: images/test\n"
                "nc: 6\n"
                "names:\n"
                "  0: person\n"
                "  1: mask\n"
                "  2: gloves\n"
                "  3: lab_coat\n"
                "  4: goggles\n"
                "  5: coverall\n",
                encoding="utf-8", newline="\n")
            report["data_yaml_updated"] = True
        print(f"\n[write] {DATA_YAML} -> nc=6 含 coverall")
    else:
        report["data_yaml_updated"] = False

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                          encoding="utf-8")

    print("\n===== 结果 =====")
    print(f"处理样本          : {report['samples_total']}")
    for k, v in sorted(report["written"].items()):
        print(f"  {k:16s}: {v}")
    print(f"因无有效框跳过     : {len(report['dropped_files'])}")
    print(f"重名冲突           : {len(report.get('name_conflicts', []))}")
    print(f"报告               : {OUT_REPORT.relative_to(ROOT)}")
    if not do_apply:
        print("\n[提示] DRY-RUN 未改动任何文件，确认后加 --apply")


if __name__ == "__main__":
    main()
