#!/usr/bin/env python3
"""merge_glove_datasets.py — 将两个公开手套数据集入库到 data/lab_ppe（可干跑、可回滚）

来源与策略（均为 CC BY 4.0，审计报告见 data/*_dedup_report.txt）：

1. safety_gloves (data/raw/safety_gloves, 3373 张)
   类别: Gloves=0, NO-Gloves=1
   映射: Gloves → gloves(1)；NO-Gloves 框丢弃，图保留为背景负样本（治裸手误检）

2. ppes_kaxsi (data/raw/ppes_kaxsi, 11978 张，审计后 11973)
   类别: glove=0, goggles=1, helmet=2, mask=3, no_glove=4, no_goggles=5,
         no_helmet=6, no_mask=7, no_shoes=8, shoes=9
   映射: glove→gloves(1), goggles→goggles(3), mask→mask(0)；
         no_*/helmet/shoes 框丢弃（背景）
   说明: 同图内的 goggles/mask 框一并映射保留——若只保留 glove 框而丢掉
         goggles/mask 框，画面中真实的护目镜/口罩会变成"无标注正样本"，
         反而教会模型在这些位置输出背景（压制 mask/goggles 类），危害更大。
         ppes 审计剔除的 5 张近重复（data/ppes_kaxsi_leak_images.txt）不入库。

通用规则:
   * 文件名加来源前缀 sgv_ / ppes_，split 对应搬运（train→train 等）
   * 只搬运图片与转换后的标注，原始数据留在 data/raw/ 不动
   * 默认干跑；--apply 才复制文件；重复运行自动跳过已存在文件（幂等）
   * 支持源前缀过滤 --only {sgv,ppes}
"""
from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / "data" / "lab_ppe"
SGV = ROOT / "data" / "raw" / "safety_gloves"
PPES = ROOT / "data" / "raw" / "ppes_kaxsi"
LEAK = ROOT / "data" / "ppes_kaxsi_leak_images.txt"

IMG_EXT = {".jpg", ".jpeg", ".png"}

# 源类别 id → 项目类别 id；None = 丢弃该框（图保留）
MAP_SGV = {0: 1, 1: None}                                   # Gloves, NO-Gloves
MAP_PPES = {0: 1, 1: 3, 3: 0, 2: None, 4: None, 5: None,
            6: None, 7: None, 8: None, 9: None}             # glove,goggles,helmet,mask,no_*...


def load_leak() -> set[str]:
    if not LEAK.exists():
        return set()
    out = set()
    for ln in LEAK.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            out.add(ln.split()[0])
    return out


def build_plan(src: Path, mapping: dict[int, int | None], prefix: str,
               exclude: set[str]) -> tuple[list, Counter, Counter, int, dict[int, str]]:
    """返回 (任务列表, 源框统计, 入库后框统计, 负样本图数, 源类别名)"""
    # 读源 data.yaml 类别名（用于打印）
    src_names: dict[int, str] = {}
    yml = src / "data.yaml"
    if yml.exists():
        try:
            import yaml
            src_names = {i: str(n) for i, n in
                         enumerate(yaml.safe_load(yml.read_text(encoding="utf-8")).get("names", []))}
        except Exception:
            pass
    jobs: list[tuple[Path, Path, Path, str]] = []   # (img_src, img_dst, lbl_src, lbl_dst内容)
    src_boxes: Counter = Counter()
    dst_boxes: Counter = Counter()
    n_neg = 0
    for sp in ("train", "valid", "test"):
        idir, ldir = src / sp / "images", src / sp / "labels"
        if not idir.is_dir():
            continue
        for ip in sorted(idir.iterdir()):
            if ip.suffix.lower() not in IMG_EXT:
                continue
            rel = f"{sp}/{ip.name}"
            if rel in exclude:
                continue
            lp = ldir / (ip.stem + ".txt")
            lines_out: list[str] = []
            if lp.exists():
                for ln in lp.read_text(encoding="utf-8").splitlines():
                    p = ln.split()
                    if len(p) != 5:
                        continue
                    cid = int(float(p[0]))
                    src_boxes[cid] += 1
                    tgt = mapping.get(cid)
                    if tgt is None:
                        continue
                    dst_boxes[tgt] += 1
                    lines_out.append(" ".join([str(tgt)] + p[1:]))
            if not lines_out:
                n_neg += 1          # 负样本：无保留框，写空标注
            dst_name = prefix + ip.name
            jobs.append((ip, DST / "images" / sp / dst_name, lp,
                         "\n".join(lines_out) + ("\n" if lines_out else "")))
    return jobs, src_boxes, dst_boxes, n_neg, src_names


def fmt(c: Counter, names: dict[int, str] | None = None) -> str:
    names = names or {0: "mask", 1: "gloves", 2: "lab_coat", 3: "goggles"}
    return "  ".join(f"{names.get(k, k)}:{v}" for k, v in sorted(c.items())) or "(无)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--only", choices=["sgv", "ppes"], help="只处理其中一个来源")
    ap.add_argument("--apply", action="store_true", help="真正复制文件（默认干跑）")
    args = ap.parse_args()

    leak = load_leak()
    plans = []
    if args.only in (None, "sgv"):
        plans.append(("sgv  safety_gloves", SGV, MAP_SGV, "sgv_", set()))
    if args.only in (None, "ppes"):
        plans.append(("ppes ppes_kaxsi", PPES, MAP_PPES, "ppes_", leak))

    grand = {"imgs": 0, "neg": 0, "skip": 0}
    for title, src, mapping, prefix, exclude in plans:
        jobs, sb, db, n_neg, src_names = build_plan(src, mapping, prefix, exclude)
        dup = sum(1 for _, d, _, _ in jobs if d.exists())
        print(f"\n===== {title} =====")
        print(f"图片: {len(jobs)} 张入库"
              + (f"（另 {len(exclude)} 张审计剔除不入库）" if exclude else ""))
        print(f"  其中负样本（空标注背景图）: {n_neg} 张")
        print(f"  目标文件名冲突（已存在，将跳过）: {dup}")
        print(f"  源框统计:   {fmt(sb, src_names)}")
        print(f"  入库框统计: {fmt(db)}")
        drop = {k: v for k, v in sb.items() if k not in mapping or mapping[k] is None}
        dn = {src_names.get(k, k): v for k, v in drop.items()}
        if drop:
            print(f"  丢弃框（留作背景）: " +
                  "  ".join(f"{k}:{v}" for k, v in sorted(dn.items())))
        grand["imgs"] += len(jobs) - dup
        grand["neg"] += n_neg
        grand["skip"] += dup

    # 项目现状
    cur_img = sum(len(list((DST / "images" / sp).iterdir())) for sp in ("train", "valid", "test"))
    cur_box: Counter = Counter()
    for sp in ("train", "valid", "test"):
        for lp in (DST / "labels" / sp).glob("*.txt"):
            for ln in lp.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    cur_box[int(float(ln.split()[0]))] += 1

    # 预计入库后
    add_box: Counter = Counter()
    for title, src, mapping, prefix, exclude in plans:
        _, _, db, _, _ = build_plan(src, mapping, prefix, exclude)
        add_box += db

    print("\n===== 汇总 =====")
    print(f"拟入库图片: {grand['imgs']} 张（负样本 {grand['neg']}）  冲突跳过: {grand['skip']}")
    print(f"项目现状:   {cur_img} 张  框 {fmt(cur_box)}")
    fut = cur_box + add_box
    print(f"入库后预计: {cur_img + grand['imgs']} 张  框 {fmt(fut)}")

    if not args.apply:
        print("\n[干跑] 未复制任何文件。确认后加 --apply 执行。")
        return 0

    moved = 0
    for title, src, mapping, prefix, exclude in plans:
        jobs, _, _, _, _ = build_plan(src, mapping, prefix, exclude)
        for isrc, idst, lsrc, content in jobs:
            ldst = DST / "labels" / idst.parent.name / (idst.stem + ".txt")
            if idst.exists() and ldst.exists():
                continue
            if not idst.exists():
                shutil.copy2(isrc, idst)
            ldst.write_text(content, encoding="utf-8")
            moved += 1
    print(f"\n[完成] 复制 {moved} 张图 + 标注。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
