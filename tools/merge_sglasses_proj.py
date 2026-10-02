#!/usr/bin/env python3
"""merge_sglasses_proj.py — 把 sglasses_proj 精简后的 safety_goggles 图入库 data/lab_ppe

来源：data/raw/sglasses_proj（CC BY 4.0，Roboflow example-workspace-ofjvo/safety-goggles-detection-project v2）
类别：0=glasses, 1=no_glasses, 2=safety_goggles
策略（用户 2026-10-02 拍板）：
  * 只取含 safety_goggles 框的图；映射 2 → goggles(3)
  * glasses / no_glasses 框一律丢弃（不在项目类别空间内，丢弃即"背景"，正是要教模型别喊眼镜）
  * 精简：原图家族（.rf. 前）每家族留 1 张；再对 goggle-* 源系列限 30 张（seed=7 确定抽样）
    最终 48 张
  * split 路由：源 train → train，源 valid → valid，源 test → train（红线：新增数据不进 test）

用法：
    python tools/merge_sglasses_proj.py            # 干跑
    python tools/merge_sglasses_proj.py --apply    # 执行
"""
from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "raw" / "sglasses_proj"
DST = ROOT / "data" / "lab_ppe"
CAND = ROOT / "data" / "sglasses_proj_candidates.txt"
PREFIX = "sgl_"
SEED = 7
SERIES_CAP = 30
GOOGLES_CLS = 3


def series_of(stem: str) -> str:
    s = stem.split(".rf.")[0]
    s = re.sub(r"[_-]+(png|jpg|jpeg)$", "", s, flags=re.I)
    s = re.sub(r"[_-]*\d+$", "", s)
    return s or "other"


def build_keep() -> list[str]:
    cand = [ln.strip() for ln in CAND.read_text(encoding="utf-8").splitlines() if ln.strip()]
    fam: dict[str, list[str]] = defaultdict(list)
    for c in cand:
        fam[c.split("/")[1].split(".rf.")[0]].append(c)
    keep = [sorted(v, key=lambda c: (c.split("/")[0] != "train", len(c)))[0]
            for v in fam.values()]
    by: dict[str, list[str]] = defaultdict(list)
    for k in keep:
        by[series_of(k.split("/")[1].split(".rf.")[0])].append(k)
    random.seed(SEED)
    out: list[str] = []
    for _, lst in by.items():
        out += random.sample(sorted(lst), min(SERIES_CAP, len(lst)))
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真正复制（默认干跑）")
    args = ap.parse_args()

    keep = build_keep()
    jobs = []
    dropped_boxes = Counter()
    for rel in keep:
        sp, stem = rel.split("/")          # stem 已去掉扩展名（含 .rf. 哈希）
        img_name = stem + ".jpg"
        src_lp = SRC / sp / "labels" / (stem + ".txt")
        lines = [l.split() for l in src_lp.read_text(encoding="utf-8").splitlines() if l.strip()]
        out_lines = []
        for p in lines:
            cid = int(float(p[0]))
            if cid == 2:
                out_lines.append(" ".join([str(GOOGLES_CLS)] + p[1:]))
            else:
                dropped_boxes[cid] += 1
        dst_sp = sp if sp in ("train", "valid") else "train"
        dst_stem = PREFIX + stem
        jobs.append((sp, img_name, dst_sp, dst_stem, out_lines))

    n_box = sum(len(j[4]) for j in jobs)
    print(f"拟入库: {len(jobs)} 张，goggles 框 {n_box} 个")
    print(f"  按目标 split: {dict(Counter(j[2] for j in jobs))}")
    print(f"  丢弃框: " + "  ".join(f"{k}:{v}" for k, v in sorted(dropped_boxes.items())))
    cur_img = sum(len(list((DST / 'images' / sp).iterdir())) for sp in ("train", "valid", "test"))
    cur_box = Counter()
    for sp in ("train", "valid", "test"):
        for lp in (DST / "labels" / sp).glob("*.txt"):
            for ln in lp.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    cur_box[int(float(ln.split()[0]))] += 1
    print(f"\n现状: {cur_img} 张  " + "  ".join(
        f"{['mask','gloves','lab_coat','goggles'][k]}:{v}" for k, v in sorted(cur_box.items())))
    fut = cur_box.copy()
    fut[GOOGLES_CLS] += n_box
    print(f"入库后: {cur_img + len(jobs)} 张  " + "  ".join(
        f"{['mask','gloves','lab_coat','goggles'][k]}:{v}" for k, v in sorted(fut.items())))

    if not args.apply:
        print("\n[干跑] 未复制。确认后加 --apply。")
        return 0

    done = 0
    for sp, img_name, dst_sp, dst_stem, out_lines in jobs:
        idst = DST / "images" / dst_sp / (dst_stem + ".jpg")
        ldst = DST / "labels" / dst_sp / (dst_stem + ".txt")
        if idst.exists() and ldst.exists():
            continue
        if not idst.exists():
            shutil.copy2(SRC / sp / "images" / img_name, idst)
        ldst.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
        done += 1
    print(f"\n[完成] 入库 {done} 张图 + 标注。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
