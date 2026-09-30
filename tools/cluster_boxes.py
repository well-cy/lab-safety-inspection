#!/usr/bin/env python3
"""cluster_boxes.py — 标注框位置聚类（毒标签批量定位用）

用途：监控视频抽帧来自**静止机位**，同一个误标往往在成百上千帧里重复出现在
几乎相同的坐标上。把同类框按 (xc,yc,w,h) 聚类，就能一眼看出"哪些框是同一处
重复错误"，从而把人工判定从"逐张看 274 张"压缩到"看几个簇的代表帧"。

两种用法：
  1) 只看分布（默认）——输出簇表，判断有多少是重复误标：
     python tools/cluster_boxes.py --class goggles --prefix ppes_ --min-size 5
  2) 与人工判定对照（--index + --keep/--drop）——验证"判对的簇 vs 判错的簇"
     是否干净可分；可分则整簇批量清理：
     python tools/cluster_boxes.py --class goggles --prefix ppes_ \
         --index outputs/_goggles_clean_batch01_zoom/index.txt --keep 5,7,9,10,12,13,16,23,28,29

输出：簇表 + 每簇代表帧清单（可直接喂给 make_box_zoom_sheet.py 抽检）。
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "lab_ppe"
CLASS_ID = {"mask": 0, "gloves": 1, "lab_coat": 2, "goggles": 3}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="cls", default="goggles")
    ap.add_argument("--split", default="train")
    ap.add_argument("--prefix", default="", help="只看文件名以此开头的图（如 ppes_）；留空=全部")
    ap.add_argument("--tol-pos", type=float, default=0.03, help="中心点聚类容差（归一化）")
    ap.add_argument("--tol-size", type=float, default=0.02, help="宽高聚类容差（归一化）")
    ap.add_argument("--min-size", type=int, default=5, help="只列出 >= 该框数的簇")
    ap.add_argument("--index", default="", help="复查对照表（与人工判定对照时用）")
    ap.add_argument("--keep", default="", help="人工判定为正确的编号")
    ap.add_argument("--drop", default="", help="人工判定为错误的编号")
    ap.add_argument("--no-backup", dest="with_backup", action="store_false",
                    help="不把 _label_backup 里已清理的框纳入聚类（默认纳入）")
    ap.add_argument("--list-out", default="", help="把最大几个簇的代表帧写成清单文件")
    ap.add_argument("--top", type=int, default=3, help="--list-out 导出前几个簇的代表帧")
    args = ap.parse_args()

    cls_id = CLASS_ID.get(args.cls.strip().lower(), -1) if not args.cls.strip().isdigit() \
        else int(args.cls)
    lab_dir = DATA / "labels" / args.split
    img_dir = DATA / "images" / args.split
    img_stems = {p.stem for p in img_dir.iterdir()} if img_dir.is_dir() else set()

    boxes = []          # (stem, xc, yc, w, h, batch)
    for p in sorted(lab_dir.glob("*.txt")):
        if args.prefix and not p.name.startswith(args.prefix):
            continue
        if p.stem not in img_stems:
            continue
        for ln in p.read_text(encoding="utf-8").splitlines():
            s = ln.split()
            if len(s) == 5 and s[0] == str(cls_id):
                xc, yc, w, h = (float(x) for x in s[1:])
                m = re.match(r"^([A-Za-z_]+?)[\d_]", p.stem)
                boxes.append((p.stem, xc, yc, w, h, m.group(1) if m else "?"))
    print(f"[范围] {args.split} / {args.cls} / 前缀'{args.prefix or '*'}' → {len(boxes)} 个框")

    # 已清理过的框已不在标签里，从 data/_label_backup 补齐，否则聚类结果缺了这些框、
    # 人工判定对照会出现"找不到坐标"的假象（本 bug 于 2026-09-30 实测踩到）。
    bk = ROOT / "data" / "_label_backup"
    if args.with_backup and bk.is_dir():
        have = {b[0] for b in boxes}
        added = 0
        for f in bk.rglob("*.txt"):
            stem = f.stem
            if stem in have or (args.prefix and not f.name.startswith(args.prefix)):
                continue
            if img_stems and stem not in img_stems:
                continue
            got = False
            for ln in f.read_text(encoding="utf-8").splitlines():
                s = ln.split()
                if len(s) == 5 and s[0] == str(cls_id):
                    xc, yc, w, h = (float(x) for x in s[1:])
                    m = re.match(r"^([A-Za-z_]+?)[\d_]", stem)
                    boxes.append((stem, xc, yc, w, h, m.group(1) if m else "?"))
                    got = True
            added += 1 if got else 0
        print(f"[备份补齐] 从 _label_backup 追加 {added} 个历史框（已清理的框重新纳入聚类）")

    # 贪心聚类（按 xc 排序后聚合）
    clusters: list[list] = []
    for b in sorted(boxes, key=lambda x: (x[1], x[2])):
        for cl in clusters:
            r = cl[0]
            if (abs(b[1] - r[1]) < args.tol_pos and abs(b[2] - r[2]) < args.tol_pos
                    and abs(b[3] - r[3]) < args.tol_size and abs(b[4] - r[4]) < args.tol_size):
                cl.append(b)
                break
        else:
            clusters.append([b])
    clusters.sort(key=len, reverse=True)

    print(f"\n{'簇':>3} {'框数':>5}  位置(xc,yc)      尺寸(w,h)      批次构成")
    print("-" * 92)
    for i, cl in enumerate(clusters, 1):
        n = len(cl)
        if n < args.min_size:
            continue
        xc = sum(b[1] for b in cl) / n
        yc = sum(b[2] for b in cl) / n
        w = sum(b[3] for b in cl) / n
        h = sum(b[4] for b in cl) / n
        bc = collections.Counter(b[5] for b in cl)
        print(f"{i:>3} {n:>5}  ({xc:.3f},{yc:.3f})  ({w:.3f},{h:.3f})  "
              + " ".join(f"{k}:{v}" for k, v in bc.most_common()))
    print("-" * 92)
    tot = sum(len(c) for c in clusters)
    big = sum(len(c) for c in clusters if len(c) >= args.min_size)
    print(f"共 {len(clusters)} 簇 / {tot} 框；>= {args.min_size} 框的簇覆盖 {big} 框 "
          f"({big / tot * 100 if tot else 0:.1f}%)；孤立框 {sum(1 for c in clusters if len(c) == 1)}")

    # --- 与人工判定对照 ---
    if args.index and (args.keep or args.drop):
        index = {}
        for ln in pathlib.Path(args.index).read_text(encoding="utf-8").splitlines():
            if ln.startswith("#") or not ln.strip():
                continue
            p = ln.split("\t")
            if len(p) >= 2:
                index[int(p[0].strip())] = pathlib.Path(p[1].strip()).stem
        verdict = {}
        if args.keep:
            keep = {int(x) for x in args.keep.replace(" ", "").split(",") if x.strip()}
        else:
            drop = {int(x) for x in args.drop.replace(" ", "").split(",") if x.strip()}
            keep = set(index) - drop
        for num, stem in index.items():
            verdict[stem] = "OK" if num in keep else "BAD"
        by_stem = {}
        for b in boxes:
            by_stem.setdefault(b[0], b)
        print(f"\n{'编号':>4} {'判定':<5} {'簇':>4}  位置(xc,yc)")
        print("-" * 56)
        stat = collections.defaultdict(collections.Counter)
        for i, cl in enumerate(clusters, 1):
            for b in cl:
                if b[0] in verdict:
                    stat[i][verdict[b[0]]] += 1
        for num in sorted(index):
            stem = index[num]
            b = by_stem.get(stem)
            if not b:
                print(f"{num:>4} {verdict[stem]:<5} {'-':>4}  (找不到坐标)")
                continue
            ci = next((i for i, cl in enumerate(clusters, 1) if b[0] in {x[0] for x in cl}), -1)
            print(f"{num:>4} {verdict[stem]:<5} {ci:>4}  ({b[1]:.3f},{b[2]:.3f})")
        print("-" * 56)
        print("簇 → 判定统计 (OK=判为护目镜, BAD=判为眼镜)")
        for i in sorted(stat):
            print(f"  簇 {i:>3}: " + " ".join(f"{k}:{v}" for k, v in sorted(stat[i].items())))

    if args.list_out and clusters:
        reps = []
        for cl in clusters[: args.top]:
            reps += [b[0] for b in cl]
        out = pathlib.Path(args.list_out)
        out.write_text("\n".join(f"{args.split}/{s}.jpg" for s in dict.fromkeys(reps)) + "\n",
                       encoding="utf-8")
        print(f"\n[导出] 前 {args.top} 簇共 {len(set(reps))} 张代表帧 → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
