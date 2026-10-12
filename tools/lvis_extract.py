#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LVIS -> 实验室/PPE 素材筛选工具（许婧雯 / AI 数据线）

LVIS 不是图片素材库，而是"COCO 图 + 1203 类长尾标注"。所以"找素材"= 从标注里
筛出目标类 -> 取出对应图片 -> （可选）转成本项目 5 类 YOLO 标签。

本项目 5 类映射（LVIS 类名 -> 本仓库类索引）:
    person -> 0, mask -> 1, gloves -> 2, lab_coat -> 3, goggles -> 4

2026-10-10 实测结论（val split 19809 图）:
  - goggles: LVIS 的 goggles 是滑雪镜，候选 327 张全是雪山/滑雪场场景，
    与实验室护目镜域不符 -> 已放弃挖掘（urls.txt 已清）。
  - lab_coat: 仅 3 张图；000000289229 是马展照（© 水印）已删，
    可用仅 000000225721（真实验室场景）、000000355830（影棚白大褂）。
  - LVIS 图源自 COCO，部分带摄影师水印，入库前必须逐张查水印 + 场景合规。

用法（在仓库根目录、用 .venv/Scripts/python.exe 跑）:

  1) 只统计、不下载图片（默认自动下 64MB 的 lvis_v1_val.json）:
       .venv/Scripts/python.exe tools/lvis_extract.py scan

  2) 导出候选图清单（url + stem），并可选下载图片:
       .venv/Scripts/python.exe tools/lvis_extract.py export --names lab_coat,goggles --download

  3) 把已下载的图转成本项目 5 类 YOLO 标签:
       .venv/Scripts/python.exe tools/lvis_extract.py yolo --images-dir outputs/lvis_mine/images

产出统一放在 outputs/lvis_mine/ 下，不碰 data/lab_ppe（红线数据集）。
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "lvis_mine"

ANN_URL = {
    "val": "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip",      # 64 MB, 19809 图
    "train": "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip",  # 350 MB, 100170 图
}
ANN_ZIP = {"val": "lvis_v1_val.json.zip", "train": "lvis_v1_train.json.zip"}

# LVIS 名字(下划线形式) -> 本项目类索引。None = 只导出不转标签
PROJ_CLASS = {
    "person": 0,
    "mask": 1,
    "gloves": 2,
    "lab_coat": 3,
    "goggles": 4,
}
# 常见 LVIS 别名 -> 规范名
ALIAS = {
    "glove": "gloves",
    "goggle": "goggles",
    "labcoat": "lab_coat",
    "laboratory_coat": "lab_coat",
    "face_mask": "mask",
    "facemask": "mask",
}
# 额外值得关注的 PPE/器材类（不参与转标签，仅统计）
EXTRA_WATCH = [
    "apron", "hairnet", "coverall", "gasmask", "earplug", "helmet", "safety_pin",
    "fire_extinguisher", "microscope", "syringe", "visor", "suit_(clothing)",
    "jumpsuit", "medical_suit", "safety_vest", "vest", "glasses", "bottle", "jar",
]


def norm(name: str) -> str:
    n = name.strip().lower().replace(" ", "_").replace("-", "_")
    return ALIAS.get(n, n)


def fname(im: dict) -> str:
    """LVIS 的 images 没有 file_name 字段，文件名从 coco_url 末段取。"""
    return Path(im["coco_url"]).name


def get_ann(split: str, cache_dir: Path, refresh: bool = False) -> Path:
    """返回本地 lvis_v1_<split>.json 路径，必要时下载解压。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    json_path = cache_dir / f"lvis_v1_{split}.json"
    if json_path.exists() and not refresh:
        return json_path
    zp = cache_dir / ANN_ZIP[split]
    if not zp.exists() or refresh:
        print(f"[下载] {ANN_URL[split]}  ->  {zp}", file=sys.stderr)
        urlretrieve(ANN_URL[split], zp)
    print(f"[解压] {zp.name}", file=sys.stderr)
    with zipfile.ZipFile(zp) as z:
        inner = z.namelist()[0]
        json_path.write_bytes(z.read(inner))
    return json_path


def load(json_path: Path):
    with open(json_path, encoding="utf-8") as f:
        d = json.load(f)
    imgs = {im["id"]: im for im in d["images"]}
    by_cat = defaultdict(list)
    for a in d["annotations"]:
        by_cat[a["category_id"]].append(a)
    return d, imgs, by_cat


def cmd_scan(args) -> None:
    json_path = get_ann(args.split, Path(args.cache), args.refresh)
    d, imgs, by_cat = load(json_path)

    want = [norm(x) for x in args.names.split(",")] if args.names else sorted(PROJ_CLASS)
    print(f"\nLVIS v1 / {args.split}: {len(d['images'])} 图, "
          f"{len(d['annotations'])} 标注, {len(d['categories'])} 类\n")

    rows, missing = [], []
    for c in d["categories"]:
        n = norm(c["name"])
        if n in want:
            anns = by_cat.get(c["id"], [])
            rows.append((n, c["id"], c["instance_count"], len(anns),
                         len({a["image_id"] for a in anns}), c.get("frequency", "?"), c["name"]))
    for w in want:
        if not any(r[0] == w for r in rows):
            missing.append(w)

    rows.sort(key=lambda r: -r[3])
    print(f"{'本项目类':<10}{'LVIS id':>8}{'instance':>10}{'本split实例':>12}{'图片数':>8}  freq  LVIS 原名")
    print("-" * 82)
    for n, cid, total, got, nimg, freq, raw in rows:
        proj = PROJ_CLASS.get(n)
        flag = f"-> 类{proj}" if proj is not None else "       "
        print(f"{n:<10}{cid:>8}{total:>10}{got:>12}{nimg:>8}  {freq:<4}  {raw} {flag}")
    if missing:
        print(f"\n[注意] 以下名字没匹配到类别: {missing}")

    print(f"\n-- 其他值得看的 PPE/器材类（{args.split} 实测）--")
    for c in sorted(d["categories"], key=lambda x: -x["instance_count"]):
        if c["name"] in EXTRA_WATCH:
            anns = by_cat.get(c["id"], [])
            print(f"  {c['name']:<24} id={c['id']:<5} 实例={c['instance_count']:<6} "
                  f"图={len({a['image_id'] for a in anns})}")

    OUT.mkdir(parents=True, exist_ok=True)
    rep = OUT / f"scan_{args.split}.txt"
    with open(rep, "w", encoding="utf-8") as f:
        f.write(f"LVIS v1 {args.split}: {len(d['images'])} images\n")
        for n, cid, total, got, nimg, freq, raw in rows:
            f.write(f"{n}\t{cid}\t{total}\t{got}\t{nimg}\t{freq}\t{raw}\n")
    print(f"\n[写出] {rep.relative_to(ROOT)}")


def cmd_export(args) -> None:
    json_path = get_ann(args.split, Path(args.cache), args.refresh)
    d, imgs, by_cat = load(json_path)
    want = {norm(x) for x in args.names.split(",")}
    tag = args.tag or "_".join(sorted(want))

    picked = defaultdict(list)  # img_id -> [ann]
    for c in d["categories"]:
        if norm(c["name"]) in want:
            for a in by_cat.get(c["id"], []):
                picked[a["image_id"]].append(a)

    outdir = OUT / tag
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) url 清单（含每图命中的类）
    listpath = outdir / "urls.txt"
    lines = []
    for iid, anns in sorted(picked.items()):
        im = imgs[iid]
        cls = ",".join(sorted({norm(next(c["name"] for c in d["categories"] if c["id"] == a["category_id"]))
                               for a in anns}))
        lines.append(f"{im['coco_url']}\t{fname(im)}\t{cls}")
    listpath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[写出] {listpath.relative_to(ROOT)}  ({len(lines)} 张候选图)")

    # 2) 可选下载
    if args.download:
        imgdir = outdir / "images"
        imgdir.mkdir(exist_ok=True)
        for i, (iid, anns) in enumerate(sorted(picked.items()), 1):
            im = imgs[iid]
            dst = imgdir / fname(im)
            if not dst.exists():
                urlretrieve(im["coco_url"], dst)
            if i % 25 == 0:
                print(f"  ... {i}/{len(picked)}")
        print(f"[下载] {imgdir.relative_to(ROOT)}  ({len(picked)} 张)")

    # 3) 顺带导出该批的 YOLO 标签（图片下载后可配合 yolo 子命令核对）
    print("\n提示：下载完后可跑 "
          f"`tools/lvis_extract.py yolo --images-dir outputs/lvis_mine/{tag}/images`")


def cmd_yolo(args) -> None:
    json_path = get_ann(args.split, Path(args.cache), args.refresh)
    d, imgs, by_cat = load(json_path)
    img_dir = Path(args.images_dir)
    if not img_dir.is_absolute():
        img_dir = ROOT / img_dir

    name_by_id = {c["id"]: norm(c["name"]) for c in d["categories"]}
    labels = defaultdict(list)
    for a in d["annotations"]:
        proj = PROJ_CLASS.get(name_by_id[a["category_id"]])
        if proj is None:
            continue
        im = imgs[a["image_id"]]
        fn = fname(im)
        if not (img_dir / fn).exists():
            continue
        w, h = im["width"], im["height"]
        x, y, bw, bh = a["bbox"]
        if bw <= 1 or bh <= 1:      # 过滤极小的噪声框（可调）
            continue
        cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
        labels[fn].append(f"{proj} {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")

    labdir = Path(args.out or OUT / "labels")
    labdir.mkdir(parents=True, exist_ok=True)
    for fn, rows in labels.items():
        (labdir / (Path(fn).stem + ".txt")).write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"[写出] {labdir}  {len(labels)} 个标签文件, {sum(len(v) for v in labels.values())} 个框")
    print("提醒：这批图的场景五花八门（LVIS 是老 COCO 图），进训练集前必须人工过一遍场景合规。")


def main() -> None:
    p = argparse.ArgumentParser(description="从 LVIS 标注筛实验室/PPE 素材")
    p.add_argument("--cache", default=str(ROOT / "data" / "_lvis"), help="标注缓存目录")
    p.add_argument("--split", default="val", choices=["val", "train"])
    p.add_argument("--refresh", action="store_true", help="重新下载标注")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="统计各类实例/图片数（默认全部目标类）")
    s.add_argument("--names", default="", help="逗号分隔，如 glove,goggles,mask,lab_coat")
    s.set_defaults(func=cmd_scan)

    e = sub.add_parser("export", help="导出候选图 url 清单/图片")
    e.add_argument("--names", required=True, help="逗号分隔，如 lab_coat,goggles")
    e.add_argument("--tag", default="", help="输出子目录名")
    e.add_argument("--download", action="store_true", help="顺带下载图片")
    e.set_defaults(func=cmd_export)

    y = sub.add_parser("yolo", help="把已下载的图转成本项目 5 类 YOLO 标签")
    y.add_argument("--images-dir", required=True)
    y.add_argument("--out", default="")
    y.set_defaults(func=cmd_yolo)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
