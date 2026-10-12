#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
screen_dirty_graphics.py —— 用图像特征自动筛查"非实拍脏图"（商品目录图/广告图/纯文字图）

背景：人工只看几何 flags 覆盖不到"商品图、广告图、报价单"这类域污染，
      这类图有明显统计特征，可以自动筛掉大部分，减少人工翻图量。

判据（基于 256x256 缩放后的 HSV / 局部方差 / 边缘密度）：
  1. graphic_white  商品目录图：白底占比高 + 大面积平坦区域
  2. graphic_color  广告图：单一色调占比高 + 大面积平坦区域
  3. blank_doc      纯文字/空白文档：白底占比极高 + 边缘极少

用法:
    python tools/screen_dirty_graphics.py --class mask                      # 干跑，只出清单
    python tools/screen_dirty_graphics.py --class mask --thumbs outputs/x  # 同时导出缩略图版
    python tools/screen_dirty_graphics.py --class mask --apply              # 生成剔除索引（配合 remove_bad_images.py）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "data" / "lab_ppe"
NAMES = ["mask", "gloves", "lab_coat", "goggles"]


def imread_unicode(p: Path):
    """cv2.imread 不支持中文路径，用 imdecode 兜底。"""
    try:
        data = np.fromfile(str(p), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def image_stats(img) -> dict:
    small = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0]
    s = hsv[..., 1].astype(np.float32) / 255.0
    v = hsv[..., 2].astype(np.float32) / 255.0

    near_white = (s < 0.18) & (v > 0.82)
    white_ratio = float(near_white.mean())

    sat = s > 0.40
    if sat.sum() > 50:
        hist = np.bincount((h[sat] // 30).astype(np.int32), minlength=6)
        dom_ratio = float(hist.max() / sat.sum())
    else:
        dom_ratio = 0.0

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mu = cv2.blur(gray, (5, 5))
    mu2 = cv2.blur(gray * gray, (5, 5))
    var = np.clip(mu2 - mu * mu, 0, None)
    flat_ratio = float((var < 12).mean())

    edges = cv2.Canny(gray.astype(np.uint8), 80, 180)
    edge_ratio = float((edges > 0).mean())

    return dict(white=white_ratio, dom=dom_ratio, flat=flat_ratio, edge=edge_ratio)


def classify(st: dict) -> str | None:
    # 1. 商品目录图：白底占比很高 + 绝大部分区域平坦（排版图特征）
    if st["white"] > 0.50 and st["flat"] > 0.60:
        return "graphic_white"
    # 2. 广告/排版图：单色调主导且近乎全图平坦（照片必有纹理起伏，不会全平）
    if st["dom"] > 0.55 and st["flat"] > 0.62:
        return "graphic_color"
    # 3. 纯文字/空白文档
    if st["white"] > 0.45 and st["edge"] < 0.02:
        return "blank_doc"
    return None


def iter_labeled(cls_id: int, splits: list[str]):
    for sp in splits:
        for lp in sorted((DS / "labels" / sp).glob("*.txt")):
            lines = [l for l in lp.read_text(encoding="utf-8").splitlines() if l.strip()]
            if any(l.split()[0] in (str(cls_id), f"{cls_id}.0") for l in lines):
                yield sp, lp


def extra_stats(img) -> dict:
    """额外指标：亮度（域偏移参考）、彩色度（识黑白老照片）。"""
    small = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    v = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[..., 2].astype(np.float32) / 255.0
    b, g, r = small[..., 0].astype(np.float32), small[..., 1].astype(np.float32), small[..., 2].astype(np.float32)
    colorf = float((np.abs(r - g).mean() + np.abs(g - b).mean() + np.abs(r - b).mean()) / 3.0)
    return dict(bright=float(v.mean()), colorf=colorf)


def scan_raw(d: Path, tsv_out: Path | None) -> int:
    """裸目录模式：筛查一批**未标注**候选图（网页采集池、B1 自采原片等）。"""
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in {e.lower() for e in exts})
    print(f"[筛查/裸目录] {d}")
    print(f"  受检图片 {len(imgs)} 张")

    rows, hits, unread = [], {}, []
    gray_like = []
    for p in imgs:
        img = imread_unicode(p)
        if img is None:
            unread.append(p.name)
            rows.append((p.name, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "unreadable"))
            continue
        h, w = img.shape[:2]
        st = image_stats(img)
        ex = extra_stats(img)
        tag = classify(st)
        if tag is None and ex["colorf"] < 6.0:
            tag = "grayscale_like"
            gray_like.append(p.name)
        if tag:
            hits.setdefault(tag, []).append(p.name)
        rows.append((p.name, w, h, round(ex["bright"], 4), round(ex["colorf"], 3),
                     round(st["white"], 4), round(st["dom"], 4),
                     round(st["flat"], 4), round(st["edge"], 4), tag or "ok"))

    for tag, files in sorted(hits.items()):
        print(f"  {tag}: {len(files)} 张")
    n_hit = sum(len(v) for v in hits.values())
    print(f"  合计可疑 {n_hit} 张（{n_hit/max(len(imgs),1):.1%}）")
    if unread:
        print(f"  无法读取 {len(unread)} 张")

    if tsv_out:
        tsv_out.parent.mkdir(parents=True, exist_ok=True)
        hdr = ["name", "width", "height", "bright", "colorf",
               "white", "dom", "flat", "edge", "flag"]
        with tsv_out.open("w", encoding="utf-8", newline="") as f:
            f.write("\t".join(hdr) + "\n")
            for r in rows:
                f.write("\t".join(str(x) for x in r) + "\n")
        print(f"[明细] {tsv_out}")

    if hits:
        out = d.parent / "flags_dirty_raw.txt"
        allf = sorted(f for v in hits.values() for f in v)
        tag_of = {f: t for t, files in hits.items() for f in files}
        out.write_text(
            "# 裸目录脏图筛查命中（非实拍/黑白老照片）\n" +
            "\n".join(f"{i+1}\t{f}\tdirty:{tag_of[f]}" for i, f in enumerate(allf)) + "\n",
            encoding="utf-8")
        print(f"[清单] {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="cls", default="", help="类名或 id：mask/gloves/lab_coat/goggles（数据集模式必填）")
    ap.add_argument("--splits", default="train,valid,test")
    ap.add_argument("--dir", default="", help="裸目录模式：直接筛查该目录下所有图片（无需标注）")
    ap.add_argument("--tsv", default="", help="裸目录模式：输出逐图指标明细 TSV")
    ap.add_argument("--thumbs", default="", help="导出命中图的缩略图版目录（可选）")
    ap.add_argument("--out", default="", help="剔除索引输出路径（默认 data/flags_dirty_<class>.txt）")
    ap.add_argument("--apply", action="store_true", help="写出剔除索引（不移动文件，移动用 remove_bad_images.py）")
    args = ap.parse_args()

    if args.dir:
        return scan_raw(Path(args.dir),
                        Path(args.tsv) if args.tsv else None)
    if not args.cls:
        ap.error("需要 --class（数据集模式）或 --dir（裸目录模式）")

    tok = args.cls.strip().lower()
    cid = int(tok) if tok.isdigit() else NAMES.index(tok)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    hits: dict[str, list[str]] = {}
    total = 0
    for sp, lp in iter_labeled(cid, splits):
        img = imread_unicode(DS / "images" / sp / (lp.stem + ".jpg"))
        if img is None:  # 尝试其它扩展名
            for ext in (".png", ".jpeg", ".JPG", ".PNG"):
                img = imread_unicode(DS / "images" / sp / (lp.stem + ext))
                if img is not None:
                    break
        if img is None:
            continue
        total += 1
        tag = classify(image_stats(img))
        if tag:
            hits.setdefault(tag, []).append(f"{sp}/{lp.stem}")

    print(f"[筛查] 类别={NAMES[cid]}  受检图片 {total} 张")
    for tag, files in sorted(hits.items()):
        print(f"  {tag}: {len(files)} 张")
    n_hit = sum(len(v) for v in hits.values())
    print(f"  合计可疑 {n_hit} 张（{n_hit/max(total,1):.1%}）")

    # 剔除索引（按家族扩展交给 remove_bad_images.py，这里按文件列出）
    all_files = sorted(f for v in hits.values() for f in v)
    if all_files:
        tag_of = {f: t for t, files in hits.items() for f in files}
        out = Path(args.out) if args.out else (ROOT / "data" / f"flags_dirty_{NAMES[cid]}.txt")
        # ① 带编号的剔除索引（给 remove_bad_images.py 用）
        lines = [f"{i+1}\t{f}.jpg\tdirty:{tag_of[f]}" for i, f in enumerate(all_files)]
        out.write_text(f"# 自动筛查可疑脏图（非实拍：商品图/广告图/文档）\n" + "\n".join(lines) + "\n",
                       encoding="utf-8")
        # ② 无编号清单（给 make_contact_sheet.py --list 用）
        sheet = out.with_name(out.stem + "_sheet.txt")
        sheet.write_text("# 拼版用清单: split/文件名 <TAB> 标签\n" +
                         "\n".join(f"{f}.jpg\tdirty:{tag_of[f]}" for f in all_files) + "\n",
                         encoding="utf-8")
        print(f"[清单] {out}")
        print(f"[拼版清单] {sheet}")

    if args.thumbs and all_files:
        sys.path.insert(0, str(ROOT / "tools"))
        import importlib
        sheet_mod = importlib.import_module("make_contact_sheet")
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "tools" / "make_contact_sheet.py"),
                        "--list", str(out), "--root", str(DS), "--only-class", NAMES[cid],
                        "--out", args.thumbs], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
