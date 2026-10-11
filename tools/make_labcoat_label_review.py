# -*- coding: utf-8 -*-
"""生成 lab_coat 单类画框工作区（复用 make_box_editor 编辑器）。

背景：data/raw/labcoat_imagenetcv/data/{train,val,test}/lab_coat/ 下 944 张
经自动筛查通过的白大褂图片是**分类图**，没有任何检测框，不能直接混训。
本工具按批生成编辑器工作区，让她逐图人工画 lab_coat 框（类别固定 3=lab_coat）。

批次语义（与 person_review 一致）：她翻完整批 → 有改动的图在编辑器导出贴回
→ 回「这批 OK」→ 我整批落库（她贴的文本覆盖有改动的图，未改动的按她画的框入库）。

用法:
  $PY tools/make_labcoat_label_review.py --limit 60 --start 0 --tag lc13a
  $PY tools/make_labcoat_label_review.py --limit 60 --start 60 --tag lc13b
  # 只看某 split：
  $PY tools/make_labcoat_label_review.py --split train --limit 60 --tag lc13a
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from make_box_editor import HTML  # noqa: E402

SRC = ROOT / "data" / "raw" / "labcoat_imagenetcv" / "data"
CLEAN = ROOT / "outputs" / "labcoat_imagenetcv_screen" / "clean.txt"
LAB_COAT_CLS = 3   # 五类口径 0=person 1=mask 2=gloves 3=lab_coat 4=goggles


def imsize(p: Path):
    img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    return (img.shape[1], img.shape[0]) if img is not None else (0, 0)


def load_clean() -> list[tuple[int, str]]:
    """读 clean.txt → [(编号, 相对路径)]（相对 SRC，形如 train/lab_coat/xxx.jpg）"""
    rows = []
    if not CLEAN.exists():
        return rows
    for ln in CLEAN.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        p = ln.split("\t")
        if len(p) >= 2 and p[0].strip().isdigit():
            rows.append((int(p[0]), p[1].strip()))
    return rows


def imread_unicode(p: Path):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--tag", default="lc13a")
    ap.add_argument("--split", default="", help="只取某 split：train/val/test（默认全部）")
    ap.add_argument("--max-side", type=int, default=1600,
                    help="预览图最长边超过此值则生成缩图（坐标按原图归一化，不影响标签精度）")
    ap.add_argument("--out", default="")
    ap.add_argument("--done-file", default="data/_labcoat_label_done.txt",
                    help="已完成图清单（落库时追加），生成新批次时自动跳过")
    ap.add_argument("--exclude-export", default="", help="编辑器导出 txt，其 ### 图视为已完成")
    ap.add_argument("--stems", default="", help="逗号分隔 stem：强制批次内容（复核重做用）")
    ap.add_argument("--note", default="", help="批次备注")
    args = ap.parse_args()

    all_rows = load_clean()
    if not all_rows:
        print("  clean.txt 为空，先跑 tools/screen_labcoat_imagenetcv.py")
        return 1
    print(f"  clean 清单 {len(all_rows)} 张")

    # 已完成排除
    excl: set[str] = set()
    for s in args.exclude_export.split(","):
        s = s.strip()
        if not s:
            continue
        sp = Path(s)
        if sp.exists():
            for ln in sp.read_text(encoding="utf-8").splitlines():
                if ln.strip().startswith("###"):
                    excl.add(ln.strip()[3:].strip())
    done_p = ROOT / args.done_file
    if args.done_file and done_p.exists():
        for ln in done_p.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                excl.add(ln.strip())
    if excl:
        print(f"  已完成剔除 {len(excl)} 张（{args.done_file} + 导出）")

    items = []       # (编号, 相对路径, Path)
    for no, rel in all_rows:
        if args.split and not rel.startswith(args.split + "/"):
            continue
        stem = Path(rel).stem
        if stem in excl:
            continue
        items.append((no, rel, SRC / rel))

    force = [s.strip() for s in args.stems.split(",") if s.strip()]
    if force:
        by_stem = {Path(r[1]).stem: r for r in items}
        by_name = {r[2].name: r for r in items}
        batch, missing = [], []
        for s in force:
            r = by_stem.get(s) or by_name.get(s) or by_stem.get(Path(s).stem)
            (batch if r else missing).append(r if r else s)
        print(f"  强制批次：指定 {len(force)}，命中 {len(batch)}"
              + (f"，缺失 {missing}" if missing else ""))
    else:
        print(f"  待标注队列 {len(items)} 张，取 {args.start}~{args.start + args.limit - 1}")
        batch = items[args.start:args.start + args.limit]

    if not batch:
        print("  空批")
        return 0

    out_dir = ROOT / "outputs" / f"labcoat_label_review_{args.tag}"
    ws = ROOT / "data" / "_prelabel" / f"labcoat_{args.tag}" / "labels"
    view = out_dir / "_view"
    ws.mkdir(parents=True, exist_ok=True)
    view.mkdir(parents=True, exist_ok=True)

    imgs = []
    scope = "复核重做" if force else f"队列 {args.start}+{len(batch)}"
    if args.split:
        scope += f" · split={args.split}"
    if args.note:
        scope += f" · {args.note}"
    index = [f"# lab_coat 人工画框 · 批 {args.tag} · {scope}",
             "# 这些图来自 images.cv 1.3K 白大褂分类图（经自动筛查通过），**原本没有任何检测框**",
             "# 重点：给每件白大褂/实验服画一个框（按 4 = lab_coat）。另：若画面里还看清了",
             "#   person(1) / mask(2) / gloves(3) / goggles(5)，一并标上更佳（可自由切类别）。",
             "#   空白拖拽=新建框 · 数字键 1-5 切类别 · 拖框内=移动 · 拖右下白手柄=缩放 · 点右上红×=删框",
             "#   不合适进训练的图（非实验/医疗场景、无白大褂、水印重、看不清）→ 按 X 或点「剔除此图」",
             "#   ⚠️ 口径已定案（2026-10-11，路线 A）：连体防护服 / Coverall 一律【算 lab_coat】，请照样按 4 画框，不要剔除、不要留空！",
             "#   一张图多件白大褂 → 画多个框；确认无白大褂就剔图，别留空",
             "# 编号\t文件\t原图编号\t尺寸"]
    for i, (no, rel, ip) in enumerate(batch):
        w, h = imsize(ip)
        if w == 0 or h == 0:
            print(f"  ⚠ 读图失败，跳过：{rel}")
            continue
        vp, vw, vh = ip, w, h
        if max(w, h) > args.max_side:
            s = args.max_side / max(w, h)
            vw, vh = max(1, int(w * s)), max(1, int(h * s))
            vp = view / (Path(rel).stem + ".jpg")
            if not vp.exists():
                im = imread_unicode(ip)
                cv2.imencode(".jpg", cv2.resize(im, (vw, vh), interpolation=cv2.INTER_AREA),
                             [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(str(vp))
        # 分类图：无任何既有框。若同批重复生成，可从工作区读回上次进度
        rows = []
        wf = ws / (Path(rel).stem + ".txt")
        if wf.exists():
            for ln in wf.read_text(encoding="utf-8").splitlines():
                q = ln.split()
                if len(q) == 5:
                    c, cx, cy, bw, bh = int(q[0]), *[float(v) for v in q[1:5]]
                    rows.append([c, round(min(max(cx - bw / 2, 0.0), 1.0), 6),
                                 round(min(max(cy - bh / 2, 0.0), 1.0), 6),
                                 round(bw, 6), round(bh, 6)])
        imgs.append({"no": i + 1, "name": Path(rel).stem + ".jpg",
                     "path": "file:///" + str(vp).replace("\\", "/"),
                     "nw": vw, "nh": vh, "boxes": rows, "rel": rel, "orig": no})
        index.append(f"{i + 1}\t{rel}\t{no}\t{w}x{h}")

    html = (HTML.replace("__TITLE__", f"lab_coat 画框 · 批 {args.tag} · {len(imgs)} 张")
            .replace("__DATA__", json.dumps(
                {"imgs": [{"name": o["name"], "path": o["path"], "nw": o["nw"],
                           "nh": o["nh"], "boxes": o["boxes"], "no": o["no"]} for o in imgs],
                 "focus": 0}, ensure_ascii=False)))
    (out_dir / "editor.html").write_text(html, encoding="utf-8")
    (out_dir / "index.txt").write_text("\n".join(index) + "\n", encoding="utf-8")
    # 映射表：stem → 相对路径（落库端据此把标签放回 data/lab_ppe）
    (out_dir / "_stem2rel.tsv").write_text(
        "\n".join(f"{Path(o['rel']).stem}\t{o['rel']}" for o in imgs) + "\n", encoding="utf-8")
    print(f"  editor.html → {out_dir / 'editor.html'}")
    print(f"  工作区标签   → {ws}")
    print(f"  本批 {len(imgs)} 张（编辑器默认类别请按 4 选 lab_coat，或直接拖拽即 lab_coat）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
