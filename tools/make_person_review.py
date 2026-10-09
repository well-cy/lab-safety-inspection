# -*- coding: utf-8 -*-
"""生成 person 补标人工审核工作区（复用 make_box_editor 编辑器）。

对指定 split 的存量图（非网图批），把「现有五类标签 + COCO person 候选(conf≥阈值)」
合成到工作区 data/_prelabel/person_<tag>/labels/，并生成 editor.html 供她逐图确认。

批次语义（与 r3 相同）：她翻完整批 → 有改动的图在编辑器导出贴回 → 回"这批 OK"
→ 我整批落库（她贴的文本覆盖有改动的图）。

用法:
  $PY tools/make_person_review.py --split train --limit 60 --start 0 --tag p0_a
  $PY tools/make_person_review.py --split train --limit 60 --start 60 --tag p0_b
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from make_box_editor import HTML  # noqa: E402

import re as _re


def fam_of(n: str) -> str:
    """与 survey_person 同口径（内联避免其模块级 stdout 包装冲突）。"""
    if n.startswith("ppes_"):
        return "ppes_cctv"
    if n.startswith("lcsi_"):
        return "lcsi_crop"
    for pre, tag in (("websr3_", "websr3"), ("websr2_", "websr2"), ("webs_", "webs_r1"), ("up0_", "up0")):
        if n.startswith(pre):
            return tag
    if _re.match(r"^[gG]\d", n):
        return "g_digit"
    if _re.match(r"^[MFZz]\d", n):
        return "MFZ"
    if n.startswith("sgv"):
        return "sgv"
    if n.startswith("gozluk"):
        return "gozluk"
    if _re.match(r"^\d", n):
        return "digit"
    return "other"

LAB = ROOT / "data" / "lab_ppe"
PRE = ROOT / "data" / "_prelabel" / "person_backfill"
CFCSV = PRE / "person_backfill.csv"
WEBS = "data/raw/web_lab_scene/labels_5cls"


def imsize(p: Path):
    img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    return (img.shape[1], img.shape[0]) if img is not None else (0, 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--conf-min", type=float, default=0.40)
    ap.add_argument("--tag", default="p0_a")
    ap.add_argument("--max-side", type=int, default=1600)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    # 网图批（已有精修 person）排除
    webs = {Path(f).stem for f in Path(ROOT / WEBS).glob("*.txt")}
    # COCO conf 表
    conf = {}
    for r in csv.DictReader(open(CFCSV, encoding="utf-8-sig")):
        if r["confs"] and r["split"] == args.split:
            conf[r["image"][:-4]] = [float(x) for x in r["confs"].split(";")]


    items = []          # (stem, ip, n_ppe)
    for ip in sorted((LAB / "images" / args.split).glob("*")):
        stem = ip.stem
        if stem in webs:
            continue
        lf = LAB / "labels" / args.split / (stem + ".txt")
        rows = [l.split() for l in lf.read_text(encoding="utf-8").splitlines() if l.strip()] if lf.exists() else []
        items.append((stem, ip, sum(1 for r in rows if r and int(r[0]) >= 1)))
    p0 = sorted([x for x in items if x[2] > 0], key=lambda x: (fam_of(x[0]), x[0]))
    p1 = sorted([x for x in items if x[2] == 0], key=lambda x: (fam_of(x[0]), x[0]))
    queue = p0 + p1
    print(f"[{args.split}] 存量图 {len(items)}（P0 有PPE框 {len(p0)} / P1 空标签 {len(p1)}），总队列 {len(queue)}")
    print(f"  队列位置 {args.start}~{args.start + args.limit - 1}（P0 至位置 {len(p0) - 1}）")
    batch = queue[args.start:args.start + args.limit]
    if not batch:
        print("  空批")
        return 0

    out_dir = ROOT / "outputs" / f"person_review_{args.tag}"
    ws = ROOT / "data" / "_prelabel" / f"person_{args.tag}" / "labels"
    view = out_dir / "_view"
    ws.mkdir(parents=True, exist_ok=True)
    view.mkdir(parents=True, exist_ok=True)

    imgs = []
    index = [f"# person 补标人工审核 · 批 {args.tag} · split={args.split} · 队列 {args.start}+{len(batch)}",
             "# 图上黄框=person 候选（COCO conf≥%.2f，未确认）+ 现有 PPE 框" % args.conf_min,
             "# 操作：删错框(×)、拖动/缩放、空白拖拽新建；导出贴回 AI；整批没问题直接回「OK」",
             "# 编号\t文件\t族\t现有框\t候选框"]
    for i, (stem, ip, n_ppe) in enumerate(batch):
        w, h = imsize(ip)
        vp, vw, vh = ip, w, h
        if max(w, h) > args.max_side:
            s = args.max_side / max(w, h)
            vw, vh = max(1, int(w * s)), max(1, int(h * s))
            vp = view / (stem + ".jpg")
            if not vp.exists():
                im = cv2.imdecode(np.fromfile(str(ip), dtype=np.uint8), cv2.IMREAD_COLOR)
                cv2.imencode(".jpg", cv2.resize(im, (vw, vh), interpolation=cv2.INTER_AREA),
                             [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tofile(str(vp))
        # 合成标签：现有 + 候选
        cur = (LAB / "labels" / args.split / (stem + ".txt"))
        rows = [l.split() for l in cur.read_text(encoding="utf-8").splitlines() if l.strip()] if cur.exists() else []
        cands = []
        cf = conf.get(stem, [])
        pf = PRE / "labels" / args.split / (stem + ".txt")
        if pf.exists():
            plines = [l.split() for l in pf.read_text(encoding="utf-8").splitlines() if l.strip()]
            for j, r in enumerate(plines):
                if j < len(cf) and cf[j] >= args.conf_min:
                    cands.append(["0"] + [f"{float(v):.6f}" for v in r[1:5]])
        merged = [[int(r[0])] + [round(float(v), 6) for v in r[1:5]] for r in rows]
        merged += [[int(r[0])] + [round(float(v), 6) for v in r[1:5]] for r in cands]
        (ws / (stem + ".txt")).write_text(
            "\n".join(" ".join(str(v) for v in r) for r in merged) + ("\n" if merged else ""),
            encoding="utf-8", newline="\n")
        imgs.append({"no": i + 1, "name": ip.name,
                     "path": "file:///" + str(vp).replace("\\", "/"),
                     "nw": vw, "nh": vh, "boxes": merged})
        index.append(f"{i + 1}\t{ip.name}\t{fam_of(stem)}\t{n_ppe}\t{len(cands)}")

    html = (HTML.replace("__TITLE__", f"person 补标 · 批 {args.tag} · {len(imgs)} 张")
            .replace("__DATA__", json.dumps({"imgs": imgs, "focus": 0}, ensure_ascii=False)))
    (out_dir / "editor.html").write_text(html, encoding="utf-8")
    (out_dir / "index.txt").write_text("\n".join(index) + "\n", encoding="utf-8")
    print(f"  editor.html → {out_dir / 'editor.html'}")
    print(f"  工作区标签   → {ws}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
