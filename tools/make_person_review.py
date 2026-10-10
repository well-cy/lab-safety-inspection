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


def iou_cxcywh(a, b) -> float:
    """两个 YOLO 中心式框 (cls,cx,cy,w,h) 的 IoU。"""
    ax1, ay1 = a[1] - a[3] / 2, a[2] - a[4] / 2
    ax2, ay2 = a[1] + a[3] / 2, a[2] + a[4] / 2
    bx1, by1 = b[1] - b[3] / 2, b[2] - b[4] / 2
    bx2, by2 = b[1] + b[3] / 2, b[2] + b[4] / 2
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    u = a[3] * a[4] + b[3] * b[4] - ix * iy
    return (ix * iy) / u if u > 0 else 0.0


def parse_export(txt: str) -> dict:
    """解析编辑器导出：### stem → [(cls,cx,cy,w,h), ...]"""
    out, cur = {}, None
    for ln in txt.splitlines():
        ln = ln.strip()
        if ln.startswith("###"):
            cur = ln[3:].strip()
            out[cur] = []
        elif cur and len(ln.split()) == 5:
            p = ln.split()
            out[cur].append((int(p[0]), *(round(float(v), 6) for v in p[1:5])))
    return out


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
    ap.add_argument("--exclude-export", default="",
                    help="编辑器导出 txt；其中 ### 列出的图视为已完成，从队列剔除")
    ap.add_argument("--exclude-all", action="store_true",
                    help="自动排除 outputs/person_review_*/_her_export.txt 里所有已完成图（防批次回退）")
    ap.add_argument("--done-file", default="data/_person_review_done.txt",
                    help="已完成图清单（apply_person_review 落库时追加）")
    ap.add_argument("--stems", default="",
                    help="逗号分隔图名/stem：强制批次内容（忽略已完成剔除与队列切片，用于复核重做）")
    ap.add_argument("--note", default="", help="批次备注，写进 index 标题")
    ap.add_argument("--pending-vs-export", default="",
                    help="旧导出 txt：库内框若在其中消失（IoU<0.5）则在图上标「★待确认」（复核重做用）")
    args = ap.parse_args()

    excl: set[str] = set()
    sources = [Path(p) for p in args.exclude_export.split(",") if p.strip()]
    if args.exclude_all:
        sources += sorted(ROOT.glob("outputs/person_review_*/_her_export.txt"))
    for src in sources:
        for ln in src.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln.startswith("###"):
                excl.add(ln[3:].strip())
    # 已完成清单（apply_person_review 落库时追加；含"未改动即通过"的图）
    done_p = ROOT / args.done_file
    if args.done_file and done_p.exists():
        for ln in done_p.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                excl.add(ln.strip())
    print(f"  已完成剔除 {len(excl)} 图（导出 {len(sources)} 份 + {args.done_file}）")

    # 网图批（已有精修 person）排除
    webs = {Path(f).stem for f in Path(ROOT / WEBS).glob("*.txt")}
    # COCO conf 表
    conf = {}
    for r in csv.DictReader(open(CFCSV, encoding="utf-8-sig")):
        if r["confs"] and r["split"] == args.split:
            conf[r["image"][:-4]] = [float(x) for x in r["confs"].split(";")]

    # 旧导出（用于把"库内有、她上次删掉"的框标成 ★待确认）
    pend_exp = {}
    if args.pending_vs_export:
        pend_exp = parse_export(Path(args.pending_vs_export).read_text(encoding="utf-8"))
        print(f"  待确认标记源：{args.pending_vs_export}（{len(pend_exp)} 图）")


    items = []          # (stem, ip, n_ppe) 通过排除筛选的
    all_items = []      # 不受排除影响的全量（供 --stems 强制批次取用）
    for ip in sorted((LAB / "images" / args.split).glob("*")):
        stem = ip.stem
        lf = LAB / "labels" / args.split / (stem + ".txt")
        rows = [l.split() for l in lf.read_text(encoding="utf-8").splitlines() if l.strip()] if lf.exists() else []
        rec = (stem, ip, sum(1 for r in rows if r and int(r[0]) >= 1))
        all_items.append(rec)
        if stem in webs or stem in excl:
            continue
        items.append(rec)

    force = [s.strip() for s in args.stems.split(",") if s.strip()]
    if force:
        by_stem = {r[0]: r for r in all_items}
        by_name = {r[1].name: r for r in all_items}
        batch, missing = [], []
        for s in force:
            r = by_stem.get(s) or by_name.get(s) or by_stem.get(Path(s).stem)
            if r is None:
                missing.append(s)
            else:
                batch.append(r)
        print(f"  强制批次：指定 {len(force)} 图，命中 {len(batch)}"
              + (f"，缺失 {missing}" if missing else ""))
    else:
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
    scope = "复核重做" if force else f"队列 {args.start}+{len(batch)}"
    if args.note:
        scope += f" · {args.note}"
    index = [f"# person 补标人工审核 · 批 {args.tag} · split={args.split} · {scope}",
             "# 图上黄虚线框=person 候选（COCO conf≥%.2f，未确认）+ 现有 PPE 框" % args.conf_min,
             "# 操作：删错框(×)、拖动/缩放、空白拖拽新建；导出贴回 AI；整批没问题直接回「OK」",
             "# 框样式：实线=库内已有标注 · 黄虚线「候选」=COCO person 未确认 · ★待确认=上次复核中消失、本次被保留的框",
             "# 编号\t文件\t族\t现有框\t候选框\t待确认"]
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
        cand_rows = [[0] + [round(float(v), 6) for v in r[1:5]] for r in cands]
        # 去重：候选若与已有框（同为 person 且 IoU≥0.85 / 任意类且 IoU≥0.95）重叠，
        # 则丢弃——否则编辑器会叠画两个框，落库还会写出重复 person 行。
        _iou_cls = iou_cxcywh

        kept_cands, n_dropped = [], 0
        for c in cand_rows:
            if any((c[0] == m[0] and _iou_cls(c, m) >= 0.85) or _iou_cls(c, m) >= 0.95
                   for m in merged) \
               or any(_iou_cls(c, k) >= 0.85 for k in kept_cands):
                n_dropped += 1
                continue
            kept_cands.append(c)
        cand_rows = kept_cands
        if n_dropped:
            print(f"   {stem[:44]}: 丢弃重复候选 {n_dropped} 个")
        (ws / (stem + ".txt")).write_text(
            "\n".join(" ".join(str(v) for v in r) for r in merged + cand_rows)
            + ("\n" if merged or cand_rows else ""),
            encoding="utf-8", newline="\n")
        # 注意：库内/prelabel 标签 = YOLO cxcywh（中心式），编辑器内部 = 左上角 xywh。
        # 不转换会让每个框整体右下偏移半个框（表现为"所有框都是偏的"）。这里做 cxcywh → xywh。
        # 第 6 位 cand：1=COCO 候选（编辑器画虚线+「候选」前缀），0=库内原有标注。
        def _xywh(c, cx, cy, bw, bh, cand):
            x = min(max(cx - bw / 2, 0.0), 1.0)
            y = min(max(cy - bh / 2, 0.0), 1.0)
            return [c, round(x, 6), round(y, 6), round(bw, 6), round(bh, 6), cand]
        edit_boxes = [_xywh(*b, 0) for b in merged] + [_xywh(*b, 1) for b in cand_rows]
        # 库内框在旧导出里消失 → 标 ★待确认（cand=2），供她复核时优先看
        n_pend = 0
        if pend_exp and stem in pend_exp:
            ex = pend_exp[stem]
            for k, m in enumerate(merged):
                if not any(e[0] == m[0] and iou_cxcywh(e, m) >= 0.5 for e in ex):
                    edit_boxes[k][5] = 2
                    n_pend += 1
        imgs.append({"no": i + 1, "name": ip.name,
                     "path": "file:///" + str(vp).replace("\\", "/"),
                     "nw": vw, "nh": vh, "boxes": edit_boxes})
        index.append(f"{i + 1}\t{ip.name}\t{fam_of(stem)}\t{n_ppe}\t{len(cand_rows)}\t{n_pend}")

    html = (HTML.replace("__TITLE__", f"person 补标 · 批 {args.tag} · {len(imgs)} 张")
            .replace("__DATA__", json.dumps({"imgs": imgs, "focus": 0}, ensure_ascii=False)))
    (out_dir / "editor.html").write_text(html, encoding="utf-8")
    (out_dir / "index.txt").write_text("\n".join(index) + "\n", encoding="utf-8")
    print(f"  editor.html → {out_dir / 'editor.html'}")
    print(f"  工作区标签   → {ws}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
