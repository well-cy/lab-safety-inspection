# -*- coding: utf-8 -*-
"""抽样评估 COCO person 检测在 lab_ppe 主库图上的行为。

目的：为 E6 §3「整库 5 类化 · person 补标」定阈值与人工审组织方式。
只读输入图片与现有标签，不写任何数据集目录。

输出:
  outputs/labppe_5cls_survey/report.txt   各族检出率/框数/置信度分布
  outputs/labppe_5cls_survey/sheet_XX.jpg 可视化图版（现有4类框 + person 候选框）
  outputs/labppe_5cls_survey/index.txt    逐图对照表

用法:
  PY=.venv/Scripts/python.exe
  $PY tools/survey_person.py --per-fam 25 --splits train,valid
"""
from __future__ import annotations

import argparse
import glob
import io
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

LAB = ROOT / "data" / "lab_ppe"
NAMES4 = {0: "mask", 1: "gloves", 2: "lab_coat", 3: "goggles"}
CLR4 = {0: (0, 165, 255), 1: (0, 200, 0), 2: (200, 0, 200), 3: (255, 120, 0)}
CLR_P = (0, 220, 220)          # person 候选 = 黄
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def fam_of(n: str) -> str:
    if n.startswith("ppes_"):
        return "ppes_cctv"
    if n.startswith("lcsi_"):
        return "lcsi_crop"
    if n.startswith("websr3_"):
        return "websr3"
    if n.startswith("websr2_"):
        return "websr2"
    if n.startswith("webs_"):
        return "webs_r1"
    if n.startswith("up0_"):
        return "up0"
    if re.match(r"^[gG]\d", n):
        return "g_digit"
    if re.match(r"^[MFZz]\d", n):
        return "MFZ"
    if n.startswith("sgv"):
        return "sgv"
    if n.startswith("gozluk"):
        return "gozluk"
    if re.match(r"^\d", n):
        return "digit"
    return "other"


def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_u(p, img):
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(p.suffix, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if ok:
        buf.tofile(str(p))
    return bool(ok)


def read_lbl(sp, stem):
    f = LAB / "labels" / sp / (stem + ".txt")
    if not f.exists():
        return []
    out = []
    for ln in f.read_text(encoding="utf-8").splitlines():
        t = ln.split()
        if len(t) == 5:
            out.append((int(t[0]), *[float(v) for v in t[1:]]))
    return out


def fit(img, size):
    h, w = img.shape[:2]
    s = min(size / w, size / h)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LANCZOS4)
    canvas = np.full((size, size, 3), 240, np.uint8)
    x0, y0 = (size - nw) // 2, (size - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = r
    return canvas, (x0, y0, nw, nh)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-fam", type=int, default=25)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--coco", default=str(ROOT / "ai" / "model" / "yolov8n_coco.pt"))
    ap.add_argument("--out", default="outputs/labppe_5cls_survey")
    ap.add_argument("--splits", default="train,valid")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--cell", type=int, default=430)
    args = ap.parse_args()

    # 分层抽样：每族等距取 per-fam 张
    byfam = defaultdict(list)
    for sp in args.splits.split(","):
        for p in sorted(glob.glob(str(LAB / "images" / sp / "*"))):
            if Path(p).suffix.lower() not in IMG_EXT:
                continue
            byfam[fam_of(Path(p).name)].append((sp, Path(p)))
    picks = []
    for f, lst in sorted(byfam.items()):
        n = len(lst)
        k = min(args.per_fam, n)
        idx = [round(i * (n - 1) / max(1, k - 1)) for i in range(k)] if k > 1 else [0]
        picks += [lst[i] for i in {*idx}]
    print(f"[抽样] 族数={len(byfam)} 总图={sum(len(v) for v in byfam.values())} 抽样={len(picks)}")

    from ultralytics import YOLO
    coco = YOLO(args.coco)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    stat = defaultdict(lambda: dict(sample=0, with_p=0, boxes=0, confs=[], ppbox0=0))
    index = []
    per = args.cols * args.rows
    sheet_no = 0
    for s in range(0, len(picks), per):
        chunk = picks[s:s + per]
        sheet_no += 1
        cw, ch = args.cell + 20, args.cell + 20 + 30
        sheet = np.full((args.rows * ch, args.cols * cw, 3), 255, np.uint8)
        for k, (sp, ip) in enumerate(chunk):
            img = imread_u(ip)
            if img is None:
                continue
            h, w = img.shape[:2]
            r, c = divmod(k, args.cols)
            dets = []
            res = coco.predict(img, conf=args.conf, iou=0.6, imgsz=args.imgsz,
                               device=args.device, verbose=False)[0]
            for b in res.boxes:
                if int(b.cls.item()) != 0:      # COCO 0 = person
                    continue
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                dets.append((float(b.conf.item()), x1, y1, x2, y2))
            cur = read_lbl(sp, ip.stem)
            n_person_lbl = sum(1 for t in cur if t[0] == 0)
            vis = img.copy()
            for cid, cx, cy, bw, bh in cur:      # 现有 4 类框
                x1 = int((cx - bw / 2) * w); y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w); y2 = int((cy + bh / 2) * h)
                col = CLR4.get(cid, (128, 128, 128))
                cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
                cv2.putText(vis, NAMES4.get(cid, str(cid)), (x1, max(14, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3)
                cv2.putText(vis, NAMES4.get(cid, str(cid)), (x1, max(14, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)
            for cf, x1, y1, x2, y2 in dets:      # person 候选
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), CLR_P, 3)
                tag = f"PERSON {cf:.2f}"
                cv2.putText(vis, tag, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 3)
                cv2.putText(vis, tag, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 0), 1)
            f = fam_of(ip.name)
            st = stat[f]
            st["sample"] += 1
            st["boxes"] += len(dets)
            st["confs"] += [d[0] for d in dets]
            if dets:
                st["with_p"] += 1
            if not cur:
                st["ppbox0"] += 1
            index.append(f"{sp}/{ip.name}\tfam={f}\tperson候选={len(dets)}\t现框={len(cur)}")
            vis, _ = fit(vis, args.cell)
            cell = np.full((ch, cw, 3), 245, np.uint8)
            cell[10:10 + args.cell, 10:10 + args.cell] = vis
            cv2.putText(cell, f"{ip.name[:40]} [{f}] P={len(dets)} lbl={len(cur)}",
                        (10, args.cell + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
            sheet[r * ch:r * ch + ch, c * cw:c * cw + cw] = cell
        imwrite_u(out_dir / f"sheet_{sheet_no:02d}.jpg", sheet)
        print(f"  图版 sheet_{sheet_no:02d}.jpg ({len(chunk)} 格)")

    lines = [f"# 抽样评估 COCO person（conf={args.conf} imgsz={args.imgsz}）",
             f"# 族数={len(byfam)} 抽样={len(picks)} 张", ""]
    lines.append(f'{"族":<12}{"抽样":>5}{"有P图":>7}{"占比":>7}{"P框":>6}{"框/图":>7}{"空标签":>7}{"均conf":>8}{"min":>6}{"max":>6}')
    tp = tw = tb = 0
    for f, d in sorted(stat.items(), key=lambda x: -x[1]["sample"]):
        n = d["sample"]
        c = d["confs"]
        lines.append(f'{f:<12}{n:>5}{d["with_p"]:>7}{d["with_p"]/n*100:>6.0f}%{d["boxes"]:>6}'
                     f'{d["boxes"]/n:>7.2f}{d["ppbox0"]:>7}'
                     f'{(sum(c)/len(c) if c else 0):>8.2f}{(min(c) if c else 0):>6.2f}{(max(c) if c else 0):>6.2f}')
        tp += d["boxes"]; tw += d["with_p"]; tb += n
    lines.append("")
    lines.append(f"合计 抽样={tb} 有P图={tw}({tw/max(1,tb)*100:.0f}%) P框={tp} 框/图={tp/max(1,tb):.2f}")
    (out_dir / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "index.txt").write_text("\n".join(index) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n输出 {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
