# -*- coding: utf-8 -*-
"""E6 §3 整库 5 类化 · person 全库预标注（只读数据集，草稿写 data/_prelabel/）。

对 data/lab_ppe 全部图跑 COCO person 检测，产出:
  data/_prelabel/person_backfill/labels/<split>/<stem>.txt   person 候选框（class 0）
  data/_prelabel/person_backfill/person_backfill.csv          逐图清单（含 conf 列表与族）
  outputs/person_backfill/sheet_XX.jpg                        核查图版（现有框+person候选）

网图批 123 图（labels_5cls 已有精修 person）同样跑一遍用于对照，不直接采信。

用法:
  PY=.venv/Scripts/python.exe
  $PY tools/prelabel_person_all.py            # 干跑看计划
  $PY tools/prelabel_person_all.py --apply    # 推理写草稿
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

LAB = ROOT / "data" / "lab_ppe"
CLR4 = {0: (0, 165, 255), 1: (0, 200, 0), 2: (200, 0, 200), 3: (255, 120, 0)}
NAMES4 = {0: "mask", 1: "gloves", 2: "lab_coat", 3: "goggles"}
CLR_P = (0, 220, 220)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def fam_of(n: str) -> str:
    if n.startswith("ppes_"):
        return "ppes_cctv"
    if n.startswith("lcsi_"):
        return "lcsi_crop"
    for pre, tag in (("websr3_", "websr3"), ("websr2_", "websr2"), ("webs_", "webs_r1"), ("up0_", "up0")):
        if n.startswith(pre):
            return tag
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
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", default=str(ROOT / "ai" / "model" / "yolov8n_coco.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45, help="person NMS（收紧抑同人多框）")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--splits", default="train,valid,test")
    ap.add_argument("--tag", default="person_backfill")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--cell", type=int, default=430)
    ap.add_argument("--no-sheets", action="store_true")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    files = []
    for sp in args.splits.split(","):
        files += [(sp, Path(p)) for p in sorted(glob.glob(str(LAB / "images" / sp / "*")))
                  if Path(p).suffix.lower() in IMG_EXT]
    print(f"[输入] {len(files)} 张（split={args.splits}）")
    if not args.apply:
        print("[干跑] 加 --apply 才推理。草稿 → data/_prelabel/%s/  图版 → outputs/%s/" % (args.tag, args.tag))
        return 0

    lab_dir = ROOT / "data" / "_prelabel" / args.tag / "labels"
    out_dir = ROOT / "outputs" / args.tag
    lab_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for sp in args.splits.split(","):
        (lab_dir / sp).mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    coco = YOLO(args.coco)
    print(f"[模型] {Path(args.coco).name} conf={args.conf} iou={args.iou} imgsz={args.imgsz} device={args.device}")

    rows_csv = []
    per = args.cols * args.rows
    sheet_no = 0
    done = 0
    for s in range(0, len(files), per):
        chunk = files[s:s + per]
        sheet = None
        if not args.no_sheets:
            sheet_no += 1
            cw, ch = args.cell + 20, args.cell + 50
            sheet = np.full((args.rows * ch, args.cols * cw, 3), 255, np.uint8)
        for k, (sp, ip) in enumerate(chunk):
            img = imread_u(ip)
            if img is None:
                rows_csv.append([sp, ip.name, fam_of(ip.name), "UNREADABLE", ""])
                continue
            h, w = img.shape[:2]
            res = coco.predict(img, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                               device=args.device, verbose=False)[0]
            dets = []
            for b in res.boxes:
                if int(b.cls.item()) != 0:
                    continue
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                dets.append((float(b.conf.item()), x1, y1, x2, y2))
            dets.sort(key=lambda x: -x[0])
            # 草稿：仅 person 框，class 0
            lines = [f"0 {((x1+x2)/2)/w:.6f} {((y1+y2)/2)/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}"
                     for _, x1, y1, x2, y2 in dets]
            (lab_dir / sp / (ip.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""),
                                                           encoding="utf-8", newline="\n")
            rows_csv.append([sp, ip.name, fam_of(ip.name), len(dets),
                             ";".join(f"{d[0]:.3f}" for d in dets)])
            if sheet is not None:
                vis = img.copy()
                for cid, cx, cy, bw, bh in read_lbl(sp, ip.stem):
                    x1 = int((cx - bw / 2) * w); y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w); y2 = int((cy + bh / 2) * h)
                    col = CLR4.get(cid, (128, 128, 128))
                    cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
                    cv2.putText(vis, NAMES4.get(cid, str(cid)), (x1, max(14, y1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3)
                    cv2.putText(vis, NAMES4.get(cid, str(cid)), (x1, max(14, y1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)
                for cf, x1, y1, x2, y2 in dets:
                    cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), CLR_P, 3)
                    cv2.putText(vis, f"P{cf:.2f}", (int(x1), int(y1) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3)
                    cv2.putText(vis, f"P{cf:.2f}", (int(x1), int(y1) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                r, c = divmod(k, args.cols)
                vis = fit(vis, args.cell)
                cell = np.full((ch, cw, 3), 245, np.uint8)
                cell[10:10 + args.cell, 10:10 + args.cell] = vis
                cv2.putText(cell, f"{ip.name[:42]} [{fam_of(ip.name)}] P={len(dets)}",
                            (10, args.cell + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
                cv2.putText(cell, f"{sp}/{ip.stem[:40]}", (10, args.cell + 44),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)
                sheet[r * ch:r * ch + ch, c * cw:c * cw + cw] = cell
        if sheet is not None:
            imwrite_u(out_dir / f"sheet_{sheet_no:02d}.jpg", sheet)
        done += len(chunk)
        ndet = sum(int(r[3]) for r in rows_csv if r[3] not in ("UNREADABLE",))
        print(f"  进度 {done}/{len(files)}  person候选累计 {ndet}", flush=True)

    with open(ROOT / "data" / "_prelabel" / args.tag / "person_backfill.csv", "w",
              encoding="utf-8-sig", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["split", "image", "family", "n_person", "confs"])
        wcsv.writerows(rows_csv)
    print(f"完成: {len(rows_csv)} 图 → 草稿 {lab_dir}  清单 person_backfill.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
