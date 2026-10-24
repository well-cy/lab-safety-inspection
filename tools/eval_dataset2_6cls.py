# -*- coding: utf-8 -*-
"""dataset2 独立外部测试（6 类现役模型）。

与 tools/eval_e3_dataset2.py 的差异：
  - 旧脚本用 SH17 + runs/e2/... 的双模型方案；本脚本评估单个现役 6 类模型
    ai/model/labppe_v8s.pt（同时输出 person + 全部 PPE）。
  - dataset2 原生无 coverall 类，故 coverall 不计入评估（无 GT 可比）。

口径：conf=0.35, imgsz=640, IoU@0.5, 11 点插值 mAP@50。
dataset2 原生 10 类 -> 系统类别 GT 映射：
  0 Mask->mask  1 Glasses->goggles  2 Person->person  3 Gloves->gloves  4 Lab Coat->lab_coat
  5~8 (No XXX 违规状态) 与 9 Shoes 丢弃

用法：
  NO_PROXY='*' .venv/Scripts/python.exe tools/eval_dataset2_6cls.py
"""
from __future__ import annotations

import io
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from ai.detector import PPEDetector

CONF = 0.35
IOU_THR = 0.5
SYS_CLASSES = ["person", "mask", "gloves", "lab_coat", "goggles", "coverall"]
SYS_ID = {c: i for i, c in enumerate(SYS_CLASSES)}
DS2_GT_MAP = {0: "mask", 1: "goggles", 2: "person", 3: "gloves", 4: "lab_coat"}
DS2_ROOT = ROOT / "data" / "raw" / "dataset2"
MODEL = ROOT / "ai" / "model" / "labppe_v8s.pt"
EVAL_CLASSES = ["person", "mask", "gloves", "lab_coat", "goggles"]


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area_u if area_u > 0 else 0.0


def match_predictions(preds, gts, cls):
    p = [d for d in preds if d["cls"] == cls]
    g = [t for t in gts if t["cls"] == cls]
    tp = np.zeros(len(p))
    matched = np.zeros(len(g), dtype=bool)
    order = np.argsort([-d["conf"] for d in p])
    for idx in order:
        best_j, best_v = -1, IOU_THR
        for j, gt in enumerate(g):
            if matched[j]:
                continue
            v = iou(p[idx]["box"], gt["box"])
            if v >= best_v:
                best_v, best_j = v, j
        if best_j >= 0:
            tp[idx] = 1
            matched[best_j] = True
    return tp, (1 - tp), (1 - matched).astype(int)


def ap11(tp, fp, conf, n_gt):
    if n_gt == 0:
        return np.nan
    sort = np.argsort(-np.asarray(conf))
    tp_s, fp_s = np.asarray(tp)[sort], np.asarray(fp)[sort]
    cum_tp, cum_fp = np.cumsum(tp_s), np.cumsum(fp_s)
    rec = cum_tp / n_gt
    prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        idx = np.where(rec >= t)[0]
        ap += (np.max(prec[idx]) if len(idx) else 0.0) / 11.0
    return ap


def load_gt():
    gts = []
    n_img = 0
    for split in ("train", "valid", "test"):
        img_dir = DS2_ROOT / split / "images"
        lbl_dir = DS2_ROOT / split / "labels"
        if not img_dir.is_dir():
            continue
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            n_img += 1
            txt = lbl_dir / (img_path.stem + ".txt")
            if not txt.exists():
                continue
            for line in txt.read_text().strip().splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                raw = int(float(p[0]))
                sysc = DS2_GT_MAP.get(raw)
                if sysc is None:
                    continue
                cx, cy, bw, bh = (float(p[1]) * w, float(p[2]) * h,
                                  float(p[3]) * w, float(p[4]) * h)
                gts.append({
                    "img": img_path.name, "cls": SYS_ID[sysc],
                    "box": np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]),
                })
    return n_img, gts


def run_model(det, names_wanted):
    preds = []
    for split in ("train", "valid", "test"):
        img_dir = DS2_ROOT / split / "images"
        if not img_dir.is_dir():
            continue
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            dets = det.detect(img)
            for d in dets:
                if d.cls_name in names_wanted:
                    preds.append({"img": img_path.name, "cls": SYS_ID[d.cls_name],
                                  "conf": d.conf, "box": np.array([d.x1, d.y1, d.x2, d.y2])})
    return preds


def evaluate(preds, gts, classes, title):
    print("\n===== %s =====" % title)
    print("%-10s%6s%6s%6s%6s%9s%9s%9s%9s" % (
        "类别", "GT", "TP", "FP", "FN", "P", "R", "F1", "AP@50"))
    aps = []
    for c in classes:
        tp, fp, fn = match_predictions(preds, gts, c)
        n_gt = int(tp.sum() + fn.sum())
        P = tp.sum() / max(tp.sum() + fp.sum(), 1e-9)
        R = tp.sum() / max(tp.sum() + fn.sum(), 1e-9)
        F1 = 2 * P * R / max(P + R, 1e-9)
        a = ap11(tp, fp, [d["conf"] for d in preds if d["cls"] == c], n_gt)
        if not np.isnan(a):
            aps.append(a)
        print("%-10s%6d%6d%6d%6d%9.1f%%%9.1f%%%9.1f%%%9.3f" % (
            SYS_CLASSES[c], n_gt, int(tp.sum()), int(fp.sum()), int(fn.sum()),
            P * 100, R * 100, F1 * 100, a))
    if aps:
        print("%-10s%42s%9.4f" % ("mAP@50", "", float(np.mean(aps))))
    return float(np.mean(aps)) if aps else float("nan")


def main():
    n_img, gts = load_gt()
    print("dataset2 external_test: %d 张, 系统类别 GT 框 %d 个" % (n_img, len(gts)))
    print("GT 分布:", dict(Counter(SYS_CLASSES[t["cls"]] for t in gts)))

    det = PPEDetector(str(MODEL), conf_threshold=CONF)
    print("\n模型:", MODEL)
    print("模型类别:", det.model.names)

    preds = run_model(det, set(EVAL_CLASSES))
    m_ppe = evaluate(preds,
                     gts,
                     [SYS_ID[c] for c in ("mask", "gloves", "lab_coat", "goggles")],
                     "现役 6 类模型 - dataset2（4 PPE 类，与旧基线同口径）")
    m_all = evaluate(preds, gts, [SYS_ID[c] for c in EVAL_CLASSES],
                     "现役 6 类模型 - dataset2（含 person 的 5 类）")

    print("\n对照（历史 4 类基线，dataset2 251 张）：")
    print("  mask F1 71.1%% / AP50 0.545")
    print("  gloves F1 64.9%% / AP50 0.449")
    print("  lab_coat F1 89.1%% / AP50 0.901")
    print("  goggles F1 22.1%% / AP50 0.160")
    print("  4 类 mAP@50 = 0.5137")
    print("\n现役 6 类模型 4 PPE mAP@50 = %.4f  (Δ = %+.4f)" % (
        m_ppe, m_ppe - 0.5137))

    out = ROOT / "outputs" / "e3_dataset2_6cls_preds.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump([{**p, "box": [round(float(v), 1) for v in p["box"]]} for p in preds],
                  f, ensure_ascii=False, indent=1)
    print("\n原始预测已保存: %s" % out)


if __name__ == "__main__":
    main()
