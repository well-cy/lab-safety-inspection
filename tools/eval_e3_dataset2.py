# -*- coding: utf-8 -*-
"""E3-3 dataset2 独立外部测试（external_test）：SH17 vs LAB_PPE 双模型评估。

口径（与 tests/evaluate_model.py 及原验收一致）：
  - 系统类别空间: person / mask / gloves / lab_coat / goggles（helmet dataset2 无 GT，不计）
  - conf=0.35, imgsz=640, IoU@0.5, 11 点插值 mAP@50
  - SH17: ai/model/sh17_yolov8s.pt，经 CLASS_ALIAS 映射到系统类别（person + 4 PPE）
  - LAB_PPE: runs/e2/labppe_v8s/weights/best.pt，原生 4 类即系统类别（仅 4 PPE，person 按双模型方案仍由 SH17 负责）

dataset2 原生 10 类 -> 系统 5 类 GT 映射:
  0 Mask->mask  1 Glasses->goggles  2 Person->person  3 Gloves->gloves  4 Lab Coat->lab_coat
  5~8 (No XXX, 违规状态非可检测物体) 与 9 Shoes 丢弃

dataset2 仅作 external_test，本脚本只读不写 data/raw/dataset2。
"""
import sys
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai.detector import PPEDetector

# ---- 口径常量 ----
CONF = 0.35
IOU_THR = 0.5
SYS_CLASSES = ["person", "mask", "gloves", "lab_coat", "goggles"]
SYS_ID = {c: i for i, c in enumerate(SYS_CLASSES)}
DS2_GT_MAP = {0: "mask", 1: "goggles", 2: "person", 3: "gloves", 4: "lab_coat"}
DS2_ROOT = ROOT / "data" / "raw" / "dataset2"
SH17 = ROOT / "ai" / "model" / "sh17_yolov8s.pt"
LABPPE = ROOT / "runs" / "e2" / "labppe_v8s" / "weights" / "best.pt"


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
    """dataset2 251 张 -> 系统类别 GT 列表"""
    gts = []
    n_img = 0
    for split in ("train", "valid", "test"):
        img_dir = DS2_ROOT / split / "images"
        lbl_dir = DS2_ROOT / split / "labels"
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


def run_model(det: PPEDetector, gts_by_img):
    """模型推理 -> 系统类别预测列表"""
    preds = []
    for split in ("train", "valid", "test"):
        img_dir = DS2_ROOT / split / "images"
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            if img_path.name not in gts_by_img:
                continue
            dets = det.detect(str(img_path))
            for d in dets:
                if d.cls_name in SYS_ID:
                    preds.append({"img": img_path.name, "cls": SYS_ID[d.cls_name],
                                  "conf": d.conf, "box": np.array([d.x1, d.y1, d.x2, d.y2])})
    return preds


def evaluate(preds, gts, classes, title):
    print(f"\n===== {title} =====")
    print(f"{'类别':<10}{'GT':>6}{'TP':>6}{'FP':>6}{'FN':>6}"
          f"{'P':>8}{'R':>8}{'F1':>8}{'AP@50':>8}")
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
        print(f"{SYS_CLASSES[c]:<10}{n_gt:>6}{int(tp.sum()):>6}{int(fp.sum()):>6}"
              f"{int(fn.sum()):>6}{P:>8.1%}{R:>8.1%}{F1:>8.1%}{a:>8.3f}")
    print(f"{'mAP@50':<10}{'':>30}{np.mean(aps):>40.4f}" if aps else "无有效 AP")


def main():
    n_img, gts = load_gt()
    print(f"dataset2 external_test: {n_img} 张, 系统5类 GT 框 {len(gts)} 个")
    from collections import Counter
    cnt = Counter(SYS_CLASSES[t["cls"]] for t in gts)
    print("GT 分布:", dict(cnt))

    gts_by_img = {t["img"] for t in gts}
    # 图片无 GT 框也应有（背景图），补全
    for split in ("train", "valid", "test"):
        for p in sorted((DS2_ROOT / split / "images").iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                gts_by_img.add(p.name)

    print("\n加载 SH17 ...")
    sh17 = PPEDetector(str(SH17), conf_threshold=CONF)
    preds_sh17 = run_model(sh17, gts_by_img)

    print("加载 LAB_PPE best.pt ...")
    labppe = PPEDetector(str(LABPPE), conf_threshold=CONF)
    preds_lab = run_model(labppe, gts_by_img)

    # PPE 4 类（两模型均可评估）；person 仅 SH17（双模型方案中 person 由 SH17 负责）
    ppe4 = [SYS_ID[c] for c in ("mask", "gloves", "lab_coat", "goggles")]
    evaluate(preds_sh17, gts, ppe4, "SH17 原模型 - dataset2 external_test（4 PPE 类）")
    evaluate(preds_lab, gts, ppe4, "LAB_PPE best.pt - dataset2 external_test（4 PPE 类）")
    evaluate(preds_sh17, gts, [SYS_ID["person"]], "SH17 原模型 - dataset2（person，双模型方案保留项）")

    # 保存原始预测供复查
    import json
    out = ROOT / "outputs" / "e3_dataset2_preds.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "sh17": [{**p, "box": [round(float(v), 1) for v in p["box"]]} for p in preds_sh17],
            "lab_ppe": [{**p, "box": [round(float(v), 1) for v in p["box"]]} for p in preds_lab],
        }, f, ensure_ascii=False, indent=1)
    print(f"\n原始预测已保存: {out}")


if __name__ == "__main__":
    main()
