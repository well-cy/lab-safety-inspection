# -*- coding: utf-8 -*-
"""
模型评估脚本（验收阶段交付，不接入主系统，不影响运行）。

用途：把"项目组自采实验室场景测试集"的标注放进来后，自动计算
  - 逐类 Precision / Recall / F1
  - mAP@50（11 点插值法，逐类 AP 再取均值）
  - 混淆矩阵（GT 类别 x 预测类别）

数据格式约定（与系统业务类别一致，方便标注）：
  目录结构:
    <data>/
        images/   *.jpg|*.png
        labels/   同名 .txt   (YOLO 格式: cls x_center y_center w h，归一化)
  类别 id（系统类别）:
    0 person  1 mask  2 gloves  3 lab_coat  4 goggles  5 helmet
  说明: 自采数据建议只标注这 6 类（person 必标，PPE 按实际出现标注）。
        lab_coat 请只标注"真正的实验服/白大褂"，用于评估真实验服检测能力，
        而不依赖 SH17 的 medical-suit/safety-suit 映射。

用法:
  python tests/evaluate_model.py \
      --model ai/model/sh17_yolov8s.pt \
      --data data/self_collected \
      --conf 0.35 --iou 0.5

  若数据集为标准 YOLO 结构(images/train|val|test + labels)且想用 ultralytics
  官方的 mAP@50，可加 --data_yaml data/dataset.yaml 走 model.val()（需含测试集）。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ultralytics import YOLO
from ai.detector import CLASS_ALIAS  # 原始类别 -> 系统类别映射

# 系统类别 -> id（评估用）
SYS_CLASSES = ["person", "mask", "gloves", "lab_coat", "goggles", "helmet"]
SYS_ID = {c: i for i, c in enumerate(SYS_CLASSES)}


def iou(a, b):
    """a,b 均为 [x1,y1,x2,y2]"""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_u = area_a + area_b - inter
    return inter / area_u if area_u > 0 else 0.0


def parse_label(txt_path, w, h):
    """解析 YOLO txt(归一化 cx,cy,w,h) -> [{cls,x1,y1,x2,y2}]"""
    boxes = []
    if not txt_path.exists():
        return boxes
    for line in txt_path.read_text().strip().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cls = int(float(p[0]))
        cx, cy, bw, bh = (float(p[1]) * w, float(p[2]) * h,
                          float(p[3]) * w, float(p[4]) * h)
        x1, y1 = cx - bw / 2, cy - bh / 2
        x2, y2 = cx + bw / 2, cy + bh / 2
        if cls in SYS_ID.values():
            boxes.append({"cls": cls, "box": np.array([x1, y1, x2, y2])})
    return boxes


def match_predictions(preds, gts, cls, iou_thr):
    """对某个类别匹配 TP/FP/FN（one-to-one 贪心，优先高 IoU）"""
    p = [d for d in preds if d["cls"] == cls]
    g = [t for t in gts if t["cls"] == cls]
    tp = np.zeros(len(p))
    matched_g = np.zeros(len(g), dtype=bool)
    # 按置信度降序匹配
    order = np.argsort([-d["conf"] for d in p])
    for idx in order:
        best_j, best_iou = -1, iou_thr
        for j, gt in enumerate(g):
            if matched_g[j]:
                continue
            v = iou(p[idx]["box"], gt["box"])
            if v >= best_iou:
                best_iou, best_j = v, j
        if best_j >= 0:
            tp[idx] = 1
            matched_g[best_j] = True
    fp = 1 - tp
    fn = (1 - matched_g).astype(int)
    return tp, fp, fn


def ap11(tp, fp, conf, n_gts):
    """11 点插值法 AP@50"""
    if n_gts == 0:
        return np.nan
    sort = np.argsort(-conf)
    tp_s, fp_s = tp[sort], fp[sort]
    cum_tp = np.cumsum(tp_s); cum_fp = np.cumsum(fp_s)
    rec = cum_tp / n_gts
    prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        idx = np.where(rec >= t)[0]
        p = np.max(prec[idx]) if len(idx) else 0.0
        ap += p / 11.0
    return ap


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--model", default=str(ROOT / "ai/model/sh17_yolov8s.pt"))
    ap_.add_argument("--data", required=True, help="含 images/ 与 labels/ 的目录")
    ap_.add_argument("--conf", type=float, default=0.35)
    ap_.add_argument("--iou", type=float, default=0.5)
    ap_.add_argument("--data_yaml", default=None,
                     help="可选：标准 YOLO dataset.yaml，走 ultralytics 官方 mAP@50 验证")
    args = ap_.parse_args()

    img_dir = Path(args.data) / "images"
    lbl_dir = Path(args.data) / "labels"
    images = sorted(p for p in img_dir.iterdir()
                    if p.suffix.lower() in (".jpg", ".png", ".jpeg"))
    print(f"测试图片: {len(images)} 张, 目录 {args.data}")

    model = YOLO(args.model)

    # 官方 mAP（可选，需标准 YOLO 数据集结构）
    if args.data_yaml:
        print("== ultralytics 官方验证 ==")
        res = model.val(data=args.data_yaml, conf=args.conf, iou=args.iou,
                        project="outputs/eval", name="official", exist_ok=True)
        print(f"mAP@50={res.box.map50:.4f}  mAP@50-95={res.box.map:.4f}")

    # 自实现逐类 P/R/F1 + mAP@50 + 混淆矩阵
    preds_all, gts_all = [], []
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        r = model.predict(str(img_path), conf=args.conf, verbose=False)[0]
        for b in r.boxes:
            raw = model.names[int(b.cls[0])]
            sysc = CLASS_ALIAS.get(raw)
            if sysc in SYS_ID:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                preds_all.append({"cls": SYS_ID[sysc], "conf": float(b.conf[0]),
                                  "box": np.array([x1, y1, x2, y2])})
        gts_all.extend(parse_label(lbl_dir / (img_path.stem + ".txt"), w, h))

    n_gts = {c: sum(1 for t in gts_all if t["cls"] == c) for c in SYS_ID.values()}
    conf_mat = np.zeros((len(SYS_CLASSES) + 1, len(SYS_CLASSES) + 1), dtype=int)

    tp_by, fp_by, fn_by = {}, {}, {}
    for c in SYS_ID.values():
        tp, fp, fn = match_predictions(preds_all, gts_all, c, args.iou)
        tp_by[c], fp_by[c], fn_by[c] = tp, fp, fn
        # 混淆矩阵：行=GT类，列=预测类(+最后列为背景/漏检)
        g_mask = np.array([t["cls"] == c for t in gts_all])
        for i, d in enumerate(preds_all):
            if d["cls"] == c and tp[i] == 0:
                # 该预测是 FP，落到某个 GT(任意类)上则记到对应GT类(忽略背景)，否则记背景列
                matched = False
                for j, gt in enumerate(gts_all):
                    if iou(d["box"], gt["box"]) >= args.iou:
                        conf_mat[gt["cls"] + 1, c + 1] += 1
                        matched = True
                        break
                if not matched:
                    conf_mat[0, c + 1] += 1
        conf_mat[c + 1, len(SYS_CLASSES)] += int(np.sum(fn))  # 漏检

    print("\n== 逐类指标 (IoU@%.2f) ==" % args.iou)
    print(f"{'类别':<12}{'GT':>6}{'TP':>6}{'FP':>6}{'FN':>6}"
          f"{'P':>9}{'R':>9}{'F1':>9}")
    aps = []
    for c in SYS_ID.values():
        name = SYS_CLASSES[c]
        tp, fp, fn = tp_by[c], fp_by[c], fn_by[c]
        P = tp.sum() / max(tp.sum() + fp.sum(), 1e-9)
        R = tp.sum() / max(tp.sum() + fn.sum(), 1e-9)
        F1 = 2 * P * R / max(P + R, 1e-9)
        a = ap11(tp, fp, np.array([d["conf"] for d in preds_all
                                   if d["cls"] == c]), n_gts[c])
        if not np.isnan(a):
            aps.append(a)
        print(f"{name:<12}{n_gts[c]:>6}{int(tp.sum()):>6}{int(fp.sum()):>6}"
              f"{int(fn.sum()):>6}{P:>9.1%}{R:>9.1%}{F1:>9.1%}")

    map50 = float(np.nanmean(aps)) if aps else 0.0
    print(f"\nmAP@50 (11点插值, 系统6类): {map50:.4f}")

    print("\n== 混淆矩阵 (行=GT, 列=预测; 最后列=漏检; 首行=背景误检) ==")
    header = " " * 10 + "".join(f"{c:<10}" for c in SYS_CLASSES) + f"{'漏检':<10}"
    print(header)
    for gi, name in enumerate(["背景"] + SYS_CLASSES):
        row = " ".join(f"{x:<10}" for x in conf_mat[gi])
        print(f"{name:<10}{row}")


if __name__ == "__main__":
    main()
