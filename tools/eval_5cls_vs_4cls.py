# -*- coding: utf-8 -*-
"""5 类新模型 vs 4 类基线：dataset2 external_test 同口径对比。

复用 tools/eval_e3_dataset2.py 的口径（conf=0.35, imgsz=640, IoU@0.5, 11点 mAP@50）
与 outputs/e3_dataset2_preds.json 中已生成的预测，避免重复推理。

注意：5 类模型自带 person，因此 person 类也可直接评估（旧 4 类需借 SH17）。
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CONF = 0.35
IOU_THR = 0.5
SYS_CLASSES = ["person", "mask", "gloves", "lab_coat", "goggles"]
SYS_ID = {c: i for i, c in enumerate(SYS_CLASSES)}
DS2_GT_MAP = {0: "mask", 1: "goggles", 2: "person", 3: "gloves", 4: "lab_coat"}
DS2_ROOT = ROOT / "data" / "raw" / "dataset2"
PREDS_JSON = ROOT / "outputs" / "e3_dataset2_preds.json"


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / area_u if area_u > 0 else 0.0


def match_predictions(preds, gts, cls):
    p = [d for d in preds if d["cls"] == cls]
    g = [t for t in gts if t["cls"] == cls]
    tp = np.zeros(len(p))
    matched = np.zeros(len(g), dtype=bool)
    for idx in np.argsort([-d["conf"] for d in p]):
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
    s = np.argsort(-np.asarray(conf))
    tp_s, fp_s = np.asarray(tp)[s], np.asarray(fp)[s]
    cum_tp, cum_fp = np.cumsum(tp_s), np.cumsum(fp_s)
    rec = cum_tp / n_gt
    prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
    return sum((np.max(prec[np.where(rec >= t)[0]]) if np.any(rec >= t) else 0.0)
               for t in np.linspace(0, 1, 11)) / 11.0


def load_gt():
    gts, n_img = [], 0
    all_imgs = set()
    for split in ("train", "valid", "test"):
        img_dir = DS2_ROOT / split / "images"
        lbl_dir = DS2_ROOT / split / "labels"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            n_img += 1
            all_imgs.add(img_path.name)
            txt = lbl_dir / (img_path.stem + ".txt")
            if not txt.exists():
                continue
            for line in txt.read_text().strip().splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                sysc = DS2_GT_MAP.get(int(float(p[0])))
                if sysc is None:
                    continue
                cx, cy, bw, bh = (float(p[1])*w, float(p[2])*h,
                                  float(p[3])*w, float(p[4])*h)
                gts.append({"img": img_path.name, "cls": SYS_ID[sysc],
                            "box": np.array([cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2])})
    return n_img, gts, all_imgs


def evaluate(preds, gts, classes, title):
    print("=" * 78)
    print(title)
    print("=" * 78)
    print(f"{'class':<10}{'n_gt':>6}{'TP':>6}{'FP':>6}{'FN':>6}"
          f"{'P':>8}{'R':>8}{'F1':>8}{'AP50':>8}")
    aps = []
    rows = {}
    for c in classes:
        tp, fp, fn = match_predictions(preds, gts, c)
        n_gt = int(tp.sum() + fn.sum())
        P = tp.sum() / max(tp.sum() + fp.sum(), 1e-9)
        R = tp.sum() / max(n_gt, 1e-9)
        F1 = 2 * P * R / max(P + R, 1e-9)
        a = ap11(tp, fp, [d["conf"] for d in preds if d["cls"] == c], n_gt)
        if not np.isnan(a):
            aps.append(a)
        rows[SYS_CLASSES[c]] = {"n_gt": n_gt, "TP": int(tp.sum()), "FP": int(fp.sum()),
                                "FN": int(fn.sum()), "P": round(float(P), 4),
                                "R": round(float(R), 4), "F1": round(float(F1), 4),
                                "AP50": None if np.isnan(a) else round(float(a), 4)}
        print(f"{SYS_CLASSES[c]:<10}{n_gt:>6}{int(tp.sum()):>6}{int(fp.sum()):>6}"
              f"{int(fn.sum()):>6}{P:>8.1%}{R:>8.1%}{F1:>8.1%}{a:>8.3f}")
    map50 = float(np.mean(aps)) if aps else float("nan")
    print(f"{'mAP@50':<10}{'':>30}{map50:>40.4f}")
    print()
    return {"title": title, "classes": rows, "mAP50": round(map50, 4)}


def main():
    n_img, gts, all_imgs = load_gt()
    print(f"dataset2 external_test: {n_img} 张, 系统5类 GT 框 {len(gts)} 个\n")
    from collections import Counter
    print("GT 分布:", dict(Counter(SYS_CLASSES[t["cls"]] for t in gts)), "\n")

    with io.open(PREDS_JSON, encoding="utf-8") as f:
        raw = json.load(f)

    ppe4 = [SYS_ID[c] for c in ("mask", "gloves", "lab_coat", "goggles")]
    all5 = [SYS_ID[c] for c in SYS_CLASSES]

    # 预测里 cls 已是系统类别索引（run_e3_retest 产出）
    preds_new = [{"cls": int(p["cls"]), "conf": float(p["conf"]),
                  "box": np.array(p["box"], dtype=float)} for p in raw["lab_ppe"]]
    preds_sh17 = [{"cls": int(p["cls"]), "conf": float(p["conf"]),
                   "box": np.array(p["box"], dtype=float)} for p in raw["sh17"]]

    r1 = evaluate(preds_new, gts, ppe4,
                  "【新】5类模型 - dataset2 external_test（4 PPE 类，与旧基线同口径）")
    r2 = evaluate(preds_new, gts, all5,
                  "【新】5类模型 - dataset2 external_test（含 person，5 类全量）")
    r3 = evaluate(preds_sh17, gts, [SYS_ID["person"]],
                  "【对照】SH17 - dataset2（person 参考）")

    # 4 类历史基线（来自 E3 复测记录）
    base = {"mask": {"F1": 0.711, "AP50": 0.545},
            "gloves": {"F1": 0.649, "AP50": 0.449},
            "lab_coat": {"F1": 0.891, "AP50": 0.901},
            "goggles": {"F1": 0.221, "AP50": 0.160}}
    print("=" * 78)
    print("与 4 类基线对比（同口径 dataset2 external_test）")
    print("=" * 78)
    print(f"{'class':<10}{'旧F1':>9}{'新F1':>9}{'ΔF1':>9}"
          f"{'旧AP50':>10}{'新AP50':>10}{'ΔAP50':>10}")
    for c in ("mask", "gloves", "lab_coat", "goggles"):
        o, n = base[c], r1["classes"][c]
        print(f"{c:<10}{o['F1']:>9.3f}{n['F1']:>9.3f}{n['F1']-o['F1']:>+9.3f}"
              f"{o['AP50']:>10.3f}{n['AP50']:>10.3f}{n['AP50']-o['AP50']:>+10.3f}")
    old_map = np.mean([base[c]["AP50"] for c in base])
    print(f"{'mAP@50':<10}{'':>9}{'':>9}{'':>9}{old_map:>10.4f}"
          f"{r1['mAP50']:>10.4f}{r1['mAP50']-old_map:>+10.4f}")

    out = {"n_img": n_img, "new_ppe4": r1, "new_all5": r2, "sh17_person": r3,
           "baseline_4cls": base, "baseline_mAP50": round(float(old_map), 4)}
    op = ROOT / "outputs" / "e5_5cls_vs_4cls_eval.json"
    with io.open(op, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n结果已保存: {op}")


if __name__ == "__main__":
    main()
