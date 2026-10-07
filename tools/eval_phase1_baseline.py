# -*- coding: utf-8 -*-
"""Phase-1 正式评测基线：单权重 × 单数据集 → 逐类 P / R / F1 / AP@50 + mAP50。

用途
----
1. 建立 **当前 SH17 权重** 在真实实验室场景测试集上的基线（原模型指标）；
2. 微调后用 **完全相同的测试集** 重跑（`--tag` 换成新权重名），得到新模型指标；
3. 两次报告可直接对比，输出给组长的实验结论表。

口径（与 E3 报告、tests/evaluate_model.py 一致，保证历史可比）
--------------------------------------------------------------
- 类别空间: person / mask / gloves / lab_coat / goggles
- conf=0.35, imgsz=640, IoU@0.5, 11 点插值 AP@50
- 预测侧: 走项目检测层 `ai.detector.PPEDetector`（SH17 原始 17 类经 CLASS_ALIAS 映射）
- GT 侧: 数据集原生类别名 → 系统类别（别名表见 GT_ALIAS）；违规状态类/非系统类丢弃

用法
----
    # SH17 基线（dataset2 external_test）
    python tools/eval_phase1_baseline.py --weights ai/model/sh17_yolov8s.pt \
        --dataset data/raw/dataset2 --tag baseline_sh17

    # 微调后同口径复评
    python tools/eval_phase1_baseline.py --weights ai/model/candidates/xxx.pt \
        --dataset data/raw/dataset2 --tag finetuned_v1

只读数据集，不写任何数据目录；结果写 outputs/phase1_eval/<tag>.{json,md}
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SYS_CLASSES = ["person", "mask", "gloves", "lab_coat", "goggles", "helmet"]
SYS_ID = {c: i for i, c in enumerate(SYS_CLASSES)}
EVAL_CLASSES = ["person", "mask", "gloves", "lab_coat", "goggles"]

# 数据集原生类别名 -> 系统类别；None = 丢弃
GT_ALIAS = {
    "person": "person",
    "mask": "mask", "face-mask-medical": "mask", "facemask": "mask", "face mask": "mask",
    "gloves": "gloves", "glove": "gloves",
    "lab coat": "lab_coat", "lab_coat": "lab_coat", "labcoat": "lab_coat",
    "glasses": "goggles", "goggles": "goggles", "safety_goggles": "goggles",
    "safety goggles": "goggles", "face-guard": "goggles", "face guard": "goggles",
    "helmet": "helmet",
    # 明确丢弃：违规状态类 / 非系统类
    "no gloves": None, "no_gloves": None, "no glasses": None, "no_glasses": None,
    "no lab coat": None, "no_lab_coat": None, "no mask": None, "no_mask": None,
    "shoes": None, "head": None, "face": None, "ear": None, "ear-mufs": None,
    "hands": None, "foot": None, "tools": None, "safety-vest": None,
    "medical-suit": None, "safety-suit": None,   # 连体防护服 ≠ 实验服，评估时不认
}


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_predictions(preds, gts, cls, iou_thr=0.5):
    p = sorted([d for d in preds if d["cls"] == cls], key=lambda d: -d["conf"])
    g = [t for t in gts if t["cls"] == cls]
    tp = np.zeros(len(p))
    matched = np.zeros(len(g), dtype=bool)
    for i, d in enumerate(p):
        best_j, best_v = -1, iou_thr
        for j, gt in enumerate(g):
            if matched[j]:
                continue
            v = iou(d["box"], gt["box"])
            if v >= best_v:
                best_v, best_j = v, j
        if best_j >= 0:
            tp[i] = 1
            matched[best_j] = True
    return tp, 1 - tp, (1 - matched).astype(int)


def ap11(tp, fp, conf, n_gt) -> float:
    """11 点插值 AP@50（与 E3 报告同口径）。"""
    if n_gt == 0:
        return float("nan")
    order = np.argsort(-np.asarray(conf))
    tp_s, fp_s = np.asarray(tp)[order], np.asarray(fp)[order]
    cum_tp, cum_fp = np.cumsum(tp_s), np.cumsum(fp_s)
    rec = cum_tp / n_gt
    prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        idx = np.where(rec >= t)[0]
        ap += (np.max(prec[idx]) if len(idx) else 0.0) / 11.0
    return float(ap)


def load_dataset(ds_root: Path):
    """读 YOLO 布局数据集 -> (图片列表, GT 框列表, 原生类别名, 图片尺寸表)"""
    yml = ds_root / "data.yaml"
    names = {}
    if yml.exists():
        meta = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
        raw_names = meta.get("names", {})
        names = ({int(k): str(v) for k, v in raw_names.items()}
                 if isinstance(raw_names, dict) else {i: str(v) for i, v in enumerate(raw_names)})

    items, gts, sizes, dropped = [], [], {}, Counter()
    for split in ("train", "valid", "test"):
        idir, ldir = ds_root / split / "images", ds_root / split / "labels"
        if not idir.is_dir():
            continue
        for ip in sorted(idir.iterdir()):
            if ip.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                continue
            img = cv2.imdecode(np.fromfile(str(ip), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            key = f"{split}/{ip.name}"
            items.append({"key": key, "path": ip, "split": split, "name": ip.name})
            sizes[key] = (w, h)
            lp = ldir / (ip.stem + ".txt")
            if not lp.exists():
                continue
            for line in lp.read_text(encoding="utf-8").splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                raw_id = int(float(p[0]))
                raw_name = names.get(raw_id, str(raw_id))
                sysc = GT_ALIAS.get(raw_name.strip().lower())
                if sysc is None:
                    dropped[raw_name] += 1
                    continue
                if sysc not in EVAL_CLASSES:
                    dropped[raw_name] += 1
                    continue
                cx, cy, bw, bh = (float(p[1]) * w, float(p[2]) * h,
                                  float(p[3]) * w, float(p[4]) * h)
                gts.append({"img": key, "cls": SYS_ID[sysc],
                            "box": np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])})
    return items, gts, names, sizes, dropped


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-1 基线评测（逐类 P/R/AP50 + mAP50）")
    ap.add_argument("--weights", required=True, help="待评权重 .pt")
    ap.add_argument("--dataset", required=True, help="YOLO 布局数据集根目录（含 data.yaml）")
    ap.add_argument("--tag", required=True, help="结果标记（输出文件名）")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--tile-recheck", action="store_true",
                    help="启用切片复检（系统生产链路默认关闭，保持与 E3 口径一致）")
    ap.add_argument("--outdir", default="outputs/phase1_eval")
    ap.add_argument("--limit", type=int, default=0, help="只评测前 N 张（冒烟测试用）")
    args = ap.parse_args()

    ds_root = (ROOT / args.dataset) if not Path(args.dataset).is_absolute() else Path(args.dataset)
    weights = (ROOT / args.weights) if not Path(args.weights).is_absolute() else Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"权重不存在: {weights}")
    if not (ds_root / "data.yaml").exists() and not any((ds_root / s / "images").is_dir()
                                                       for s in ("train", "valid", "test")):
        raise SystemExit(f"数据集不存在或布局不对: {ds_root}")

    items, gts, names, sizes, dropped = load_dataset(ds_root)
    if args.limit:
        items = items[:args.limit]
        keep = {it["key"] for it in items}
        gts = [g for g in gts if g["img"] in keep]
    with_gt = {g["img"] for g in gts}
    n_bg = sum(1 for it in items if it["key"] not in with_gt)
    gt_cnt = Counter(SYS_CLASSES[t["cls"]] for t in gts)
    print(f"数据集 {ds_root}：{len(items)} 图（其中空场景/无 GT 框 {n_bg} 张），"
          f"评估类 GT 框 {len(gts)} 个")
    print("  原生类别:", {names.get(k, k): v for k, v in
                       sorted(Counter(int(float(l.split()[0]))
                                      for it in items
                                      for l in ((ds_root / it["split"] / "labels" /
                                                 (Path(it["name"]).stem + ".txt")).read_text(
                                                    encoding="utf-8").splitlines()
                                                if (ds_root / it["split"] / "labels" /
                                                    (Path(it["name"]).stem + ".txt")).exists() else [])
                                      if len(l.split()) >= 5).items())})
    print("  GT 分布:", dict(gt_cnt))
    if dropped:
        print("  丢弃类别框数:", dict(dropped))

    from ai.detector import PPEDetector
    det = PPEDetector(str(weights), conf_threshold=args.conf, imgsz=args.imgsz)
    print(f"\n加载权重 {weights.name}，推理中（{len(items)} 图，tile_recheck="
          f"{args.tile_recheck}）...")

    preds, bg_fp = [], Counter()
    for i, it in enumerate(items, 1):
        img = cv2.imdecode(np.fromfile(str(it["path"]), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        for d in det.detect(img, tile_recheck=args.tile_recheck):
            if d.cls_name not in SYS_ID:
                continue
            preds.append({"img": it["key"], "cls": SYS_ID[d.cls_name], "conf": float(d.conf),
                          "box": np.array([d.x1, d.y1, d.x2, d.y2])})
            if it["key"] not in with_gt:
                bg_fp[SYS_CLASSES[SYS_ID[d.cls_name]]] += 1
        if i % 25 == 0:
            print(f"  已推理 {i}/{len(items)}")

    # 逐类指标
    rows, aps = [], []
    for c in EVAL_CLASSES:
        cid = SYS_ID[c]
        tp, fp, fn = match_predictions(preds, gts, cid, args.iou)
        n_gt = int(tp.sum() + fn.sum())
        P = tp.sum() / max(tp.sum() + fp.sum(), 1e-9)
        R = tp.sum() / max(tp.sum() + fn.sum(), 1e-9)
        F1 = 2 * P * R / max(P + R, 1e-9)
        a = ap11(tp, fp, [d["conf"] for d in preds if d["cls"] == cid], n_gt)
        if not np.isnan(a):
            aps.append(a)
        rows.append({"cls": c, "n_gt": n_gt, "tp": int(tp.sum()), "fp": int(fp.sum()),
                     "fn": int(fn.sum()), "precision": P, "recall": R, "f1": F1,
                     "ap50": a, "bg_fp": bg_fp.get(c, 0)})
    map50 = float(np.mean(aps)) if aps else float("nan")
    macro_p = float(np.mean([r["precision"] for r in rows]))
    macro_r = float(np.mean([r["recall"] for r in rows]))

    print("\n" + "=" * 78)
    print(f"{'类别':<10}{'GT':>6}{'TP':>6}{'FP':>6}{'FN':>6}{'P':>9}{'R':>9}{'F1':>9}{'AP50':>9}{'空场景FP':>9}")
    for r in rows:
        print(f"{r['cls']:<10}{r['n_gt']:>6}{r['tp']:>6}{r['fp']:>6}{r['fn']:>6}"
              f"{r['precision']:>9.1%}{r['recall']:>9.1%}{r['f1']:>9.1%}"
              f"{r['ap50']:>9.3f}{r['bg_fp']:>9}")
    print(f"\nmAP@50 = {map50:.4f}   macro-P = {macro_p:.1%}   macro-R = {macro_r:.1%}")

    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "tag": args.tag, "weights": str(weights), "dataset": str(ds_root),
        "conf": args.conf, "imgsz": args.imgsz, "iou": args.iou,
        "tile_recheck": args.tile_recheck, "n_images": len(items),
        "n_background_images": n_bg, "n_gt_boxes": len(gts),
        "gt_distribution": dict(gt_cnt), "mAP50": map50,
        "macro_precision": macro_p, "macro_recall": macro_r, "per_class": rows,
        "background_fp_by_class": dict(bg_fp),
    }
    (outdir / f"{args.tag}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    md = [f"# Phase-1 评测基线 —— `{args.tag}`", "",
          f"- 权重：`{weights.relative_to(ROOT) if weights.is_relative_to(ROOT) else weights}`",
          f"- 测试集：`{ds_root.relative_to(ROOT) if ds_root.is_relative_to(ROOT) else ds_root}`"
          f"（{len(items)} 图，其中无 GT 框的空场景 {n_bg} 张）",
          f"- 口径：conf={args.conf} / imgsz={args.imgsz} / IoU={args.iou} / 11 点 AP@50，"
          f"tile_recheck={args.tile_recheck}", "",
          "| 类别 | GT | TP | FP | FN | P | R | F1 | AP@50 | 空场景FP |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['cls']} | {r['n_gt']} | {r['tp']} | {r['fp']} | {r['fn']} | "
                  f"{r['precision']:.1%} | {r['recall']:.1%} | {r['f1']:.1%} | "
                  f"{r['ap50']:.3f} | {r['bg_fp']} |")
    md += ["", f"**mAP@50 = {map50:.4f}** ｜ macro-P = {macro_p:.1%} ｜ macro-R = {macro_r:.1%}", ""]
    (outdir / f"{args.tag}.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n结果已保存：{outdir / (args.tag + '.json')}\n            {outdir / (args.tag + '.md')}")


if __name__ == "__main__":
    main()
