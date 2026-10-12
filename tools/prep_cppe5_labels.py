"""CPPE-5 可用池 -> 项目五类标签预生成。

- 读 outputs/cppe5_screen/picked_index.csv 的 468 张可用图
- COCO 标注映射：Coverall(1)/Face_Shield(2) 丢弃；Gloves(3)->2, Goggles(4)->4, Mask(5)->1
- person(0) 用 yolov8n_coco 检测补齐（CPPE-5 无 person 类）
- 输出 data/raw/cppe5/prep/labels/<stem>.txt（YOLO cxcywh 归一化）+ prep/person_det.json

用法：
    python tools/prep_cppe5_labels.py [--conf 0.40] [--imgsz 1280] [--limit N]
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CPPE = ROOT / "data" / "raw" / "cppe5"
PICKED = CPPE / "picked"
PREP = CPPE / "prep"
LBL_OUT = PREP / "labels"
PERSON_JSON = PREP / "person_det.json"
COCO = {sp: CPPE / "extracted" / "annotations" / f"{sp}.json" for sp in ("train", "test")}

# COCO category_id(1..5) -> 项目五类 id
MAP_PPE = {3: 2, 4: 4, 5: 1}   # Gloves->2, Goggles->4, Mask->1
DROP = {1, 2}                  # Coverall, Face_Shield

CLS_NAME = {0: "person", 1: "mask", 2: "gloves", 3: "lab_coat", 4: "goggles"}


def imread_u(p: Path):
    try:
        return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def clamp01(v: float) -> float:
    return float(max(0.0, min(1.0, float(v))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.40, help="person 检测置信度阈值")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张（调试用）")
    ap.add_argument("--no-person", action="store_true", help="跳过 person 检测（只出 PPE 框）")
    ap.add_argument("--model", default="ai/model/yolov8n_coco.pt")
    args = ap.parse_args()

    picked: list[str] = []
    with open(ROOT / "outputs/cppe5_screen/picked_index.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            picked.append(r["file"])
    if args.limit:
        picked = picked[: args.limit]
    print(f"可用图 {len(picked)} 张")

    # COCO 索引
    meta: dict[str, dict] = {}
    for sp, jp in COCO.items():
        d = json.load(open(jp, encoding="utf-8"))
        imgs = {im["id"]: im for im in d["images"]}
        anns = collections.defaultdict(list)
        for a in d["annotations"]:
            anns[a["image_id"]].append(a)
        for iid, im in imgs.items():
            meta[im["file_name"]] = {
                "split": sp, "w": im["width"], "h": im["height"],
                "anns": anns.get(iid, []),
            }

    LBL_OUT.mkdir(parents=True, exist_ok=True)

    model = None
    if not args.no_person:
        from ultralytics import YOLO
        model = YOLO(str(ROOT / args.model))
        print(f"person 检测模型: {args.model} conf>={args.conf} imgsz={args.imgsz}")

    stat = collections.Counter()
    person_stat = collections.Counter()
    person_records: dict[str, list] = {}
    skipped: list[str] = []
    sizes: list[tuple[int, int]] = []

    for k, fname in enumerate(picked, 1):
        m = meta.get(fname)
        if not m:
            skipped.append(fname)
            continue
        stem = Path(fname).stem
        W, H = int(m["w"]), int(m["h"])
        rows: list[tuple[int, float, float, float, float]] = []

        # --- PPE 框：COCO [x,y,w,h] 左上角绝对坐标 -> cxcywh 归一化 ---
        for a in m["anns"]:
            cid = a["category_id"]
            if cid in DROP:
                stat["drop_" + {1: "coverall", 2: "face_shield"}[cid]] += 1
                continue
            newc = MAP_PPE.get(cid)
            if newc is None:
                continue
            x, y, bw, bh = a["bbox"]
            if bw <= 0 or bh <= 0:
                stat["ppe_zero_size"] += 1
                continue
            cx = clamp01((x + bw / 2) / W)
            cy = clamp01((y + bh / 2) / H)
            nw = clamp01(bw / W)
            nh = clamp01(bh / H)
            rows.append((newc, cx, cy, nw, nh))
            stat["ppe_" + CLS_NAME[newc]] += 1

        # --- person 检测 ---
        if model is not None:
            img = imread_u(PICKED / fname)
            if img is None:
                skipped.append(fname)
                continue
            ih, iw = img.shape[:2]
            sizes.append((iw, ih))
            res = model.predict(img, classes=[0], conf=args.conf,
                                imgsz=args.imgsz, verbose=False)[0]
            pboxes = []
            if res.boxes is not None and len(res.boxes):
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                for (x1, y1, x2, y2), cf in zip(xyxy, confs):
                    cx = clamp01(((x1 + x2) / 2) / iw)
                    cy = clamp01(((y1 + y2) / 2) / ih)
                    nw = clamp01((x2 - x1) / iw)
                    nh = clamp01((y2 - y1) / ih)
                    if nw <= 0 or nh <= 0:
                        continue
                    pboxes.append((0, cx, cy, nw, nh))
                    person_records.setdefault(fname, []).append(
                        [round(float(cf), 3), round(cx, 6), round(cy, 6),
                         round(nw, 6), round(nh, 6)])
            rows.extend(pboxes)
            person_stat["imgs_with_person" if pboxes else "imgs_no_person"] += 1
            person_stat["person_boxes"] += len(pboxes)

        lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in rows]
        (LBL_OUT / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        stat["labeled_imgs"] += 1

        if k % 50 == 0:
            print(f"  {k}/{len(picked)} ...")

    if not args.no_person:
        PERSON_JSON.write_text(json.dumps(person_records, ensure_ascii=False, indent=0),
                               encoding="utf-8")

    print("\n=== PPE 映射 ===")
    for k in sorted(stat):
        print(f"  {k}: {stat[k]}")
    if not args.no_person:
        print("\n=== person 检测 ===")
        for k in sorted(person_stat):
            print(f"  {k}: {person_stat[k]}")
        if sizes:
            ws = sorted(s[0] for s in sizes)
            hs = sorted(s[1] for s in sizes)
            print(f"  尺寸中位 {ws[len(ws)//2]}x{hs[len(hs)//2]}  "
                  f"最大 {ws[-1]}x{hs[-1]}")
    if skipped:
        print(f"\n跳过 {len(skipped)}: {skipped[:8]}")
    print(f"\n标签输出 -> {LBL_OUT}")
    return 0


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    raise SystemExit(main())
