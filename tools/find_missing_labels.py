# -*- coding: utf-8 -*-
"""底座漏标复查 —— 模型推理 vs 人工标签，找「高置信但无标签」的漏标目标。

背景：来源域手术 + 多轮清洗后，保留池里仍有原口径未标注的目标（尤其 mask、远景白大褂）。
本脚本对 train/valid 全量图推理（labppe_v8s），预测框与人工标签比对：
同类 IoU < 阈值 且 conf 足够高 → 漏标候选（补标可零采集成本增加有效框）。

安全约束：
- **只读数据集**：候选写 outputs/missing_labels_20261008/，不改任何标签
- 幂等续跑：结果 CSV 逐张追加，重跑自动跳过已处理图
- --limit 控制单轮张数（防环境回收），配 bash 循环续跑

用法:
    PY=.venv/Scripts/python.exe
    $PY tools/find_missing_labels.py --limit 300          # 分段跑
    $PY tools/find_missing_labels.py --report             # 只汇总+出图版（全部跑完后）
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "lab_ppe"
OUT = ROOT / "outputs" / "missing_labels_20261008"
CSV_PATH = OUT / "scan_progress.csv"
CAND_PATH = OUT / "candidates.csv"
NAMES = {0: "mask", 1: "gloves", 2: "lab_coat", 3: "goggles"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PAD = 10
BAR_H = 34


def imread_u(p: pathlib.Path):
    buf = np.fromfile(str(p), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def imwrite_u(p: pathlib.Path, img) -> bool:
    ok, buf = cv2.imencode(p.suffix, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if ok:
        buf.tofile(str(p))
    return bool(ok)


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def load_done() -> set[str]:
    if not CSV_PATH.exists():
        return set()
    with open(CSV_PATH, encoding="utf-8") as f:
        return {r["key"] for r in csv.DictReader(f)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "ai" / "model" / "labppe_v8s.pt"))
    ap.add_argument("--conf", type=float, default=0.45, help="候选漏标的置信度下限")
    ap.add_argument("--iou", type=float, default=0.30, help="与已有同类标签的匹配阈值")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--splits", default="train,valid")
    ap.add_argument("--limit", type=int, default=0, help="本轮最多处理多少张（0=不限）")
    ap.add_argument("--report", action="store_true", help="只汇总+出图版")
    ap.add_argument("--exclude-substr", default="8533991",
                    help="report 时排除文件名含此子串的候选（默认排除已判 coverall 口径的 8533991；传空=不排除）")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if args.report:
        return report(args)

    from ultralytics import YOLO
    model = YOLO(args.weights)

    done = load_done()
    jobs = []
    for sp in args.splits.split(","):
        idir = DATASET / "images" / sp
        for p in sorted(idir.iterdir()):
            if p.suffix.lower() in IMG_EXT and f"{sp}/{p.name}" not in done:
                jobs.append((sp, p))
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"[任务] 待扫 {len(jobs)} 张（已完成 {len(done)}）")
    if not jobs:
        print("[完成] 全部已扫。用 --report 出图版。")
        return 0

    new_file = not CSV_PATH.exists()
    with open(CSV_PATH, "a", encoding="utf-8", newline="") as fcsv, \
         open(CAND_PATH, "a", encoding="utf-8", newline="") as fcand:
        w = csv.writer(fcsv)
        if new_file:
            w.writerow(["key", "n_pred", "n_cand"])
        wc = csv.writer(fcand)
        if not CAND_PATH.exists() or CAND_PATH.stat().st_size == 0:
            wc.writerow(["split", "file", "cid", "cid_name", "conf", "x1", "y1", "x2", "y2"])
        n_cand_total = 0
        for i, (sp, ip) in enumerate(jobs, 1):
            img = imread_u(ip)
            if img is None:
                w.writerow([f"{sp}/{ip.name}", 0, 0])
                continue
            h, wd = img.shape[:2]
            res = model.predict(img, conf=args.conf, iou=0.5, imgsz=args.imgsz, device="cpu",
                                verbose=False)[0]
            preds = []
            for b in res.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                preds.append((int(b.cls.item()), float(b.conf.item()), (x1, y1, x2, y2)))
            # 人工标签
            gt = []
            lp = DATASET / "labels" / sp / (ip.stem + ".txt")
            if lp.exists():
                for ln in lp.read_text(encoding="utf-8").splitlines():
                    p = ln.split()
                    if len(p) >= 5 and p[0].lstrip("-").isdigit():
                        cx, cy, bw, bh = map(float, p[1:5])
                        gt.append((int(p[0]), (wd * (cx - bw / 2), h * (cy - bh / 2),
                                               wd * (cx + bw / 2), h * (cy + bh / 2))))
            cands = []
            for cid, cf, box in preds:
                matched = any(gcid == cid and iou(box, gbox) >= args.iou for gcid, gbox in gt)
                if not matched:
                    cands.append((cid, cf, box))
                    wc.writerow([sp, ip.name, cid, NAMES.get(cid, cid), f"{cf:.3f}",
                                 *[f"{v:.1f}" for v in box]])
            n_cand_total += len(cands)
            w.writerow([f"{sp}/{ip.name}", len(preds), len(cands)])
            fcsv.flush()
            fcand.flush()
            if i % 100 == 0:
                print(f"  进度 {i}/{len(jobs)}  累计候选 {n_cand_total}")
    print(f"[完成] 本轮 {len(jobs)} 张，候选 {n_cand_total}。剩 "
          f"{len(load_done())} 已扫。")
    return 0


def report(args) -> int:
    if not CAND_PATH.exists():
        print("[空] 无候选文件")
        return 1
    rows = list(csv.DictReader(open(CAND_PATH, encoding="utf-8")))
    if not rows:
        print("[空] 无候选")
        return 1
    excl = args.exclude_substr or ""
    if excl:
        kept = [r for r in rows if excl not in r["file"]]
        print(f"[口径过滤] 排除含 {excl!r} 的候选 {len(rows)-len(kept)} 个（已定口径，无需复核）")
        rows = kept
        if not rows:
            print("[空] 过滤后无候选")
            return 1
    # 汇总
    import collections
    by_cls = collections.Counter(r["cid_name"] for r in rows)
    by_split = collections.Counter(r["split"] for r in rows)
    confs = [float(r["conf"]) for r in rows]
    lines = [
        "# 底座漏标复查 —— 汇总",
        f"- 扫描范围: data/lab_ppe train+valid（详见 scan_progress.csv）",
        f"- 候选总数: {len(rows)}（conf>={args.conf} 且与同类标签 IoU<{args.iou}）",
        f"- 按类别: " + "  ".join(f"{k}={v}" for k, v in by_cls.most_common()),
        f"- 按split: " + "  ".join(f"{k}={v}" for k, v in by_split.most_common()),
        f"- conf 分布: min={min(confs):.2f} 中位={sorted(confs)[len(confs)//2]:.2f} max={max(confs):.2f}",
        "- 提醒: 高 conf 候选优先人工看；mask/lab_coat 候选价值最大；goggles 候选可能含普通眼镜（注意口径）",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # 图版：按 conf 降序取前 N 图（每图画 候选红实线 + 已有标签灰细线）
    by_img = {}
    for r in rows:
        by_img.setdefault((r["split"], r["file"]), []).append(r)
    ranked = sorted(by_img.items(),
                    key=lambda kv: max(float(x["conf"]) for x in kv[1]), reverse=True)
    top = ranked[: 120]
    cols, cell = 4, 470
    per = cols * 3
    sheets = 0
    index_lines = ["# 漏标候选图版对照表（红实线=候选漏标框，灰细线=已有标签）",
                   "# 复核口径: 红框里的东西该不该有标签? 回复编号即可（如: 1,3,5 补; 2 剔图）"]
    for s in range(0, len(top), per):
        chunk = top[s:s + per]
        sheets += 1
        sheet = np.full((3 * (cell + BAR_H + PAD * 2), cols * (cell + PAD * 2), 3), 255, np.uint8)
        for k, ((sp, fn), items) in enumerate(chunk):
            r, c = divmod(k, cols)
            img = imread_u(DATASET / "images" / sp / fn)
            if img is None:
                continue
            h, w = img.shape[:2]
            # 已有标签灰线
            lp = DATASET / "labels" / sp / (pathlib.Path(fn).stem + ".txt")
            if lp.exists():
                for ln in lp.read_text(encoding="utf-8").splitlines():
                    p = ln.split()
                    if len(p) < 5:
                        continue
                    cx, cy, bw, bh = map(float, p[1:5])
                    cv2.rectangle(img, (int(w * (cx - bw / 2)), int(h * (cy - bh / 2))),
                                  (int(w * (cx + bw / 2)), int(h * (cy + bh / 2))),
                                  (150, 150, 150), 1)
            for r_ in items:
                x1, y1, x2, y2 = float(r_["x1"]), float(r_["y1"]), float(r_["x2"]), float(r_["y2"])
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                tag = f"{r_['cid_name']} {float(r_['conf']):.2f}"
                ty = max(16, int(y1) - 5)
                cv2.putText(img, tag, (int(x1), ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3)
                cv2.putText(img, tag, (int(x1), ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            hh, ww = img.shape[:2]
            sc = min(cell / ww, cell / hh)
            img = cv2.resize(img, (max(1, int(ww * sc)), max(1, int(hh * sc))))
            cellimg = np.full((cell + BAR_H + PAD * 2, cell + PAD * 2, 3), 245, np.uint8)
            cellimg[PAD:PAD + img.shape[0], PAD:PAD + img.shape[1]] = img
            idx = s + k + 1
            cv2.putText(cellimg, str(idx), (PAD + 4, PAD + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(cellimg, f"{sp}/{fn[:34]} ({len(items)}候选)", (PAD, cell + PAD + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
            sy, sx = r * (cell + BAR_H + PAD * 2), c * (cell + PAD * 2)
            sheet[sy:sy + cellimg.shape[0], sx:sx + cellimg.shape[1]] = cellimg
            index_lines.append(f"{idx:>4}\t{sp}/{fn}\t候选 {len(items)}")
        imwrite_u(OUT / f"sheet_{sheets:02d}.jpg", sheet)
        print(f"图版 {sheets:02d} → sheet_{sheets:02d}.jpg")
    (OUT / "index.txt").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    print(f"共 {sheets} 版 / {len(top)} 图（按最高 conf 排序）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
