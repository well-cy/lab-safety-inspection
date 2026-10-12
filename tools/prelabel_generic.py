# -*- coding: utf-8 -*-
"""通用预标注草稿（任意图片目录）——双模型：labppe(4类) + COCO(person)。

背景：B1 SOP 说"自动预标注做不到"，但仓库现有 ai/model/labppe_v8s.pt（4 类 PPE）
与 ai/model/yolov8n_coco.pt（person），两者拼起来正好覆盖 E6 业务 5 类口径：
    person=0  mask=1  gloves=2  lab_coat=3  goggles=4
（labppe 权重内部为 mask=0/gloves=1/lab_coat=2/goggles=3，本脚本自动平移 +1）

安全约束：
- 只读输入图片；标签草稿写 data/_prelabel/<tag>/labels/，**不碰数据集**
- 默认干跑，--apply 才推理与写盘
- 出带框图版供人工按编号复核

用法:
    PY=.venv/Scripts/python.exe
    $PY tools/prelabel_generic.py --src data/test/uploads --tag uploads_20261008 --apply
    $PY tools/prelabel_generic.py --src D:/B1_raw --tag b1_r1 --apply --limit 300
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# E6 业务 5 类
E6_NAMES = {0: "person", 1: "mask", 2: "gloves", 3: "lab_coat", 4: "goggles"}
# labppe 权重(cid) → E6 cid
LABPPE2E6 = {0: 1, 1: 2, 2: 3, 3: 4}
# BGR 配色：person 黄 / mask 橙 / gloves 绿 / lab_coat 紫 / goggles 蓝
COLORS = {0: (0, 220, 220), 1: (0, 165, 255), 2: (0, 200, 0), 3: (200, 0, 200), 4: (255, 120, 0)}
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


def fit(img, size: int):
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
    ap.add_argument("--src", required=True, help="图片目录（递归）")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ppe-weights", default=str(ROOT / "ai" / "model" / "labppe_v8s.pt"))
    ap.add_argument("--coco-weights", default=str(ROOT / "ai" / "model" / "yolov8n_coco.pt"))
    ap.add_argument("--conf", type=float, default=0.30, help="PPE 类置信度阈值")
    ap.add_argument("--person-conf", type=float, default=0.40)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--cell", type=int, default=470)
    ap.add_argument("--limit", type=int, default=0, help="本次最多处理多少张（防环境回收，0=不限）")
    ap.add_argument("--no-person", action="store_true", help="跳过 COCO person 推理")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    src = pathlib.Path(args.src)
    if not src.is_absolute():
        src = ROOT / src
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT)
    if args.limit:
        files = files[: args.limit]
    print(f"[输入] {src} → {len(files)} 张")

    out_dir = ROOT / "outputs" / f"_prelabel_{args.tag}"
    lab_dir = ROOT / "data" / "_prelabel" / args.tag / "labels"
    if not args.apply:
        print("[干跑] 加 --apply 才推理。计划输出:")
        print(f"  图版 {out_dir}/sheet_XX.jpg")
        print(f"  草稿 {lab_dir}/<name>.txt")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    ppe = YOLO(args.ppe_weights)
    coco = None if args.no_person else YOLO(args.coco_weights)
    print(f"[模型] PPE={pathlib.Path(args.ppe_weights).name}  person={'skip' if coco is None else pathlib.Path(args.coco_weights).name}")

    per = args.cols * args.rows
    cell_w, cell_h = args.cell + PAD * 2, args.cell + PAD * 2 + BAR_H
    index_lines, stats, sheets = [], {k: 0 for k in E6_NAMES}, 0

    for s in range(0, len(files), per):
        chunk = files[s:s + per]
        sheets += 1
        sheet = np.full((args.rows * cell_h, args.cols * cell_w, 3), 255, np.uint8)
        for k, ip in enumerate(chunk):
            r, c = divmod(k, args.cols)
            img = imread_u(ip)
            if img is None:
                continue
            h, w = img.shape[:2]
            dets = []
            res = ppe.predict(img, conf=args.conf, iou=0.5, imgsz=args.imgsz,
                              device=args.device, verbose=False)[0]
            for b in res.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                cid = LABPPE2E6.get(int(b.cls.item()))
                if cid is None:
                    continue
                dets.append((cid, float(b.conf.item()), x1, y1, x2, y2))
            if coco is not None:
                res2 = coco.predict(img, conf=args.person_conf, iou=0.6, imgsz=args.imgsz,
                                    device=args.device, verbose=False)[0]
                for b in res2.boxes:
                    if int(b.cls.item()) != 0:      # COCO 0=person
                        continue
                    x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                    dets.append((0, float(b.conf.item()), x1, y1, x2, y2))
            dets.sort(key=lambda x: -x[1])

            vis = img.copy()
            lines = []
            for cid, cf, x1, y1, x2, y2 in dets:
                col = COLORS.get(cid, (0, 200, 200))
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), col, 3)
                tag = f"{E6_NAMES[cid]} {cf:.2f}"
                ty = max(18, int(y1) - 6)
                cv2.putText(vis, tag, (int(x1), ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4)
                cv2.putText(vis, tag, (int(x1), ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                lines.append(f"{cid} {((x1+x2)/2)/w:.6f} {((y1+y2)/2)/h:.6f} {bw:.6f} {bh:.6f}")
                stats[cid] += 1
            (lab_dir / (ip.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""),
                                                     encoding="utf-8", newline="\n")
            cnt = {}
            for cid, *_ in dets:
                cnt[E6_NAMES[cid]] = cnt.get(E6_NAMES[cid], 0) + 1
            summary = " ".join(f"{k}{v}" for k, v in cnt.items()) or "无检出"
            vis = fit(vis, args.cell)
            gap = PAD if pathlib.Path(str(args.tag)).name else 10
            cell = np.full((cell_h, cell_w, 3), 245, np.uint8)
            cell[PAD:PAD + args.cell, PAD:PAD + args.cell] = vis
            idx = s + k + 1
            cv2.putText(cell, f"{idx}", (PAD + 6, PAD + 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)
            cv2.putText(cell, f"{ip.name[:38]} | {summary}", (PAD, PAD + args.cell + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
            sy, sx = r * cell_h, c * cell_w
            sheet[sy:sy + cell_h, sx:sx + cell_w] = cell
            index_lines.append(f"{idx:>4}\t{ip.parent.name}/{ip.name}\t{summary}")
        imwrite_u(out_dir / f"sheet_{sheets:02d}.jpg", sheet)
        print(f"  图版 {sheets:02d}: {len(chunk)} 格 → sheet_{sheets:02d}.jpg")

    total = sum(stats.values())
    (out_dir / "index.txt").write_text(
        "\n".join([f"# {args.tag} 预标注草稿对照表（PPE conf={args.conf} / person conf={args.person_conf} imgsz={args.imgsz}）",
                   "# 类别口径（E6 5 类）: 0=person 1=mask 2=gloves 3=lab_coat 4=goggles",
                   "# 复核口径：框位置/类别对不对？多了说删、少了说补、该剔图的直接说剔。回复编号即可",
                   f"# 合计 {len(files)} 张 / {total} 框: " + " ".join(f"{E6_NAMES[k]}={v}" for k, v in stats.items())]
                  + index_lines) + "\n",
        encoding="utf-8")
    print(f"\n完成: {len(files)} 张 / {total} 框  " + " ".join(f"{E6_NAMES[k]}={v}" for k, v in stats.items()))
    print(f"  图版 {out_dir}")
    print(f"  草稿 {lab_dir}（未并入数据集）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
