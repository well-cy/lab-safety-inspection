# -*- coding: utf-8 -*-
"""E4 效果速测：用 LAB_PPE 权重对指定图片目录推理，出可视化拼版 + 统计。

用途（改数据后快速看效果，不跑完整 E3 套件）：
  1) 对新入库的 web 图直接推理，看 lab_coat 跨域召回是否改善
  2) 对 dataset2 的真实实验室帧看漏检情况
  3) 对 B1 自采照片做预标注质检

用法：
  .venv/Scripts/python.exe tools/infer_check.py --weights runs/e4/labppe_v8s/weights/best.pt \
      --dir data/raw/web_lab_keep_r2/images --out outputs/infer_check/web_r2
  # 只看每类命中，不存图：加 --stat-only
  # 调整阈值：--conf 0.25

注意：
- 本机 GPU 为 MX450/2GB，默认 device=cpu（推理 209 张约 2-4 分钟）；有独显可 --device 0
- 类别固定 mask=0 / gloves=1 / lab_coat=2 / goggles=3（与 data/lab_ppe/data.yaml 一致）
- 只读输入图片，不写任何数据集目录
"""
from __future__ import annotations

import argparse
import collections
import statistics as st
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NAMES = ["mask", "gloves", "lab_coat", "goggles"]
CLR = {0: (255, 150, 40), 1: (0, 150, 255), 2: (90, 220, 60), 3: (200, 60, 220)}


def imread(p: Path):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite(p: Path, img) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(p.suffix, img)
    if ok:
        buf.tofile(str(p))


def main() -> int:
    ap = argparse.ArgumentParser(description="LAB_PPE 权重推理速测")
    ap.add_argument("--weights", required=True, help="best.pt 路径")
    ap.add_argument("--dir", required=True, help="图片目录（递归一层）")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "infer_check"), help="输出目录")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0, help="最多处理几张（0=全部）")
    ap.add_argument("--sheet", type=int, default=24, help="拼版每页格数")
    ap.add_argument("--stat-only", action="store_true", help="只统计，不存可视化")
    args = ap.parse_args()

    wp = Path(args.weights)
    if not wp.exists():
        print(f"[错误] 权重不存在: {wp}")
        print("  → 需向组长索取 runs/e2/labppe_v8s/weights/best.pt（或 E4 训练产物）")
        return 2
    src = Path(args.dir)
    if not src.exists():
        print(f"[错误] 图片目录不存在: {src}")
        return 2

    from ultralytics import YOLO
    model = YOLO(str(wp))
    print(f"模型: {wp}")
    print(f"类别: {model.names}")

    if len(model.names) != 4:
        cnt = collections.Counter(model.names.values()) if isinstance(model.names, dict) else None
        print(f"\n[错误] 该权重有 {len(model.names)} 个类别，不是 LAB_PPE 的 4 类"
              f"（mask/gloves/lab_coat/goggles）")
        print("  很可能是官方 COCO 权重或 SH17 权重，拿错了。请用 runs/*/labppe_*/weights/best.pt")
        if cnt and "person" in cnt:
            print("  （检测到 'person' 类 -> COCO/SH17 系列）")
        return 3

    imgs = sorted(p for p in src.iterdir()
                  if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
    if args.limit:
        imgs = imgs[: args.limit]
    print(f"待推理: {len(imgs)} 张  conf={args.conf} imgsz={args.imgsz} device={args.device}\n")

    out_dir = Path(args.out)
    stat_c = collections.Counter()       # 类别命中图数
    stat_b = collections.Counter()       # 类别框数
    areas = collections.defaultdict(list)
    no_det = []
    tiles = []

    for i, p in enumerate(imgs, 1):
        r = model.predict(str(p), conf=args.conf, iou=args.iou,
                          imgsz=args.imgsz, device=args.device, verbose=False)[0]
        img = imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        hits = collections.Counter()
        if r.boxes is not None and len(r.boxes):
            for b in r.boxes:
                c = int(b.cls.item())
                cf = float(b.conf.item())
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                hits[c] += 1
                stat_b[c] += 1
                areas[c].append(((x2 - x1) * (y2 - y1)) / (w * h))
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                              CLR.get(c, (0, 0, 255)), max(2, w // 300))
                cv2.putText(img, f"{NAMES[c]} {cf:.2f}" if c < 4 else f"c{c} {cf:.2f}",
                            (int(x1) + 3, max(14, int(y1) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, max(0.45, w / 1500),
                            CLR.get(c, (0, 0, 255)), max(1, w // 700))
        for c in hits:
            stat_c[c] += 1
        if not hits:
            no_det.append(p.name)
        if not args.stat_only:
            cv2.putText(img, p.name, (6, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.5, w / 1400), (255, 255, 255), max(1, w // 800))
            tw = 320
            tiles.append(cv2.resize(img, (tw, int(h * tw / w)), interpolation=cv2.INTER_AREA))
        if i % 25 == 0:
            print(f"  {i}/{len(imgs)}", flush=True)

    print("\n" + "=" * 60)
    print(f"推理完成: {len(imgs)} 张")
    print(f"{'类别':<10}{'命中图数':>9}{'框数':>8}{'面积中位':>10}")
    for c in range(4):
        am = st.median(areas[c]) if areas[c] else float("nan")
        print(f"{NAMES[c]:<10}{stat_c[c]:>9}{stat_b[c]:>8}{am:>10.4f}")
    extra = [c for c in stat_b if c >= 4]
    for c in extra:
        print(f"  [越界类 {c}] {stat_c[c]} 张 / {stat_b[c]} 框（权重类别数超出 4 类，检查模型）")
    print(f"零检测图: {len(no_det)} 张")
    if no_det[:10]:
        print(f"  例: {', '.join(no_det[:10])}")
    lab_hit = stat_c.get(2, 0)
    print(f"\n[lab_coat 关键指标] 命中 {lab_hit} / {len(imgs)} 张 = {lab_hit/max(1,len(imgs))*100:.1f}%")

    if not args.stat_only and tiles:
        per = args.sheet
        cols = 4 if per <= 24 else 8
        for pg in range(0, len(tiles), per):
            chunk = tiles[pg:pg + per]
            rows_n = (len(chunk) + cols - 1) // cols
            H = max(t.shape[0] for t in chunk)
            canvas = np.full((rows_n * (H + 8), cols * 324, 3), 24, np.uint8)
            for k, t in enumerate(chunk):
                rr, cc = divmod(k, cols)
                canvas[rr * (H + 8):rr * (H + 8) + t.shape[0],
                       cc * 324:cc * 324 + t.shape[1]] = t
            fp = out_dir / f"sheet_{pg // per + 1:02d}.jpg"
            imwrite(fp, canvas)
            print(f"  拼版 -> {fp}")
        print(f"\n拼版目录: {out_dir}（{out_dir}/sheet_*.jpg）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
