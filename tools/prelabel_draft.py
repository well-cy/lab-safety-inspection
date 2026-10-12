"""模型预标注草稿 —— 对「疑似漏标」的图跑一遍当前权重，产出可确认的带框图版 + 标签草稿。

场景：空标签图审计（audit_empty_labels.py）挑出的疑似漏标图，其中一部分是
目标域好图（实验室白大褂、防护服产品图等），值得补标而不是剔除。
人工从零画框成本高，因此先用模型标一遍 → 出带框图版供人工确认/增删 →
确认后再由调用方把草稿标签并入数据集。

安全约束：
- **不直接改数据集**：标签草稿写到 data/_prelabel/<tag>/，合并是另一步（人工确认后）
- 默认干跑（只打印计划）；--apply 才写文件
- 输出图版供人工核对，编号唯一对应

用法:
    PY=.venv/Scripts/python.exe
    $PY tools/prelabel_draft.py --list data/_backfill_candidates.txt --tag backfill_20261001 --apply
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "lab_ppe"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 与 draw_labels.py / visualize_b3.py 一致的 BGR 配色
COLORS = {0: (0, 165, 255), 1: (0, 200, 0), 2: (160, 0, 255), 3: (200, 0, 0)}
FALLBACK = (0, 200, 200)
BG = 245
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
    """等比缩放并居中放入 size×size 画布。"""
    h, w = img.shape[:2]
    s = size / max(h, w)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    r = cv2.resize(img, (nw, nh), interpolation=interp)
    canvas = np.full((size, size, 3), BG, np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = r
    return canvas, (x0, y0, s)


def main() -> int:
    ap = argparse.ArgumentParser(description="模型预标注草稿（供人工确认后并入数据集）")
    ap.add_argument("--list", required=True, help="清单（每行 split/文件名）")
    ap.add_argument("--tag", required=True, help="输出标记名")
    ap.add_argument("--weights", default=str(ROOT / "ai" / "model" / "labppe_v8s.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--cell", type=int, default=620, help="每格边长（像素）")
    ap.add_argument("--apply", action="store_true", help="写图版与标签草稿（默认只打印计划）")
    args = ap.parse_args()

    lp = pathlib.Path(args.list)
    rels = [l.split("\t")[0].strip().replace("\\", "/") for l in
            lp.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    items = []
    for rel in rels:
        if "/" not in rel:
            continue
        split, name = rel.split("/", 1)
        ip = DATASET / "images" / split / name
        if ip.exists():
            items.append((split, ip, name))

    print(f"[清单] {lp} → {len(items)} 张")
    if not items:
        return 1
    if not args.apply:
        print("[干跑] 不加 --apply 不跑模型、不写文件。计划处理:")
        for split, ip, name in items:
            print(f"    {split}/{name}")
        return 0

    wp = pathlib.Path(args.weights)
    from ultralytics import YOLO
    model = YOLO(str(wp))
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))

    out_dir = ROOT / "outputs" / f"_prelabel_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    lab_dir = ROOT / "data" / "_prelabel" / args.tag / "labels"
    lab_dir.mkdir(parents=True, exist_ok=True)

    per = args.cols * args.rows
    cell_w, cell_h = args.cell + PAD * 2, args.cell + PAD * 2 + BAR_H
    index_lines, draft_lines = [], []
    sheets = 0
    total_boxes = 0

    for s in range(0, len(items), per):
        chunk = items[s:s + per]
        sheets += 1
        sheet = np.full((args.rows * cell_h, args.cols * cell_w, 3), 255, np.uint8)
        for k, (split, ip, name) in enumerate(chunk):
            r, c = divmod(k, args.cols)
            img = imread_u(ip)
            if img is None:
                continue
            h, w = img.shape[:2]
            res = model.predict(str(ip), conf=args.conf, iou=0.5, imgsz=args.imgsz,
                                device=args.device, verbose=False)[0]
            boxes = []
            if res.boxes is not None and len(res.boxes):
                for b in res.boxes:
                    x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                    cid = int(b.cls.item())
                    cf = float(b.conf.item())
                    boxes.append((cid, cf, x1, y1, x2, y2))
            boxes.sort(key=lambda x: -x[1])
            total_boxes += len(boxes)

            vis = img.copy()
            lines = []
            for cid, cf, x1, y1, x2, y2 in boxes:
                col = COLORS.get(cid, FALLBACK)
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), col, 3)
                tag = f"{names.get(cid, cid)} {cf:.2f}"
                ty = max(18, int(y1) - 6)
                cv2.putText(vis, tag, (int(x1), ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 4)
                cv2.putText(vis, tag, (int(x1), ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                lines.append(f"{cid} {((x1+x2)/2)/w:.6f} {((y1+y2)/2)/h:.6f} {bw:.6f} {bh:.6f}")
            draft_lines.append((split, name, lines))

            sq, (ox, oy, sc) = fit(vis, args.cell)
            cell = np.full((cell_h, cell_w, 3), 255, np.uint8)
            cell[PAD:PAD + args.cell, PAD:PAD + args.cell] = sq
            idx = s + k + 1
            badge_w = 18 + 14 * len(str(idx))
            cv2.rectangle(cell, (PAD, PAD), (PAD + badge_w, PAD + 30), (0, 0, 0), -1)
            cv2.putText(cell, str(idx), (PAD + 9, PAD + 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (255, 255, 255), 2)
            cv2.putText(cell, f"{name[:40]}  |  {len(boxes)} 框", (PAD, PAD + args.cell + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
            sy, sx = r * cell_h, c * cell_w
            sheet[sy:sy + cell_h, sx:sx + cell_w] = cell
            index_lines.append(f"{idx:>4}\t{split}/{name}\t模型检出 {len(boxes)} 框")

        imwrite_u(out_dir / f"sheet_{sheets:02d}.jpg", sheet)
        print(f"  图版 {sheets:02d}: {len(chunk)} 格 → {out_dir / f'sheet_{sheets:02d}.jpg'}")

    for split, name, lines in draft_lines:
        d = lab_dir / split
        d.mkdir(parents=True, exist_ok=True)
        (d / (pathlib.Path(name).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""),
                                                            encoding="utf-8")

    (out_dir / "index.txt").write_text(
        "\n".join([f"# {args.tag} 预标注草稿对照表（conf={args.conf} imgsz={args.imgsz}）",
                   "# 编号 <TAB> split/文件名 <TAB> 模型检出",
                   "# 确认口径：框位置/类别对不对，多了删、少了说，回复编号即可"] + index_lines) + "\n",
        encoding="utf-8")

    print(f"\n完成：{len(items)} 张，共 {total_boxes} 框")
    print(f"  图版: {out_dir}")
    print(f"  标签草稿: {lab_dir}（未并入数据集，待人工确认）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
