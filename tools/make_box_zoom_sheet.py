#!/usr/bin/env python3
"""make_box_zoom_sheet.py — 标注框「框内放大」拼版（毒标签清洗专用）

为什么不直接用 make_contact_sheet.py：
  它出的是**整图缩略图**。监控视频抽帧里 goggles 框只有几十像素，
  缩到 340px 的格子里根本看不清是"护目镜"还是"普通眼镜"，
  人工判定等于靠猜。本工具**一个框一格**，格子左边放原图（带全部框、用于定位），
  右边放该框的**放大特写**（含框线），判断依据一目了然。

与 make_contact_sheet.py 的分工：
  - 判断"图里有没有这个目标" → make_contact_sheet.py（整图，看上下文）
  - 判断"这个框对不对/是什么" → 本工具（放大，看框内细节）

用法：
  # 毒标签专查：某个类别每个框放大一格
  python tools/make_box_zoom_sheet.py --list data/_goggles_clean_list.txt \
      --class goggles --limit 30 --out outputs/_goggles_clean_batch01_zoom

  # 不指定 --list 时扫描全库含该类别的图（配合 --one-per-family 降人工量）
  python tools/make_box_zoom_sheet.py --class goggles --one-per-family \
      --random 30 --out outputs/_goggles_zoom_sample

输出（都在 --out 目录下）：
  sheet_01.jpg …   图版（编号连续，跨版累计；一个目标框 = 一格）
  index.txt        编号 → 图片 / 框序号 对照表（回报问题时用编号）
  items.txt        本次拼入的图清单（去重后），可直接再喂给本工具或画框工具
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from draw_labels import draw, imread_u, load_names  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ROOTDIR = ROOT / "data" / "lab_ppe"
CLASS_ID = {"mask": 0, "gloves": 1, "lab_coat": 2, "goggles": 3}
BG = 250
PAD = 6
BAR_H = 36
MIN_CROP_PX = 96          # 框太小则至少裁这么大一块，保证周围有上下文


def fit_square(img: np.ndarray, size: int) -> np.ndarray:
    """等比缩放到 size×size 画布内（留白居中），不拉伸变形。"""
    h, w = img.shape[:2]
    s = size / max(h, w)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    r = cv2.resize(img, (nw, nh), interpolation=interp)
    canvas = np.full((size, size, 3), BG, np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = r
    return canvas


def clip_text(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "~"


def parse_targets(tokens: str, root: Path) -> set[int]:
    """类别名/数字 id → id 集合；同时读 root 下 data.yaml 里的类别名。"""
    name2id = dict(CLASS_ID)
    yml = root / "data.yaml"
    if yml.exists():
        try:
            import yaml
            for i, nm in enumerate(yaml.safe_load(yml.read_text(encoding="utf-8")).get("names", [])):
                name2id[str(nm).strip().lower()] = i
        except Exception:
            pass
    out = set()
    for tok in tokens.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        if tok.isdigit():
            out.add(int(tok))
        elif tok in name2id:
            out.add(name2id[tok])
        else:
            print(f"[警告] 无法识别的类别: {tok}")
    return out


def build_jobs(args, root: Path):
    """返回 [{img, lab, boxes:[(k, x1, y1, x2, y2)]}]，每项一个目标框一条记录。"""
    names = load_names(root, args.names)
    targets = parse_targets(args.class_id, root)
    print(f"[类别] 目标 {sorted(targets)}  映射={names}")

    rels = []
    if args.list:
        for ln in Path(args.list).read_text(encoding="utf-8").splitlines():
            rel = ln.split("\t")[0].strip()
            if rel and not rel.startswith("#"):
                rels.append(rel)
    else:
        for sp in ("train", "valid", "test"):
            idir = root / "images" / sp
            if idir.is_dir():
                rels += [f"{sp}/{p.name}" for p in sorted(idir.iterdir())
                         if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]

    recs, pending = [], []
    for rel in rels:
        sp, name = rel.split("/", 1)
        stem = Path(name).stem
        ip = root / "images" / sp / name
        lp = root / "labels" / sp / (stem + ".txt")
        if not ip.exists() or not lp.exists():
            continue
        h = w = None
        img = imread_u(ip)
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = []
        for line in lp.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            try:
                cid = int(float(p[0]))
            except ValueError:
                continue
            if cid not in targets:
                continue
            xc, yc, bw, bh = map(float, p[1:])
            x1, x2 = (xc - bw / 2) * w, (xc + bw / 2) * w
            y1, y2 = (yc - bh / 2) * h, (yc + bh / 2) * h
            boxes.append((x1, y1, x2, y2))
        if boxes:
            pending.append({"rel": rel, "sp": sp, "ip": ip, "lp": lp, "boxes": boxes})

    print(f"[候选] 含目标框的图 {len(pending)} 张 / 框 {sum(len(r['boxes']) for r in pending)} 个")

    if args.one_per_family:
        seen, kept = set(), []
        for r in pending:
            fam = r["ip"].name.split(".rf.")[0]
            if fam in seen:
                continue
            seen.add(fam)
            kept.append(r)
        pending = kept
        print(f"[每族取一] 剩 {len(pending)} 张")

    if args.random and args.random < len(pending):
        random.seed(args.seed)
        pending = random.sample(pending, args.random)
        print(f"[随机抽检] 抽出 {len(pending)} 张（seed={args.seed}）")

    for r in pending:
        for k, (x1, y1, x2, y2) in enumerate(r["boxes"]):
            recs.append({"rel": r["rel"], "sp": r["sp"], "ip": r["ip"], "lp": r["lp"],
                         "k": k, "box": (x1, y1, x2, y2), "names": names})

    # 阶段内限制条数（按"格"计数）
    if args.limit:
        recs = recs[: args.limit]
    return recs


def crop_box(img: np.ndarray, box, pad: float):
    """按框裁剪：框外扩 pad×max(框边长)，并保证最小窗口 MIN_CROP_PX。"""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    m = pad * max(bw, bh)
    cx1, cy1, cx2, cy2 = int(x1 - m), int(y1 - m), int(x2 + m), int(y2 + m)
    # 最小窗口：以框中心为锚点补足
    ccx, ccy = (x1 + x2) / 2, (y1 + y2) / 2
    if cx2 - cx1 < MIN_CROP_PX:
        cx1, cx2 = int(ccx - MIN_CROP_PX / 2), int(ccx + MIN_CROP_PX / 2)
    if cy2 - cy1 < MIN_CROP_PX:
        cy1, cy2 = int(ccy - MIN_CROP_PX / 2), int(ccy + MIN_CROP_PX / 2)
    cx1, cy1 = max(0, cx1), max(0, cy1)
    cx2, cy2 = min(w, cx2), min(h, cy2)
    if cx2 <= cx1 or cy2 <= cy1:
        return None, None
    crop = img[cy1:cy2, cx1:cx2].copy()
    rel = (int(x1 - cx1), int(y1 - cy1), int(x2 - cx1), int(y2 - cy1))  # 框在裁剪图中的位置
    return crop, rel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="", help="图片清单（每行 <split>/<文件名>[\\t标签]）；不传则扫描全库")
    ap.add_argument("--root", default=str(ROOTDIR), help="数据集根目录（默认 data/lab_ppe）")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--class", dest="class_id", required=True,
                    help="要看/放大的类别（mask/gloves/lab_coat/goggles 或数字 id，可逗号分隔）")
    ap.add_argument("--names", default="", help="手工类别映射，如 --names mask,gloves,lab_coat,goggles")
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=250, help="左侧原图定位格边长（像素）")
    ap.add_argument("--crop", type=int, default=310, help="右侧放大特写格边长（像素）")
    ap.add_argument("--pad", type=float, default=0.6, help="放大时框外扩比例（0.6=框边长 60%% 的留白）")
    ap.add_argument("--random", type=int, default=0, help="随机抽 N 张（按图计）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="本批最多出多少格（按框计，0=不限）")
    ap.add_argument("--one-per-family", action="store_true", help="每个原始照片家族只留 1 张")
    ap.add_argument("--start-index", type=int, default=1, help="编号起始（多批续编时用）")
    ap.add_argument("--max-sheets", type=int, default=0, help="最多出多少版（0=不限）")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    recs = build_jobs(args, root)
    if not recs:
        print("[空] 没有可拼的目标框，请检查 --list / --class / --root")
        return 1

    per = args.cols * args.rows
    cell_w = args.ctx + args.crop + PAD * 3
    cell_h = max(args.ctx, args.crop) + BAR_H + PAD * 2
    index_lines, item_lines, skipped = [], [], []
    idx = args.start_index
    sheets = 0

    for s in range(0, len(recs), per):
        if args.max_sheets and sheets >= args.max_sheets:
            break
        chunk = recs[s:s + per]
        sheets += 1
        sheet = np.full((args.rows * cell_h, args.cols * cell_w, 3), 255, np.uint8)
        for k, rec in enumerate(chunk):
            r, c = divmod(k, args.cols)
            img = imread_u(rec["ip"])
            if img is None:
                skipped.append(str(rec["ip"]))
                continue

            # 左：原图（画全部框，用于定位）
            ctx = draw(rec["ip"], rec["lp"], rec["names"], None)
            if ctx is None:
                skipped.append(str(rec["ip"]))
                continue
            crop, rel = crop_box(img, rec["box"], args.pad)
            if crop is None:
                skipped.append(str(rec["ip"]))
                continue
            cv2.rectangle(crop, rel[:2], rel[2:], (0, 0, 255), 2)  # 框线：BGR 红

            cell = np.full((cell_h, cell_w, 3), 255, np.uint8)
            y0, x0 = PAD, PAD
            # 左侧原图
            cell[y0:y0 + args.ctx, x0:x0 + args.ctx] = fit_square(ctx, args.ctx)
            cv2.rectangle(cell, (x0, y0), (x0 + args.ctx - 1, y0 + args.ctx - 1), (150, 150, 150), 1)
            cv2.putText(cell, "full", (x0 + 4, y0 + args.ctx - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (90, 90, 90), 1)
            # 右侧放大
            x1 = x0 + args.ctx + PAD
            cell[y0:y0 + args.crop, x1:x1 + args.crop] = fit_square(crop, args.crop)
            cv2.rectangle(cell, (x1, y0), (x1 + args.crop - 1, y0 + args.crop - 1), (150, 150, 150), 1)
            cv2.putText(cell, "zoom", (x1 + 4, y0 + args.crop - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (90, 90, 90), 1)

            # 编号徽标
            badge_w = 16 + 13 * len(str(idx))
            cv2.rectangle(cell, (x0, y0), (x0 + badge_w, y0 + 28), (0, 0, 0), -1)
            cv2.putText(cell, str(idx), (x0 + 8, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 2)

            # 底部文字：文件名 + 框序号
            ty = y0 + max(args.ctx, args.crop) + 22
            cv2.putText(cell, clip_text(rec["ip"].stem, 34), (x0, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (30, 30, 30), 1)
            cv2.putText(cell, f"box #{rec['k'] + 1}", (x1, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 230), 1)

            sy, sx = r * cell_h, c * cell_w
            sheet[sy:sy + cell_h, sx:sx + cell_w] = cell

            index_lines.append(f"{idx:>4}\t{rec['sp']}/{rec['ip'].name}\tbox #{rec['k'] + 1}")
            item_lines.append(f"{rec['sp']}/{rec['ip'].name}")
            idx += 1

        cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(
            str(out / f"sheet_{sheets:02d}.jpg"))
        print(f"  图版 {sheets:02d}: {len(chunk)} 格 → {out / f'sheet_{sheets:02d}.jpg'}")

    header = [
        f"# {args.out} 复查对照表 —— 编号 → 图片 / 框（{len(index_lines)} 格，共 {sheets} 版）",
        "# 每格 = 一个目标框：左=原图(定位)  右=该框放大(判定内容)",
        "# 口径：判「框里是不是护目镜」——不是则回报该编号（如：3、9 是眼镜）",
        "# 格式: 编号 <TAB> split/文件名 <TAB> 框序号",
    ]
    (out / "index.txt").write_text("\n".join(header + index_lines) + "\n", encoding="utf-8")
    (out / "items.txt").write_text("\n".join(dict.fromkeys(item_lines)) + "\n", encoding="utf-8")

    print(f"\n完成：{len(index_lines)} 格 / {sheets} 版 → {out}/")
    print(f"  对照表: {out / 'index.txt'}")
    if skipped:
        print(f"  [跳过] {len(skipped)} 格读图失败")
    return 0


if __name__ == "__main__":
    sys.exit(main())
