"""lab_coat 画框批复核拼图生成器（lc13 系）。

读 `outputs/labcoat_label_review_<tag>/` 下的 `_her_export.txt` + `_stem2rel.tsv`，
生成两张拼版：
  - `_review_kept_<N>.jpg`     : 保留图（画上她的框，按类别着色）
  - `_review_rejected_<N>.jpg` : 剔除图（从 `data/_rejected/<tag>_reject_<日期>/` 读原图）

用法：
  python tools/make_labcoat_review_sheet.py --tag lc13f
  python tools/make_labcoat_review_sheet.py --tag lc13f --reject-dir data/_rejected/lc13f_reject_20261011
"""
import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "data" / "raw" / "labcoat_imagenetcv" / "data"

CELL = 260
COLS = 6
IMG_NAME = {0: "person", 1: "mask", 2: "gloves", 3: "lab_coat", 4: "goggles"}
# BGR
COLORS = {0: (0, 200, 0), 1: (255, 120, 0), 2: (0, 180, 255), 3: (255, 0, 255), 4: (0, 0, 255)}


def imread_cn(p):
    """cv2 不能直接读中文路径，用 fromfile+imdecode。"""
    try:
        buf = np.fromfile(str(p), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def parse_export(exp_path):
    """返回 (kept: {stem: [(cls,cx,cy,w,h)...]}, rejected: [stem...])。

    剔除优先：某图出现 `!REJECT` 后，其已收集的框全部丢弃。
    """
    kept, rejected = {}, []
    cur, boxes, is_rej = None, [], False

    def flush():
        if cur is None:
            return
        if is_rej:
            rejected.append(cur)
        else:
            kept[cur] = boxes

    with open(exp_path, encoding="utf-8") as fh:
        for line in fh:
            s = line.rstrip("\n")
            if s.startswith("### "):
                flush()
                cur, boxes, is_rej = s[4:].strip(), [], False
            elif s.strip().startswith("!REJECT"):
                is_rej = True
            elif s.strip() and cur:
                parts = s.split()
                if len(parts) == 5:
                    try:
                        boxes.append(tuple(float(x) for x in parts))
                    except ValueError:
                        pass
    flush()
    return kept, rejected


def tile(img, caption, w=CELL, h=CELL, boxes=None):
    """把图缩放进 cell 并居中，画框 + 底部文件名。"""
    if img is None:
        img = np.full((h, w, 3), 225, np.uint8)
    else:
        H, W = img.shape[:2]
        if boxes:
            for (c, cx, cy, bw, bh) in boxes:
                x1, y1 = int((cx - bw / 2) * W), int((cy - bh / 2) * H)
                x2, y2 = int((cx + bw / 2) * W), int((cy + bh / 2) * H)
                col = COLORS.get(int(c), (255, 255, 255))
                cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
                lbl = IMG_NAME.get(int(c), str(int(c)))
                cv2.putText(img, lbl, (x1, max(12, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        s = min(w / W, h / H)
        nw, nh = max(1, int(W * s)), max(1, int(H * s))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w, 3), 245, np.uint8)
    if img.shape[0] <= h and img.shape[1] <= w:
        oy, ox = (h - img.shape[0]) // 2, (w - img.shape[1]) // 2
        canvas[oy:oy + img.shape[0], ox:ox + img.shape[1]] = img
    else:
        canvas = img[:h, :w]
    cv2.putText(canvas, caption[:22], (4, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (60, 60, 60), 1, cv2.LINE_AA)
    return canvas


def sheet(tiles, out_path, cols=COLS):
    if not tiles:
        print(f"  [跳过] 无内容：{out_path.name}")
        return
    rows = math.ceil(len(tiles) / cols)
    grid = np.full((rows * CELL, cols * CELL, 3), 255, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        grid[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL] = t
    cv2.imwrite(str(out_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"  {out_path.relative_to(ROOT)}  ({len(tiles)} 格)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="批次标签，如 lc13f")
    ap.add_argument("--export", default=None, help="默认 outputs/labcoat_label_review_<tag>/_her_export.txt")
    ap.add_argument("--stem2rel", default=None, help="默认 outputs/labcoat_label_review_<tag>/_stem2rel.tsv")
    ap.add_argument("--reject-dir", default=None, help="默认 data/_rejected/<tag>_reject_<今天>")
    ap.add_argument("--cols", type=int, default=COLS)
    args = ap.parse_args()

    out_dir = ROOT / "outputs" / f"labcoat_label_review_{args.tag}"
    exp = Path(args.export) if args.export else out_dir / "_her_export.txt"
    s2r_path = Path(args.stem2rel) if args.stem2rel else out_dir / "_stem2rel.tsv"
    if not exp.exists():
        sys.exit(f"找不到导出文件：{exp}")
    if not s2r_path.exists():
        sys.exit(f"找不到 stem→rel 映射：{s2r_path}")

    rej_dir = Path(args.reject_dir) if args.reject_dir else None
    if rej_dir is None:
        cands = sorted((ROOT / "data" / "_rejected").glob(f"{args.tag}_reject_*"))
        rej_dir = cands[-1] if cands else None

    s2r = {}
    with open(s2r_path, encoding="utf-8") as fh:
        for line in fh:
            a = line.rstrip("\n").split("\t")
            if len(a) >= 2:
                s2r[a[0]] = a[1]

    kept, rejected = parse_export(exp)
    print(f"[{args.tag}] 导出解析：保留 {len(kept)} · 剔除 {len(rejected)}")

    # 保留拼版（按批次顺序）
    k_tiles = []
    for stem in s2r:
        if stem in kept:
            rel = s2r[stem]
            img = imread_cn(SRC_ROOT / rel)
            k_tiles.append(tile(img, stem, boxes=kept[stem]))
    sheet(k_tiles, out_dir / f"_review_kept_{len(k_tiles)}.jpg", args.cols)

    # 剔除拼版（从 _rejected 读）
    r_tiles = []
    for stem in rejected:
        rel = s2r.get(stem, "")
        p = None
        if rel and rej_dir:
            p = rej_dir / rel
            if not p.exists():
                stem_path = rej_dir / Path(rel).parent / (Path(rel).stem + ".jpg")
                p = stem_path if stem_path.exists() else None
        img = imread_cn(p) if p is not None else None
        if img is None and p is not None:
            print(f"  [警告] 剔除原图读取失败：{p}")
        r_tiles.append(tile(img, stem))
    sheet(r_tiles, out_dir / f"_review_rejected_{len(r_tiles)}.jpg", args.cols)


if __name__ == "__main__":
    main()
