#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""screen_labcoat_imagenetcv.py —— 筛查 images.cv lab coat 分类图（1,299 张，无 bbox）

背景：组长下载的 `e58z9vu3fcusg7wzjk91n.zip` 是 images.cv 的 lab_coat **分类**数据集，
      原始分辨率、彩色、无增强，但**没有检测框**，且来源是网络图（电商/生活照/展会混杂）。
      按项目场景红线（训练图必须是**实验室/科研/医疗操作**场景）必须先筛。

复用 `screen_dirty_graphics.py` 的统计判据（白底商品图 / 单色广告图 / 文档图 / 黑白老照片），
另加本数据集特有的四类判据：
  A. 电商商品图/平铺图  —— 白底占比高 + 平坦（沿用 graphic_white）
  B. 拼图/多宫格        —— 检测规则网格线（衣服多角度拼版、九宫格）
  C. 水印/文字叠加      —— 角/边区域高频细线条 + 低饱和
  D. 低质量/糊图        —— 拉普拉斯方差过低；尺寸过小

用法:
    $PY tools/screen_labcoat_imagenetcv.py                      # 干跑 + 出明细 TSV + 拼版清单
    $PY tools/screen_labcoat_imagenetcv.py --sheets             # 同时导出抽检拼版 JPG
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from screen_dirty_graphics import image_stats, classify, extra_stats, imread_unicode  # noqa: E402

SRC = ROOT / "data" / "raw" / "labcoat_imagenetcv" / "data"
OUT = ROOT / "outputs" / "labcoat_imagenetcv_screen"


def gridline_score(img) -> tuple[float, float]:
    """检测规则网格线（拼图/多宫格）：行/列方向长直线占比。

    拼版图会把 2x2 / 3x3 子图用白/黑分隔线切开，这种线在 Hough 里表现为
    近乎水平或垂直、且贯穿较长距离的线段。
    """
    h, w = img.shape[:2]
    small = cv2.resize(img, (320, int(320 * h / max(w, 1))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                            minLineLength=int(small.shape[1] * 0.55), maxLineGap=6)
    if lines is None:
        return 0.0, 0.0
    H, W = small.shape[:2]
    # OpenCV 版本差异：旧版返回 (N,1,4)，新版可能返回 (N,4) —— 统一 ravel
    lines = np.asarray(lines).reshape(-1, 4)
    n_h = n_v = 0
    for x1, y1, x2, y2 in lines:
        dx, dy = abs(int(x2) - int(x1)), abs(int(y2) - int(y1))
        if dy <= 2 and dx >= 0.5 * W:
            n_h += 1
        elif dx <= 2 and dy >= 0.5 * H:
            n_v += 1
    # 归一化：2 条以上分隔线即可疑
    return min(n_h / 4.0, 1.0), min(n_v / 4.0, 1.0)


def watermark_score(img) -> float:
    """粗判水印/叠字：图像边缘条带的高频细边密度（文字/水印会留下密集短边）。

    仅作参考指标——水印需人工确认，不单独作为剔除依据。
    """
    h, w = img.shape[:2]
    small = cv2.resize(img, (320, int(320 * h / max(w, 1))), interpolation=cv2.INTER_AREA)
    H, W = small.shape[:2]
    band = int(H * 0.12)
    strips = [small[0:band], small[H - band:H], small[:, 0:band], small[:, W - band:W]]
    scores = []
    for s in strips:
        g = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
        e = cv2.Canny(g, 90, 200)
        scores.append(float((e > 0).mean()))
    return float(np.mean(scores))


def blur_score(img) -> float:
    """拉普拉斯方差：越低越糊。"""
    h, w = img.shape[:2]
    small = cv2.resize(img, (320, int(320 * h / max(w, 1))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def judge(img, path: Path) -> dict:
    h, w = img.shape[:2]
    st = image_stats(img)
    ex = extra_stats(img)
    tag = classify(st)
    gh, gv = gridline_score(img)
    wm = watermark_score(img)
    bl = blur_score(img)

    flags = []
    if tag == "graphic_white":
        flags.append("ecom_flatlay")          # 白底商品/平铺图
    elif tag == "graphic_color":
        flags.append("ad_graphic")
    elif tag == "blank_doc":
        flags.append("blank_doc")
    if gh >= 0.5 or gv >= 0.5:
        flags.append("collage_grid")          # 拼图/多宫格
    if ex["colorf"] < 6.0:
        flags.append("grayscale_like")
    if bl < 60:
        flags.append("blurry")                # 糊图
    if min(w, h) < 220:
        flags.append("too_small")
    if w * h > 4_000_000:
        flags.append("huge_res")              # 分辨率异常高（多为原图下载，非问题，仅记录）

    return dict(w=w, h=h, bright=round(ex["bright"], 4), colorf=round(ex["colorf"], 3),
                white=round(st["white"], 4), dom=round(st["dom"], 4),
                flat=round(st["flat"], 4), edge=round(st["edge"], 4),
                gridH=round(gh, 3), gridV=round(gv, 3), wm=round(wm, 4), blur=round(bl, 1),
                flags="|".join(flags) if flags else "ok")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", action="store_true", help="导出抽检拼版 JPG")
    ap.add_argument("--per-sheet", type=int, default=48)
    args = ap.parse_args()

    exts = {".jpg", ".jpeg", ".png"}
    imgs = sorted(p for p in SRC.rglob("*") if p.suffix.lower() in exts)
    print(f"[筛查] {SRC}")
    print(f"  受检图片 {len(imgs)} 张")

    OUT.mkdir(parents=True, exist_ok=True)
    rows, hits, unread = [], {}, []
    for p in imgs:
        img = imread_unicode(p)
        if img is None:
            unread.append(p.name)
            rows.append((p.name, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "unreadable"))
            continue
        r = judge(img, p)
        # 相对 SRC 的路径（含 train/val/test 与 lab_coat 层级），保证可回溯
        name = p.relative_to(SRC).as_posix()
        rows.append((name, r["w"], r["h"], r["bright"], r["colorf"], r["white"], r["dom"],
                     r["flat"], r["edge"], r["gridH"], r["gridV"], r["wm"], r["blur"], r["flags"]))
        for f in r["flags"].split("|"):
            if f and f != "ok" and f != "huge_res":
                hits.setdefault(f, []).append(name)

    print()
    for t, files in sorted(hits.items(), key=lambda x: -len(x[1])):
        print(f"  {t:16s} {len(files):5d} 张")
    allf = sorted({f for v in hits.values() for f in v})
    print(f"\n  合计可疑 {len(allf)} 张（{len(allf)/max(len(imgs),1):.1%}）")
    print(f"  干净可用（无 flags）{len(imgs)-len(allf)-len(unread)} 张")
    if unread:
        print(f"  无法读取 {len(unread)} 张")

    tsv = OUT / "screen_detail.tsv"
    hdr = ["name", "width", "height", "bright", "colorf", "white", "dom", "flat", "edge",
           "gridH", "gridV", "watermark", "blur", "flags"]
    with tsv.open("w", encoding="utf-8", newline="") as f:
        f.write("\t".join(hdr) + "\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    print(f"\n[明细] {tsv}")

    flag_of = {f: t for t, files in hits.items() for f in files}
    lst = OUT / "suspect.txt"
    lst.write_text("# labcoat_imagenetcv 自动筛查可疑图（待人工复核）\n" +
                   "\n".join(f"{i+1}\t{n}\t{flag_of[n]}" for i, n in enumerate(allf)) + "\n",
                   encoding="utf-8")
    print(f"[可疑清单] {lst}")

    clean = [r[0] for r in rows if r[13] in ("ok", "huge_res")]
    cl = OUT / "clean.txt"
    cl.write_text("# labcoat_imagenetcv 自动筛查通过（无 flags，建议进入人工细看/标注候选）\n" +
                  "\n".join(f"{i+1}\t{n}" for i, n in enumerate(sorted(clean))) + "\n",
                  encoding="utf-8")
    print(f"[通过清单 {len(clean)} 张] {cl}")

    if args.sheets:
        make_sheets(allf, OUT, args.per_sheet, "suspect")
        make_sheets(sorted(clean), OUT, args.per_sheet, "clean")
    return 0


def make_sheets(names: list[str], out: Path, per: int, tag: str) -> None:
    """把清单里的图拼成抽检版（每版 per 张，5 列）。"""
    if not names:
        return
    cell, cols = 300, 5
    rows_n = (per + cols - 1) // cols
    for page in range((len(names) + per - 1) // per):
        chunk = names[page * per:(page + 1) * per]
        canvas = np.full((rows_n * (cell + 22), cols * cell, 3), 245, np.uint8)
        for i, n in enumerate(chunk):
            p = SRC / n
            img = imread_unicode(p)
            if img is None:
                continue
            h, w = img.shape[:2]
            s = cell / max(h, w)
            thumb = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
            r, c = divmod(i, cols)
            y0, x0 = r * (cell + 22), c * cell
            th, tw = thumb.shape[:2]
            canvas[y0:y0 + th, x0:x0 + tw] = thumb
            cv2.putText(canvas, n.split("/")[-1][:16], (x0 + 3, y0 + cell + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)
        fp = out / f"_{tag}_p{page+1:02d}.jpg"
        cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 82])[1].tofile(str(fp))
    print(f"[拼版] {out}/_{tag}_p*.jpg  共 {(len(names)+per-1)//per} 版")


if __name__ == "__main__":
    sys.exit(main())
