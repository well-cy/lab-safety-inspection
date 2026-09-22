#!/usr/bin/env python3
"""make_contact_sheet.py — 把带框图片拼成大图版（网格），用于人工快速复查标注质量

为什么需要它：一张张点开看图太慢。本工具把 N 张图缩略拼成一张大图（默认 5×4=20 张/版），
每格左上角有**编号**、下方有文件名与问题标签。人工只需在大图版上找编号，
回报"第 3 版第 7 号错了"即可，不必逐个打开文件。

画框口径完全复用 draw_labels.py（同样的颜色、同样的类别名映射），保证与既有审计一致。

用法：
  # 1) 按问题清单拼版（配合 audit_lab_ppe_boxes.py 导出的 data/flags_*.txt）
  python tools/make_contact_sheet.py --list data/flags_gloves.txt --root data/lab_ppe \
      --only-class gloves --out outputs/gloves_review/flags

  # 2) 随机抽检（从数据集里随机挑含该类的图）
  python tools/make_contact_sheet.py --root data/lab_ppe --only-class mask \
      --random 80 --seed 42 --out outputs/mask_review/random

  # 3) 自定义版式；--full 额外导出原尺寸带框图（想放大细看时用）
  python tools/make_contact_sheet.py --list data/flags_gloves.txt --cols 4 --rows 5 \
      --thumb 400 --full --out outputs/gloves_review/flags

输出（都在 --out 目录下）：
  sheet_01.jpg …   图版（编号连续，跨版累计）
  index.txt        编号 → 文件名 / 问题标签 对照表（回报问题时用编号）
  items.txt        本次拼入的图清单，可直接再喂给本工具或 draw_labels.py
  full/            仅当传 --full：原尺寸带框图，文件名前缀编号
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from draw_labels import draw, imread_u, load_names, resolve_label  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ROOTDIR = ROOT / "data" / "lab_ppe"
CLASS_ID = {"mask": 0, "gloves": 1, "lab_coat": 2, "goggles": 3}
BAR_H = 30
BG = 250
PAD = 4


def fit_square(img: np.ndarray, size: int) -> np.ndarray:
    """等比缩放到 size×size 画布内（留白居中），避免小图被拉变形。"""
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


def build_jobs(args, root: Path):
    """返回 [(img_path, label_path, names, targets, flag_text)]"""
    targets = None
    if args.only_class:
        targets = set()
        # 除项目内置映射外，也接受该 root 下 data.yaml 里的类别名（大小写不敏感）
        name2id = dict(CLASS_ID)
        yml = root / "data.yaml"
        if yml.exists():
            try:
                import yaml
                for i, nm in enumerate(yaml.safe_load(yml.read_text(encoding="utf-8")).get("names", [])):
                    name2id[str(nm).strip().lower()] = i
            except Exception:
                pass
        for tok in args.only_class.split(","):
            tok = tok.strip().lower()
            if tok.isdigit():
                targets.add(int(tok))
            elif tok in name2id:
                targets.add(name2id[tok])
        print(f"[类别过滤] 只画类别 {sorted(targets)}")

    jobs = []
    if args.list:
        names = load_names(Path(args.root) if args.root else ROOTDIR, args.names)
        print(f"[清单模式] {args.list}  类别映射={names}")
        for ln in Path(args.list).read_text(encoding="utf-8").splitlines():
            parts = ln.split("\t")
            rel = parts[0].strip()
            flags = parts[1].strip() if len(parts) > 1 else ""
            if not rel or rel.startswith("#"):
                continue
            sp, name = rel.split("/", 1)
            stem = Path(name).stem
            if (root / sp / "images").is_dir():      # raw 数据集布局
                ip = root / sp / "images" / name
                lp = root / sp / "labels" / (stem + ".txt")
            else:                                     # 项目统一布局
                ip = root / "images" / sp / name
                lp = root / "labels" / sp / (stem + ".txt")
            jobs.append((ip, lp, names, targets, flags))
    else:
        names = load_names(root, args.names)
        print(f"[全库扫描] {root}  类别映射={names}")
        if (root / "labels").is_dir():
            label_dirs = [(sp, root / "labels" / sp, root / "images" / sp)
                          for sp in sorted(p.name for p in (root / "labels").iterdir() if p.is_dir())]
        else:  # raw 数据集布局: <root>/<split>/{images,labels}
            label_dirs = [(sp, root / sp / "labels", root / sp / "images")
                          for sp in ("train", "valid", "test") if (root / sp / "labels").is_dir()]
        for sp, ld, idir in label_dirs:
            for lp in sorted(ld.glob("*.txt")):
                lines = [l.split() for l in lp.read_text(encoding="utf-8").splitlines() if l.strip()]
                if targets is not None:
                    ok = any(len(p) == 5 and p[0].lstrip("-").isdigit()
                             and int(float(p[0])) in targets for p in lines)
                    if not ok:
                        continue
                elif not lines:
                    continue
                ip = idir / (lp.stem + ".jpg")
                if not ip.exists():
                    alt = [p for p in idir.glob(lp.stem + ".*")
                           if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
                    if not alt:
                        continue
                    ip = alt[0]
                jobs.append((ip, lp, names, targets, ""))

    # 类别筛图（清单模式下也要过滤掉不含目标类的）
    if targets is not None:
        before = len(jobs)
        kept = []
        for ip, lp, nm, tg, fl in jobs:
            if not lp.exists():
                continue
            for line in lp.read_text(encoding="utf-8").splitlines():
                p = line.split()
                if len(p) == 5 and p[0].lstrip("-").isdigit() and int(float(p[0])) in targets:
                    kept.append((ip, lp, nm, tg, fl))
                    break
        jobs = kept
        print(f"[类别筛图] 含目标类别的 {len(jobs)}/{before} 张")

    # 批次过滤：只看文件名（.rf. 之前）匹配正则的图，用于按"批次/来源"抽检
    if args.filter_regex:
        rx = re.compile(args.filter_regex)
        before = len(jobs)
        jobs = [j for j in jobs if rx.match(j[0].name.split(".rf.")[0])]
        print(f"[批次过滤] {args.filter_regex} 命中 {len(jobs)}/{before} 张")

    # 每族取一：同一原始照片（.rf. 之前的部分）只留一张代表，人工量降到 1/N
    if args.one_per_family:
        seen, kept = set(), []
        for job in jobs:
            fam = job[0].name.split(".rf.")[0]
            if fam in seen:
                continue
            seen.add(fam)
            kept.append(job)
        jobs = kept
        print(f"[每族取一] 剩 {len(jobs)} 张（每个原始照片家族 1 张代表）")

    if args.random and args.random < len(jobs):
        random.seed(args.seed)
        jobs = random.sample(jobs, args.random)
        print(f"[随机抽检] 抽出 {len(jobs)} 张（seed={args.seed}）")

    if args.limit:
        jobs = jobs[: args.limit]
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="", help="图片清单（<split>/<文件名>[\\t标签]），如 data/flags_gloves.txt")
    ap.add_argument("--root", default=str(ROOTDIR), help="数据集根目录（默认 data/lab_ppe）")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--only-class", default="", help="只看/只画该类（mask/gloves/lab_coat/goggles 或数字 id）")
    ap.add_argument("--names", default="", help="手工类别映射，如 --names mask,gloves,lab_coat,goggles")
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--thumb", type=int, default=360, help="单格缩略图边长（像素）")
    ap.add_argument("--random", type=int, default=0, help="随机抽 N 张")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--full", action="store_true", help="额外导出原尺寸带框图到 <out>/full/")
    ap.add_argument("--filter-regex", default="",
                    help="只保留文件名（.rf. 之前）匹配该正则的图，如 '^(screenshot|IMG-\\\\d{8}-WA)'")
    ap.add_argument("--one-per-family", action="store_true",
                    help="同一原始照片家族只留 1 张代表（按批次判断时用，人工量降到 1/N）")
    ap.add_argument("--start-index", type=int, default=1, help="编号起始（多批续编时用）")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.full:
        (out / "full").mkdir(exist_ok=True)

    jobs = build_jobs(args, root)
    if not jobs:
        print("[空] 没有可拼的图，请检查 --list / --root / --only-class")
        return 1

    per = args.cols * args.rows
    cell_w, cell_h = args.thumb + PAD * 2, args.thumb + BAR_H + PAD * 2
    index_lines, item_lines = [], []
    idx = args.start_index
    skipped, sheets = [], 0

    for s in range(0, len(jobs), per):
        chunk = jobs[s:s + per]
        sheets += 1
        sheet = np.full((args.rows * cell_h, args.cols * cell_w, 3), 255, np.uint8)
        for k, (ip, lp, names, tg, flags) in enumerate(chunk):
            r, c = divmod(k, args.cols)
            if not ip.exists():
                skipped.append(str(ip))
                continue
            img = draw(ip, lp, names, tg)
            if img is None:
                skipped.append(str(ip))
                continue

            if args.full:
                full = img.copy()
                hh, ww = full.shape[:2]
                cv2.rectangle(full, (0, 0), (ww - 1, hh - 1), (0, 0, 0), 2)
                cv2.rectangle(full, (6, 6), (6 + 34 * len(str(idx)) + 20, 62), (0, 0, 0), -1)
                cv2.putText(full, str(idx), (16, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                            (255, 255, 255), 3)
                cv2.imencode(".jpg", full)[1].tofile(
                    str(out / "full" / f"{idx:03d}_{ip.stem[:50]}.jpg"))

            cell = np.full((cell_h, cell_w, 3), 255, np.uint8)
            thumb = fit_square(img, args.thumb)
            y0, x0 = PAD, PAD
            cell[y0:y0 + args.thumb, x0:x0 + args.thumb] = thumb
            cv2.rectangle(cell, (x0, y0), (x0 + args.thumb - 1, y0 + args.thumb - 1),
                          (150, 150, 150), 1)

            # 编号徽标
            badge_w = 16 + 12 * len(str(idx))
            cv2.rectangle(cell, (x0, y0), (x0 + badge_w, y0 + 26), (0, 0, 0), -1)
            cv2.putText(cell, str(idx), (x0 + 8, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)

            # 底部文字：文件名 + 问题标签
            ty = y0 + args.thumb + 20
            cv2.putText(cell, clip_text(ip.stem, 30), (x0, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (30, 30, 30), 1)
            if flags:
                cv2.putText(cell, clip_text(flags, 34), (x0, ty + 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.34, (0, 0, 230), 1)

            sy, sx = r * cell_h, c * cell_w
            sheet[sy:sy + cell_h, sx:sx + cell_w] = cell

            rel = f"{lp.parent.name}/{ip.name}" if lp.parent.name in ("train", "valid", "test") else ip.name
            index_lines.append(f"{idx:>4}\t{rel}\t{flags}")
            item_lines.append(f"{lp.parent.name}/{ip.name}")
            idx += 1

        cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(
            str(out / f"sheet_{sheets:02d}.jpg"))
        print(f"  图版 {sheets:02d}: {len(chunk)} 格 → {out / f'sheet_{sheets:02d}.jpg'}")

    header = [
        f"# {args.out} 复查对照表 —— 编号 → 图片文件（{len(index_lines)} 张，共 {sheets} 版）",
        "# 回报问题时只需说「第X版 编号N 有问题 + 原因」",
        "# 格式: 编号 <TAB> split/文件名 <TAB> 问题标签",
    ]
    (out / "index.txt").write_text("\n".join(header + index_lines) + "\n", encoding="utf-8")
    (out / "items.txt").write_text("\n".join(item_lines) + "\n", encoding="utf-8")

    print(f"\n完成：{len(index_lines)} 张 / {sheets} 版 → {out}/")
    print(f"  对照表: {out / 'index.txt'}")
    if skipped:
        print(f"  [跳过] {len(skipped)} 张读图/路径失败，例如 {skipped[:3]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
