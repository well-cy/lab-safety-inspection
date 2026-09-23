#!/usr/bin/env python3
"""draw_labels.py — 通用 YOLO 标注画框工具（审计抽检用）

对指定图片目录里的每张图，读取同名 .txt 标注（支持类名文本和数字 id 两种格式），
画出所有框，输出到可视化目录。替代写死抽检逻辑的 visualize_b3.py，可对任意数据集复用
（B3 干净子集、新下载的公开数据集等）。

用法：
  python tools/draw_labels.py                          # 默认画 data/b3_clean 全部三个 split
  python tools/draw_labels.py --src data/raw/labcoat_si/train/images --out outputs/labcoat_si_check
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

# 类名 → BGR 颜色（与 visualize_b3.py 口径一致，便于对照）
CLS = {"Lab_coat": (0, 165, 255), "lab_coat": (0, 165, 255),
       "Gloves": (0, 200, 0), "gloves": (0, 200, 0),
       "Goggles": (160, 0, 255), "goggles": (160, 0, 255),
       "mask": (200, 0, 0), "face": (0, 200, 200)}
NUM_COLORS = [(0, 165, 255), (0, 200, 0), (160, 0, 255), (200, 0, 0), (0, 200, 200)]


def imread_u(path: Path):
    """兼容中文/韩文等非 ASCII 文件名（cv2.imread 在 Windows 下读不了）。"""
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def imwrite_u(path: Path, img):
    ok, buf = cv2.imencode(".jpg", img)
    if ok:
        buf.tofile(str(path))
    return ok


def load_names(src: Path, override: str = ""):
    """从 --src 向上找 data.yaml，把数字 id 映射为类别名用于显示。"""
    if override:
        return {i: n for i, n in enumerate(override.split(","))}
    d = src.parent if src.is_file() else src
    for _ in range(4):
        yml = d / "data.yaml"
        if yml.exists():
            try:
                import yaml
                meta = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
                nm = meta.get("names")
                if isinstance(nm, list):
                    return {i: n for i, n in enumerate(nm)}
                if isinstance(nm, dict):
                    return {int(k): v for k, v in nm.items()}
            except Exception:
                pass
        if d.parent == d:
            break
        d = d.parent
    return {}


def color_of(cls: str):
    if cls in CLS:
        return CLS[cls]
    try:  # 数字 id
        return NUM_COLORS[int(cls) % len(NUM_COLORS)]
    except ValueError:
        return (255, 255, 255)


def resolve_label(img_path: Path) -> Path:
    """按图片路径推断同名标注文件位置，兼容两种目录布局：
        raw 数据集布局   <root>/<split>/images/x.jpg -> <root>/<split>/labels/x.txt
        项目统一布局     <root>/images/<split>/x.jpg -> <root>/labels/<split>/x.txt
    """
    stem = img_path.stem + ".txt"
    a = img_path.parent.parent / "labels" / stem          # 布局 A
    if a.exists() or (img_path.parent.parent / "labels").is_dir():
        return a
    return img_path.parent.parent.parent / "labels" / img_path.parent.name / stem  # 布局 B


def draw(img_path: Path, lbl_path: Path, names: dict, targets: set | None = None,
         force_color: tuple | None = None):
    """force_color: BGR 元组；给定时所有框统一用该色（人工复查区分多套拼版用）。"""
    img = imread_u(img_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    if not lbl_path.exists():
        cv2.putText(img, "NO LABEL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return img
    for line in lbl_path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        cls, xc, yc, bw, bh = p[0], *map(float, p[1:])
        # 只画指定类别（按类别审计用）
        if targets is not None:
            try:
                if int(float(cls)) not in targets:
                    continue
            except ValueError:
                continue
        # 数字 id → 类别名（如 1 → lab_coat）
        try:
            show = names.get(int(float(cls)), cls)
        except ValueError:
            show = cls
        l, r = int((xc - bw / 2) * w), int((xc + bw / 2) * w)
        t, b = int((yc - bh / 2) * h), int((yc + bh / 2) * h)
        c = force_color if force_color is not None else (color_of(show) if show in CLS else color_of(cls))
        cv2.rectangle(img, (l, t), (r, b), c, 3)
        cv2.putText(img, show, (max(l, 5), max(t - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", default=[],
                    help="图片目录（可多次传）；默认为 data/b3_clean 的三个 split")
    ap.add_argument("--out", default="outputs/draw_labels_check", help="输出目录")
    ap.add_argument("--limit", type=int, default=0, help="每个目录最多画几张（0=全部）")
    ap.add_argument("--names", default="",
                    help="手工指定类别映射，如 --names mask,gloves,lab_coat,goggles（默认自动读 data.yaml）")
    ap.add_argument("--list", default="",
                    help="按清单画框：文件里每行一个相对 --root 的图片路径（如 train/xxx.jpg）")
    ap.add_argument("--root", default="",
                    help="--list 的根目录（数据集根，如 data/raw/labcoat_si）")
    ap.add_argument("--random", type=int, default=0,
                    help="从待画图中随机抽 N 张（抽查用，0=不随机）")
    ap.add_argument("--only-class", default="",
                    help="只画指定类别（数字 id 或类名，逗号分隔），如 --only-class gloves,1；"
                         "同时只挑含该类别的图片（按类别抽检用）")
    args = ap.parse_args()

    targets = None
    if args.only_class:
        targets = set()
        for tok in args.only_class.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if tok.isdigit():
                targets.add(int(tok))
            else:
                rev = {"mask": 0, "gloves": 1, "lab_coat": 2, "goggles": 3, "face": 0,
                       "glove": 1, "goggle": 1}
                if tok.lower() in rev:
                    targets.add(rev[tok.lower()])
                else:
                    print(f"[警告] 无法识别的类别: {tok}（用数字 id 或 mask/gloves/lab_coat/goggles）")
        print(f"[类别过滤] 只画类别 {sorted(targets)}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 模式一：按清单画（配合去重剔除清单做入库前抽查）
    jobs = []  # (img_path, lbl_path, names, targets)
    if args.list:
        root = Path(args.root)
        names = load_names(root, args.names)
        print(f"[清单模式] root={root}  类别映射={names}")
        for ln in Path(args.list).read_text(encoding="utf-8").splitlines():
            rel = ln.split("\t")[0].strip()
            if not rel or rel.startswith("#"):
                continue
            # 清单格式: <split>/<文件名>（与 dedup_new_dataset 的 leak 清单一致）。
            # 兼容两种目录布局：
            #   raw 数据集布局   <root>/<split>/images/<文件名>   （dataset1/2、labcoat_si、b3）
            #   项目统一布局     <root>/images/<split>/<文件名>   （data/lab_ppe）
            sp, name = rel.split("/", 1)
            stem = Path(name).stem
            if (root / sp / "images").is_dir():
                ip = root / sp / "images" / name
                lp = root / sp / "labels" / (stem + ".txt")
            else:
                ip = root / "images" / sp / name
                lp = root / "labels" / sp / (stem + ".txt")
            jobs.append((ip, lp, names, targets))
    else:
        srcs = args.src or [f"data/b3_clean/{s}/images" for s in ("train", "valid", "test")]
        for src in map(Path, srcs):
            if not src.exists():
                print(f"[跳过] 目录不存在: {src}")
                continue
            names = load_names(src, args.names)
            print(f"[目录模式] {src} → 类别映射 {names or '(按原始 id 显示)'}")
            imgs = sorted(p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            if args.limit:
                imgs = imgs[: args.limit]
            jobs += [(p, resolve_label(p), names, targets) for p in imgs]

    # 按类别筛图：只保留标注里含目标类别的图片
    if targets is not None:
        before = len(jobs)
        kept = []
        for ip, lp, nm, tg in jobs:
            if not lp.exists():
                continue
            for line in lp.read_text(encoding="utf-8").splitlines():
                p = line.split()
                if len(p) == 5 and p[0].lstrip("-").isdigit() and int(float(p[0])) in targets:
                    kept.append((ip, lp, nm, tg))
                    break
        jobs = kept
        print(f"[类别筛图] 含目标类别的图片 {len(jobs)}/{before} 张")

    if args.random and args.random < len(jobs):
        import random
        jobs = random.sample(jobs, args.random)
        print(f"[随机抽查] 抽取 {len(jobs)} 张")

    n = 0
    for ip, lp, names, tg in jobs:
        if not ip.exists():
            print(f"[跳过] 图片不存在: {ip}")
            continue
        img = draw(ip, lp, names, tg)
        if img is None:
            print(f"[跳过] 读图失败: {ip}")
            continue
        imwrite_u(out / f"{ip.stem[:60]}.jpg", img)
        n += 1
    print(f"完成：共画 {n} 张 → {out}/")


if __name__ == "__main__":
    main()
