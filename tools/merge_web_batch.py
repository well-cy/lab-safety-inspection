# -*- coding: utf-8 -*-
"""
把 labeler.html 导出的标注 ZIP 合并进主数据集（data/lab_ppe）。

背景：labeler.html 内部用「左上角+宽高」存储框，2026-09-24 之前导出的 ZIP
     直接写左上角口径（YOLO 要求中心点），导致框整体偏移。该 bug 已修，但
     历史 ZIP 仍需要转换——本脚本会自动判别口径并统一转换，避免再踩。

流程：
    1. 解包 ZIP（或直接读已解包目录）到 --work
    2. 口径判别：若多数框按「中心点」解释会越界、按「左上角」解释不越界 -> 判为左上角，转换
    3. 裁剪越界框到 [0,1]；剔除退化框（宽或高 < 0.005）
    4. 合法性校验：类别 ∈ {0,1,2,3}、坐标有限、无 NaN
    5. 撞名检查（与主数据集）
    6. 90/10 分层入 train/valid（--seed，默认 42），**不进 test**
    7. 负样本（无框图）写空 txt 保留
    8. 全库复核：图文配对、框数统计

用法：
    python tools/merge_web_batch.py --zip <导出.zip> --images <图片目录>            # 干跑
    python tools/merge_web_batch.py --zip <导出.zip> --images <图片目录> --apply    # 落盘
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import random
import shutil
import statistics as st
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "data" / "lab_ppe"
CLASSES = ["mask", "gloves", "lab_coat", "goggles"]
MIN_SIDE = 0.005


def load_boxes(text: str):
    boxes, bad = [], 0
    for line in text.splitlines():
        t = line.split()
        if not t:
            continue
        if len(t) != 5:
            bad += 1
            continue
        try:
            c = int(float(t[0]))
            x, y, w, h = (float(v) for v in t[1:5])
        except ValueError:
            bad += 1
            continue
        if c not in (0, 1, 2, 3) or not all(map(lambda v: v == v and abs(v) < 1e6, (x, y, w, h))):
            bad += 1
            continue
        boxes.append((c, x, y, w, h))
    return boxes, bad


def overhang(boxes, mode: str) -> float:
    """返回「越界程度」总和：中心口径与左上角口径各算一次，谁小算谁的对。"""
    tot = 0.0
    for _, x, y, w, h in boxes:
        if mode == "center":
            x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
        else:  # topleft
            x1, y1, x2, y2 = x, y, x + w, y + h
        tot += max(0.0, -x1) + max(0.0, -y1) + max(0.0, x2 - 1) + max(0.0, y2 - 1)
    return tot


def to_center(boxes, mode: str):
    """统一转成 YOLO 中心点口径，并裁剪到图像边界。返回 (boxes, n_clipped, n_dropped)。"""
    out, clipped, dropped = [], 0, 0
    for c, x, y, w, h in boxes:
        if mode == "topleft":
            cx, cy = x + w / 2, y + h / 2
        else:
            cx, cy = x, y
        x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
        x2, y2 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
        if x2 - x1 < MIN_SIDE or y2 - y1 < MIN_SIDE:
            dropped += 1
            continue
        if abs(x1 - (cx - w / 2)) > 1e-9 or abs(x2 - (cx + w / 2)) > 1e-9 \
           or abs(y1 - (cy - h / 2)) > 1e-9 or abs(y2 - (cy + h / 2)) > 1e-9:
            clipped += 1
        out.append((c, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
    return out, clipped, dropped


def main() -> int:
    ap = argparse.ArgumentParser(description="labeler 导出 ZIP -> 主数据集入库")
    ap.add_argument("--zip", required=True, help="labeler 导出的 zip；也可传已解包目录")
    ap.add_argument("--images", required=True, help="图片目录（与 zip 内 labels 的 stem 对齐）")
    ap.add_argument("--work", default=str(ROOT / "data" / "raw" / "_merge_work"), help="解包/转换工作目录")
    ap.add_argument("--valid-ratio", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--apply", action="store_true", help="真正写入主数据集（默认只干跑）")
    args = ap.parse_args()

    src = Path(args.zip)
    img_dir = Path(args.images)
    work = Path(args.work)

    # ---------- 0. 按批次隔离工作目录（不清空、不删除任何文件）----------
    # work 默认 data/raw/_merge_work，按 ZIP 内容指纹派生子目录，避免跨批次标签混入
    if str(work) == str(ROOT / "data" / "raw" / "_merge_work"):
        if src.is_file():
            fp = (str(src.stat().st_size) + str(src.stat().st_mtime))
        else:
            fp = "|".join(sorted(p.name for p in src.glob("labels/*.txt")))
        tag = hashlib.sha1(fp.encode("utf-8")).hexdigest()[:10]
        work = work / tag
    raw_labels = work / "labels_raw"
    yolo_labels = work / "labels_yolo"

    # ---------- 1. 取标签 ----------
    if src.is_dir():
        cand = src / "labels"
        raw_labels = cand if cand.is_dir() else src
    else:
        with zipfile.ZipFile(src) as z:
            if z.testzip() is not None:
                print("[错误] ZIP CRC 校验失败")
                return 2
            raw_labels.mkdir(parents=True, exist_ok=True)
            for n in z.namelist():
                if n.startswith("labels/") and n.endswith(".txt"):
                    (raw_labels / Path(n).name).write_bytes(z.read(n))
            for n in ("classes.txt", "_export_report.txt", "_unlabeled.txt", "_skipped.txt"):
                if n in z.namelist():
                    (work / n).write_bytes(z.read(n))
    txts = sorted(raw_labels.glob("*.txt"))
    print(f"标签文件: {len(txts)}  图片目录: {img_dir}")

    # ---------- 2. 口径判别 ----------
    all_boxes, bad_total = [], 0
    per_file = {}
    for f in txts:
        b, bad = load_boxes(f.read_text(encoding="utf-8"))
        per_file[f.stem] = b
        all_boxes += b
        bad_total += bad
    tot_c, tot_t = overhang(all_boxes, "center"), overhang(all_boxes, "topleft")
    mode = "topleft" if tot_t < tot_c else "center"
    print(f"框总数 {len(all_boxes)}，非法行 {bad_total}")
    print(f"越界程度：按中心口径 {tot_c:.3f} / 按左上角口径 {tot_t:.3f}  -> 判定为 [{mode}] 口径")
    if mode == "topleft":
        print("  （labeler 旧版导出格式，自动转换为 YOLO 中心点口径）")

    # ---------- 3/4. 转换 + 校验 ----------
    yolo_labels.mkdir(parents=True, exist_ok=True)
    n_box = n_clip = n_drop = 0
    final = {}
    for stem, boxes in per_file.items():
        conv, clip, drop = to_center(boxes, mode)
        n_box += len(conv); n_clip += clip; n_drop += drop
        final[stem] = conv
        (yolo_labels / f"{stem}.txt").write_text(
            "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in conv),
            encoding="utf-8")
    print(f"转换后 {n_box} 框（裁剪 {n_clip}，剔除 {n_drop}）")
    if n_box:
        for i, name in enumerate(CLASSES):
            sub = [b for b in final.values() for b in b if b[0] == i]
            if not sub:
                continue
            areas = sorted(b[3] * b[4] for b in sub)
            print(f"  {name:9} n={len(sub):4}  面积中位 {st.median(areas):.3f}")

    have_img = {p.stem for p in img_dir.iterdir() if p.is_file()}

    # 只保留「本批图片目录里真实存在」的标签，防止工作目录残留的其他批次标签混入
    foreign = sorted(set(per_file) - have_img)
    if foreign:
        print(f"[警告] 丢弃 {len(foreign)} 个不属于本批的标签：{', '.join(foreign[:5])}…")
        for s in foreign:
            per_file.pop(s, None)
            (yolo_labels / f"{s}.txt").unlink(missing_ok=True)

    # 用户标为「不合格」的图直接排除，不作为负样本入库
    skip = set()
    sk = work / "_skipped.txt"
    if sk.exists():
        skip = {l.strip().rsplit(".", 1)[0] for l in sk.read_text(encoding="utf-8").splitlines() if l.strip()}
    print(f"标为不合格（排除）: {len(skip)} 张")

    # ---------- 5. 撞名 ----------
    existing = set()
    for sp in ("train", "valid", "test"):
        existing |= {p.stem for p in (DS / "images" / sp).iterdir()}
    clash = sorted(existing & have_img)
    missing_img = sorted(set(final) - have_img)
    print(f"撞名 {len(clash)} 张{': ' + ', '.join(clash[:5]) if clash else ''}")
    print(f"有标签无图 {len(missing_img)} 张{': ' + ', '.join(missing_img[:5]) if missing_img else ''}")

    pool = sorted((have_img - existing - skip) - (set(final) - have_img))
    labeled = [s for s in pool if final.get(s)]
    neg = [s for s in pool if not final.get(s)]
    print(f"待入库 {len(pool)} 张：有框 {len(labeled)}  负样本 {len(neg)}")

    # ---------- 6/7. 划分 ----------
    random.seed(args.seed)
    lab_sh = labeled[:]; random.shuffle(lab_sh)
    neg_sh = neg[:]; random.shuffle(neg_sh)
    n_val = max(1, round(len(labeled) * args.valid_ratio))
    n_val_neg = max(0, round(len(neg) * args.valid_ratio))
    valid_set = set(lab_sh[:n_val]) | set(neg_sh[:n_val_neg])
    train_set = set(pool) - valid_set
    print(f"train {len(train_set)}  valid {len(valid_set)}（seed={args.seed}，不进 test）")

    if not args.apply:
        print("\n[干跑] 未写入主数据集。确认无误后加 --apply")
        return 0

    for stem in sorted(pool):
        sp = "train" if stem in train_set else "valid"
        img = next(img_dir.glob(stem + ".*"))
        shutil.copy2(img, DS / "images" / sp / img.name)
        (DS / "labels" / sp / f"{stem}.txt").write_text(
            "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in final.get(stem, [])),
            encoding="utf-8")

    # ---------- 8. 全库复核 ----------
    tot_img = tot_box = 0
    counter = collections.Counter()
    orphan = 0
    for sp in ("train", "valid", "test"):
        imgs = {p.stem for p in (DS / "images" / sp).iterdir()}
        labs = {p.stem for p in (DS / "labels" / sp).glob("*.txt")}
        orphan += len(labs - imgs)
        tot_img += len(imgs)
        for lp in (DS / "labels" / sp).glob("*.txt"):
            if lp.stem not in imgs:
                continue
            for line in lp.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    counter[int(float(line.split()[0]))] += 1
                    tot_box += 1
    print(f"\n入库后全库: {tot_img} 张 / {tot_box} 框  孤儿标签 {orphan}")
    print("  " + " / ".join(f"{CLASSES[i]} {counter[i]}" for i in range(4)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
