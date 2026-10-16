# -*- coding: utf-8 -*-
"""把 lab_coat 画框工作区落库进 data/lab_ppe（labcoat_imagenetcv 批的落库端）。

流程：
  1. 读编辑器导出 txt（她贴回的改动）→ 覆盖工作区标签
  2. 读 _stem2rel.tsv → 定位 data/raw/labcoat_imagenetcv/data/<rel>
  3. 按 split 映射到 data/lab_ppe 的 split（test→test, val→valid, train→train）
  4. tar 备份 labels/；图片以 lc13_ 前缀**复制**进 data/lab_ppe/images/<split>/
  5. 标签 cxcywh 直写 data/lab_ppe/labels/<split>/<stem>.txt（类别 3=lab_coat）
  6. !REJECT 的图不写标签，图（原图）移入 data/_rejected/<reject-tag>/
  7. 落库后格式校验 + 类别对账

铁律：只移不删（用 shutil.move / os.replace，绝不 os.unlink）。

用法:
  $PY tools/apply_labcoat_label.py --export outputs/labcoat_label_review_lc13a/_her_export.txt \
      --workspace data/_prelabel/labcoat_lc13a/labels \
      --stem2rel outputs/labcoat_label_review_lc13a/_stem2rel.tsv \
      --accept-from-ws --dry        # 先干跑看统计
  # 确认后去掉 --dry 执行
"""
from __future__ import annotations

import argparse
import io
import re
import shutil
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "data" / "lab_ppe"
RAW = ROOT / "data" / "raw" / "labcoat_imagenetcv" / "data"
SPLITS = ("train", "valid", "test")
NAMES = ["person", "mask", "gloves", "lab_coat", "goggles"]
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
PREFIX = "lc13_"
# raw split 名 → data/lab_ppe split 名（保持与库内既有口径一致：val → valid）
SPLIT_MAP = {"train": "train", "val": "valid", "test": "test"}


def parse_export(txt: str):
    """解析编辑器导出 → (items, rejected, reasons)

    注意：编辑器在「先画了框、后点了剔除」时会写成
        ### <stem>
        !REJECT
        3 0.5 0.4 0.3 0.5
    即剔除标记与框行**同时存在**（自相矛盾）。本函数以**剔除为先**：
    只要出现 !REJECT，该图一律剔除，其框行丢弃（但记录下来供人工复核）。
    """
    items, rejected, reasons, cur = {}, set(), {}, None
    conflict = {}   # stem -> 该图在剔除标记之后仍写出的框数（提示她可能误操作）
    for ln in txt.splitlines():
        ln = ln.strip()
        if ln.startswith("###"):
            cur = ln[3:].strip()
            items.setdefault(cur, [])
            continue
        if not cur:
            continue
        if ln.upper().startswith("!REJECT"):
            rejected.add(cur)
            reasons[cur] = ln[7:].strip()
            items[cur] = []          # 清空（若之前已收框），且保留键避免后续 KeyError
            continue
        if len(ln.split()) == 5:
            p = ln.split()
            items.setdefault(cur, []).append(
                (int(p[0]), *(round(float(v), 6) for v in p[1:5])))
            if cur in rejected:
                conflict[cur] = conflict.get(cur, 0) + 1
    # 剔除图不保留任何框
    for s in rejected:
        items[s] = []
    return items, rejected, reasons, conflict


def backup_labels() -> Path | None:
    lab = LAB / "labels"
    if not lab.exists():
        return None
    ts = time.strftime("%Y%m%d_%H%M%S")
    dst = LAB / "_label_backup" / f"labels_labcoat_lc13_{ts}.tar.gz"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dst, "w:gz") as tf:
        tf.add(lab, arcname="labels")
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default="", help="编辑器导出 txt（有改动/剔除的图）")
    ap.add_argument("--workspace", required=True, help="工作区标签目录")
    ap.add_argument("--stem2rel", required=True, help="_stem2rel.tsv")
    ap.add_argument("--accept-from-ws", action="store_true",
                    help="批次内未出现在导出中的图，直接采用工作区标签（她回「都对」时用）")
    ap.add_argument("--skip-stems", default="",
                    help="逗号分隔 stem：跳过不落库（待补标/复核，如漏标 lab_coat 的图）")
    ap.add_argument("--reject-tag", default="", help="剔除去向目录名，默认 lc13_reject_<日期>")
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    ws = Path(args.workspace)
    if not ws.exists():
        print(f"  工作区不存在：{ws}")
        return 1

    # 写前全量备份（默认；--no-backup 跳过）
    if not args.dry and not args.no_backup:
        bp = backup_labels()
        print(f"  已备份 labels/ → {bp}" if bp else "  ⚠ labels/ 不存在，跳过备份")

    # stem → rel 映射
    s2r = {}
    for ln in Path(args.stem2rel).read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        p = ln.split("\t")
        if len(p) >= 2:
            s2r[p[0].strip()] = p[1].strip()
    print(f"  映射表 {len(s2r)} 条")

    # 1) 应用导出：覆盖工作区标签
    n_exp, rejects, reasons = 0, set(), {}
    conflict = {}
    if args.export and Path(args.export).exists():
        items, rejects, reasons, conflict = parse_export(
            Path(args.export).read_text(encoding="utf-8"))
        for stem, boxes in items.items():
            if stem in rejects:
                continue
            (ws / f"{stem}.txt").write_text(
                "\n".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in boxes)
                + ("\n" if boxes else ""), encoding="utf-8", newline="\n")
            n_exp += 1
        print(f"  导出覆盖 {n_exp} 图 · 整图剔除 {len(rejects)} 图")
        if conflict:
            print(f"  ⚠ 剔除标记与框行并存 {len(conflict)} 图（已按剔除处理，框行丢弃）：")
            for s, k in list(conflict.items())[:10]:
                print(f"     · {s}（{k} 个框被丢弃）")
    else:
        print("  无导出（--export 为空/不存在），仅采用工作区标签")

    # 2) 逐图落库
    splits_txt: dict[str, list[str]] = {s: [] for s in SPLITS}
    stat = Counter()
    moved_rej, missing_src = [], []
    rej_root = ROOT / "data" / "_rejected" / (
        args.reject_tag or f"lc13_reject_{time.strftime('%Y%m%d')}")

    for stem, rel in sorted(s2r.items()):
        if stem in {s.strip() for s in args.skip_stems.split(",") if s.strip()}:
            stat["跳过（待补标）"] += 1
            continue
        raw_split = rel.split("/", 1)[0]
        lab_split = SPLIT_MAP.get(raw_split)
        if lab_split is None:
            stat["未知split"] += 1
            continue
        src_img = RAW / rel
        if not src_img.exists():
            missing_src.append(rel)
            stat["源图缺失"] += 1
            continue
        new_stem = PREFIX + stem
        if stem in rejects:
            # 整图剔除：原图移入 _rejected（只移不删）
            dst = rej_root / rel
            if not args.dry:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.move(str(src_img), str(dst))
                # 若工作区有残留标签，一并移走（归档）
                wf = ws / f"{stem}.txt"
                if wf.exists():
                    shutil.move(str(wf), str(dst.with_suffix(".ws.txt")))
            moved_rej.append((rel, reasons.get(stem, "")))
            stat["剔除"] += 1
            continue

        wf = ws / f"{stem}.txt"
        rows = []
        if wf.exists():
            for ln in wf.read_text(encoding="utf-8").splitlines():
                q = ln.split()
                if len(q) == 5:
                    rows.append((int(q[0]), *(round(float(v), 6) for v in q[1:5])))
        if not rows:
            stat["无框（跳过）"] += 1
            continue

        # 校验：允许五类 0-4（她实际会顺手把 person/gloves 等一起标）；
        # 每张保留图必须至少有 1 个 lab_coat(3)（本批来源就是白大褂图）
        bad = [b for b in rows if b[0] not in (0, 1, 2, 3, 4)
               or not (0 <= b[1] <= 1 and 0 <= b[2] <= 1
                       and 0 < b[3] <= 1 and 0 < b[4] <= 1)]
        if bad:
            stat["坐标/类别异常"] += 1
            print(f"  ⚠ {rel}: 异常框 {bad[:3]}")
            continue
        if not any(b[0] == 3 for b in rows):
            stat["无lab_coat框（警告保留）"] += 1
            print(f"  ⚠ {rel}: 保留了但没有 lab_coat 框（类别 "
                  f"{sorted({b[0] for b in rows})}），请复核")

        lab_txt = LAB / "labels" / lab_split / (new_stem + ".txt")
        img_dst = LAB / "images" / lab_split / (new_stem + src_img.suffix.lower())
        if not args.dry:
            lab_txt.parent.mkdir(parents=True, exist_ok=True)
            img_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src_img), str(img_dst))
            lab_txt.write_text(
                "\n".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in rows)
                + "\n", encoding="utf-8", newline="\n")
        splits_txt[lab_split].append(new_stem)
        stat[f"入库·{lab_split}"] += 1
        stat["总框数"] += len(rows)

    print("\n  ── 落库统计 ──")
    for k, v in stat.most_common():
        print(f"   {k:16s} {v}")
    for s in SPLITS:
        print(f"   {s}: {len(splits_txt[s])} 图")
    if moved_rej:
        print(f"\n  剔除 {len(moved_rej)} 图 → {rej_root}")
        for rel, why in moved_rej[:10]:
            print(f"   · {rel}  {why}")
    if missing_src:
        print(f"\n  ⚠ 源图缺失 {len(missing_src)}：{missing_src[:5]}")

    if args.dry:
        print("\n  干跑结束。去掉 --dry 执行落库。")
        return 0

    # 3) 备份在写之前（若 dry 已跳过）
    # 备份应在写之前做：这里重排——为稳妥，先做一次全量备份再继续后续校验
    print("\n  完成落库。建议马上跑 tools/audit_lab_ppe_boxes.py 做全库格式校验。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
