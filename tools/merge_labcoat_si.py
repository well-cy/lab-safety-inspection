"""labcoat_si 入库合并：把审计通过的图片并入项目统一数据集 data/lab_ppe。

背景（见 docs/labcoat_si_audit.md）:
    labcoat_si 566 张中，129 张与 B3 同源已剔除，437 张审计通过可入库。
    目的：补充项目最缺的 lab_coat 类（dataset1 转换后仅 1696 框，真实场景效果差）。

合并规则（组长确认版）:
    1. 只并入 train split，不动 data/lab_ppe 的 valid/test
       —— 保持 dataset1 的验证/测试口径不变，评估可比；
          且已核实 labcoat_si 与 dataset1/dataset2 零重复，不构成泄漏。
    2. 类别重映射：labcoat_si 的 lab_coat(1) -> 项目 lab_coat(2)
       类别表：0=face, 1=lab_coat          （labcoat_si）
               0=mask, 1=gloves, 2=lab_coat, 3=goggles （项目）
       face 类项目用不上，标注行丢弃（face-only 图变成无框负样本，仍保留图片）。
    3. 文件名加前缀 lcsi_ ，来源一眼可辨，也避免与 dataset1 同名冲突。
    4. 不修改 data/raw/ 原始数据；每行标注写入 manifest.csv 便于追溯。

用法:
    python tools/merge_labcoat_si.py --dry-run     # 只看统计，不动文件（建议先跑）
    python tools/merge_labcoat_si.py               # 实际合并

前置条件（顺序不能颠倒）:
    先跑 tools/convert_lab_ppe.py 生成 data/lab_ppe/ ，再跑本脚本。
    若之后重跑 convert_lab_ppe.py，manifest.csv 会被覆盖（合并记录丢失），
    需再跑一次本脚本补齐（图片已存在会自动跳过，幂等）。
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "raw" / "labcoat_si"
KEEP_LIST = ROOT / "data" / "labcoat_si_keep_images.txt"
DST = ROOT / "data" / "lab_ppe"
DEST_SPLIT = "train"
PREFIX = "lcsi_"
REPORT = ROOT / "data" / "labcoat_si_merge_report.txt"

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PROJECT_LAB_COAT = 2  # 项目 4 类体系里的 lab_coat


def load_src_names(src: Path) -> dict[int, str]:
    """读 labcoat_si 的 data.yaml，拿到 id -> 类名。"""
    y = src / "data.yaml"
    if not y.exists():
        sys.exit(f"[错误] 找不到 {y}")
    names = yaml.safe_load(y.read_text(encoding="utf-8")).get("names")
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    return {int(k): v for k, v in (names or {}).items()}


def parse_keep_list(path: Path) -> list[str]:
    """清单格式: <split>/<文件名>，忽略 # 注释与空行。"""
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        rel = ln.strip()
        if not rel or rel.startswith("#"):
            continue
        out.append(rel)
    return out


def existing_manifest_rows(dst: Path) -> set[str]:
    """已合并过的 new_filename 集合（用于幂等判断）。"""
    mf = dst / "manifest.csv"
    if not mf.exists():
        return set()
    with mf.open(encoding="utf-8-sig", newline="") as f:
        return {r["new_filename"] for r in csv.DictReader(f) if r.get("new_filename")}


def append_manifest(dst: Path, rows: list[list[str]]) -> None:
    mf = dst / "manifest.csv"
    header = ["source_dataset", "split", "original_filename", "new_filename",
              "original_class_id", "original_class_name", "new_class_id",
              "new_class_name", "action"]
    if mf.exists():
        with mf.open(encoding="utf-8-sig", newline="") as f:
            first = f.readline().strip().lstrip("\ufeff")
        if first:
            header = next(csv.reader([first]))
    new_file = not mf.exists()
    with mf.open("a" if not new_file else "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(header)
        w.writerows(rows)


def count_boxes(label_dir: Path) -> Counter:
    c = Counter()
    for lb in label_dir.glob("*.txt"):
        for line in lb.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) >= 5:
                c[int(p[0])] += 1
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC), help="labcoat_si 根目录")
    ap.add_argument("--keep-list", default=str(KEEP_LIST), help="入库清单（含 split/文件名）")
    ap.add_argument("--dst", default=str(DST), help="目标数据集根目录（含 images/labels 子目录）")
    ap.add_argument("--dest-split", default=DEST_SPLIT, choices=["train", "valid", "test"])
    ap.add_argument("--prefix", default=PREFIX, help="新文件名前缀")
    ap.add_argument("--dry-run", action="store_true", help="只统计不落盘")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    keep_rel = parse_keep_list(Path(args.keep_list))

    if not (dst / "images" / args.dest_split).exists():
        sys.exit(f"[错误] {dst}/images/{args.dest_split} 不存在，请先跑 tools/convert_lab_ppe.py")

    names = load_src_names(src)
    lab_id = next((i for i, n in names.items() if n.lower().replace(" ", "_") == "lab_coat"), None)
    if lab_id is None:
        sys.exit(f"[错误] labcoat_si 类别表里找不到 lab_coat：{names}")
    print(f"labcoat_si 类别表: {names}")
    print(f"重映射: lab_coat({lab_id}) -> 项目 lab_coat({PROJECT_LAB_COAT})，其余类别丢弃\n")

    before = count_boxes(dst / "labels" / args.dest_split)
    n_before_img = len(list((dst / "images" / args.dest_split).glob("*")))
    merged_names = existing_manifest_rows(dst)

    rows: list[list[str]] = []
    st = Counter()
    new_boxes = Counter()
    used_new = set()

    for rel in keep_rel:
        sp, _, fname = rel.rpartition("/")
        if not sp:
            st["bad_entry"] += 1
            continue
        img = src / sp / "images" / fname
        if not img.exists():
            st["missing_image"] += 1
            print(f"  [警告] 图片不存在，跳过: {rel}")
            continue

        new_name = f"{args.prefix}{img.name}"
        if new_name in used_new:
            st["dup_name"] += 1
            print(f"  [警告] 新文件名重复，跳过: {new_name}")
            continue
        used_new.add(new_name)

        if new_name in merged_names:
            st["already_merged"] += 1
            continue

        # 解析标注：只留 lab_coat，映射成项目 id
        lbl = src / sp / "labels" / (img.stem + ".txt")
        out_lines, dropped = [], []
        if lbl.exists():
            for i, line in enumerate(lbl.read_text(encoding="utf-8").splitlines(), 1):
                p = line.split()
                if len(p) < 5:
                    st["malformed"] += 1
                    continue
                cid = int(p[0])
                if cid == lab_id:
                    out_lines.append(f"{PROJECT_LAB_COAT} " + " ".join(p[1:5]))
                    new_boxes["lab_coat"] += 1
                else:
                    dropped.append((cid, names.get(cid, "?")))
        else:
            st["missing_label"] += 1

        if not args.dry_run:
            shutil.copy2(img, dst / "images" / args.dest_split / new_name)
            (dst / "labels" / args.dest_split / (Path(new_name).stem + ".txt")).write_text(
                "\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8"
            )

        for cid, cname in dropped:
            rows.append(["labcoat_si", sp, img.name, new_name, str(cid), cname, "", "", "drop_unused_class"])
            st["dropped_boxes"] += 1
        if out_lines:
            rows.append(["labcoat_si", sp, img.name, new_name, str(lab_id),
                         names.get(lab_id, "lab_coat"), str(PROJECT_LAB_COAT), "lab_coat", "keep"])
        else:
            rows.append(["labcoat_si", sp, img.name, new_name, "", "无标注", "", "", "keep_as_negative"])
            st["negative_samples"] += 1
        st["merged_images"] += 1

    if not args.dry_run and rows:
        append_manifest(dst, rows)

    after = count_boxes(dst / "labels" / args.dest_split) if not args.dry_run else before
    n_after_img = len(list((dst / "images" / args.dest_split).glob("*"))) if not args.dry_run else n_before_img

    names_4 = {0: "mask", 1: "gloves", 2: "lab_coat", 3: "goggles"}
    lines = []
    lines.append("labcoat_si 合并报告（并入 data/lab_ppe/train）")
    lines.append("=" * 52)
    lines.append(f"模式: {'DRY-RUN 仅统计，未落盘' if args.dry_run else '实际合并'}")
    lines.append(f"入库清单: {args.keep_list}（{len(keep_rel)} 张）")
    lines.append("")
    lines.append(f"成功并入图片:        {st['merged_images']} 张")
    lines.append(f"  其中无框负样本:    {st['negative_samples']} 张（原图为纯 face 或空标注）")
    lines.append(f"新增 lab_coat 框:    {new_boxes['lab_coat']} 个")
    lines.append(f"丢弃的 face 框:      {st['dropped_boxes']} 个")
    lines.append(f"已存在跳过(幂等):    {st['already_merged']} 张")
    lines.append("")
    lines.append("异常计数:")
    for k in ("missing_image", "missing_label", "malformed", "dup_name", "bad_entry"):
        if st[k]:
            lines.append(f"  {k}: {st[k]}")
    lines.append("")
    lines.append(f"train split 图片数:   {n_before_img} -> {n_after_img}")
    lines.append("train split 各类框数（合并后）:")
    for cid in sorted(names_4):
        b, a = before.get(cid, 0), after.get(cid, 0)
        lines.append(f"  {cid} {names_4[cid]:9s} {b:6d} -> {a:6d}  (+{a - b})")
    if args.dry_run:
        lines.append("")
        lines.append("（dry-run 模式，上面合并后的数字等同预演值，实际未写入）")

    text = "\n".join(lines)
    print("\n" + text)
    if not args.dry_run:
        REPORT.write_text(text + "\n", encoding="utf-8")
        print(f"\n报告已写入: {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
