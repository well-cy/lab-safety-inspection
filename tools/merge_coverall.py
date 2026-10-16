"""ppedetect Coverall 入库合并：并入新图 + 给已入库同源图补 lab_coat 框。

背景（2026-09-28 用户拍板"连体防护服并入"）:
    E2 模型对"连体防护服"零识别（当初建数据特意剔除了 Coverall）。
    ppedetect（saba/ppedetection v7，CC BY 4.0）train 364 张里 329 张含 Coverall。

干跑实测（2026-09-28）:
    300/329 张与 data/lab_ppe 现有图片 MD5 完全相同（早年手套合并已入库），
    且经逐框比对两边为同一套标注——但这些图现有标注缺 Coverall 框，
    等于教模型"穿防护服=没穿实验服"，必须补框。
    29 张为全新图片，直接并入。

规则:
    1. 全新图（MD5 不在库中）: 映射后并入 train（cov_ 前缀）
         Coverall->2 lab_coat, Gloves->1 gloves, Goggles->3 goggles,
         Mask->0 mask, Face_Shield 丢弃
    2. 已在库中的图（MD5 相同）:
         - 仅当其在 train split 时，把缺失的 Coverall->lab_coat 框
           （6 位小数）补进现有标注文件（逐行精确比对，幂等）；
           改前先备份到 data/_label_backup_coverall_<日期>/
         - valid/test 一律不碰（口径可比 + external_test 红线），只记录
    3. 每个框一行写入 manifest.csv（action=keep/patch/drop）
    4. 不修改 data/raw/ 原始数据

用法:
    python tools/merge_coverall.py --dry-run   # 只看统计，不动文件
    python tools/merge_coverall.py             # 实际执行
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from collections import Counter
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "raw" / "ppedetect"
DST = ROOT / "data" / "lab_ppe"
DEST_SPLIT = "train"
PREFIX = "cov_"
REPORT = ROOT / "data" / "coverall_merge_report.txt"
MANIFEST = DST / "manifest.csv"
BACKUP = ROOT / "data" / f"_label_backup_coverall_{date.today():%Y-%m-%d}"

CLASS_MAP = {
    "Coverall": (2, "lab_coat"),
    "Gloves": (1, "gloves"),
    "Goggles": (3, "goggles"),
    "Mask": (0, "mask"),
    "Face_Shield": None,
}
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def md5_of(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt_box(vals: list[str]) -> str:
    """与现有标注一致的 6 位小数格式。"""
    return " ".join(f"{float(v):.6f}" for v in vals)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    names_raw = yaml.safe_load((SRC / "data.yaml").read_text(encoding="utf-8"))["names"]
    src_names = {i: n for i, n in enumerate(names_raw)}

    print("[1/4] 扫描现有图片 MD5 ...")
    existing_md5: dict[str, tuple[str, Path, Path]] = {}  # digest -> (img_name, img_path, label_path)
    for split_dir in sorted((DST / "images").iterdir()):
        if not split_dir.is_dir():
            continue
        split = split_dir.name
        for p in split_dir.iterdir():
            if p.suffix.lower() in IMG_EXT:
                lbl = DST / "labels" / split / (p.stem + ".txt")
                existing_md5[md5_of(p)] = (p.name, p, lbl)
    print(f"      现有图片 {len(existing_md5)} 张")

    print("[2/4] 扫描 ppedetect/train 标注 ...")
    jobs = []
    total = 0
    for lp in sorted((SRC / "train" / "labels").glob("*.txt")):
        total += 1
        lines = [l.split() for l in lp.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not any(len(p) == 5 and src_names.get(int(float(p[0]))) == "Coverall" for p in lines):
            continue
        ip = SRC / "train" / "images" / (lp.stem + ".jpg")
        if not ip.exists():
            alt = [q for q in (SRC / "train" / "images").glob(lp.stem + ".*")
                   if q.suffix.lower() in IMG_EXT]
            if not alt:
                print(f"      [警告] 有标注无图: {lp.name}")
                continue
            ip = alt[0]
        jobs.append((ip, lines))
    print(f"      train {total} 张，含 Coverall 的 {len(jobs)} 张")

    print("[3/4] 分类处理 ...")
    stats: Counter = Counter()
    box_stats: Counter = Counter()
    md5_dup: list[str] = []
    manifest_rows: list[list[str]] = []
    new_files: list[tuple[Path, Path, str]] = []          # (src_img, dst_img, label_txt)
    patches: list[tuple[Path, str]] = []                  # (existing_label, append_line)

    for ip, lines in jobs:
        digest = md5_of(ip)
        cov_lines = [p for p in lines
                     if len(p) == 5 and src_names.get(int(float(p[0]))) == "Coverall"]
        if digest not in existing_md5:
            new_name = PREFIX + ip.name
            out, skip = [], False
            for p in lines:
                if len(p) != 5:
                    stats["坏标注行"] += 1
                    continue
                src_name = src_names.get(int(float(p[0])))
                mapped = CLASS_MAP.get(src_name)
                if mapped is None:
                    manifest_rows.append(["ppedetect", DEST_SPLIT, ip.name, new_name,
                                          p[0], src_name or "?", "", "", "drop"])
                    box_stats["drop:" + (src_name or "?")] += 1
                    continue
                new_id, new_nm = mapped
                out.append(" ".join([str(new_id), fmt_box(p[1:])]))
                manifest_rows.append(["ppedetect", DEST_SPLIT, ip.name, new_name,
                                      p[0], src_name, str(new_id), new_nm, "keep"])
                box_stats["keep:" + new_nm] += 1
            if skip:
                continue
            dst_img = DST / "images" / DEST_SPLIT / new_name
            dst_lbl = DST / "labels" / DEST_SPLIT / (new_name.rsplit(".", 1)[0] + ".txt")
            new_files.append((ip, dst_img, "\n".join(out)))
            existing_md5[digest] = (new_name, dst_img, dst_lbl)  # 源内部同图去重
            stats["新图并入"] += 1
        else:
            img_name, _, lbl_path = existing_md5[digest]
            split = lbl_path.parent.name
            if split != DEST_SPLIT:
                stats[f"同图在{split}跳过"] += 1
                md5_dup.append(f"[{split}] {ip.name} == {img_name}（不碰 {split}）")
                continue
            have = set(lbl_path.read_text(encoding="utf-8").splitlines()) if lbl_path.exists() else set()
            added = 0
            for p in cov_lines:
                line = "2 " + fmt_box(p[1:])
                if line in have:
                    continue
                patches.append((lbl_path, line))
                manifest_rows.append(["ppedetect", split, ip.name, img_name,
                                      p[0], "Coverall", "2", "lab_coat", "patch"])
                have.add(line)
                added += 1
                box_stats["patch:lab_coat"] += 1
            md5_dup.append(f"[train] {ip.name} == {img_name}（补 lab_coat 框 {added} 个）")
            stats["同图补框"] += 1

    rep = ["ppedetect Coverall 合并报告（并入/补框 data/lab_ppe/train）",
           "=" * 56,
           f"模式: {'DRY-RUN（未写任何文件）' if args.dry_run else '实际执行'}",
           f"ppedetect train {total} 张，含 Coverall {len(jobs)} 张",
           f"新图并入: {stats['新图并入']}    同图补框: {stats['同图补框']}",
           f"valid/test 同图跳过: {stats['同图在valid跳过'] + stats['同图在test跳过']}",
           f"坏标注行: {stats['坏标注行']}",
           "框统计: " + "  ".join(f"{k}={v}" for k, v in sorted(box_stats.items())),
           ""]
    rep += [f"  {x}" for x in md5_dup[:40]]
    if len(md5_dup) > 40:
        rep.append(f"  ...（其余 {len(md5_dup) - 40} 条见 manifest）")
    print("\n".join(rep))

    if args.dry_run:
        print("(dry-run 未写任何文件)")
        return

    print("[4/4] 写入 ...")
    if patches:
        BACKUP.mkdir(parents=True, exist_ok=True)
        done: set[Path] = set()
        for lbl, line in patches:
            if lbl not in done:
                shutil.copy2(lbl, BACKUP / lbl.name)
                done.add(lbl)
        for lbl, line in patches:
            with lbl.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        print(f"      补框 {len(patches)} 个（原文件备份在 {BACKUP.name}/）")
    for ip, dst_img, txt in new_files:
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ip, dst_img)
        (DST / "labels" / DEST_SPLIT).mkdir(parents=True, exist_ok=True)
        (DST / "labels" / DEST_SPLIT / (dst_img.stem + ".txt")).write_text(
            txt + ("\n" if txt else ""), encoding="utf-8")
    if manifest_rows:
        with MANIFEST.open("a", encoding="utf-8-sig", newline="") as f:
            csv.writer(f).writerows(manifest_rows)
    rep += ["", f"完成: 新图 {len(new_files)} 张，补框 {len(patches)} 个",
            f"备份: {BACKUP}"]
    (BACKUP.parent / "coverall_merge_report.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
    print(f"完成: 新图 {len(new_files)} 张 / 补框 {len(patches)} 个 → 报告 {REPORT}")


if __name__ == "__main__":
    main()
