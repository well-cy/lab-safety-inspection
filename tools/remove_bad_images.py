#!/usr/bin/env python3
"""remove_bad_images.py — 按人工复查结论从 data/lab_ppe 剔除坏标注图（可干跑、可回滚）

配合流程：make_contact_sheet.py 拼图版 → 人工看图回报编号 → 本脚本执行剔除。

编号来源：make_contact_sheet.py 输出的 index.txt（编号 <TAB> split/文件名 <TAB> 标签）。
判定输入二选一：
  --pass  "1,2,5-9"    人工判**合格**的编号，其余全部剔除（她常用的回报格式）
  --reject "1,2,5-9"   人工判**不合格**的编号，只剔这些

家族扩展（默认开）：Roboflow 把同一张原图导出成多个增强副本，文件名形如
  <原名>.rf.<hash>.jpg。若某副本被判错，同一原图的其它副本几乎必然同错
  （同图同标注，只是增强不同），所以按 .rf. 前缀跨 split 一起剔。
  用 --no-expand-family 可关闭。

安全设计：
  * 默认 dry-run，只打印影响面，不动任何文件；加 --apply 才真正执行
  * 剔除 = **移动**到 data/_rejected/<标记>/（保留 split 结构），不是删除；
    原始下载还在 data/raw/，随时可整体恢复
  * 同时报告连带损失：被剔图里其它类别的框也会一起消失，干跑里会列清楚

用法：
  # 干跑（推荐先跑）
  python tools/remove_bad_images.py --index outputs/gloves_review/flags/index.txt \
      --pass "2,3,7,10,88-125" --tag gloves_review

  # 确认无误后执行
  python tools/remove_bad_images.py --index outputs/gloves_review/flags/index.txt \
      --pass "2,3,7,10,88-125" --tag gloves_review --apply
"""
from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DS = ROOT / "data" / "lab_ppe"
NAMES = {0: "mask", 1: "gloves", 2: "lab_coat", 3: "goggles"}
IMG_EXT = {".jpg", ".jpeg", ".png"}


def parse_nums(spec: str) -> set[int]:
    """解析 "1,2,5-9,12" 这类编号串。"""
    out: set[int] = set()
    for tok in spec.replace("，", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(tok))
    return out


def load_index(path: Path) -> dict[int, str]:
    idx2rel: dict[int, str] = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split("\t")
        idx2rel[int(parts[0])] = parts[1].strip()
    return idx2rel


def scan_dataset(ds: Path) -> dict[str, dict[str, Path]]:
    """split -> {stem: 图片路径}"""
    out: dict[str, dict[str, Path]] = {}
    for idir in sorted((ds / "images").glob("*")):
        if not idir.is_dir():
            continue
        out[idir.name] = {p.stem: p for p in idir.iterdir() if p.suffix.lower() in IMG_EXT}
    return out


def count_boxes(ds: Path, splits: dict[str, dict[str, Path]]) -> Counter:
    c: Counter = Counter()
    for sp, stems in splits.items():
        for stem in stems:
            lp = ds / "labels" / sp / f"{stem}.txt"
            if not lp.exists():
                continue
            for line in lp.read_text(encoding="utf-8").splitlines():
                p = line.split()
                if len(p) == 5:
                    try:
                        c[int(float(p[0]))] += 1
                    except ValueError:
                        pass
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="make_contact_sheet 输出的 index.txt")
    ap.add_argument("--pass", dest="pass_spec", default="", help="合格编号串，其余剔除")
    ap.add_argument("--reject", dest="reject_spec", default="", help="不合格编号串，只剔这些")
    ap.add_argument("--dataset", default=str(DEFAULT_DS))
    ap.add_argument("--tag", default="review", help="回滚目录名/报告名前缀")
    ap.add_argument("--no-expand-family", action="store_true", help="不按 .rf. 原图家族扩展")
    ap.add_argument("--apply", action="store_true", help="真正执行移动（默认只干跑）")
    ap.add_argument("--reason", default="", help="剔除原因，写进报告")
    args = ap.parse_args()

    if bool(args.pass_spec) == bool(args.reject_spec):
        ap.error("--pass 和 --reject 必须二选一")

    idx2rel = load_index(Path(args.index))
    if not idx2rel:
        print(f"[错误] index 为空: {args.index}")
        return 1

    nums = parse_nums(args.pass_spec) | parse_nums(args.reject_spec)
    bad_idx = nums & set(idx2rel)
    unknown = sorted(nums - set(idx2rel))
    if unknown:
        print(f"[警告] 编号超出 index 范围(1-{max(idx2rel)})，已忽略: {unknown}")
    if args.pass_spec:                       # 合格 → 其余全剔
        reject = {i for i in idx2rel} - bad_idx
    else:                                    # 不合格 → 只剔这些
        reject = bad_idx
    if not reject:
        print("[空] 没有要剔除的编号")
        return 1

    ds = Path(args.dataset)
    splits = scan_dataset(ds)
    stem2split = {stem: sp for sp, m in splits.items() for stem in m}

    reject_items: dict[str, str] = {}        # split/name -> 来源(判定/家族)
    for i in sorted(reject):
        rel = idx2rel[i]
        sp, name = rel.split("/", 1)
        reject_items[f"{sp}/{name}"] = "人工判定"

    # 家族扩展
    extra: list[str] = []
    if not args.no_expand_family:
        fam_of = lambda name: name.split(".rf.")[0]
        bad_fams = {fam_of(n) for _, n in (r.split("/", 1) for r in reject_items)}
        for sp, m in splits.items():
            for stem, ip in m.items():
                key = f"{sp}/{ip.name}"
                if key in reject_items:
                    continue
                if fam_of(ip.name) in bad_fams:
                    reject_items[key] = "同原图家族扩展"
                    extra.append(key)

    print(f"判定输入: {'合格' if args.pass_spec else '不合格'} {len(nums)} 个编号"
          f" → 初步剔除 {len(reject)} 张"
          + (f" + 家族扩展 {len(extra)} 张" if extra else ""))
    print(f"合计拟剔除 {len(reject_items)} 张\n")

    # 影响面统计
    per_split: Counter = Counter()
    cls_lose: Counter = Counter()
    fam_cnt: Counter = Counter()
    neg_imgs: list[str] = []
    for key, src in reject_items.items():
        sp, name = key.split("/", 1)
        stem = Path(name).stem
        per_split[sp] += 1
        fam_cnt[Path(name).name.split(".rf.")[0]] += 1
        lp = ds / "labels" / sp / f"{stem}.txt"
        n_box = 0
        if lp.exists():
            for line in lp.read_text(encoding="utf-8").splitlines():
                p = line.split()
                if len(p) == 5:
                    try:
                        cls_lose[int(float(p[0]))] += 1
                        n_box += 1
                    except ValueError:
                        pass
        if n_box == 0:
            neg_imgs.append(key)

    all_boxes = count_boxes(ds, splits)
    print("按 split:".ljust(14) + "  ".join(f"{k}={v}" for k, v in sorted(per_split.items())))
    print("按类别框数:")
    for cid, cname in NAMES.items():
        lose = cls_lose.get(cid, 0)
        print(f"  {cname:<10} 剔 {lose:>5} / 现 {all_boxes.get(cid, 0):>5}"
              f"  → 剩 {all_boxes.get(cid, 0) - lose:>5}")
    if neg_imgs:
        print(f"无框负样本被剔 {len(neg_imgs)} 张: {neg_imgs[:5]}{' …' if len(neg_imgs) > 5 else ''}")
    print(f"\n原图家族 TOP12（共 {len(fam_cnt)} 个）:")
    for fam, n in fam_cnt.most_common(12):
        print(f"  {n:>3}  {fam}")

    if not args.apply:
        print("\n[干跑] 未改动任何文件。确认无误后加 --apply 执行。")
        return 0

    # 执行：移动到 data/_rejected/<tag>/<日期>/<split>/
    tag = f"{args.tag}_{date.today():%Y%m%d}"
    dest_root = ds.parent / "_rejected" / tag
    moved_img = moved_lbl = 0
    for key in reject_items:
        sp, name = key.split("/", 1)
        stem = Path(name).stem
        ddir = dest_root / "images" / sp
        ldir = dest_root / "labels" / sp
        ddir.mkdir(parents=True, exist_ok=True)
        ldir.mkdir(parents=True, exist_ok=True)
        ip = ds / "images" / sp / name
        lp = ds / "labels" / sp / f"{stem}.txt"
        if ip.exists():
            shutil.move(str(ip), str(ddir / name))
            moved_img += 1
        if lp.exists():
            shutil.move(str(lp), str(ldir / lp.name))
            moved_lbl += 1

    remain_imgs = sum(len(m) for m in splits.values()) - moved_img
    remain_boxes = count_boxes(ds, splits)
    lines = [
        f"剔除报告  tag={tag}  数据集={ds}",
        f"判定依据: {args.reason or '(见 index 与任务卡)'}",
        f"剔除图片 {moved_img} 张 / 标注 {moved_lbl} 份（已移动到 {dest_root}，非删除，可整体移回恢复）",
        f"其中家族扩展 {len(extra)} 张；无框负样本 {len(neg_imgs)} 张",
        "按 split: " + "  ".join(f"{k}={v}" for k, v in sorted(per_split.items())),
        "类别框数变化:",
        *[f"  {NAMES[cid]:<10} {all_boxes.get(cid,0):>5} → {remain_boxes.get(cid,0):>5}"
          f"（剔 {cls_lose.get(cid,0)}）" for cid in NAMES],
        f"剩余图片总数约 {remain_imgs}",
        "",
        "剔除清单（split/文件名<TAB>来源）:",
        *[f"{k}\t{v}" for k, v in sorted(reject_items.items())],
    ]
    rep = ds.parent / f"{args.tag}_remove_report.txt"
    rep.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[完成] 移动 {moved_img} 图 + {moved_lbl} 标注 → {dest_root}")
    print(f"报告: {rep}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
