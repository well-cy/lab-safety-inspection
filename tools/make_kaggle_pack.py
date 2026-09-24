# -*- coding: utf-8 -*-
"""打包 lab_ppe 数据集为 Kaggle 可直接上传的骨架目录 + zip。

产出（默认 outputs/kaggle_upload/）：
  lab_ppe_kaggle/                 <- 目录骨架，可整体上传为 Kaggle Dataset
    data.yaml                     <- path 已改为 Kaggle 挂载路径
    images/{train,valid,test}/*   <- 硬链接（不占额外空间）
    labels/{train,valid,test}/*
  lab_ppe_kaggle.zip              <- 若加 --zip，额外产出单文件压缩包

用法：
  # 1) 先干跑看体积
  .venv/Scripts/python.exe tools/make_kaggle_pack.py
  # 2) 确认后落盘（--apply），需要单文件再加 --zip
  .venv/Scripts/python.exe tools/make_kaggle_pack.py --apply --zip

说明：
- 图片用**硬链接**（同盘瞬时完成、不占空间）；若跨盘会自动回退为复制
- **不含 test 的用途限制**：Kaggle 训练时只用 train/valid；test 保留在同一包里
  仅供训完自查，禁止用于调参（与项目 external_test 红线一致）
- data.yaml 的 path 写成 /kaggle/input/lab-ppe/lab_ppe_kaggle，
  上传后数据集 slug 若不是 lab-ppe，需改这一行
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "lab_ppe"
OUT = ROOT / "outputs" / "kaggle_upload"
NAME = "lab_ppe_kaggle"
KAGGLE_PATH = "/kaggle/input/lab-ppe/lab_ppe_kaggle"

SPLITS = ("train", "valid", "test")
EXTS = (".jpg", ".jpeg", ".png", ".bmp")

DATA_YAML = f"""# Lab-PPE 4 类检测数据集（Kaggle 版）
# 上传为 Kaggle Dataset 后，slug 若不是 lab-ppe，请同步改下面的 path
path: {KAGGLE_PATH}
train: images/train
val: images/valid
test: images/test
nc: 4
names:
  0: mask
  1: gloves
  2: lab_coat
  3: goggles
"""


def link_or_copy(src: Path, dst: Path) -> str:
    """优先硬链接；失败则复制。返回 'link' / 'copy' / 'skip'。

    幂等：目标已存在且大小一致时直接跳过（不做任何删除，
    避免触发沙箱批量删除护栏）。
    """
    if dst.exists():
        try:
            if dst.stat().st_size == src.stat().st_size:
                return "skip"
        except OSError:
            pass
        return "skip"          # 已在位，交给人工核对，不删不覆盖
    try:
        dst.hardlink_to(src)
        return "link"
    except Exception:  # noqa: BLE001  跨盘/权限等
        shutil.copy2(src, dst)
        return "copy"


def main() -> int:
    ap = argparse.ArgumentParser(description="打包 lab_ppe 供 Kaggle 上传")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--apply", action="store_true", help="真正落盘（默认只统计）")
    ap.add_argument("--zip", action="store_true", help="额外产出 zip 单文件")
    args = ap.parse_args()

    out = Path(args.out)
    root = out / NAME
    if not SRC.is_dir():
        print(f"[错误] 找不到数据集: {SRC}")
        return 2

    plan = {}
    for sp in SPLITS:
        imgs = [p for p in (SRC / "images" / sp).iterdir()
                if p.is_file() and p.suffix.lower() in EXTS]
        labs = [p for p in (SRC / "labels" / sp).glob("*.txt")]
        size = sum(p.stat().st_size for p in imgs)
        plan[sp] = (imgs, labs, size)

    print("=" * 66)
    print(f"{'split':<8}{'图片':>8}{'标签':>8}{'体积MB':>10}")
    print("=" * 66)
    tot_i = tot_l = 0
    tot_s = 0
    for sp in SPLITS:
        imgs, labs, size = plan[sp]
        print(f"{sp:<8}{len(imgs):>8}{len(labs):>8}{size/1048576:>10.1f}")
        tot_i += len(imgs); tot_l += len(labs); tot_s += size
    print("-" * 66)
    print(f"{'合计':<8}{tot_i:>8}{tot_l:>8}{tot_s/1048576:>10.1f}")
    print()

    # 图文配对检查
    bad = 0
    for sp in SPLITS:
        imgs, labs, _ = plan[sp]
        si = {p.stem for p in imgs}
        sl = {p.stem for p in labs}
        d = len(si ^ sl)
        if d:
            bad += d
            print(f"[警告] {sp}: 图文不配对 {d} 个")
    if not bad:
        print("图文配对: 全部一致")
    print(f"\n目标目录: {root}")
    print(f"data.yaml path = {KAGGLE_PATH}")

    if not args.apply:
        print("\n[干跑] 未落盘。确认无误后加 --apply --zip")
        return 0

    # ---------- 落盘 ----------
    stats = {"link": 0, "copy": 0, "skip": 0}
    for sp in SPLITS:
        imgs, labs, _ = plan[sp]
        (root / "images" / sp).mkdir(parents=True, exist_ok=True)
        (root / "labels" / sp).mkdir(parents=True, exist_ok=True)
        for p in imgs:
            stats[link_or_copy(p, root / "images" / sp / p.name)] += 1
        for p in labs:
            stats[link_or_copy(p, root / "labels" / sp / p.name)] += 1
    (root / "data.yaml").write_text(DATA_YAML, encoding="utf-8")
    print(f"\n骨架完成：新建硬链接 {stats['link']} / 复制 {stats['copy']} / 已存在跳过 {stats['skip']}")

    if args.zip:
        zp = out / f"{NAME}.zip"
        n = 0
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as z:
            for p in sorted(root.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(out))
                    n += 1
        mb = zp.stat().st_size / 1048576
        print(f"压缩包: {zp}  ({n} 个文件, {mb:.1f} MB, ZIP_DEFLATED)")
        print("  注：图片本身已是 JPEG，压缩率有限；如需更小可用 7z 分卷")

    print(f"\n[完成] 上传位置: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
