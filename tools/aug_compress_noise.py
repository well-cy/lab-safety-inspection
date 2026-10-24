"""强压缩 / 噪声增广：模拟真实场景的画质退化。

背景：09-28 演示实测显示，真实巡检图大量来自微信/QQ 传图、摄像头抓拍、网图反复转存，
这些图带有肉眼可见的 JPEG 压缩伪影（"彩虹噪点"）。E4 模型在 10841 张高清训练图上
收敛，遇到这类退化图时弱目标（lab_coat / mask）置信度会从 0.8+ 掉到 0.05 以下直接丢失。
本脚本对现有 train 集做强压缩 + 噪声 + 分辨率重采样的离线增广，不改变标注。

设计约束（遵守项目数据纪律）：
- 只增广 train；valid / test 一律不动（保持评测口径可比较）
- 只改像素、不改标注：标签文件逐行原样复制
- 输出到独立目录，确认无误后再合并进主数据集；不删除、不覆盖任何原文件
- 幂等：目标已存在即跳过，可中断后重跑续做
- 默认干跑（--dry-run），不加参数不会写任何文件

用法：
    PY=.venv/Scripts/python.exe
    # 1) 干跑：只看会生成多少张
    $PY tools/aug_compress_noise.py --ratio 0.3 --dry-run
    # 2) 生成到独立目录
    $PY tools/aug_compress_noise.py --ratio 0.3 --out data/lab_ppe_aug
    # 3) 合并进主数据集 train（复制，原文件不动；文件名带 _augcq 不会冲突）
    $PY tools/aug_compress_noise.py --merge-from data/lab_ppe_aug
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = ROOT / "data" / "lab_ppe"
SUFFIX = "_augcq"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def degrade(img: np.ndarray, rng: random.Random, quality: tuple[int, int],
            noise: tuple[float, float], scale: tuple[float, float]) -> np.ndarray:
    """随机组合退化：分辨率降采样 -> 高斯噪声 -> 强 JPEG 压缩"""
    out = img
    # 1) 降采样再放大（模拟低分辨率抓拍 / 网图缩放）
    s = rng.uniform(*scale)
    if s < 0.98:
        h, w = out.shape[:2]
        small = cv2.resize(out, (max(8, int(w * s)), max(8, int(h * s))),
                           interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    # 2) 高斯噪声（模拟传感器噪声 / 反复转存的色度噪点）
    sigma = rng.uniform(*noise)
    if sigma > 0:
        noise_map = np.random.normal(0, sigma, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise_map, 0, 255).astype(np.uint8)
    # 3) 强 JPEG 压缩（真正的杀手：15~45 质量下块效应+振铃伪影）
    q = rng.randint(*quality)
    ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if ok:
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return out


def collect(src: Path, splits: list[str]) -> list[tuple[Path, Path]]:
    """返回 [(图片路径, 对应标签路径)]，标签缺失的图片跳过"""
    pairs: list[tuple[Path, Path]] = []
    for split in splits:
        img_dir = src / "images" / split
        lab_dir = src / "labels" / split
        if not img_dir.is_dir():
            print(f"  [跳过] 目录不存在: {img_dir}")
            continue
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() not in IMG_EXT:
                continue
            lab = lab_dir / f"{p.stem}.txt"
            pairs.append((p, lab if lab.exists() else None))
    return pairs


def run_generate(args) -> int:
    src = Path(args.src).resolve()
    out = Path(args.out).resolve()
    rng = random.Random(args.seed)

    pairs = collect(src, args.splits)
    if not pairs:
        print("没有找到可增广的图片，检查 --src 路径")
        return 1

    n_target = int(len(pairs) * args.ratio)
    sample = rng.sample(pairs, min(n_target, len(pairs)))
    print(f"源图 {len(pairs)} 张 | 增广比例 {args.ratio:.0%} → 计划生成 {len(sample)} 张")

    if args.dry_run:
        print("\n[dry-run] 未写任何文件。示例输出：")
        for img, lab in sample[:5]:
            tag = "有标签" if lab else "无标签(跳过该图)"
            print(f"  {img.name} -> {out.name}/images/.../{img.stem}{SUFFIX}.jpg  [{tag}]")
        if not args.limit:
            print(f"  ... 共 {len(sample)} 张")
        print(f"\n实际执行去掉 --dry-run 即可（输出目录: {out}）")
        return 0

    done = skip = fail = 0
    for i, (img_path, lab_path) in enumerate(sample):
        if args.limit and done >= args.limit:
            print(f"达到 --limit {args.limit}，停止")
            break
        if lab_path is None:
            skip += 1
            continue
        for split in args.splits:
            if split in img_path.parts:
                split_name = split
                break
        else:
            split_name = args.splits[0]

        dst_img_dir = out / "images" / split_name
        dst_lab_dir = out / "labels" / split_name
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lab_dir.mkdir(parents=True, exist_ok=True)
        dst_img = dst_img_dir / f"{img_path.stem}{SUFFIX}.jpg"
        dst_lab = dst_lab_dir / f"{img_path.stem}{SUFFIX}.txt"

        if dst_img.exists() and dst_lab.exists():
            skip += 1  # 幂等：已生成过
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            fail += 1
            print(f"  [警告] 读取失败，跳过: {img_path.name}")
            continue
        try:
            aug = degrade(img, rng, args.quality, args.noise, args.scale)
            cv2.imwrite(str(dst_img), aug,
                        [int(cv2.IMWRITE_JPEG_QUALITY), args.out_quality])
            dst_lab.write_text(lab_path.read_text(encoding="utf-8"), encoding="utf-8")
            done += 1
        except Exception as e:  # 单张失败不中断
            fail += 1
            print(f"  [警告] 处理失败 {img_path.name}: {e}")
        if (i + 1) % 500 == 0:
            print(f"  进度 {i + 1}/{len(sample)} | 生成 {done} 跳过 {skip} 失败 {fail}")

    print(f"\n完成：生成 {done} | 跳过（已存在/无标签）{skip} | 失败 {fail}")
    print(f"输出目录: {out}")
    print("\n下一步：确认抽检后合并进主数据集")
    print(f"  .venv/Scripts/python.exe tools/aug_compress_noise.py --merge-from {args.out}")
    return 0


def run_merge(args) -> int:
    src = Path(args.merge_from).resolve()
    dst = DEFAULT_SRC
    if not (src / "images").is_dir():
        print(f"源目录不含 images/：{src}")
        return 1
    total = 0
    for split in args.splits:
        simg, slab = src / "images" / split, src / "labels" / split
        dimg, dlab = dst / "images" / split, dst / "labels" / split
        if not simg.is_dir():
            continue
        dimg.mkdir(parents=True, exist_ok=True)
        dlab.mkdir(parents=True, exist_ok=True)
        for p in sorted(simg.iterdir()):
            if p.suffix.lower() not in IMG_EXT:
                continue
            lab = slab / f"{p.stem}.txt"
            if not lab.exists():
                print(f"  [跳过] 无标签: {p.name}")
                continue
            t_img, t_lab = dimg / p.name, dlab / f"{p.stem}.txt"
            if t_img.exists():
                continue  # 幂等
            if args.dry_run:
                continue
            shutil.copy2(p, t_img)
            shutil.copy2(lab, t_lab)
            total += 1
    if args.dry_run:
        print(f"[dry-run] 将合并 {sum(1 for s in args.splits if (src / 'images' / s).is_dir())} 个 split 的增广图到 {dst}")
    else:
        print(f"合并完成：{total} 张（原文件未被修改或删除）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="强压缩/噪声增广（只增广 train，不改标注）")
    ap.add_argument("--src", default=str(DEFAULT_SRC), help="源数据集根目录（含 images/ labels/）")
    ap.add_argument("--out", default=str(ROOT / "data" / "lab_ppe_aug"), help="输出目录")
    ap.add_argument("--splits", nargs="+", default=["train"], help="处理的 split，默认仅 train")
    ap.add_argument("--ratio", type=float, default=0.3, help="增广比例，默认 0.3（30%%）")
    ap.add_argument("--quality", nargs=2, type=int, default=[15, 45], metavar=("MIN", "MAX"),
                    help="模拟的 JPEG 质量区间，默认 15 45（越低越糊）")
    ap.add_argument("--out-quality", type=int, default=70, help="落盘时的 JPEG 质量，默认 70")
    ap.add_argument("--noise", nargs=2, type=float, default=[0, 18], metavar=("MIN", "MAX"),
                    help="高斯噪声 sigma 区间，默认 0 18")
    ap.add_argument("--scale", nargs=2, type=float, default=[0.45, 0.85], metavar=("MIN", "MAX"),
                    help="降采样比例区间，默认 0.45 0.85")
    ap.add_argument("--seed", type=int, default=42, help="随机种子，保证可复现")
    ap.add_argument("--limit", type=int, default=0, help="最多生成多少张（0=不限，试跑用）")
    ap.add_argument("--dry-run", action="store_true", help="只统计不写文件")
    ap.add_argument("--merge-from", default=None, help="把已生成的增广目录合并进主数据集")
    args = ap.parse_args()

    # 数据安全：禁止把增广结果写回源目录
    if args.merge_from:
        return run_merge(args)
    if Path(args.out).resolve() == Path(args.src).resolve():
        print("拒绝执行：--out 不能等于 --src（会污染原始数据集）")
        return 2
    if "valid" in args.splits or "test" in args.splits:
        print("警告：valid/test 属于评测口径，增广它们会让指标不可比。")
        print("如确认要这么做，请手动修改脚本；脚本默认拒绝。")
        return 2
    return run_generate(args)


if __name__ == "__main__":
    sys.exit(main())
