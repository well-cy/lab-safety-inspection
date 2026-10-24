#!/usr/bin/env python3
"""crosslib_phash.py — 跨库 pHash 红线去重（四关审计第①关）

用途：外部数据集入库前，与红线库（data/raw/dataset1、data/raw/dataset2=external_test）
      以及主库 data/lab_ppe 做近重复比对。任何一张与红线库近重复的外部图都必须剔除，
      防止 external_test 泄漏进训练；与主库近重复的图则无增量价值，一并报告。

方法：src 库逐图算 pHash，against 库算 pHash 后以 numpy 向量化汉明距离暴力比对。
      命中（距离 <= thr）输出报告 + 剔除索引（<split>/<文件名>，喂 reject_by_list.py）。

用法：
    python tools/crosslib_phash.py --src data/raw/goggles_ppe \
        --against data/raw/dataset1,data/raw/dataset2,data/lab_ppe \
        --thr 8 --report data/goggles_ppe_crosslib_report.txt \
        --index data/goggles_ppe_crosslib_index.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def imread_unicode(p: Path):
    try:
        return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def phash(img, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    size = hash_size * highfreq_factor
    gray = cv2.cvtColor(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(np.float32(gray))
    low = dct[:hash_size, :hash_size].flatten()
    med = np.median(low[1:])
    h = 0
    for b in (low > med).astype(np.uint64):
        h = (h << 1) | int(b)
    return h


def list_images(root: Path) -> list[Path]:
    """递归列出一个根下所有图片（不关心标注布局）。"""
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMG_EXT
                  and not p.name.startswith("_"))


def split_of(ip: Path, src_root: Path) -> str:
    """raw 布局取 <split> 段，取不到就用父目录名。"""
    try:
        rel = ip.relative_to(src_root)
        return rel.parts[0] if len(rel.parts) > 1 else "."
    except ValueError:
        return "."


POP = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)


def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (n,) uint64, b: (m,) uint64 -> (n, m) uint8 汉明距离，分块防爆内存。"""
    out = np.empty((len(a), len(b)), dtype=np.uint8)
    for i in range(0, len(a), 256):
        x = a[i:i + 256, None] ^ b[None, :]
        d = POP[(x & np.uint64(0xFFFF)).astype(np.uint64)].astype(np.uint8)
        d += POP[((x >> np.uint64(16)) & np.uint64(0xFFFF)).astype(np.uint64)]
        d += POP[((x >> np.uint64(32)) & np.uint64(0xFFFF)).astype(np.uint64)]
        d += POP[((x >> np.uint64(48)) & np.uint64(0xFFFF)).astype(np.uint64)]
        out[i:i + 256] = d
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="待审库根目录")
    ap.add_argument("--against", required=True,
                    help="逗号分隔的对照库根目录（红线库在前）")
    ap.add_argument("--thr", type=int, default=8,
                    help="pHash 汉明距离阈值（默认 8；红线比对可放宽容错）")
    ap.add_argument("--report", default="", help="报告输出路径")
    ap.add_argument("--index", default="", help="剔除索引输出路径")
    args = ap.parse_args()

    src_root = Path(args.src)
    against_roots = [Path(p) for p in args.against.split(",") if p.strip()]

    src_imgs = list_images(src_root)
    print(f"[src] {src_root}: {len(src_imgs)} 张，计算 pHash…")
    src_hashes, src_ok = [], []
    for i, ip in enumerate(src_imgs):
        img = imread_unicode(ip)
        if img is None:
            continue
        src_hashes.append(phash(img))
        src_ok.append(ip)
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(src_imgs)}")
    H_src = np.array([np.uint64(h) for h in src_hashes], dtype=np.uint64)

    report: list[str] = []
    idx_lines: list[str] = []
    n = 0
    for ar in against_roots:
        a_imgs = list_images(ar)
        if not a_imgs:
            print(f"[against] {ar}: 0 张，跳过")
            continue
        print(f"[against] {ar}: {len(a_imgs)} 张，计算 pHash…")
        a_hashes = []
        for i, ip in enumerate(a_imgs):
            img = imread_unicode(ip)
            if img is not None:
                a_hashes.append(phash(img))
            else:
                a_hashes.append(None)
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{len(a_imgs)}")
        keep = [j for j, h in enumerate(a_hashes) if h is not None]
        H_a = np.array([np.uint64(a_hashes[j]) for j in keep], dtype=np.uint64)
        a_imgs_ok = [a_imgs[j] for j in keep]

        D = hamming_matrix(H_src, H_a)
        hit_rows, hit_cols = np.where(D <= args.thr)
        print(f"  命中（<= {args.thr}）: {len(hit_rows)} 对")

        reported: set[int] = set()
        for r, c in zip(hit_rows, hit_cols):
            report.append(f"[{ar.name}] dist={int(D[r, c])}  "
                          f"src={src_ok[r].relative_to(src_root)}  <->  "
                          f"{a_imgs_ok[c].relative_to(ar)}")
            if r not in reported:
                reported.add(r)
                n += 1
                idx_lines.append(f"{n}\t{split_of(src_ok[r], src_root)}/{src_ok[r].name}"
                                 f"\tcrosslib:{ar.name}")

    n_hit_imgs = len({ln.split("\t")[1] for ln in idx_lines})
    out_rep = Path(args.report) if args.report else (ROOT / "data" / "crosslib_phash_report.txt")
    out_rep.write_text(
        f"# 跨库 pHash 比对报告 —— src={src_root}  against={args.against}  阈值<={args.thr}\n"
        f"# src 共 {len(src_ok)} 张，命中 {n_hit_imgs} 张需处理\n\n"
        + ("\n".join(report) + "\n" if report else "(无命中)\n"), encoding="utf-8")
    print(f"\n[报告] {out_rep}")

    if args.index:
        ip = Path(args.index)
        if idx_lines:
            ip.write_text("# 跨库红线近重复剔除清单\n" + "\n".join(idx_lines) + "\n",
                          encoding="utf-8")
            print(f"[剔除索引] {ip}  ({len(idx_lines)} 行)")
        else:
            print("[剔除索引] 无命中，未写索引文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
