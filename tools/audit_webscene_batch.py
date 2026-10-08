# -*- coding: utf-8 -*-
"""audit_webscene_batch.py —— web_lab_scene 48 张网图入库前四关审计（2026-10-08）

批次：data/raw/web_lab_scene/images（48 张，index=outputs/web_lab_scene_review/index.txt）
草稿：data/_prelabel/webscene_r1/labels（E6 五类 0=person/1=mask/2=gloves/3=lab_coat/4=goggles）

四关：
  ① 红线去重：与 data/raw/dataset1、data/raw/dataset2(external_test 红线)、data/lab_ppe 做 pHash 比对
  ② 批内近重复：48 张两两 pHash
  ③ 脏图筛查：由 tools/screen_dirty_graphics.py --dir 另跑（本脚本汇总其 TSV）
  ④ 标注几何审计：格式/越界/超小/超大面积/极端宽高比/同图同类重复框(IoU>0.9)/贴边框

注意：本脚本自己列图，**不跳过 "_" 开头文件名**（crosslib_phash.list_images 会跳过，
      而本批文件名恰好全部以 "_" 或中文开头，直接复用会漏检）。

用法: python tools/audit_webscene_batch.py
输出: data/webscene_batch_audit.txt
"""
from __future__ import annotations

import io
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "raw" / "web_lab_scene" / "images"
LBL = ROOT / "data" / "_prelabel" / "webscene_r1" / "labels"
INDEX = ROOT / "outputs" / "web_lab_scene_review" / "index.txt"
REPORT = ROOT / "data" / "webscene_batch_audit.txt"
DIRTY_TSV = ROOT / "outputs" / "webscene_dirty_stats.tsv"

REF_LIBS = {
    "dataset1": ROOT / "data" / "raw" / "dataset1",
    "dataset2(红线)": ROOT / "data" / "raw" / "dataset2",
    "lab_ppe(主库)": ROOT / "data" / "lab_ppe",
}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLS = {0: "person", 1: "mask", 2: "gloves", 3: "lab_coat", 4: "goggles"}

THR_CROSS = 8     # 跨库近重复阈值
THR_SELF = 8      # 批内近重复阈值
MIN_AREA = 0.0005
MAX_AREA = 0.90
ASPECT = 6.0
IOU_DUP = 0.90
EDGE_TOL = 0.02


def imread_u(p: Path):
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


POP = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.empty((len(a), len(b)), dtype=np.uint8)
    for i in range(0, len(a), 256):
        x = a[i:i + 256, None] ^ b[None, :]
        d = POP[(x & np.uint64(0xFFFF)).astype(np.uint64)].astype(np.uint8)
        d += POP[((x >> np.uint64(16)) & np.uint64(0xFFFF)).astype(np.uint64)]
        d += POP[((x >> np.uint64(32)) & np.uint64(0xFFFF)).astype(np.uint64)]
        d += POP[((x >> np.uint64(48)) & np.uint64(0xFFFF)).astype(np.uint64)]
        out[i:i + 256] = d
    return out


def list_all_images(root: Path):
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMG_EXT)


def load_index():
    out = []
    for line in INDEX.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("编号"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            out.append((int(parts[0]), parts[1]))
    return out


def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> int:
    rep: list[str] = []
    idx = load_index()
    rep.append(f"# web_lab_scene 批次入库前审计  {len(idx)} 图  (2026-10-08)")
    rep.append("")

    # ---- 载入源库 ----
    stems = {s for _, s in idx}
    src_files = [SRC / s for _, s in idx]
    missing = [p.name for p in src_files if not p.exists()]
    hashes, ok_names = [], []
    for p in src_files:
        img = imread_u(p)
        if img is None:
            missing.append(p.name)
            continue
        hashes.append(phash(img))
        ok_names.append(p.name)
    H = np.array([np.uint64(h) for h in hashes], dtype=np.uint64)
    rep.append(f"## 读图\n可用 {len(ok_names)} 张；缺失/不可读 {len(missing)} 张 {missing}")
    rep.append("")

    # ---- 第①关 红线去重 ----
    rep.append("## 第①关 红线/主库 pHash 去重 (thr<=8)")
    for name, rp in REF_LIBS.items():
        if not rp.exists():
            rep.append(f"  [{name}] 目录不存在，跳过")
            continue
        refs = list_all_images(rp)
        rh = []
        rn = []
        for p in refs:
            img = imread_u(p)
            if img is None:
                continue
            rh.append(phash(img))
            rn.append(p)
        if not rh:
            rep.append(f"  [{name}] 0 张可用，跳过")
            continue
        R = np.array([np.uint64(h) for h in rh], dtype=np.uint64)
        D = hamming(H, R)
        n_hit = int((D <= THR_CROSS).sum())
        rep.append(f"  [{name}] 对照 {len(rn)} 张，命中<= {THR_CROSS}: {n_hit} 对")
        for r, c in zip(*np.where(D <= THR_CROSS)):
            rep.append(f"      dist={int(D[r, c])}  {ok_names[r]}  <->  "
                       f"{rn[c].relative_to(rp)}")
        mins = D.min(axis=1)
        near = [(int(m), ok_names[i]) for i, m in enumerate(mins) if m <= 14]
        if near:
            rep.append(f"      最近邻(d<=14)提示: " +
                       "; ".join(f"{n}({d})" for d, n in sorted(near)))
    rep.append("")

    # ---- 第②关 批内近重复 ----
    rep.append("## 第②关 批内近重复 (thr<=8)")
    D2 = hamming(H, H)
    np.fill_diagonal(D2, 255)
    pairs = [(int(D2[i, j]), ok_names[i], ok_names[j])
             for i in range(len(ok_names)) for j in range(i + 1, len(ok_names))
             if D2[i, j] <= THR_SELF]
    if pairs:
        for d, a, b in sorted(pairs):
            rep.append(f"  dist={d}  {a}  <->  {b}")
    else:
        rep.append("  无（48 张两两 pHash 均 > 8）")
    rep.append("")

    # ---- 第④关 标注几何 ----
    rep.append("## 第④关 标注几何审计")
    per_cls = Counter()
    n_boxes = 0
    problems = []
    img_no_label = []
    sizes = []
    for num, stem in idx:
        stem_no_ext = Path(stem).stem
        lp = LBL / f"{stem_no_ext}.txt"
        if not lp.exists():
            img_no_label.append(f"#{num} {stem_no_ext}")
            continue
        rows = []
        for ln in lp.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            p = ln.split()
            if len(p) != 5:
                problems.append(f"#{num} 列数!=5: {ln}")
                continue
            c = int(p[0])
            cx, cy, w, h = (float(v) for v in p[1:5])
            if c not in CLS:
                problems.append(f"#{num} 非法类别 {c}")
            if not all(-0.001 <= v <= 1.001 for v in (cx, cy, w, h)):
                problems.append(f"#{num} 越界 {ln}")
            if w <= 0 or h <= 0:
                problems.append(f"#{num} 零尺寸 {ln}")
                continue
            per_cls[CLS[c]] += 1
            n_boxes += 1
            x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            area = w * h
            sizes.append((area, w / h))
            if area < MIN_AREA:
                problems.append(f"#{num} 超小框 area={area:.5f} {ln}")
            if area > MAX_AREA:
                problems.append(f"#{num} 超大框 area={area:.3f} {ln}")
            if w / h > ASPECT or h / w > ASPECT:
                problems.append(f"#{num} 极端宽高比 {w/h:.2f} {ln}")
            if min(x0, y0) < -EDGE_TOL or max(x1, y1) > 1 + EDGE_TOL:
                problems.append(f"#{num} 框越界图片 {ln}")
            rows.append((c, (x0, y0, x1, y1)))
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if rows[i][0] == rows[j][0] and iou(rows[i][1], rows[j][1]) > IOU_DUP:
                    problems.append(f"#{num} 同图同类重复框 IoU={iou(rows[i][1], rows[j][1]):.2f}")
    rep.append(f"  标签文件 {len(idx) - len(img_no_label)}/{len(idx)}；框 {n_boxes} 个")
    rep.append(f"  按类别: {dict(per_cls)}")
    if sizes:
        ar = np.array([a for a, _ in sizes])
        asp = np.array([r for _, r in sizes])
        rep.append(f"  面积: 中位 {np.median(ar):.4f}  min {ar.min():.5f}  max {ar.max():.4f}")
        rep.append(f"  宽高比(w/h): 中位 {np.median(asp):.2f}  min {asp.min():.2f}  max {asp.max():.2f}")
    if img_no_label:
        rep.append(f"  !! 缺标签: {img_no_label}")
    if problems:
        rep.append(f"  !! 问题 {len(problems)} 条:")
        for p in problems:
            rep.append(f"     {p}")
    else:
        rep.append("  ✅ 无几何问题")
    rep.append("")

    # ---- 第③关 汇总 ----
    rep.append("## 第③关 脏图筛查 (screen_dirty_graphics --dir)")
    if DIRTY_TSV.exists():
        lines = DIRTY_TSV.read_text(encoding="utf-8").splitlines()
        hdr = lines[0].split("\t")
        flags = Counter()
        hits = []
        for ln in lines[1:]:
            q = ln.split("\t")
            if len(q) < len(hdr):
                continue
            f = q[-1]
            flags[f] += 1
            if f != "ok":
                hits.append((q[0], f, q[1], q[2], q[4], q[5]))
        rep.append(f"  受检 {len(lines) - 1} 张；标记分布 {dict(flags)}")
        for n, f, w, h, b, c in hits:
            rep.append(f"    {f}: {n}  ({w}x{h}, bright={b}, colorf={c})")
    else:
        rep.append("  （未找到 TSV，先跑 screen_dirty_graphics.py --dir ... --tsv ...）")
    rep.append("")
    rep.append("## 结论待办：① 有命中则剔除；② 有近重复则按系列整组处理；④ 无问题")

    REPORT.write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep))
    print(f"\n[报告] {REPORT}")
    return 1 if (problems or missing or pairs or img_no_label) else 0


if __name__ == "__main__":
    raise SystemExit(main())
