#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""audit_lcsi_subset.py — labcoat_si（lcsi_*）子集专项四关审计（只读）

背景：主库 data/lab_ppe 中 lcsi_* 家族共 431 张（train 369 + valid 62），
      来自 Roboflow 公开集 jeiotech/labcoat_si（单类 lab_coat 裁剪图为主）。
      它占全库图数 ~18% 却只贡献 ~7% 框，且含大量空标签负样本与单类裁剪图，
      存在"域偏置 + 图框比失真"风险。本脚本对该子集做一次专项审计。

四关（与外部数据入库口径一致）：
  ① 红线去重：lcsi 逐图 pHash 对 data/raw/dataset1 + data/raw/dataset2(=external_test)
  ② 同源/重复：lcsi 家族内部近重复连通分量 + lcsi 对主库"非 lcsi"部分的近重复
  ③ 脏图筛查：商品图/广告图/纯文字图 + 灰图/黑白 + 亮度异常
  ④ 标注一致性：坏行/越界/超小/超大/极端宽高比/重复框/贴边 + 结构指标
      （图框比、空标签率、单类占比、分辨率、长宽比）

用法：
    python tools/audit_lcsi_subset.py                     # 干跑，只出报告
    python tools/audit_lcsi_subset.py --sheet outputs/lcsi_audit   # 同时出核查页
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from crosslib_phash import phash, hamming_matrix, imread_unicode  # noqa: E402
from screen_dirty_graphics import image_stats, classify, extra_stats  # noqa: E402

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

LAB = ROOT / "data" / "lab_ppe"
D1 = ROOT / "data" / "raw" / "dataset1"
D2 = ROOT / "data" / "raw" / "dataset2"
CACHE = ROOT / "outputs" / "_phash_cache_lcsi.json"
REPORT = ROOT / "data" / "lcsi_subset_audit_report.txt"
INDEX = ROOT / "data" / "lcsi_subset_reject_index.txt"

PREFIX = "lcsi_"
NAMES = {0: "mask", 1: "gloves", 2: "lab_coat", 3: "goggles"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

EDGE_TOL = 0.05      # 越界容差
MIN_AREA = 0.0005    # 超小框
MAX_AREA = 0.90      # 超大框
ASPECT = 6.0         # 极端宽高比
IOU_DUP = 0.90       # 同图同类重复框
TOL = 0.07           # 几何框匹配容差（与其他工具一致）

SPLITS = ["train", "valid"]


# ---------------------------------------------------------------- 基础工具
def list_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMG_EXT
                  and not p.name.startswith("_"))


def split_of(ip: Path, root: Path) -> str:
    try:
        rel = ip.relative_to(root)
        return rel.parts[0] if len(rel.parts) > 1 else "."
    except ValueError:
        return "."


def load_cache() -> dict:
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_cache(c: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(c), encoding="utf-8")


def hash_many(paths: list[Path], cache: dict, tag: str) -> tuple[list[Path], np.ndarray]:
    ok, hs = [], []
    for i, p in enumerate(paths):
        key = f"{p}|{p.stat().st_mtime_ns}|{p.stat().st_size}"
        h = cache.get(key)
        if h is None:
            img = imread_unicode(p)
            if img is None:
                continue
            h = str(phash(img))
            cache[key] = h
        ok.append(p)
        hs.append(np.uint64(int(h)))
        if (i + 1) % 1000 == 0:
            print(f"    [{tag}] {i+1}/{len(paths)}")
    return ok, np.array(hs, dtype=np.uint64)


def read_label(sp: str, stem: str) -> tuple[list[list[float]], list[str]]:
    lp = LAB / "labels" / sp / (stem + ".txt")
    if not lp.exists():
        return [], []
    rows, bad = [], []
    for ln in lp.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) != 5:
            bad.append(s)
            continue
        try:
            c = int(float(parts[0]))
            xy = [float(x) for x in parts[1:]]
        except ValueError:
            bad.append(s)
            continue
        if c not in NAMES:
            bad.append(s)
            continue
        rows.append([c] + xy)
    return rows, bad


def fam_of(name: str) -> str:
    """按文件名把 lcsi 拆成子上游家族（用于域分析）。"""
    n = name[len(PREFIX):] if name.startswith(PREFIX) else name
    n = re.sub(r"\.rf\.[0-9a-f]+$", "", n)   # 去 roboflow 后缀
    n = re.sub(r"\.(jpg|jpeg|png|webp)$", "", n, flags=re.I)
    if re.match(r"^-?\d{4}-\d{2}-\d{2}-", n):
        return "png-2024-05-20 系列"
    if n.startswith("KakaoTalk"):
        return "KakaoTalk 手机照"
    if n.startswith("frame_"):
        return "视频抽帧 frame_*"
    if "crop_margin" in n:
        return "crop_margin 裁剪图"
    if re.match(r"^LC\d", n, re.I):
        return "LC## 编号图"
    if re.match(r"^CoatGoggle", n, re.I):
        return "CoatGoggle 组合图"
    if re.match(r"^Lab_coat_", n, re.I) or re.match(r"^labcoat", n, re.I):
        return "Lab_coat 命名图"
    if re.match(r"^suggested-", n, re.I):
        return "suggested-* 网页建议图"
    if re.match(r"^\d+$", n):
        return "纯数字名（上游池）"
    if re.match(r"^\d+_\d+", n):
        return "数字_数字 裁剪块"
    return "其他"


def iou(a: list[float], b: list[float]) -> float:
    """cxcywh 归一化 IoU。"""
    ax1, ay1, ax2, ay2 = a[1]-a[3]/2, a[2]-a[4]/2, a[1]+a[3]/2, a[2]+a[4]/2
    bx1, by1, bx2, by2 = b[1]-b[3]/2, b[2]-b[4]/2, b[1]+b[3]/2, b[2]+b[4]/2
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    u = a[3]*a[4] + b[3]*b[4] - inter
    return inter / u if u > 0 else 0.0


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=int, default=8, help="pHash 阈值（默认 8）")
    ap.add_argument("--sheet", default="", help="核查页输出目录")
    args = ap.parse_args()
    thr = args.thr

    R: list[str] = []

    def out(s: str = "") -> None:
        R.append(s)
        print(s)

    # ---- 收集 lcsi 子集
    lcsi: list[tuple[str, Path]] = []
    for sp in SPLITS:
        for p in sorted((LAB / "images" / sp).glob(PREFIX + "*")):
            lcsi.append((sp, p))
    print(f"[lcsi] 共 {len(lcsi)} 张")

    # ---- 读标注 + 图像元信息
    meta: dict[Path, dict] = {}
    boxes_total = Counter()
    empty_imgs: list[str] = []
    bad_lines: list[tuple[str, str]] = []
    geo = defaultdict(list)
    fam_stat = defaultdict(lambda: dict(imgs=0, boxes=0, empty=0))

    for sp, p in lcsi:
        rows, bad = read_label(sp, p.stem)
        img = imread_unicode(p)
        h = w = 0
        if img is not None:
            h, w = img.shape[:2]
        fam = fam_of(p.name)
        meta[p] = dict(sp=sp, rows=rows, h=h, w=w, fam=fam,
                       area=(w * h) if w and h else 0)
        fam_stat[fam]["imgs"] += 1
        fam_stat[fam]["boxes"] += len(rows)
        if not rows:
            empty_imgs.append(f"{sp}/{p.name}")
            fam_stat[fam]["empty"] += 1
        for r in rows:
            boxes_total[r[0]] += 1
        for b in bad:
            bad_lines.append((f"{sp}/{p.name}", b))
        # 几何
        for i, r in enumerate(rows):
            c, cx, cy, bw, bh = r
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                geo["非法值"].append(f"{sp}/{p.name}#{i} {r}")
                continue
            if (cx - bw/2 < -EDGE_TOL or cy - bh/2 < -EDGE_TOL
                    or cx + bw/2 > 1 + EDGE_TOL or cy + bh/2 > 1 + EDGE_TOL):
                geo["越界"].append(f"{sp}/{p.name}#{i}")
            a = bw * bh
            if a < MIN_AREA:
                geo["超小框"].append(f"{sp}/{p.name}#{i} area={a:.5f}")
            if a > MAX_AREA:
                geo["超大框"].append(f"{sp}/{p.name}#{i} area={a:.3f}")
            if bw / bh > ASPECT or bh / bw > ASPECT:
                geo["极端宽高比"].append(f"{sp}/{p.name}#{i} {bw/bh:.1f}")
            m = min(cx - bw/2, cy - bh/2, 1 - (cx + bw/2), 1 - (cy + bh/2))
            if m < 0.05:
                geo["贴边"].append(f"{sp}/{p.name}#{i}")
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if rows[i][0] == rows[j][0] and iou(rows[i], rows[j]) > IOU_DUP:
                    geo["重复框"].append(f"{sp}/{p.name}#{i}~#{j} IoU={iou(rows[i],rows[j]):.2f}")

    n_box = sum(boxes_total.values())
    print(f"[lcsi] 框 {n_box}，空标签 {len(empty_imgs)}，坏行 {len(bad_lines)}")

    # ================================================= ① 红线去重
    cache = load_cache()
    cache0 = len(cache)

    print("[①] lcsi pHash…")
    l_ok, H_l = hash_many([p for _, p in lcsi], cache, "lcsi")
    print("[①] 红线库 pHash…")
    d2_imgs = list_images(D2)
    d2_ok, H_d2 = hash_many(d2_imgs, cache, "dataset2")
    d1_imgs = list_images(D1)
    d1_ok, H_d1 = hash_many(d1_imgs, cache, "dataset1")
    if len(cache) != cache0:
        save_cache(cache)

    red_hits: list[str] = []
    n_d2 = n_d1 = 0
    for tag, ok, H in (("dataset2(external_test)", d2_ok, H_d2),
                       ("dataset1(train源)", d1_ok, H_d1)):
        if len(H) == 0:
            continue
        D = hamming_matrix(H_l, H)
        rr, cc = np.where(D <= thr)
        print(f"[①] vs {tag}: {len(rr)} 对命中")
        for r, c in zip(rr, cc):
            sp = meta[l_ok[r]]["sp"] if l_ok[r] in meta else "?"
            red_hits.append(f"[{tag}] dist={int(D[r,c])} lcsi={sp}/{l_ok[r].name} "
                            f"<-> {ok[c].relative_to(ok[c].parents[2]) if len(ok[c].parts)>2 else ok[c].name}")
        if tag.startswith("dataset2"):
            n_d2 = len(set(rr.tolist()))
        else:
            n_d1 = len(set(rr.tolist()))

    # ================================================= ② 同源 / 内部重复
    print("[②] lcsi 内部两两…")
    Di = hamming_matrix(H_l, H_l)
    np.fill_diagonal(Di, 255)
    parent = list(range(len(l_ok)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    ii, jj = np.where(Di <= thr)
    for a, b in zip(ii.tolist(), jj.tolist()):
        union(a, b)
    comps = defaultdict(list)
    for k in range(len(l_ok)):
        comps[find(k)].append(k)
    dup_comps = {r: v for r, v in comps.items() if len(v) > 1}
    dup_imgs = sum(len(v) for v in dup_comps.values())
    print(f"[②] 内部近重复：{len(dup_comps)} 组，涉及 {dup_imgs} 张")

    # lcsi vs 主库非 lcsi
    print("[②] 主库非 lcsi pHash…")
    rest = [p for sp in SPLITS for p in sorted((LAB / "images" / sp).glob("*"))
            if not p.name.startswith(PREFIX) and p.suffix.lower() in IMG_EXT]
    r_ok, H_r = hash_many(rest, cache, "rest")
    save_cache(cache)
    cross_hits: list[str] = []
    n_cross = 0
    if len(H_r):
        Dx = hamming_matrix(H_l, H_r)
        xr, xc = np.where(Dx <= thr)
        print(f"[②] lcsi vs 主库非 lcsi：{len(xr)} 对")
        for r, c in zip(xr, xc):
            sp = meta[l_ok[r]]["sp"]
            cross_hits.append(f"dist={int(Dx[r,c])} lcsi={sp}/{l_ok[r].name} <-> {r_ok[c].name}")
        n_cross = len(set(xr.tolist()))

    # ================================================= ③ 脏图筛查
    print("[③] 脏图特征…")
    dirty = defaultdict(list)
    for sp, p in lcsi:
        img = imread_unicode(p)
        if img is None:
            dirty["unreadable"].append(f"{sp}/{p.name}")
            continue
        st = image_stats(img)
        ex = extra_stats(img)
        tag = classify(st)
        if tag is None and ex["colorf"] < 6.0:
            tag = "grayscale_like(黑白/灰图)"
        if ex["bright"] < 0.15:
            tag = (tag + "+dark" if tag else "dark(过暗)")
        if tag:
            dirty[tag].append(f"{sp}/{p.name}")
    for k, v in sorted(dirty.items()):
        print(f"[③] {k}: {len(v)} 张")

    # ================================================= ④ 汇总输出
    src_fam = Counter(meta[p]["fam"] for _, p in lcsi)
    empty_by_fam = Counter(meta[p]["fam"] for _, p in lcsi if not meta[p]["rows"])
    box_by_fam = defaultdict(int)
    for _, p in lcsi:
        box_by_fam[meta[p]["fam"]] += len(meta[p]["rows"])

    out("=" * 78)
    out("lcsi_* 子集专项审计报告（data/lab_ppe，labcoat_si 来源）")
    out("=" * 78)
    out(f"子集规模：{len(lcsi)} 张（train {sum(1 for s,_ in lcsi if s=='train')} / "
        f"valid {sum(1 for s,_ in lcsi if s=='valid')}）")
    out(f"标注框：{n_box} 个  " + ", ".join(f"{NAMES[k]}={v}" for k, v in sorted(boxes_total.items())))
    out(f"空标签（纯负样本）：{len(empty_imgs)} 张（{len(empty_imgs)/max(len(lcsi),1):.1%}）")
    out(f"坏行：{len(bad_lines)}")
    out(f"全库占比参考：lcsi 占全库图数 431/2440 = 17.7%，贡献框 {n_box}/6146 = {n_box/6146:.1%}")
    out()

    out("## 0. 子上游家族分布")
    out(f"{'家族':<24}{'图数':>6}{'框数':>6}{'空标签':>8}{'框/图':>8}")
    for fam, n in src_fam.most_common():
        e = empty_by_fam.get(fam, 0)
        b = box_by_fam.get(fam, 0)
        out(f"{fam:<24}{n:>6}{b:>6}{e:>8}{b/max(n,1):>8.2f}")
    out()

    out("## ① 红线去重（pHash <= %d）" % thr)
    out(f"vs dataset2(external_test) 命中图片：{n_d2} 张")
    out(f"vs dataset1(训练源) 命中图片：{n_d1} 张")
    if red_hits:
        out("命中明细：")
        out("\n".join(red_hits[:400]))
    else:
        out("(无命中)")
    out()

    out("## ② 同源 / 近重复")
    out(f"内部近重复连通分量：{len(dup_comps)} 组，涉及 {dup_imgs} 张")
    for r, v in sorted(dup_comps.items(), key=lambda x: -len(x[1]))[:40]:
        names = [f"{meta[l_ok[k]]['sp']}/{l_ok[k].name}" for k in v]
        out(f"  组(≥{len(v)}): " + " | ".join(names))
    out(f"lcsi vs 主库非 lcsi 命中：{n_cross} 张")
    if cross_hits:
        out("  明细：")
        out("\n".join("  " + s for s in cross_hits[:200]))
    out()

    out("## ③ 脏图 / 灰图筛查")
    for k, v in sorted(dirty.items()):
        out(f"{k}: {len(v)} 张")
        out("  " + " | ".join(v[:80]))
    out()

    out("## ④ 标注一致性 / 几何")
    for k in ["非法值", "越界", "超小框", "超大框", "极端宽高比", "重复框", "贴边"]:
        v = geo.get(k, [])
        tag = "（不计问题，仅统计）" if k == "贴边" else ""
        out(f"{k}: {len(v)} {tag}")
        if k != "贴边" and v:
            out("  " + "\n  ".join(v[:60]))
    out()
    if bad_lines:
        out("坏行明细：")
        out("\n".join(f"  {f}: {s}" for f, s in bad_lines[:80]))
        out()

    out("## 5. 结构与分辨率")
    areas = [m["area"] for m in meta.values() if m["area"]]
    if areas:
        a = np.array(areas)
        out(f"像素面积 中位数 {int(np.median(a)):,}  最小 {int(a.min()):,}  最大 {int(a.max()):,}")
        small = int((a < 200*200).sum())
        out(f"小于 200x200 的图：{small} 张（{small/len(areas):.1%}）")
    ar = [m["w"]/m["h"] for m in meta.values() if m["h"]]
    if ar:
        arr = np.array(ar)
        out(f"长宽比 中位数 {np.median(arr):.2f}  极窄(>2.0) {int((arr>2.0).sum())} 张  "
            f"极扁(<0.5) {int((arr<0.5).sum())} 张")
    # 单类占比
    multi = sum(1 for _, p in lcsi if len({r[0] for r in meta[p]["rows"]}) >= 2)
    out(f"含 >=2 个类别的图：{multi} 张（{multi/len(lcsi):.1%}）")
    out(f"有框图平均框数：{n_box/max(len(lcsi)-len(empty_imgs),1):.2f}  "
        f"（全库均值 {6146/2440:.2f}）")
    out()
    out("注：本报告为只读审计，未修改任何数据。剔除清单见 "
        f"{INDEX.relative_to(ROOT)}")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(R) + "\n", encoding="utf-8")
    print(f"\n[报告] {REPORT}")

    # ---- 剔除索引（仅红线致命命中；其余待人工裁定）
    lines = []
    seen = set()
    for s in red_hits:
        m = re.search(r"lcsi=(train|valid)/(\S+)", s)
        if m and "dataset2" in s:
            k = f"{m.group(1)}/{m.group(2)}"
            if k not in seen:
                seen.add(k)
                lines.append(f"{k}\t红线:dataset2近重复")
    INDEX.write_text("# lcsi 子集剔除索引（仅①关 dataset2 external_test 红线命中，致命）\n"
                     + ("\n".join(lines) + "\n" if lines else "# (无 dataset2 红线命中)\n"),
                     encoding="utf-8")
    print(f"[剔除索引] {INDEX}  ({len(lines)} 行)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
