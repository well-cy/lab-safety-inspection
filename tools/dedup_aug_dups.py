# -*- coding: utf-8 -*-
"""dedup_aug_dups.py — 同源增强副本去重（抓「翻转 / 调色 / 灰度化」变体）

背景：主库 .rf. 图里同一 Roboflow 源图被导出成 2~3 份——flip / hue / 灰度化等
      增强各存一份，标签也各存一套（内容还不总一致）。后果：
        ① 训练集虚高、模型过拟合这几张图；
        ② 同源副本跨 split → 评估虚高（数据泄漏）；
        ③ 人工补标（person）时同一源图重复审 2~3 遍，还容易只标一份、其余成背景毒点。

方法（分两步，避免"连续帧被串成大组"的误判）：
  1) **同源分组只认基础名**（去掉 .rf.<md5>）：这是 Roboflow 增强副本的硬证据；
  2) 视觉哈希只用来**验证**组内关系，不跨基础名串联：
       g   灰度 pHash        e   梯度幅值 pHash（颜色无关，抓调色/灰度化）
       每个再算「水平翻转后」的哈希 → 距离取 min(原-原, 原-翻转)
     同时输出 d_same / d_flip，判断"图像是否被镜像"以及"标签是否跟着镜像"。
  3) 逐副本判决：
       drop_subset   副本标签（经几何变换后）被代表完全覆盖 → 冗余，安全剔
       drop_broken   图像被镜像但标签没跟着镜像（标注错位毒数据）→ 剔
       review_*      副本带有代表没有的框 / 标签对不上 → 保留并标记人工看
       keep_far      同源但视觉差异过大（非增强副本）→ 保留

用法:
    python tools/dedup_aug_dups.py                  # 干跑，出报告 + 目检图版
    python tools/dedup_aug_dups.py --apply          # 落盘：冗余副本移入 _rejected/
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import cv2
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "data" / "lab_ppe"
SPLITS = ["train", "valid", "test"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NAMES = ["person", "mask", "gloves", "lab_coat", "goggles"]
TFS = ("identity", "fliph", "flipv", "rot180")


def imread_u(p: Path):
    try:
        return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def _phash_gray(gray: np.ndarray, hash_size: int = 8, hf: int = 4) -> int:
    size = hash_size * hf
    g = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(g))
    low = dct[:hash_size, :hash_size].flatten()
    med = np.median(low[1:])
    h = 0
    for b in (low > med).astype(np.uint64):
        h = (h << 1) | int(b)
    return h


def features_of(img: np.ndarray):
    """返回 dict(h=[g,gf,e,ef], sat, w, h)。"""
    h0, w0 = img.shape[:2]
    small = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    sat = float(hsv[:, :, 1].mean())
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.normalize(cv2.magnitude(gx, gy), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return dict(h=[_phash_gray(gray), _phash_gray(cv2.flip(gray, 1)),
                   _phash_gray(mag), _phash_gray(cv2.flip(mag, 1))],
                sat=sat, ww=w0, hh=h0)


def pixel_gate(pa: Path, pb: Path, size: int = 128) -> float:
    """像素级复核（安全闸门）：min(直接, 镜像) 的**灰度**平均绝对差。

    用灰度是有意的——Roboflow 的"灰度化/调色"增强会造出饱和度为 0 的副本，
    拿彩色像素差比会把这类真副本误判成"不像"。哈希是 64 位摘要，可能把
    "结构不同"的图判成近重复；灰度像素差只验结构，是最后一道闸门。
    """
    a = imread_u(pa)
    b = imread_u(pb)
    if a is None or b is None:
        return 999.0
    a = cv2.cvtColor(cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.int16)
    b = cv2.cvtColor(cv2.resize(b, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.int16)
    return float(min(np.abs(a - b).mean(), np.abs(a - cv2.flip(b, 1)).mean()))


TINY_AREA = 0.02      # 面积占比 <2%（416px 下约 34×34）视为"垃圾小框"


def unmatched_of(ref_rows, rows, kind, thr_iou=0.5):
    """副本 rows 中在 ref（经 kind 变换）里找不到对应的框。"""
    if not ref_rows or not rows:
        return list(rows or [])
    cand = _tf(ref_rows, kind) if kind in TFS else ref_rows
    used = set()
    out = []
    for r in rows:
        bv, bj = -1.0, -1
        for j, c in enumerate(cand):
            if j in used:
                continue
            v = _iou(c, r)
            if v > bv:
                bv, bj = v, j
        if bv >= thr_iou:
            used.add(bj)
        else:
            out.append(r)
    return out


def base_of(name: str) -> str:
    s = re.sub(r"\.(jpg|jpeg|png|bmp|webp)$", "", name, flags=re.I)
    return re.sub(r"\.rf\.[0-9a-f]+$", "", s, flags=re.I)


POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def ham(a: int, b: int) -> int:
    return int(POP[np.frombuffer(np.uint64(a ^ b).tobytes(), dtype=np.uint8)].sum())


# ---------- 标签 ----------

def read_rows(lab: Path, split: str, name: str):
    """返回 [(cls:int, cx, cy, w, h), ...]；文件不存在返回 None。"""
    f = lab / split / (Path(name).stem + ".txt")
    if not f.exists():
        return None
    out = []
    for ln in f.read_text(encoding="utf-8").splitlines():
        p5 = ln.split()
        if len(p5) == 5:
            out.append((int(p5[0]), float(p5[1]), float(p5[2]), float(p5[3]), float(p5[4])))
    return out


def _tf(rows, kind):
    out = []
    for r in rows:
        c, cx, cy, w, h = int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4])
        if kind == "fliph":
            cx = 1.0 - cx
        elif kind == "flipv":
            cy = 1.0 - cy
        elif kind == "rot180":
            cx, cy = 1.0 - cx, 1.0 - cy
        out.append((c, cx, cy, w, h))
    return out


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a[1] - a[3] / 2, a[2] - a[4] / 2, a[1] + a[3] / 2, a[2] + a[4] / 2
    bx1, by1, bx2, by2 = b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    return inter / (a[3] * a[4] + b[3] * b[4] - inter + 1e-9)


def best_kind(ref_rows, rows, thr_iou=0.50):
    """找把 ref 变换后与 rows 最匹配的变换 → (kind, 类别一致覆盖率, 位置覆盖率)。

    cov_cls 要求类别一致；cov_pos 只看位置（用于区分"类别判得不同"与"真的多标了物体"）。
    阈值 0.50 是有意放宽：同源副本的框常有小幅重画/抖动（IoU 0.6~0.85），
    用 0.9 会把同一物体误判成两个。
    """
    if ref_rows is None or rows is None or len(rows) == 0:
        return ("identity" if rows is not None else None), 0.0, 0.0
    n = len(rows)
    best = (None, -1.0, -1.0)
    for kind in TFS:
        cand = _tf(ref_rows, kind)
        used = set()
        hit_cls = 0
        for r in rows:
            for j, c in enumerate(cand):
                if j in used or int(c[0]) != int(r[0]):
                    continue
                if _iou(c, r) >= thr_iou:
                    used.add(j)
                    hit_cls += 1
                    break
        used2 = set()
        hit_pos = 0
        for r in rows:
            bv, bj = -1.0, -1
            for j, c in enumerate(cand):
                if j in used2:
                    continue
                v = _iou(c, r)
                if v > bv:
                    bv, bj = v, j
            if bv >= thr_iou:
                used2.add(bj)
                hit_pos += 1
        pair = (hit_cls / n, hit_pos / n)
        if pair > (best[1], best[2]):
            best = (kind, pair[0], pair[1])
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(LAB))
    ap.add_argument("--thr", type=int, default=6, help="视觉哈希汉明距离阈值（同源验证）")
    ap.add_argument("--far", type=int, default=12, help="超过此距离视为非同源，保留")
    ap.add_argument("--pix", type=float, default=45.0, help="像素复核阈值（平均绝对差，越小越严）")
    ap.add_argument("--cache", default="data/_aug_hash_cache2.json")
    ap.add_argument("--report", default="data/aug_dup_report.txt")
    ap.add_argument("--sheet-dir", default="outputs/aug_dup_review")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--apply-splits", default="train,valid",
                    help="允许落盘的 split（test 属评测基线，默认挂起）")
    ap.add_argument("--reject-dir", default="")
    args = ap.parse_args()

    root = Path(args.root)
    imgs = []
    for sp in SPLITS:
        d = root / "images" / sp
        if d.exists():
            imgs += [(sp, p) for p in sorted(d.iterdir())
                     if p.is_file() and p.suffix.lower() in IMG_EXT]
    print(f"扫描 {root}: {len(imgs)} 张")

    cachep = ROOT / args.cache
    cache = {}
    if cachep.exists():
        try:
            cache = json.loads(cachep.read_text(encoding="utf-8"))
        except Exception:
            cache = {}
    F = {}
    for sp, p in imgs:
        k = f"{sp}/{p.name}"
        if k in cache and isinstance(cache[k], dict):
            F[k] = cache[k]
            continue
        img = imread_u(p)
        if img is None:
            continue
        F[k] = features_of(img)
        cache[k] = F[k]
    cachep.parent.mkdir(parents=True, exist_ok=True)
    cachep.write_text(json.dumps(cache), encoding="utf-8")

    lab = root / "labels"
    meta = {}
    for sp, p in imgs:
        k = f"{sp}/{p.name}"
        rows = read_rows(lab, sp, p.name)
        f = F.get(k)
        meta[k] = dict(split=sp, name=p.name, rows=rows, base=base_of(p.name),
                       sat=(f or {}).get("sat", 0.0),
                       area=(f or {}).get("ww", 0) * (f or {}).get("hh", 0),
                       nbox=len(rows) if rows else 0,
                       ncls=len({int(r[0]) for r in rows}) if rows else 0,
                       miss=(rows is None))
    keys = sorted(meta)

    by_base = defaultdict(list)
    for k in keys:
        by_base[meta[k]["base"]].append(k)
    dup_bases = {b: v for b, v in by_base.items() if len(v) > 1}

    def same_side_counts(ks):
        """组内"同侧(未镜像)"邻居数：原始朝向那张的同侧邻居最少（变体多为它的镜像）。"""
        cnt = {k: 0 for k in ks}
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                a, b2 = ks[i], ks[j]
                ha, hb = F.get(a), F.get(b2)
                if not ha or not hb:
                    continue
                ds = min(ham(ha["h"][0], hb["h"][0]), ham(ha["h"][2], hb["h"][2]))
                df = min(ham(ha["h"][0], hb["h"][1]), ham(ha["h"][2], hb["h"][3]))
                if ds < df:
                    cnt[a] += 1
                    cnt[b2] += 1
        return cnt

    plan = []          # 拟剔
    review = []        # 待人工复核
    group_info = []

    for b, ks in sorted(dup_bases.items(), key=lambda x: -len(x[1])):
        ss = same_side_counts(ks)

        def score(k, _ss=ss):
            """代表优先：类别多 → 框多 → 彩色(非灰度化) → 原始朝向 → 像素大。"""
            m = meta[k]
            return (m["ncls"], m["nbox"], round(m["sat"]), -_ss.get(k, 0), m["area"])

        keeper = max(ks, key=score)
        members = []
        for k in ks:
            if k == keeper:
                members.append((k, "KEEP", "-", 0, 1.0, 1.0, 0))
                continue
            hk, hd = F.get(keeper), F.get(k)
            d_same = min(ham(hk["h"][0], hd["h"][0]), ham(hk["h"][2], hd["h"][2])) if hk and hd else 99
            d_flip = min(ham(hk["h"][0], hd["h"][1]), ham(hk["h"][2], hd["h"][3])) if hk and hd else 99
            d = min(d_same, d_flip)
            kind, cov_cls, cov_pos = best_kind(meta[keeper]["rows"], meta[k]["rows"])
            mirrored = d_flip < d_same and d_flip <= args.thr and d_same > args.thr
            nk = meta[k]["nbox"]
            if d > args.far:
                v = "keep_far"
            elif meta[k]["rows"] is None:
                v = "review_missing_label"
            elif nk == 0 and meta[keeper]["nbox"] > 0:
                v = "drop_empty"            # 同源空标签副本 → 背景毒点
            elif kind is None:
                v = "review_mismatch"
            elif (mirrored and kind == "identity") or                  ((not mirrored) and kind in ("fliph", "flipv", "rot180") and d <= args.thr):
                v = "drop_broken"           # 图像几何变换与标签几何变换不一致 → 标注错位
            elif cov_pos >= 0.999 and cov_cls < 0.999:
                v = "review_class_conflict"  # 同一位置类别判得不同
            elif cov_pos >= 0.999 and nk <= meta[keeper]["nbox"]:
                v = "drop_subset"           # 标签被代表完全覆盖 → 冗余
            else:
                extra = unmatched_of(meta[keeper]["rows"], meta[k]["rows"], kind)
                if extra and all(r[3] * r[4] < TINY_AREA for r in extra):
                    v = "drop_tiny_extra"   # 多出来的框全是垃圾小框 → 冗余
                else:
                    v = "review_extra"      # 副本有代表没有的真实物体
            px = pixel_gate(root / "images" / meta[keeper]["split"] / meta[keeper]["name"],
                            root / "images" / meta[k]["split"] / meta[k]["name"])
            px = px if not isinstance(px, float) else round(px, 1)
            if v.startswith("drop") and px > args.pix:
                v = "review_pixel_diff"     # 哈希说像、灰度像素说不像 → 交人工
            row = (keeper, k, v, kind or "-", d_same, d_flip, cov_pos)
            (plan if v.startswith("drop") else review).append(row)
            members.append((k, v, kind or "-", d, cov_cls, cov_pos, mirrored))
        group_info.append((b, keeper, members))

    drop_keys = {r[1] for r in plan}
    leak_groups = [gi for gi in group_info if len({meta[k]["split"] for k, *_ in gi[2]}) > 1]

    # ---------- 输出 ----------
    L = []
    def out(s=""):
        L.append(s)
        print(s)

    out(f"# 同源增强副本去重报告  ({date.today().isoformat()})")
    out(f"# root={root}  阈值 thr={args.thr}(同源验证) / far={args.far}(判非同源)")
    out(f"# 扫描 {len(keys)} 张；唯一基础名 {len(by_base)}；同源多副本基础名 {len(dup_bases)}")
    out(f"# 同源组 {len(group_info)}；拟剔 {len(plan)}；待人工复核 {len(review)}；跨 split 泄漏组 {len(leak_groups)}")
    out()
    vc = Counter(r[2] for r in plan) + Counter(r[2] for r in review)
    out(f"# 判决分布: {dict(vc)}")
    out("#   drop_subset=标签被代表完全覆盖(冗余)  drop_broken=图像镜像但标签未镜像(错位毒数据)")
    out("#   review_extra=副本带有代表没有的框  review_mismatch=标签对不上  keep_far=同源但视觉差异过大")
    out()
    if leak_groups:
        out(f"# 同源组跨 split 泄漏 {len(leak_groups)} 组：")
        for b, keeper, members in leak_groups[:40]:
            out(f"   LEAK {b[:40]}: " + " | ".join(
                f'{meta[k]["split"]}/{meta[k]["name"][:44]}[{v}]' for k, v, *_ in members))
        out()
    out("=" * 110)
    for i, (b, keeper, members) in enumerate(group_info, 1):
        out(f"### 组{i:03d}  源={b[:60]}  成员 {len(members)}")
        for k, v, kind, d, cc, cp, mir in sorted(members, key=lambda x: (x[1] != "KEEP", x[0])):
            m = meta[k]
            out(f"   {v:<22} {m['split']:<5} {m['name'][:52]:<54} box={m['nbox']} cls={m['ncls']} "
                f"sat={m['sat']:.0f} d={d:<3} kind={kind:<8} cov_cls={cc:.2f} cov_pos={cp:.2f} mir={int(bool(mir))}")
    out("=" * 110)
    out("# 剔除清单（split\\t文件\\t保留代表\\t判决\\tkind\\td_same\\td_flip\\tcov）")
    for keeper, d, v, kind, ds, df, cov in sorted(plan, key=lambda x: (meta[x[1]]["split"], meta[x[1]]["name"])):
        out(f"{meta[d]['split']}\t{meta[d]['name']}\t{meta[keeper]['name']}\t{v}\t{kind}\t{ds}\t{df}\t{cov:.2f}")
    if review:
        out()
        out("# 待人工复核清单（未剔，仅提示）")
        for keeper, k, v, kind, ds, df, cov in sorted(review, key=lambda x: (x[2], meta[x[1]]["split"], meta[x[1]]["name"])):
            out(f"{v}\t{meta[k]['split']}/{meta[k]['name']}\t代表={meta[keeper]['name']}\tkind={kind}\tcov={cov:.2f}\td_same={ds}\td_flip={df}")

    rep = ROOT / args.report
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n报告 → {rep}")

    idxp = ROOT / "data/_aug_dup_drop_index.txt"
    idxp.write_text("\n".join(f"{meta[k]['split']}/{meta[k]['name']}" for k in sorted(drop_keys)) + "\n",
                    encoding="utf-8")
    print(f"剔除索引 → {idxp}  ({len(drop_keys)})")

    # ---------- 目检图版（只画含 drop/review 的组） ----------
    sdir = ROOT / args.sheet_dir
    sdir.mkdir(parents=True, exist_ok=True)
    CELL, COLS = 300, 4
    interesting = [gi for gi in group_info if any(v != "KEEP" for _, v, *_ in gi[2])]
    sheets = 0
    for si in range(0, len(interesting), COLS):
        chunk = interesting[si:si + COLS]
        rows_n = max(len(gi[2]) for gi in chunk)
        sh = np.full((rows_n * (CELL + 26), COLS * (CELL + 10), 3), 255, np.uint8)
        for ci, (b, keeper, members) in enumerate(chunk):
            for ri, (k, v, kind, d, cc, cp, mir) in enumerate(sorted(members, key=lambda x: (x[1] != "KEEP", x[0]))):
                m = meta[k]
                im = imread_u(root / "images" / m["split"] / m["name"])
                if im is None:
                    continue
                h, w = im.shape[:2]
                s = min(CELL / w, CELL / h)
                r = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))))
                y0 = ri * (CELL + 26) + (CELL - r.shape[0]) // 2
                x0 = ci * (CELL + 10) + (CELL - r.shape[1]) // 2
                sh[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
                col = (0, 150, 0) if v == "KEEP" else ((0, 0, 220) if v.startswith("drop") else (0, 150, 220))
                cv2.rectangle(sh, (ci * (CELL + 10), ri * (CELL + 26)),
                              (ci * (CELL + 10) + CELL, ri * (CELL + 26) + CELL), col, 2)
                cap = f"{v[:12]} {m['name'][:22]}"
                cv2.putText(sh, cap, (ci * (CELL + 10) + 2, ri * (CELL + 26) + CELL + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3)
                cv2.putText(sh, cap, (ci * (CELL + 10) + 2, ri * (CELL + 26) + CELL + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
        ok, buf = cv2.imencode(".jpg", sh, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if ok:
            (sdir / f"sheet_{si // COLS + 1:02d}.jpg").write_bytes(buf.tobytes())
            sheets += 1
    print(f"目检图版 {sheets} 版 → {sdir}")

    if not args.apply:
        print("\n[干跑] 未改动任何文件。加 --apply 落盘。")
        return 0

    tag = args.reject_dir or f"data/_rejected/aug_dup_{date.today().strftime('%Y%m%d')}"
    rdir = ROOT / tag
    allow = {s.strip() for s in args.apply_splits.split(",") if s.strip()}
    held = [r for r in plan if meta[r[1]]["split"] not in allow]
    if held:
        Path(ROOT / "data/_aug_dup_held.txt").write_text(
            "\n".join(f"{meta[k]['split']}/{meta[k]['name']}\t代表={meta[kp]['name']}\t{v}"
                      for kp, k, v, *_ in sorted(held, key=lambda x: (meta[x[1]]["split"], meta[x[1]]["name"])))
            + "\n", encoding="utf-8")
        print(f"[挂起] {len(held)} 张不在 {sorted(allow)}：清单 data/_aug_dup_held.txt")
    moved, failed = 0, []
    man = ["# 同源增强副本剔除清单  被剔 -> 保留代表", ""]
    for keeper, k, v, kind, ds, df, cov in sorted(plan, key=lambda x: (meta[x[1]]["split"], meta[x[1]]["name"])):
        m = meta[k]
        if m["split"] not in allow:
            continue
        src = root / "images" / m["split"] / m["name"]
        dst = rdir / m["split"] / m["name"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            continue
        try:
            os.replace(src, dst)
        except Exception:
            try:
                alt = dst.with_name("x_" + dst.name)
                os.replace(src, alt)
                dst = alt
            except Exception as e:
                failed.append(f"{m['split']}/{m['name']}: {e}")
                continue
        ls = root / "labels" / m["split"] / (Path(m["name"]).stem + ".txt")
        w1 = "no-label"
        if ls.exists():
            try:
                os.replace(ls, rdir / m["split"] / ls.name)
                w1 = "label-moved"
            except Exception as e:
                w1 = f"label-fail:{e}"
        moved += 1
        man.append(f"{m['split']}/{m['name']}\t->\t{meta[keeper]['split']}/{meta[keeper]['name']}\t{v}\t{kind}\t"
                   f"d_same={ds}\td_flip={df}\tcov={cov:.2f}\t{w1}")
    (rdir / "_prune_manifest.csv").parent.mkdir(parents=True, exist_ok=True)
    mpath = rdir / "_prune_manifest.csv"
    head = man[:2]                 # 注释两行
    body = man[2:]
    if mpath.exists():             # 多次运行追加，不覆盖历史
        old = mpath.read_text(encoding="utf-8").rstrip("\n").split("\n")
        body = [l for l in old[2:] if l.strip()] + body
    mpath.write_text("\n".join(head + body) + "\n", encoding="utf-8")
    print(f"\n[落盘] 移动 {moved} 张（含标签）→ {rdir}；失败 {len(failed)}；清单累计 {len(body)} 条")
    for f in failed[:10]:
        print("   FAIL", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
