#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""lcsi_prune_apply.py — lcsi 子集域瘦身执行器（默认干跑，--apply 落盘）

方案（10-09 她授权"做你觉得最好的方案"）：
  保留真实拍摄场景（实验室/监控/手机实拍），剔除电商目录域与冗余：
    R1  d=0 完全冗余副本（每分量留 1，优先 train/框多/含非 lab_coat 类）
    R2  lcsi_27（与 websr3_14 同源，r3 方案 B 挂账）
    R3  空标签负样本中"商品图/纯色棚拍"域（graphic_* 命中；grayscale 与 frame_* 同域负样本保留）
    R4  电商目录正样本族整族：LC##、Lab_coat_*、suggested-*
    R5  其他(纯数字名)有框且 graphic_* 命中（白底目录模特图）
    R6  png-2024-05-20 系列中 graphic_* 命中（目录图；grayscale 保留）
    R7  20250228_* 同一摆拍连拍裁剪块：只留 4 张代表
    R8  crop_margin 同源裁剪块：只留 3 张代表
    R9  frame_* 有框连续帧：每 3 帧留 1
    R10 漏标毒点 3 张（KakaoTalk_…_10 手套未标 / CoatGoggle8 护目镜未标 / -2024-05-20-1418 口罩未标）
  保护（一律不剔）：
    P1 含 mask/gloves/goggles 框的图（lcsi 全族仅 ~10 张，除 lcsi_27 外）
    P2 多类图
  收尾：
    F1  保留集内 d=0 保底去重
    F2  修复 -2024-05-20-143341 重复框（IoU 0.94，删 #1，先备份标签）
    F3  剩余 lcsi 重检跨 split 泄漏，valid 侧移回 train（口径：连通分量整组同 split）
    F4  全库校验（数量/类别对账/坏行/孤儿/跨 split 重名）

剔除 = 移动到 data/_rejected/lcsi_prune_20261009/<split>/，附 _prune_manifest.csv（可整体移回）。
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from crosslib_phash import phash, hamming_matrix, imread_unicode  # noqa: E402
from audit_lcsi_subset import fam_of, read_label  # noqa: E402
from screen_dirty_graphics import image_stats, classify, extra_stats  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

LAB = ROOT / "data" / "lab_ppe"
PREFIX = "lcsi_"
REJ = ROOT / "data" / "_rejected" / "lcsi_prune_20261009"
MANIFEST = REJ / "_prune_manifest.csv"

# 漏标毒点（人工目检确认：dirty #6/#75/#76，可回收）
POISON = {
    "lcsi_-2024-05-20-141815_png.rf.55449c78d88b45257ff39d03790aa8b2.jpg",  # 口罩未标（dirty#6/realscene#7）
    "lcsi_CoatGoggle8_jpg.rf.88cd862da983a141d70e638afa96754f.jpg",         # 护目镜未标
    "lcsi_KakaoTalk_20240321_043006418_10_jpg.rf.bb710996b242fe5db8c86b6d45eb899d.jpg",  # 蓝手套未标
}
DUPBOX_FIX = "lcsi_-2024-05-20-143341_png.rf.9571bf22f8aa0b2c419c10edf323a29b"  # 删 #1（IoU 0.94）


def collect() -> list[tuple[str, Path]]:
    out = []
    for sp in ["train", "valid"]:
        out += [(sp, p) for p in sorted((LAB / "images" / sp).glob(PREFIX + "*"))]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真正落盘（默认干跑）")
    args = ap.parse_args()

    imgs = collect()
    info = {}
    for sp, p in imgs:
        rows, _ = read_label(sp, p.stem)
        img = imread_unicode(p)
        st = image_stats(img) if img is not None else {}
        ex = extra_stats(img) if img is not None else {}
        flag = classify(st) if st else None
        if flag is None and ex and ex["colorf"] < 6.0:
            flag = "grayscale_like"
        fam = fam_of(p.name)
        info[p] = dict(sp=sp, rows=rows, fam=fam, flag=flag or "",
                       n=len(rows), cls={r[0] for r in rows})

    # P1 保护：含非 lab_coat 框
    protect = {p for p, d in info.items() if d["cls"] - {2}}
    # R2 挂账
    rej27 = {p for p in info if p.name.startswith("lcsi_27_jpg.rf.f79da049")}

    # ---- R1 d=0 冗余
    H, ok = [], []
    for sp, p in imgs:
        im = imread_unicode(p)
        if im is not None:
            ok.append(p)
            H.append(phash(im))
    H = np.array([np.uint64(h) for h in H], dtype=np.uint64)
    D = hamming_matrix(H, H)
    np.fill_diagonal(D, 255)
    parent = list(range(len(ok)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(*np.where(D == 0)):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra
    comps = defaultdict(list)
    for k in range(len(ok)):
        comps[find(k)].append(k)

    def score(p: Path):
        d = info[p]
        return (0 if d["sp"] == "valid" else 1, len(d["cls"] - {2}), d["n"])

    dup_drop = set()
    for v in comps.values():
        if len(v) <= 1:
            continue
        keep_k = sorted(v, key=lambda k: score(ok[k]), reverse=True)[0]
        for k in v:
            if k != keep_k:
                dup_drop.add(ok[k])

    # ---- 族规则
    drop: dict[Path, str] = {}
    for p, d in info.items():
        if p in rej27:                       # 挂账项优先于一切保护
            drop[p] = "R2_dup_with_websr3_14"
            continue
        if p in protect:
            continue
        fam, flag, n = d["fam"], d["flag"], d["n"]
        if p in rej27:
            drop[p] = "R2_dup_with_websr3_14"
        elif p in dup_drop:
            drop[p] = "R1_exact_dup_d0"
        elif n == 0 and fam != "视频抽帧 frame_*" and flag != "grayscale_like":
            # 空标签负样本：仅保留同域的监控帧与灰图；其余为电商服装目录图
            #（chkC 图版证实：有纹理背景的目录图 flag 抓不到，需按"空标签"直接剔）
            drop[p] = "R3_catalog_negative"
        elif fam in ("LC## 编号图", "Lab_coat 命名图", "suggested-* 网页建议图"):
            drop[p] = "R4_catalog_family"
        elif fam == "其他" and n > 0 and flag in ("graphic_white", "graphic_color", "graphic_color+dark"):
            drop[p] = "R5_catalog_positive"
        elif fam == "png-2024-05-20 系列" and flag in ("graphic_white", "graphic_color", "graphic_color+dark"):
            drop[p] = "R6_catalog_positive"
        elif p.name in POISON:
            drop[p] = "R10_missed_label_poison"
    # R7/R8/R9 组内配额（在未被上面剔除的图里做抽稀）
    for fam, keep_n, tag in [("数字_数字 裁剪块", 4, "R7_same_shoot_redundant"),
                             ("crop_margin 裁剪图", 3, "R8_same_source_redundant")]:
        members = sorted([p for p, d in info.items() if d["fam"] == fam and p not in drop and p not in protect],
                         key=lambda p: p.name)
        if len(members) > keep_n:
            step = max(1, len(members) // keep_n)
            for p in members:
                if members.index(p) % step != 0:
                    drop[p] = tag
    members = sorted([p for p, d in info.items() if d["fam"] == "视频抽帧 frame_*" and d["n"] > 0
                      and p not in drop and p not in protect], key=lambda p: p.name)
    for i, p in enumerate(members):
        if i % 3 != 0:
            drop.setdefault(p, "R9_video_frame_thin")

    # F1 保底：保留集内 d=0 去重
    keep = [p for p in info if p not in drop]
    seen_h = {}
    for p in sorted(keep):
        im = imread_unicode(p)
        if im is None:
            continue
        h = phash(im)
        if h in seen_h:
            drop[p] = "F1_keep_set_dup_d0"
        else:
            seen_h[h] = p

    # ---- 汇总
    by_reason = Counter(drop.values())
    n_box_drop = sum(info[p]["n"] for p in drop)
    cls_drop = Counter()
    for p in drop:
        for r in info[p]["rows"]:
            cls_drop[r[0]] += 1
    cls_keep = Counter()
    for p in info:
        if p not in drop:
            for r in info[p]["rows"]:
                cls_keep[r[0]] += 1
    print(f"lcsi 共 {len(imgs)} 张 / {sum(d['n'] for d in info.values())} 框")
    print(f"剔除 {len(drop)} 张 / {n_box_drop} 框：" +
          ", ".join(f"{k}={v}" for k, v in sorted(by_reason.items())))
    print(f"保留 {len(imgs) - len(drop)} 张 / {sum(cls_keep.values())} 框  "
          f"类别：mask={cls_keep[0]} gloves={cls_keep[1]} lab_coat={cls_keep[2]} goggles={cls_keep[3]}")
    print(f"剔除类别：mask={cls_drop[0]} gloves={cls_drop[1]} lab_coat={cls_drop[2]} goggles={cls_drop[3]}")
    kept_tab = Counter((info[p]["fam"], info[p]["flag"] or "clean",
                        "empty" if info[p]["n"] == 0 else "box") for p in info if p not in drop)
    print("保留构成：")
    for k in sorted(kept_tab):
        print(f"    {k[0]:<22}{k[1]:<16}{k[2]:<6}{kept_tab[k]}")

    if not args.apply:
        print("\n[干跑] 未落盘。加 --apply 执行。")
        return 0

    # ---- 落盘：移动到 _rejected
    REJ.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    moved = 0
    with MANIFEST.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "filename", "family", "flag", "n_boxes", "reason", "moved_at"])
        for sp, p in imgs:
            if p not in drop:
                continue
            d = REJ / sp
            d.mkdir(exist_ok=True)
            dst_img = d / p.name
            dst_lbl = d / (p.stem + ".txt")
            os.replace(p, dst_img)
            lp = LAB / "labels" / sp / (p.stem + ".txt")
            if lp.exists():
                os.replace(lp, dst_lbl)
            w.writerow([sp, p.name, d := info[p]["fam"], info[p]["flag"], info[p]["n"],
                        drop[p], datetime.now().isoformat(timespec="seconds")])
            moved += 1
    print(f"[移动] {moved} 张 -> {REJ}")

    # ---- F2 重复框修复
    for sp in ["train", "valid"]:
        lp = LAB / "labels" / sp / (DUPBOX_FIX + ".txt")
        if lp.exists():
            bak = ROOT / "data" / "_label_backup" / "lcsi_prune_20261009" / (DUPBOX_FIX + ".txt")
            bak.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(lp, bak)
            rows = [l for l in lp.read_text(encoding="utf-8").splitlines() if l.strip()]
            if len(rows) >= 2:
                lp.write_text("\n".join(rows[:1]) + "\n", encoding="utf-8")
                print(f"[F2] {DUPBOX_FIX} 重复框已删 #1（{len(rows)}->1），备份 {bak.name}")
            break

    # ---- F3 剩余 lcsi 跨 split 泄漏复检 + valid 侧移 train
    rest = collect()
    if rest:
        H2, ok2 = [], []
        for sp, p in rest:
            im = imread_unicode(p)
            ok2.append(p)
            H2.append(phash(im) if im is not None else 0)
        H2 = np.array([np.uint64(h) for h in H2], dtype=np.uint64)
        D2 = hamming_matrix(H2, H2)
        np.fill_diagonal(D2, 255)
        par = list(range(len(ok2)))

        def find2(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        for a, b in zip(*np.where(D2 <= 8)):
            ra, rb = find2(int(a)), find2(int(b))
            if ra != rb:
                par[rb] = ra
        groups = defaultdict(list)
        for k in range(len(ok2)):
            groups[find2(k)].append(k)
        moved_v = 0
        for v in groups.values():
            sps = {info[ok2[k]]["sp"] for k in v if ok2[k] in info} or \
                  {("train" if (LAB / "labels" / "train" / (ok2[k].stem + ".txt")).exists() else "valid") for k in v}
            if len(sps) > 1:
                for k in v:
                    p = ok2[k]
                    cur = "train" if (LAB / "labels" / "train" / (p.stem + ".txt")).exists() else "valid"
                    if cur == "valid":
                        os.replace(LAB / "images" / "valid" / p.name, LAB / "images" / "train" / p.name)
                        os.replace(LAB / "labels" / "valid" / (p.stem + ".txt"),
                                   LAB / "labels" / "train" / (p.stem + ".txt"))
                        moved_v += 1
        print(f"[F3] 剩余 lcsi 跨 split 组 valid 侧移回 train：{moved_v} 张")

    # ---- F4 全库校验
    tot_i = tot_b = 0
    bad = 0
    names = set()
    cls_all = Counter()
    for sp in ["train", "valid", "test"]:
        ims = list((LAB / "images" / sp).glob("*"))
        lbs = {p.stem for p in (LAB / "labels" / sp).glob("*.txt")}
        orph = sum(1 for p in ims if p.stem not in lbs)
        miss = sum(1 for s in lbs if not (LAB / "images" / sp / (s + ".jpg")).exists()
                   and not any((LAB / "images" / sp / (s + e)).exists() for e in (".png", ".jpeg")))
        for lp in (LAB / "labels" / sp).glob("*.txt"):
            for ln in lp.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    parts = ln.split()
                    if len(parts) != 5:
                        bad += 1
                    else:
                        cls_all[int(float(parts[0]))] += 1
                        tot_b += 1
        for p in ims:
            if p.name in names:
                print(f"[F4][警告] 跨 split 重名：{p.name}")
            names.add(p.name)
        tot_i += len(ims)
        print(f"[F4] {sp}: imgs={len(ims)} labels={len(lbs)} 孤儿图={orph} 缺图={miss}")
    print(f"[F4] 全库 {tot_i} 图 / {tot_b} 框  坏行={bad}  "
          f"mask={cls_all[0]} gloves={cls_all[1]} lab_coat={cls_all[2]} goggles={cls_all[3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
