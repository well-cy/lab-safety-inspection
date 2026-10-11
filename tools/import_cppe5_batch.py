# -*- coding: utf-8 -*-
"""CPPE-5 终审清单入库（2026-10-10 她拍板：468 全采用 → 自动清洗 412 → 剔 15 空挂 + 3 毒标注 = 394）。

流程：
  1. picked_index.csv + final_clean.json（412：已剔 39 泄漏 + 33 批内同图冗余）
  2. 再剔 EXCLUDE_NO（15 空挂 + 3 毒标注 + EXCLUDE_MANUAL 点名剔除，按 picked 编号）
  3. 标签几何清洗：越界裁剪、大半出图丢弃、w/h<0.01 丢弃、同类 IoU>=0.95 去重
  4. 划分：第②关 dist<=8 近重复连通分量整组同 split，按编号序轮转凑 ~10% valid，不进 test
  5. PNG→JPG(q95)，改名 cppe5_<no>_<md5[:8]>.jpg，写 import_manifest_cppe5.csv
  6. 入库前 tar 备份主库 train/valid 标签；入库后全库格式校验 + 按类别对账
"""
import csv
import hashlib
import io
import json
import re
import shutil
import sys
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
PICKED = ROOT / "data/raw/cppe5/picked"
LABELS = ROOT / "data/raw/cppe5/prep/labels"
INDEX = ROOT / "outputs/cppe5_screen/picked_index.csv"
CLEAN = ROOT / "outputs/cppe5_screen/final_clean.json"
AUDIT = ROOT / "data/webscene_batch_audit_cppe5.txt"
MANIFEST = ROOT / "data/raw/cppe5/import_manifest_cppe5.csv"
LIB = ROOT / "data/lab_ppe"
NAMES = ["person", "mask", "gloves", "lab_coat", "goggles"]

# 她拍板「按你说的剔」：15 空挂（挂椅背/站立的未穿戴防护服被标 person+mask+gloves）
EXCLUDE_HANGING = {611, 612, 613, 652, 707, 712, 714, 733, 737, 751, 838, 839, 851, 853, 858}
# AI 目检 + 画框核实：毒标注
#   70.png  mask/gloves 框锚在 (1,1) 巨型越界废框；621/909.png goggles/mask 框在桌面未佩戴装备
EXCLUDE_POISON = {60: "mask/gloves 框锚(1,1)越界废框", 503: "goggles/mask 框在桌面未佩戴", 770: "goggles/mask 框在桌面未佩戴"}
# 她 2026-10-11 点名整图剔除（入库后追加，此处登记以保证脚本重跑不会复活）
EXCLUDE_MANUAL = {153: "火葬场混乱人群（框全乱、场景不符实验室/医疗操作红线）"}


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[1] - a[3] / 2, a[2] - a[4] / 2, a[1] + a[3] / 2, a[2] + a[4] / 2
    bx1, by1, bx2, by2 = b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = a[3] * a[4] + b[3] * b[4] - inter
    return inter / ua if ua > 0 else 0.0


def clean_label(path: Path):
    """返回 (rows, dropped) ；rows=[(c,cx,cy,w,h)] 已清洗。"""
    rows, dropped = [], []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        c, cx, cy, w, h = ln.split()
        c, cx, cy, w, h = int(c), float(cx), float(cy), float(w), float(h)
        x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
        x2, y2 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
        nw, nh = x2 - x1, y2 - y1
        if nw * nh < 0.4 * w * h:
            dropped.append((c, "大半出图"))
            continue
        if nw < 0.01 or nh < 0.01:
            dropped.append((c, "过小"))
            continue
        row = (c, round((x1 + x2) / 2, 6), round((y1 + y2) / 2, 6), round(nw, 6), round(nh, 6))
        if any(r[0] == row[0] and iou(r, row) >= 0.95 for r in rows):
            dropped.append((c, "同类重复"))
            continue
        rows.append(row)
    return rows, dropped


def main():
    idx = {int(r["no"]): r for r in csv.DictReader(INDEX.open(encoding="utf-8"))}
    clean = json.load(CLEAN.open(encoding="utf-8"))
    final_files = list(clean["final"])          # 412
    drop_leak = set(clean["drop_leak"])
    drop_dup = set(clean["drop_dup"])

    no2file = {no: r["file"] for no, r in idx.items() if r["file"] in set(final_files)}
    ex_hang = {no2file[n] for n in EXCLUDE_HANGING if n in no2file}
    ex_poison = {no2file[n]: EXCLUDE_POISON[n] for n in EXCLUDE_POISON if n in no2file}
    ex_manual = {no2file[n] for n in EXCLUDE_MANUAL if n in no2file}
    keep = [f for f in final_files
            if f not in ex_hang and f not in ex_poison and f not in ex_manual]
    print(f"picked 468 → 泄漏 {len(drop_leak)} + 批内冗余 {len(drop_dup)} → 412；"
          f"空挂 {len(ex_hang)} + 毒标注 {len(ex_poison)} + 点名剔 {len(ex_manual)} "
          f"→ **入库 {len(keep)}**")

    # ---- 标签清洗 ----
    boxdata, drops = {}, []
    for f in keep:
        rows, dd = clean_label(LABELS / (f.rsplit(".", 1)[0] + ".txt"))
        boxdata[f] = rows
        if dd:
            drops.append((f, dd))
    n_drop = sum(len(d) for _, d in drops)
    print(f"几何清洗：{len(keep)} 图，丢框 {n_drop} 个（{sum(1 for _, d in drops for _ in d)}）")
    for f, dd in drops[:12]:
        print(f"  {f}: {Counter(c for c, _ in dd)}")

    # ---- 近重复连通分量（第②关 dist<=8）----
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    sec = ""
    for ln in AUDIT.read_text(encoding="utf-8").splitlines():
        if ln.startswith("## "):
            sec = ln
            continue
        m = re.match(r"dist=(\d+)\s+(\S+)\s+<->\s+(\S+)", ln.strip())
        if m and "第②关" in sec and int(m.group(1)) <= 8:
            a, b = m.group(2), m.group(3)
            if a in boxdata and b in boxdata:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
    comps = defaultdict(list)
    for f in boxdata:
        comps[find(f)].append(f)
    comps = sorted([sorted(v, key=lambda x: int(x.split(".")[0])) for v in comps.values()],
                   key=lambda g: int(g[0].split(".")[0]))
    print(f"连通分量 {len(comps)} 组（最大 {max(len(g) for g in comps)} 张）")

    # ---- 划分：按编号序轮转凑 valid ≈10%，整组同 split ----
    target = max(1, round(len(keep) * 0.10))
    valid, vc = [], 0
    for g in comps:                      # 每 10 组取 1 组，直到够量
        if vc >= target:
            break
        valid.append(g)
        vc += len(g)
    vsplit = {f for g in valid for f in g}
    print(f"valid {len(vsplit)} 图 / train {len(keep) - len(vsplit)} 图（不进 test）")

    # ---- 备份 ----
    ts = time.strftime("%Y%m%d_%H%M%S")
    bk = LIB / "_label_backup" / f"labels_tv_cppe5_{ts}.tar.gz"
    with tarfile.open(bk, "w:gz") as tf:
        for sp in ("train", "valid"):
            for p in sorted((LIB / "labels" / sp).glob("*.txt")):
                tf.add(p, arcname=f"{sp}/{p.name}")
    print(f"备份 → {bk.relative_to(ROOT)}")

    # ---- 落盘 ----
    rows_out = []
    per_cls = Counter()
    for f in sorted(keep, key=lambda x: int(x.split(".")[0])):
        no = next(n for n, r in idx.items() if r["file"] == f)
        src = PICKED / f
        md5 = hashlib.md5(src.read_bytes()).hexdigest()[:8]
        name = f"cppe5_{no}_{md5}"
        sp = "valid" if f in vsplit else "train"
        img = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_COLOR)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        assert ok
        buf.tofile(str(LIB / "images" / sp / f"{name}.jpg"))
        txt = "\n".join(" ".join(str(v) for v in r) for r in boxdata[f]) + ("\n" if boxdata[f] else "")
        (LIB / "labels" / sp / f"{name}.txt").write_text(txt, encoding="utf-8", newline="\n")
        per_cls.update(r[0] for r in boxdata[f])
        rows_out.append({"no": no, "src": f"picked/{f}", "new_name": name + ".jpg", "split": sp,
                         "md5": md5, "boxes": len(boxdata[f]),
                         "classes": "|".join(sorted(NAMES[c] for c in {r[0] for r in boxdata[f]}))})
    with MANIFEST.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"入库 {len(rows_out)} 图 / {sum(per_cls.values())} 框 {dict(sorted(per_cls.items()))}")
    print(f"清单 → {MANIFEST.relative_to(ROOT)}")

    # ---- 全库校验 ----
    tot = Counter()
    bad = 0
    for sp in ("train", "valid", "test"):
        c = Counter()
        n = 0
        for p in (LIB / "labels" / sp).glob("*.txt"):
            n += 1
            imgp = LIB / "images" / sp / (p.stem + ".jpg")
            if not imgp.exists():
                print(f"!! 孤儿标签 {p}")
            for ln in p.read_text(encoding="utf-8").splitlines():
                parts = ln.split()
                if len(parts) != 5 or not (0 <= int(parts[0]) <= 4):
                    bad += 1
                    continue
                vals = list(map(float, parts[1:]))
                if not all(0.0 <= v <= 1.0 for v in vals):
                    bad += 1
                    continue
                c[int(parts[0])] += 1
        imgs = len(list((LIB / "images" / sp).glob("*.jpg")))
        if imgs != n:
            print(f"!! {sp} 图/标签数不符 {imgs}/{n}")
        print(f"{sp}: {n} 图 / {sum(c.values())} 框 {dict(sorted(c.items()))}")
        tot.update(c)
    print(f"全库 {sum(tot.values())} 框 {dict(sorted(tot.items()))} 坏行 {bad}")


if __name__ == "__main__":
    main()
