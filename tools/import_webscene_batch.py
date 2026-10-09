# -*- coding: utf-8 -*-
"""import_webscene_batch.py —— web_lab_scene 网图入库 data/lab_ppe

--batch r1（默认）：images 48 张小红书网图；按博主系列整组划分
--batch r2：images_r2 19 张 stock 视频截图；批内审计无近重复 → 每张自成系列按比例划分

类别口径：
  - 主库当前是四类（person 未纳入），因此入库时 **person 框丢弃**、其余四类整体左移一位：
      1(mask)->0  2(gloves)->1  3(lab_coat)->2  4(goggles)->3
  - 5 类完整草稿另存 data/raw/web_lab_scene/labels_5cls/，等主库做 5 类化（person 补标）
    时用同 stem 一键回补 person 框，不丢信息。

划分：**系列整组进同一 split**，防跨 split 泄漏；小系列优先给 valid（目标 ~15%），
      大系列留 train；**不进 test**。

命名：非 ASCII 原名对训练管线不友好，统一改 ASCII: webs{b}_<编号>_<md5[:8]>.jpg，
      原名/系列/split 写入 import_manifest{_b}.csv 备查。

用法:
    python tools/import_webscene_batch.py [--batch r2]            # 干跑
    python tools/import_webscene_batch.py [--batch r2] --apply    # 落盘
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import re
import shutil
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent

_ap = argparse.ArgumentParser()
_ap.add_argument("--batch", default="", help="批次后缀，如 r2")
_ap.add_argument("--apply", action="store_true")
_ap.add_argument("--src", default="", help="覆盖图片目录")
_ap.add_argument("--lbl", default="", help="覆盖五类草稿标签目录")
_ap.add_argument("--index", default="", help="覆盖索引文件")
_ap.add_argument("--arch", default="", help="五类草稿存档目录")
_ap.add_argument("--manifest", default="", help="溯源表路径")
_ap.add_argument("--prefix", default="", help="入库文件名前缀（默认 webs<batch>）")
_args = _ap.parse_args()
_b = _args.batch
_sfx = f"_{_b}" if _b else ""
PREFIX = _args.prefix or (f"webs{_b}" if _b else "webs")


def _p(v, default):
    if not v:
        return default
    q = Path(v)
    return q if q.is_absolute() else ROOT / q


SRC_IMG = _p(_args.src, ROOT / "data" / "raw" / "web_lab_scene" / f"images{_sfx}")
SRC_LBL5 = _p(_args.lbl, ROOT / "data" / "_prelabel" / f"webscene{_sfx}" / "labels")
ARCH_LBL5 = _p(_args.arch, ROOT / "data" / "raw" / "web_lab_scene" / "labels_5cls")
MANIFEST = _p(_args.manifest, ROOT / "data" / "raw" / "web_lab_scene" / f"import_manifest{_sfx}.csv")
DS = ROOT / "data" / "lab_ppe"
INDEX = _p(_args.index, ROOT / "outputs" / f"web_lab_scene_review{_sfx}" / "index.txt")

CLASSES = ["mask", "gloves", "lab_coat", "goggles"]
CLS5 = {0: "person", 1: "mask", 2: "gloves", 3: "lab_coat", 4: "goggles"}
REM = {1: 0, 2: 1, 3: 2, 4: 3}     # E6 五类 -> 主库四类
VALID_RATIO = 0.15


def load_index():
    out = []
    for line in INDEX.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("编号"):
            continue
        p = line.split("\t")
        if len(p) >= 2:
            out.append((int(p[0]), p[1]))
    return out


def series_of(stem: str) -> str:
    """从文件名推系列（同一篇笔记/同一组图）。stock/uploads 批每张自成系列。"""
    if stem.startswith("stock_"):
        return stem            # r2：19 张批内互不近重复（审计②），按张独立划分
    if stem.startswith("img_"):
        return stem            # uploads 第 0 批：批内 0 近重复，按张独立划分
    base = stem.split("_来自小红书")[0]
    m = re.match(r"^_(\d+)_(.+)$", base)      # 形如 _1_青春药发光
    if m:
        return m.group(2)
    return re.sub(r"_\d+_[^_]*$", "", base)   # 形如 STEM Girlie_x_HannaH


def main() -> int:
    apply = _args.apply

    idx = load_index()
    have = set()
    for sp in ("train", "valid", "test"):
        have |= {p.stem for p in (DS / "images" / sp).iterdir()}

    # ---- 读草稿 + 分类统计 ----
    items, per5, drop_person = [], collections.Counter(), 0
    for num, fname in idx:
        stem = Path(fname).stem
        lp = SRC_LBL5 / f"{stem}.txt"
        if not lp.exists():
            print(f"!! 缺草稿标签 #{num} {stem}")
            return 1
        boxes4, keep5 = [], []
        for ln in lp.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            p = ln.split()
            c = int(p[0])
            vals = [float(v) for v in p[1:5]]
            per5[CLS5[c]] += 1
            keep5.append((c, vals))
            if c == 0:
                drop_person += 1
                continue
            boxes4.append((REM[c], vals))
        items.append(dict(num=num, fname=fname, stem=stem, series=series_of(stem),
                          boxes4=boxes4, keep5=keep5))

    print(f"批次 {len(items)} 张；草稿框 {sum(per5.values())}（{dict(per5)}）")
    print(f"入库四类框 {sum(len(i['boxes4']) for i in items)}，丢弃 person 框 {drop_person}")

    # ---- 系列整组划分 ----
    ser = collections.defaultdict(list)
    for it in items:
        ser[it["series"]].append(it)
    order = sorted(ser, key=lambda s: (len(ser[s]), s))
    target = round(len(items) * VALID_RATIO)
    valid_ser, tot = [], 0
    for s in order:
        if tot + len(ser[s]) > target and tot > 0:
            continue
        valid_ser.append(s)
        tot += len(ser[s])
        if tot >= target:
            break
    for it in items:
        it["split"] = "valid" if it["series"] in valid_ser else "train"
    print(f"\n系列 {len(ser)} 个；valid 取 {len(valid_ser)} 个系列 / {tot} 张 "
          f"(target={target})：")
    for s in valid_ser:
        print(f"    valid 系列: {s}  ({len(ser[s])} 张)")
    print(f"train {len(items) - tot} 张 / valid {tot} 张")

    # ---- 命名 + 撞名 ----
    for it in items:
        raw = (SRC_IMG / it["fname"]).read_bytes()
        it["md5"] = hashlib.md5(raw).hexdigest()
        it["new"] = f"{PREFIX}_{it['num']:02d}_{it['md5'][:8]}"
    clash = sorted({i["new"] for i in items} & have)
    print(f"与主库撞名: {len(clash)} {clash[:3]}")

    # ---- 类别分布对照 ----
    newc = collections.Counter()
    for it in items:
        for c, _ in it["boxes4"]:
            newc[c] += 1
    old = collections.Counter()
    for sp in ("train", "valid", "test"):
        for lp in (DS / "labels" / sp).glob("*.txt"):
            for ln in lp.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    old[int(ln.split()[0])] += 1
    print("\n类别   现在   新增   合计")
    for i, n in enumerate(CLASSES):
        print(f"{n:9} {old[i]:5}  +{newc[i]:4}  {old[i] + newc[i]:6}")

    if not apply:
        print("\n[干跑] 未写盘。确认后加 --apply")
        return 0

    # ---- 落盘 ----
    ARCH_LBL5.mkdir(parents=True, exist_ok=True)
    for it in items:
        sp = it["split"]
        shutil.copy2(SRC_IMG / it["fname"], DS / "images" / sp / f"{it['new']}.jpg")
        (DS / "labels" / sp / f"{it['new']}.txt").write_text(
            "".join(f"{c} {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {v[3]:.6f}\n"
                    for c, v in it["boxes4"]), encoding="utf-8")
        # 5 类草稿存档（含 person）
        (ARCH_LBL5 / f"{it['new']}.txt").write_text(
            "".join(f"{c} {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {v[3]:.6f}\n"
                    for c, v in it["keep5"]), encoding="utf-8")

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", encoding="utf-8") as f:
        f.write("new_name,split,num,series,original_name,boxes_4cls,boxes_5cls\n")
        for it in sorted(items, key=lambda x: x["num"]):
            f.write(f"{it['new']},{it['split']},{it['num']},"
                    f"\"{it['series']}\",\"{it['fname']}\",{len(it['boxes4'])},{len(it['keep5'])}\n")

    # ---- 全库复核 ----
    tot_img = tot_box = orphan = 0
    counter = collections.Counter()
    for sp in ("train", "valid", "test"):
        imgs = {p.stem for p in (DS / "images" / sp).iterdir()}
        labs = {p.stem for p in (DS / "labels" / sp).glob("*.txt")}
        orphan += len(labs - imgs)
        tot_img += len(imgs)
        for lp in (DS / "labels" / sp).glob("*.txt"):
            if lp.stem not in imgs:
                continue
            for ln in lp.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    counter[int(ln.split()[0])] += 1
                    tot_box += 1
    print(f"\n入库后全库: {tot_img} 张 / {tot_box} 框  孤儿标签 {orphan}")
    print("  " + " / ".join(f"{CLASSES[i]} {counter[i]}" for i in range(4)))
    print(f"[manifest] {MANIFEST}")
    print(f"[5类存档] {ARCH_LBL5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
