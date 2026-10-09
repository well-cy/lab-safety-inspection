# -*- coding: utf-8 -*-
"""E6 §3 lab_ppe 整库 5 类化迁移（原子手术执行器）。

三步（--apply 一次完成，默认干跑）:
  1. 快照备份 labels/{train,valid,test} + data.yaml → data/_label_backup/lab_ppe_5cls_<date>/
  2. 全库类号 +1（0..3 → 1..4），幂等保护（data.yaml 非 nc:4 即拒绝）
  3. 网图批 123 图以 labels_5cls 内容整体替换（person 0 + 非 person 1..4），
     替换前逐字段校验"非 person 框 == 主库现状框（左移后）"
  4. data.yaml → nc:5 五类
  5. 复检: 0 坏行 / 类别对账

用法:
  PY=.venv/Scripts/python.exe
  $PY tools/migrate_5cls.py            # 干跑
  $PY tools/migrate_5cls.py --apply
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import io
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

LAB = ROOT / "data" / "lab_ppe"
FIVE = ROOT / "data" / "raw" / "web_lab_scene" / "labels_5cls"
SPLITS = ["train", "valid", "test"]
NAMES5 = {0: "person", 1: "mask", 2: "gloves", 3: "lab_coat", 4: "goggles"}


def rd(p: Path):
    return [l.split() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def norm(rows):
    return [(int(r[0]), round(float(r[1]), 6), round(float(r[2]), 6),
             round(float(r[3]), 6), round(float(r[4]), 6)) for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    # ① 幂等/健康前置校验
    dy = (LAB / "data.yaml").read_text(encoding="utf-8")
    if "nc: 4" not in dy.replace("nc:4", "nc: 4"):
        print("[中止] data.yaml 不是 nc:4 —— 可能已迁移过。")
        return 1
    bad = 0
    for sp in SPLITS:
        for f in glob.glob(str(LAB / "labels" / sp / "*.txt")):
            for ln in open(f, encoding="utf-8"):
                t = ln.split()
                if not t:
                    continue
                if len(t) != 5 or not t[0].isdigit() or int(t[0]) > 3:
                    bad += 1
    if bad:
        print(f"[中止] 现有标签 {bad} 坏行，先修复再迁移。")
        return 1
    print("[校验] nc:4 ✓ 全库 0 坏行 ✓")

    # ② 逐字段校验 labels_5cls 非 person 框 == 主库（左移后）
    idx = {}
    for sp in SPLITS:
        for p in glob.glob(str(LAB / "images" / sp / "*")):
            idx[Path(p).stem] = sp
    five_files = sorted(FIVE.glob("*.txt"))
    mism = []
    for f in five_files:
        stem = f.stem
        if stem not in idx:
            mism.append((stem, "5cls 图不在主库"))
            continue
        sp = idx[stem]
        c5 = rd(f)
        nonp = norm([(int(r[0]) - 1, *r[1:]) for r in c5 if int(r[0]) != 0])
        cur = norm(rd(LAB / "labels" / sp / (stem + ".txt")))
        if nonp != cur:
            mism.append((stem, f"5cls非P {len(nonp)} 框 ≠ 主库 {len(cur)} 框或坐标不同"))
    if mism:
        print(f"[中止] labels_5cls 与主库不一致 {len(mism)} 张：")
        for m in mism[:20]:
            print("  ", m[0], m[1])
        return 1
    n_person = sum(sum(1 for r in rd(f) if int(r[0]) == 0) for f in five_files)
    print(f"[校验] labels_5cls {len(five_files)} 张全部一致，person 框 {n_person} ✓")

    # ③ 干跑输出计划
    tot = Counter()
    for sp in SPLITS:
        for f in glob.glob(str(LAB / "labels" / sp / "*.txt")):
            for ln in open(f, encoding="utf-8"):
                if ln.strip():
                    tot[int(ln.split()[0])] += 1
    print(f"[计划] 迁移框数 {sum(tot.values())}（mask{tot[0]}/gloves{tot[1]}/lab_coat{tot[2]}/goggles{tot[3]} → +1）")
    print(f"[计划] 网图批替换 {len(five_files)} 图，并入 person {n_person} 框")
    if not args.apply:
        print("[干跑] 加 --apply 执行。")
        return 0

    # ④ 备份校验（沙箱对批量建文件会 SIGTERM，备份必须用单文件 tar 归档，外部完成）
    arch = ROOT / "data" / "_label_backup" / "lab_ppe_5cls_20261009.tar.gz"
    expect = sum(len(glob.glob(str(LAB / "labels" / sp / "*.txt"))) for sp in SPLITS) + 1  # + data.yaml
    if arch.exists():
        import tarfile
        with tarfile.open(arch, "r:gz") as tf:
            n = sum(1 for m in tf.getmembers() if m.isfile())
        if n != expect:
            print(f"[中止] 备份归档不完整 {n}/{expect}")
            return 1
        print(f"[备份] 归档已验证 {arch.name}（{n} 文件）")
    else:
        print(f"[中止] 缺备份归档 {arch}")
        print("先执行: tar -C data/lab_ppe -czf data/_label_backup/lab_ppe_5cls_20261009.tar.gz labels data.yaml")
        return 1

    # ⑤ 类号 +1
    for sp in SPLITS:
        for f in glob.glob(str(LAB / "labels" / sp / "*.txt")):
            fp = Path(f)
            rows = rd(fp)
            out = "\n".join(f"{int(r[0]) + 1} {r[1]} {r[2]} {r[3]} {r[4]}" for r in rows)
            fp.write_text(out + ("\n" if rows else ""), encoding="utf-8", newline="\n")
    print("[迁移] 全库类号 +1 完成")

    # ⑥ 网图批替换
    for f in five_files:
        sp = idx[f.stem]
        rows = rd(f)
        out = "\n".join(f"{r[0]} {r[1]} {r[2]} {r[3]} {r[4]}" for r in rows)
        (LAB / "labels" / sp / (f.stem + ".txt")).write_text(out + ("\n" if rows else ""),
                                                             encoding="utf-8", newline="\n")
    print(f"[回补] {len(five_files)} 图替换为 5cls 标签（person {n_person} 框并入）")

    # ⑦ data.yaml
    (LAB / "data.yaml").write_text(
        "path: D:/ZuoYe/lab-safety-inspection/data/lab_ppe\n"
        "train: images/train\nval: images/valid\ntest: images/test\nnc: 5\nnames:\n"
        "  0: person\n  1: mask\n  2: gloves\n  3: lab_coat\n  4: goggles\n",
        encoding="utf-8")
    print("[yaml] nc:5 五类")

    # ⑧ 复检
    bad2 = 0
    cnt = Counter()
    nfiles = 0
    for sp in SPLITS:
        for f in glob.glob(str(LAB / "labels" / sp / "*.txt")):
            nfiles += 1
            for ln in open(f, encoding="utf-8"):
                t = ln.split()
                if not t:
                    continue
                if len(t) != 5 or not t[0].isdigit() or int(t[0]) > 4:
                    bad2 += 1
                else:
                    cnt[int(t[0])] += 1
    print(f"[复检] 标签 {nfiles} 文件 / {sum(cnt.values())} 框 / 坏行 {bad2}")
    print("[复检] 分布:", " ".join(f"{NAMES5[k]}={v}" for k, v in sorted(cnt.items())))
    return 0 if bad2 == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
