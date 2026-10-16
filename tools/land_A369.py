# -*- coding: utf-8 -*-
"""A档369 预标注审核批次落库 train/valid（test 红线不动）。

四个批次的终稿来源：
  m1  -> outputs/_A369/staged/m1/*.txt        （用户导出已应用）+ 恢复被误删的原始 gloves 框
  m2  -> outputs/_A369/staged/m2/*.txt        （用户导出已应用）
  c35 -> outputs/_A369/editor/ws/ws_a369_c35/labels/*.txt （无导出=未改动=接受）
  u8  -> outputs/_A369/editor/ws/ws_a369_u8/labels/*.txt  （无导出=未改动=接受）

流程：组终稿 -> 校验 -> MD5 去重(对全库+批内) -> 备份 -> 90/10 train/valid 落库。
默认干跑；--apply 才写 data/lab_ppe。
"""
import argparse
import hashlib
import random
import shutil
import tarfile
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "outputs" / "_A369"
PKG = A / "A档_6类预标注包"
STAGED = A / "staged"
WS = A / "editor" / "ws"
DATA = ROOT / "data" / "lab_ppe"
N_CLASSES = 6
SEED = 20261016

IMG_EXT = {".jpg"}


def parse_index(path: Path):
    names = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        col = s.split("\t")[1] if "\t" in s else s  # index 格式：编号\t文件\t族\t...
        if Path(col).suffix.lower() in IMG_EXT:
            names.append(col)
    return names


def load_label(path: Path, stem: str):
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 5:
            raise SystemExit(f"[坏行] {stem}: {s!r}")
        cls = int(parts[0])
        if not (0 <= cls < N_CLASSES):
            raise SystemExit(f"[类别越界] {stem}: cls={cls}")
        vals = [float(x) for x in parts[1:5]]
        for v in vals:
            if v < -0.001 or v > 1.001:
                raise SystemExit(f"[坐标越界] {stem}: {s}")
        vals = [min(max(v, 0.0), 1.0) for v in vals]
        out.append(" ".join([str(cls)] + [f"{v:.6f}" for v in vals]))
    return out


def box_key(line: str):
    parts = line.split()
    return (parts[0],) + tuple(round(float(x), 4) for x in parts[1:5])


def md5_of(p: Path, chunk=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真正写入 data/lab_ppe")
    args = ap.parse_args()

    pkg_images = sorted(
        p for p in (PKG / "images").iterdir() if p.suffix.lower() in IMG_EXT
    )
    pkg_names = {p.name: p for p in pkg_images}
    print(f"包内图片: {len(pkg_images)}")

    # 1. 批次成员
    batches = {}
    for b in ("u8", "c35", "m1", "m2"):
        batches[b] = parse_index(A / "editor" / f"a369_{b}" / "index.txt")
        print(f"  批{b}: index {len(batches[b])} 张")
    all_names = [n for b in batches for n in batches[b]]
    assert len(all_names) == len(set(all_names)) == len(pkg_names) == 369, (
        f"批次成员覆盖异常: 总数={len(all_names)} 去重={len(set(all_names))}"
    )

    # 2. 组终稿
    final, restore_log = {}, []
    for b, names in batches.items():
        for name in names:
            stem = Path(name).stem
            if b in ("m1", "m2"):
                sp = STAGED / b / f"{stem}.txt"
                if not sp.exists():
                    raise SystemExit(f"[缺staged] {b}/{stem}.txt")
                lines = load_label(sp, stem)
                if b == "m1":  # 恢复被误删的原始 gloves 框（skill 硬约束④）
                    op = PKG / "labels_original" / f"{stem}.txt"
                    staged_keys = {box_key(l) for l in lines}
                    for ol in load_label(op, stem):
                        if ol.split()[0] == "2" and box_key(ol) not in staged_keys:
                            lines.append(ol)
                            restore_log.append((stem, ol))
            else:
                wp = WS / f"ws_a369_{b}" / "labels" / f"{stem}.txt"
                if not wp.exists():
                    raise SystemExit(f"[缺ws] {b}/{stem}.txt")
                lines = load_label(wp, stem)
            final[name] = lines
    print(f"终稿组齐: {len(final)} 张 | m1 恢复误删 gloves 框: {len(restore_log)} 个")

    # 3. 迷你框扫描（真误拖特征 w/h<=0.02）
    minis = [
        (n, l) for n, ls in final.items() for l in ls
        if float(l.split()[3]) <= 0.02 or float(l.split()[4]) <= 0.02
    ]
    for n, l in minis:
        print(f"  [迷你框] {n}: {l}")

    # 4. MD5 去重：对全库 + 批内
    lib_md5 = {}
    for split in ("train", "valid", "test"):
        for p in (DATA / "images" / split).iterdir():
            lib_md5.setdefault(md5_of(p), p.name)
    seen, dup_lib, dup_internal = {}, [], []
    for name in sorted(final):
        h = md5_of(pkg_names[name])
        if h in lib_md5:
            dup_lib.append((name, lib_md5[h]))
        elif h in seen:
            dup_internal.append((name, seen[h]))
        else:
            seen[h] = name
    land = [n for n in sorted(final) if n not in {d[0] for d in dup_lib + dup_internal}]
    print(f"去重: 与库内重复 {len(dup_lib)} | 批内重复 {len(dup_internal)} | 待落 {len(land)}")
    for n, old in dup_lib + dup_internal:
        print(f"  [重复] {n} == {old}")

    # 5. 现名冲突
    existing = {p.name for s in ("train", "valid") for p in (DATA / "images" / s).iterdir()}
    clash = [n for n in land if n in existing]
    if clash:
        raise SystemExit(f"[文件名冲突] {clash[:5]}")

    # 6. 统计
    def stat(names):
        c = Counter()
        for n in names:
            for l in final[n]:
                c[int(l.split()[0])] += 1
        return c

    before = stat([])
    lib_boxes = Counter()
    for s in ("train", "valid"):
        for p in (DATA / "labels" / s).iterdir():
            for ln in p.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    lib_boxes[int(ln.split()[0])] += 1
    new_boxes = stat(land)
    print("落库框类别增量:", dict(sorted(new_boxes.items())))
    print("库内框类别(train+valid):", dict(sorted(lib_boxes.items())))

    if not args.apply:
        print("\n[干跑] 加 --apply 执行落库。")
        return

    # 7. 备份现有 train/valid 标签
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = DATA / "_label_backup" / f"labels_pre_A369_{ts}.tar.gz"
    with tarfile.open(bak, "w:gz") as tf:
        for s in ("train", "valid"):
            tf.add(DATA / "labels" / s, arcname=f"labels_{s}")
    print(f"备份: {bak.name}")

    # 8. 90/10 切分并落库
    rng = random.Random(SEED)
    shuffled = land[:]
    rng.shuffle(shuffled)
    n_valid = max(1, round(len(shuffled) * 0.1))
    to_valid, to_train = set(shuffled[:n_valid]), set(shuffled[n_valid:])
    for name in land:
        split = "valid" if name in to_valid else "train"
        shutil.copy2(pkg_names[name], DATA / "images" / split / name)
        (DATA / "labels" / split / f"{Path(name).stem}.txt").write_text(
            "\n".join(final[name]) + "\n", encoding="utf-8"
        )
    print(f"落库完成: train +{len(land) - len(to_valid)} / valid +{len(to_valid)}")

    # 9. 终态对账
    for s in ("train", "valid", "test"):
        ni = len(list((DATA / "images" / s).iterdir()))
        nl = len(list((DATA / "labels" / s).iterdir()))
        print(f"  {s}: img={ni} lbl={nl} {'✅' if ni == nl else '❌'}")


if __name__ == "__main__":
    main()
