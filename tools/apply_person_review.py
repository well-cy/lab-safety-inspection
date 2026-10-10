# -*- coding: utf-8 -*-
"""把 make_box_editor 导出的改动标签落库（person 补标流程的落库端）。

编辑器只导出「改过的图」（buildText 只遍历 dirty），因此导出文本里的每一图
整图覆盖库内同名标签；未出现在导出里的图维持原样。

用法:
  $PY tools/apply_person_review.py --export outputs/person_review_p0_a/_her_export.txt \
      --restore-stems "stem1,stem2,..."        # 这些图恢复库内 class-4（护目镜）框
可选:
  --split train            落库 split
  --no-backup              跳过 tar 备份（默认全量备份 labels/ 到 _label_backup/）
  --dry                    只干跑，打印将写入的内容统计
落库后自动做全库格式校验 + 按类别对账。
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "data" / "lab_ppe"
SPLITS = ("train", "valid", "test")
NAMES = ["person", "mask", "gloves", "lab_coat", "goggles"]


def parse_export(txt: str) -> dict[str, list[tuple]]:
    items, cur = {}, None
    for ln in txt.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if ln.startswith("###"):
            cur = ln[3:].strip()
            items[cur] = []
        elif cur and len(ln.split()) == 5:
            p = ln.split()
            items[cur].append((int(p[0]), *(round(float(v), 6) for v in p[1:5])))
    return items


def read_rows(p: Path) -> list[tuple]:
    if not p.exists():
        return []
    rows = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        q = ln.split()
        if len(q) == 5:
            rows.append((int(q[0]), *(float(v) for v in q[1:5])))
    return rows


def near(a, b, eps=1e-4):
    return a[0] == b[0] and all(abs(a[k] - b[k]) < eps for k in range(1, 5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--restore-stems", default="")
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    exp = parse_export(Path(args.export).read_text(encoding="utf-8"))
    restore = {s.strip() for s in args.restore_stems.split(",") if s.strip()}
    LBL = LAB / "labels" / args.split

    # ---- 合成最终标签 ----
    final: dict[str, list[tuple]] = {}
    stats = {"imgs": len(exp), "restored": 0, "bad_stem": []}
    for stem, rows in exp.items():
        lib = read_rows(LBL / (stem + ".txt"))
        out = list(rows)
        if stem in restore:
            for r in lib:
                if r[0] == 4 and not any(near(r, s) for s in out):
                    out.append(r)
                    stats["restored"] += 1
        # 收尾：裁剪到 [0,1]，去掉零面积框
        clean = []
        for c, cx, cy, w, h in out:
            x1 = min(max(cx - w / 2, 0.0), 1.0)
            y1 = min(max(cy - h / 2, 0.0), 1.0)
            x2 = min(max(cx + w / 2, 0.0), 1.0)
            y2 = min(max(cy + h / 2, 0.0), 1.0)
            if x2 - x1 < 1e-4 or y2 - y1 < 1e-4:
                continue
            clean.append((c, round((x1 + x2) / 2, 6), round((y1 + y2) / 2, 6),
                          round(x2 - x1, 6), round(y2 - y1, 6)))
        final[stem] = clean
        if not lib and not clean:
            stats["bad_stem"].append(stem)

    if args.dry:
        for stem, rows in final.items():
            c = Counter(r[0] for r in rows)
            tag = " [恢复goggles]" if stem in restore else ""
            print(f"{stem[:44]:46} {len(rows)}框 person{c[0]} {tag}")
        print(f"\n干跑 {stats['imgs']} 图，恢复 goggles 框 {stats['restored']} 个")
        return 0

    # ---- 备份 ----
    if not args.no_backup:
        ts = time.strftime("%Y%m%d")
        bk = LAB / "_label_backup" / f"labels_{args.split}_person_{ts}.tar.gz"
        bk.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(bk, "w:gz") as tf:
            for f in sorted((LAB / "labels" / args.split).glob("*.txt")):
                tf.add(f, arcname=f.name)
        print(f"备份 → {bk.relative_to(ROOT)}（{len(list((LAB / 'labels' / args.split).glob('*.txt')))} 文件）")

    # ---- 落库 ----
    for stem, rows in final.items():
        txt = "\n".join(" ".join(str(v) for v in r) for r in rows) + ("\n" if rows else "")
        (LBL / (stem + ".txt")).write_text(txt, encoding="utf-8", newline="\n")
    print(f"落库 {len(final)} 图（恢复 goggles 框 {stats['restored']} 个）")

    # ---- 全库校验 + 对账 ----
    bad = orphan_img = orphan_lab = 0
    tot = Counter()
    for sp in SPLITS:
        c = Counter()
        n = 0
        for f in (LAB / "labels" / sp).glob("*.txt"):
            n += 1
            for ln in f.read_text(encoding="utf-8").splitlines():
                p = ln.split()
                if len(p) != 5:
                    bad += 1
                    continue
                try:
                    k = int(p[0])
                    vals = [float(v) for v in p[1:5]]
                except ValueError:
                    bad += 1
                    continue
                if not (0 <= k <= 4) or any(v < -1e-4 or v > 1.0001 for v in vals):
                    bad += 1
                    continue
                c[k] += 1
        imgs = {p.stem for p in (LAB / "images" / sp).iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
        labs = {f.stem for f in (LAB / "labels" / sp).glob("*.txt")}
        orphan_img += len(imgs - labs)
        orphan_lab += len(labs - imgs)
        tot.update(c)
        print(f"{sp}: {n} 图 / {sum(c.values())} 框  " +
              " / ".join(f"{NAMES[k]}{c[k]}" for k in range(5)))
    print(f"全库: {sum(tot.values())} 框  " + " / ".join(f"{NAMES[k]}{tot[k]}" for k in range(5)),
          f" 坏行{bad} 孤儿图{orphan_img} 孤儿标签{orphan_lab}")
    return 0 if bad == 0 and orphan_img == 0 and orphan_lab == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
