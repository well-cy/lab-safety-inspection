"""uploads 第0批她判 #44-#95 落地（第二批）。

判定来源：她 2026-10-08 回报（38 条），逐图核实后定案：
- 剔图 27 张（重复/车间/综艺/新闻病房/海报误检），移入 _rejected（与第一批同 tag，可回滚）
- 删框 15（防毒面具/针织衫/眼镜误框、person 多框合并、桌上黑布/鞋误框手套）
- 修框 10（person 全身合并、goggles 偏差、lab_coat 双框合并）
- 补框 7（lab_coat×3、mask×2、person×1、lab_coat×1）

用法:
    python tools/apply_uploads_verdict2.py          # 干跑
    python tools/apply_uploads_verdict2.py --apply  # 执行
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "data" / "test" / "uploads"
LBL = ROOT / "data" / "_prelabel" / "uploads_20261008" / "labels"
BK = ROOT / "data" / "_label_backup" / "uploads_verdict2_20261008"
REJ = ROOT / "data" / "_rejected" / "uploads_reject_20261008"  # 与第一批同 tag

# ---------------- 剔图（她判重复/非实验室/错误过多；每组保留已修的一张） ----------------
REJECTS = [
    ("img_76c39075.jpg",  "#45 重复(#28)"),
    ("img_85d0176e.jpg",  "#48 重复(#37)"),
    ("img_8797ad08.jpg",  "#50 重复(#11/#43组)"),
    ("img_884458c3.jpg",  "#51 车间钻床(与#5/#13同源)"),
    ("img_9530da51.jpg",  "#57 重复(#27已修)"),
    ("img_96035d36.jpg",  "#58 她判重复(同场景连拍,非像素重复)+防毒面具误标口罩"),
    ("img_9aabd540.jpg",  "#60 重复(#37组)"),
    ("img_9bae4800.jpg",  "#61 重复(#10已修)"),
    ("img_a06d3652.jpg",  "#62 综艺演播厅非实验室(Kapil Sharma Show)"),
    ("img_a4f506e0.jpg",  "#64 重复(#17已修)"),
    ("img_a69e84c4.jpg",  "#65 重复(#53, phash)"),
    ("img_a69f30b4.jpg",  "#66 她判重复(#58同系列连拍,非像素重复)"),
    ("img_b21b2bfb.jpg",  "#67 重复(#23/#24/#80组)"),
    ("img_b46f1a2c.jpg",  "#69 她判剔除(错误过多:防毒面具+耐高温手套)"),
    ("img_b8e70e32.jpg",  "#73 她判剔除(新闻图病房,Global Times水印)"),
    ("img_ba04f02b.jpg",  "#74 她判剔除(≡#9)"),
    ("img_c4136295.jpg",  "#78 重复剔除(≡#9)"),
    ("img_c6688fff.jpg",  "#80 重复(#23/#24/#67组)"),
    ("img_cd35df6f.jpg",  "#81 重复(#33已修)"),
    ("img_da17e46e.jpg",  "#82 重复(#51车间)"),
    ("img_da442253.jpg",  "#83 她判重复(白大褂商品图家族,非像素重复)"),
    ("img_e145c38a.jpg",  "#84 重复(#75)"),
    ("img_eb7a4a3f.jpg",  "#85 车间钻床非实验室"),
    ("img_f573ca58.png",  "#89 她判剔除(办公室海报人物误检+衬衫误标lab_coat)"),
    ("img_f72b0ee4.jpg",  "#90 她判剔除(车间铣床)"),
    ("img_fac73146.jpeg", "#91 重复(#46)"),
    ("img_fff68bb0.jpg",  "#95 她判剔除(≡#78/#9病房组)"),
]

# ---------------- 删框 (stem, old_index, class, 说明) ----------------
DELS = [
    ("img_721ca04b", 0, 0, "#44 person双框合并删一"),
    ("img_7f430e3b", 3, 0, "#47 person双框合并删一"),
    ("img_8dc96304", 3, 3, "#53 防护服不算实验服"),
    ("img_8fbb49f6", 0, 3, "#54 白色针织衫不算实验服"),
    ("img_8fda114e", 0, 4, "#55 眼镜误框护目镜"),
    ("img_91dee0ed", 0, 0, "#56 person双框合并删一"),
    ("img_b5ccdccf", 4, 2, "#70 桌上黑布误框手套"),
    ("img_b5ccdccf", 7, 3, "#70 lab_coat双框合并删一"),
    ("img_b725100a", 3, 0, "#72 person三框合并删一"),
    ("img_b725100a", 4, 0, "#72 person三框合并删二"),
    ("img_bca17019", 4, 0, "#76 跨两人第三框"),
    ("img_bda59d18", 4, 0, "#77 person上下双框合并删一"),
    ("img_c45d2240", 1, 0, "#79 person双框合并删一"),
    ("img_ec6b1b8d", 2, 0, "#86 person双框合并删一"),
    ("img_ed8a1e3e", 2, 2, "#87 白鞋误框手套"),
]

# ---------------- 修框 (stem, old_index, cx, cy, w, h) ----------------
SETS = [
    ("img_721ca04b", 2, 0.515, 0.585, 0.970, 0.830),  # #44 person 全身合并
    ("img_7f430e3b", 2, 0.427, 0.608, 0.428, 0.692),  # #47 person 全身合并
    ("img_91dee0ed", 2, 0.464, 0.514, 0.738, 0.973),  # #56 person 全身合并
    ("img_b5ccdccf", 2, 0.764, 0.625, 0.342, 0.694),  # #70 lab_coat 双框合并
    ("img_b5ccdccf", 6, 0.600, 0.325, 0.100, 0.068),  # #70 goggles 偏差修框
    ("img_b725100a", 0, 0.500, 0.525, 1.000, 0.950),  # #72 person 特写合并
    ("img_b725100a", 2, 0.616, 0.384, 0.639, 0.343),  # #72 goggles 偏差修框
    ("img_bda59d18", 1, 0.497, 0.497, 0.921, 0.904),  # #77 person 全身合并
    ("img_c45d2240", 0, 0.625, 0.510, 0.750, 0.970),  # #79 person 合并
    ("img_ec6b1b8d", 1, 0.510, 0.509, 0.960, 0.982),  # #86 person 合并
]

# ---------------- 补框 (stem, class, cx, cy, w, h, 说明) ----------------
ADDS = [
    ("img_721ca04b", 3, 0.680, 0.688, 0.640, 0.625, "#44 实验服补框"),
    ("img_7d4c235e", 3, 0.481, 0.493, 0.648, 0.870, "#46 商品图白大褂补框"),
    ("img_8fbb49f6", 1, 0.551, 0.312, 0.408, 0.220, "#54 口罩补框"),
    ("img_8fda114e", 1, 0.581, 0.427, 0.375, 0.286, "#55 口罩补框"),
    ("img_f34aa3f7", 0, 0.431, 0.555, 0.435, 0.736, "#88 女科学家person补框"),
    ("img_f34aa3f7", 3, 0.440, 0.687, 0.417, 0.473, "#88 女科学家实验服补框"),
    ("img_f34aa3f7", 3, 0.786, 0.500, 0.424, 0.475, "#88 男科学家实验服补框"),
]

CLS = {0: "person", 1: "mask", 2: "gloves", 3: "lab_coat", 4: "goggles"}


def validate_line(line: str) -> bool:
    p = line.split()
    if len(p) != 5:
        return False
    try:
        c = int(float(p[0]))
        vals = [float(v) for v in p[1:]]
    except ValueError:
        return False
    if c not in (0, 1, 2, 3, 4):
        return False
    return all(0.0 <= v <= 1.0 for v in vals)


def read_lines(stem: str) -> list[str]:
    lp = LBL / f"{stem}.txt"
    if not lp.exists():
        return []
    return [l for l in lp.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_lines(stem: str, lines: list[str]) -> None:
    (LBL / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    mode = "执行" if args.apply else "干跑"

    # 备份所有涉改标签
    stem_set = {s for s, *_ in DELS} | {s for s, *_ in SETS} | {s for s, *_ in ADDS}
    rej_stems = {Path(f).stem for f, _ in REJECTS}
    if args.apply:
        BK.mkdir(parents=True, exist_ok=True)
        n_bk = 0
        for st in sorted(stem_set | rej_stems):
            lp = LBL / f"{st}.txt"
            if lp.exists():
                shutil.copy2(lp, BK / lp.name)
                n_bk += 1
        print(f"[备份] {n_bk} 个标签 -> {BK.relative_to(ROOT)}")

    # 1) 剔图
    n_rej = 0
    for fname, why in REJECTS:
        ip, lp = IMG / fname, LBL / f"{Path(fname).stem}.txt"
        if not ip.exists():
            print(f"  [跳过] {fname} 不存在（可能已剔）")
            continue
        print(f"  [剔图] {fname}  {why}")
        if args.apply:
            (REJ / "images").mkdir(parents=True, exist_ok=True)
            (REJ / "labels").mkdir(parents=True, exist_ok=True)
            os.replace(ip, REJ / "images" / fname)
            if lp.exists():
                os.replace(lp, REJ / "labels" / lp.name)
        n_rej += 1

    # 2) 删框（index 从大到小避免错位）
    n_del = 0
    by_stem: dict[str, list] = {}
    for st, idx, cls, why in DELS:
        by_stem.setdefault(st, []).append((idx, cls, why))
    for st, items in by_stem.items():
        lines = read_lines(st)
        for idx, cls, why in sorted(items, key=lambda t: -t[0]):
            if idx >= len(lines):
                print(f"  [错误] {st} [{idx}] 越界（共 {len(lines)} 行）")
                continue
            c = int(float(lines[idx].split()[0]))
            tag = "OK" if c == cls else f"类别不符(实为{CLS.get(c, '?')})"
            print(f"  [删框] {st} [{idx}] {CLS.get(cls, '?')}  {tag}  {why}")
            if args.apply and c == cls:
                del lines[idx]
        if args.apply:
            write_lines(st, lines)
        n_del += len(items)

    # 3) 修框
    n_set = 0
    set_by_stem: dict[str, list] = {}
    for st, idx, cx, cy, w, h in SETS:
        set_by_stem.setdefault(st, []).append((idx, cx, cy, w, h))
    for st, items in set_by_stem.items():
        lines = read_lines(st)
        for idx, cx, cy, w, h in items:
            if idx >= len(lines):
                print(f"  [错误] {st} [{idx}] 越界")
                continue
            p = lines[idx].split()
            print(f"  [修框] {st} [{idx}] {CLS.get(int(float(p[0])), '?')} -> ({cx},{cy},{w},{h})")
            if args.apply:
                lines[idx] = f"{int(float(p[0]))} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        if args.apply:
            write_lines(st, lines)
        n_set += len(items)

    # 4) 补框
    n_add = 0
    add_by_stem: dict[str, list] = {}
    for st, cls, cx, cy, w, h, why in ADDS:
        add_by_stem.setdefault(st, []).append((cls, cx, cy, w, h, why))
    for st, items in add_by_stem.items():
        lines = read_lines(st)
        for cls, cx, cy, w, h, why in items:
            print(f"  [补框] {st} +{CLS.get(cls, '?')} ({cx},{cy},{w},{h})  {why}")
            if args.apply:
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        if args.apply:
            write_lines(st, lines)
        n_add += len(items)

    # 5) 全库格式校验 + 对账
    if args.apply:
        bad = 0
        cnt: dict[str, int] = {}
        n_img, n_box = 0, 0
        for lp in sorted(LBL.glob("*.txt")):
            n_img += 1
            for line in lp.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                if not validate_line(line):
                    print(f"  [坏行] {lp.name}: {line!r}")
                    bad += 1
                    continue
                cnt[CLS[int(float(line.split()[0]))]] = cnt.get(CLS[int(float(line.split()[0]))], 0) + 1
                n_box += 1
        print(f"\n[校验] 坏行 {bad}；草稿 {n_img} 图 / {n_box} 框")
        for k in ("person", "mask", "gloves", "lab_coat", "goggles"):
            print(f"    {k}: {cnt.get(k, 0)}")

    print(f"\n{mode}汇总: 剔图 {n_rej}  删框 {n_del}  修框 {n_set}  补框 {n_add}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
