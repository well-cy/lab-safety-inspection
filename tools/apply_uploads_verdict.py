"""uploads 第 0 批她判（#1–#43）落地：删框/修框/补框 + 剔图。

- 草稿：data/_prelabel/uploads_20261008/labels/（0=person 1=mask 2=gloves 3=lab_coat 4=goggles）
- 图：data/test/uploads/
- 备份：data/_label_backup/uploads_verdict_20261008/
- 剔图：图+草稿标签移动到 data/_rejected/uploads_reject_20261008/{images,labels}（os.replace，永不删除）

用法: python tools/apply_uploads_verdict.py [--apply]
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LBL = ROOT / "data" / "_prelabel" / "uploads_20261008" / "labels"
IMG = ROOT / "data" / "test" / "uploads"
BK = ROOT / "data" / "_label_backup" / "uploads_verdict_20261008"
REJ = ROOT / "data" / "_rejected" / "uploads_reject_20261008"
CLS = {0: "person", 1: "mask", 2: "gloves", 3: "lab_coat", 4: "goggles"}

# (stem, old_index, class, 动作说明) —— old_index 以 dump 行序为准
DELS = [
    ("img_006c8f46", 2, 4, "#1 眼镜不是护目镜"),
    ("img_01cc0be0", 2, 3, "#3 左下角试剂瓶不是实验服"),
    ("img_0a4aeefa", 1, 3, "#9 防护服不算 lab_coat"),
    ("img_104e87bc", 4, 3, "#10 没有人穿实验服"),
    ("img_2c0a9a74", 2, 1, "#18 口罩是防毒面具"),
    ("img_2c0a9a74", 3, 2, "#18 手套是劳保手套"),
    ("img_369557a4", 4, 1, "#22 耳钉误框为口罩"),
    ("img_3afdb65b", 1, 2, "#24 没戴手套"),
    ("img_47e8a531", 4, 3, "#26 桌子误框为实验服"),
    ("img_5cedb3af", 5, 3, "#33 防护服不算 lab_coat"),
    ("img_5ffe514b", 1, 1, "#36 防毒面具不算口罩"),
    ("img_64166c86", 3, 2, "#37 手臂误框为手套"),
    ("img_704df2fb", 1, 1, "#42 防毒面具不算口罩"),
    ("img_704df2fb", 2, 1, "#42 防毒面具不算口罩"),
]
# (stem, old_index, new cx cy w h)
SETS = [
    ("img_15720e9f", 0, 0.513, 0.534, 0.902, 0.440),  # #12 goggles 框扩到全镜
    ("img_2905d870", 1, 0.528, 0.561, 0.804, 0.879),  # #17 person 合并为全身
    ("img_5794be96", 1, 0.514, 0.546, 0.953, 0.910),  # #31 person 合并为全身
]
ADDS = [
    ("img_03450963", 2, 0.161, 0.255, 0.204, 0.136, "#4 补左手丁腈手套"),
    ("img_03450963", 2, 0.854, 0.248, 0.214, 0.126, "#4 补右手丁腈手套"),
    ("img_056ba768", 1, 0.774, 0.508, 0.059, 0.069, "#7 补最右人口罩"),
    ("img_056ba768", 4, 0.778, 0.432, 0.074, 0.091, "#7 补最右人护目镜"),
    ("img_0a4aeefa", 1, 0.614, 0.798, 0.083, 0.082, "#9 补病床男口罩"),
    ("img_2905d870", 4, 0.527, 0.250, 0.259, 0.111, "#17 补护目镜"),
    ("img_3170b5e9", 3, 0.453, 0.461, 0.131, 0.274, "#20 补后面人实验服"),
    ("img_392f916c", 4, 0.375, 0.160, 0.123, 0.040, "#23 补护目镜"),
    ("img_405ce451", 3, 0.543, 0.460, 0.347, 0.459, "#25 补实验服"),
    ("img_492d78d6", 3, 0.519, 0.552, 0.784, 0.581, "#27 补实验服"),
    ("img_4b608c01", 1, 0.529, 0.333, 0.106, 0.122, "#28 补口罩"),
    ("img_4b608c01", 4, 0.536, 0.267, 0.144, 0.063, "#28 补护目镜"),
    ("img_5794be96", 4, 0.490, 0.220, 0.335, 0.147, "#31 补护目镜"),
    ("img_71be509e", 0, 0.323, 0.522, 0.647, 0.613, "#43 补人物框"),
]
REJECTS = [
    ("img_08687aa5.jpg", "#8 她判剔除(大堂摆拍防毒面具)"),
    ("img_6728ca30.jpg", "#39 她判剔除(温室非实验室)"),
    ("img_0347c108.jpg", "#5 车间钻床场景(既定口径)+劳保手套; ≡#13"),
    ("img_1e5a37d8.jpg", "#13 车间钻床场景(既定口径); ≡#5"),
    ("img_5d81db77.jpg", "#34 ≡#22 完全同图(md5同)"),
    ("img_5e948d90.jpg", "#35 ≡#23 完全同图"),
    ("img_5b68d326.jpg", "#32 ≡#24 完全同图"),
    ("img_658ce72f.jpg", "#38 ≡#28 完全同图"),
    ("img_6a7108a5.jpg", "#40 ≡#36 完全同图"),
    ("img_3181c97c.jpg", "#21 ≡#1 完全同图(md5同)"),
]
TOUCHED = {d[0] for d in DELS} | {s[0] for s in SETS} | {a[0] for a in ADDS}


def validate(path: Path) -> None:
    for l in path.read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        q = l.split()
        assert len(q) == 5, (path, l)
        c = int(float(q[0]))
        assert c in CLS and all(0 <= float(v) <= 1 for v in q[1:]), (path, l)


def main() -> int:
    apply = "--apply" in sys.argv
    n_del = n_set = n_add = 0
    for stem, idx, c, note in DELS:
        p = LBL / f"{stem}.txt"
        lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        q = lines[idx].split()
        assert len(q) == 5 and int(float(q[0])) == c, (stem, idx, lines[idx], note)
        n_del += 1
    print(f"干跑: 删框 {len(DELS)} 修框 {len(SETS)} 补框 {len(ADDS)} 剔图 {len(REJECTS)}")
    if not apply:
        return 0

    BK.mkdir(parents=True, exist_ok=True)
    for stem in TOUCHED:
        shutil.copy2(LBL / f"{stem}.txt", BK / f"{stem}.txt")
    print(f"备份 {len(TOUCHED)} 个草稿标签 -> {BK.name}/")

    # 删框（从大到小的 index 逐文件处理避免错位）
    by_file: dict[str, list] = {}
    for stem, idx, c, note in DELS:
        by_file.setdefault(stem, []).append((idx, c, note))
    for stem, items in by_file.items():
        p = LBL / f"{stem}.txt"
        lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        for idx, c, note in sorted(items, key=lambda x: -x[0]):
            q = lines[idx].split()
            assert int(float(q[0])) == c, (stem, idx)
            del lines[idx]
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # 修框
    for stem, idx, cx, cy, w, h in SETS:
        p = LBL / f"{stem}.txt"
        lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        q = lines[idx].split()
        lines[idx] = f"{q[0]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # 补框
    for stem, c, cx, cy, w, h, note in ADDS:
        p = LBL / f"{stem}.txt"
        with open(p, "a", encoding="utf-8") as f:
            f.write(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    # 全部写后校验
    for stem in TOUCHED:
        validate(LBL / f"{stem}.txt")
    print(f"删{n_del}框/修{len(SETS)}框/补{len(ADDS)}框 完成, 格式校验通过")

    # 剔图
    (REJ / "images").mkdir(parents=True, exist_ok=True)
    (REJ / "labels").mkdir(parents=True, exist_ok=True)
    for name, note in REJECTS:
        src_i, src_l = IMG / name, LBL / (Path(name).stem + ".txt")
        if not src_i.exists():
            print(f"  [跳过] {name} 不存在")
            continue
        import os
        os.replace(src_i, REJ / "images" / name)
        if src_l.exists():
            os.replace(src_l, REJ / "labels" / src_l.name)
        print(f"  剔 {name}  {note}")
    print("剔图完成 ->", REJ.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
