# -*- coding: utf-8 -*-
"""E1-3 类别转换：dataset1 (13类) -> 4类，输出到 data/lab_ppe/（标准 YOLO 目录）。

转换规则（用户确认版）:
    mask                       -> 0 mask
    glove / Glove / Gloves     -> 1 gloves
    lab coat / Lab coat        -> 2 lab_coat
    Goggles / googles / safety goggles -> 3 goggles
    Coverall / Face_Shield / lab shoe / protective head cap -> 丢弃（该行标注删除，图片保留）

原则:
    - 不修改 data/raw/ 原始数据
    - 生成独立目录 data/lab_ppe/
    - 每个标注行写入 manifest.csv 便于追溯
"""
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "raw" / "dataset1"
DST = ROOT / "data" / "lab_ppe"

# 显式映射表（键为 data.yaml 中的原始类名，需完全匹配；大小写敏感按 Roboflow 原样）
KEEP = {
    "mask": 0, "Mask": 0,
    "glove": 1, "Glove": 1, "Gloves": 1,
    "lab coat": 2, "Lab coat": 2,
    "Goggles": 3, "googles": 3, "safety goggles": 3,
    # 作者误将版本号写进类名的脏类，语义即 safety goggles
    "safety goggles - v1 2023-08-02 12-26pm": 3,
}
DROP = {"Coverall", "Face_Shield", "lab shoe", "protective head cap"}

SPLITS = ["train", "valid", "test"]


def main() -> int:
    src_yaml = SRC / "data.yaml"
    if not src_yaml.exists():
        print(f"[error] 找不到 {src_yaml}，请先下载数据集")
        return 1

    meta = yaml.safe_load(src_yaml.read_text(encoding="utf-8"))
    names = meta.get("names")
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    print("dataset1 原始类别:")
    for i, n in sorted(names.items()):
        tag = "KEEP" if n in KEEP else ("DROP" if n in DROP else "??未知??")
        print(f"  {i:>2}  {n!r:25} -> {tag}")

    unmapped = [n for n in names.values() if n not in KEEP and n not in DROP]
    if unmapped:
        print(f"\n[error] 存在未映射类别，需人工确认后更新脚本映射表: {unmapped}")
        return 1

    # 建目录
    for sp in SPLITS:
        (DST / "images" / sp).mkdir(parents=True, exist_ok=True)
        (DST / "labels" / sp).mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    stats = Counter()
    used_names = {}  # 新文件名 -> 原路径, 防跨split重名

    for sp in SPLITS:
        img_dir = SRC / sp / "images"
        lbl_dir = SRC / sp / "labels"
        if not img_dir.exists():
            print(f"[warn] {sp} 无 images 目录，跳过")
            continue
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            stem = img_path.stem
            lbl_path = lbl_dir / (stem + ".txt")

            # 新文件名（防跨 split 冲突）
            new_stem = stem
            if new_stem in used_names:
                new_stem = f"{sp}_{stem}"
            used_names[new_stem] = img_path

            # 解析标注
            lines_out = []
            if lbl_path.exists():
                for line_no, line in enumerate(
                    lbl_path.read_text(encoding="utf-8").strip().splitlines(), 1
                ):
                    parts = line.split()
                    if len(parts) == 5:
                        # 标准 bbox: cls cx cy w h
                        vals = [float(x) for x in parts[1:]]
                        cls_raw = parts[0]
                    elif len(parts) > 5 and len(parts) % 2 == 1:
                        # 多边形分割标注 (YOLO-seg): cls x1 y1 x2 y2 ...
                        # 转为外接矩形 bbox: min/max 顶点
                        pts = [float(x) for x in parts[1:]]
                        xs, ys = pts[0::2], pts[1::2]
                        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
                        cls_raw = parts[0]
                        vals = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
                        stats["poly_to_bbox"] += 1
                    else:
                        manifest_rows.append([
                            "dataset1", sp, img_path.name, new_stem + img_path.suffix,
                            "", f"(第{line_no}行格式异常)", "", "", "drop_malformed",
                        ])
                        stats["malformed_lines"] += 1
                        continue
                    cid = int(cls_raw)
                    cname = names.get(cid, f"<未知id:{cid}>")
                    if cname in KEEP:
                        new_id = KEEP[cname]
                        lines_out.append(f"{new_id} " + " ".join(f"{v:.6f}" for v in vals))
                        manifest_rows.append([
                            "dataset1", sp, img_path.name, new_stem + img_path.suffix,
                            cid, cname, new_id,
                            {0: "mask", 1: "gloves", 2: "lab_coat", 3: "goggles"}[new_id],
                            "keep",
                        ])
                        stats[f"keep_{ {0:'mask',1:'gloves',2:'lab_coat',3:'goggles'}[new_id] }"] += 1
                    else:
                        manifest_rows.append([
                            "dataset1", sp, img_path.name, new_stem + img_path.suffix,
                            cid, cname, "", "", "drop_class",
                        ])
                        stats[f"drop_{cname}"] += 1

            # 复制图片与标签
            shutil.copy2(img_path, DST / "images" / sp / (new_stem + img_path.suffix))
            (DST / "labels" / sp / (new_stem + ".txt")).write_text(
                "\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8"
            )
            stats[f"img_{sp}"] += 1
            if not lines_out:
                stats[f"empty_label_{sp}"] += 1

    # data.yaml
    data_yaml = {
        "path": str(DST).replace("\\", "/"),
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test",
        "nc": 4,
        "names": {0: "mask", 1: "gloves", 2: "lab_coat", 3: "goggles"},
    }
    (DST / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    # manifest.csv
    with open(DST / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["source_dataset", "split", "original_filename", "new_filename",
                    "original_class_id", "original_class_name",
                    "new_class_id", "new_class_name", "action"])
        w.writerows(manifest_rows)

    print("\n===== 转换统计 =====")
    for k in sorted(stats):
        print(f"  {k:28} {stats[k]}")
    print(f"\nmanifest 行数: {len(manifest_rows)}")
    print(f"输出目录: {DST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
