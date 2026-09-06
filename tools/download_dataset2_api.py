# -*- coding: utf-8 -*-
"""E1-1b 通过 Roboflow REST API 逐张下载 dataset2 (masters-ppe/lab-ppe, 251张)。

原因：该项目从未发布 dataset version（versions=0），SDK 的 version.download() 不可用。
改用 search API 列出图片 + images/{id} API 获取标注与原图 URL。

输出（原生10类YOLO格式，不映射）:
    data/raw/dataset2/
    ├─ train/images, train/labels
    ├─ valid/images, valid/labels
    ├─ test/images,  test/labels
    ├─ data.yaml
    └─ image_list.csv  (id, name, split)
"""
import csv
import time
from pathlib import Path

import requests

API_KEY = "nNQADL3zz9PMVKGnkYc1"
BASE = "https://api.roboflow.com"
WS, PROJ = "masters-ppe", "lab-ppe"

ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / "data" / "raw" / "dataset2"

# 原生类别（保持数据集原样，便于后续独立决策如何映射）
NATIVE_NAMES = ["Mask", "Glasses", "Person", "Gloves", "Lab Coat",
                "No Gloves", "No Glasses", "No Lab Coat", "No Mask", "Shoes"]
NATIVE_ID = {n: i for i, n in enumerate(NATIVE_NAMES)}


def search_images():
    out, offset = [], 0
    while True:
        r = requests.post(
            f"{BASE}/{WS}/{PROJ}/search?api_key={API_KEY}",
            json={"offset": offset, "limit": 250, "in_dataset": True,
                  "fields": ["id", "name", "split"]}, timeout=60)
        r.raise_for_status()
        data = r.json()
        batch = data.get("results", [])
        out += batch
        if len(out) >= data.get("total", 0) or not batch:
            break
        offset += len(batch)
    return out


def main():
    imgs = search_images()
    print(f"共 {len(imgs)} 张图片")
    rows = []
    n_fail = 0
    for i, meta in enumerate(imgs, 1):
        iid, name, split = meta["id"], meta["name"], meta.get("split", "train")
        split = {"train": "train", "valid": "valid", "test": "test"}.get(split, "train")
        img_dir = DST / split / "images"
        lbl_dir = DST / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        img_path = img_dir / name
        lbl_path = lbl_dir / (Path(name).stem + ".txt")
        if img_path.exists() and lbl_path.exists():
            rows.append((iid, name, split))
            continue
        try:
            det = requests.get(
                f"{BASE}/{WS}/{PROJ}/images/{iid}?api_key={API_KEY}", timeout=60)
            det.raise_for_status()
            info = det.json()["image"]
            ann = info.get("annotation") or {}
            W, H = float(ann.get("width") or 0), float(ann.get("height") or 0)
            lines = []
            for b in ann.get("boxes", []):
                cls = NATIVE_ID.get(b["label"])
                if cls is None or not W or not H:
                    continue
                cx = float(b["x"]) / W
                cy = float(b["y"]) / H
                w = float(b["width"]) / W
                h = float(b["height"]) / H
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            # 下载原图
            src = info["urls"]["original"]
            img_data = requests.get(src, timeout=120).content
            img_path.write_bytes(img_data)
            lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            rows.append((iid, name, split))
            if i % 25 == 0:
                print(f"  [{i}/{len(imgs)}] {name} boxes={len(lines)}")
        except Exception as e:
            n_fail += 1
            print(f"  [FAIL {i}] {name}: {e}")
        time.sleep(0.15)
    # data.yaml
    (DST / "data.yaml").write_text(
        f"path: {str(DST).replace(chr(92), '/')}\n"
        f"train: train/images\nval: valid/images\ntest: test/images\n"
        f"nc: {len(NATIVE_NAMES)}\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(NATIVE_NAMES)),
        encoding="utf-8")
    with open(DST / "image_list.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "filename", "split"])
        w.writerows(rows)
    print(f"\n完成: {len(rows)} 张成功, {n_fail} 张失败")
    print(f"输出: {DST}")


if __name__ == "__main__":
    main()
