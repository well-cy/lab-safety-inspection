#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""stage_webscene_r3.py —— 把 images/ 下新到的一批网图收进 images_r3/ 暂存区（2026-10-09）

来源（用户 10-09 放入 data/raw/web_lab_scene/images/）：
  A) 17 张 unsplash 直接下载的 jpg（national-cancer-institute-* 等）
  B) 23 张 png 屏幕截图（images/Screenshots/）
  共 40 张。r1 的 48 张原图在 images/ 根目录，**不能混进来**——
  用 r1 manifest 的 original_name + 文件名规则双重排除。

动作：只读源 → 统一转 JPG（PNG 转档，JPEG 原样字节复制）→ 写 images_r3/
      ASCII 命名：A 组 r3u_<NN>_<slug8>.jpg   B 组 r3s_<NN>_<hhmmss>.jpg
      写 stage_manifest.csv 溯源（new_name, group, original_name, w, h）。

用法: python tools/stage_webscene_r3.py            # 干跑
      python tools/stage_webscene_r3.py --apply    # 落盘
"""
from __future__ import annotations

import argparse
import csv
import io

import re
import sys
from pathlib import Path

from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "data" / "raw" / "web_lab_scene" / "images"
SHOTS = IMG / "Screenshots"
DST = ROOT / "data" / "raw" / "web_lab_scene" / "images_r3"
MANIFEST = ROOT / "data" / "raw" / "web_lab_scene" / "stage_manifest_r3.csv"
R1_MANIFEST = ROOT / "data" / "raw" / "web_lab_scene" / "import_manifest.csv"


def r1_names() -> set[str]:
    out = set()
    if R1_MANIFEST.exists():
        with R1_MANIFEST.open(encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                out.add(row["original_name"])
    return out


def slug_of(name: str) -> str:
    """national-cancer-institute-Ccf83bIVY3Q-unsplash (1).jpg -> ccf83bivy3q"""
    m = re.search(r"-([A-Za-z0-9_\-]{8,})-unsplash", name)
    if m:
        return m.group(1)[:8].lower()
    return re.sub(r"[^A-Za-z0-9]", "", Path(name).stem)[:8].lower() or "x"


def hhmmss(name: str) -> str:
    m = re.search(r"(\d{6})\.png$", name)
    return m.group(1) if m else re.sub(r"[^0-9]", "", Path(name).stem)[-6:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    already = r1_names()
    items = []

    # A) unsplash jpg：images/ 根目录下的英文名文件（排除 r1 的中文原名）
    for p in sorted(IMG.iterdir()):
        if not p.is_file() or p.suffix.lower() not in (".jpg", ".jpeg"):
            continue
        if p.name in already:
            continue
        items.append(("A", p))
    # B) 截图：Screenshots/*.png
    for p in sorted(SHOTS.glob("*.png")):
        items.append(("B", p))

    print(f"发现 {len(items)} 张待收（A 组 {sum(1 for g,_ in items if g=='A')} / B 组 {sum(1 for g,_ in items if g=='B')}）")
    if not items:
        print("没有新文件。")
        return 0

    rows = []
    na = nb = 0
    for grp, p in items:
        if grp == "A":
            na += 1
            new = f"r3u_{na:02d}_{slug_of(p.name)}.jpg"
        else:
            nb += 1
            new = f"r3s_{nb:02d}_{hhmmss(p.name)}.jpg"
        try:
            im = Image.open(p)
            w, h = im.size
            fmt = im.format
        except Exception as e:  # noqa: BLE001
            print(f"  !! 打不开 {p.name}: {e}")
            continue
        rows.append({"new_name": new, "group": grp, "original_name": p.name, "w": w, "h": h, "fmt": fmt, "_src": p})
        flag = "转JPG" if grp == "B" else "原样"
        print(f"  [{grp}] {p.name[:52]:<52} -> {new}  {w}x{h} ({flag})")

    print(f"\n合计 {len(rows)} 张。")
    if not args.apply:
        print("\n[干跑] 未写盘。确认后加 --apply")
        return 0

    DST.mkdir(parents=True, exist_ok=True)
    for r in rows:
        out = DST / r["new_name"]
        if r["group"] == "B":
            Image.open(r["_src"]).convert("RGB").save(out, "JPEG", quality=92)
        else:
            out.write_bytes(r["_src"].read_bytes())  # 复制（保留源文件，便于复查）
    with MANIFEST.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["new_name", "group", "original_name", "w", "h", "fmt"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in ("new_name", "group", "original_name", "w", "h", "fmt")})
    print(f"\n[落盘] {len(rows)} 张 -> {DST}")
    print(f"[落盘] 溯源表 -> {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
