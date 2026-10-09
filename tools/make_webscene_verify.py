# -*- coding: utf-8 -*-
"""生成网图修框抽查页面 verify.html：每张卡显示修后带框图（grid_fixed/gNN.jpg）。

用法（r1 默认路径）: python tools/make_webscene_verify.py
用法（自定义批次）:  python tools/make_webscene_verify.py --batch r2 --title "stock 批次"
    --batch X  → 使用 outputs/web_lab_scene_review_X/ 与 data/_prelabel/webscene_X/
"""
import argparse
import io
import sys
import urllib.parse
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ap = argparse.ArgumentParser()
ap.add_argument("--batch", default="", help="批次后缀，如 r2 → review_r2 / webscene_r2")
ap.add_argument("--title", default="")
ap.add_argument("--tips", default="")
args = ap.parse_args()

_b = f"_{args.batch}" if args.batch else ""
ROOT = Path(__file__).resolve().parent.parent
REVIEW = ROOT / "outputs" / f"web_lab_scene_review{_b}"
INDEX = REVIEW / "index.txt"
GRID = REVIEW / "grid_fixed"
OUT = REVIEW / "verify.html"
LBLDIR = ROOT / "data" / "_prelabel" / f"webscene{_b}" / "labels"

idx = {}
for line in INDEX.read_text(encoding="utf-8").splitlines():
    if not line or line.startswith("#") or line.startswith("编号"):
        continue
    p = line.split("\t")
    if len(p) >= 2:
        idx[int(p[0])] = p[1]

cards = []
for n in sorted(idx):
    name = idx[n]
    g = GRID / f"g{n:02d}.jpg"
    nbox = "?"
    if g.exists():
        # 从文件名下划线前缀拿不到框数，直接数标签
        lbl = LBLDIR / (Path(name).stem + ".txt")
        if lbl.exists():
            nbox = sum(1 for ln in lbl.read_text(encoding="utf-8").splitlines() if ln.strip())
    url = "file:///" + str(g).replace("\\", "/")
    cards.append(
        f'<div class="card"><div class="no">#{n}</div>'
        f'<a href="{url}" target="_blank"><img src="{url}" loading="lazy" alt="g{n:02d}"></a>'
        f'<div class="fn">{nbox} 框 | {name}</div></div>'
    )

n_total = len(cards)
box_total = 0
for n in sorted(idx):
    name = idx[n]
    lbl = LBLDIR / (Path(name).stem + ".txt")
    if lbl.exists():
        box_total += sum(1 for ln in lbl.read_text(encoding="utf-8").splitlines() if ln.strip())

title = args.title or f"web_lab_scene 修框抽查 · 共 {n_total} 张（{box_total} 框）"
tips = args.tips or (
    "图上为<b>修后预标注框</b>（绿=person 红=mask 橙=gloves 紫=lab_coat 黄=goggles，编号=框序）。"
    "点图放大看细节。<b>回报方式：按编号说</b>，如「1 中间女生B大褂框多了」「15 手套框偏了」「33 OK」。")

html = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>""" + title + """</title>
<style>
 body{font-family:-apple-system,"Microsoft YaHei",sans-serif;background:#1b1d21;color:#e6e6e6;margin:0;padding:16px}
 h1{font-size:18px;margin:0 0 8px}
 .tips{color:#9aa0a6;font-size:13px;margin-bottom:14px;line-height:1.7}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}
 .card{background:#26292e;border-radius:8px;padding:8px;position:relative}
 .no{position:absolute;top:6px;left:6px;background:#0a84ff;color:#fff;border-radius:5px;
      padding:1px 7px;font-size:13px;font-weight:600;z-index:2}
 img{width:100%;height:240px;object-fit:contain;background:#111;border-radius:5px;display:block}
 .fn{font-size:11px;color:#8b9099;margin-top:5px;word-break:break-all;line-height:1.35}
</style></head><body>
<h1>""" + title + """</h1>
<div class="tips">""" + tips + """</div>
<div class="grid">""" + "".join(cards) + """</div>
</body></html>"""

OUT.write_text(html, encoding="utf-8")
print(f"完成 -> {OUT}（{len(cards)} 张卡片）")
