# -*- coding: utf-8 -*-
"""生成网图修框抽查页面 verify.html：每张卡显示修后带框图（grid_fixed/gNN.jpg）。

用法: python tools/make_webscene_verify.py
"""
import io
import sys
import urllib.parse
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "outputs" / "web_lab_scene_review" / "index.txt"
GRID = ROOT / "outputs" / "web_lab_scene_review" / "grid_fixed"
OUT = ROOT / "outputs" / "web_lab_scene_review" / "verify.html"

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
        lbl = ROOT / "data" / "_prelabel" / "webscene_r1" / "labels" / (Path(name).stem + ".txt")
        if lbl.exists():
            nbox = sum(1 for ln in lbl.read_text(encoding="utf-8").splitlines() if ln.strip())
    url = "file:///" + str(g).replace("\\", "/")
    cards.append(
        f'<div class="card"><div class="no">#{n}</div>'
        f'<a href="{url}" target="_blank"><img src="{url}" loading="lazy" alt="g{n:02d}"></a>'
        f'<div class="fn">{nbox} 框 | {name}</div></div>'
    )

html = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>web_lab_scene 修框抽查（49 张）</title>
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
<h1>web_lab_scene 修框抽查 · 共 49 张（修后 196 框）</h1>
<div class="tips">图上为<b>修后预标注框</b>（绿=person 红=mask 橙=gloves 紫=lab_coat 黄=goggles，编号=框序）。
点图放大看细节。<b>回报方式：按编号说</b>，如「1 中间女生B大褂框多了」「15 手套框偏了」「33 OK」。<br>
本次修正概要：拆一框多人（#1/#3）、删重复框（#2/#10/#13/#23/#24/#37/#47）、删误检（仪器/枪身/纸盒/管架/普通眼镜/推头顶护目镜/裸手）、补漏检（person/大褂/手套/口罩）、扩短框到底部。#14/#35 原标注即正确未动。</div>
<div class="grid">""" + "".join(cards) + """</div>
</body></html>"""

OUT.write_text(html, encoding="utf-8")
print(f"完成 -> {OUT}（{len(cards)} 张卡片）")
