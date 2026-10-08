# -*- coding: utf-8 -*-
"""为 data/raw/web_lab_scene/ 的网图生成「编号查看页」（分诊用，无标签）。

把目录下的图片按文件名排序编号，生成一个可点击放大的 HTML 查看页，
并输出 index.txt 供她按编号回报判定。

用法:
    python tools/make_webscene_viewer.py --dir data/raw/web_lab_scene \
        --out outputs/web_lab_scene_review
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/raw/web_lab_scene")
    ap.add_argument("--out", default="outputs/web_lab_scene_review")
    args = ap.parse_args()

    src = Path(args.dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    imgs = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT)
    vids = sorted(p for p in src.iterdir() if p.suffix.lower() in (".mp4", ".mov"))

    # index.txt
    lines = ["# web_lab_scene 网图收集清单（按文件名排序编号）", "# 编号\t文件名"]
    for i, p in enumerate(imgs, 1):
        lines.append(f"{i}\t{p.name}")
    (out / "index.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    cards = []
    for i, p in enumerate(imgs, 1):
        rel = p.resolve().as_uri()
        name = html.escape(p.name)
        cards.append(
            f'<div class="card"><div class="no">#{i}</div>'
            f'<a href="{rel}" target="_blank"><img src="{rel}" loading="lazy" alt="{name}"></a>'
            f'<div class="fn" title="{name}">{name}</div></div>'
        )
    vidnote = ""
    if vids:
        vidnote = "<p>另有视频：" + "、".join(html.escape(v.name) for v in vids) + "</p>"

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>web_lab_scene 网图分诊（{len(imgs)} 张）</title>
<style>
 body{{font-family:-apple-system,"Microsoft YaHei",sans-serif;background:#1b1d21;color:#e6e6e6;margin:0;padding:16px}}
 h1{{font-size:18px;margin:0 0 8px}}
 .tips{{color:#9aa0a6;font-size:13px;margin-bottom:14px;line-height:1.7}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}}
 .card{{background:#26292e;border-radius:8px;padding:8px;position:relative}}
 .no{{position:absolute;top:6px;left:6px;background:#0a84ff;color:#fff;border-radius:5px;
      padding:1px 7px;font-size:13px;font-weight:600;z-index:2}}
 img{{width:100%;height:200px;object-fit:contain;background:#111;border-radius:5px;display:block}}
 .fn{{font-size:11px;color:#8b9099;margin-top:5px;word-break:break-all;line-height:1.35}}
</style></head><body>
<h1>web_lab_scene 网图分诊 · 共 {len(imgs)} 张</h1>
<div class="tips">点图可放大原图。判定看三点：① 是不是真实验室场景 ② 有没有人穿 PPE（白大褂/口罩/手套/护目镜）③ 有没有水印/logo/贴纸。<br>
回报方式：按编号说，如「3 有水印」「5 不是实验室」「7 好图」。</div>
{vidnote}
<div class="grid">{''.join(cards)}</div>
</body></html>"""
    (out / "view.html").write_text(doc, encoding="utf-8")
    print(f"[完成] {len(imgs)} 张 → {out/'view.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
