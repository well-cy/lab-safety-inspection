"""把 uploads 第 0 批预标注草稿生成「原图查看器」（E6 五类口径）。

- 图：data/test/uploads/<name>.jpg
- 草稿：data/_prelabel/uploads_20261008/labels/<name>.txt（0=person 1=mask 2=gloves 3=lab_coat 4=goggles）
- 编号：沿用 outputs/_prelabel_uploads_20261008/index.txt（她按编号回报）

用法:
    python tools/make_uploads_viewer.py --out outputs/_prelabel_uploads_20261008
"""
from __future__ import annotations

import argparse
import html
import os
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "data" / "test" / "uploads"
LBL_DIR = ROOT / "data" / "_prelabel" / "uploads_20261008" / "labels"
CLS = {0: "person", 1: "mask", 2: "gloves", 3: "lab_coat", 4: "goggles"}


def read_index(path: Path) -> dict:
    m = {}
    if not path.exists():
        return m
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = [p for p in line.split("\t") if p != ""]
        if len(parts) < 2:
            continue
        try:
            m[int(parts[0].strip())] = Path(parts[1].strip()).name
        except ValueError:
            continue
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/_prelabel_uploads_20261008")
    ap.add_argument("--index", default="outputs/_prelabel_uploads_20261008/index.txt")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = read_index(ROOT / args.index)
    rel2no = {v: k for k, v in idx.items()}
    nxt = (max(idx) if idx else 0) + 1

    items = []
    for img in sorted(IMG_DIR.glob("*")):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        lbl = LBL_DIR / (img.stem + ".txt")
        no = rel2no.get(img.name)
        if no is None:
            no = nxt
            nxt += 1
        boxes = []
        if lbl.exists():
            for line in lbl.read_text(encoding="utf-8").splitlines():
                p = line.split()
                if len(p) >= 5:
                    boxes.append((CLS.get(int(float(p[0])), p[0]),
                                  float(p[1]), float(p[2]), float(p[3]), float(p[4])))
        items.append(dict(no=no, img=img, boxes=boxes))
    items.sort(key=lambda d: d["no"])

    cards = []
    for d in items:
        im = cv2.imread(str(d["img"]))
        h_px, w_px = (im.shape[0], im.shape[1]) if im is not None else (1, 1)
        rel = os.path.relpath(d["img"], out_dir).replace("\\", "/")
        cls_set = "|".join(sorted({b[0] for b in d["boxes"]}))
        counts: dict[str, int] = {}
        for b in d["boxes"]:
            counts[b[0]] = counts.get(b[0], 0) + 1
        tally = " ".join(f"{k}{v}" for k, v in counts.items())
        ov = []
        for name, cx, cy, bw, bh in d["boxes"]:
            ov.append(f'<div class="b k-{name}" style="left:{(cx-bw/2)*100:.3f}%;'
                      f'top:{(cy-bh/2)*100:.3f}%;width:{bw*100:.3f}%;height:{bh*100:.3f}%">'
                      f'<i>{html.escape(name)}</i></div>')
        cards.append(
            f'<div class="card" data-cls="{cls_set}">'
            f'<div class="head"><span class="no">#{d["no"]}</span>'
            f'<span class="fn">{html.escape(d["img"].name)}</span>'
            f'<span class="tally">{html.escape(tally) or "无框"}</span></div>'
            f'<div class="wrap"><img loading="lazy" src="{html.escape(rel)}">'
            f'<div class="ov">{"".join(ov)}</div></div>'
            f'<div class="p"><a href="{html.escape(rel)}" target="_blank">新标签打开原图</a>'
            f'<span class="path">{html.escape(str(d["img"]))}</span></div></div>')

    page = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>uploads 第 0 批预标注草稿 · 分诊查看器</title>
<style>
:root{{--bg:#f6f7f9;--card:#fff;--fg:#1d2129;--mut:#86909c;--line:#e5e6eb;}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}}
header{{position:sticky;top:0;z-index:9;background:rgba(255,255,255,.94);
backdrop-filter:blur(6px);border-bottom:1px solid var(--line);padding:10px 16px}}
h1{{font-size:16px;margin:0 0 6px}}
.bar{{display:flex;gap:8px;flex-wrap:wrap;align-items:center}}
.bar button{{border:1px solid var(--line);background:#fff;border-radius:999px;
padding:3px 12px;cursor:pointer;font-size:13px;color:var(--fg)}}
.bar button.on{{background:var(--fg);color:#fff;border-color:var(--fg)}}
.hint{{color:var(--mut);font-size:12px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(430px,1fr));
gap:14px;padding:14px 16px 40px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px}}
.head{{display:flex;gap:8px;align-items:center;margin-bottom:4px;flex-wrap:wrap}}
.no{{font-weight:700;font-size:16px}}
.fn{{font-size:11px;color:var(--mut);word-break:break-all}}
.tally{{font-size:11px;color:var(--mut)}}
.wrap{{position:relative;background:#000;border-radius:6px;overflow:hidden}}
.wrap img{{display:block;width:100%;height:auto}}
.ov{{position:absolute;inset:0}}
.b{{position:absolute;border-radius:2px;border:2px solid}}
.b.k-person{{border-color:#f59f00}}.b.k-mask{{border-color:#7048e8}}
.b.k-gloves{{border-color:#0c8599}}.b.k-lab_coat{{border-color:#2f9e44}}
.b.k-goggles{{border-color:#e8590c}}
.b i{{position:absolute;left:0;top:-15px;font-size:10px;font-style:normal;
padding:0 4px;border-radius:3px;white-space:nowrap;color:#fff}}
.b.k-person i{{background:#f59f00}}.b.k-mask i{{background:#7048e8}}
.b.k-gloves i{{background:#0c8599}}.b.k-lab_coat i{{background:#2f9e44}}
.b.k-goggles i{{background:#e8590c}}
.p{{display:flex;gap:10px;align-items:center;margin-top:6px}}
.p a{{color:#1971c2;text-decoration:none;font-size:12px;white-space:nowrap}}
.p .path{{font-size:10px;color:var(--mut);word-break:break-all}}
body.noperson .b.k-person{{display:none}}
</style></head><body>
<header>
<h1>uploads 第 0 批预标注草稿 · 共 {len(items)} 张（分诊用）</h1>
<div class="bar">
<button data-f="all" class="on">全部</button>
<button data-f="mask">mask</button>
<button data-f="lab_coat">lab_coat</button>
<button data-f="goggles">goggles</button>
<button data-f="gloves">gloves</button>
<button id="tg">隐藏 person 框</button>
<span class="hint">框=模型草稿（E6 五类）；按编号回报：框错说删/漏说补/整图不要说剔</span>
</div></header>
<div class="grid">{"".join(cards)}</div>
<script>
document.querySelectorAll('.bar button[data-f]').forEach(b=>b.onclick=()=>{{
  document.querySelectorAll('.bar button[data-f]').forEach(x=>x.classList.remove('on'));
  b.classList.add('on');
  const f=b.dataset.f;
  document.querySelectorAll('.card').forEach(c=>{{
    c.style.display=(f==='all'||c.dataset.cls.split('|').includes(f))?'':'none';
  }});
}});
document.getElementById('tg').onclick=()=>{{
  document.body.classList.toggle('noperson');
  document.getElementById('tg').classList.toggle('on');
}};
</script></body></html>"""
    (out_dir / "view.html").write_text(page, encoding="utf-8")
    print(f"[完成] {len(items)} 张 → {out_dir / 'view.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
