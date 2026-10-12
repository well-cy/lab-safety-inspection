"""把漏标候选（candidates.csv）生成「原图查看器」：
  - index_full.csv : 编号 → 原图绝对路径 / 标签绝对路径 / 候选类别 / conf / 框
  - view.html      : 浏览器直接打开，原图全分辨率 + 候选红框 + 已有标签灰框，可筛选

用法:
    python tools/make_candidate_viewer.py \
        --cand outputs/missing_labels_20261008/candidates.csv \
        --out  outputs/missing_labels_20261008 \
        --index outputs/missing_labels_20261008/index.txt \
        --exclude-substr 4267250
"""
from __future__ import annotations

import argparse
import csv
import html
import os
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "lab_ppe"
CLS = {0: "mask", 1: "gloves", 2: "lab_coat", 3: "goggles"}  # lab_ppe 原生类别


def read_index(path: Path) -> dict:
    """index.txt → {编号: (split, file)}"""
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
            m[int(parts[0].strip())] = parts[1].strip()
        except ValueError:
            continue
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--index", default="")
    ap.add_argument("--exclude-substr", default="", help="已定口径、无需复核的文件名子串")
    ap.add_argument("--title", default="漏标候选原图查看器")
    args = ap.parse_args()

    cand_path = Path(args.cand)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(cand_path, encoding="utf-8")))
    if args.exclude_substr:
        rows = [r for r in rows if args.exclude_substr not in r["file"]]

    # 按图聚合，conf 降序（与图版排序一致）
    by_img: dict[tuple, list] = {}
    for r in rows:
        by_img.setdefault((r["split"], r["file"]), []).append(r)
    ranked = sorted(by_img.items(),
                    key=lambda kv: max(float(x["conf"]) for x in kv[1]), reverse=True)

    # 图版编号优先；图版未覆盖的（图版只取前 120）续编
    seen_no = read_index(Path(args.index)) if args.index else {}
    rel2no = {v.replace("\\", "/"): k for k, v in seen_no.items()}
    next_no = (max(seen_no) if seen_no else 0) + 1

    items = []
    for (sp, fn), cands in ranked:
        img_abs = DATASET / "images" / sp / fn
        lbl_abs = DATASET / "labels" / sp / (Path(fn).stem + ".txt")
        no = rel2no.get(f"{sp}/{fn}")
        in_sheet = no is not None
        if no is None:
            no = next_no
            next_no += 1
        exists = img_abs.exists()

        boxes = []
        if exists and lbl_abs.exists():
            h = w = None
            for line in lbl_abs.read_text(encoding="utf-8").splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                c = int(float(p[0]))
                boxes.append((CLS.get(c, str(c)), float(p[1]), float(p[2]),
                              float(p[3]), float(p[4])))

        items.append(dict(no=no, in_sheet=in_sheet, split=sp, file=fn,
                          img=img_abs, lbl=lbl_abs, exists=exists,
                          cands=cands, boxes=boxes))

    items.sort(key=lambda d: d["no"])

    # ---------- index_full.csv ----------
    csv_path = out_dir / "index_full.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["编号", "在图版内", "split", "文件名", "原图绝对路径", "标签绝对路径",
                    "原图是否存在", "候选类别", "候选conf", "候选框(x1,y1,x2,y2)"])
        for d in items:
            cl = ", ".join(c["cid_name"] for c in d["cands"])
            cf = ", ".join(c["conf"] for c in d["cands"])
            bx = "; ".join(
                "%s:(%s,%s,%s,%s)" % (c["cid_name"], c["x1"], c["y1"], c["x2"], c["y2"])
                for c in d["cands"])
            w.writerow([d["no"], "是" if d["in_sheet"] else "否", d["split"], d["file"],
                        str(d["img"]), str(d["lbl"]),
                        "是" if d["exists"] else "否（已剔除）", cl, cf, bx])

    # ---------- view.html ----------
    cards = []
    for d in items:
        w_px = h_px = None
        if d["exists"]:
            im = cv2.imread(str(d["img"]))
            if im is not None:
                h_px, w_px = im.shape[:2]
        rel = os.path.relpath(d["img"], out_dir).replace("\\", "/")
        tag = "图版内" if d["in_sheet"] else "图版外"
        cls_set = "|".join(sorted({c["cid_name"] for c in d["cands"]}))
        confs = [float(c["conf"]) for c in d["cands"]]
        head = (f'<span class="no">#{d["no"]}</span>'
                f'<span class="tag">{tag}</span>'
                f'<span class="cls c-{cls_set.split("|")[0]}">{html.escape(cls_set)}</span>'
                f'<span class="conf">conf {min(confs):.2f}'
                + (f'–{max(confs):.2f}' if len(confs) > 1 else "") + '</span>')

        if not d["exists"]:
            cards.append(
                f'<div class="card dead" data-cls="{cls_set}">{head}'
                f'<div class="fn">{html.escape(d["file"])}</div>'
                f'<div class="gone">原图已不在数据集内（已剔除/移出）</div></div>')
            continue

        ov = []
        for name, x, y, ww, hh in d["boxes"]:
            ov.append(f'<div class="b gray" data-k="{name}" style="left:{(x-ww/2)*100:.3f}%;'
                      f'top:{(y-hh/2)*100:.3f}%;width:{ww*100:.3f}%;height:{hh*100:.3f}%">'
                      f'<i>{html.escape(name)}</i></div>')
        for c in d["cands"]:
            ov.append(f'<div class="b red" data-k="{c["cid_name"]}" style="'
                      f'left:{float(c["x1"])/w_px*100:.3f}%;top:{float(c["y1"])/h_px*100:.3f}%;'
                      f'width:{(float(c["x2"])-float(c["x1"]))/w_px*100:.3f}%;'
                      f'height:{(float(c["y2"])-float(c["y1"]))/h_px*100:.3f}%">'
                      f'<i>候选 {html.escape(c["cid_name"])} {float(c["conf"]):.2f}</i></div>')

        cards.append(
            f'<div class="card" data-cls="{cls_set}">{head}'
            f'<div class="fn" title="{html.escape(str(d["img"]))}">{html.escape(d["file"])}</div>'
            f'<div class="wrap"><img loading="lazy" src="{html.escape(rel)}">'
            f'<div class="ov">{"".join(ov)}</div></div>'
            f'<div class="p"><a href="{html.escape(rel)}" target="_blank">新标签打开原图</a>'
            f'<span class="path">{html.escape(str(d["img"]))}</span></div></div>')

    n_sheet = sum(1 for d in items if d["in_sheet"])
    page = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>{html.escape(args.title)}</title>
<style>
:root{{--bg:#f6f7f9;--card:#fff;--fg:#1d2129;--mut:#86909c;--line:#e5e6eb;
--red:#e03131;--gray:#9aa4b2;}}
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
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:10px;overflow:hidden}}
.card.dead{{opacity:.55}}
.head{{display:flex;gap:8px;align-items:center;margin-bottom:4px;flex-wrap:wrap}}
.no{{font-weight:700;font-size:16px}}
.tag{{font-size:11px;color:var(--mut);border:1px solid var(--line);border-radius:4px;padding:0 5px}}
.cls{{font-size:11px;padding:1px 7px;border-radius:999px;color:#fff}}
.c-mask{{background:#7048e8}}.c-gloves{{background:#0c8599}}
.c-lab_coat{{background:#2f9e44}}.c-goggles{{background:#e8590c}}
.conf{{font-size:11px;color:var(--mut)}}
.fn{{font-size:11px;color:var(--mut);word-break:break-all;margin-bottom:6px}}
.wrap{{position:relative;background:#000;border-radius:6px;overflow:hidden}}
.wrap img{{display:block;width:100%;height:auto}}
.ov{{position:absolute;inset:0}}
.b{{position:absolute;border-radius:2px}}
.b.gray{{border:1px solid rgba(154,164,178,.75)}}
.b.red{{border:2px solid var(--red);box-shadow:0 0 0 1px rgba(255,255,255,.6) inset}}
.b i{{position:absolute;left:0;top:-15px;font-size:10px;font-style:normal;
padding:0 4px;border-radius:3px;white-space:nowrap;color:#fff}}
.b.gray i{{background:rgba(134,144,156,.85);display:none}}
.b.red i{{background:var(--red)}}
.p{{display:flex;gap:10px;align-items:center;margin-top:6px}}
.p a{{color:#1971c2;text-decoration:none;font-size:12px;white-space:nowrap}}
.p .path{{font-size:10px;color:var(--mut);word-break:break-all}}
.gone{{color:var(--red);font-size:12px;padding:14px 0}}
body.hidegray .b.gray{{display:none}}
</style></head><body>
<header>
<h1>{html.escape(args.title)} · 共 {len(items)} 张
（图版内 {n_sheet} / 图版外续编 {len(items)-n_sheet}）</h1>
<div class="bar">
<button data-f="all" class="on">全部</button>
<button data-f="mask">mask</button>
<button data-f="lab_coat">lab_coat</button>
<button data-f="goggles">goggles</button>
<button data-f="gloves">gloves</button>
<button id="tg">隐藏灰框</button>
<span class="hint">红框=候选漏标（待判定）; 灰框=已有标签; 编号与图版一致，"图版外"为 conf 较低的补充候选</span>
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
  document.body.classList.toggle('hidegray');
  document.getElementById('tg').classList.toggle('on');
}};
</script></body></html>"""
    (out_dir / "view.html").write_text(page, encoding="utf-8")

    print(f"[完成] {len(items)} 张（图版内 {n_sheet}，图版外 {len(items)-n_sheet}）")
    print(f"  清单 → {csv_path}")
    print(f"  查看器 → {out_dir / 'view.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
