# -*- coding: utf-8 -*-
"""A 档 369 张预标注审核编辑器生成器（适配包结构，复用 make_box_editor 的 HTML 模板）。

批次拆分：
  a369_u8   8 张「模型未检出 person」——候选含当前管线 person 输出；
            其中 2 张 UV 紫光盲区（连 0.05 都无输出）用 lab_coat 原框 1.15 倍外扩作 person 初始候选
  a369_c35  35 张 CONFLICT——标签用 step2 清理后的 labels_merged_v2（原标实线 + 清理后保留候选）
  a369_m1/a369_m2  主批 NEEDS_REVIEW（322 张）——原标实线 + preannotated 候选

工作区标签输出到 <out>/ws_<批>/labels/（不写 data/，落库时用 --accept-from-ws 指到此处）。
候选去重口径与 make_person_review 相同：同类 IoU≥0.85 / 任意类 IoU≥0.95 / 候选间 IoU≥0.85 → 丢弃。
"""
import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from make_box_editor import HTML  # noqa: E402

CLSMAP = {0: "person", 1: "mask", 2: "gloves", 3: "lab_coat", 4: "goggles", 5: "coverall"}


def iou_cxcywh(a, b) -> float:
    _, ax, ay, aw, ah = a
    _, bx, by, bw, bh = b
    ax1, ay1, ax2, ay2 = ax - aw / 2, ay - ah / 2, ax + aw / 2, ay + ah / 2
    bx1, by1, bx2, by2 = bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = aw * ah + bw * bh - inter
    return inter / ua if ua > 0 else 0.0


def fam369(name: str) -> str:
    """251020 连续视频帧族：按 来源+日期+小时 聚族；SafetyLab 按短名聚族。"""
    if name.startswith("251020__"):
        m = name.split("__", 1)[1]
        return "251020_" + m[:13]          # 到小时（-2024-05-20-14…）
    return "safety_" + name.replace("SafetyLab__", "").split("_jpeg")[0].split("_png")[0][:8]


def imsize(p: Path):
    im = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    return im.shape[1], im.shape[0]


def read_labels(p: Path):
    if not p.exists():
        return []
    return [[int(float(l[0]))] + [float(v) for v in l[1:5]]
            for l in (x.split() for x in p.read_text(encoding="utf-8").splitlines()) if len(l) >= 5]


def dedup(merged, cands):
    kept, n = [], 0
    for c in cands:
        if any((c[0] == m[0] and iou_cxcywh(c, m) >= 0.85) or iou_cxcywh(c, m) >= 0.95 for m in merged) \
           or any(iou_cxcywh(c, k) >= 0.85 for k in kept):
            n += 1
            continue
        kept.append(c)
    return kept, n


def xywh(cls, cx, cy, w, h, cand):
    return [cls, round(min(max(cx - w / 2, 0.0), 1.0), 6), round(min(max(cy - h / 2, 0.0), 1.0), 6),
            round(w, 6), round(h, 6), cand]


def build_batch(tag, items, src, out, ws_root, title, max_side=1600):
    out_dir = out / tag
    ws = ws_root / f"ws_{tag}" / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    ws.mkdir(parents=True, exist_ok=True)
    imgs, index = [], []
    items = sorted(items, key=lambda t: (fam369(t[0]), t[0]))
    for i, (fname, merged, cands) in enumerate(items):
        ip = src / "images" / fname
        w, h = imsize(ip)
        merged = [[m[0]] + [round(v, 6) for v in m[1:5]] for m in merged]
        cands, n_drop = dedup(merged, [[c[0]] + [round(v, 6) for v in c[1:5]] for c in cands])
        (ws / (ip.stem + ".txt")).write_text(
            "\n".join(" ".join(str(v) for v in r) for r in merged + cands)
            + ("\n" if merged or cands else ""), encoding="utf-8", newline="\n")
        edit = [xywh(*b, 0) for b in merged] + [xywh(*b, 1) for b in cands]
        imgs.append({"no": i + 1, "name": fname,
                     "path": "file:///" + str(ip.resolve()).replace("\\", "/"),
                     "nw": w, "nh": h, "boxes": edit})
        index.append(f"{i + 1}\t{ip.name}\t{fam369(fname)}\t{len(merged)}\t{len(cands)}\t0")
    html = (HTML.replace("__TITLE__", f"{title} · {len(imgs)} 张")
            .replace("__DATA__", json.dumps({"imgs": imgs, "focus": 0}, ensure_ascii=False)))
    (out_dir / "editor.html").write_text(html, encoding="utf-8")
    header = ["# A档369 人工审核 · " + title,
              "# 绿/实线=原数据集标注（默认保留） · 黄虚线「候选」=模型预标注（需确认）",
              "# 操作：删错框(×)、拖动/缩放、空白拖拽新建、X 键整图剔除；导出贴回 AI；整批没问题直接回「OK」",
              "# 编号\t文件\t族\t原标框\t候选框\t待确认"]
    (out_dir / "index.txt").write_text("\n".join(header + index) + "\n", encoding="utf-8")
    print(f"  {tag}: {len(imgs)} 张 → {out_dir / 'editor.html'}（丢弃重复候选 {sum(1 for _ in ())} 由日志计）")
    return out_dir, ws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="outputs/_A369/A档_6类预标注包")
    ap.add_argument("--merged-v2", default="outputs/_A369/step2/labels_merged_v2")
    ap.add_argument("--out", default="outputs/_A369/editor")
    args = ap.parse_args()

    src = ROOT / args.src
    v2 = ROOT / args.merged_v2
    out = ROOT / args.out
    ws_root = out / "ws"
    csvp = src / "manifests" / "A档369张_预标注审核清单.csv"
    rows = list(csv.DictReader(open(csvp, encoding="utf-8-sig")))
    print(f"A档 {len(rows)} 张，状态:", {s: sum(1 for r in rows if r['status'] == s)
                                       for s in ('READY', 'NEEDS_REVIEW', 'CONFLICT')})

    # 当前管线 person 输出（8 张无 person 图专用）
    miss_rows = [r for r in rows if r["missing_person"] == "是"]
    miss_set = {r["filename"] for r in miss_rows}
    pipeline_person = {}
    if miss_rows:
        import os
        os.environ.setdefault("NO_PROXY", "*")
        sys.path.insert(0, str(ROOT))
        from ai.detector import PPEDetector
        from backend.config import config
        det = PPEDetector(weights="ai/model/labppe_v8s.pt", conf_threshold=config.CONF_THRESHOLD,
                          conf_person=config.CONF_PERSON, conf_goggles=config.CONF_GOGGLES, imgsz=1280)
        for r in miss_rows:
            ip = src / "images" / r["filename"]
            im = cv2.imdecode(np.fromfile(str(ip), dtype=np.uint8), cv2.IMREAD_COLOR)
            H, W = im.shape[:2]
            d = det.detect(im, conf=config.CONF_THRESHOLD, tile_recheck=True)
            ps = [[x.conf, x.x1 / W, x.y1 / H, (x.x2 - x.x1) / W, (x.y2 - x.y1) / H]
                  for x in d if x.cls_name == "person"]
            pipeline_person[r["filename"]] = ps
            print(f"  管线 person[{r['filename'][:40]}]: {[(round(c,3)) for c,*_ in ps]}")
    # UV 盲区（管线 person=0）→ lab_coat 外扩 1.15 作 person 候选（cxcywh）
    uv_person = {}
    for r in miss_rows:
        if pipeline_person.get(r["filename"]):
            continue
        stem = r["filename"].rsplit(".", 1)[0]
        orig = read_labels(src / "labels_original" / (stem + ".txt"))
        uv_person[r["filename"]] = [[cx, cy, w * 1.15, min(h * 1.15, 1.0)]
                                    for c, cx, cy, w, h in orig if c == 3]

    # 组装各批
    u_items, c_items, m_items = [], [], []
    for r in rows:
        fn = r["filename"]
        stem = fn.rsplit(".", 1)[0]
        orig = read_labels(src / "labels_original" / (stem + ".txt"))
        pre = read_labels(src / "labels_preannotated" / (stem + ".txt"))
        # 缺 person 的 8 张（READY 4 + CONFLICT 4）全部进 U 批，补 person 候选
        if fn in miss_set:
            if r["status"] == "CONFLICT":
                mv2 = read_labels(v2 / (stem + ".txt"))
                v2set = {tuple(x) for x in mv2}
                keep = [c for c in pre if tuple(c) in v2set]
            else:
                keep = list(pre)
            for c, cx, cy, w, h in pipeline_person.get(fn, []):
                keep.append([0, cx, cy, w, h])
            for cx, cy, w, h in uv_person.get(fn, []):
                keep.append([0, cx, cy, w, h])
            u_items.append((fn, orig, keep))
        elif r["status"] == "CONFLICT":
            mv2 = read_labels(v2 / (stem + ".txt"))
            v2set = {tuple(x) for x in mv2}
            keep = [c for c in pre if tuple(c) in v2set]
            c_items.append((fn, orig, keep))
        elif r["status"] == "NEEDS_REVIEW":
            m_items.append((fn, orig, pre))
        # 其余 READY（4 张）直接入库

    print(f"分组: U(无person)={len(u_items)} C(CONFLICT)={len(c_items)} M(主批)={len(m_items)} READY=直接入库")
    half = (len(m_items) + 1) // 2
    build_batch("a369_u8", u_items, src, out, ws_root, "批U·8张无person图（含2张UV紫光盲区）")
    build_batch("a369_c35", c_items, src, out, ws_root, "批C·35张CONFLICT（已自动清理误检，请复核）")
    build_batch("a369_m1", m_items[:half], src, out, ws_root, "批M1·主批前半")
    build_batch("a369_m2", m_items[half:], src, out, ws_root, "批M2·主批后半")
    print("全部编辑器生成完毕；审核顺序建议 U → C → M1 → M2")


if __name__ == "__main__":
    sys.exit(main())
