# -*- coding: utf-8 -*-
"""把 make_box_editor 导出的改动标签落库（person 补标流程的落库端）。

编辑器只导出「改过的图」（buildText 只遍历 dirty），因此导出文本里的每一图
整图覆盖库内同名标签；未出现在导出里的图维持原样。

用法:
  $PY tools/apply_person_review.py --export outputs/person_review_p0_a/_her_export.txt \
      --restore-stems "stem1,stem2,..."        # 这些图恢复库内 class-4（护目镜）框
  $PY tools/apply_person_review.py --export outputs/person_review_p0_b/_her_export.txt \
      --keep-classes 3                         # 库内 class-3(lab_coat) 框被删则以原框保留（防误删）
  $PY tools/apply_person_review.py --export outputs/person_review_p0_b/_her_export.txt \
      --accept-from-ws data/_prelabel/person_p0_b/labels \
      --batch-index outputs/person_review_p0_b/index.txt
                                               # 她回「未改动的图+候选都对」时：
                                               # 批次内未出现在导出中的图，直接采用工作区标签（库内+候选）
可选:
  --split train            落库 split
  --reject-tag p0i_watermark
                           整图剔除的去向目录名，默认 <批名>_reject_<日期>
                           例：data/_rejected/<reject-tag>/
  --no-backup              跳过 tar 备份（默认全量备份 labels/ 到 _label_backup/）
  --dry                    只干跑，打印将写入的内容统计

整图剔除（2026-10-11 新增，与编辑器 X 键配套）:
  编辑器导出里出现
      ### <文件名>
      !REJECT [原因文字]
  即表示「这张图整张剔除」。落库时该图**不写标签**，而是把
  images/<split>/<name>.* 与 labels/<split>/<name>.txt 一并 **移到**
  data/_rejected/<reject-tag>/（只移不删，可随时回滚），并从工作区标签删除。
落库后自动做全库格式校验 + 按类别对账。
"""
from __future__ import annotations

import argparse
import io
import re
import shutil
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "data" / "lab_ppe"
SPLITS = ("train", "valid", "test")
NAMES = ["person", "mask", "gloves", "lab_coat", "goggles"]
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
REJECT_WORDS = ("reject", "剔除", "丢弃", "删图")


def parse_export(txt: str):
    """解析编辑器导出。

    返回 (items, rejected, reasons)：
      items   : {stem: [(cls,cx,cy,w,h), ...]}
      rejected: {stem}          出现 `!REJECT [原因]` 行 → 整图剔除
      reasons : {stem: 原因文字}
    兼容旧格式（无 !REJECT 行时 rejected 为空）。
    """
    items: dict[str, list[tuple]] = {}
    rejected: set[str] = set()
    reasons: dict[str, str] = {}
    cur = None
    for ln in txt.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("###"):
            cur = s[3:].strip()
            items.setdefault(cur, [])
        elif s.startswith("!") and cur is not None:
            parts = s[1:].split(None, 1)
            if parts and parts[0].lower() in REJECT_WORDS:
                rejected.add(cur)
                if len(parts) > 1:
                    reasons[cur] = parts[1].strip()
        elif cur is not None and len(s.split()) == 5:
            p = s.split()
            items[cur].append((int(p[0]), *(round(float(v), 6) for v in p[1:5])))
    return items, rejected, reasons


def read_rows(p: Path) -> list[tuple]:
    if not p.exists():
        return []
    rows = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        q = ln.split()
        if len(q) == 5:
            rows.append((int(q[0]), *(float(v) for v in q[1:5])))
    return rows


def near(a, b, eps=1e-4):
    return a[0] == b[0] and all(abs(a[k] - b[k]) < eps for k in range(1, 5))


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[1] - a[3] / 2, a[2] - a[4] / 2, a[1] + a[3] / 2, a[2] + a[4] / 2
    bx1, by1, bx2, by2 = b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    u = a[3] * a[4] + b[3] * b[4] - inter
    return inter / u if u > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--restore-stems", default="")
    ap.add_argument("--keep-classes", default="",
                    help="库内这些类的框若在导出中消失（IoU<0.5 视为被删）则以原框保留，防误删")
    ap.add_argument("--accept-from-ws", default="",
                    help="工作区标签目录；批次内未出现在导出中的图直接采用其内容（库内+候选）")
    ap.add_argument("--batch-index", default="", help="批次 index.txt，用于确定批次图集")
    ap.add_argument("--done-file", default="data/_person_review_done.txt",
                    help="已完成（已落库）图清单，追加式；空串则跳过")
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--reject-tag", default="",
                    help="整图剔除去向目录名（data/_rejected/<tag>/），默认 <批名>_reject_<日期>")
    args = ap.parse_args()

    exp, rejected, reasons = parse_export(Path(args.export).read_text(encoding="utf-8"))
    exp = {k: v for k, v in exp.items() if k not in rejected}   # 剔除图不写标签
    restore = {s.strip() for s in args.restore_stems.split(",") if s.strip()}
    keep = {int(x) for x in args.keep_classes.split(",") if x.strip()}
    LBL = LAB / "labels" / args.split
    if rejected:
        print(f"整图剔除 {len(rejected)} 张（不写标签，落库时移入 _rejected/）：")
        for s in sorted(rejected):
            print(f"   ✕ {s}" + (f"  [{reasons[s]}]" if reasons.get(s) else ""))

    # ---- 待落库计划：导出图 + 批次内未改动图（采用工作区标签） ----
    plan: dict[str, list[tuple]] = {k: list(v) for k, v in exp.items()}
    accepted: list[str] = []
    if args.accept_from_ws and args.batch_index:
        ws_dir = Path(args.accept_from_ws)
        if not ws_dir.is_absolute():
            ws_dir = ROOT / ws_dir
        stems = []
        for ln in Path(args.batch_index).read_text(encoding="utf-8").splitlines():
            if ln.startswith("#") or not ln.strip():
                continue
            stems.append(Path(ln.split("\t")[1]).stem)
        for s in stems:
            if s in plan or s in rejected:
                continue
            wsf = ws_dir / (s + ".txt")
            if not wsf.exists():
                print(f"!! 工作区缺标签，跳过：{s}")
                continue
            ws_rows = read_rows(wsf)
            lib = read_rows(LBL / (s + ".txt"))
            miss = [r for r in lib if not any(near(r, w) for w in ws_rows)]
            if miss:
                print(f"!! 工作区标签未覆盖库内框（{len(miss)} 个），跳过：{s}")
                continue
            plan[s] = ws_rows
            accepted.append(s)

    # ---- 合成最终标签 ----
    final: dict[str, list[tuple]] = {}
    stats = {"imgs": len(plan), "n_exp": len(exp), "n_acc": len(accepted), "n_rej": len(rejected),
             "restored": 0, "kept": 0, "bad_stem": []}
    for stem, rows in plan.items():
        lib = read_rows(LBL / (stem + ".txt"))
        out = list(rows)
        if stem in restore:
            for r in lib:
                if r[0] == 4 and not any(near(r, s) for s in out):
                    out.append(r)
                    stats["restored"] += 1
        if keep:
            for r in lib:
                if r[0] in keep and not any(
                        s[0] == r[0] and iou(r, s) >= 0.5 for s in out):
                    out.append(r)
                    stats["kept"] += 1
                    stats.setdefault("kept_detail", []).append(
                        (stem, r[0], r[1], r[2], r[3], r[4]))

        # 收尾：裁剪到 [0,1]，去掉零面积框
        clean = []
        for c, cx, cy, w, h in out:
            x1 = min(max(cx - w / 2, 0.0), 1.0)
            y1 = min(max(cy - h / 2, 0.0), 1.0)
            x2 = min(max(cx + w / 2, 0.0), 1.0)
            y2 = min(max(cy + h / 2, 0.0), 1.0)
            if x2 - x1 < 1e-4 or y2 - y1 < 1e-4:
                continue
            clean.append((c, round((x1 + x2) / 2, 6), round((y1 + y2) / 2, 6),
                          round(x2 - x1, 6), round(y2 - y1, 6)))
        final[stem] = clean
        if not lib and not clean:
            stats["bad_stem"].append(stem)

    if args.dry:
        for stem, rows in final.items():
            c = Counter(r[0] for r in rows)
            tags = []
            if stem in restore:
                tags.append("恢复goggles")
            if stem in accepted:
                tags.append("未改动→采纳工作区(库内+候选)")
            print(f"{stem[:44]:46} {len(rows)}框 person{c[0]}"
                  + ("  [" + "/".join(tags) + "]" if tags else ""))
        print(f"\n干跑：导出图 {stats['n_exp']} + 未改动采纳 {stats['n_acc']} = {stats['imgs']} 图"
              f"，恢复 goggles 框 {stats['restored']} 个，防误删保留 {stats['kept']} 个")
        if rejected:
            rtag = args.reject_tag or f"{Path(args.export).parent.name.replace('person_review_', 'p0_')}_reject_{time.strftime('%Y%m%d')}"
            print(f"干跑：整图剔除 {len(rejected)} 张 → data/_rejected/{rtag}/（图+标签移动，不写标签）")
            for s in sorted(rejected):
                found = next((p for p in (LAB / 'images' / args.split).glob(s + '.*')
                              if p.suffix.lower() in IMG_EXT), None)
                print(f"   ✕ {s[:50]:52} "
                      + (f"移 {found.name}" if found else "!! 库内找不到图（跳过）"))
        for st, c, cx, cy, w, h in stats.get("kept_detail", []):
            print(f"  保留 {NAMES[c]:9} {st[:44]:46} cx{cx:.3f} cy{cy:.3f} w{w:.3f} h{h:.3f}")
        return 0

    # ---- 备份 ----
    if not args.no_backup:
        ts = time.strftime("%Y%m%d_%H%M%S")
        bk = LAB / "_label_backup" / f"labels_{args.split}_person_{ts}.tar.gz"
        bk.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(bk, "w:gz") as tf:
            for f in sorted((LAB / "labels" / args.split).glob("*.txt")):
                tf.add(f, arcname=f.name)
        print(f"备份 → {bk.relative_to(ROOT)}（{len(list((LAB / 'labels' / args.split).glob('*.txt')))} 文件）")

    # ---- 整图剔除：图+标签移入 data/_rejected/（只移不删，可回滚） ----
    if rejected:
        rtag = args.reject_tag or \
            f"{Path(args.export).parent.name.replace('person_review_', 'p0_')}_reject_{time.strftime('%Y%m%d')}"
        rdir = ROOT / "data" / "_rejected" / rtag
        rdir.mkdir(parents=True, exist_ok=True)
        ws_dir = Path(args.accept_from_ws) if args.accept_from_ws else None
        if ws_dir and not ws_dir.is_absolute():
            ws_dir = ROOT / ws_dir
        moved, missing = [], []
        for s in sorted(rejected):
            ip = next((p for p in (LAB / "images" / args.split).glob(s + ".*")
                       if p.suffix.lower() in IMG_EXT), None)
            if ip is None:
                missing.append(s)
                continue
            shutil.move(str(ip), str(rdir / ip.name))
            lf = LBL / (s + ".txt")
            if lf.exists():
                shutil.move(str(lf), str(rdir / lf.name))
            if ws_dir is not None and (ws_dir / (s + ".txt")).exists():
                (ws_dir / (s + ".txt")).unlink()
            moved.append((s, reasons.get(s, "")))
        print(f"整图剔除 {len(moved)} 张 → data/_rejected/{rtag}/")
        for s, rs in moved:
            print(f"   ✕ {s[:50]:52}" + (f" [{rs}]" if rs else ""))
        if missing:
            print(f"!! 库内找不到以下图（已跳过）：{missing}")
        stats["n_rej"] = len(moved)

    # ---- 落库 ----
    for stem, rows in final.items():
        txt = "\n".join(" ".join(str(v) for v in r) for r in rows) + ("\n" if rows else "")
        (LBL / (stem + ".txt")).write_text(txt, encoding="utf-8", newline="\n")
    print(f"落库 {len(final)} 图（导出 {stats['n_exp']} + 未改动采纳 {stats['n_acc']}；"
          f"恢复 goggles 框 {stats['restored']} 个；防误删保留框 {stats['kept']} 个）"
          + (f"；整图剔除 {stats['n_rej']} 张" if rejected else ""))

    # ---- 已完成清单（防止下一批重复推送已审图） ----
    if args.done_file:
        df = Path(args.done_file)
        if not df.is_absolute():
            df = ROOT / df
        old = set()
        if df.exists():
            old = {l.strip() for l in df.read_text(encoding="utf-8").splitlines() if l.strip()}
        new = old | set(final) | rejected
        df.parent.mkdir(parents=True, exist_ok=True)
        df.write_text("\n".join(sorted(new)) + "\n", encoding="utf-8", newline="\n")
        print(f"已完成清单 → {df.relative_to(ROOT)}（{len(old)} → {len(new)} 图，含剔除图）")

    # ---- 全库校验 + 对账 ----
    bad = orphan_img = orphan_lab = 0
    tot = Counter()
    for sp in SPLITS:
        c = Counter()
        n = 0
        for f in (LAB / "labels" / sp).glob("*.txt"):
            n += 1
            for ln in f.read_text(encoding="utf-8").splitlines():
                p = ln.split()
                if len(p) != 5:
                    bad += 1
                    continue
                try:
                    k = int(p[0])
                    vals = [float(v) for v in p[1:5]]
                except ValueError:
                    bad += 1
                    continue
                if not (0 <= k <= 4) or any(v < -1e-4 or v > 1.0001 for v in vals):
                    bad += 1
                    continue
                c[k] += 1
        imgs = {p.stem for p in (LAB / "images" / sp).iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
        labs = {f.stem for f in (LAB / "labels" / sp).glob("*.txt")}
        orphan_img += len(imgs - labs)
        orphan_lab += len(labs - imgs)
        tot.update(c)
        print(f"{sp}: {n} 图 / {sum(c.values())} 框  " +
              " / ".join(f"{NAMES[k]}{c[k]}" for k in range(5)))
    print(f"全库: {sum(tot.values())} 框  " + " / ".join(f"{NAMES[k]}{tot[k]}" for k in range(5)),
          f" 坏行{bad} 孤儿图{orphan_img} 孤儿标签{orphan_lab}")
    return 0 if bad == 0 and orphan_img == 0 and orphan_lab == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
