"""空标签图审计 —— 找出「标注为空但模型高置信检出 PPE」的疑似漏标毒样本。

用法:
    python tools/audit_empty_labels.py --splits train,valid --conf 0.5

产出:
    outputs/_empty_audit/empty_label_audit.tsv      逐图检出明细
    data/_empty_audit_suspects_<split>.txt          疑似毒样本清单（可直接喂 make_contact_sheet.py）
    data/_empty_audit_clean_<split>.txt             模型也认为无 PPE 的（真负样本，保留）

口径说明:
    空标签图 = 标签 txt 为空/不存在目标框。它们等价于「整图负样本」，在图里
    确实有 PPE 时就是毒样本（教模型"这里什么都没有"）。本工具用当前权重
    自动分诊，人工只需复核 suspect 名单。
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def imread(p: pathlib.Path):
    """cv2.imread 不认中文路径，用 imdecode 兜底。"""
    buf = np.fromfile(str(p), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def find_image(split: str, stem: str) -> pathlib.Path | None:
    d = ROOT / "data" / "lab_ppe" / "images" / split
    for ext in IMG_EXT:
        p = d / (stem + ext)
        if p.exists():
            return p
    return None


def family(stem: str) -> str:
    if stem.startswith("ppes_kp"):
        return "ppes_kp*(kp机位)"
    if stem.startswith("ppes_"):
        return "ppes_其他(packing/img)"
    import re
    if re.match(r"^\d+_png_png", stem):
        return "NNN_png_png"
    if re.match(r"^w-\d+_jpeg", stem):
        return "w-N_jpeg"
    return "其他"


def main() -> int:
    ap = argparse.ArgumentParser(description="空标签图审计（疑似漏标毒样本）")
    ap.add_argument("--splits", default="train,valid")
    ap.add_argument("--weights", default=str(ROOT / "ai" / "model" / "labppe_v8s.pt"))
    ap.add_argument("--conf", type=float, default=0.5, help="判定阈值")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "_empty_audit"))
    args = ap.parse_args()

    wp = pathlib.Path(args.weights)
    if not wp.exists():
        print(f"[错误] 权重不存在: {wp}")
        return 2

    from ultralytics import YOLO
    model = YOLO(str(wp))
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    print(f"模型: {wp.name} | 类别: {names} | conf={args.conf} imgsz={args.imgsz}")

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv = out_dir / "empty_label_audit.tsv"

    rows = []
    stat = collections.defaultdict(lambda: collections.Counter())   # family -> split
    clsimg = collections.Counter()
    tot = 0

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        lst = ROOT / "data" / f"_empty_labels_{split}.txt"
        if not lst.exists():
            print(f"[跳过] 缺少清单 {lst}")
            continue
        stems = [l.strip() for l in lst.read_text(encoding="utf-8").splitlines() if l.strip()]
        if args.limit:
            stems = stems[: args.limit]
        print(f"\n--- {split}: {len(stems)} 张空标签图")
        for i, stem in enumerate(stems, 1):
            ip = find_image(split, stem)
            if ip is None:
                continue
            tot += 1
            r = model.predict(str(ip), conf=args.conf, iou=0.5, imgsz=args.imgsz,
                              device=args.device, verbose=False)[0]
            dets = []
            if r.boxes is not None and len(r.boxes):
                for b in r.boxes:
                    dets.append((int(b.cls.item()), float(b.conf.item())))
            dets.sort(key=lambda x: -x[1])
            fam = family(stem)
            stat[fam][split] += 1
            if dets:
                stat[fam][split + "|脏"] += 1
                for c, _ in dets:
                    clsimg[c] += 1
            desc = " ".join(f"{names.get(c, c)}:{cf:.2f}" for c, cf in dets)
            rows.append((split, ip.name, fam, len(dets), desc))
            if i % 100 == 0:
                print(f"    {i}/{len(stems)} ... 已标脏 {sum(1 for r_ in rows if r_[3])}")

    tsv.write_text("\n".join(f"{s}\t{n}\t{f}\t{c}\t{d}" for s, n, f, c, d in rows) + "\n",
                   encoding="utf-8")

    print("\n================ 汇总 ================")
    print(f"{'家族':<24}{'总数':>7}{'疑似脏':>8}{'占比':>8}   按 split")
    for fam, c in sorted(stat.items(), key=lambda x: -(sum(v for k, v in x[1].items() if '|' not in k))):
        base = sum(v for k, v in c.items() if "|" not in k)
        dirty = sum(v for k, v in c.items() if "|" in k)
        per = " ".join(f"{k.replace('|脏','*')}={v}" for k, v in sorted(c.items()))
        print(f"{fam:<24}{base:>7}{dirty:>8}{(dirty/base*100 if base else 0):>7.1f}%   {per}")
    print(f"\n类别命中图数: " + " ".join(f"{names.get(k, k)}={v}" for k, v in clsimg.most_common()))

    for split in {r[0] for r in rows}:
        sus = [r for r in rows if r[0] == split and r[3]]
        clean = [r for r in rows if r[0] == split and not r[3]]
        sus.sort(key=lambda r: -max([float(x.split(":")[1]) for x in r[4].split()] or [0]))
        (ROOT / "data" / f"_empty_audit_suspects_{split}.txt").write_text(
            "\n".join(f"{split}/{r[1]}" for r in sus) + "\n", encoding="utf-8")
        (ROOT / "data" / f"_empty_audit_clean_{split}.txt").write_text(
            "\n".join(f"{split}/{r[1]}" for r in clean) + "\n", encoding="utf-8")
        print(f"\n[{split}] 疑似毒样本 {len(sus)} / 真负样本 {len(clean)}")
        print(f"   清单: data/_empty_audit_suspects_{split}.txt")
        for r in sus[:12]:
            print(f"     {r[1][:52]:<54} {r[4]}")
    print(f"\n明细 TSV: {tsv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
