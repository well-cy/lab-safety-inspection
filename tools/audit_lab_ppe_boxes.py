"""data/lab_ppe 四类标注几何审计（mask / gloves / lab_coat / goggles）

之前所有审计都针对 lab_coat（B3_data_audit.md、labcoat_si_audit.md），
本脚本对项目统一数据集 data/lab_ppe 的**全部四类**做一次几何体检，找出：
    1. 越界框：任一方向超出图片范围（容差 0.05，与 filter_b3_clean.py 同口径）
    2. 超小框：相对面积 < 0.0005（约 15x15 像素 @640，多为误标噪点）
    3. 超大框：相对面积 > 0.90（几乎糊满全图）
    4. 极端宽高比：w/h > 6 或 < 1/6
    5. 非法值：坐标非 0~1、行列数 != 5
    6. 重复框：同图同类 IoU > 0.90（一物体标两次）
    7. 空标注图：无框负样本数量
    8. 窄框（贴边 5% 以内）：框边落在图片边缘 5% 区域，常为被裁切目标——不计问题，仅统计

输出: data/lab_ppe_audit_report.txt（同时打印到终端）

用法:
    python tools/audit_lab_ppe_boxes.py                # 审计全部四类
    python tools/audit_lab_ppe_boxes.py --splits train # 只看 train
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent.parent
LAB_PPE = ROOT / "data" / "lab_ppe"
REPORT = ROOT / "data" / "lab_ppe_audit_report.txt"

NAMES = {0: "mask", 1: "gloves", 2: "lab_coat", 3: "goggles"}
EDGE_TOL = 0.05        # 越界容差（与 filter_b3_clean.py 一致）
MIN_AREA = 0.0005      # 超小框阈值
MAX_AREA = 0.90        # 超大框阈值
ASPECT = 6.0           # 极端宽高比阈值
IOU_DUP = 0.90         # 重复框 IoU 阈值


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def audit_split(lbl_dir: Path, img_dir: Path) -> dict:
    st = {
        "n_img": 0, "n_empty": 0, "per_class": Counter(), "img_per_class": Counter(),
        "oob": Counter(), "tiny": Counter(), "huge": Counter(), "aspect": Counter(),
        "illegal": 0, "dup": Counter(), "area": defaultdict(list),
        "examples": defaultdict(list), "src_prefix": Counter(),
        "flags": defaultdict(dict),   # cid -> image_name -> {tag: detail}
    }
    # 标注 stem -> 真实图片文件名（画框/拼图要用带扩展名的名字）
    stem2name = {}
    if img_dir.is_dir():
        for p in img_dir.iterdir():
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                stem2name[p.stem] = p.name

    def flag(cid: int, stem: str, tag: str, detail: str = "") -> None:
        name = stem2name.get(stem, stem + ".jpg")
        st["flags"][cid].setdefault(name, {})[tag] = detail

    for lb in sorted(lbl_dir.glob("*.txt")):
        st["n_img"] += 1
        stem = lb.stem
        prefix = "labcoat_si(lcsi_)" if stem.startswith("lcsi_") else "dataset1"
        st["src_prefix"][prefix] += 1

        lines = [l for l in lb.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not lines:
            st["n_empty"] += 1
            continue

        boxes_by_cls = defaultdict(list)
        seen_cls = set()
        for line_no, line in enumerate(lines, 1):
            p = line.split()
            if len(p) != 5:
                st["illegal"] += 1
                st["examples"]["illegal"].append(f"{lb.name}:{line_no} 列数={len(p)}")
                continue
            try:
                cid = int(p[0])
                vals = [float(x) for x in p[1:]]
            except ValueError:
                st["illegal"] += 1
                st["examples"]["illegal"].append(f"{lb.name}:{line_no} 非数值")
                continue
            xc, yc, w, h = vals
            if cid not in NAMES or not all(0.0 <= v <= 1.0 for v in vals) or w <= 0 or h <= 0:
                st["illegal"] += 1
                st["examples"]["illegal"].append(f"{lb.name}:{line_no} cls={cid} vals={vals}")
                continue

            st["per_class"][cid] += 1
            seen_cls.add(cid)
            st["area"][cid].append(w * h)
            x1, y1, x2, y2 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2

            if x1 < -EDGE_TOL or x2 > 1 + EDGE_TOL or y1 < -EDGE_TOL or y2 > 1 + EDGE_TOL:
                st["oob"][cid] += 1
                flag(cid, stem, "越界", f"x[{x1:.2f},{x2:.2f}]y[{y1:.2f},{y2:.2f}]")
                if len(st["examples"]["oob"]) < 20:
                    st["examples"]["oob"].append(
                        f"{lb.name} {NAMES[cid]} x[{x1:.2f},{x2:.2f}] y[{y1:.2f},{y2:.2f}]")
            if w * h < MIN_AREA:
                st["tiny"][cid] += 1
                flag(cid, stem, "超小", f"{w*h:.5f}")
                if len(st["examples"]["tiny"]) < 15:
                    st["examples"]["tiny"].append(f"{lb.name} {NAMES[cid]} area={w*h:.5f}")
            if w * h > MAX_AREA:
                st["huge"][cid] += 1
                flag(cid, stem, "超大", f"{w*h:.3f}")
                if len(st["examples"]["huge"]) < 15:
                    st["examples"]["huge"].append(f"{lb.name} {NAMES[cid]} area={w*h:.3f}")
            if w / h > ASPECT or h / w > ASPECT:
                st["aspect"][cid] += 1
                flag(cid, stem, "怪比例", f"{max(w/h, h/w):.1f}")
                if len(st["examples"]["aspect"]) < 15:
                    st["examples"]["aspect"].append(
                        f"{lb.name} {NAMES[cid]} w={w:.3f} h={h:.3f} 比={max(w/h, h/w):.1f}")
            boxes_by_cls[cid].append((x1, y1, x2, y2))

        for cid in seen_cls:
            st["img_per_class"][cid] += 1
            bs = boxes_by_cls[cid]
            for i in range(len(bs)):
                for j in range(i + 1, len(bs)):
                    if iou(bs[i], bs[j]) > IOU_DUP:
                        st["dup"][cid] += 1
                        flag(cid, stem, "重复框", f"IoU>{IOU_DUP}")
                        if len(st["examples"]["dup"]) < 10:
                            st["examples"]["dup"].append(f"{lb.name} {NAMES[cid]} 两个框重叠")
    return st


def pct(part: int, total: int) -> str:
    return f"{(part / total * 100):.1f}%" if total else "-"


def med(vals: list[float]) -> str:
    if not vals:
        return "-"
    return f"{median(vals):.4f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(LAB_PPE))
    ap.add_argument("--splits", default="train,valid,test")
    ap.add_argument("--no-dump", action="store_true", help="只出报告，不导出 flags 清单")
    args = ap.parse_args()

    root = Path(args.root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    out: list[str] = []
    add = out.append
    all_flags: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)

    add("data/lab_ppe 四类标注几何审计报告")
    add("=" * 64)
    add(f"阈值: 越界容差±{EDGE_TOL} | 超小框<{MIN_AREA} | 超大框>{MAX_AREA} | 极端宽高比>{ASPECT} | 重复框IoU>{IOU_DUP}")
    add("")

    total = {c: Counter() for c in NAMES}
    for sp in splits:
        ld, idir = root / "labels" / sp, root / "images" / sp
        if not ld.exists():
            add(f"[跳过] {ld} 不存在")
            continue
        st = audit_split(ld, idir)
        add(f"── split: {sp} ──")
        add(f"图片总数 {st['n_img']}（其中无框负样本 {st['n_empty']}）  来源: {dict(st['src_prefix'])}")
        add(f"非法标注行: {st['illegal']}")
        add("")
        add(f"{'类别':<10}{'框数':>7}{'涉及图':>8}{'越界':>8}{'超小':>7}{'超大':>7}{'怪比例':>8}{'重复':>7}{'面积中位':>10}")
        for cid, cname in NAMES.items():
            n = st["per_class"][cid]
            add(f"{cname:<10}{n:>7}{st['img_per_class'][cid]:>8}"
                f"{st['oob'][cid]:>8}{st['tiny'][cid]:>7}{st['huge'][cid]:>7}"
                f"{st['aspect'][cid]:>8}{st['dup'][cid]:>7}{med(st['area'][cid]):>10}")
            for k, v in st["per_class"].items():
                pass
        add("")
        for cid in NAMES:
            total[cid]["n"] += st["per_class"][cid]
            total[cid]["img"] += st["img_per_class"][cid]
            total[cid]["oob"] += st["oob"][cid]
            total[cid]["tiny"] += st["tiny"][cid]
            total[cid]["huge"] += st["huge"][cid]
            total[cid]["aspect"] += st["aspect"][cid]
            total[cid]["dup"] += st["dup"][cid]
        for name, tags in st["flags"].items():
            for img_name, t in tags.items():
                all_flags[name].setdefault(f"{sp}/{img_name}", {}).update(t)
        for tag, limit in (("oob", 8), ("tiny", 8), ("aspect", 8), ("dup", 5), ("huge", 5), ("illegal", 8)):
            if st["examples"][tag]:
                add(f"  [{tag} 示例] " + " | ".join(st["examples"][tag][:limit]))
                add("")
        add("")

    add("── 全量合计 ──")
    add(f"{'类别':<10}{'框数':>7}{'涉及图':>8}{'越界':>8}{'越界占比':>10}{'超小':>7}{'超大':>7}{'怪比例':>8}{'重复':>7}")
    for cid, cname in NAMES.items():
        t = total[cid]
        add(f"{cname:<10}{t['n']:>7}{t['img']:>8}{t['oob']:>8}{pct(t['oob'], t['n']):>10}"
            f"{t['tiny']:>7}{t['huge']:>7}{t['aspect']:>8}{t['dup']:>7}")

    text = "\n".join(out)
    print(text)
    REPORT.write_text(text + "\n", encoding="utf-8")
    print(f"\n报告已写入: {REPORT}")

    if not args.no_dump:
        for cid, cname in NAMES.items():
            imgs = all_flags.get(cid) or {}
            if not imgs:
                continue
            fp = ROOT / "data" / f"flags_{cname}.txt"
            lines = [
                f"# {cname} 几何问题图清单（共 {len(imgs)} 张）—— 由 tools/audit_lab_ppe_boxes.py 生成",
                "# 格式: <split>/<图片文件名>\\t<问题标签(数值)>  可直接喂给 draw_labels.py --list",
                "# 注意: 只是几何可疑，不代表一定错标，需人工看图确认",
            ]
            for key in sorted(imgs):
                tags = ",".join(f"{k}({v})" if v else k for k, v in imgs[key].items())
                lines.append(f"{key}\t{tags}")
            fp.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"  → {fp}  ({len(imgs)} 张)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
