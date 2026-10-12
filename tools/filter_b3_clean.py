#!/usr/bin/env python3
"""filter_b3_clean.py — 按「越界框 + 可疑偏移 + 空标注」三档过滤 B3，产出干净子集 data/b3_clean/

背景（2026-09-20 标注质量审计，v2）：
  B3 共 456 图 / 589 框。v1 仅剔除越界框图片（231 张，51%），但人工抽检发现：
  ① 存在「偏移但恰好没出界」的 Lab_coat 框（如 001_0043：框中心 0.776，右边缘 1.003）；
  ② 存在图内明显有实验服、标注却为 0 字节的漏标图（如 001_0191、015_0241）。
  故 v2 采用三档分级：

  bad     —— 任一类别框越界（边界超出 [-0.05, 1.05]）        → 剔除
  suspect —— 无越界但 Lab_coat 框中心 > 0.65（B3 主体基本居中，
             疑似同源右偏），或标注为空（需人工确认是否真负样本） → 待人工复核
  clean   —— 有标注、无越界、无可疑 Lab_coat 框              → 进干净子集

用法：
  python tools/filter_b3_clean.py            # 生成 data/b3_clean/ + 各档清单
  python tools/filter_b3_clean.py --dry-run  # 只统计，不复制

注意：
  clean 仅代表「机器可判定的干净」，抽检时仍建议打开
  outputs/draw_labels_check/（tools/draw_labels.py 生成）人工确认框贴合度。
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "raw" / "b3"
DST = ROOT / "data" / "b3_clean"
MARGIN = 0.05       # 越界判定容差
XC_SUSPECT = 0.65   # Lab_coat 中心可疑阈值（B3 主体基本居中）


def classify(label_path: Path):
    """返回 (档位, 原因)。档位: bad / suspect / clean"""
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        boxes.append((p[0], *map(float, p[1:])))

    if not boxes:
        return "suspect", "空标注（需人工确认是否真负样本）"

    for cls, xc, yc, w, h in boxes:
        l, r, t, b = xc - w / 2, xc + w / 2, yc - h / 2, yc + h / 2
        if l < -MARGIN or r > 1 + MARGIN or t < -MARGIN or b > 1 + MARGIN:
            return "bad", f"{cls} 越界 left={l:.2f} right={r:.2f} top={t:.2f} bottom={b:.2f}"

    for cls, xc, yc, w, h in boxes:
        if cls == "Lab_coat" and xc > XC_SUSPECT:
            return "suspect", f"Lab_coat 中心右偏 xc={xc:.2f}（疑似未重映射）"

    return "clean", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只统计，不复制文件")
    args = ap.parse_args()

    stats = {}   # split -> [total, clean, suspect, bad]
    lists = {"suspect": [], "bad": [], "clean": []}

    for split in ("train", "valid", "test"):
        img_dir = SRC / split / "images"
        lbl_dir = SRC / split / "labels"
        cnt = [0, 0, 0, 0]
        for img in sorted(img_dir.glob("*.jpg")):
            cnt[0] += 1
            lbl = lbl_dir / (img.stem + ".txt")
            tier, reason = classify(lbl) if lbl.exists() else ("bad", "标注文件缺失")
            cnt[{"clean": 1, "suspect": 2, "bad": 3}[tier]] += 1
            lists[tier].append(f"{split}/{img.name}\t{reason}" if reason else f"{split}/{img.name}")
            if tier == "clean" and not args.dry_run:
                for kind, f in (("images", img), ("labels", lbl)):
                    if f.exists():
                        d = DST / split / kind
                        d.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(f, d / f.name)
        stats[split] = cnt

    DST.mkdir(parents=True, exist_ok=True)
    (DST / "clean_list.txt").write_text("\n".join(lists["clean"]), encoding="utf-8")
    (ROOT / "data" / "b3_suspect_images.txt").write_text("\n".join(lists["suspect"]), encoding="utf-8")
    (ROOT / "data" / "b3_oob_images.txt").write_text("\n".join(lists["bad"]), encoding="utf-8")

    print("=" * 60)
    print(f"{'split':8s}{'总数':>6s}{'clean':>8s}{'suspect':>9s}{'bad':>6s}")
    tot = [0, 0, 0, 0]
    for sp, c in stats.items():
        for i in range(4):
            tot[i] += c[i]
        print(f"{sp:8s}{c[0]:>6d}{c[1]:>8d}{c[2]:>9d}{c[3]:>6d}")
    print("-" * 60)
    print(f"{'合计':8s}{tot[0]:>6d}{tot[1]:>8d}{tot[2]:>9d}{tot[3]:>6d}")
    print()
    print(f"clean 子集: {DST}" + ("（dry-run 未实际复制）" if args.dry_run else ""))
    print("清单: data/b3_clean/clean_list.txt | data/b3_suspect_images.txt | data/b3_oob_images.txt")
    print("复核建议: python tools/draw_labels.py 画框后人工过一遍 suspect 清单")


if __name__ == "__main__":
    main()
