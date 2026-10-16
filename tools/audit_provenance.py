"""训练底座来源域审计

用途：把 data/lab_ppe 的每张图归到「来源数据集 → 场景域」，并统计各类框数、
空标签数，回答「有多少训练数据其实不是实验室场景 / 框选口径可疑」。

归属优先级：
  1) data/lab_ppe/manifest.csv 的 source_dataset（权威，覆盖 dataset1/labcoat_si/ppedetect）
  2) 文件名模式（ppes_* 家族、gozluk、stock 视频帧、产品图等）

输出：docs/E6_训练底座来源域审计.md + data/lab_ppe_provenance.csv
"""
from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "data" / "lab_ppe"
SPLITS = ("train", "valid", "test")
CLASS_NAMES = ["mask", "gloves", "lab_coat", "goggles"]


def strip_rf(stem: str) -> str:
    return stem.split(".rf.")[0]


# ---- 场景域归属规则（先匹配先生效）----
# (域名, 场景判定, 判定关键词/正则)
DOMAIN_RULES = [
    ("ppes_*", "实验室外-工业CCTV", r"^ppes[_-]|^NVR|^PPE_?Data"),
    ("gozluk/eyeglass", "实验室外-眼镜电商图", r"^sgl_gozluk|^gozluk|^eyeglass|^sgl_goggle|^safety_goggles_"),
    ("stock视频帧", "实验室外-通用视频素材", r"^VID_|uhd_|istock|adpp|^\d{8}_\d{4}_|fps_mp4|^13456599"),
    ("KakaoTalk", "实验室外-手机照", r"^KakaoTalk"),
    ("手套产品图", "实验室外-商品图", r"Glove|glove|gloves[-_]|Wetsuit|golf|gant|perchatk|HE_Gloves"),
    ("lcsi_*", "实验室-穿戴场景", r"^lcsi"),
    ("web_*", "实验室-网页图", r"^web_?p?\d"),
    ("sgv/sgl", "实验室外-眼镜网图(sglasses)", r"^sgl_|^sgv_"),
]

# manifest 的 source_dataset → 场景域
MANIFEST_DOMAIN = {
    "labcoat_si": "实验室-实验服场景",
    "dataset1": "混合(dataset1)",
    "ppedetect": "实验室外-工业CCTV",
}


def domain_of(name: str, manifest_src: str | None) -> tuple[str, str]:
    if manifest_src:
        return manifest_src, MANIFEST_DOMAIN.get(manifest_src, f"manifest:{manifest_src}")
    base = strip_rf(name)
    for tag, scene, pat in DOMAIN_RULES:
        if re.search(pat, base):
            return tag, scene
    return "其他/未归类", "未归类"


def load_manifest() -> dict[str, str]:
    p = LAB / "manifest.csv"
    if not p.exists():
        return {}
    m: dict[str, str] = {}
    with p.open(encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            m.setdefault(r["new_filename"], r["source_dataset"])
    return m


# ---- 细分组：把 dataset1 再拆成「手套商品图 / 护目镜网图 / 穿戴演示」----
def load_d1_subsets() -> dict[str, set[str]]:
    """dataset1 每图 → 该图包含的原始类别名集合。"""
    out: dict[str, set[str]] = defaultdict(set)
    p = LAB / "manifest.csv"
    if not p.exists():
        return out
    with p.open(encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["source_dataset"] == "dataset1":
                out[r["new_filename"]].add(r["original_class_name"])
    return out


def fine_group(name: str, man: dict[str, str], d1: dict[str, set[str]]) -> str:
    """返回细分组名。前缀 X- 表示「实验室外/口径可疑」，~ 表示「实验室但口径需复核」。"""
    if name in man:
        s = man[name]
        if s == "labcoat_si":
            return "实验室-白大褂场景(labcoat_si)"
        if s == "ppedetect":
            return "X-工业CCTV(ppedetect)"
        cls = d1.get(name, set())
        low = {c.lower() for c in cls}
        if low <= {"glove", "gloves"}:
            return "X-手套商品图(dataset1-A)"
        if len(cls) == 1 and list(cls)[0].startswith("safety goggles"):
            return "X-护目镜网图(dataset1-B)"
        if any("lab coat" in c for c in low):
            return "~穿戴演示-含白大褂(dataset1-C1)"
        return "X-穿戴演示-无白大褂(dataset1-C2)"
    base = strip_rf(name)
    if re.search(r"^ppes[_-]|^NVR|^PPE_?Data", base):
        return "X-工业CCTV(ppes_*)"
    if re.search(r"^web_?p?\d", base):
        return "实验室-网页图(web_*)"
    if re.search(r"^sgl_|^sgv_|^gozluk|^safety_goggles", base):
        return "X-眼镜网图(外部)"
    if re.search(r"^lcsi", base):
        return "实验室-实验服场景(lcsi)"
    return "X-未归类/其他"


def main() -> None:
    man = load_manifest()
    stats: dict[str, dict] = defaultdict(
        lambda: {"scene": "", "imgs": Counter(), "boxes": Counter(), "empty": Counter()}
    )
    per_file_rows = []

    for sp in SPLITS:
        idir = LAB / "images" / sp
        for ip in sorted(idir.iterdir()):
            if ip.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            base = strip_rf(ip.name)
            src = man.get(ip.name)
            tag, scene = domain_of(ip.name, src)
            lp = LAB / "labels" / sp / (ip.stem + ".txt")
            lines = [l.split() for l in lp.read_text().splitlines() if l.strip()] if lp.exists() else []
            cls = Counter(int(float(p[0])) for p in lines)
            st = stats[tag]
            st["scene"] = scene
            st["imgs"][sp] += 1
            st["boxes"][sp] += len(lines)
            if not lines:
                st["empty"][sp] += 1
            for k, v in cls.items():
                st.setdefault("cls", Counter())[(sp, k)] += v
            per_file_rows.append(
                (sp, ip.name, src or tag, scene, len(lines), cls[0], cls[1], cls[2], cls[3])
            )

    # ---- Markdown 报告 ----
    out_md = ROOT / "docs" / "E6_训练底座来源域审计.md"
    L: list[str] = []
    L.append("# E6 训练底座来源域审计（框选口径复核）\n")
    L.append("> 生成工具：`tools/audit_provenance.py`　数据：`data/lab_ppe`\n")
    L.append("> 背景：她指出「之前框选的手套/护目镜/实验服标准都不准确，很多不该算入，且要求是实验室场景」。")
    L.append("> 本审计回答：现有训练集里，**哪些来源不是实验室场景、各类框分别来自哪里**。\n")

    for sp in SPLITS:
        n_img = sum(st["imgs"][sp] for st in stats.values())
        n_box = sum(st["boxes"][sp] for st in stats.values())
        L.append(f"\n## {sp}（{n_img} 图 / {n_box} 框）\n")
        L.append("| 来源域 | 场景判定 | 图数 | 占比 | 框数 | mask | gloves | lab_coat | goggles | 空标签 |")
        L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        ranked = sorted(stats.items(), key=lambda kv: -kv[1]["imgs"][sp])
        for tag, st in ranked:
            ni = st["imgs"][sp]
            if ni == 0:
                continue
            c = st.get("cls", Counter())
            L.append(
                f"| `{tag}` | {st['scene']} | {ni} | {ni / n_img * 100:.1f}% | {st['boxes'][sp]} | "
                f"{c[(sp,0)]} | {c[(sp,1)]} | {c[(sp,2)]} | {c[(sp,3)]} | {st['empty'][sp]} |"
            )
    out_md.write_text("\n".join(L) + "\n", encoding="utf-8")

    # ---- CSV ----
    out_csv = ROOT / "data" / "lab_ppe_provenance.csv"
    with out_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "filename", "provenance", "scene", "n_box", "mask", "gloves", "lab_coat", "goggles"])
        w.writerows(per_file_rows)

    # ---- 细分组 × 类别 矩阵 ----
    d1 = load_d1_subsets()
    all_mat: dict[str, tuple[dict, Counter]] = {}
    for sp in ("train", "valid"):
        mat: dict[str, Counter] = defaultdict(Counter)
        imgs: Counter = Counter()
        for ip in (LAB / "images" / sp).iterdir():
            if ip.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            g = fine_group(ip.name, man, d1)
            imgs[g] += 1
            lp = LAB / "labels" / sp / (ip.stem + ".txt")
            lines = [l.split() for l in lp.read_text().splitlines() if l.strip()] if lp.exists() else []
            if not lines:
                mat[g]["空标签"] += 1
            for p in lines:
                mat[g][CLASS_NAMES[int(float(p[0]))]] += 1
        L.append(f"\n### {sp} 细分来源 × 类别框数\n")
        L.append("| 细分来源 | 图片 | 框 | mask | gloves | lab_coat | goggles | 空标签 |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for g in sorted(imgs, key=lambda x: -(sum(mat[x][c] for c in CLASS_NAMES))):
            row = mat[g]
            L.append(
                f"| {g} | {imgs[g]} | {sum(row[c] for c in CLASS_NAMES)} | "
                + " | ".join(str(row[c]) for c in CLASS_NAMES)
                + f" | {row['空标签']} |"
            )
        L.append(
            f"| **合计** | **{sum(imgs.values())}** | "
            f"**{sum(sum(mat[g][c] for c in CLASS_NAMES) for g in imgs)}** | "
            + " | ".join(f"**{sum(mat[g][c] for g in imgs)}**" for c in CLASS_NAMES)
            + f" | **{sum(mat[g]['空标签'] for g in imgs)}** |"
        )
        all_mat[sp] = (mat, imgs)
    out_md.write_text("\n".join(L) + "\n", encoding="utf-8")

    # ---- 结论（静态，随工具输出）----
    L += [
        "",
        "## 结论\n",
        "1. **train 里明确非实验室/口径可疑的图约 64%**（手套商品图 + 护目镜网图 + 工业 CCTV + 眼镜网图 + 未归类）。",
        "2. **gloves 7,478 框中约 57% 来自非实验室来源**（dataset1-A 商品图 2,430 框 + ppes_* 工业 CCTV 1,821 框等）——滑雪手套、骑行手套、",
        "   高尔夫手套、潜水手套、防割工作手套，甚至厨房海绵都被标成了 gloves。",
        "3. **goggles 2,895 框中约 63% 来自网图/工业**（dataset1-B 的 gozluk_* 土耳其眼镜网 + safety_goggles 产品图 1,654 框等）。",
        "4. **lab_coat 2,389 框中约 84% 来自实验室场景**（dataset1-C1 穿戴演示 1,411 + labcoat_si 405 + web_* 201）——三类里最干净，",
        "   但仍混有 336 框来自「无白大褂的穿戴演示」。",
        "5. **valid 集的 84.3% 是 ppes_* 工业 CCTV**——我们一直在「工厂监控画面」上验证模型，与实验室场景几乎无关，",
        "   这解释了 valid mAP 与真实场景表现的巨大落差。test 同理（82.2% ppes_*）。",
        "6. 实验室场景占比最高的来源：`dataset1-C1`（穿戴演示，含白大褂标注）、`labcoat_si`（432 图，jeiotech 实验服）、`web_*`（网页图）。",
        "",
        "## 建议处置（等确认后执行）\n",
        "- **保留**：`dataset1-C1`、`dataset1-C2`（先人工确认场景）、`labcoat_si`、`web_*`。",
        "- **移出训练/验证集**（不删，移入 `data/_rejected/` 可整体移回）：dataset1-A、dataset1-B、ppes_*、ppedetect、外部眼镜网图、未归类。",
        "- **重建 valid/test**：按「实验室场景优先」从保留池重新划分，并与 dataset2（external_test 红线，251 图）配合使用。",
        "- 分诊证据：`outputs/phase2_provenance/sheet_*/sheet_01.jpg`（每组 48 张抽样图版）。",
    ]
    out_md.write_text("\n".join(L) + "\n", encoding="utf-8")

    # ---- 控制台摘要 ----
    for sp in ("train", "valid"):
        mat, imgs = all_mat[sp]
        print(f"===== {sp}: {sum(imgs.values())} 图 =====")
        hdr = f"{'细分来源':<34}{'图':>6}{'框':>7}{'mask':>7}{'gloves':>8}{'coat':>7}{'gog':>6}{'空标签':>7}"
        print(hdr)
        for g in sorted(imgs, key=lambda x: -(sum(mat[x][c] for c in CLASS_NAMES))):
            r = mat[g]
            print(
                f"{g:<34}{imgs[g]:>6}{sum(r[c] for c in CLASS_NAMES):>7}{r['mask']:>7}"
                f"{r['gloves']:>8}{r['lab_coat']:>7}{r['goggles']:>6}{r['空标签']:>7}"
            )
        print()
    print(f"\n报告: {out_md.relative_to(ROOT)}")
    print(f"逐图表: {out_csv.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
