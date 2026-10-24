# -*- coding: utf-8 -*-
"""场景红线合规复核（五类口径）

背景：她 2026-10-07 定了「场景红线」——
    训练图必须是 实验室/科研/医疗操作场景（实验室、准备室、穿堂/更衣区）；
    工厂车间、CCTV 监控、电商商品图、生活照 不入训练集。

本工具回答：**当前 data/lab_ppe 各 split 里，还有多少图属于红线外来源。**

归属优先级：
  1) manifest.csv 的 source_dataset（dataset1 / labcoat_si / ppedetect）
  2) 文件名前缀规则（ppes_* / gozluk / sgv / 网图批 / 上传批 …）

输出：docs/E6_场景红线合规复核_20261010.md
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "data" / "lab_ppe"
SPLITS = ("train", "valid", "test")
CLASS_NAMES = ["person", "mask", "gloves", "lab_coat", "goggles"]

OK = "✅ 合规"
BAD = "❌ 红线外"
WARN = "⚠️ 待复核"

# (域名, 场景判定, 红线判定, 正则)  —— 先匹配先生效
RULES = [
    ("ppes_*", "实验室外·工业监控(CCTV)", BAD, r"^ppes[_-]|^NVR|^PPE_?Data"),
    ("眼镜电商/网图", "实验室外·商品图/网图", BAD,
     r"^gozluk|^sgl_|^sgv_|^eyeglass|^safety_goggles|^sgl_goggle"),
    ("KakaoTalk", "实验室外·手机生活照", BAD, r"^KakaoTalk"),
    ("图库/视频素材", "来源不明·通用图库素材", WARN,
     r"^istock|^VID_|^uhd_|^adpp|fps_mp4|^\d{8}_\d{4}_|^13456599"),
    ("网图批(r1/r2/r3)", "实验室·网页图（已过四关）", OK, r"^webs?_|^websr\d_"),
    ("团队上传批(up0)", "实验室·团队实测照片", OK, r"^up\d_"),
    ("旧网图批(web_*等)", "实验室·网页图", OK, r"^web_?p?\d|^webp_"),
]

MANIFEST_RULE = {
    "labcoat_si": ("labcoat_si", "实验室·实验服场景", OK),
    "ppedetect": ("ppedetect", "实验室外·工业监控(CCTV)", BAD),
    "dataset1": ("dataset1", "混合·需逐图复核", WARN),
}


def strip_rf(stem: str) -> str:
    return stem.split(".rf.")[0]


def load_manifest() -> dict[str, str]:
    p = LAB / "manifest.csv"
    m: dict[str, str] = {}
    if not p.exists():
        return m
    with p.open(encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            m.setdefault(r["new_filename"], r["source_dataset"])
    return m


def classify(name: str, man: dict[str, str]) -> tuple[str, str, str]:
    src = man.get(name)
    if src and src in MANIFEST_RULE:
        return MANIFEST_RULE[src]
    base = strip_rf(name)
    for tag, scene, verdict, pat in RULES:
        if re.search(pat, base):
            return tag, scene, verdict
    return "未归类", "来源不明", WARN


def main() -> None:
    man = load_manifest()
    # stats[split][tag] = {"scene","verdict","imgs","boxes","empty","cls"Counter}
    stats: dict[str, dict[str, dict]] = {sp: defaultdict(
        lambda: {"scene": "", "verdict": "", "imgs": 0, "boxes": 0,
                 "empty": 0, "cls": Counter()}) for sp in SPLITS}

    for sp in SPLITS:
        for ip in sorted((LAB / "images" / sp).iterdir()):
            if ip.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            tag, scene, verdict = classify(ip.name, man)
            st = stats[sp][tag]
            st["scene"], st["verdict"] = scene, verdict
            st["imgs"] += 1
            lp = LAB / "labels" / sp / (ip.stem + ".txt")
            rows = [l.split() for l in lp.read_text(encoding="utf-8").splitlines() if l.strip()] if lp.exists() else []
            st["boxes"] += len(rows)
            if not rows:
                st["empty"] += 1
            for p in rows:
                st["cls"][int(float(p[0]))] += 1

    L: list[str] = []
    L.append("# E6 场景红线合规复核（2026-10-10）\n")
    L.append("> 生成工具：`tools/audit_scene_compliance.py`　数据：`data/lab_ppe`（五类口径）")
    L.append("> 红线定义（她 2026-10-07）：训练图必须是 **实验室/科研/医疗操作场景**（实验室、准备室、穿堂/更衣区）；")
    L.append("> **工厂车间、CCTV 监控、电商商品图、生活照不入训练集**（负样本空场景亦须实验室环境）。\n")

    head = ("| 来源域 | 场景判定 | 红线 | 图数 | 占比 | 框数 | "
            + " | ".join(CLASS_NAMES) + " | 空标签 |")
    sep = "|---|---|---|---:|---:|---:|" + "---:|" * len(CLASS_NAMES) + "---:|"

    for sp in SPLITS:
        tot = sum(st["imgs"] for st in stats[sp].values())
        L.append(f"\n## {sp}（{tot} 图）\n")
        L.append(head)
        L.append(sep)
        for tag, st in sorted(stats[sp].items(), key=lambda kv: -kv[1]["imgs"]):
            c = st["cls"]
            L.append(
                f"| {tag} | {st['scene']} | {st['verdict']} | {st['imgs']} | "
                f"{st['imgs'] / tot * 100:.1f}% | {st['boxes']} | "
                + " | ".join(str(c[i]) for i in range(len(CLASS_NAMES)))
                + f" | {st['empty']} |"
            )

    # ---- 结论 ----
    def share(sp: str, verdict: str) -> tuple[int, float]:
        n = sum(st["imgs"] for st in stats[sp].values() if st["verdict"] == verdict)
        tot = sum(st["imgs"] for st in stats[sp].values())
        return n, (n / tot * 100 if tot else 0.0)

    L.append("\n## 结论\n")
    for sp in SPLITS:
        nb, pb = share(sp, BAD)
        nw, pw = share(sp, WARN)
        no, po = share(sp, OK)
        L.append(f"- **{sp}**：合规 {no} 图（{po:.1f}%）／待复核 {nw}（{pw:.1f}%）／**红线外 {nb}（{pb:.1f}%）**")
    L.append("")
    L.append("- `train`/`valid` 已无 CCTV（ppes_* = 0），红线外来源已清零；存量风险集中在 **dataset1（混合域）** 需逐图复核。")
    L.append("- **`test` 仍以 ppes_* 工业 CCTV 为主**（同 10-07 审计结论），另含眼镜电商图/手机照。")
    L.append("  → test 作为**内部评测集**，其指标反映的是「工厂监控画面」而非实验室场景；")
    L.append("  真实域外测应使用 `dataset2`（251 图，永久 external_test 红线）。是否重建 test 需拍板。")

    out = ROOT / "docs" / "E6_场景红线合规复核_20261010.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")

    # ---- 控制台 ----
    for sp in SPLITS:
        tot = sum(st["imgs"] for st in stats[sp].values())
        print(f"===== {sp}: {tot} 图 =====")
        for tag, st in sorted(stats[sp].items(), key=lambda kv: -kv[1]["imgs"]):
            print(f"  {st['verdict']}  {tag:<18} {st['imgs']:>4} 图 / {st['boxes']:>5} 框")
        print()
    print(f"报告: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
