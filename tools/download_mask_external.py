# -*- coding: utf-8 -*-
"""引入外部 PPE 数据集（mask 补充为主）。

背景：mask 是四类中最薄的一类（收紧去重后仅 1354 图 / 1799 框），且域单一
（ppes/legacy 两来源），需要外部数据补充外观多样性。

候选（均 Roboflow Universe，CC BY 4.0）：
  - epidetect/ppe-wlllw v2   1758 图  7 类 HELMET/PERSON/VEST/MASK/GLASS/GLOVE/BOOTS
                                      MASK 259 框；工业安全域，独立来源
  - saba/ppedetection   v7    364 图  5 类 Goggles/Gloves/Coverall/Mask/Face_Shield
                                      Mask 394 框（密度 1.08/图，含不规范佩戴），类映射最契合

输出（原生 YOLOv8 格式，**不映射**，保留 data.yaml 原类名便于独立决策）:
    data/raw/<tag>/{train,valid,test}/{images,labels}/ + data.yaml

幂等：目标目录已存在 data.yaml 则跳过，可安全重复执行。

用法:
    python tools/download_mask_external.py --list              # 列出目标
    python tools/download_mask_external.py --only ppedetect    # 只下某一个
    python tools/download_mask_external.py --only ppedetect --only ppe_wlllw
    python tools/download_mask_external.py --all
"""
import argparse
from pathlib import Path

from roboflow import Roboflow

API_KEY = "nNQADL3zz9PMVKGnkYc1"

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

# tag -> (workspace, project, version, 说明)
TARGETS = {
    "ppe_wlllw": ("epidetect", "ppe-wlllw", 2,
                  "工业 PPE 7 类（MASK 259 框 / 1758 图）"),
    "ppedetect": ("saba", "ppedetection", 7,
                  "工业 PPE 5 类（Mask 394 框 / 364 图，类映射最契合）"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description="下载外部 PPE 数据集（mask 补充）")
    ap.add_argument("--only", action="append", default=[], metavar="TAG",
                    help="只下载指定 tag（可重复）")
    ap.add_argument("--all", action="store_true", help="下载全部目标")
    ap.add_argument("--version", type=int, default=0, metavar="N",
                    help="覆盖目标版本号；默认版本的服务端导出卡住时，"
                         "可用老版本的现成导出（如 --version 1）")
    ap.add_argument("--list", action="store_true", help="仅列出目标")
    args = ap.parse_args()

    if args.list:
        for tag, (ws, proj, ver, desc) in TARGETS.items():
            print(f"{tag:<10} {ws}/{proj} v{ver}   {desc}")
        return 0

    keys = list(TARGETS) if (args.all or not args.only) else args.only
    unknown = [k for k in keys if k not in TARGETS]
    if unknown:
        print(f"未知 tag: {unknown}；可用: {list(TARGETS)}")
        return 2

    RAW.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=API_KEY)
    for tag in keys:
        ws, proj, ver, desc = TARGETS[tag]
        if args.version:
            ver = args.version
        dst = RAW / tag
        if (dst / "data.yaml").exists():
            n = sum(1 for p in dst.rglob("*")
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            print(f"[跳过] {tag} 已存在（{n} 张图）: {dst}")
            continue
        print(f"\n[下载] {tag} = {ws}/{proj} v{ver}  {desc}")
        print(f"       目标: {dst}")
        rf.workspace(ws).project(proj).version(ver).download(
            "yolov8", location=str(dst))
        n = sum(1 for p in dst.rglob("*")
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        print(f"[完成] {tag}: {n} 张图 -> {dst}")

    print("\n全部完成。下一步三关审计：")
    print("  1) 红线去重（与 dataset1/dataset2 同源）")
    print("  2) python tools/dedup_phash.py 同源/近重复检测")
    print("  3) python tools/screen_dirty_graphics.py 水印/图库图筛查")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
