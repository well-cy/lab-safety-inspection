# -*- coding: utf-8 -*-
"""引入外部护目镜数据集（实验室安全护目镜补充）。

背景：demo 实测反复出现「眼镜被当成护目镜」「透明护目镜漏检」；E5 P0-3 计划
补 150~200 框护目镜正面照。ppes_* 削减 + 毒标签清洗后 goggles 已从 5199 降到
2845 框，且拔掉了 kp 机位一大片「眼镜当护目镜」的毒标签，需要用**干净的
护目镜域数据**补回来。

候选（均 Roboflow Universe，CC BY 4.0，独立来源）：
  - database-sjrvw/safety-goggles---ppe  411 图  5 类 Goggles/Gloves/Mask/
                                                 Coverall/Face_Shield（类映射契合）
  - example-workspace-ofjvo/safety-goggles-detection-project
                                          300 图  3 类 glasses/safety_goggles/
                                                 no_glasses（直接针对眼镜≠护目镜）
  - maddie-kajj5/safetyglasses-gvwfi      650 图  goggles/no goggles（机加工场景）

输出（原生 YOLOv8 格式，**不做类映射**，保留 data.yaml 原类名便于逐批决策）:
    data/raw/<tag>/{train,valid,test}/{images,labels}/ + data.yaml

幂等：目标目录已存在 data.yaml 则跳过。

用法:
    python tools/download_goggles_external.py --list
    python tools/download_goggles_external.py --all
    python tools/download_goggles_external.py --only goggles_ppe
"""
import argparse
from pathlib import Path

from roboflow import Roboflow

API_KEY = "nNQADL3zz9PMVKGnkYc1"
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

# tag -> (workspace, project, version, 说明)
TARGETS = {
    "goggles_ppe": ("database-sjrvw", "safety-goggles---ppe", 1,
                    "411 图 5 类 Goggles/Gloves/Mask/Coverall/Face_Shield"),
    "sglasses_proj": ("example-workspace-ofjvo", "safety-goggles-detection-project", 2,
                      "300 图 3 类 glasses/safety_goggles/no_glasses（眼镜 vs 护目镜）"),
    "sglasses_shop": ("maddie-kajj5", "safetyglasses-gvwfi", 5,
                      "650 图 goggles/no goggles（机加工/车间场景）"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description="下载外部护目镜数据集")
    ap.add_argument("--only", action="append", default=[], metavar="TAG")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for tag, (ws, proj, ver, desc) in TARGETS.items():
            print(f"{tag:<15} {ws}/{proj} v{ver}   {desc}")
        return 0

    keys = list(TARGETS) if (args.all or not args.only) else args.only
    bad = [k for k in keys if k not in TARGETS]
    if bad:
        print(f"[错误] 未知 tag: {bad}")
        return 2

    rf = Roboflow(api_key=API_KEY)
    for tag in keys:
        ws, proj, ver, desc = TARGETS[tag]
        dst = RAW / tag
        if (dst / "data.yaml").exists():
            print(f"[跳过] {tag} 已存在（{dst}）")
            continue
        print(f"[下载] {tag}  {ws}/{proj} v{ver}  {desc}")
        try:
            ds = rf.workspace(ws).project(proj).version(ver).download(
                "yolov8", location=str(dst))
            print(f"    → {ds.location}")
        except Exception as e:      # noqa: BLE001
            print(f"    [失败] {type(e).__name__}: {str(e)[:160]}")
    print("\n完成。下一步：拼版人工核域（实验室 vs 工业 vs 车间）→ 四关审计 → 入库。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
