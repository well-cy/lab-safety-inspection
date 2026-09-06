# -*- coding: utf-8 -*-
"""E1-1 下载两个 Roboflow LAB_PPE 数据集（YOLOv8 格式）到 data/raw/。

用法:
    python tools/download_datasets.py <ROBOFLOW_API_KEY>

下载目标:
    dataset1: workspace=vitor-ferraz-marini  project=lab_ppe-g8hja  (约7.4k张, 13类)
    dataset2: workspace=masters-ppe          project=lab-ppe        (约251张, 10类)

输出:
    data/raw/dataset1/   (train/ valid/ test/ data.yaml)
    data/raw/dataset2/   (train/ valid/ test/ data.yaml)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

DATASETS = [
    {"key": "dataset1", "workspace": "vitor-ferraz-marini", "project": "lab_ppe-g8hja"},
    {"key": "dataset2", "workspace": "masters-ppe", "project": "lab-ppe"},
]


def main(api_key: str) -> None:
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    for ds in DATASETS:
        dest = RAW / ds["key"]
        if (dest / "data.yaml").exists():
            print(f"[skip] {ds['key']} 已存在于 {dest}")
            continue
        print(f"[download] {ds['workspace']}/{ds['project']} -> {dest}")
        project = rf.workspace(ds["workspace"]).project(ds["project"])
        # 选择最新版本（Roboflow 会返回所有版本，取最大版本号）
        versions = project.versions()
        latest = max(versions, key=lambda v: int(getattr(v, "version", 0) or 0))
        ver_num = getattr(latest, "version", "?")
        print(f"  最新版本: v{ver_num}")
        latest.download("yolov8", location=str(dest))
        print(f"[done] {ds['key']} 下载完成: {dest}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
