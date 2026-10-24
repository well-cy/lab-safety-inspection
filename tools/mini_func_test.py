# -*- coding: utf-8 -*-
"""
本地 CPU 迷你功能测试：抽 100 张图、YOLOv8n、2 epoch，验证
「建迷你数据集 -> 训练 -> 出 best.pt -> 推理」整条链路。
只验证链路，不追求精度。全部产物在 outputs/mini_test/。
"""
import os
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "lab_ppe"
MINI = ROOT / "data" / "_mini_ppe"
random.seed(42)


def build(split, n):
    imgs = SRC / "images" / split
    labels = SRC / "labels" / split
    files = [p for p in imgs.glob("*") if (labels / (p.stem + ".txt")).exists()]
    random.shuffle(files)
    picked = files[:n]
    for p in picked:
        dst_img = MINI / "images" / split / p.name
        dst_lbl = MINI / "labels" / split / (p.stem + ".txt")
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        if not dst_img.exists():
            try:
                os.link(p, dst_img)
            except OSError:
                shutil.copy2(p, dst_img)
        if not dst_lbl.exists():
            try:
                os.link(labels / (p.stem + ".txt"), dst_lbl)
            except OSError:
                shutil.copy2(labels / (p.stem + ".txt"), dst_lbl)
    return len(picked)


if __name__ == "__main__":
    n_tr = build("train", 100)
    n_va = build("valid", 30)
    print(f"mini 数据集就绪: train={n_tr} valid={n_va}")

    yaml = MINI / "data.yaml"
    yaml.write_text(
        f"path: {MINI.as_posix()}\n"
        "train: images/train\n"
        "val: images/valid\n"
        "nc: 4\n"
        "names:\n  0: mask\n  1: gloves\n  2: lab_coat\n  3: goggles\n",
        encoding="utf-8",
    )

    import torch

    torch.set_num_threads(8)
    from ultralytics import YOLO

    PT = ROOT / "outputs" / "mini_test" / "yolov8n.pt"
    m = YOLO(str(PT))
    r = m.train(
        data=str(yaml),
        epochs=2,
        imgsz=320,
        batch=8,
        device="cpu",
        workers=0,
        project=str(ROOT / "outputs" / "mini_test"),
        name="v8n_mini",
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    best = Path(r.save_dir) / "weights" / "best.pt"
    print("BEST:", best)
    print("MINI_TRAIN_DONE")
