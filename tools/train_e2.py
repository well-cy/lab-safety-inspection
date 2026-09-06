# -*- coding: utf-8 -*-
"""E2 训练 YOLOv8s baseline（lab_ppe 4类）。

约定：
- 官方 YOLOv8s 预训练权重（首次运行自动下载 yolov8s.pt）
- epochs=50, imgsz=640, batch=16（RTX 4060 Laptop 8GB 安全值）
- 其余超参保持 Ultralytics 默认，不做多余调整
- 严禁读取 dataset2（data.yaml 只指向 data/lab_ppe）
"""
import time
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "lab_ppe" / "data.yaml"
RUNS = ROOT / "runs" / "e2"

def main():
    t0 = time.time()
    model = YOLO("yolov8s.pt")  # 官方预训练权重，自动下载
    results = model.train(
        data=str(DATA),
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=8,
        project=str(RUNS),
        name="labppe_v8s",
        exist_ok=True,
        seed=42,
        verbose=True,
    )
    dt = time.time() - t0
    print(f"\n[E2] 训练完成，总耗时 {dt/60:.1f} 分钟")
    print(f"[E2] save_dir: {results.save_dir}")
    print(f"[E2] best: {Path(results.save_dir) / 'weights' / 'best.pt'}")

if __name__ == "__main__":
    main()
