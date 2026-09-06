# -*- coding: utf-8 -*-
"""
生成 MVP 演示用测试视频。

说明：第一阶段尚无项目组自采实验室视频，此脚本用公开测试图片合成一段
模拟视频：人员在画面中从"普通区域"移动进入"实验操作区"，
用于演示 ROI 触发 + 违规判断 + 截图 + 事件生成。

正式验证阶段将使用项目组自采实验室场景数据进行测试。
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "data" / "test"
OUT = TEST_DIR / "demo_lab_video.mp4"


def load_img(p):
    img = cv2.imread(str(p))
    if img is None:
        raise FileNotFoundError(p)
    return img


def make_video(image_paths, out_path=OUT, fps=25, duration_s=16, size=(1280, 720)):
    """将图片依次粘贴到移动窗口中，模拟人员走入实验操作区"""
    W, H = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
    total_frames = fps * duration_s
    n_img = len(image_paths)
    seg = total_frames // n_img

    for i, path in enumerate(image_paths):
        img = load_img(path)
        img = cv2.resize(img, (480, 640))  # 竖版人像块
        for f in range(seg):
            canvas = np.full((H, W, 3), 235, dtype=np.uint8)
            # 背景网格模拟实验室
            for gx in range(0, W, 80):
                cv2.line(canvas, (gx, 0), (gx, H), (205, 205, 205), 1)
            for gy in range(0, H, 80):
                cv2.line(canvas, (0, gy), (W, gy), (205, 205, 205), 1)
            cv2.putText(canvas, "Laboratory (simulated demo)", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 120, 120), 2)
            # 人物块从画面左侧（普通区域）移动到中部（实验操作区）
            t = f / max(seg - 1, 1)
            x = int(40 + t * 400)
            y = int((H - 640) / 2)
            # 简单阴影
            cv2.rectangle(canvas, (x + 6, y + 6), (x + 486, y + 646), (180, 180, 180), -1)
            canvas[y:y + 640, x:x + 480] = img
            writer.write(canvas)
    writer.release()
    print(f"OK: {out_path} ({total_frames} frames @ {fps}fps)")
    return out_path


if __name__ == "__main__":
    imgs = [
        TEST_DIR / "t03_nih_pipetting.png",     # 穿实验服+手套 → 进入区域后仍可能缺口罩
        TEST_DIR / "t05_pcr_scientist.jpg",     # 全套 PPE
        TEST_DIR / "t08_grad_chem_lab.jpg",     # 学生做实验
    ]
    imgs = [p for p in imgs if p.exists()]
    if not imgs:
        print("没有找到测试图片，请先运行下载脚本")
        sys.exit(1)
    make_video(imgs)
