# -*- coding: utf-8 -*-
"""
AI 检测层：封装 YOLO 模型，输出统一的目标检测结果。

底层模型：SH17 数据集官方发布的 YOLOv8s 权重（ultralytics 框架）。
本模块负责：
1. 加载预训练模型
2. 将 SH17 的 17 个原始类别映射为系统业务类别
   （person / mask / gloves / lab_coat / goggles / helmet）
3. 输出统一的 Detection 结构供后续模块使用
"""
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# ---------- 类别映射 ----------
# SH17 原始类别（以模型文件内嵌 names 为准）:
# person, head, face, glasses, face-mask-medical, face-guard, ear,
# ear-mufs, hands, gloves, foot, shoes, safety-vest, tools, helmet,
# medical-suit, safety-suit
#
# 系统业务类别（第一版范围）:
# person, mask, gloves, lab_coat, goggles, helmet
CLASS_ALIAS = {
    "person": "person",
    "head": None, "face": None, "ear": None, "ear-mufs": None,
    "earmuffs": None, "hands": None, "foot": None, "tools": None,
    "face-mask": "mask",
    "face-mask-medical": "mask",
    "facemask": "mask",
    "mask": "mask",
    "gloves": "gloves",
    "glove": "gloves",
    "glasses": "goggles",
    "face-guard": "goggles",
    "helmet": "helmet",
    "safety-vest": None,
    "medical-suit": "lab_coat",
    "safety-suit": "lab_coat",
    "shoes": None,
}

# 系统支持的业务类别
SYSTEM_CLASSES = ["person", "mask", "gloves", "lab_coat", "goggles", "helmet"]

# 业务类别中文名称
CLASS_CN = {
    "person": "人员", "mask": "口罩", "gloves": "手套",
    "lab_coat": "实验服", "goggles": "护目镜", "helmet": "安全帽",
}


@dataclass
class Detection:
    """一次检测结果的统一结构（坐标为像素绝对值 xyxy）"""
    cls_name: str          # 系统业务类别
    raw_cls: str           # 模型原始类别名
    conf: float            # 置信度
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self):
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def area(self):
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def to_dict(self):
        return {
            "cls": self.cls_name, "cls_cn": CLASS_CN.get(self.cls_name, self.cls_name),
            "conf": round(self.conf, 3),
            "bbox": [round(self.x1, 1), round(self.y1, 1),
                     round(self.x2, 1), round(self.y2, 1)],
        }


def _normalize(name: str) -> str:
    return str(name).strip().lower().replace("_", "-").replace(" ", "-")


def map_class(raw_name: str) -> str | None:
    """把模型原始类别名映射为系统业务类别，不支持的返回 None"""
    key = _normalize(raw_name)
    if key in CLASS_ALIAS:
        return CLASS_ALIAS[key]
    # 兜底：子串匹配
    if "person" in key:
        return "person"
    if "mask" in key:
        return "mask"
    if "glove" in key:
        return "gloves"
    if "glass" in key or "goggle" in key:
        return "goggles"
    if "helmet" in key or "hardhat" in key:
        return "helmet"
    if "suit" in key or "coat" in key:
        return "lab_coat"
    return None


class PPEDetector:
    """YOLO PPE 检测器"""

    def __init__(self, weights: str, conf_threshold: float = 0.35, device: str = "auto"):
        from ultralytics import YOLO  # 延迟导入，加快模块加载
        self.weights_path = str(weights)
        self.model = YOLO(self.weights_path)
        self.conf_threshold = conf_threshold
        if device == "auto":
            import torch
            self.device = 0 if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        # 记录模型原始类别
        self.raw_names = {int(k): str(v) for k, v in self.model.names.items()}
        self.supported_map = {i: map_class(v) for i, v in self.raw_names.items()}

    @property
    def device_name(self) -> str:
        if isinstance(self.device, int):
            import torch
            return torch.cuda.get_device_name(self.device)
        return str(self.device)

    def detect(self, frame: np.ndarray, conf: float | None = None) -> list[Detection]:
        """对单帧图像（BGR ndarray）执行检测，返回业务类别 Detection 列表"""
        results = self.model.predict(
            frame, conf=conf or self.conf_threshold, device=self.device,
            verbose=False, imgsz=640,
        )
        detections: list[Detection] = []
        if not results:
            return detections
        r = results[0]
        boxes = r.boxes
        if boxes is None:
            return detections
        for i in range(len(boxes)):
            cls_idx = int(boxes.cls[i].item())
            raw = self.raw_names.get(cls_idx, str(cls_idx))
            mapped = self.supported_map.get(cls_idx)
            if mapped is None:
                continue  # 非业务类别（head/ear/tools 等），跳过
            xyxy = boxes.xyxy[i].tolist()
            detections.append(Detection(
                cls_name=mapped, raw_cls=raw,
                conf=float(boxes.conf[i].item()),
                x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3],
            ))
        return detections

    @staticmethod
    def filter_by_class(detections, cls_name):
        return [d for d in detections if d.cls_name == cls_name]
