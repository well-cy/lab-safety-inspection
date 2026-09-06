# -*- coding: utf-8 -*-
"""
ROI / 区域判断模块。

第一阶段采用归一化矩形 ROI（在设置页面配置，取值 0~1），
同一张画面可配置多个区域；第一个命中的区域作为人员所在区域。
人员判定依据：person 检测框中心点是否落入区域矩形内。
"""
from dataclasses import dataclass


@dataclass
class AreaROI:
    """一个受控区域的定义（归一化坐标 0~1 的矩形）"""
    area_id: int
    name: str            # 区域名称，例如 "实验操作区"
    x1: float
    y1: float
    x2: float
    y2: float
    required_ppe: list   # 该区域要求佩戴的 PPE 列表, 如 ["mask","gloves","lab_coat"]

    def contains_center(self, det) -> bool:
        """person 检测框中心点是否落入该区域（det 需为归一化坐标）"""
        cx = det.x1 + (det.x2 - det.x1) / 2
        cy = det.y1 + (det.y2 - det.y1) / 2
        return self.x1 <= cx <= self.x2 and self.y1 <= cy <= self.y2


def normalize_detection(det, frame_w: int, frame_h: int):
    """把像素坐标的检测框转换为归一化坐标（浅拷贝替换坐标值）"""
    from ai.detector import Detection
    return Detection(
        cls_name=det.cls_name, raw_cls=det.raw_cls, conf=det.conf,
        x1=det.x1 / frame_w, y1=det.y1 / frame_h,
        x2=det.x2 / frame_w, y2=det.y2 / frame_h,
    )


def locate_person(person_det, frame_w: int, frame_h: int,
                  areas: list[AreaROI]) -> tuple[bool, AreaROI | None]:
    """
    判断人员处于哪个区域。
    返回 (in_roi, area)。in_roi=False 表示在普通区域（无强制 PPE 要求）。
    """
    norm = normalize_detection(person_det, frame_w, frame_h)
    for area in areas:
        if area.contains_center(norm):
            return True, area
    return False, None
