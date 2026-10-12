# -*- coding: utf-8 -*-
"""
目标关联层：Person-PPE 空间匹配模块。

解决"画面中多人，PPE 归属哪个人员"的问题。
采用简单可解释的方法：
1. 优先：PPE 检测框中心点落在哪个 person 框内 → 归属该人员
2. 兜底：若中心点不在任何 person 框内，归属"中心点距离最近且距离小于
   该 person 框对角线长度 0.5 倍"的人员（避免远处误归属）

第一阶段不使用姿态估计 / tracking，保持逻辑独立、可解释、可替换。
"""
from dataclasses import dataclass, field

from ai.detector import Detection, CLASS_CN, _iou


@dataclass
class PersonState:
    """单个人的检测与 PPE 佩戴状态"""
    person_id: int                      # 在当前帧内的编号
    bbox: Detection                     # person 检测框
    ppe: dict = field(default_factory=dict)   # {ppe_type: Detection}
    in_roi: bool = False                # 是否在实验操作区（由外部 ROI 模块填写）
    area_name: str = ""                 # 所在区域名称
    virtual: bool = False               # True=无 person 模型时的虚拟聚合人员

    @property
    def has_ppe(self) -> dict:
        """返回各 PPE 是否佩戴（True=检测到）"""
        return {k: (k in self.ppe) for k in ["mask", "gloves", "lab_coat", "goggles", "helmet"]}

    def to_dict(self):
        return {
            "person_id": self.person_id,
            "bbox": self.bbox.to_dict()["bbox"],
            "in_roi": self.in_roi,
            "area": self.area_name,
            "ppe": {k: v.to_dict() for k, v in self.ppe.items()},
        }


def _center_in_box(px, py, box: Detection) -> bool:
    return box.x1 <= px <= box.x2 and box.y1 <= py <= box.y2


def _diag(box: Detection) -> float:
    return ((box.x2 - box.x1) ** 2 + (box.y2 - box.y1) ** 2) ** 0.5


def _near_duplicate(a: Detection, b: Detection,
                    iou_thr: float = 0.85, contain_thr: float = 0.8) -> bool:
    """判断两个 person 框是否为「同一人的重复框」。

    必须同时满足：IoU 很高（>=0.85）且小框大部分被大框包含。
    只用 IoU>=0.5 会误杀拥抱/并肩等真实重叠的两个人
    （2026-10-11 网页实测：双人拥抱 IoU≈0.52 被误合并为 1 人）。
    """
    iou = _iou(a, b)
    if iou < iou_thr:
        return False
    iw = min(a.x2, b.x2) - max(a.x1, b.x1)
    ih = min(a.y2, b.y2) - max(a.y1, b.y1)
    inter = max(0.0, iw) * max(0.0, ih)
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    smaller = min(area_a, area_b)
    return smaller > 0 and inter / smaller >= contain_thr


def _dedup_persons(persons: list[Detection]) -> list[Detection]:
    """合并「同一人」的近重复 person 框（保留置信度最高者）。

    背景：YOLO/NMS 在密集人群可能对同一人输出两个几乎相同的框，
    若不去重会被当成两个人（其中一人无 PPE → 误报违规）。
    判定标准必须从严（近重复才合并）：真实拥挤场景中两人框
    IoU 达 0.5+ 很常见，宽松阈值会把真人合并掉。
    """
    if len(persons) <= 1:
        return list(persons)
    kept: list[Detection] = []
    for p in sorted(persons, key=lambda d: d.conf, reverse=True):
        if not any(_near_duplicate(p, k) for k in kept):
            kept.append(p)
    return kept


def match_person_ppe(detections: list[Detection],
                     max_dist_ratio: float = 0.5) -> list[PersonState]:
    """
    输入：一帧内的全部 Detection（含 person 与各类 PPE）
    输出：PersonState 列表（每人绑定了属于自己的 PPE）
    """
    persons = _dedup_persons([d for d in detections if d.cls_name == "person"])
    ppes = [d for d in detections if d.cls_name != "person"]

    states = [PersonState(person_id=i + 1, bbox=p) for i, p in enumerate(persons)]

    # 4 类自制模型（E2/E4）不输出 person：把全部 PPE 聚合到一个虚拟人员，
    # bbox 取所有 PPE 的外包络，保证规则引擎 / 标注 / 统计链路照常工作
    if not persons and ppes:
        virt = Detection(
            cls_name="person", raw_cls="virtual-person", conf=1.0,
            x1=min(d.x1 for d in ppes), y1=min(d.y1 for d in ppes),
            x2=max(d.x2 for d in ppes), y2=max(d.y2 for d in ppes),
        )
        states = [PersonState(person_id=1, bbox=virt, virtual=True)]

    for ppe in ppes:
        px, py = ppe.center
        # 规则1：中心点在 person 框内
        owner = None
        for st in states:
            if _center_in_box(px, py, st.bbox):
                if owner is None or st.bbox.area < owner.bbox.area:
                    owner = st  # 多人重叠时归属更小（更可能是本人）的框
        # 规则2：扩张框兜底 —— person 框常切掉伸出的手臂/腿部（弯腰操作显微镜、
        # 伸手取试剂等），手套框在原始框外但显然属于该人。
        # 把 person 框向四周扩张 40%，PPE 中心落在扩张框内即归属；
        # 多人扩张框重叠时归属面积最小（最贴近 PPE）的人。
        if owner is None and states:
            def expand_box(st, ratio: float = 0.4):
                b = st.bbox
                w, h = b.x2 - b.x1, b.y2 - b.y1
                return (b.x1 - w * ratio, b.y1 - h * ratio,
                        b.x2 + w * ratio, b.y2 + h * ratio)

            cands = [st for st in states
                     if expand_box(st)[0] <= px <= expand_box(st)[2]
                     and expand_box(st)[1] <= py <= expand_box(st)[3]]
            if cands:
                owner = min(cands, key=lambda st: st.bbox.area)
        # 规则3：最近人员兜底（PPE 离所有人都很远时，仍按距离兜底）
        if owner is None and states:
            def dist(st):
                cx, cy = st.bbox.center
                return ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            nearest = min(states, key=dist)
            if dist(nearest) <= max_dist_ratio * _diag(nearest.bbox):
                owner = nearest
        if owner is not None:
            # 同类 PPE 保留置信度更高的
            old = owner.ppe.get(ppe.cls_name)
            if old is None or ppe.conf > old.conf:
                owner.ppe[ppe.cls_name] = ppe
    return states
