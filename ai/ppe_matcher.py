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


def _contains_ppe_center(p: Detection, ppes: list[Detection] | None) -> bool:
    """person 框内含 mask/goggles/lab_coat 中心 → 强真人信号（脸部/躯干 PPE）。"""
    if not ppes:
        return False
    for d in ppes:
        if d.cls_name in ("mask", "goggles", "lab_coat") and \
                _center_in_box(d.center[0], d.center[1], p):
            return True
    return False


def _is_fragment(p: Detection, others: list[Detection],
                 ppes: list[Detection] | None = None) -> bool:
    """碎片框判定：person 框只是某人的一小部分（手臂/头部误检）。

    判据①：中心点落在另一个面积 ≥2 倍且置信度更高的 person 框内。
    （宿主必须更高置信：若"宿主"是低置信联合框而 p 是高置信真人，
    应由 _is_contained_lowconf 淘汰联合框，而不是反过来杀掉真人。
    且真人框内通常含 mask/goggles 等面部 PPE —— 有 PPE 豁免。）
    判据②：面积不足同帧最大 person 框的 3%（极远小人/碎片）。
    """
    if not others:
        return False
    if not _contains_ppe_center(p, ppes):
        for o in others:
            if o.conf <= p.conf:
                continue
            if o.area >= 2.0 * p.area and _center_in_box(p.center[0], p.center[1], o):
                return True
    largest = max(o.area for o in others)
    return p.area < 0.03 * largest


def _is_contained_lowconf(p: Detection, others: list[Detection]) -> bool:
    """「被包含的低置信框」判定（联合框猎手）。

    场景：联合框比真人框更大、置信度更低（套住 1~2 个真人），
    碎片判定只检查"小框是否被大框包含"轮不到它——因为它是大框。
    本判定反向：若 p 的框**包含**某个中心在 p 内、面积 ≥2 倍差的
    更高置信框（即 p 套着一个更可信的真人大框），p 大概率是联合框。
    判据：存在更高置信框 o，o.center 在 p 内、o 被 p 包含 ≥80%、
    且 p.area >= 2.0 * o.area。
    拥抱场景联合框与真人比仅 1.7 倍 → 不会误杀。
    """
    for o in others:
        if o.conf <= p.conf or o.area == 0:
            continue
        iw = min(p.x2, o.x2) - max(p.x1, o.x1)
        ih = min(p.y2, o.y2) - max(p.y1, o.y1)
        inter = max(0.0, iw) * max(0.0, ih)
        if (p.area >= 2.0 * o.area
                and _center_in_box(o.center[0], o.center[1], p)
                and inter / o.area >= 0.8):
            return True
    return False


def _is_union_box(p: Detection, others: list[Detection]) -> bool:
    """联合框判定：一个框"套住"了 >=2 个更高置信度的真人框（YOLO 密集人群产物）。

    判据：>=2 个 conf 比 p 高 0.1+ 的 person 框与 p 的交集合计覆盖 p 面积 50%+。
    真人框不会被多个更高置信度框大面积覆盖（那样 NMS/真值都会偏向保留大框）。
    """
    covered = 0.0
    count = 0
    for o in others:
        if o.conf < p.conf + 0.1:
            continue
        iw = min(p.x2, o.x2) - max(p.x1, o.x1)
        ih = min(p.y2, o.y2) - max(p.y1, o.y1)
        inter = max(0.0, iw) * max(0.0, ih)
        if inter > 0:
            covered += inter
            count += 1
    return count >= 2 and p.area > 0 and covered >= 0.5 * p.area


def _dedup_persons(persons: list[Detection],
                   ppes: list[Detection] | None = None) -> list[Detection]:
    """person 框清理：近重复合并 + 碎片框/联合框剔除（保留置信度最高者）。

    背景：YOLO 在密集人群会输出三类伪 person 框——
    ① 同一人近重复框（NMS 漏抑）② 肢体/头部碎片框 ③ 套住多人的联合框。
    伪框不清理会把真人挤掉 PPE 归属或凭空多出"NO PPE"幻影人员。
    判定从严迭代删除：删掉一个后其余框的关系可能变化，故循环直到稳定。
    """
    if len(persons) <= 1:
        return list(persons)
    kept = sorted(persons, key=lambda d: d.conf, reverse=True)
    changed = True
    while changed and len(kept) > 1:
        changed = False
        result: list[Detection] = []
        for i, p in enumerate(kept):
            higher = kept[:i]  # 置信度更高的已保留框
            if (_is_fragment(p, kept[:i] + kept[i + 1:], ppes)
                    or _is_contained_lowconf(p, higher)
                    or _is_union_box(p, higher)):
                changed = True
                continue
            result.append(p)
        kept = result
    # 最后做近重复合并（保留 conf 最高者）
    out: list[Detection] = []
    for p in kept:
        if not any(_near_duplicate(p, k) for k in out):
            out.append(p)
    return out


def match_person_ppe(detections: list[Detection],
                     max_dist_ratio: float = 0.5) -> list[PersonState]:
    """
    输入：一帧内的全部 Detection（含 person 与各类 PPE）
    输出：PersonState 列表（每人绑定了属于自己的 PPE）
    """
    persons = _dedup_persons([d for d in detections if d.cls_name == "person"],
                             ppes=[d for d in detections if d.cls_name != "person"])
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

    # 各类 PPE 的解剖学合理带：PPE 中心相对 person 框顶部的比例（rel_y）。
    # 口罩/护目镜只可能出现在头部（框顶部 ~45%）；手套在躯干活动范围；
    # 站立者的"手套"落到框底部 12%（脚部）几乎必然是错检或他人的手。
    # 用于多人框重叠时消歧：坐姿者框往往高大（含椅子），其 PPE 中心
    # 也会落进旁边站立者的小框内且贴近底部 —— 旧逻辑"取面积最小"
    # 会把坐姿者的 PPE 全部判给站立者（img_d020e947 实测案例）。
    _PPE_BAND = {
        "mask": (0.0, 0.45),
        "goggles": (0.0, 0.45),
        "helmet": (0.0, 0.35),
        "gloves": (0.08, 0.88),
        "lab_coat": (0.05, 0.92),
        "coverall": (0.05, 0.92),
    }

    def _cand(st):
        """返回 (exact中心在框内, plausible解剖学合理, st)"""
        b = st.bbox
        w, h = b.x2 - b.x1, b.y2 - b.y1
        exact = b.x1 <= px <= b.x2 and b.y1 <= py <= b.y2
        expanded = (b.x1 - w * 0.4) <= px <= (b.x2 + w * 0.4) \
            and (b.y1 - h * 0.4) <= py <= (b.y2 + h * 0.4)
        lo, hi = _PPE_BAND.get(ppe.cls_name, (0.0, 1.0))
        rel = (py - b.y1) / h if h > 0 else 0.5
        plausible = lo <= rel <= hi
        return (exact or expanded), exact, plausible, st

    for ppe in ppes:
        px, py = ppe.center
        owner = None
        # 统一候选集：中心落在 person 框内（规则1）或 40% 扩张框内（规则2）。
        # 优先级：解剖学合理 > 精确包含 > 面积最小。
        # （旧逻辑分两步且"面积最小"优先，导致坐姿人员的 PPE 被
        #   重叠的站立者小框整体抢走。）
        cands = [c for c in (_cand(st) for st in states) if c[0]]
        if cands:
            best = max(cands, key=lambda c: (c[2], c[1], -c[3].bbox.area))
            owner = best[3]
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
