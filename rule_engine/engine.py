# -*- coding: utf-8 -*-
"""
安全规则引擎（Safety Rule Engine）。

输入：
  - PersonState 列表（已由 ppe_matcher 完成人员-PPE 绑定、已由 roi 模块标记区域）
  - 各区域的 PPE 要求（从数据库 safety_rules 读取，可配置，非硬编码）

输出：
  - ViolationEvent 列表

违规等级规则（第一版）：
  - 正常：区域内所有要求的 PPE 均已佩戴
  - 一般违规：缺少 1 种要求的 PPE
  - 严重违规：同时缺少 2 种及以上要求的 PPE
  - 人员在普通区域（非受控区域）：不做 PPE 强制判断，视为"不评估"
"""
from dataclasses import dataclass, field
from datetime import datetime

from ai.detector import CLASS_CN

SEVERITY_NONE = "正常"
SEVERITY_MINOR = "一般违规"
SEVERITY_MAJOR = "严重违规"

VIOLATION_LABEL = {
    "mask": "未佩戴口罩",
    "gloves": "未佩戴手套",
    "lab_coat": "未穿实验服",
    "goggles": "未佩戴护目镜",
    "helmet": "未佩戴安全帽",
}


@dataclass
class ViolationEvent:
    """一次违规事件"""
    person_id: int
    area_name: str                 # 所在区域
    violation_types: list          # 违规类型列表（中文标签）
    missing_ppe: list              # 缺失的 PPE 英文类别
    severity: str                  # 正常/一般违规/严重违规
    timestamp: str                 # 事件时间（系统时间）
    frame_time: float = 0.0        # 视频内时间（秒），图片为 0
    screenshot: str = ""           # 截图相对路径
    person_bbox: list = field(default_factory=list)

    @property
    def is_violation(self) -> bool:
        return self.severity in (SEVERITY_MINOR, SEVERITY_MAJOR)

    def to_dict(self):
        return {
            "person_id": self.person_id,
            "area": self.area_name,
            "violation_types": self.violation_types,
            "severity": self.severity,
            "timestamp": self.timestamp,
            "frame_time": round(self.frame_time, 2),
            "screenshot": self.screenshot,
        }


class SafetyRuleEngine:
    """可配置的安全规则引擎"""

    def __init__(self, default_severity_minor: int = 1):
        # 缺少 N 种为一般违规，超过为严重违规（可配置）
        self.minor_threshold = default_severity_minor

    def evaluate(self, person_states, frame_time: float = 0.0,
                 now: datetime | None = None) -> list[ViolationEvent]:
        """
        对一帧内所有人员进行规则判断。
        person_states: PersonState 列表（in_roi / area_name 已由上游填写）
        """
        now = now or datetime.now()
        events = []
        for st in person_states:
            if not st.in_roi:
                # 普通区域：不做强制 PPE 判断
                continue
            required = list(st.area_required) if hasattr(st, "area_required") else []
            if not required:
                required = []  # 区域未配置规则则不评估
            missing = [p for p in required if p not in st.ppe]
            if not missing:
                severity = SEVERITY_NONE
            elif len(missing) <= self.minor_threshold:
                severity = SEVERITY_MINOR
            else:
                severity = SEVERITY_MAJOR
            events.append(ViolationEvent(
                person_id=st.person_id,
                area_name=st.area_name,
                violation_types=[VIOLATION_LABEL.get(m, m) for m in missing],
                missing_ppe=missing,
                severity=severity,
                timestamp=now.strftime("%Y-%m-%d %H:%M:%S"),
                frame_time=frame_time,
                person_bbox=[st.bbox.x1, st.bbox.y1, st.bbox.x2, st.bbox.y2],
            ))
        return events


def annotate_person_states(person_states, areas, frame_w, frame_h):
    """
    把 ROI 判断结果回填到 PersonState 上（in_roi / area_name / area_required）。
    areas: AreaROI 列表
    """
    from ai.roi import locate_person
    for st in person_states:
        in_roi, area = locate_person(st.bbox, frame_w, frame_h, areas)
        st.in_roi = in_roi
        st.area_name = area.name if area else "普通区域"
        st.area_required = area.required_ppe if area else []
    return person_states
