# -*- coding: utf-8 -*-
"""
视频处理层：图片 / 视频 / 摄像头 统一处理管线。

管线：OpenCV 读帧 → YOLO 检测 → Person-PPE 匹配 → ROI 区域判断
      → 规则引擎 → 画面标注 → 违规截图 → 违规事件
"""
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ai.detector import PPEDetector, Detection, CLASS_CN
from ai.ppe_matcher import match_person_ppe
from ai.roi import AreaROI
from rule_engine.engine import (
    SafetyRuleEngine, ViolationEvent, annotate_person_states,
    SEVERITY_MINOR, SEVERITY_MAJOR, SEVERITY_NONE,
)

# BGR 颜色
COLOR_OK = (0, 200, 0)
COLOR_MINOR = (0, 140, 255)      # 橙色：一般违规
COLOR_MAJOR = (0, 0, 255)        # 红色：严重违规
COLOR_PPE = (200, 200, 0)
COLOR_ROI = (255, 180, 0)

VIOLATION_COOLDOWN_S = 10.0   # 同一组违规的截图冷却时间（秒，视频内时间）


def _cn(name):
    return CLASS_CN.get(name, name)


def draw_annotations(frame, person_states, events, areas):
    """在画面上绘制 ROI、person 框、PPE 框、违规标注"""
    h, w = frame.shape[:2]
    # ROI 区域
    for area in areas:
        p1 = (int(area.x1 * w), int(area.y1 * h))
        p2 = (int(area.x2 * w), int(area.y2 * h))
        cv2.rectangle(frame, p1, p2, COLOR_ROI, 2)
        cv2.putText(frame, f"{area.name}", (p1[0] + 4, p1[1] + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_ROI, 2)
        req = "/".join(_cn(p) for p in area.required_ppe)
        cv2.putText(frame, f"required: {req}", (p1[0] + 4, p1[1] + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_ROI, 1)

    event_map = {e.person_id: e for e in events}
    # person 框
    for st in person_states:
        ev = event_map.get(st.person_id)
        if ev is None:
            color, tag = COLOR_OK, "OK"
        elif ev.severity == SEVERITY_MINOR:
            color, tag = COLOR_MINOR, "WARN"
        elif ev.severity == SEVERITY_MAJOR:
            color, tag = COLOR_MAJOR, "VIOLATION"
        else:
            color, tag = COLOR_OK, "OK"
        x1, y1, x2, y2 = map(int, [st.bbox.x1, st.bbox.y1, st.bbox.x2, st.bbox.y2])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # 佩戴摘要
        worn = "+".join(_cn(k) for k in st.ppe.keys()) or "no PPE"
        cv2.putText(frame, f"P{st.person_id} {tag}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, worn, (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        # 违规文字
        if ev is not None and ev.is_violation:
            text = "; ".join(ev.violation_types) + f" [{ev.severity}]"
            cv2.putText(frame, text, (x1, y2 + 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    # PPE 框
    for st in person_states:
        for ppe in st.ppe.values():
            x1, y1, x2, y2 = map(int, [ppe.x1, ppe.y1, ppe.x2, ppe.y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PPE, 1)
            cv2.putText(frame, f"{_cn(ppe.cls_name)} {ppe.conf:.2f}",
                        (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_PPE, 1)
    return frame


class InspectionPipeline:
    """巡检处理管线：封装 detector + matcher + roi + rule engine"""

    def __init__(self, detector: PPEDetector, engine: SafetyRuleEngine | None = None,
                 screenshot_dir="outputs/screenshots"):
        self.detector = detector
        self.engine = engine or SafetyRuleEngine()
        root = Path(__file__).resolve().parent.parent
        self.screenshot_dir = root / screenshot_dir
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

    def analyze_frame(self, frame, areas: list[AreaROI], frame_time=0.0):
        """对单帧执行完整分析，返回 (person_states, events, detections)"""
        h, w = frame.shape[:2]
        dets = self.detector.detect(frame)
        states = match_person_ppe(dets)
        states = annotate_person_states(states, areas, w, h)
        events = self.engine.evaluate(states, frame_time=frame_time)
        return states, events, dets


# ---------------- 图片处理 ----------------

def process_image(pipeline: InspectionPipeline, image_path, areas,
                  save_path=None) -> dict:
    """处理单张图片，返回结果 dict（含标注图、人员状态、违规事件）"""
    t0 = time.perf_counter()
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    states, events, dets = pipeline.analyze_frame(img, areas)
    annotated = draw_annotations(img.copy(), states, events, areas)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # 违规截图
    for ev in events:
        if ev.is_violation:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            rel = f"img_{ts}_p{ev.person_id}.jpg"
            cv2.imwrite(str(pipeline.screenshot_dir / rel), annotated)
            ev.screenshot = rel
    if save_path:
        cv2.imwrite(str(save_path), annotated)

    return {
        "person_states": [s.to_dict() for s in states],
        "events": [e.to_dict() for e in events],
        "detections": [d.to_dict() for d in dets],
        "elapsed_ms": round(elapsed_ms, 1),
        "annotated": annotated,
    }


# ---------------- 视频处理 ----------------

def process_video(pipeline: InspectionPipeline, video_path, areas,
                  output_path, stride=2, save_annotated=True) -> dict:
    """
    处理视频文件。
    stride: 检测帧间隔（每 stride 帧做一次 YOLO 推理，中间帧复用上一次结果，
            保证演示流畅同时提升处理速度）
    运行结束后统计：总帧数、检测帧数、处理 FPS、违规事件（去重后）、标注视频路径。
    """
    t0 = time.perf_counter()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if save_annotated:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps_in, (frame_w, frame_h))

    all_events: list[ViolationEvent] = []
    last_states, last_events, last_dets = [], [], []
    last_result = None            # 最近一次检测的原始帧缓存（避免重复推理绘制错位）
    n_frames = n_infer = 0
    cooldown: dict[tuple, float] = {}   # (违规集合) -> 上次截图的视频时间

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        n_frames += 1
        frame_time = n_frames / fps_in
        if n_frames % stride == 1 or stride == 1:
            states, events, dets = pipeline.analyze_frame(frame, areas, frame_time)
            last_states, last_events, last_dets = states, events, dets
            n_infer += 1
        else:
            # 复用上一帧结果（时间戳刷新）
            states, events = last_states, last_events
            for ev in events:
                ev.frame_time = frame_time

        annotated = draw_annotations(frame.copy(), states, events, areas)

        # 事件收集 + 截图（带冷却去重）
        for ev in events:
            if not ev.is_violation:
                continue
            key = (ev.person_id, tuple(sorted(ev.missing_ppe)))
            if frame_time - cooldown.get(key, -1e9) >= VIOLATION_COOLDOWN_S:
                cooldown[key] = frame_time
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                rel = f"vid_{ts}_p{ev.person_id}.jpg"
                cv2.imwrite(str(pipeline.screenshot_dir / rel), annotated)
                ev.screenshot = rel
                ev.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                all_events.append(ev)

        if writer is not None:
            writer.write(annotated)

    cap.release()
    if writer is not None:
        writer.release()
    elapsed = time.perf_counter() - t0
    return {
        "total_frames": n_frames,
        "inferred_frames": n_infer,
        "video_fps": round(n_frames / elapsed, 1) if elapsed > 0 else 0,
        "elapsed_s": round(elapsed, 1),
        "events": [e.to_dict() for e in all_events],
        "output_video": str(output_path) if save_annotated else "",
        "frame_size": [frame_w, frame_h],
    }
