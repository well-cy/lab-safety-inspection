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
    "medical-suit": "coverall",
    "safety-suit": "coverall",
    "coverall": "coverall",
    "coveralls": "coverall",
    "shoes": None,
}

# 系统支持的业务类别
SYSTEM_CLASSES = ["person", "mask", "gloves", "lab_coat", "goggles", "coverall", "helmet"]

# 业务类别中文名称
CLASS_CN = {
    "person": "人员", "mask": "口罩", "gloves": "手套",
    "lab_coat": "实验服", "goggles": "护目镜", "helmet": "安全帽",
    "coverall": "连体防护服",
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
    # 兜底：子串匹配（顺序重要，coverall 必须排在 lab_coat 之前判断）
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
    if "coverall" in key:
        return "coverall"
    if "suit" in key or "coat" in key:
        return "lab_coat"
    return None


def _iou(a, b) -> float:
    """两个框的 IoU（xyxy 像素坐标）"""
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    return inter / (area_a + area_b - inter)


class PersonDetector:
    """COCO 预训练人员检测器（person 类锚点）

    自制 4 类 PPE 模型不含 person 类，规则层只能把全部 PPE 聚合成 1 个
    "虚拟人员"，导致：多人场景被合并成 1 人、PPE 归属错乱、无人物的
    产品图也报"1 人"。本检测器提供真实 person 框作为锚点，交给
    ppe_matcher 做 Person-PPE 空间归属，从而支持逐人合规判定。

    刻意与 PPE 判定解耦：只输出 person，不参与任何 PPE 类别判断。
    """

    _PERSON_CLASS_ID = 0  # COCO 类别表中 person = 0
    # 人员框合理性过滤：手/手臂等局部被误认成 person 时框极小（实测 1~3% 画面，
    # 同期真人 42%），这类碎片若当锚点会凭空造出多个虚拟人 + 假违规
    _MIN_AREA_RATIO = 0.03        # 绝对下限：不足画面 3% 的框丢弃
    _MIN_REL_TO_LARGEST = 0.15    # 相对下限：不足最大人员框 15% 的框丢弃（保留远景小人）

    def __init__(self, weights: str, conf_threshold: float = 0.35,
                 device: str = "auto", imgsz: int = 640):
        from ultralytics import YOLO
        self.weights_path = str(weights)
        self.model = YOLO(self.weights_path)
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        if device == "auto":
            import torch
            self.device = 0 if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def detect(self, frame: np.ndarray,
               conf: float | None = None) -> list[Detection]:
        """返回画面中的人员 Detection 列表（cls_name 恒为 "person"）"""
        results = self.model.predict(
            frame, conf=conf or self.conf_threshold, classes=[self._PERSON_CLASS_ID],
            device=self.device, verbose=False, imgsz=self.imgsz,
        )
        out: list[Detection] = []
        if not results or results[0].boxes is None:
            return out
        h, w = frame.shape[:2]
        frame_area = float(h * w)
        boxes = results[0].boxes
        cand: list[Detection] = []
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].tolist()
            cand.append(Detection(
                cls_name="person", raw_cls="person",
                conf=float(boxes.conf[i].item()),
                x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3],
            ))
        if not cand:
            return out
        largest = max((d.x2 - d.x1) * (d.y2 - d.y1) for d in cand)
        for d in cand:
            area = (d.x2 - d.x1) * (d.y2 - d.y1)
            if area < self._MIN_AREA_RATIO * frame_area:
                continue  # 碎块（手/臂局部）
            if area < self._MIN_REL_TO_LARGEST * largest:
                continue  # 相对过小，判为同场景内的误检碎片
            out.append(d)
        return out


class PPEDetector:
    """YOLO PPE 检测器"""

    def __init__(self, weights: str, conf_threshold: float = 0.35, device: str = "auto",
                 imgsz: int = 640):
        from ultralytics import YOLO  # 延迟导入，加快模块加载
        self.weights_path = str(weights)
        self.model = YOLO(self.weights_path)
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
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

    def detect(self, frame: np.ndarray, conf: float | None = None,
               tile_recheck: bool = False) -> list[Detection]:
        """对单帧图像（BGR ndarray）执行检测，返回业务类别 Detection 列表

        双尺度：imgsz 与 640 不同时，两个尺度各推理一次取并集（不同目标
        在不同分辨率下置信度差异大，如小目标护目镜需 1280、大画面防毒
        面具在 640 更稳）。
        tile_recheck：额外做 2×2 带重叠切片复检（大幅提升小目标召回，
        如正面透明护目镜），代价是推理耗时约 ×2.5，仅建议图片模式开启。
        """
        scales = sorted({640, self.imgsz})
        detections: list[Detection] = []
        for sz in scales:
            results = self.model.predict(
                frame, conf=conf or self.conf_threshold, device=self.device,
                verbose=False, imgsz=sz,
                iou=0.5,  # 收紧 NMS：默认 0.7 会保留"套住多人"的联合框
            )
            if not results:
                continue
            boxes = results[0].boxes
            if boxes is None:
                continue
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
        # 先对全图结果做合理性过滤，再做切片支持判断：
        # 否则底部鞋子等全图误检会错误地"支持"切片手套准入（②img_ed8a1e3e 连锁误检）
        detections = self._filter_implausible(detections, frame)
        if tile_recheck:
            detections += self._tile_detect(frame, conf, full_dets=detections)
        merged = self._nms_merge(detections) if (len(scales) > 1 or tile_recheck) else detections
        return self._filter_implausible(merged, frame)

    # 全图级合理性过滤：手套是肢体末端，正常框不会超过画面 20%
    # （白大褂袖子/衣料被误认成手套时框可达画面 25%~33%：特写构图 33%、
    #   img_64166c86 整条防护服袖子 24.6%——原 0.25 上限被擦线混过，收紧到 0.20）
    _GLOVES_MAX_AREA_RATIO = 0.20
    # 手套框若整体位于画面底部 ~12%（y1 > 0.88h），实际是鞋/脚（白鞋被认成手套，
    # 如②img_ed8a1e3e 鞋子 0.52）；正常站姿下手很少低到画面最底部
    _GLOVES_BOTTOM_Y = 0.86

    def _filter_implausible(self, detections: list[Detection],
                            frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        out = []
        for d in detections:
            if d.cls_name == "gloves":
                if (d.x2 - d.x1) * (d.y2 - d.y1) > self._GLOVES_MAX_AREA_RATIO * w * h:
                    continue  # 面积超限（袖子/衣料误检）
                # 画面底部（鞋/脚误检）——但与 person 框实质重叠的是"长在身上的手"
                # （2026-10-12 修复：img_89128025 特写构图，主研究员的手在画面
                #   最底部 y1=0.88h 被 0.86 规则误杀，conf 高达 0.637；
                #   原防护案例 img_ed8a1e3e 的鞋误检在换 6 类模型后已不复出现）
                if d.y1 > self._GLOVES_BOTTOM_Y * h \
                        and not self._overlaps_person_body(d, detections):
                    continue
                if self._is_reflection(d, detections, frame, h):
                    continue  # 玻璃台面倒影误检
                # 手套不应占"其所叠 person 框"的 35%+（牛仔裤/衣料误检，
                # 如 lc13_GMG6BHZE45MT 病人蓝牛仔裤被认成手套 0.45，
                # 占该 person 框 45%）——真手再近也很少超过所属人体 1/3
                if self._glove_dominates_person(d, detections):
                    continue
            out.append(d)
        return out

    # 倒影判据：手套框位于画面底部 25%，正上方有横向重叠的护目镜/口罩框
    # （玻璃台面把 PPE 镜像到下方，如 img_104e87bc 台面倒影被认成手套 0.66）
    _REFLECT_BOTTOM_Y = 0.75

    # 手套占所属 person 框面积比上限（lc13_GMG6BHZE45MT 牛仔裤误检 45%）
    _GLOVE_PERSON_RATIO = 0.35

    def _overlaps_person_body(self, d: Detection,
                              detections: list[Detection]) -> bool:
        """手套框与某 person 框实质重叠（≥35% 手套面积）→ 长在身上的手，非鞋/脚。"""
        g_area = (d.x2 - d.x1) * (d.y2 - d.y1)
        if g_area <= 0:
            return False
        for o in detections:
            if o.cls_name != "person":
                continue
            iw = min(d.x2, o.x2) - max(d.x1, o.x1)
            ih = min(d.y2, o.y2) - max(d.y1, o.y1)
            if iw > 0 and ih > 0 and iw * ih >= 0.35 * g_area:
                return True
        return False

    def _glove_dominates_person(self, d: Detection,
                                detections: list[Detection]) -> bool:
        """手套框面积超过与其重叠的最小 person 框的 35% → 衣料/裤料误检。"""
        g_area = (d.x2 - d.x1) * (d.y2 - d.y1)
        for o in detections:
            if o.cls_name != "person":
                continue
            iw = min(d.x2, o.x2) - max(d.x1, o.x1)
            ih = min(d.y2, o.y2) - max(d.y1, o.y1)
            if iw > 0 and ih > 0:
                p_area = (o.x2 - o.x1) * (o.y2 - o.y1)
                if p_area > 0 and g_area >= self._GLOVE_PERSON_RATIO * p_area:
                    return True
        return False

    def _is_reflection(self, d: Detection, detections: list[Detection],
                       frame: np.ndarray, h: float) -> bool:
        if d.y1 <= self._REFLECT_BOTTOM_Y * h:
            return False
        # 真手长在身上：手套框大部分（≥50%）落在某个 person 框内 → 是人体的一部分，
        # 不可能是台面倒影。（2026-10-12 修复：img_17c92cec 4K 竖构图，弯腰伏案者
        # 自己的手在画面底部、正对自己上方的 mask —— 完美满足旧判据，
        # 3 个 conf 0.71~0.87 的真手套被当成"口罩倒影"全部误杀。
        # 真倒影浮在人身体外的台面上，不会整体落在 person 框里。）
        for o in detections:
            if o.cls_name != "person":
                continue
            iw = min(d.x2, o.x2) - max(d.x1, o.x1)
            ih = min(d.y2, o.y2) - max(d.y1, o.y1)
            if iw > 0 and ih > 0 and (iw * ih) / max(d.area, 1e-6) >= 0.5:
                return False
        dw = d.x2 - d.x1
        for o in detections:
            if o.cls_name not in ("goggles", "mask") or o is d:
                continue
            x_overlap = min(d.x2, o.x2) - max(d.x1, o.x1)
            if x_overlap > 0.5 * min(dw, o.x2 - o.x1) and d.y1 > o.y2 + 0.03 * h:
                return True  # 底部 + 正上方有同位 PPE = 台面倒影
        return False

    # 切片复检类与逐类过滤：lab_coat 全图对竖长/特写构图不稳，也纳入切片
    _TILE_CLASSES = {"gloves", "mask", "goggles", "lab_coat"}
    _TILE_MAX_AREA = {"gloves": 0.08, "mask": 0.08, "goggles": 0.08,
                      "lab_coat": 0.75}  # 框占切片面积上限（白墙填满切片的误检≈100%）
    _TILE_GOGGLES_MAX_AR = 1.5   # 护目镜框宽高比上限：真护目镜≈0.65（含绑带偏高），普通眼镜误检≈2.2（扁宽）

    def _tile_detect(self, frame: np.ndarray, conf: float | None = None,
                     overlap: float = 0.25,
                     full_dets: list[Detection] | None = None) -> list[Detection]:
        """2×2 带重叠切片推理，坐标平移回原图后返回（不再做 NMS，由上层统一合并）

        gloves 特殊分级准入：
        - 全图已有手套检出（full_has_gloves）→ 切片结果直接收；
        - 否则需"强证据"：conf≥0.65 且框占切片 6%~30% 且框内非肤色
          （裸手放大后易被误检：老照片持械手 0.68、病床裸手 0.66、
          ③图裸手 0.52——都是肤色+小框；真彩色手套如 teal/蓝/银箔
          置信度高、框大、非肤色，如 teal 0.79、蓝手套 0.85）。
        """
        h, w = frame.shape[:2]
        th, tw = h // 2, w // 2
        full_has_gloves = bool(full_dets) and any(d.cls_name == "gloves" for d in full_dets)
        out: list[Detection] = []
        for ty in (0, h - th):
            for tx in (0, w - tw):
                tile = frame[ty:ty + th, tx:tx + tw]
                results = self.model.predict(
                    tile, conf=conf or self.conf_threshold, device=self.device,
                    verbose=False, imgsz=self.imgsz,
                )
                if not results or results[0].boxes is None:
                    continue
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    cls_idx = int(boxes.cls[i].item())
                    mapped = self.supported_map.get(cls_idx)
                    if mapped is None or mapped not in self._TILE_CLASSES:
                        continue
                    xyxy = boxes.xyxy[i].tolist()
                    bw, bh = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
                    area_ratio = bw * bh / (tw * th)
                    if mapped == "gloves":
                        if area_ratio > self._GLOVES_TILE_MAX_AREA:
                            continue  # 放大切片里的超大框 = 衣料误检（坏图 0.87@33%），丢弃
                        if not full_has_gloves:
                            if (float(boxes.conf[i].item()) < self._GLOVES_TILE_MIN_CONF
                                    or area_ratio < self._GLOVES_TILE_MIN_AREA
                                    or self._skin_ratio(tile, xyxy) > self._GLOVES_SKIN_MAX):
                                continue  # 弱证据 + 肤色 = 裸手误检，丢弃
                    elif area_ratio > self._TILE_MAX_AREA[mapped]:
                        continue  # 放大切片里的超大框 = 误检，丢弃
                    if mapped == "goggles" and bw / max(bh, 1e-6) > self._TILE_GOGGLES_MAX_AR:
                        continue  # 扁宽框 = 普通眼镜误检，丢弃
                    out.append(Detection(
                        cls_name=mapped,
                        raw_cls=self.raw_names.get(cls_idx, str(cls_idx)),
                        conf=float(boxes.conf[i].item()),
                        x1=xyxy[0] + tx, y1=xyxy[1] + ty,
                        x2=xyxy[2] + tx, y2=xyxy[3] + ty,
                    ))
        return out

    # 切片手套"强证据"准入参数（全图无手套支持时）
    _GLOVES_TILE_MAX_AREA = 0.30   # 真手套可到 26%（⑤蓝手套），衣料误检 33% 被拦
    _GLOVES_TILE_MIN_CONF = 0.65   # 裸手误检普遍 ≤0.68 且多为 0.5 上下
    _GLOVES_TILE_MIN_AREA = 0.06   # 裸手小碎框 1%~4%；真手套在切片里 ≥8%
    _GLOVES_SKIN_MAX = 0.35        # 框内肤色像素占比上限：裸手高、彩色/银色手套接近 0

    @staticmethod
    def _skin_ratio(tile: np.ndarray, xyxy: list[float]) -> float:
        """框内（中心 70% 区域）YCrCb 肤色像素占比，用于区分裸手与真实手套"""
        h, w = tile.shape[:2]
        x1, y1, x2, y2 = (max(0, int(v)) for v in xyxy)
        roi = tile[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        ch, cw = roi.shape[:2]
        roi = roi[int(ch * 0.15):int(ch * 0.85) or ch, int(cw * 0.15):int(cw * 0.85) or cw]
        if roi.size == 0:
            return 0.0
        ycc = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        skin = ((ycc[:, :, 0] > 40) & (ycc[:, :, 1] >= 135) & (ycc[:, :, 1] <= 180)
                & (ycc[:, :, 2] >= 85) & (ycc[:, :, 2] <= 135))
        return float(skin.mean())

    @staticmethod
    def _nms_merge(detections: list[Detection], iou_thr: float = 0.55) -> list[Detection]:
        """类内贪心 NMS：合并多尺度重复框，保留高置信度。

        ⚠️ 必须**按类别各自独立**做抑制（2026-10-12 修复）。

        历史 bug：早期实现只用一个全局 merged 列表做跨类比较
        （`any(_iou(d, m) > iou_thr for m in merged)`），而 merged 里混着
        所有已处理的类别。后果是**穿白大褂的人被自己的 lab_coat 框吃掉** ——
        person 框与 lab_coat 框 IoU 常达 0.56~0.69 > 0.55。实测：
          t09 实验室合影 7 人 → 4 人；t02 砷实验室 1 人 → 0 人；
          t07 雌激素实验 1 人 → 0 人；img_b8ab4fdf 3 人 → 2 人。
        更隐蔽的是外层 `for cls in {d.cls_name ...}` 遍历字符串集合，
        集合迭代顺序受 PYTHONHASHSEED 随机化影响 → **谁抑制谁每次重启都变**，
        表现为「有时人少一个、有时白大褂少一个」的幽灵 bug。
        """
        merged: list[Detection] = []
        by_cls: dict[str, list[Detection]] = {}
        for d in detections:
            by_cls.setdefault(d.cls_name, []).append(d)
        for group in by_cls.values():          # 每类一个独立 kept 列表
            group = sorted(group, key=lambda d: -d.conf)
            kept: list[Detection] = []
            for d in group:
                if any(_iou(d, m) > iou_thr for m in kept):
                    continue                    # 只与同类比较
                kept.append(d)
            merged.extend(kept)
        merged.sort(key=lambda d: -d.conf)
        return merged

    @staticmethod
    def filter_by_class(detections, cls_name):
        return [d for d in detections if d.cls_name == cls_name]
