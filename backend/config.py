# -*- coding: utf-8 -*-
"""
后端集中配置。

对应分工任务「集中管理配置：模型路径、上传目录、阈值、区域规则等」，
同时修复技术债 #1（模型权重路径硬编码在 backend/main.py，换模型必须改源码）。

设计原则
--------
1. **集中**：所有可调项只在本文件定义，其他模块不得再写死路径与阈值。
2. **可覆盖**：全部支持环境变量覆盖（统一前缀 ``LABSAFETY_``），
   因此"换模型"只需设置环境变量，不必修改任何 .py 文件。
3. **零依赖**：只用标准库，不为团队增加安装负担（不需要 pydantic-settings）。
4. **不改默认值**：未设置任何环境变量时，行为与收口前**完全一致**。

覆盖示例
--------
.. code-block:: bash

    # 切到 E4 训练好的模型（技术债 #1 的目标场景）
    export LABSAFETY_MODEL_PATH=~/labppe_v8s_best.pt
    # 提高置信度阈值、强制 CPU
    export LABSAFETY_CONF_THRESHOLD=0.45
    export LABSAFETY_DEVICE=cpu

未在本文件列出的项（如模型原始类别映射 ``CLASS_ALIAS``、系统业务类别
``SYSTEM_CLASSES``）属于**业务契约**，不通过配置暴露，改动需走契约评审。
"""
import os
from pathlib import Path

# 环境变量统一前缀，避免与其他软件冲突
_ENV_PREFIX = "LABSAFETY_"


def _raw(name: str) -> str | None:
    """读取环境变量；空字符串视为未设置"""
    value = os.environ.get(_ENV_PREFIX + name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_str(name: str, default: str) -> str:
    return _raw(name) or default


def _env_int(name: str, default: int) -> int:
    raw = _raw(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(
            f"环境变量 {_ENV_PREFIX + name} 必须为整数，当前值：{raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = _raw(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(
            f"环境变量 {_ENV_PREFIX + name} 必须为数字，当前值：{raw!r}") from exc


def _env_path(name: str, default: Path) -> Path:
    raw = _raw(name)
    return Path(raw).expanduser() if raw else default


class Config:
    """
    集中配置对象。

    实例化时读取环境变量，因此可在进程启动前通过环境变量覆盖各项配置。
    """

    def __init__(self) -> None:
        root = Path(__file__).resolve().parent.parent
        self.ROOT = root

        # ---------- 路径 ----------
        # 默认模型：优先使用自训 5 类 Lab-PPE 模型（person/mask/gloves/lab_coat/goggles），
        # 该模型不存在时回退 SH17 官方权重（17 类，经 CLASS_ALIAS 映射到系统类别）。
        # 仍可用 LABSAFETY_MODEL_PATH 环境变量强制指定任意权重。
        # 说明：模型权重不入 git 仓库（见 .gitignore / DEVELOPMENT_GUIDE），
        #      新成员克隆后若两个权重都没有，启动时会给出 503 + 下载提示。
        _labppe = root / "ai" / "model" / "labppe_v8s.pt"
        _default_model = _labppe if _labppe.exists() else (
            root / "ai" / "model" / "sh17_yolov8s.pt")
        self.MODEL_PATH = _env_path("MODEL_PATH", _default_model)
        self.DB_PATH = _env_path(
            "DB_PATH", root / "backend" / "labsafety.db")
        self.STATIC_DIR = _env_path(
            "STATIC_DIR", root / "backend" / "static")
        self.OUTPUTS_DIR = _env_path(
            "OUTPUTS_DIR", root / "outputs")
        self.UPLOAD_DIR = _env_path(
            "UPLOAD_DIR", root / "data" / "test" / "uploads")
        # 以下三个默认挂载在 OUTPUTS_DIR 下，便于整体搬迁
        self.ANNOTATED_DIR = _env_path(
            "ANNOTATED_DIR", self.OUTPUTS_DIR / "annotated")
        self.VIDEO_DIR = _env_path(
            "VIDEO_DIR", self.OUTPUTS_DIR / "videos")
        self.SCREENSHOT_DIR = _env_path(
            "SCREENSHOT_DIR", self.OUTPUTS_DIR / "screenshots")

        # ---------- 检测（模型推理） ----------
        # 置信度阈值：低于该值的检测框丢弃
        self.CONF_THRESHOLD = _env_float("CONF_THRESHOLD", 0.35)
        # person 类的独立阈值（2026-10-14 方案 C，见 docs/E5_11）。
        # 重叠/遮挡人群场景下，真人的置信度常被 YOLO 的"联合框"压低
        # （实测 img_9491b9ea 中间男生仅 0.290 → 3 人只检出 2 人），
        # 而 PPE 类一旦跟着降阈值会引入假阳性（同图右女被塞假 mask 0.339，
        # 把"严重违规"洗成"一般违规"）。故只对 person 放宽。
        #
        # 取值 0.28 的依据（实测两案例夹逼）：
        #   要救的真人之最低 conf = 0.290（img_9491b9ea 中男）
        #   不想要的低置信框之最高 conf = 0.274（img_8e909ac5 遮挡半身，
        #     检出后 PPE 全空 → 会凭空多一条"三项全违规"假阳性）
        # 0.28 落在两者之间：救回真人且不引入假阳性。
        # ⚠️ 注意此值贴近实测样本，若有新的漏检/误报样本需重新标定。
        # 设为与 CONF_THRESHOLD 相同即恢复旧行为。
        self.CONF_PERSON = _env_float("CONF_PERSON", 0.28)
        # "auto" = 有 CUDA 用 GPU，否则 CPU；也可显式指定 "cpu" / 0 / 1
        self.DEVICE = _env_str("DEVICE", "auto")

        # ---------- 视频处理 ----------
        # 每 stride 帧做一次推理，中间帧复用上次结果
        self.VIDEO_STRIDE = _env_int("VIDEO_STRIDE", 2)
        # 同一组违规的截图冷却时间（秒，视频内时间），用于事件去重
        self.VIOLATION_COOLDOWN_S = _env_float("VIOLATION_COOLDOWN_S", 10.0)
        # 视频后台任务的并发 worker 数。
        # 默认 1：视频处理是 CPU/GPU 密集型，并发跑只会互相拖慢；
        # 多余任务排队等待，避免把机器压垮。
        self.VIDEO_MAX_WORKERS = _env_int("VIDEO_MAX_WORKERS", 1)
        # 后台任务队列的待处理上限（含排队中与处理中）。
        # 防止无限提交同时耗尽内存（任务对象）与磁盘（每个视频任务在提交时
        # 就已落盘一份上传文件），超限时接口返回 429。
        self.MAX_PENDING_TASKS = _env_int("MAX_PENDING_TASKS", 20)

        # 运行期文件清理：默认保留最近 N 天（超出后由清理接口删除）
        self.CLEANUP_KEEP_DAYS = _env_int("CLEANUP_KEEP_DAYS", 7)

        # ---------- 规则引擎 ----------
        # 缺失 PPE 数量 <= 该值判为「一般违规」，超过则「严重违规」
        self.MINOR_THRESHOLD = _env_int("MINOR_THRESHOLD", 1)

        # ---------- 接口默认值 ----------
        self.DEFAULT_LAB_ID = _env_int("DEFAULT_LAB_ID", 1)
        # 新建区域时的默认 PPE 要求（仅作为 API 参数默认值；
        # 数据库中已有区域的规则不受影响，也不会改动数据库种子数据）
        self.DEFAULT_REQUIRED_PPE = _env_str(
            "DEFAULT_REQUIRED_PPE", "mask,gloves,lab_coat")

        # ---------- 业务约束（不通过环境变量暴露）----------
        # 可作为区域 PPE 要求的业务类别；person 是检测主体，不是 PPE。
        # 该列表与 ai/detector.py 的 SYSTEM_CLASSES 对齐，属契约内容，
        # 如需变更请走契约评审，不要用环境变量绕过。
        #
        # 注意：coverall（连体防护服）**故意不列入**。
        # 按 E5.7 口径（Route B'），连体防护服不满足 lab_coat 要求，
        # 它是「未穿实验服」这一违规的**证据类别**，而非区域可配置的必备 PPE。
        # 它出现在检测结果里，但不作为 required_ppe 的可选值。
        self.ALLOWED_PPE = ["mask", "gloves", "lab_coat", "goggles", "helmet"]

    def ensure_dirs(self) -> None:
        """创建运行期需要的目录（幂等）"""
        for d in (self.UPLOAD_DIR, self.ANNOTATED_DIR,
                  self.VIDEO_DIR, self.SCREENSHOT_DIR):
            d.mkdir(parents=True, exist_ok=True)

    def describe(self) -> dict:
        """
        当前生效配置快照，用于诊断与日志。
        不含任何敏感信息（本配置中没有密钥类内容）。
        """
        return {
            "root": str(self.ROOT),
            "model_path": str(self.MODEL_PATH),
            "db_path": str(self.DB_PATH),
            "upload_dir": str(self.UPLOAD_DIR),
            "annotated_dir": str(self.ANNOTATED_DIR),
            "video_dir": str(self.VIDEO_DIR),
            "screenshot_dir": str(self.SCREENSHOT_DIR),
            "conf_threshold": self.CONF_THRESHOLD,
            "conf_person": self.CONF_PERSON,
            "device": self.DEVICE,
            "video_stride": self.VIDEO_STRIDE,
            "violation_cooldown_s": self.VIOLATION_COOLDOWN_S,
            "video_max_workers": self.VIDEO_MAX_WORKERS,
            "max_pending_tasks": self.MAX_PENDING_TASKS,
            "cleanup_keep_days": self.CLEANUP_KEEP_DAYS,
            "minor_threshold": self.MINOR_THRESHOLD,
            "default_lab_id": self.DEFAULT_LAB_ID,
            "default_required_ppe": self.DEFAULT_REQUIRED_PPE,
            "allowed_ppe": list(self.ALLOWED_PPE),
        }


# 模块级单例：其他模块统一 `from backend.config import config` 使用
config = Config()
