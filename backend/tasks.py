# -*- coding: utf-8 -*-
"""
后台任务管理（独立模块）。

解决的问题（PROJECT_STATUS 已知问题 3 / 技术债 #3）：
    ``POST /api/detect/video`` 原先在 ``async def`` 端点里做**阻塞**处理
    （cv2 读帧 + YOLO 推理），会占死事件循环 —— 长视频处理期间整个 API
    都无响应，连 ``/api/dashboard`` 都打不开。演示时风险极高。

设计要点
--------
1. **用线程池执行阻塞处理**：视频处理是 CPU/GPU 密集型，改成 asyncio 协程
   并不能解决问题，必须真正把工作挪到别的线程。
2. **任务状态保存在内存中，不新增数据库表**：新增表属于数据库结构变更，
   需走契约评审；本项目为单机单进程演示，内存态已足够，且服务重启后
   任务列表自动清空，反而不会出现"报了已完成但实际没跑"的误导。
3. 提交任务**立即返回 task_id**，调用方轮询 ``GET /api/tasks/{task_id}``
   读取 ``status`` / ``progress``。

已知局限（明确记录，不隐藏）
--------------------------
- 任务是进程内内存态，**服务重启后丢失**
- 仅适用于单进程部署（uvicorn 多 worker 时各进程看不到彼此的任务）
- 不支持取消已提交的任务
"""
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Callable

# ---------- 任务状态常量 ----------
STATUS_PENDING = "pending"     # 已提交，排队等待执行
STATUS_RUNNING = "running"     # 处理中
STATUS_DONE = "done"           # 处理成功
STATUS_FAILED = "failed"       # 处理失败（异常信息在 error 字段）

# 进度回调签名：progress_cb(已处理帧数, 总帧数)
ProgressCallback = Callable[[int, int], None]

# 工作函数签名：work(progress_cb) -> 结果（建议返回 dict）
WorkFn = Callable[[ProgressCallback], Any]


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Task:
    """一个后台任务的运行时状态"""

    def __init__(self, task_id: str, kind: str):
        self.task_id = task_id
        self.kind = kind
        self.status = STATUS_PENDING
        self.created_at = _now()
        self.started_at = ""
        self.finished_at = ""
        self.processed_frames = 0
        self.total_frames = 0
        self.result: dict | None = None
        self.error = ""

    @property
    def progress(self) -> float:
        """进度 0~1。总帧数未知时为 0；已完成恒为 1。"""
        if self.status == STATUS_DONE:
            return 1.0
        if not self.total_frames:
            return 0.0
        return round(min(self.processed_frames / self.total_frames, 1.0), 4)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "kind": self.kind,
            "status": self.status,
            "progress": self.progress,
            "processed_frames": self.processed_frames,
            "total_frames": self.total_frames,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "result": self.result,
        }


class TaskManager:
    """
    线程池任务管理器。

    所有对任务字典的读写都在同一把锁下进行，worker 线程与请求线程共享它。
    """

    def __init__(self, max_workers: int = 1, max_history: int = 100):
        self._tasks: dict[str, Task] = {}
        self._lock = threading.Lock()
        self._max_history = max(1, int(max_history))
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)),
            thread_name_prefix="labsafety-task")

    # ---------------- 提交与执行 ----------------

    def submit(self, kind: str, work: WorkFn) -> str:
        """
        提交一个后台任务，立即返回 task_id（不等待执行）。

        work: 可调用对象，签名 ``work(progress_cb) -> result``；
              work 内部应定期调用 ``progress_cb(已处理帧数, 总帧数)``。
              返回值若为 dict，会放进任务的 result 字段。
        """
        task = Task(task_id=uuid.uuid4().hex[:12], kind=kind)
        with self._lock:
            self._tasks[task.task_id] = task
            self._trim_locked()
        self._executor.submit(self._run, task.task_id, work)
        return task.task_id

    def _run(self, task_id: str, work: WorkFn) -> None:
        self._update(task_id, status=STATUS_RUNNING, started_at=_now())

        def progress_cb(processed: int, total: int) -> None:
            self._update(task_id, processed_frames=int(processed),
                         total_frames=int(total))

        try:
            result = work(progress_cb)
        except Exception as exc:  # 任务失败不应拖垮服务，记录到任务状态里
            self._update(task_id, status=STATUS_FAILED, finished_at=_now(),
                         error=f"{type(exc).__name__}: {exc}")
        else:
            self._update(task_id, status=STATUS_DONE, finished_at=_now(),
                         result=result if isinstance(result, dict) else None)
        finally:
            # 任务结束时也裁剪一次：只在 submit 时裁剪的话，最后一批任务完成后
            # 会一直超出上限，直到下次提交才被清理（已被测试用例抓到）。
            self._trim()

    # ---------------- 查询 ----------------

    def get(self, task_id: str) -> Task | None:
        with self._lock:
            return self._tasks.get(task_id)

    def list_recent(self, limit: int = 20) -> list[Task]:
        with self._lock:
            items = sorted(self._tasks.values(),
                           key=lambda t: t.created_at, reverse=True)
            return items[:max(1, int(limit))]

    def stats(self) -> dict:
        with self._lock:
            counts = {s: 0 for s in (STATUS_PENDING, STATUS_RUNNING,
                                     STATUS_DONE, STATUS_FAILED)}
            for task in self._tasks.values():
                counts[task.status] = counts.get(task.status, 0) + 1
            return {
                "total": len(self._tasks),
                "active": counts[STATUS_PENDING] + counts[STATUS_RUNNING],
                **counts,
            }

    def shutdown(self, wait: bool = False) -> None:
        """关闭线程池（主要供测试使用）"""
        self._executor.shutdown(wait=wait)

    # ---------------- 内部 ----------------

    def _update(self, task_id: str, **fields) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            for key, value in fields.items():
                setattr(task, key, value)

    def _trim(self) -> None:
        with self._lock:
            self._trim_locked()

    def _trim_locked(self) -> None:
        """
        仅保留最近 max_history 个任务，避免长期运行内存无限增长。

        不变量（被 tests/test_video_async.py 断言）：
          - **已结束**（done/failed）的任务最多保留 max_history 个
          - **未结束**（pending/running）的任务永不裁剪 —— 裁剪掉会导致
            调用方轮询时突然查不到自己的任务

        调用方必须已持有 self._lock。
        """
        if len(self._tasks) <= self._max_history:
            return
        finished = sorted(
            (t for t in self._tasks.values()
             if t.status in (STATUS_DONE, STATUS_FAILED)),
            key=lambda t: t.created_at)
        for task in finished[:len(self._tasks) - self._max_history]:
            self._tasks.pop(task.task_id, None)
