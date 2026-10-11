# -*- coding: utf-8 -*-
"""
视频后台任务与进度查询测试（新增独立文件）。

运行：venv/bin/python tests/test_video_async.py

背景（PROJECT_STATUS 已知问题 3 / 技术债 #3）：
    POST /api/detect/video 原先写成 async def 却做阻塞处理
    （cv2 读帧 + YOLO 推理），会占死事件循环 —— 长视频处理期间整个 API
    都无响应，连 /api/dashboard 都打不开。演示时风险极高。

本用例验证四件事：
  1. **结构性**：检测端点不再是协程函数 —— FastAPI 会把它们放进线程池，
     这是"不阻塞事件循环"的前提（回归时会第一时间发现有人改回 async def）
  2. TaskManager 单元行为：状态流转、进度回调、失败记录、历史裁剪
  3. 异步接口闭环：提交 → 轮询进度 → 完成 → 结果落库
  4. **响应性**：后台处理视频期间，其他接口仍能快速响应（核心验收点）

注意：本文件为新增，不修改 tests/run_tests.py 的 T01~T14。
"""
import inspect
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

import backend.main as main_mod  # noqa: E402
from backend import database as db  # noqa: E402
from backend.tasks import (  # noqa: E402
    STATUS_DONE, STATUS_FAILED, STATUS_PENDING, STATUS_RUNNING, TaskManager)

RESULTS = []
DEMO_VIDEO = ROOT / "data" / "test" / "demo_lab_video.mp4"


def report(name, ok, detail=""):
    RESULTS.append({"name": name, "ok": ok, "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        print(f"       {detail}")


def wait_for_task(client, task_id, timeout=180.0, on_poll=None):
    """轮询任务直到结束；返回 (最终状态 dict, 采样到的进度列表)"""
    deadline = time.perf_counter() + timeout
    samples = []
    last = {}
    while time.perf_counter() < deadline:
        r = client.get(f"/api/tasks/{task_id}")
        last = r.json()
        samples.append(last.get("progress", 0.0))
        if on_poll is not None:
            on_poll()
        if last.get("status") in (STATUS_DONE, STATUS_FAILED):
            return last, samples
        time.sleep(0.2)
    return last, samples


def main():
    print("=" * 68)
    print("视频后台任务与进度查询测试")
    print("=" * 68)

    # ---------------- 1. 结构性断言：端点必须是同步函数 ----------------
    print("--- 1. 结构性：检测端点不再是协程（线程池前提）---")
    for fn in (main_mod.detect_image, main_mod.detect_video):
        report(f"{fn.__name__} 是同步 def（FastAPI 会放线程池）",
               not inspect.iscoroutinefunction(fn),
               f"iscoroutinefunction={inspect.iscoroutinefunction(fn)}")
    report("detect_video_async 也应为同步 def（提交动作本身不阻塞）",
           not inspect.iscoroutinefunction(main_mod.detect_video_async), "")
    report("task_manager 已创建且 worker 数 >= 1",
           main_mod.task_manager._executor._max_workers >= 1,
           f"workers={main_mod.task_manager._executor._max_workers}")

    # ---------------- 2. TaskManager 单元行为 ----------------
    print("--- 2. TaskManager 单元行为 ---")
    tm = TaskManager(max_workers=2, max_history=5)

    def ok_work(progress):
        for i in range(1, 5):
            progress(i, 4)
            time.sleep(0.02)
        return {"value": 42}

    tid = tm.submit("unit_ok", ok_work)
    report("submit 立即返回 task_id", isinstance(tid, str) and len(tid) == 12,
           f"task_id={tid}")
    report("任务初始状态为 pending", tm.get(tid).status in
           (STATUS_PENDING, STATUS_RUNNING), f"status={tm.get(tid).status}")

    for _ in range(100):
        if tm.get(tid).status == STATUS_DONE:
            break
        time.sleep(0.02)
    task = tm.get(tid)
    report("任务最终状态为 done", task.status == STATUS_DONE, f"status={task.status}")
    report("进度回调被记录（processed==total）",
           task.processed_frames == 4 and task.total_frames == 4,
           f"processed={task.processed_frames} total={task.total_frames}")
    report("完成的任务 progress == 1.0", task.progress == 1.0, f"progress={task.progress}")
    report("结果被保存到 result 字段", task.result == {"value": 42},
           f"result={task.result}")

    def bad_work(progress):
        progress(1, 10)
        raise ValueError("故意失败")

    tid2 = tm.submit("unit_fail", bad_work)
    for _ in range(100):
        if tm.get(tid2).status == STATUS_FAILED:
            break
        time.sleep(0.02)
    t2 = tm.get(tid2)
    report("失败任务状态为 failed", t2.status == STATUS_FAILED, f"status={t2.status}")
    report("失败原因写入 error（含异常类型）",
           "ValueError" in t2.error and "故意失败" in t2.error, f"error={t2.error}")
    report("失败任务的 result 为 None", t2.result is None, f"result={t2.result}")

    report("未知 task_id 返回 None", tm.get("不存在的id") is None, "")

    st = tm.stats()
    report("stats() 计数正确",
           st["total"] == 2 and st["done"] == 1 and st["failed"] == 1
           and st["active"] == 0,
           f"stats={st}")

    # 历史裁剪
    for i in range(8):
        tm.submit(f"trim_{i}", lambda p: {"i": 1})
    time.sleep(0.5)
    kept = tm.list_recent(100)
    finished_kept = [t for t in kept if t.status in (STATUS_DONE, STATUS_FAILED)]
    active_kept = [t for t in kept if t.status not in (STATUS_DONE, STATUS_FAILED)]
    report("历史裁剪生效：已结束任务不超过 max_history",
           len(finished_kept) <= 5,
           f"已结束={len(finished_kept)}（上限 5） 未结束={len(active_kept)}")
    report("未结束的任务不会被裁剪（否则调用方轮询会查不到）",
           len(kept) == len(finished_kept) + len(active_kept), "")

    report("list_recent 按创建时间倒序",
           all(tm.list_recent(10)[i].created_at >= tm.list_recent(10)[i + 1].created_at
               for i in range(len(tm.list_recent(10)) - 1)), "")
    tm.shutdown(wait=True)

    # ---------------- 3 & 4. 接口闭环 + 响应性 ----------------
    created_records = []
    try:
        with TestClient(app=main_mod.app) as client:
            print("--- 3. 异步接口：参数校验 ---")
            r = client.post("/api/detect/video/async",
                            files={"file": ("a.mp4", b"\x00\x00\x00\x18ftyp", "video/mp4")},
                            data={"lab_id": "1", "stride": "0"})
            report("stride=0 → 400",
                   r.status_code == 400 and "error" in r.json(),
                   f"HTTP {r.status_code} {r.text[:60]}")

            r = client.post("/api/detect/video/async",
                            files={"file": ("a.mp4", b"\x00\x00\x00\x18ftyp", "video/mp4")},
                            data={"lab_id": "999999", "stride": "2"})
            report("实验室不存在 → 400", r.status_code == 400,
                   f"HTTP {r.status_code}")

            r = client.post("/api/detect/video/async",
                            files={"file": ("empty.mp4", b"", "video/mp4")},
                            data={"lab_id": "1", "stride": "2"})
            report("空文件 → 400", r.status_code == 400, f"HTTP {r.status_code}")

            report("未知任务 → 404 且含 error 字段",
                   client.get("/api/tasks/nonexistent").status_code == 404
                   and "error" in client.get("/api/tasks/nonexistent").json(), "")

            print("--- 4. 异步接口：提交并轮询进度 ---")
            if not DEMO_VIDEO.exists():
                report("视频后台任务端到端（跳过：缺演示视频）", True,
                       f"未找到 {DEMO_VIDEO.name}")
            else:
                with open(DEMO_VIDEO, "rb") as f:
                    r = client.post("/api/detect/video/async",
                                    files={"file": (DEMO_VIDEO.name, f, "video/mp4")},
                                    data={"lab_id": "1", "stride": "2"})
                report("提交返回 HTTP 202", r.status_code == 202,
                       f"HTTP {r.status_code}")
                body = r.json()
                task_id = body.get("task_id", "")
                report("响应含 task_id 与 status_url",
                       bool(task_id) and body.get("status_url") == f"/api/tasks/{task_id}",
                       f"task_id={task_id} status_url={body.get('status_url')}")

                # ---- 响应性采样：轮询任务的同时请求其他接口 ----
                latencies = []
                dashboard_codes = []

                def on_poll():
                    t0 = time.perf_counter()
                    dr = client.get("/api/dashboard")
                    latencies.append(time.perf_counter() - t0)
                    dashboard_codes.append(dr.status_code)

                t_start = time.perf_counter()
                final, progress_samples = wait_for_task(
                    client, task_id, on_poll=on_poll)
                duration = time.perf_counter() - t_start

                report("任务最终完成", final.get("status") == STATUS_DONE,
                       f"status={final.get('status')} error={final.get('error', '')[:80]}")
                report("任务耗时合理（确实做了实际处理）", duration > 0.5,
                       f"耗时={duration:.1f}s")

                report("进度单调不减且最终为 1.0",
                       progress_samples == sorted(progress_samples)
                       and final.get("progress") == 1.0,
                       f"采样数={len(progress_samples)} 最大={max(progress_samples) if progress_samples else '-'}")

                report("进度有中间态（不是 0 直接跳 1）",
                       any(0 < p < 1 for p in progress_samples),
                       f"非端点采样值={[p for p in progress_samples if 0 < p < 1][:5]}")

                max_latency = max(latencies) if latencies else 999
                report("【核心】处理期间其他接口仍可响应（未阻塞事件循环）",
                       max_latency < 1.0,
                       f"轮询 {len(latencies)} 次 /api/dashboard，最大延迟={max_latency * 1000:.0f}ms "
                       f"（任务总耗时 {duration:.1f}s；若阻塞则每次都需 {duration:.1f}s 量级）")
                report("处理期间 /api/dashboard 全部返回 200",
                       set(dashboard_codes) == {200},
                       f"状态码集合={set(dashboard_codes)}")

                result = final.get("result") or {}
                report("结果包含 record_id / output_video_url",
                       isinstance(result.get("record_id"), int)
                       and str(result.get("output_video_url", "")).startswith("/media/videos/"),
                       f"record_id={result.get('record_id')} url={result.get('output_video_url')}")
                report("结果包含帧数与违规数",
                       result.get("total_frames", 0) > 0
                       and result.get("violation_count", -1) >= 0,
                       f"total_frames={result.get('total_frames')} "
                       f"inferred={result.get('inferred_frames')} "
                       f"violations={result.get('violation_count')}")

                if isinstance(result.get("record_id"), int):
                    created_records.append(result["record_id"])
                    conn = db.get_conn()
                    row = conn.execute("SELECT source_type, lab_id FROM detection_records"
                                       " WHERE id=?", (result["record_id"],)).fetchone()
                    conn.close()
                    report("检测记录已落库且 source_type=video",
                           row is not None and row["source_type"] == "video",
                           f"row={dict(row) if row else None}")

                r = client.get("/api/tasks?limit=5")
                listing = r.json()
                report("GET /api/tasks 列表包含该任务",
                       any(i["task_id"] == task_id for i in listing.get("items", [])),
                       f"列表条数={len(listing.get('items', []))}")
                report("列表项不含 result（避免响应过大）",
                       all("result" not in i for i in listing.get("items", [])), "")
                report("GET /api/tasks 返回 stats",
                       "stats" in listing and listing["stats"]["total"] >= 1,
                       f"stats={listing.get('stats')}")
                report("GET /api/tasks limit 越界 → 422",
                       client.get("/api/tasks?limit=0").status_code == 422, "")

    finally:
        conn = db.get_conn()
        for rid in created_records:
            conn.execute("DELETE FROM violation_events WHERE record_id=?", (rid,))
            conn.execute("DELETE FROM detection_records WHERE id=?", (rid,))
        conn.commit()
        conn.close()

    passed = sum(1 for x in RESULTS if x["ok"])
    total = len(RESULTS)
    print("=" * 68)
    print(f"结果: {passed}/{total} 通过")
    if passed != total:
        print("失败用例:")
        for x in RESULTS:
            if not x["ok"]:
                print(f"  - {x['name']}: {x['detail']}")
    print("=" * 68)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
