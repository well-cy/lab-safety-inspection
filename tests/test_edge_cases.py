# -*- coding: utf-8 -*-
"""
第二轮探索性测试：边界与数据准确性（新增独立文件）。

运行：venv/bin/python tests/test_edge_cases.py

本轮不打正向功能，专门质疑"上次没覆盖到"的假设。7 个探测点中：
  ✅ 无问题：0 帧视频、伪视频、文件名边界（中文/emoji/无扩展名/超长/大写/空格）、
            大 offset 分页、stride 大于总帧数（不崩溃）
  🔴 真实缺陷（本轮修复）：
     BUG-6  视频检测落库的 person_count 恒为 0（库内数据不准确）
     BUG-7  vtype 筛选用 LIKE 但未转义通配符 —— 传 % 或 _ 会返回**全部**记录
     BUG-8  后台任务队列无上限，可被无限提交堆积（内存 + 磁盘）

注意：本文件为新增，不修改 tests/run_tests.py 的 T01~T14。
"""
import io
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import backend.main as main_mod  # noqa: E402
from backend import database as db  # noqa: E402
from backend.config import config  # noqa: E402
from backend.database import _escape_like  # noqa: E402
from backend.tasks import QueueFull, TaskManager  # noqa: E402

RESULTS = []
DEMO_VIDEO = ROOT / "data" / "test" / "demo_lab_video.mp4"
TEST_IMG = ROOT / "data" / "test" / "t05_pcr_scientist.jpg"


def report(name, ok, detail=""):
    RESULTS.append({"name": name, "ok": ok, "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        print(f"       {detail}")


def main():
    created_records = []
    saved_tm = main_mod.task_manager
    try:
        # ============ BUG-7：LIKE 通配符转义 ============
        print("--- BUG-7 LIKE 通配符转义 ---")
        report("_escape_like 转义 % / _ / 反斜杠",
               _escape_like("%") == "\\%" and _escape_like("_") == "\\_"
               and _escape_like("a\\b") == "a\\\\b",
               f"% → {_escape_like('%')!r}, _ → {_escape_like('_')!r}, "
               f"a\\\\b → {_escape_like('a\\b')!r}")
        report("_escape_like 不影响正常中文类型",
               _escape_like("未佩戴口罩") == "未佩戴口罩", "")

        with TestClient(main_mod.app) as client:
            all_count = client.get("/api/violations?limit=1").json()["total_count"]
            pct = client.get("/api/violations?vtype=%25&limit=1").json()["total_count"]
            und = client.get("/api/violations?vtype=_&limit=1").json()["total_count"]

            report("vtype=% 不再匹配全部记录", pct == 0 and all_count > 0,
                   f"全部={all_count}  vtype=% 命中={pct}（修复前为 {all_count}）")
            report("vtype=_ 不再匹配全部记录", und == 0,
                   f"vtype=_ 命中={und}（修复前为 {all_count}）")

            real = client.get("/api/violations?vtype=未佩戴口罩&limit=1").json()
            report("正常类型筛选仍然可用", real["total_count"] > 0,
                   f"vtype=未佩戴口罩 → total_count={real['total_count']}")

            exp = client.get("/api/export/violations?fmt=csv&ppe_type=%25")
            import csv
            rows = list(csv.reader(io.StringIO(exp.content.decode("utf-8-sig"))))
            report("导出接口同样转义（口径一致）", len(rows) - 1 == 0,
                   f"ppe_type=% 导出行数={len(rows) - 1}（修复前为 {all_count}）")

            # ============ BUG-8：任务队列上限 ============
            print("--- BUG-8 后台任务队列上限 ---")
            tm = TaskManager(max_workers=1, max_pending=2)
            accepted = rejected = 0
            for _ in range(5):
                try:
                    tm.submit("blocker", lambda p: (time.sleep(0.2), {"ok": True})[1])
                    accepted += 1
                except QueueFull:
                    rejected += 1
            report("达到 max_pending 后拒绝提交",
                   accepted == 2 and rejected == 3,
                   f"接受={accepted} 拒绝={rejected}（上限 2）")
            report("stats() 暴露 max_pending 便于诊断",
                   tm.stats().get("max_pending") == 2, f"stats={tm.stats()}")
            tm.shutdown(wait=True)

            # HTTP 层：临时换成一个容量为 1 的管理器，塞满后提交应得 429
            small = TaskManager(max_workers=1, max_pending=1)
            small.submit("blocker", lambda p: (time.sleep(1.5), {"ok": True})[1])
            main_mod.task_manager = small
            before = len(list(config.UPLOAD_DIR.glob("vid_*.mp4")))
            r = client.post("/api/detect/video/async",
                            files={"file": ("a.mp4", io.BytesIO(b"\x00" * 512),
                                            "video/mp4")},
                            data={"lab_id": "1", "stride": "2"})
            after = len(list(config.UPLOAD_DIR.glob("vid_*.mp4")))
            report("队列满时返回 429 且含 error 字段",
                   r.status_code == 429 and "error" in r.json(),
                   f"HTTP {r.status_code} {r.text[:70]}")
            report("被拒绝时清理已落盘的临时文件（不占磁盘）",
                   after == before, f"拒绝前={before} 个 拒绝后={after} 个")
            small.shutdown(wait=True)
            main_mod.task_manager = saved_tm

            # ============ 边界回归：确认无问题的项 ============
            print("--- 边界回归（确认为无问题项）---")
            if TEST_IMG.exists():
                img_bytes = TEST_IMG.read_bytes()
                names = ["实验室照片.jpg", "🔬.jpg", "noext", "A" * 200 + ".jpg",
                         "UPPER.JPG", "带空格 的 文件.jpg", "a" * 100 + ".jpeg"]
                oks = 0
                for fname in names:
                    rr = client.post("/api/detect/image",
                                     files={"file": (fname, io.BytesIO(img_bytes),
                                                     "image/jpeg")},
                                     data={"lab_id": "1"})
                    if rr.status_code == 200:
                        oks += 1
                        created_records.append(rr.json().get("record_id"))
                report("文件名边界（中文/emoji/无扩展名/超长/大写/空格）均可处理",
                       oks == len(names), f"{oks}/{len(names)} 成功")

            rr = client.post("/api/detect/video",
                             files={"file": ("fake.mp4",
                                             io.BytesIO(b"\x00\x00\x00\x18ftypisom"),
                                             "video/mp4")},
                             data={"lab_id": "1", "stride": "2"})
            report("伪视频（仅文件头）→ 400 且报错可读",
                   rr.status_code == 400 and "无法打开" in rr.json().get("error", ""),
                   f"HTTP {rr.status_code}")

            with tempfile.TemporaryDirectory() as tmp:
                zero = Path(tmp) / "zero.mp4"
                vw = cv2.VideoWriter(str(zero), cv2.VideoWriter_fourcc(*"mp4v"),
                                     25.0, (64, 64))
                vw.release()          # 不写任何帧
                with open(zero, "rb") as fh:
                    rr = client.post("/api/detect/video",
                                     files={"file": ("zero.mp4", fh, "video/mp4")},
                                     data={"lab_id": "1", "stride": "2"})
                report("0 帧视频 → 400（不崩溃）", rr.status_code == 400,
                       f"HTTP {rr.status_code}")

            rr = client.get("/api/violations?limit=1&offset=999999")
            report("超大 offset 返回空列表而非报错",
                   rr.status_code == 200 and rr.json()["items"] == [],
                   f"HTTP {rr.status_code}")

            # ============ BUG-6：视频 person_count ============
            print("--- BUG-6 视频落库 person_count ---")
            if not DEMO_VIDEO.exists():
                report("视频 person_count（跳过：缺演示视频）", True, "")
            else:
                with open(DEMO_VIDEO, "rb") as fh:
                    t0 = time.perf_counter()
                    rr = client.post("/api/detect/video",
                                     files={"file": (DEMO_VIDEO.name, fh, "video/mp4")},
                                     data={"lab_id": "1", "stride": "2"})
                    dt = time.perf_counter() - t0
                report("视频检测成功", rr.status_code == 200,
                       f"HTTP {rr.status_code} 耗时 {dt:.1f}s")
                rid = rr.json().get("record_id") if rr.status_code == 200 else None
                if rid:
                    created_records.append(rid)
                    conn = db.get_conn()
                    row = conn.execute(
                        "SELECT person_count, violation_count FROM detection_records"
                        " WHERE id=?", (rid,)).fetchone()
                    conn.close()
                    report("视频记录的 person_count 不再恒为 0",
                           row is not None and row["person_count"] > 0,
                           f"person_count={row['person_count'] if row else '-'} "
                           f"violation_count={row['violation_count'] if row else '-'}")

            if DEMO_VIDEO.exists():
                with open(DEMO_VIDEO, "rb") as fh:
                    rr = client.post("/api/detect/video",
                                     files={"file": (DEMO_VIDEO.name, fh, "video/mp4")},
                                     data={"lab_id": "1", "stride": "100000"})
                body = rr.json() if rr.status_code == 200 else {}
                report("stride=100000 不崩溃（但推理帧数极少）",
                       rr.status_code == 200 and body.get("total_frames", 0) > 0,
                       f"HTTP {rr.status_code} total_frames={body.get('total_frames')} "
                       f"inferred_frames={body.get('inferred_frames')} "
                       f"（提示：上报 inferred_frames 便于用户发现 stride 不合理）")
                if body.get("record_id"):
                    created_records.append(body["record_id"])
    finally:
        main_mod.task_manager = saved_tm
        conn = db.get_conn()
        for rid in created_records:
            if isinstance(rid, int):
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
