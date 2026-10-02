# -*- coding: utf-8 -*-
"""
存储维护（清理运行期文件）测试（新增独立文件）。

运行：venv/bin/python tests/test_maintenance.py

背景（分工任务「清理临时视频、结果视频和截图目录」）：
    运行期 4 个目录持续产生文件且原先没有任何清理机制。实测本机一天测试
    就产生 54.58 MB（上传 13.52 + 标注图 6.27 + 视频 21.78 + 截图 13.01），
    长期运行磁盘只增不减。

本用例验证：
  1. 存储报告结构正确、数字自洽
  2. **默认 dry_run**（只统计不删除）—— 防止误删
  3. 按修改时间保留：旧文件清理、新文件保留
  4. 默认目标**不含 screenshots**（截图被数据库记录引用）
  5. 只在目标目录内操作，不触碰其他路径
  6. 参数校验：非法目标、越界天数

测试在临时目录中进行，通过替换 config 的目录指向实现隔离，
结束后恢复原值，不会动到真实的 outputs/ 与 uploads/。

注意：本文件为新增，不修改 tests/run_tests.py 的 T01~T14。
"""
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from backend import maintenance  # noqa: E402
from backend.config import config  # noqa: E402
from backend.main import app  # noqa: E402

RESULTS = []

# config 属性名对照：(目标名, config 属性)
DIR_ATTRS = (
    ("uploads", "UPLOAD_DIR"),
    ("annotated", "ANNOTATED_DIR"),
    ("videos", "VIDEO_DIR"),
    ("screenshots", "SCREENSHOT_DIR"),
)


def report(name, ok, detail=""):
    RESULTS.append({"name": name, "ok": ok, "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        print(f"       {detail}")


@contextmanager
def temp_dirs(base: Path):
    """把 config 的 4 个运行期目录临时指向 base 下的子目录，结束后恢复"""
    saved = {}
    for name, attr in DIR_ATTRS:
        saved[attr] = getattr(config, attr)
        d = base / name
        d.mkdir(parents=True, exist_ok=True)
        setattr(config, attr, d)
    try:
        yield base
    finally:
        for attr, old in saved.items():
            setattr(config, attr, old)


def make_file(path: Path, age_days: float, size: int = 1024) -> Path:
    """创建一个文件并把修改时间设为 age_days 天前"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    t = time.time() - age_days * 86400
    os.utime(path, (t, t))
    return path


def main():
    print("=" * 68)
    print("存储维护（清理运行期文件）测试")
    print("=" * 68)

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        # 目标目录之外的文件：用于验证"不触碰其他路径"
        outside = tmp / "outside"
        make_file(outside / "keep_me.txt", age_days=30)

        with temp_dirs(tmp):
            # ---------------- 1. 存储报告 ----------------
            print("--- 1. storage_report 结构 ---")
            make_file(config.UPLOAD_DIR / "old_a.bin", age_days=10, size=2048)
            make_file(config.UPLOAD_DIR / "sub" / "old_b.bin", age_days=10, size=1024)
            make_file(config.VIDEO_DIR / "big.mp4", age_days=1, size=4096)
            make_file(config.SCREENSHOT_DIR / "shot.jpg", age_days=0.01, size=512)

            rep = maintenance.storage_report()
            report("报告包含全部 4 个目标",
                   set(rep["targets"]) == {"uploads", "annotated", "videos",
                                           "screenshots"},
                   f"实际={sorted(rep['targets'])}")

            need = {"path", "exists", "files", "bytes", "mb", "oldest"}
            report("每个目标字段齐全",
                   all(need <= set(v) for v in rep["targets"].values()), "")

            report("uploads 文件数含子目录（递归统计）",
                   rep["targets"]["uploads"]["files"] == 2,
                   f"files={rep['targets']['uploads']['files']}")

            expect_total = sum(v["bytes"] for v in rep["targets"].values())
            report("total_bytes 等于各目标之和",
                   rep["total_bytes"] == expect_total,
                   f"total={rep['total_bytes']} sum={expect_total}")

            report("最旧文件时间已填充",
                   rep["targets"]["uploads"]["oldest"].startswith("20"), 
                   f"oldest={rep['targets']['uploads']['oldest']}")

            # ---------------- 2. 默认 dry_run ----------------
            print("--- 2. 默认 dry_run（防误删）---")
            before = sorted(p.name for p in config.UPLOAD_DIR.rglob("*") if p.is_file())
            res = maintenance.cleanup(keep_days=7)
            after = sorted(p.name for p in config.UPLOAD_DIR.rglob("*") if p.is_file())
            report("默认 dry_run=True", res["dry_run"] is True, f"dry_run={res['dry_run']}")
            report("dry_run 下文件一个都没删", before == after,
                   f"前={len(before)} 个 后={len(after)} 个")
            report("dry_run 仍统计出待删除数量",
                   res["detail"]["uploads"]["deleted"] == 2,
                   f"deleted={res['detail']['uploads']['deleted']}（2 个 10 天前文件）")
            report("dry_run 返回 note 说明未删除",
                   "未删除" in res["note"], f"note={res['note']}")

            # ---------------- 3. 按修改时间保留 ----------------
            print("--- 3. 按修改时间保留（旧删新留）---")
            make_file(config.UPLOAD_DIR / "fresh.bin", age_days=0.01, size=128)
            res = maintenance.cleanup(keep_days=7, dry_run=False)
            remaining = sorted(p.name for p in config.UPLOAD_DIR.rglob("*") if p.is_file())
            report("超期文件被删除", res["detail"]["uploads"]["deleted"] == 2,
                   f"deleted={res['detail']['uploads']['deleted']}")
            report("未超期文件被保留", remaining == ["fresh.bin"],
                   f"剩余={remaining}")
            report("freed_bytes 等于删除文件大小之和",
                   res["detail"]["uploads"]["freed_bytes"] == 3072,
                   f"freed={res['detail']['uploads']['freed_bytes']}（2048+1024）")
            report("新文件（1 天前）不受 keep_days=7 影响",
                   res["detail"]["videos"]["deleted"] == 0
                   and res["detail"]["videos"]["kept"] == 1,
                   f"deleted={res['detail']['videos']['deleted']} kept={res['detail']['videos']['kept']}")

            # ---------------- 4. 默认目标不含 screenshots ----------------
            print("--- 4. 默认目标不含 screenshots（截图被数据库引用）---")
            report("默认目标 = uploads/annotated/videos",
                   set(res["targets"]) == {"uploads", "annotated", "videos"},
                   f"targets={res['targets']}")
            shot = config.SCREENSHOT_DIR / "shot.jpg"
            report("截图文件未被默认清理", shot.exists(),
                   f"{shot.name} exists={shot.exists()}")
            report("显式指定后可清理截图",
                   maintenance.cleanup(keep_days=0, dry_run=True,
                                       targets=["screenshots"]
                                       )["detail"]["screenshots"]["deleted"] == 1, "")

            # ---------------- 5. 不触碰其他路径 ----------------
            print("--- 5. 只在目标目录内操作 ---")
            maintenance.cleanup(keep_days=0, dry_run=False,
                                targets=["uploads", "annotated", "videos",
                                         "screenshots"])
            report("目标目录之外的文件未被触碰", (outside / "keep_me.txt").exists(),
                   f"{outside.name}/keep_me.txt exists=True")

            # ---------------- 6. 参数校验 ----------------
            print("--- 6. 参数校验 ---")
            try:
                maintenance.cleanup(keep_days=7, targets=["不存在的目录"])
                report("未知目标抛 ValueError", False, "未抛异常")
            except ValueError as exc:
                report("未知目标抛 ValueError 且提示可选值",
                       "可选值" in str(exc) and "uploads" in str(exc),
                       f"{exc}")

            try:
                maintenance.cleanup(keep_days=-1)
                report("负数 keep_days 抛 ValueError", False, "未抛异常")
            except ValueError as exc:
                report("负数 keep_days 抛 ValueError", True, f"{exc}")

            # ---------------- 7. 接口层 ----------------
            print("--- 7. HTTP 接口 ---")
            with TestClient(app) as client:
                r = client.get("/api/maintenance/storage")
                report("GET /api/maintenance/storage → 200",
                       r.status_code == 200 and "targets" in r.json(),
                       f"HTTP {r.status_code}")

                r = client.post("/api/maintenance/cleanup")
                report("POST cleanup 默认 dry_run（不删文件）",
                       r.status_code == 200 and r.json()["dry_run"] is True,
                       f"HTTP {r.status_code} dry_run={r.json().get('dry_run')}")

                r = client.post("/api/maintenance/cleanup?days=-1")
                report("days=-1 → 422（参数越界）", r.status_code == 422,
                       f"HTTP {r.status_code}")

                r = client.post("/api/maintenance/cleanup?targets=bad_target")
                report("未知 target → 400 且含 error 字段",
                       r.status_code == 400 and "error" in r.json(),
                       f"HTTP {r.status_code} {r.text[:80]}")

                r = client.post("/api/maintenance/cleanup?days=7&dry_run=false"
                                "&targets=uploads,annotated,videos")
                report("显式 dry_run=false 可正常执行",
                       r.status_code == 200 and r.json()["dry_run"] is False,
                       f"HTTP {r.status_code}")

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
