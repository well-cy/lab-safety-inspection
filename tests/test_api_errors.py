# -*- coding: utf-8 -*-
"""
后端 API 异常处理与参数校验测试（新增独立文件）。

运行：venv/bin/python tests/test_api_errors.py
    （或 Windows: venv\\Scripts\\python tests\\test_api_errors.py）

背景（v0.1.0-baseline 已知问题）：
    - 检测端点未捕获 ValueError：非图片/视频上传 → 500
    - lab_id 不存在触发外键约束 → 500（契约 6.6 声明"不报错"）
    - stride=0 触发 processor 中 `n_frames % stride` 除零 → 500
    - 区域坐标无 0~1 校验、required_ppe 无白名单校验

本用例验证以上非法输入现在返回 4xx，且 400 响应体带 error 字段
（前端 apiFetch 读取的是 body.error，而非 FastAPI 默认的 detail）。

注意：本文件为新增，不修改 tests/run_tests.py 的 T01~T14。
"""
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from backend.main import app  # noqa: E402

TEST_DIR = ROOT / "data" / "test"
RESULTS = []


def report(name, ok, detail):
    RESULTS.append({"name": name, "ok": ok, "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"       {detail}")


def expect_400(name, resp):
    """期望 400，且响应体含 error 字段（前端依赖该字段显示提示）"""
    try:
        body = resp.json()
    except Exception:
        body = {}
    ok = resp.status_code == 400 and isinstance(body, dict) and "error" in body
    report(name, ok, f"HTTP {resp.status_code}  body={body}")


def expect_422(name, resp):
    """期望 FastAPI 参数校验失败 422"""
    report(name, resp.status_code == 422, f"HTTP {resp.status_code}")


def main():
    with TestClient(app) as client:
        print("=" * 62)
        print("后端 API 异常处理 / 参数校验测试")
        print("=" * 62)

        # ---------- 检测：图片 ----------
        expect_400("图片接口 - 非图片文件（原名 500）",
                   client.post("/api/detect/image",
                               files={"file": ("a.txt", io.BytesIO(b"not an image"),
                                               "text/plain")},
                               data={"lab_id": "1"}))

        expect_400("图片接口 - 空文件",
                   client.post("/api/detect/image",
                               files={"file": ("empty.jpg", io.BytesIO(b""),
                                               "image/jpeg")},
                               data={"lab_id": "1"}))

        expect_400("图片接口 - 实验室不存在（原名 500 外键崩溃）",
                   client.post("/api/detect/image",
                               files={"file": ("a.jpg", io.BytesIO(b"\xff\xd8\xff"),
                                               "image/jpeg")},
                               data={"lab_id": "9999"}))

        # ---------- 检测：视频 ----------
        expect_400("视频接口 - stride=0（原名 500 除零）",
                   client.post("/api/detect/video",
                               files={"file": ("a.mp4", io.BytesIO(b"\x00\x00\x00\x18ftyp"),
                                               "video/mp4")},
                               data={"lab_id": "1", "stride": "0"}))

        expect_400("视频接口 - stride=-5",
                   client.post("/api/detect/video",
                               files={"file": ("a.mp4", io.BytesIO(b"\x00\x00\x00\x18ftyp"),
                                               "video/mp4")},
                               data={"lab_id": "1", "stride": "-5"}))

        expect_400("视频接口 - 实验室不存在",
                   client.post("/api/detect/video",
                               files={"file": ("a.mp4", io.BytesIO(b"\x00\x00\x00\x18ftyp"),
                                               "video/mp4")},
                               data={"lab_id": "9999", "stride": "2"}))

        expect_400("视频接口 - 非视频文件（原名 500）",
                   client.post("/api/detect/video",
                               files={"file": ("a.mp4", io.BytesIO(b"totally not a video"),
                                               "video/mp4")},
                               data={"lab_id": "1", "stride": "2"}))

        # ---------- 设置 ----------
        expect_400("设置 - 实验室重名",
                   client.post("/api/settings/lab",
                               data={"name": "实验室A", "description": ""}))

        expect_400("设置 - 区域坐标越界（x1=1.5）",
                   client.post("/api/settings/area",
                               data={"lab_id": "1", "name": "越界区",
                                     "x1": "1.5", "y1": "0.1",
                                     "x2": "0.9", "y2": "0.9"}))

        expect_400("设置 - 区域坐标反向（x1>x2）",
                   client.post("/api/settings/area",
                               data={"lab_id": "1", "name": "反向区",
                                     "x1": "0.8", "y1": "0.1",
                                     "x2": "0.2", "y2": "0.9"}))

        expect_400("设置 - 非法 PPE 类型",
                   client.post("/api/settings/area",
                               data={"lab_id": "1", "name": "非法PPE区",
                                     "x1": "0.1", "y1": "0.1",
                                     "x2": "0.5", "y2": "0.5",
                                     "required_ppe": "mask,laser_gun,gun"}))

        expect_400("设置 - 区域所属实验室不存在（原名 500 外键崩溃）",
                   client.post("/api/settings/area",
                               data={"lab_id": "9999", "name": "孤儿区",
                                     "x1": "0.1", "y1": "0.1",
                                     "x2": "0.5", "y2": "0.5"}))

        # ---------- 违规查询分页参数 ----------
        expect_422("违规查询 - limit=-1", client.get("/api/violations?limit=-1"))
        expect_422("违规查询 - offset=-1", client.get("/api/violations?offset=-1"))
        expect_422("违规查询 - limit=99999（超上限 1000）",
                   client.get("/api/violations?limit=99999"))

        # ---------- 回归：合法请求不受影响 ----------
        print("-" * 62)
        img = TEST_DIR / "t05_pcr_scientist.jpg"
        if img.exists():
            with open(img, "rb") as f:
                r = client.post("/api/detect/image",
                                files={"file": (img.name, f, "image/jpeg")},
                                data={"lab_id": "1"})
            ok = r.status_code == 200 and "record_id" in r.json()
            report("回归 - 合法图片检测仍返回 200", ok, f"HTTP {r.status_code}")

        for path in ("/api/dashboard", "/api/statistics", "/api/settings",
                     "/api/violations", "/api/model/info"):
            r = client.get(path)
            report(f"回归 - GET {path}", r.status_code == 200, f"HTTP {r.status_code}")

    passed = sum(1 for x in RESULTS if x["ok"])
    total = len(RESULTS)
    print("=" * 62)
    print(f"结果: {passed}/{total} 通过")
    if passed != total:
        print("失败用例:")
        for x in RESULTS:
            if not x["ok"]:
                print(f"  - {x['name']}: {x['detail']}")
    print("=" * 62)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
