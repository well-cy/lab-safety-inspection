# -*- coding: utf-8 -*-
"""
后端健壮性测试（新增独立文件）。

运行：venv/bin/python tests/test_robustness.py

背景（本轮测试发现的真实缺陷）：
  BUG-3  /api/violations 参数不校验
         `date_to=2026-99-99` 这类 typo 被静默忽略并返回**全表**数据，
         用户会误以为"筛选生效了"；同一套筛选参数在 /api/export/violations
         会返回 400/422，两个接口口径不一致。
  BUG-4  模型权重缺失时抛 FileNotFoundError → 500 + 原始路径堆栈
         模型文件不入 git（.gitignore 排除 *.pt），新成员克隆后首次调用
         必然遇到，但提示无法指导解决。
  BUG-5  图片模式为每个违规人员各写一份完全相同的标注图
         实测一次 7 人检测产生 6 张内容相同的图片（5.9MB / 13 文件）。

注意：本文件为新增，不修改 tests/run_tests.py 的 T01~T14。
"""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

import backend.main as main_mod  # noqa: E402
from backend.main import app     # noqa: E402

RESULTS = []
SHOT_DIR = ROOT / "outputs" / "screenshots"
TEST_IMG = ROOT / "data" / "test" / "t09_lab_coat_ceremony.jpg"


def report(name, ok, detail=""):
    RESULTS.append({"name": name, "ok": ok, "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        print(f"       {detail}")


def main():
    with TestClient(app) as client:
        print("=" * 68)
        print("后端健壮性测试")
        print("=" * 68)

        # ================= BUG-3：/api/violations 参数校验 =================
        print("--- BUG-3 违规查询参数校验（与导出接口同口径）---")
        bad_date_cases = [
            ("date=abc", "date"),
            ("date_from=abc", "date_from"),
            ("date_to=2026-99-99", "date_to"),
            ("date_from=2026-02-30", "date_from"),
        ]
        for query, field in bad_date_cases:
            r = client.get(f"/api/violations?{query}")
            body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            ok = r.status_code == 400 and field in str(body.get("error", ""))
            report(f"非法日期 ?{query} → 400 + error",
                   ok, f"HTTP {r.status_code} body={str(body)[:90]}")

        r = client.get("/api/violations?date_from=2026-10-05&date_to=2026-10-01")
        report("日期区间反向 → 400",
               r.status_code == 400 and "区间无效" in r.json().get("error", ""),
               f"HTTP {r.status_code} {r.text[:80]}")

        for query in ("sort=不存在", "order=乱填", "sort=1; DROP TABLE"):
            r = client.get(f"/api/violations?{query}")
            report(f"非法排序参数 ?{query} → 422",
                   r.status_code == 422, f"HTTP {r.status_code}（期望 422）")

        r = client.get("/api/violations?date_from=2026-01-01&date_to=2026-12-31"
                       "&sort=created_at&order=asc&limit=5")
        ok = r.status_code == 200 and "total_count" in r.json()
        report("合法参数仍正常返回 200", ok, f"HTTP {r.status_code}")

        r = client.get("/api/violations?limit=5&offset=0")
        items = r.json().get("items", [])
        order_ok = all(items[i]["id"] >= items[i + 1]["id"]
                       for i in range(len(items) - 1))
        report("默认排序仍为 id 倒序（未破坏原行为）", order_ok,
               f"返回 id 序列={[i['id'] for i in items]}")

        # ================= BUG-4：模型缺失应返回 503 =================
        print("--- BUG-4 模型权重缺失时的响应 ---")
        saved_path = main_mod.MODEL_PATH
        saved_pipeline, saved_detector = main_mod._pipeline, main_mod._detector
        main_mod.MODEL_PATH = Path("/nonexistent_dir/not_exist.pt")
        main_mod._pipeline = None
        main_mod._detector = None
        try:
            r = client.get("/api/model/info")
            body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            err = str(body.get("error", ""))
            report("模型缺失时 /api/model/info 返回 503（而非 500）",
                   r.status_code == 503, f"HTTP {r.status_code}")
            report("错误信息说明「模型文件不存在」",
                   "模型文件不存在" in err, f"error={err[:70]}")
            report("错误信息包含下载指引（可自助解决）",
                   "curl" in err and "yolo8s.pt" in err,
                   f"是否含 curl={'curl' in err} 是否含 yolo8s.pt={'yolo8s.pt' in err}")

            r = client.post("/api/detect/image",
                            files={"file": ("a.jpg", b"\xff\xd8\xff", "image/jpeg")},
                            data={"lab_id": "1"})
            report("模型缺失时检测接口同样返回 503",
                   r.status_code == 503, f"HTTP {r.status_code}")
        finally:
            main_mod.MODEL_PATH = saved_path
            main_mod._pipeline = saved_pipeline
            main_mod._detector = saved_detector

        r = client.get("/api/model/info")
        report("恢复后 /api/model/info 重新可用（未污染全局状态）",
               r.status_code == 200, f"HTTP {r.status_code}")

        # ================= BUG-5：图片截图不应重复 =================
        print("--- BUG-5 图片模式截图去重 ---")
        if not TEST_IMG.exists():
            report("BUG-5 截图去重（跳过：缺测试图片）", True, f"未找到 {TEST_IMG.name}")
        else:
            before = {p.name for p in SHOT_DIR.glob("img_*.jpg")}
            with open(TEST_IMG, "rb") as f:
                r = client.post("/api/detect/image",
                                files={"file": (TEST_IMG.name, f, "image/jpeg")},
                                data={"lab_id": "1"})
            body = r.json() if r.status_code == 200 else {}
            new_files = {p.name for p in SHOT_DIR.glob("img_*.jpg")} - before
            viol_events = [e for e in body.get("events", [])
                           if e.get("severity") != "正常"]

            report("检测成功", r.status_code == 200, f"HTTP {r.status_code}")
            report("多个违规事件只新增 1 张截图（去重生效）",
                   len(new_files) == 1,
                   f"违规事件数={len(viol_events)} 新增截图数={len(new_files)} "
                   f"→ {sorted(new_files)}")

            shots = {e.get("screenshot") for e in viol_events}
            report("所有违规事件共用同一截图文件名",
                   len(shots) == 1, f"出现的截图名={shots}")

            if new_files:
                p = SHOT_DIR / sorted(new_files)[0]
                report("截图文件名符合 img_*.jpg 约定",
                       p.name.startswith("img_") and p.suffix == ".jpg",
                       f"文件名={p.name} 大小={p.stat().st_size / 1024:.0f} KB")

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
