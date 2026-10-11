# -*- coding: utf-8 -*-
"""
配置接口（settings）收敛与顺序稳定性测试（新增独立文件）。

运行：venv/bin/python tests/test_settings_api.py

背景：
  原实现中 `GET/POST/DELETE /api/settings*` 的 SQL 直接写在 backend/main.py 里
  （技术债清单 #5「设置端点裸 SQL 绕过 database 层」），且查询没有 ORDER BY，
  导致 regions 的 required_ppe 返回顺序不固定 —— 实测返回
  ["gloves","lab_coat","mask"]，与 INTERFACE_CONTRACT.md 6.8 示例的
  ["mask","gloves","lab_coat"] 不一致。

本用例验证：
  1. SQL 收敛后接口行为与响应结构**完全不变**（防回归）
  2. required_ppe 顺序稳定，且与契约示例一致
  3. 区域新建/删除的完整闭环与级联清理
  4. 参数校验仍然生效

注意：本文件为新增，不修改 tests/run_tests.py 的 T01~T14。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from backend import database as db  # noqa: E402
from backend.main import app  # noqa: E402

RESULTS = []

TEST_LAB_NAME = "自动化测试实验室_请勿手工保留"
TEST_AREA_NAME = "自动化测试区域"


def report(name, ok, detail=""):
    RESULTS.append({"name": name, "ok": ok, "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        print(f"       {detail}")


def cleanup(area_ids, lab_ids):
    conn = db.get_conn()
    for aid in area_ids:
        conn.execute("DELETE FROM safety_rules WHERE area_id=?", (aid,))
        conn.execute("DELETE FROM areas WHERE id=?", (aid,))
    for lid in lab_ids:
        conn.execute("DELETE FROM areas WHERE lab_id=?", (lid,))
        conn.execute("DELETE FROM laboratories WHERE id=?", (lid,))
    conn.commit()
    conn.close()


def main():
    created_areas = []
    created_labs = []
    try:
        with TestClient(app) as client:
            print("=" * 66)
            print("配置接口（settings）收敛与顺序稳定性测试")
            print("=" * 66)

            # ---------- 1. 响应结构与顺序（回归 + 顺序修复） ----------
            r = client.get("/api/settings")
            body = r.json()
            report("GET /api/settings 返回 200", r.status_code == 200,
                   f"HTTP {r.status_code}")

            labs = body.get("labs", [])
            lab_keys = set(labs[0].keys()) if labs else set()
            ok = {"id", "name", "description", "created_at", "areas"} <= lab_keys
            report("实验室字段结构未变", ok, f"字段={sorted(lab_keys)}")

            areas = labs[0].get("areas", []) if labs else []
            area_keys = set(areas[0].keys()) if areas else set()
            ok = ({"id", "lab_id", "name", "x1", "y1", "x2", "y2",
                   "created_at", "required_ppe"} <= area_keys)
            report("区域字段结构未变", ok, f"字段={sorted(area_keys)}")

            # 核心：顺序必须与契约 6.8 一致
            seed_area = next((a for a in areas if a["name"] == "实验操作区"), None)
            expect = ["mask", "gloves", "lab_coat"]
            ok = seed_area is not None and seed_area["required_ppe"] == expect
            report("required_ppe 顺序与契约 6.8 一致（顺序修复）", ok,
                   f"实际={seed_area['required_ppe'] if seed_area else '未找到'} "
                   f"期望={expect}")

            # 连续调用 3 次，顺序必须稳定
            orders = {tuple(a["required_ppe"])
                      for _ in range(3)
                      for a in client.get("/api/settings").json()["labs"][0]["areas"]
                      if a["name"] == "实验操作区"}
            report("连续 3 次查询顺序稳定", len(orders) == 1,
                   f"出现过的顺序集合={orders}")

            # ---------- 2. 新建实验室 ----------
            r = client.post("/api/settings/lab",
                            data={"name": TEST_LAB_NAME, "description": "自动化用例"})
            report("POST /api/settings/lab 新建成功",
                   r.status_code == 200 and r.json().get("ok") is True,
                   f"HTTP {r.status_code} body={r.text[:60]}")
            new_lab = next((x for x in db.list_labs_with_areas()
                            if x["name"] == TEST_LAB_NAME), None)
            if new_lab:
                created_labs.append(new_lab["id"])

            r = client.post("/api/settings/lab", data={"name": TEST_LAB_NAME})
            report("POST /api/settings/lab 重名返回 400",
                   r.status_code == 400 and "error" in r.json(),
                   f"HTTP {r.status_code} body={r.text[:60]}")

            # ---------- 3. 新建区域 + 顺序验证 ----------
            lab_id = created_labs[0] if created_labs else 1
            r = client.post("/api/settings/area",
                            data={"lab_id": str(lab_id), "name": TEST_AREA_NAME,
                                  "x1": "0.1", "y1": "0.2", "x2": "0.6", "y2": "0.8",
                                  "required_ppe": "goggles,helmet"})
            area_id = r.json().get("area_id") if r.status_code == 200 else None
            report("POST /api/settings/area 新建成功并返回 area_id",
                   r.status_code == 200 and isinstance(area_id, int),
                   f"HTTP {r.status_code} area_id={area_id}")
            if area_id:
                created_areas.append(area_id)

            area = next((a for lab in client.get("/api/settings").json()["labs"]
                         if lab["id"] == lab_id
                         for a in lab["areas"] if a["id"] == area_id), None)
            ok = area is not None and area["required_ppe"] == ["goggles", "helmet"]
            report("新建区域 required_ppe 保持配置顺序",
                   ok, f"实际={area['required_ppe'] if area else '未找到'}")

            coords_ok = (area is not None
                         and (area["x1"], area["y1"], area["x2"], area["y2"])
                         == (0.1, 0.2, 0.6, 0.8))
            report("新建区域坐标正确落库", coords_ok,
                   f"实际={[area[k] for k in ('x1', 'y1', 'x2', 'y2')] if area else '未找到'}")

            # ---------- 4. 参数校验仍然生效 ----------
            cases = [
                ("坐标越界 → 400",
                 {"lab_id": str(lab_id), "name": "越界", "x1": "1.5", "y1": "0",
                  "x2": "0.9", "y2": "0.9"}, 400),
                ("坐标反向 → 400",
                 {"lab_id": str(lab_id), "name": "反向", "x1": "0.8", "y1": "0.1",
                  "x2": "0.2", "y2": "0.9"}, 400),
                ("非法 PPE → 400",
                 {"lab_id": str(lab_id), "name": "非法PPE", "x1": "0.1", "y1": "0.1",
                  "x2": "0.5", "y2": "0.5", "required_ppe": "gun"}, 400),
                ("实验室不存在 → 400",
                 {"lab_id": "999999", "name": "孤儿", "x1": "0.1", "y1": "0.1",
                  "x2": "0.5", "y2": "0.5"}, 400),
            ]
            for name, data, expect in cases:
                rr = client.post("/api/settings/area", data=data)
                report(f"POST /api/settings/area {name}",
                       rr.status_code == expect,
                       f"HTTP {rr.status_code}（期望 {expect}）")

            # ---------- 5. 删除区域（含级联清理） ----------
            if area_id:
                r = client.delete(f"/api/settings/area/{area_id}")
                report("DELETE /api/settings/area 返回 ok",
                       r.status_code == 200 and r.json().get("ok") is True,
                       f"HTTP {r.status_code} body={r.text[:40]}")

                conn = db.get_conn()
                left = conn.execute("SELECT COUNT(*) c FROM safety_rules"
                                    " WHERE area_id=?", (area_id,)).fetchone()["c"]
                conn.close()
                report("删除区域级联清理 safety_rules", left == 0,
                       f"残留规则数={left}")

                gone = all(a["id"] != area_id
                           for lab in client.get("/api/settings").json()["labs"]
                           for a in lab["areas"])
                report("删除后区域不再出现在列表中", gone, "")
                created_areas.remove(area_id)

            r = client.delete("/api/settings/area/999999")
            report("DELETE 不存在的区域保持幂等（200）",
                   r.status_code == 200, f"HTTP {r.status_code}")

            # ---------- 6. 回归：检测与统计接口不受影响 ----------
            for path in ("/api/dashboard", "/api/statistics", "/api/violations",
                         "/api/model/info"):
                rr = client.get(path)
                report(f"回归 - GET {path}", rr.status_code == 200,
                       f"HTTP {rr.status_code}")

    finally:
        cleanup(created_areas, created_labs)

    passed = sum(1 for x in RESULTS if x["ok"])
    total = len(RESULTS)
    print("=" * 66)
    print(f"结果: {passed}/{total} 通过")
    if passed != total:
        print("失败用例:")
        for x in RESULTS:
            if not x["ok"]:
                print(f"  - {x['name']}: {x['detail']}")
    print("=" * 66)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
