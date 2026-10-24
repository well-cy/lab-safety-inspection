# -*- coding: utf-8 -*-
"""
统计接口正确性测试（新增独立文件）。

运行：venv/bin/python tests/test_statistics_api.py

背景（本轮测试发现的真实缺陷）：
  BUG-1  by_type 把多类型违规记录整体分组
         原实现 `GROUP BY violation_types` 作用于逗号分隔字符串，
         导致图表出现 "未佩戴手套,未穿实验服,未佩戴口罩" 这种合并类别，
         与 INTERFACE_CONTRACT 6.3 示例（每项为单个类型）不符，
         "今日违规类型统计" 图表数据无意义。
  BUG-2  statistics().by_day 取到的是**最早** 30 天
         原实现 `ORDER BY d ASC LIMIT 30`，实测造 40 天数据时返回
         08-24~09-22，完全不含今天；数据一超过 30 天，
         统计页就会显示过时区间并丢失最新数据。
         正确语义应取「最近 30 天」。

本用例用「差值断言」（先测基线再测增量），因此不受演示库中已有数据干扰。

注意：本文件为新增，不修改 tests/run_tests.py 的 T01~T14。
"""
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import database as db  # noqa: E402

RESULTS = []
TODAY = date.today().isoformat()


def report(name, ok, detail=""):
    RESULTS.append({"name": name, "ok": ok, "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        print(f"       {detail}")


def _insert(conn, day: str, types: str, severity: str = "一般违规") -> int:
    cur = conn.execute(
        "INSERT INTO detection_records(lab_id, source_type, source_name,"
        " person_count, violation_count, normal_count, process_time_ms, created_at)"
        " VALUES(1,'image','stats_test',1,1,0,1.0,?)", (f"{day} 08:00:00",))
    rid = cur.lastrowid
    conn.execute(
        "INSERT INTO violation_events(record_id, lab_id, area_name, person_id,"
        " violation_types, severity, screenshot_path, frame_time, created_at)"
        " VALUES(?,1,'实验操作区',1,?,?,'',0,?)",
        (rid, types, severity, f"{day} 08:00:00"))
    return rid


def types_to_dict(items):
    return {i["t"]: i["c"] for i in items}


def main():
    created = []
    conn = db.get_conn()

    # ---------- 造数据：3 条多类型记录（今天） ----------
    created.append(_insert(conn, TODAY, "未佩戴口罩"))
    created.append(_insert(conn, TODAY, "未穿实验服,未佩戴口罩"))
    created.append(_insert(conn, TODAY, "未佩戴手套,未穿实验服,未佩戴口罩"))
    # 一条「正常」评估记录：用于验证 by_severity 的结构
    # （演示库中不一定存在正常记录，不能假设它天然存在）
    created.append(_insert(conn, TODAY, "", "正常"))
    # ---------- 造数据：40 天，用于验证「最近 30 天」 ----------
    for i in range(40):
        d = (date.today() - timedelta(days=i)).isoformat()
        created.append(_insert(conn, d, "未佩戴口罩"))
    conn.commit()
    conn.close()

    try:
        print("=" * 68)
        print("统计接口正确性测试")
        print("=" * 68)

        # ---------- BUG-1：by_type 必须是单项类型 ----------
        st = db.statistics()
        by_type = types_to_dict(st["by_type"])

        merged = [t for t in by_type if "," in t]
        report("BUG-1 statistics.by_type 不含合并类别（无逗号）",
               not merged, f"含逗号的类别={merged or '无'}")

        dash = db.dashboard_stats()
        dash_type = types_to_dict(dash["by_type"])
        merged_d = [t for t in dash_type if "," in t]
        report("BUG-1 dashboard.by_type 不含合并类别（无逗号）",
               not merged_d, f"含逗号的类别={merged_d or '无'}")

        # ---------- BUG-1：计数正确性（差值断言） ----------
        # 基线 = 造数据之前的值；现在再插 3 条已知记录，比较增量更直观：
        #   新增记录贡献：未佩戴口罩 +3，未穿实验服 +2，未佩戴手套 +1
        before = types_to_dict(db.statistics()["by_type"])
        conn = db.get_conn()
        extra = [_insert(conn, TODAY, "未佩戴口罩"),
                 _insert(conn, TODAY, "未穿实验服,未佩戴口罩"),
                 _insert(conn, TODAY, "未佩戴手套,未穿实验服,未佩戴口罩")]
        conn.commit()
        conn.close()
        created.extend(extra)
        after = types_to_dict(db.statistics()["by_type"])

        deltas = {k: after.get(k, 0) - before.get(k, 0)
                  for k in set(before) | set(after)}
        expect = {"未佩戴口罩": 3, "未穿实验服": 2, "未佩戴手套": 1}
        ok = all(deltas.get(k, 0) == v for k, v in expect.items())
        report("BUG-1 多类型记录按单项分别计数", ok,
               f"实际增量={ {k: v for k, v in deltas.items() if v} } 期望={expect}")

        # ---------- BUG-2：by_day 取最近 30 天 ----------
        st = db.statistics()
        days = [i["d"] for i in st["by_day"]]
        report("BUG-2 by_day 条数为 30", len(days) == 30, f"实际={len(days)}")
        report("BUG-2 by_day 包含今天", days and days[-1] == TODAY,
               f"最后一天={days[-1] if days else '-'} 今天={TODAY}")
        oldest_seeded = (date.today() - timedelta(days=39)).isoformat()
        report("BUG-2 by_day 不包含最早的 40 天前数据",
               oldest_seeded not in days,
               f"40 天前={oldest_seeded}，是否出现={oldest_seeded in days}")
        report("BUG-2 by_day 按时间升序（便于前端画折线）",
               days == sorted(days), f"首={days[0] if days else '-'} 末={days[-1] if days else '-'}")
        report("BUG-2 by_day 覆盖含今天在内的连续 30 天",
               days == [(date.today() - timedelta(days=i)).isoformat()
                        for i in range(29, -1, -1)],
               f"期望 {TODAY} 往前 30 天连续覆盖")

        # ---------- dashboard 的 by_day 时间窗 ----------
        d_days = [i["d"] for i in db.dashboard_stats()["by_day"]]
        report("dashboard.by_day 包含今天且不超过 7 天",
               TODAY in d_days and len(d_days) <= 7,
               f"天数={len(d_days)} 含今天={TODAY in d_days}")

        # ---------- by_severity 结构未变 ----------
        sev = {i["s"] for i in st["by_severity"]}
        report("by_severity 含「正常」且结构未变",
               "正常" in sev and "一般违规" in sev, f"取值={sorted(sev)}")

        # ---------- 排序稳定性（同计数时按类型名升序，保证结果可重复） ----------
        runs = [tuple((i["t"], i["c"]) for i in db.statistics()["by_type"])
                for _ in range(3)]
        report("by_type 连续 3 次调用结果完全一致", len(set(runs)) == 1,
               f"不同结果数={len(set(runs))}")

    finally:
        conn = db.get_conn()
        for rid in created:
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
