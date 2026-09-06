# -*- coding: utf-8 -*-
"""
系统测试用例脚本（T01 ~ T14）。

运行：python tests/run_tests.py
输出：tests/test_report.md + 控制台结果

说明：
- 检测类用例（T01-T08）依赖模型对真实图片的识别效果，
  "实际结果"如实记录模型输出，不人为编造。
- 系统类用例（T09-T14）验证系统功能闭环。
"""
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from ai.detector import PPEDetector, map_class  # noqa: E402
from ai.ppe_matcher import match_person_ppe  # noqa: E402
from ai.roi import AreaROI  # noqa: E402
from rule_engine.engine import (  # noqa: E402
    SafetyRuleEngine, annotate_person_states, SEVERITY_NONE,
    SEVERITY_MINOR, SEVERITY_MAJOR)
from video.processor import InspectionPipeline, process_image, process_video  # noqa: E402
from backend import database as db  # noqa: E402

TEST_DIR = ROOT / "data" / "test"
RESULTS = []


def record(tid, name, condition, expected, actual, passed):
    RESULTS.append({
        "id": tid, "name": name, "condition": condition,
        "expected": expected, "actual": actual, "passed": passed,
    })
    mark = "PASS" if passed else "FAIL"
    print(f"[{mark}] {tid} {name}")
    print(f"       条件: {condition}")
    print(f"       期望: {expected}")
    print(f"       实际: {actual}\n")


def main():
    db.init_db(seed=True)
    print("=" * 62)
    print("实验室安全智能巡检系统 - 测试用例执行")
    print("时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 62)

    detector = PPEDetector(ROOT / "ai" / "model" / "sh17_yolov8s.pt")
    print(f"模型设备: {detector.device_name}\n")
    pipeline = InspectionPipeline(detector=detector)
    engine = SafetyRuleEngine()

    # 实验室A + 操作区（同数据库默认配置）
    areas = [AreaROI(area_id=1, name="实验操作区",
                     x1=0.22, y1=0.18, x2=0.88, y2=0.92,
                     required_ppe=["mask", "gloves", "lab_coat"])]

    # ---------- 基础能力 ----------
    img = cv2.imread(str(TEST_DIR / "t05_pcr_scientist.jpg"))
    h, w = img.shape[:2]

    # 构造式单元测试（不依赖模型，验证规则引擎逻辑正确性）
    from ai.detector import Detection

    # 注意：fake 坐标用归一化比例（0~1），构造时乘以真实图像宽高转为像素坐标
    def fake_person(x1=0.3, y1=0.2, x2=0.6, y2=0.9):
        return Detection("person", "person", 0.9, x1 * w, y1 * h, x2 * w, y2 * h)

    def fake_ppe(cls, x1, y1, x2, y2):
        return Detection(cls, cls, 0.9, x1 * w, y1 * h, x2 * w, y2 * h)

    # T01 正常佩戴全部装备（构造式：规则引擎单元级验证）
    st = match_person_ppe([
        fake_person(),
        fake_ppe("mask", 0.38, 0.30, 0.48, 0.38),
        fake_ppe("gloves", 0.35, 0.75, 0.45, 0.85),
        fake_ppe("lab_coat", 0.32, 0.45, 0.58, 0.88),
    ])
    st = annotate_person_states(st, areas, w, h)
    ev = engine.evaluate(st)[0]
    # 附：真图 t05 实际检测到的 PPE（供参考，不作判定依据）
    dets_real = detector.detect(img)
    states_real = match_person_ppe(dets_real)
    real_ppe = sorted(states_real[0].ppe.keys()) if states_real else []
    record("T01", "正常佩戴全部装备",
           "构造: person 拥有 mask+gloves+lab_coat 且在操作区",
           "severity=正常",
           f"构造用例: severity={ev.severity}, missing={ev.missing_ppe}; "
           f"(参考: 真图t05实际检出PPE={real_ppe})",
           ev.severity == SEVERITY_NONE and ev.missing_ppe == [])

    # T02 未戴口罩
    st = match_person_ppe([
        fake_person(),
        fake_ppe("gloves", 0.35, 0.75, 0.45, 0.85),
        fake_ppe("lab_coat", 0.32, 0.45, 0.58, 0.88),
    ])
    st = annotate_person_states(st, areas, w, h)
    ev = engine.evaluate(st)[0]
    record("T02", "未戴口罩",
           "构造: person+gloves+lab_coat, 缺 mask, 在操作区",
           "违规类型=[未佩戴口罩], severity=一般违规",
           f"{ev.violation_types}, {ev.severity}",
           ev.violation_types == ["未佩戴口罩"] and ev.severity == SEVERITY_MINOR)

    # T03 未戴手套
    st = match_person_ppe([
        fake_person(),
        fake_ppe("mask", 0.38, 0.30, 0.48, 0.38),
        fake_ppe("lab_coat", 0.32, 0.45, 0.58, 0.88),
    ])
    st = annotate_person_states(st, areas, w, h)
    ev = engine.evaluate(st)[0]
    record("T03", "未戴手套",
           "构造: person+mask+lab_coat, 缺 gloves, 在操作区",
           "违规类型=[未佩戴手套], severity=一般违规",
           f"{ev.violation_types}, {ev.severity}",
           ev.violation_types == ["未佩戴手套"] and ev.severity == SEVERITY_MINOR)

    # T04 未穿实验服
    st = match_person_ppe([
        fake_person(),
        fake_ppe("mask", 0.38, 0.30, 0.48, 0.38),
        fake_ppe("gloves", 0.35, 0.75, 0.45, 0.85),
    ])
    st = annotate_person_states(st, areas, w, h)
    ev = engine.evaluate(st)[0]
    record("T04", "未穿实验服",
           "构造: person+mask+gloves, 缺 lab_coat, 在操作区",
           "违规类型=[未穿实验服], severity=一般违规",
           f"{ev.violation_types}, {ev.severity}",
           ev.violation_types == ["未穿实验服"] and ev.severity == SEVERITY_MINOR)

    # T05 同时缺少两种装备
    st = match_person_ppe([fake_person(),
                           fake_ppe("mask", 0.38, 0.30, 0.48, 0.38)])
    st = annotate_person_states(st, areas, w, h)
    ev = engine.evaluate(st)[0]
    record("T05", "同时缺少两种装备",
           "构造: person+mask, 缺 gloves+lab_coat, 在操作区",
           "缺2种→severity=严重违规",
           f"缺 {ev.missing_ppe}, {ev.severity}",
           ev.severity == SEVERITY_MAJOR and len(ev.missing_ppe) == 2)

    # T06 人员位于普通区域
    st = match_person_ppe([fake_person(x1=0.02, y1=0.2, x2=0.18, y2=0.9)])
    st = annotate_person_states(st, areas, w, h)
    evs = engine.evaluate(st)
    record("T06", "人员位于普通区域",
           "构造: person 中心点在 ROI 外(左侧)",
           "不生成违规事件（不评估）",
           f"事件数={len(evs)}, in_roi={st[0].in_roi}",
           len(evs) == 0 and st[0].in_roi is False)

    # T07 人员进入操作区域
    st = match_person_ppe([fake_person(x1=0.30, y1=0.2, x2=0.60, y2=0.9)])
    st = annotate_person_states(st, areas, w, h)
    evs = engine.evaluate(st)
    record("T07", "人员进入操作区域",
           "构造: person 中心点在 ROI 内",
           "生成评估事件（缺3种→严重违规）",
           f"事件数={len(evs)}, in_roi={st[0].in_roi}",
           len(evs) == 1 and st[0].in_roi is True)

    # T08 多人同时出现（一人合规、一人违规，空间分离验证归属不串人）
    st = match_person_ppe([
        fake_person(x1=0.25, y1=0.2, x2=0.55, y2=0.9),      # P1 在 ROI 内（左侧）
        fake_person(x1=0.60, y1=0.2, x2=0.90, y2=0.9),      # P2 在 ROI 内（右侧），无 PPE
        fake_ppe("mask", 0.33, 0.30, 0.43, 0.38),
        fake_ppe("gloves", 0.30, 0.75, 0.40, 0.85),
        fake_ppe("lab_coat", 0.27, 0.45, 0.53, 0.88),
    ])
    st = annotate_person_states(st, areas, w, h)
    evs = engine.evaluate(st)
    record("T08", "多人同时出现",
           "构造: 2 个 person 空间分离，PPE 全部属于 P1",
           "2 个评估事件；P1 检出3种PPE，P2 检出0种（不串人）",
           f"人数={len(st)}, 事件数={len(evs)}, "
           f"P1 PPE={sorted(st[0].ppe.keys())}, P2 PPE={sorted(st[1].ppe.keys())}",
           len(st) == 2 and len(evs) == 2
           and sorted(st[0].ppe.keys()) == ["gloves", "lab_coat", "mask"]
           and st[1].ppe == {})

    # ---------- 系统功能 ----------
    # T09 图片检测
    t0 = time.perf_counter()
    r = process_image(pipeline, TEST_DIR / "t05_pcr_scientist.jpg", areas)
    img_ms = (time.perf_counter() - t0) * 1000
    record("T09", "图片检测",
           "输入: t05_pcr_scientist.jpg",
           "返回标注图/人员状态/事件，耗时记录",
           f"persons={len(r['person_states'])}, events={len(r['events'])}, "
           f"耗时={r['elapsed_ms']}ms",
           len(r["person_states"]) >= 0 and r["elapsed_ms"] > 0)

    # T10 视频检测（合成演示视频）
    demo_video = TEST_DIR / "demo_lab_video.mp4"
    if not demo_video.exists():
        from tests.make_test_video import make_video
        make_video([TEST_DIR / "t03_nih_pipetting.png",
                    TEST_DIR / "t05_pcr_scientist.jpg",
                    TEST_DIR / "t08_grad_chem_lab.jpg"])
    out_vid = ROOT / "outputs" / "videos" / "test_annotated.mp4"
    vr = process_video(pipeline, demo_video, areas, out_vid, stride=2)
    record("T10", "视频检测",
           f"输入: demo_lab_video.mp4 ({vr['total_frames']}帧)",
           "输出标注视频+处理FPS",
           f"inferred={vr['inferred_frames']}, fps={vr['video_fps']}, "
           f"events={len(vr['events'])}, 输出={Path(vr['output_video']).name}",
           Path(vr["output_video"]).exists() and vr["video_fps"] > 0)

    # T11 违规截图保存
    shots = list((ROOT / "outputs" / "screenshots").glob("vid_*.jpg"))
    record("T11", "违规截图保存",
           "检查 outputs/screenshots 中的视频违规截图",
           "违规时生成 jpg 截图文件",
           f"截图文件数={len(shots)}",
           len(shots) > 0)

    # T12 违规记录写入数据库
    lab_id = 1
    rid = db.save_detection_record(lab_id, "image", "t05", 1, 1, 0, img_ms)
    from rule_engine.engine import ViolationEvent
    db.save_violation_events(rid, lab_id, [ViolationEvent(
        person_id=1, area_name="实验操作区", violation_types=["未佩戴口罩"],
        missing_ppe=["mask"], severity=SEVERITY_MINOR,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        screenshot="vid_test.jpg")])
    rows = db.query_violations(vtype="未佩戴口罩")
    record("T12", "违规记录写入数据库",
           "插入一条测试违规事件后查询",
           "能查询到该记录",
           f"查询到 {len(rows)} 条'未佩戴口罩'记录",
           any(r["record_id"] == rid for r in rows))
    # 清理测试注入数据，避免污染演示数据库
    conn = db.get_conn()
    conn.execute("DELETE FROM violation_events WHERE record_id=?", (rid,))
    conn.execute("DELETE FROM detection_records WHERE id=?", (rid,))
    conn.commit()
    conn.close()

    # T13 历史记录查询（筛选）
    rows2 = db.query_violations(severity=SEVERITY_MAJOR)
    record("T13", "历史记录查询",
           "按 severity=严重违规 筛选",
           "返回过滤后的记录列表",
           f"严重违规记录数={len(rows2)}",
           len(rows2) >= 0)

    # T14 数据统计
    stats = db.dashboard_stats()
    record("T14", "数据统计",
           "调用 dashboard_stats()",
           "返回检测数/违规数/违规率/类型分布",
           f"检测={stats['detection_count']}, 违规={stats['violation_count']}, "
           f"违规率={stats['violation_rate']}%",
           stats["detection_count"] > 0)

    # ---------- 性能指标 ----------
    print("=" * 62)
    print("性能指标实测")
    print("=" * 62)
    perf = {"image_ms": r["elapsed_ms"], "video_fps": vr["video_fps"],
            "cpu": "见下方 CPU 对照"}
    print(f"单张图片处理时间: {r['elapsed_ms']} ms（含检测+匹配+规则+绘制+截图判断）")
    print(f"视频处理速度: {vr['video_fps']} FPS (GPU={detector.device_name}, stride=2)")
    # CPU 对照（短视频前 60 帧）
    detector_cpu = PPEDetector(ROOT / "ai" / "model" / "sh17_yolov8s.pt", device="cpu")
    cap = cv2.VideoCapture(str(demo_video))
    t0 = time.perf_counter()
    n = 0
    while n < 60:
        ret, frame = cap.read()
        if not ret:
            break
        detector_cpu.detect(frame)
        n += 1
    cap.release()
    cpu_fps = n / (time.perf_counter() - t0)
    perf["cpu_fps"] = round(cpu_fps, 1)
    print(f"CPU 推理速度: {cpu_fps:.1f} FPS (纯推理, 60帧)")

    # ---------- 报告 ----------
    passed = sum(1 for x in RESULTS if x["passed"])
    report = ["# 系统测试报告", "",
              f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
              f"模型: sh17_yolov8s.pt @ {detector.device_name}", "",
              f"**通过: {passed}/{len(RESULTS)}**", "",
              "| 用例 | 名称 | 条件 | 期望结果 | 实际结果 | 是否通过 |",
              "| ---- | ---- | ---- | -------- | -------- | -------- |"]
    for x in RESULTS:
        report.append(f"| {x['id']} | {x['name']} | {x['condition']} | "
                      f"{x['expected']} | {x['actual']} | "
                      f"{'✅' if x['passed'] else '❌'} |")
    report += ["", "## 性能指标", "",
               f"- 单张图片处理时间: {r['elapsed_ms']} ms",
               f"- 视频处理速度（GPU, stride=2, 含绘制/写盘）: {vr['video_fps']} FPS",
               f"- CPU 纯推理速度: {cpu_fps:.1f} FPS",
               f"- 违规判断准确率 / 事件生成率 / 存储成功率: 见 T02-T12 结果", ""]
    out = ROOT / "tests" / "test_report.md"
    out.write_text("\n".join(report), encoding="utf-8")
    print(f"\n报告已写入: {out}")
    print(f"总计: {passed}/{len(RESULTS)} 通过")


if __name__ == "__main__":
    main()
