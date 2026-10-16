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

后续补充（均为网页实测报图的回归守卫）：
     BUG-9  跨类 NMS 抑制穿白大褂的人（"不管多少人都只显示一个人"）
     BUG-10 倒影过滤器误杀"长在身上的真手套"
     BUG-11 坐姿者 PPE 被重叠站立者小框整体抢走
     BUG-12 画面底部的真手被"鞋/脚误检"规则无差别误杀
     BUG-13 切片复检的高包含率重复框逃逸（IoU 不够但 100% 被包含）
     BUG-14 手套错归：伸出框外的手被"精确包含优先"判给旁人（成对先验修复）
     BUG-15 成对先验副作用：相邻两人的手套全被先绑者吸走（满额上限 2 只修复）
     BUG-16 手套中心压 person 框缘 0.02px → 无精确命中时按重叠占比消歧
     BUG-17 重叠人群真人置信度被联合框压低 → person 按类阈值单独放宽（0.28）
     BUG-18 普通眼镜误判护目镜（conf 0.389 过 0.35 阈值）→ goggles 按类阈值收紧（0.44）
     BUG-19 成对先验无空间约束 → 相距很远的两人互抢手套（验收报告 S2/S3/S11/S12）
     BUG-21 验收复审：40% 扩张仍会抢部分重叠邻人的手套（S3/S12）、满额后零交集
            旁人顶替抢手套（S11）→ sibling 收紧 20% + 空间命中硬过滤 + 兜底距离改到框边
     BUG-22 验收再复审：环带内零交集手套仍被抢 → sibling 追加 overlap>0，
            S3/S12 测试点移进 20% 边界内侧
     BUG-23 验收三复审：1~20px 小交叠仍可被邻人 sibling 抢 → 收紧为相交占比
            overlap >= 0.2，并补 0<overlap<0.2 守卫（x1=580）与大幅相交正向对照
     BUG-20 联合框判定无面积约束 → 弯腰俯身三人簇中真人被"平级重叠"误杀

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
        # ============ BUG-9：跨类 NMS 抑制穿白大褂的人 ============
        # 历史 bug：_nms_merge 用单一 merged 列表做**跨类**比较，person 框
        # 会因与 lab_coat 框 IoU>0.55 而被丢弃（且 set 迭代顺序随机 →
        # 每次重启表现不同）。用户可见症状：「不管多少人都只显示一个人」。
        print("--- BUG-9 _nms_merge 必须按类独立 ---")
        from ai.detector import Detection, PPEDetector

        person_box = Detection(cls_name="person", raw_cls="person", conf=0.8,
                               x1=100, y1=100, x2=300, y2=600)
        coat_box = Detection(cls_name="lab_coat", raw_cls="lab_coat", conf=0.7,
                             x1=100, y1=100, x2=300, y2=580)  # 与 person IoU≈0.96
        merged = PPEDetector._nms_merge([person_box, coat_box], iou_thr=0.55)
        report("跨类 NMS 不再互相抑制（person + lab_coat 均保留）",
               len(merged) == 2 and {d.cls_name for d in merged} == {"person", "lab_coat"},
               f"输入 2 框（IoU≈0.96）→ 输出 {len(merged)} 框: {[d.cls_name for d in merged]}")

        dup_person = Detection(cls_name="person", raw_cls="person", conf=0.5,
                              x1=102, y1=102, x2=298, y2=598)   # 同类近重复
        merged2 = PPEDetector._nms_merge([person_box, dup_person], iou_thr=0.55)
        report("同类近重复框仍会被正确合并（保留高置信）",
               len(merged2) == 1 and merged2[0].conf == 0.8,
               f"2 个同类近重复 → {len(merged2)} 框，conf={merged2[0].conf if merged2 else None}")

        # 端到端：t09 合影真实应检出 >=5 人（修复前被 lab_coat 吃到 4 人）
        t09 = ROOT / "data" / "test" / "t09_lab_coat_ceremony.jpg"
        if t09.exists():
            from backend.main import get_image_pipeline
            from video.processor import process_image
            pipe = get_image_pipeline()
            res = process_image(pipe, t09, db.load_areas(config.DEFAULT_LAB_ID))
            n_person = len(res["person_states"])
            report("t09 合影人数不再被白大褂框吃掉（>=5 人）",
                   n_person >= 5, f"检出人数 = {n_person}（修复前为 4）")

        # ============ BUG-10：倒影过滤器误杀「长在身上的真手套」 ============
        # 历史 bug：_is_reflection 只看「手套在画面底部 25% + 正上方有 mask/goggles
        # 同位重叠」就判台面倒影。弯腰伏案者（4K 竖构图）自己的手在画面底部、
        # 正对自己上方的 mask，完美满足旧判据 → img_17c92cec 3 个 conf 0.71~0.87
        # 的真手套被全部误杀，网页显示「未佩戴手套」。
        print("--- BUG-10 _is_reflection 不得误杀手套（长在身上的手） ---")
        from ai.detector import Detection, PPEDetector as _PPE2

        # _is_reflection 只读类常量，用 __new__ 造个空实例即可（无需加载模型）
        _refl = _PPE2.__new__(_PPE2)

        def _mk(cls, cf, x1, y1, x2, y2):
            return Detection(cls_name=cls, raw_cls=cls, conf=cf,
                             x1=x1, y1=y1, x2=x2, y2=y2)

        # 场景 A（img_17c92cec 缩小复刻）：手套 71%+ 在 person 框内 + 上方同位 mask
        scene_a = [
            _mk("person", 0.86, 0, 0, 1000, 1000),
            _mk("mask",   0.74, 300, 300, 500, 380),
            _mk("gloves", 0.87, 300, 800, 460, 1000),   # y1=800 > 0.75*1000
        ]
        g = scene_a[2]
        report("身体上的手套不判倒影（≥50% 在 person 框内）",
               not _refl._is_reflection(g, scene_a, None, 1000),
               "gloves 与上方 mask 同位重叠但在 person 框内 → 必须保留")

        # 场景 B（真倒影）：手套浮在人身体框外的台面上
        scene_b = [
            _mk("person", 0.86, 0, 0, 600, 900),
            _mk("mask",   0.74, 300, 300, 500, 380),
            _mk("gloves", 0.66, 300, 950, 460, 990),    # 在 person 框(y2=900)之外
        ]
        g2 = scene_b[2]
        report("台面倒影仍会被正确过滤（person 框外）",
               _refl._is_reflection(g2, scene_b, None, 1000),
               "gloves 在 person 框外且上方同位 mask → 判倒影剔除")

        # ============ BUG-11：坐姿者 PPE 被重叠站立者小框整体抢走 ============
        print("--- BUG-11 match_person_ppe 解剖学消歧（坐姿 vs 站立重叠） ---")
        from ai.detector import Detection
        from ai.ppe_matcher import match_person_ppe as _match

        def _mk(cls, cf, x1, y1, x2, y2):
            return Detection(cls_name=cls, raw_cls=cls, conf=cf,
                             x1=x1, y1=y1, x2=x2, y2=y2)

        # img_d020e947 按比例缩小复刻：坐姿者大框（含椅子）+ 站立者小框
        # 与坐姿者上半身重叠。坐姿者的 mask/gloves 中心同时落在两个框内。
        p_seated = _mk("person", 0.83, 528, 115, 692, 500)   # 大框
        p_stand  = _mk("person", 0.74, 633,  77, 750, 273)   # 小框，重叠
        mask_s   = _mk("mask",   0.72, 623, 157, 650, 181)   # 中心(636,169) 坐姿者的
        mask_t   = _mk("mask",   0.78, 676, 118, 713, 147)   # 中心(694,133) 站立者的
        glove_s  = _mk("gloves", 0.83, 697, 243, 725, 263)   # 中心(710,253)

        sts = _match([p_seated, p_stand, mask_s, mask_t, glove_s])
        by_area = {s.bbox.area: s for s in sts}
        big = by_area[max(by_area)]      # 坐姿者
        small = by_area[min(by_area)]    # 站立者
        report("坐姿者的口罩不被站立者抢走（mask 在其头部高度）",
               "mask" in big.ppe,
               "mask 中心在坐姿者框 rel_y=0.14（脸），站立者框 rel_y=0.47（肚子，不合理）")
        report("站立者本人的 mask 保留（在其头部高度）",
               "mask" in small.ppe,
               "站立者自己的 mask 仍归站立者")
        report("显微伸手的手套归坐姿者（站立者框底部不合理）",
               "gloves" in big.ppe,
               "gloves 中心在站立者框 rel_y=0.90（脚部，不合理）→ 归坐姿者扩张框")

        # ============ BUG-12：底部手套被「鞋/脚误检」规则无差别误杀 ============
        print("--- BUG-12 底部真手不被 0.86 规则误杀（长在身上守卫） ---")
        import numpy as np
        from ai.detector import PPEDetector as _PPE3
        # 不加载模型的轻量实例（_filter_implausible 只依赖类级常量与方法）
        _det3 = object.__new__(_PPE3)

        def _mk2(cls, cf, x1, y1, x2, y2):
            return Detection(cls_name=cls, raw_cls=cls, conf=cf,
                             x1=x1, y1=y1, x2=x2, y2=y2)

        # img_89128025 复刻：主研究员的手在画面最底部（y1=0.88h），
        # 与 person 框重叠 40% 面积 → 是手不是鞋
        scene_c = [
            _mk2("person", 0.62, 380,   7, 762, 423),
            _mk2("gloves", 0.64, 303, 381, 453, 432),
        ]
        frame = np.zeros((432, 768, 3), dtype=np.uint8)
        kept = _det3._filter_implausible(scene_c, frame)
        report("画面底部但长在身上的手保留（与 person 重叠≥35%）",
               any(d.cls_name == "gloves" for d in kept),
               "特写构图手在画面底部 y1=0.88h，conf 0.637 不应被当鞋丢弃")

        # 对照：底部手套与 person 框无实质重叠（真正的鞋/脚场景）
        scene_d = [
            _mk2("person", 0.62, 380,   7, 762, 300),   # person 框在手套上方结束
            _mk2("gloves", 0.64, 303, 381, 453, 432),
        ]
        kept2 = _det3._filter_implausible(scene_d, frame)
        report("与人体无重叠的底部手套仍被过滤（鞋/脚）",
               not any(d.cls_name == "gloves" for d in kept2),
               "person 框在手套上方结束 → 底部手套判鞋/脚剔除")

        # ============ BUG-13：切片复检产生的「高包含率重复框」逃逸 ============
        # img_8e909ac5 实测：同一只右手产出 0.719(3347,1323,3795,1705) 与
        # 0.389(3515,1411,3791,1692) 两个框，IoU≈0.45 < 0.55（常规 NMS 阈值）
        # 但包含率 100% → 0.389 逃逸，随后被归属给旁边的女生，
        # 网页显示「她戴了手套」（用户报的「只有一人戴手套却归属给两人」）。
        # 修复：PPE 类追加「IoU≥0.40 且包含率≥0.85 即合并」。
        print("--- BUG-13 高包含率重复框合并（PPE 类） ---")
        from ai.detector import Detection, PPEDetector as _PPE4

        g_big = _mk2("gloves", 0.719, 3347, 1323, 3795, 1705)
        g_small = _mk2("gloves", 0.389, 3515, 1411, 3791, 1692)   # 100% 被大框包含
        m_out = _PPE4._nms_merge([g_big, g_small])
        report("同目标的高包含率重复框被合并（保留高置信）",
               len(m_out) == 1 and abs(m_out[0].conf - 0.719) < 1e-6,
               f"2 框（IoU≈0.45 / 包含率 100%）→ {len(m_out)} 框，"
               f"conf={[round(d.conf, 3) for d in m_out]}")

        # 对照①：错位重叠（真的是两只手）的手套框不应被合并
        g_a = _mk2("gloves", 0.70, 100, 100, 300, 300)
        g_b = _mk2("gloves", 0.60, 200, 200, 400, 400)   # IoU≈0.14、包含率 25%
        m_out2 = _PPE4._nms_merge([g_a, g_b])
        report("错位重叠的两只手套不误合并",
               len(m_out2) == 2, f"IoU≈0.14 → {len(m_out2)} 框")

        # 对照②：person 类不做包含合并（真人前后站位，后排只露半身，
        #        其小框 100% 落在前排大框内，但这是两个真人）
        p_big = _mk2("person", 0.90, 0, 0, 400, 250)
        p_small = _mk2("person", 0.80, 50, 50, 350, 200)  # 完全在内、IoU≈0.45
        m_out3 = _PPE4._nms_merge([p_big, p_small])
        report("person 前后站位不被包含合并（真人不是重复框）",
               len(m_out3) == 2,
               f"小 person 框 100% 落在大框内（IoU≈0.45）→ 仍保留 {len(m_out3)} 人")

        # ============ BUG-14：手套「成对先验」修错归 ============
        # img_8e909ac5 复刻：中间男的右手伸出自己框外、落进右边女生的框内，
        # 旧逻辑「精确包含优先」把这只手判给女生（她因此凭空多出一只手套）。
        # 中男的左手已先行归属本人 → 成对先验应让本人胜出。
        print("--- BUG-14 手套成对先验（伸出框外的手归本人） ---")
        from ai.ppe_matcher import match_person_ppe as _match2

        p_mid = _mk2("person", 0.78, 1963, 237, 3507, 2527)     # 中男（大框）
        p_right = _mk2("person", 0.83, 3270, 925, 4201, 2156)   # 右女（小框）
        g_left = _mk2("gloves", 0.795, 3003, 1637, 3514, 1875)  # 中男左手（框内）
        g_hand = _mk2("gloves", 0.719, 3347, 1323, 3795, 1705)  # 中男右手（伸进右女框）

        sts2 = _match2([p_mid, p_right, g_left, g_hand])
        mid = [s for s in sts2 if abs(s.bbox.area - p_mid.area) < 1e-6][0]
        right = [s for s in sts2 if abs(s.bbox.area - p_right.area) < 1e-6][0]
        report("伸出框外的手归本人（成对先验）",
               "gloves" in mid.ppe and "gloves" not in right.ppe,
               f"中男={sorted(mid.ppe)}，右女={sorted(right.ppe)}"
               "（修复前右女会被凭空赋予手套）")

        # 对照：中男自己没有任何手套检出时，不应越权抢右女框内的手套
        sts3 = _match2([p_mid, p_right, g_hand])
        mid3 = [s for s in sts3 if abs(s.bbox.area - p_mid.area) < 1e-6][0]
        right3 = [s for s in sts3 if abs(s.bbox.area - p_right.area) < 1e-6][0]
        report("无成对先验时不越权抢人（对照组）",
               "gloves" in right3.ppe and "gloves" not in mid3.ppe,
               f"中男={sorted(mid3.ppe)}，右女={sorted(right3.ppe)}")

        # ============ BUG-15：手套满额守卫（每人上限 2 只） ============
        # img_cb869a2c 复刻：两人相邻，女2 的两只手套都落在与男1 框的交界
        # 重叠区内。BUG-14 的成对先验让男1 每轮胜出 → 女2 两只全被吸走再被
        # 同类覆盖丢弃，网页表现为"只有 1 人戴手套"。
        # 手套是成对器官 → 已绑满 2 只的人必须退出竞争。
        print("--- BUG-15 手套满额守卫（每人上限 2 只） ---")
        from ai.ppe_matcher import match_person_ppe as _match3

        p_a = _mk2("person", 0.84, 686, 15, 1003, 643)    # 男1（左，大框）
        p_b = _mk2("person", 0.82, 975, 22, 1180, 644)    # 女2（右，小框）
        g1 = _mk2("gloves", 0.87, 777, 199, 838, 263)     # 男1 手 A（框内）
        g2 = _mk2("gloves", 0.81, 862, 300, 943, 351)     # 男1 手 B（框内）
        g3 = _mk2("gloves", 0.79, 964, 382, 1013, 447)    # 女2 手 A（交界，两人框内）
        g4 = _mk2("gloves", 0.67, 978, 430, 1047, 484)    # 女2 手 B（交界偏女2）

        sts4 = _match3([p_a, p_b, g1, g2, g3, g4])
        a = [s for s in sts4 if abs(s.bbox.area - p_a.area) < 1e-6][0]
        b = [s for s in sts4 if abs(s.bbox.area - p_b.area) < 1e-6][0]
        report("两人都戴手套时各自归属（满额守卫）",
               "gloves" in a.ppe and "gloves" in b.ppe,
               f"男1={sorted(a.ppe)}，女2={sorted(b.ppe)}"
               "（修复前女2 的两只手套被成对先验全部吸走）")

        # 守卫不误伤：3 只手套、两人相邻时，第 3 只按普通规则走（不凭空消失）
        sts5 = _match3([p_a, p_b, g1, g3, g4])
        a5 = [s for s in sts5 if abs(s.bbox.area - p_a.area) < 1e-6][0]
        b5 = [s for s in sts5 if abs(s.bbox.area - p_b.area) < 1e-6][0]
        n_g = (1 if "gloves" in a5.ppe else 0) + (1 if "gloves" in b5.ppe else 0)
        report("未满额时先验照常生效",
               n_g == 2,
               f"男1={sorted(a5.ppe)}，女2={sorted(b5.ppe)}（2 人共绑定 {n_g} 只）")

        # ============ BUG-16：中心点压线 → 重叠占比消歧 ============
        # img_9491b9ea 复刻：左男的手套框中心 x=2880.6 恰好压在左侧大框
        # 右缘 x=2880.58 外 0.02 像素 → 两个候选都退化为"扩张框命中"，
        # 旧规则"面积最小"让小框（右女）赢走左男伸出的手。
        # 修复：无精确命中时改用 PPE 框与 person 框的重叠占比消歧
        # （手套 50% 落在左侧大框内、仅 24% 落在右女框内）。
        print("--- BUG-16 中心点压线时按重叠占比消歧 ---")
        from ai.ppe_matcher import match_person_ppe as _match4

        p_union = _mk2("person", 0.67, 271, 444, 2881, 3929)   # 左男+中男联合大框
        p_w = _mk2("person", 0.82, 3022, 1484, 4191, 3148)     # 右女（小框）
        g_edge = _mk2("gloves", 0.74, 2607, 2027, 3154, 2736)  # 左男的手（压线伸出）

        sts6 = _match4([p_union, p_w, g_edge])
        u6 = [s for s in sts6 if abs(s.bbox.area - p_union.area) < 1e-6][0]
        w6 = [s for s in sts6 if abs(s.bbox.area - p_w.area) < 1e-6][0]
        report("压线手套归重叠占比大的一方",
               "gloves" in u6.ppe and "gloves" not in w6.ppe,
               f"大框={sorted(u6.ppe)}，右女={sorted(w6.ppe)}"
               "（修复前右女因面积最小赢走手套）")

        # 对照：手套大部分在右女框内时，占比规则应同样支持右女
        g_right = _mk2("gloves", 0.70, 3050, 2000, 3400, 2250)  # 右女躯干前的手
        sts7 = _match4([p_union, p_w, g_right])
        u7 = [s for s in sts7 if abs(s.bbox.area - p_union.area) < 1e-6][0]
        w7 = [s for s in sts7 if abs(s.bbox.area - p_w.area) < 1e-6][0]
        report("重叠占比规则双向有效（对照组）",
               "gloves" in w7.ppe and "gloves" not in u7.ppe,
               f"大框={sorted(u7.ppe)}，右女={sorted(w7.ppe)}")

        # ============ BUG-17：按类阈值（person 单独放宽） ============
        # img_9491b9ea：重叠人群里真人 conf 被联合框压低（中男 0.290），
        # 而 PPE 类不能跟着降阈值（右女会被塞假 mask 0.339，把"严重违规"
        # 洗成"一般违规"）。方案 C：person 用 conf_person，其余类用 conf。
        print("--- BUG-17 按类阈值收口（person 单独放宽） ---")
        from ai.detector import PPEDetector as _PPED

        p_low = _mk2("person", 0.290, 1721, 1253, 2847, 3306)   # 真人（低置信）
        p_ok = _mk2("person", 0.820, 3022, 1484, 4191, 3148)    # 真人（高置信）
        m_fake = _mk2("mask", 0.339, 3282, 2012, 3532, 2154)    # PPE 假阳性

        kept = _PPED._apply_class_thresholds([p_low, p_ok, m_fake], 0.35, 0.28)
        names = sorted(d.cls_name for d in kept)
        report("person 按独立阈值保留，PPE 类维持原阈值",
               names == ["person", "person"],
               f"0.35/0.28 收口后保留 {names}"
               "（低置信真人 0.290 保留；假 mask 0.339 被 0.35 挡掉）")

        # 守卫：conf_person 为 None 时短路（阈值已由上层 predict 承担，
        # 此处不再二次过滤，避免改变旧行为）
        kept2 = _PPED._apply_class_thresholds([p_low, p_ok, m_fake], 0.35, None)
        names2 = sorted(d.cls_name for d in kept2)
        report("未配置 conf_person 时不二次过滤（旧行为短路）",
               names2 == ["mask", "person", "person"],
               f"None → 原样返回 {names2}（真实流程里低置信框已被 predict 的 0.35 挡掉）")

        # 边界：conf_person 高于 conf 时也不应误挡 PPE（各自独立判断）
        m_ok = _mk2("mask", 0.45, 3282, 2012, 3532, 2154)      # 高于 0.35 的 PPE
        kept3 = _PPED._apply_class_thresholds([p_low, p_ok, m_ok], 0.35, 0.90)
        names3 = sorted(d.cls_name for d in kept3)
        report("conf_person 与 conf 相互独立",
               names3 == ["mask"],
               f"0.35/0.90 → 保留 {names3}（person 被 0.90 挡、mask 0.45 按 0.35 保留）")

        # ============ BUG-18：goggles 按类阈值收紧（普通眼镜误判护目镜） ============
        # img_169f81a5：中间男士戴普通近视镜，模型判成护目镜（conf 0.389 >
        # 全局 0.35 → 凭空多出"佩戴护目镜"）；右侧女生眼镜 0.251 险过。
        # 实测：假 ≤0.389、真 ≥0.497（img_9a04e15b，首版 0.50 曾把它漏掉），
        # → 夹逼取 0.44（唯一可行区间 0.39~0.49 的中点）。
        print("--- BUG-18 goggles 按类阈值收紧（普通眼镜假阳性） ---")

        g_fake = _mk2("goggles", 0.389, 2414, 453, 2927, 722)  # 近视镜误检
        g_edge = _mk2("goggles", 0.497, 308, 90, 518, 157)     # 真护目镜（0.50 曾漏掉）
        g_real = _mk2("goggles", 0.703, 1076, 1097, 1760, 1623)  # 真护目镜
        m_std = _mk2("mask", 0.40, 500, 500, 700, 600)          # 普通 PPE 对照

        kept4 = _PPED._apply_class_thresholds([g_fake, g_edge, g_real, m_std],
                                              0.35, 0.28, 0.44)
        names4 = sorted(d.cls_name for d in kept4)
        report("goggles 按 0.44 收口：眼镜误检被挡、两真护目镜保留",
               names4 == ["goggles", "goggles", "mask"],
               f"0.35/0.28/0.44 收口后保留 {names4}"
               "（假 goggles 0.389 被挡；0.497/0.703 保留——0.44 是唯一可行区间中点）")

        # 独立性：未配置 conf_goggles 时 goggles 仍按全局 conf 判断
        kept5 = _PPED._apply_class_thresholds([g_fake, g_real], 0.35, 0.28, None)
        names5 = sorted(d.cls_name for d in kept5)
        report("未配置 conf_goggles 时 goggles 不受影响",
               names5 == ["goggles", "goggles"],
               f"None → goggles 按 0.35 保留 {names5}（与 conf_person 独立）")

        # 守卫：conf_person 与 conf_goggles 同时为默认语义时短路（旧行为）
        kept6 = _PPED._apply_class_thresholds([g_fake, g_real], 0.35, None, None)
        report("两阈值均未配置时短路返回原列表",
               len(kept6) == 2,
               "None/None → 原样返回（推理阶段的 0.35 已承担过滤）")

        # ============ BUG-19：成对先验跨人抢手套（验收报告 S2/S3/S11/S12） ============
        # 两人相距很远、各戴一只手套：BUG-14 的成对先验没有空间约束，
        # sibling 作为最高优先级，先绑到手套的人会把帧内任意位置的其他
        # 手套抢走（名额被占、conf 更低不入库）→ 失主被误判"未戴手套"。
        # 修复：sibling 要求手套落在本人 40% 扩张框内（空间接触才构成
        # "另一只手"证据；BUG-14 的伸手场景本就在扩张框内，不受影响）。
        print("--- BUG-19 成对先验空间约束（跨人抢手套） ---")
        from ai.ppe_matcher import match_person_ppe as _match5

        p_far_l = _mk2("person", 0.90, 100, 100, 500, 900)
        p_far_r = _mk2("person", 0.88, 1500, 100, 1900, 900)
        g_far_l = _mk2("gloves", 0.80, 200, 600, 320, 700)     # 左人手套（先处理）
        g_far_r = _mk2("gloves", 0.75, 1600, 600, 1720, 700)   # 右人手套

        sts8 = _match5([p_far_l, p_far_r, g_far_l, g_far_r])
        l8 = [s for s in sts8 if abs(s.bbox.area - p_far_l.area) < 1e-6][0]
        r8 = [s for s in sts8 if abs(s.bbox.area - p_far_r.area) < 1e-6][0]
        report("相距很远的两人各戴一只手套互不抢（验收场景）",
               "gloves" in l8.ppe and "gloves" in r8.ppe,
               f"左人={sorted(l8.ppe)}，右人={sorted(r8.ppe)}"
               "（修复前右人的手套被左人抢走 → 右人误判未戴手套）")

        # 对照：手套与本人框相交但中心在框外（伸手场景）→ 成对先验仍生效。
        # 注意（BUG-22）：sibling 还要求手套与本人框 overlap>0——完全悬在
        # 扩张环带里、与本人框零交集的手套不构成"另一只手"证据，故此
        # 对照用例的手套必须跨压本人框缘（x1=480 < 框缘 500）。
        p_near_l = _mk2("person", 0.90, 100, 100, 500, 900)
        p_near_r = _mk2("person", 0.88, 700, 100, 1000, 900)
        g_own = _mk2("gloves", 0.80, 200, 600, 320, 700)       # 左人手套
        g_reach = _mk2("gloves", 0.75, 480, 500, 600, 600)     # 左人跨缘伸出的手
        sts9 = _match5([p_near_l, p_near_r, g_own, g_reach])
        l9 = [s for s in sts9 if abs(s.bbox.area - p_near_l.area) < 1e-6][0]
        r9 = [s for s in sts9 if abs(s.bbox.area - p_near_r.area) < 1e-6][0]
        report("跨压框缘的伸手场景成对先验仍生效（BUG-14 不回归）",
               "gloves" in l9.ppe and "gloves" not in r9.ppe,
               f"左人={sorted(l9.ppe)}，右人={sorted(r9.ppe)}"
               "（手套中心 x=540 在左人 20% 扩张边界 580 内且与本人框相交 → 归本人）")

        # ============ BUG-20：联合框判定误杀平级重叠真人 ============
        # img_97445215 三人俯身图：中女(0.516)框与右女(55%)、后男(16%)均有
        # 重叠，合计覆盖 71% 过了"50% 且 ≥2 框"阈值被当联合框杀掉 → 3 人只
        # 报 2 人。真联合框要装下两个人，面积至少是最大被套框的 ~2 倍
        # （img_9491b9ea 联合框为单人 4 倍）；此处右女框 = 中女框的 93%，
        # 是平级重叠不是"套住" → 加 p.area ≥ 1.8×max(o.area) 面积约束。
        print("--- BUG-20 联合框判定的面积约束（俯身三人簇） ---")
        from ai.ppe_matcher import match_person_ppe as _match6
        from ai.detector import Detection as _D2

        def _mk3(cls, conf, box):
            return _D2(cls_name=cls, raw_cls=cls, conf=conf,
                       x1=box[0], y1=box[1], x2=box[2], y2=box[3])

        p_man   = _mk3("person", 0.737, (550, 79, 801, 595))    # 后男
        p_mid   = _mk3("person", 0.516, (652, 120, 1157, 988))  # 中女（被误杀者）
        p_right = _mk3("person", 0.825, (741, 411, 1378, 1051)) # 右女
        g1 = _mk3("gloves", 0.725, (650, 771, 748, 954))
        g2 = _mk3("gloves", 0.798, (787, 861, 914, 936))
        lc = _mk3("lab_coat", 0.852, (922, 621, 1377, 1064))

        stsA = _match6([p_man, p_mid, p_right, g1, g2, lc])
        report("俯身三人簇 3 人都保留（联合框面积约束）",
               len(stsA) == 3,
               f"person 清理后剩 {len(stsA)} 人"
               "（修复前中女被当联合框杀掉，3 人只报 2 人）")

        # 对照：真联合框（面积远大于被套者）仍应被杀
        u_big = _mk3("person", 0.60, (100, 100, 1400, 900))     # 大联合框
        r_a = _mk3("person", 0.85, (150, 150, 650, 850))        # 真人A（大部分在联合框内）
        r_b = _mk3("person", 0.83, (750, 200, 1300, 880))       # 真人B
        stsB = _match6([u_big, r_a, r_b])
        areas = sorted(s.bbox.area for s in stsB)
        report("真联合框仍被面积约束杀掉（img_9491b9ea 场景不回归）",
               len(stsB) == 2,
               f"保留 {len(stsB)} 人（联合框 0.60 被杀，两位真人 0.85/0.83 保留）")

        # ============ BUG-21：验收报告 S3/S11/S12 仍未修净 ============
        # BUG-19 的 40% 扩张约束不够小：两人框部分重叠时，40% 扩张会覆盖到
        # 对方的手套（S3 部分重叠 / S12 三人排队，已有一只手套的人照样抢）；
        # 且候选集仍是全员，手套满额守卫让本人退出后，远处零交集的旁人会
        # 顶上抢走 S11 的第 3 只手套。修复（PR5 验收报告建议）：
        #   ① 空间命中硬过滤：与 PPE 无任何空间关系（框内 + 40% 扩张都够不到）
        #      的人直接不得参选，候选为空走距离兜底；
        #   ② sibling 收紧到 20% 小扩张（img_8e909ac5 实测伸手仅超框 4%）；
        #   ③ 兜底距离改「PPE 中心到框边」（img_aa379dec：手伸出 56% 框宽，
        #      到框心距离被放大导致没人认领）。
        print("--- BUG-21 验收 S3/S11/S12（sibling 小扩张 + 空间硬过滤） ---")
        from ai.ppe_matcher import match_person_ppe as _match7

        # S3：两人部分重叠（重叠 150px），各戴手套。
        # g2 中心 670 特意放进 P1 的 20% 扩张环带（边界 700）内但与 P1 框
        # 零交集（框 x1=640 > P1 缘 600）——锁死 overlap>0 约束（BUG-22）。
        s3_p1 = _mk2("person", 0.90, 100, 100, 600, 900)
        s3_p2 = _mk2("person", 0.88, 450, 100, 950, 900)
        s3_g1 = _mk2("gloves", 0.80, 200, 600, 320, 700)   # P1 框内
        s3_g2 = _mk2("gloves", 0.78, 640, 600, 760, 700)   # P2 框内、P1 框外（环带内）
        s3 = _match7([s3_p1, s3_p2, s3_g1, s3_g2])
        s3_a = [s for s in s3 if abs(s.bbox.area - s3_p1.area) < 1e-6][0]
        s3_b = [s for s in s3 if abs(s.bbox.area - s3_p2.area) < 1e-6][0]
        report("S3 部分重叠两人各戴手套互不抢",
               "gloves" in s3_a.ppe and "gloves" in s3_b.ppe,
               f"P1={sorted(s3_a.ppe)}，P2={sorted(s3_b.ppe)}"
               "（g2 中心 670 在 P1 20% 环带内但与 P1 框零交集 → 不得抢）")

        # S11：3 只手套全在 A 框内，A 名额满 → 远处零交集的 B 不得抢第 3 只
        s11_a = _mk2("person", 0.90, 100, 100, 600, 900)
        s11_b = _mk2("person", 0.88, 1500, 100, 1900, 900)
        s11_g = [_mk2("gloves", 0.86, 200, 600, 320, 700),
                 _mk2("gloves", 0.84, 340, 620, 460, 720),
                 _mk2("gloves", 0.60, 200, 300, 320, 400)]
        s11 = _match7([s11_a, s11_b] + s11_g)
        s11_a2 = [s for s in s11 if abs(s.bbox.area - s11_a.area) < 1e-6][0]
        s11_b2 = [s for s in s11 if abs(s.bbox.area - s11_b.area) < 1e-6][0]
        report("S11 满额者的第 3 只手套不被远处零交集者抢走",
               "gloves" not in s11_b2.ppe,
               f"A={sorted(s11_a2.ppe)}，B={sorted(s11_b2.ppe)}"
               "（修复前 B 顶替满额退出的 A 抢走第 3 只）")

        # S12：三人排队各有手套。g3 中心 1120 放进 P2 的 20% 环带
        # （1050~1150）内但与 P2 框零交集（框 x1=1060 > P2 缘 1050）。
        s12_ps = [_mk2("person", 0.90, 100, 100, 600, 900),
                  _mk2("person", 0.88, 550, 100, 1050, 900),
                  _mk2("person", 0.86, 1000, 100, 1500, 900)]
        s12_gs = [_mk2("gloves", 0.82, 200, 600, 320, 700),
                  _mk2("gloves", 0.80, 650, 600, 770, 700),
                  _mk2("gloves", 0.78, 1060, 600, 1180, 700)]
        s12 = _match7(s12_ps + s12_gs)
        s12_own = sorted(tuple(sorted(s.ppe)) for s in s12)
        report("S12 三人各自有手套互不抢",
               s12_own == [("gloves",), ("gloves",), ("gloves",)],
               f"三人 PPE={s12_own}（修复前中间人的手套被左邻 sibling 抢走）")

        # 对照：手大幅伸出框外（img_aa379dec，伸出 56% 框宽）→ 距离兜底仍认领
        r_p = _mk2("person", 0.71, 53, 17, 259, 325)
        r_g = _mk2("gloves", 0.67, 340, 110, 408, 203)      # 中心 (374,157)
        r_sts = _match7([r_p, r_g])
        r_own = [s.ppe for s in r_sts if abs(s.bbox.area - r_p.area) < 1e-6][0]
        report("伸出手大幅出框时距离兜底仍归最近本人",
               "gloves" in r_own,
               f"本人={sorted(r_own)}（修复前硬过滤后无人认领 → 误判未戴手套）")

        # ============ BUG-23：sibling 追加 overlap>=0.2（验收再复审） ============
        # BUG-22 的 overlap>0 挡不住小交叠：手套绝大部分面积在本人框内时，
        # 部分重叠邻人与该手套仅有 1~20px 交叠（占比 ≈0.01~0.05），
        # 邻人的 sibling 仍会成立并抢走。收紧为相交占比 ≥ 0.2。
        # 守卫：手套与本人框 0 < overlap < 0.2（x1=580，与 P1 缘 600 交叠 20px，
        # 占比 0.167）→ 本人 sibling 不得生效，手套交还规则链（邻人 exact 命中）。
        b23_p1 = _mk2("person", 0.90, 100, 100, 600, 900)
        b23_p2 = _mk2("person", 0.88, 450, 100, 950, 900)
        b23_g = _mk2("gloves", 0.80, 580, 600, 700, 700)   # P2 exact 命中；与 P1 交叠 20px
        b23 = _match7([b23_p1, b23_p2, b23_g])
        # 注意：P1/P2 面积相同（500×800），不能按面积匹配（会取到同一 state），
        # 必须按 bbox 坐标匹配
        b23_a = [s for s in b23 if abs(s.bbox.x1 - 100) < 1e-6][0]
        b23_b = [s for s in b23 if abs(s.bbox.x1 - 450) < 1e-6][0]
        report("小交叠（0<overlap<0.2）不构成成对先验证据（BUG-23 守卫）",
               "gloves" not in b23_a.ppe and "gloves" in b23_b.ppe,
               f"P1={sorted(b23_a.ppe)}，P2={sorted(b23_b.ppe)}"
               "（g 与 P1 交叠仅 20px 占比 0.167 → P1 sibling 不得抢；P2 exact 命中）")

        # 正向对照：手套大部分面积与本人框相交（占比 >0.2）→ sibling 照常生效。
        # 复刻真实伸手形态（img_8e909ac5 / img_9491b9ea g_edge 占比 0.439）：
        # 手套 120px 宽，60px 在本人框内（占比 0.5），且本人已绑另一只手套。
        b23b_p1 = _mk2("person", 0.90, 100, 100, 600, 900)
        b23b_p2 = _mk2("person", 0.88, 560, 100, 950, 900)   # 与 P1 重叠 40px
        b23b_own = _mk2("gloves", 0.80, 200, 600, 320, 700)  # P1 已绑的第一只
        b23b_g = _mk2("gloves", 0.78, 540, 600, 660, 700)    # 60/120 在 P1 内（0.5）
        b23b = _match7([b23b_p1, b23b_p2, b23b_own, b23b_g])
        b23b_a = [s for s in b23b if abs(s.bbox.x1 - 100) < 1e-6][0]
        b23b_b = [s for s in b23b if abs(s.bbox.x1 - 560) < 1e-6][0]
        report("大幅相交（overlap>=0.2）的成对先验不回归（BUG-23 正向对照）",
               "gloves" in b23b_a.ppe and "gloves" not in b23b_b.ppe,
               f"P1={sorted(b23b_a.ppe)}，P2={sorted(b23b_b.ppe)}"
               "（g 60/120 在 P1 内占比 0.5 → P1 sibling 胜出；P2 不得抢）")

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
