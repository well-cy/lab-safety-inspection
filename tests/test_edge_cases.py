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
