# -*- coding: utf-8 -*-
"""
集中配置模块测试（新增独立文件）。

运行：venv/bin/python tests/test_config.py

背景（分工任务「集中管理配置」+ 技术债 #1）：
  收口前，以下值散落硬编码，换模型/换目录必须改源码：
    - 模型权重路径       backend/main.py
    - 上传/输出目录      backend/main.py
    - 数据库路径         backend/database.py
    - 置信度阈值 0.35    ai/detector.py 构造默认值
    - 视频 stride 2      backend/main.py 表单默认值
    - 违规截图冷却 10s   video/processor.py 模块常量
    - 缺 PPE 分级阈值 1  rule_engine/engine.py 构造默认值

本用例验证两件事：
  1. **默认值与原硬编码完全一致**（即收口未改变任何行为）
  2. 每一项都可通过 LABSAFETY_* 环境变量覆盖（换模型无需改代码）

注意：本文件为新增，不修改 tests/run_tests.py 的 T01~T14。
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.config import Config, config  # noqa: E402

RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append({"name": name, "ok": ok, "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        print(f"       {detail}")


def build_config(**env) -> Config:
    """临时设置 LABSAFETY_* 环境变量并构造一个新 Config，随后恢复环境"""
    saved = {}
    try:
        for name, value in env.items():
            key = "LABSAFETY_" + name
            saved[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        return Config()
    finally:
        for key, old in saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def expect_value_error(name, **env):
    try:
        build_config(**env)
    except ValueError as exc:
        report(name, "LABSAFETY_" in str(exc), f"异常信息={exc}")
    except Exception as exc:  # pragma: no cover
        report(name, False, f"抛出了非 ValueError：{type(exc).__name__}: {exc}")
    else:
        report(name, False, "未抛出任何异常（非法值被静默接受）")


def main():
    print("=" * 70)
    print("集中配置模块测试")
    print("=" * 70)

    # ---------- 1. 默认值必须与原硬编码一致 ----------
    print("--- 1. 默认值与原硬编码一致性（收口不得改变行为）---")
    expectations = [
        ("MODEL_PATH", "ai/model/sh17_yolov8s.pt"),
        ("DB_PATH", "backend/labsafety.db"),
        ("UPLOAD_DIR", "data/test/uploads"),
        ("ANNOTATED_DIR", "outputs/annotated"),
        ("VIDEO_DIR", "outputs/videos"),
        ("SCREENSHOT_DIR", "outputs/screenshots"),
    ]
    for attr, suffix in expectations:
        actual = str(getattr(config, attr))
        ok = actual.replace("\\", "/").endswith(suffix)
        report(f"默认 {attr} 指向 {suffix}", ok, f"实际={actual}")

    scalars = [
        ("CONF_THRESHOLD", 0.35),
        ("DEVICE", "auto"),
        ("VIDEO_STRIDE", 2),
        ("VIOLATION_COOLDOWN_S", 10.0),
        ("MINOR_THRESHOLD", 1),
        ("DEFAULT_LAB_ID", 1),
        ("DEFAULT_REQUIRED_PPE", "mask,gloves,lab_coat"),
    ]
    for attr, expect in scalars:
        actual = getattr(config, attr)
        report(f"默认 {attr} == {expect!r}", actual == expect, f"实际={actual!r}")

    report("默认 ALLOWED_PPE 为 5 类 PPE",
           config.ALLOWED_PPE == ["mask", "gloves", "lab_coat", "goggles", "helmet"],
           f"实际={config.ALLOWED_PPE}")

    # ---------- 2. 环境变量覆盖 ----------
    print("--- 2. 环境变量覆盖（换模型无需改代码）---")
    fake = "/tmp/custom_model.pt"
    c = build_config(MODEL_PATH=fake, CONF_THRESHOLD="0.55", DEVICE="cpu",
                     VIDEO_STRIDE="5", VIOLATION_COOLDOWN_S="3.5",
                     MINOR_THRESHOLD="2", DEFAULT_LAB_ID="7",
                     DEFAULT_REQUIRED_PPE="mask,goggles",
                     DB_PATH="/tmp/custom.db", UPLOAD_DIR="/tmp/up")

    report("MODEL_PATH 可覆盖（技术债 #1 的目标场景）",
           str(c.MODEL_PATH) == fake, f"实际={c.MODEL_PATH}")
    report("CONF_THRESHOLD 可覆盖（float）",
           c.CONF_THRESHOLD == 0.55, f"实际={c.CONF_THRESHOLD}")
    report("DEVICE 可覆盖（str）", c.DEVICE == "cpu", f"实际={c.DEVICE}")
    report("VIDEO_STRIDE 可覆盖（int）", c.VIDEO_STRIDE == 5, f"实际={c.VIDEO_STRIDE}")
    report("VIOLATION_COOLDOWN_S 可覆盖（float）",
           c.VIOLATION_COOLDOWN_S == 3.5, f"实际={c.VIOLATION_COOLDOWN_S}")
    report("MINOR_THRESHOLD 可覆盖（int）",
           c.MINOR_THRESHOLD == 2, f"实际={c.MINOR_THRESHOLD}")
    report("DEFAULT_LAB_ID 可覆盖（int）",
           c.DEFAULT_LAB_ID == 7, f"实际={c.DEFAULT_LAB_ID}")
    report("DEFAULT_REQUIRED_PPE 可覆盖（str）",
           c.DEFAULT_REQUIRED_PPE == "mask,goggles", f"实际={c.DEFAULT_REQUIRED_PPE}")
    report("DB_PATH 可覆盖（未设置时不影响默认值）",
           str(c.DB_PATH) == "/tmp/custom.db", f"实际={c.DB_PATH}")
    report("UPLOAD_DIR 可覆盖", str(c.UPLOAD_DIR) == "/tmp/up", f"实际={c.UPLOAD_DIR}")
    report("ALLOWED_PPE 不受环境变量影响（业务约束不走配置）",
           c.ALLOWED_PPE == config.ALLOWED_PPE, f"实际={c.ALLOWED_PPE}")

    # ---------- 3. 边界与非法输入 ----------
    print("--- 3. 边界与非法输入 ---")
    c = build_config(DEVICE="", MODEL_PATH="")
    report("空字符串视为未设置（回退默认值）",
           c.DEVICE == "auto" and str(c.MODEL_PATH).endswith("sh17_yolov8s.pt"),
           f"DEVICE={c.DEVICE} MODEL_PATH={c.MODEL_PATH}")

    c = build_config(DEVICE="  cpu  ", CONF_THRESHOLD=" 0.5 ")
    report("首尾空白被裁剪",
           c.DEVICE == "cpu" and c.CONF_THRESHOLD == 0.5,
           f"DEVICE={c.DEVICE!r} CONF_THRESHOLD={c.CONF_THRESHOLD}")

    c = build_config(MODEL_PATH="~/models/custom.pt")
    ok = str(c.MODEL_PATH).startswith(str(Path.home()))
    report("MODEL_PATH 支持 ~ 展开", ok, f"实际={c.MODEL_PATH}")

    expect_value_error("非法整数（VIDEO_STRIDE=abc）抛 ValueError 且含变量名",
                       VIDEO_STRIDE="abc")
    expect_value_error("非法浮点（CONF_THRESHOLD=高）抛 ValueError",
                       CONF_THRESHOLD="高")
    expect_value_error("非法整数（MINOR_THRESHOLD=1.5）抛 ValueError",
                       MINOR_THRESHOLD="1.5")

    # ---------- 4. 目录创建 ----------
    print("--- 4. ensure_dirs ---")
    with tempfile.TemporaryDirectory() as tmp:
        c = build_config(UPLOAD_DIR=f"{tmp}/up",
                         ANNOTATED_DIR=f"{tmp}/ann",
                         VIDEO_DIR=f"{tmp}/vid",
                         SCREENSHOT_DIR=f"{tmp}/shot")
        c.ensure_dirs()
        made = [Path(f"{tmp}/{n}").is_dir()
                for n in ("up", "ann", "vid", "shot")]
        report("ensure_dirs 创建全部运行期目录", all(made), f"逐个结果={made}")
        c.ensure_dirs()
        report("ensure_dirs 幂等（重复调用不报错）", True, "")

    # ---------- 5. describe() ----------
    print("--- 5. describe() ---")
    d = config.describe()
    need = {"root", "model_path", "db_path", "upload_dir", "annotated_dir",
            "video_dir", "screenshot_dir", "conf_threshold", "device",
            "video_stride", "violation_cooldown_s", "minor_threshold",
            "default_lab_id", "default_required_ppe", "allowed_ppe"}
    report("describe() 字段完整", need <= set(d),
           f"缺少={sorted(need - set(d)) or '无'}")
    report("describe() 全部为可序列化基础类型",
           all(isinstance(v, (str, int, float, list)) for v in d.values()),
           f"类型={ {k: type(v).__name__ for k, v in d.items() if not isinstance(v, (str, int, float, list))} or '全部合规'}")

    # ---------- 6. 与 main.py 的接线验证（集成） ----------
    print("--- 6. 集成：main.py 确实在用 config ---")
    import backend.main as main_mod

    report("main.ROOT 来自 config.ROOT",
           main_mod.ROOT == config.ROOT, f"main={main_mod.ROOT}")
    report("main.MODEL_PATH 来自 config.MODEL_PATH",
           main_mod.MODEL_PATH == config.MODEL_PATH, f"main={main_mod.MODEL_PATH}")
    report("main.ALLOWED_PPE 来自 config.ALLOWED_PPE",
           main_mod.ALLOWED_PPE == config.ALLOWED_PPE, "")
    report("main.OUT_IMG_DIR / OUT_VID_DIR 来自 config",
           main_mod.OUT_IMG_DIR == config.ANNOTATED_DIR
           and main_mod.OUT_VID_DIR == config.VIDEO_DIR, "")

    import backend.database as db_mod
    report("database.DB_PATH 来自 config.DB_PATH",
           db_mod.DB_PATH == config.DB_PATH, f"db={db_mod.DB_PATH}")

    import backend.export as export_mod
    report("export 模块可正常导入（未破坏）", export_mod.router is not None, "")

    # 管线参数接线（首次会加载模型，约数秒）
    try:
        pipe = main_mod.get_pipeline()
        report("管线 detector.conf_threshold 来自 config",
               pipe.detector.conf_threshold == config.CONF_THRESHOLD,
               f"实际={pipe.detector.conf_threshold}")
        report("管线 engine.minor_threshold 来自 config",
               pipe.engine.minor_threshold == config.MINOR_THRESHOLD,
               f"实际={pipe.engine.minor_threshold}")
        report("管线 cooldown_s 来自 config",
               pipe.cooldown_s == config.VIOLATION_COOLDOWN_S,
               f"实际={pipe.cooldown_s}")
        report("管线 screenshot_dir 来自 config",
               Path(pipe.screenshot_dir) == config.SCREENSHOT_DIR,
               f"实际={pipe.screenshot_dir}")
    except Exception as exc:  # pragma: no cover
        report("管线参数接线验证", False, f"异常: {type(exc).__name__}: {exc}")

    # ---------- 7. video/processor 的默认值未被破坏 ----------
    print("--- 7. 向后兼容：processor 默认值不变 ---")
    from video.processor import InspectionPipeline, VIOLATION_COOLDOWN_S
    from unittest.mock import MagicMock

    p = InspectionPipeline(detector=MagicMock())
    report("InspectionPipeline 不传 cooldown_s 时沿用模块常量",
           p.cooldown_s == VIOLATION_COOLDOWN_S == 10.0,
           f"实际={p.cooldown_s}（模块常量={VIOLATION_COOLDOWN_S}）")
    p2 = InspectionPipeline(detector=MagicMock(), cooldown_s=2.5)
    report("InspectionPipeline 可显式指定 cooldown_s",
           p2.cooldown_s == 2.5, f"实际={p2.cooldown_s}")

    passed = sum(1 for x in RESULTS if x["ok"])
    total = len(RESULTS)
    print("=" * 70)
    print(f"结果: {passed}/{total} 通过")
    if passed != total:
        print("失败用例:")
        for x in RESULTS:
            if not x["ok"]:
                print(f"  - {x['name']}: {x['detail']}")
    print("=" * 70)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
