# -*- coding: utf-8 -*-
"""E3 复测运行器（零侵入包装）。

背景：E3 原始脚本（tools/eval_e3_dataset2.py / eval_e3_t01t10.py）硬编码模型路径
`runs/e2/labppe_v8s/weights/best.pt`，该路径已不存在（runs/ 被清理）。现网模型为
`ai/model/labppe_v8s.pt`。

本运行器不修改任何原始脚本，只：
  1) import 两个 E3 模块
  2) 把模块级常量 LABPPE 重绑到实际模型路径
  3) 依次调用两者的 main()

用法:
  $PY tools/run_e3_retest.py              # 使用 ai/model/labppe_v8s.pt
  $PY tools/run_e3_retest.py --model path/to/best.pt
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import cv2
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

DEFAULT_MODEL = ROOT / "ai" / "model" / "labppe_v8s.pt"


def _patch_detector_for_paths() -> None:
    """生产 detector.detect() 已改为接收 BGR ndarray；E3 老脚本传的是路径字符串。
    此处给 PPEDetector 打一个兼容补丁：detect(str_path) 自动读图后转发。
    （零侵入：不改 ai/detector.py，只在运行时包装。）"""
    from ai.detector import PPEDetector

    _orig = PPEDetector.detect

    def detect_compat(self, frame, conf=None, tile_recheck=False):
        if isinstance(frame, (str, Path)):
            arr = cv2.imdecode(np.fromfile(str(frame), dtype=np.uint8), cv2.IMREAD_COLOR)
            if arr is None:
                return []
            return _orig(self, arr, conf, tile_recheck)
        return _orig(self, frame, conf, tile_recheck)

    PPEDetector.detect = detect_compat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL),
                    help="LAB_PPE 权重路径（默认 ai/model/labppe_v8s.pt）")
    ap.add_argument("--skip-dataset2", action="store_true")
    ap.add_argument("--skip-t01t10", action="store_true")
    args = ap.parse_args()

    model = Path(args.model)
    if not model.exists():
        print(f"[ERROR] 模型不存在: {model}")
        return 1
    print(f"[E3] LAB_PPE 模型 → {model}")

    _patch_detector_for_paths()

    if not args.skip_dataset2:
        import eval_e3_dataset2 as m1
        m1.LABPPE = model
        print("\n" + "#" * 78)
        print("# E3-3 dataset2 external_test 双模型评估")
        print("#" * 78)
        m1.main()

    if not args.skip_t01t10:
        import eval_e3_t01t10 as m2
        m2.LABPPE = model
        print("\n" + "#" * 78)
        print("# E3-4 t01-t10 A/B 对比")
        print("#" * 78)
        m2.main()

    return 0


if __name__ == "__main__":
    sys.exit(main())
