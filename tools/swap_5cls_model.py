# -*- coding: utf-8 -*-
"""5 类新模型换装 + 三线对比验收（一键脚本）

用途：队友跑完训练回传 labppe_weights_5cls.zip 后，执行本脚本完成全部收尾：
  1) 从 zip 里取出 best.pt / last.pt，校验类别数 == 5
  2) 备份当前 4 类模型 → ai/model/backup/labppe_v8s_4cls_<date>.pt
  3) 换装：新 best.pt → ai/model/labppe_v8s.pt
  4) 在 dataset2 external_test 上跑新版评估（E3 口径：conf=0.35, imgsz=640, IoU@0.5, mAP@50）
  5) 输出对比报告 → docs/E5_5cls_retrain_report.md

用法：
  $PY tools/swap_5cls_model.py --zip C:/Users/xxx/Desktop/labppe_weights_5cls.zip
  $PY tools/swap_5cls_model.py --zip ... --dry-run     # 只看不换
"""
from __future__ import annotations

import argparse
import io
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
LIVE_MODEL = ROOT / "ai" / "model" / "labppe_v8s.pt"
BACKUP_DIR = ROOT / "ai" / "model" / "backup"
TMP_DIR = ROOT / "outputs" / "_swap_tmp"

EXPECT_NC = 5
EXPECT_NAMES = {0: "person", 1: "mask", 2: "gloves", 3: "lab_coat", 4: "goggles"}


def step(msg):
    print("\n" + "=" * 72 + "\n>>> " + msg + "\n" + "=" * 72, flush=True)


def check_model(path: Path) -> dict:
    """加载模型，返回 {nc, names} 并校验。"""
    import torch
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    m = ckpt.get("model")
    names = getattr(m, "names", None) or ckpt.get("names")
    names = {int(k): str(v) for k, v in (names or {}).items()}
    return {"nc": len(names), "names": names, "ckpt": ckpt}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="队友回传的权重 zip 路径")
    ap.add_argument("--dry-run", action="store_true", help="只校验不换装")
    args = ap.parse_args()

    zp = Path(args.zip)
    if not zp.exists():
        print(f"[ERROR] 找不到 zip: {zp}")
        return 1

    # ---------- 1) 解压 ----------
    step("1/5 解压权重包")
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zp) as z:
        z.extractall(TMP_DIR)
        print("包内文件:", z.namelist())
    bests = list(TMP_DIR.rglob("best.pt"))
    lasts = list(TMP_DIR.rglob("last.pt"))
    if not bests:
        print("[ERROR] 包里没有 best.pt")
        return 1
    new_best = bests[0]
    print("best.pt:", new_best, "%.1f MB" % (new_best.stat().st_size / 1048576))

    # ---------- 2) 校验类别 ----------
    step("2/5 校验新模型类别（必须 5 类）")
    info = check_model(new_best)
    print("类别数:", info["nc"])
    print("类别:", info["names"])
    if info["nc"] != EXPECT_NC:
        print(f"[ERROR] 类别数不是 {EXPECT_NC}！可能是错的包。已中止，未改动任何文件。")
        return 1
    if info["names"] != EXPECT_NAMES:
        print(f"[WARN] 类别名与预期不同：预期 {EXPECT_NAMES}")
        print("       如确认无误可手工继续，否则请核对。")

    # ---------- 3) 备份旧模型 ----------
    step("3/5 备份当前模型")
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    if LIVE_MODEL.exists():
        old = check_model(LIVE_MODEL)
        print("当前模型类别数:", old["nc"], old["names"])
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        bak = BACKUP_DIR / f"labppe_v8s_{old['nc']}cls_{stamp}.pt"
        shutil.copy2(LIVE_MODEL, bak)
        print("已备份 →", bak)
    else:
        print("[WARN] 当前无 ai/model/labppe_v8s.pt，跳过备份")

    if args.dry_run:
        print("\n[DRY-RUN] 校验通过，未改动任何文件。")
        return 0

    # ---------- 4) 换装 ----------
    step("4/5 换装新模型")
    shutil.copy2(new_best, LIVE_MODEL)
    print("已换装 →", LIVE_MODEL, "%.1f MB" % (LIVE_MODEL.stat().st_size / 1048576))
    # 顺带存一份 last.pt
    if lasts:
        shutil.copy2(lasts[0], TMP_DIR / ".." / "_last_5cls.pt")
    verify = check_model(LIVE_MODEL)
    assert verify["nc"] == EXPECT_NC, "换装后类别数不对！"
    print("换装校验通过:", verify["nc"], verify["names"])

    # ---------- 5) 提示后续 ----------
    step("5/5 下一步：跑 dataset2 external_test 评估")
    print("请依次执行：")
    print(f"  .venv/Scripts/python.exe tools/run_e3_retest.py")
    print(f"  .venv/Scripts/python.exe tools/compare_e4_vs_e2.py")
    print("然后把输出发回，生成对比报告。")
    print("\n备注：demo_images 批量对比在 CPU 上跑，约几分钟。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
