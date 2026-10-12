# -*- coding: utf-8 -*-
"""模型换装 + 验收（一键脚本，自动适配 5 类 / 6 类 / 4 类）

用途：队友跑完训练回传 labppe_weights_*.zip 后，执行本脚本完成全部收尾：
  1) 从 zip 里取出 best.pt，自动识别类别数（不写死，兼容 4/5/6 类）
  2) 校验类别名是否与项目体系一致（person/mask/gloves/lab_coat/goggles[/coverall]）
  3) 备份当前模型 → ai/model/backup/labppe_v8s_<旧nc>cls_<date>.pt
  4) 换装：新 best.pt → ai/model/labppe_v8s.pt
  5) 提示后续评估命令

用法：
  $PY tools/swap_5cls_model.py --zip C:/Users/xxx/Desktop/labppe_weights_6cls.zip
  $PY tools/swap_5cls_model.py --zip ... --dry-run        # 只看不换
  $PY tools/swap_5cls_model.py --zip ... --expect-nc 6    # 强制要求 6 类

历史：
  2026-10-11 初版，仅支持 5 类（EXPECT_NC=5）
  2026-10-12 改为自动识别，支持 coverall 独立成类后的 6 类模型
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

# 项目类别体系的「标准前缀」：模型类别名必须是它的前缀（顺序也要一致）
CANON_ORDER = ["person", "mask", "gloves", "lab_coat", "goggles", "coverall"]


def step(msg):
    print("\n" + "=" * 72 + "\n>>> " + msg + "\n" + "=" * 72, flush=True)


def check_model(path: Path) -> dict:
    """加载模型，返回 {nc, names}"""
    import torch
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    m = ckpt.get("model")
    names = getattr(m, "names", None) or ckpt.get("names")
    names = {int(k): str(v) for k, v in (names or {}).items()}
    return {"nc": len(names), "names": names, "ckpt": ckpt}


def validate_names(names: dict) -> tuple[bool, str]:
    """校验类别名与项目体系是否一致（允许前缀匹配，6 类是 5 类的超集）"""
    got = [names[k] for k in sorted(names)]
    expect = CANON_ORDER[:len(got)]
    if got == expect:
        return True, "与项目体系完全一致"
    # 常见别名归一后比较
    alias = {"labcoat": "lab_coat", "lab-coat": "lab_coat", "glove": "gloves",
             "goggle": "goggles", "face-mask": "mask", "coveralls": "coverall"}
    norm = [alias.get(g.lower().replace(" ", "_"), g.lower().replace(" ", "_")) for g in got]
    if norm == expect:
        return True, f"归一化后一致（原始名: {got}）"
    return False, f"不一致：得到 {got}，期望 {expect}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True,
                    help="队友回传的权重 zip，或直接给单个 best.pt / *.pt 文件")
    ap.add_argument("--dry-run", action="store_true", help="只校验不换装（不备份、不写盘）")
    ap.add_argument("--expect-nc", type=int, default=None,
                    help="强制要求类别数（默认自动识别，兼容 4/5/6 类）")
    args = ap.parse_args()

    zp = Path(args.zip)
    if not zp.exists():
        print(f"[ERROR] 找不到文件: {zp}")
        return 1

    # ---------- 1) 取出 best.pt（兼容 zip 包 / 单个 .pt 文件） ----------
    lasts = []
    if zp.is_file() and zp.suffix.lower() == ".pt":
        step("1/5 使用指定的权重文件（非 zip）")
        print("权重文件:", zp, "%.1f MB" % (zp.stat().st_size / 1048576))
        new_best = zp
    else:
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

    # ---------- 2) 校验类别（自动识别） ----------
    step("2/5 校验新模型类别（自动识别）")
    info = check_model(new_best)
    print("类别数:", info["nc"])
    print("类别:", info["names"])

    if args.expect_nc is not None and info["nc"] != args.expect_nc:
        print(f"[ERROR] 类别数为 {info['nc']}，你要求的是 {args.expect_nc}。已中止，未改动任何文件。")
        return 1
    if info["nc"] not in (5, 6):
        print(f"[ERROR] 类别数为 {info['nc']}，本项目只支持 5 类（无 coverall）或 6 类（含 coverall）。已中止。")
        return 1
    ok, why = validate_names(info["names"])
    if not ok:
        print(f"[ERROR] 类别名校验失败：{why}")
        print("        请确认这是本项目的模型包。已中止，未改动任何文件。")
        return 1
    print("类别名校验通过：", why)

    # 6 类模型额外提示
    if info["nc"] == 6:
        print("\n[提示] 这是 6 类模型（含 coverall 连体防护服）。")
        print("       换装前请确认 data/lab_ppe/data.yaml 已是 nc=6，")
        print("       且 ai/detector.py 的 CLASS_ALIAS 已把 coverall 映射到 'coverall'。")
    else:
        print("\n[提示] 这是 5 类模型（无 coverall）。")

    # ---------- 3) 备份旧模型 ----------
    if args.dry_run:
        print("\n[DRY-RUN] 不备份、不换装。若确认无误，去掉 --dry-run 重跑即可。")
        return 0

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

    # ---------- 4) 换装 ----------
    step("4/5 换装新模型")
    shutil.copy2(new_best, LIVE_MODEL)
    print("已换装 →", LIVE_MODEL, "%.1f MB" % (LIVE_MODEL.stat().st_size / 1048576))
    if lasts:
        dst_last = ROOT / "ai" / "model" / f"labppe_v8s_{info['nc']}cls_last.pt"
        shutil.copy2(lasts[0], dst_last)
        print("顺带留存 last.pt →", dst_last)
    verify = check_model(LIVE_MODEL)
    assert verify["nc"] == info["nc"], "换装后类别数不对！"
    print("换装校验通过:", verify["nc"], verify["names"])

    # ---------- 5) 提示后续 ----------
    step("5/5 下一步：跑评估")
    print("请依次执行：")
    print("  .venv/Scripts/python.exe tools/run_e3_retest.py")
    print("  .venv/Scripts/python.exe tools/compare_e4_vs_e2.py")
    print()
    print("然后跑回归确认无破坏：")
    print("  NO_PROXY='*' .venv/Scripts/python.exe tests/run_tests.py          # 期望 14/14")
    print("  NO_PROXY='*' .venv/Scripts/python.exe tests/test_edge_cases.py    # 期望 17/17")
    print("  NO_PROXY='*' .venv/Scripts/python.exe tests/test_robustness.py    # 期望 19/19")
    if info["nc"] == 6:
        print()
        print("[6 类专项验证建议] 用含连体服的图测：")
        print("  data/lab_ppe/images/train/fsaz_jpg.rf.24313af2908e963674e5624b20f166e1.jpg")
        print("  预期：检出 coverall，且那些人被判「未穿实验服」")
    return 0


if __name__ == "__main__":
    sys.exit(main())
