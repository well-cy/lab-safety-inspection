"""漏标候选复核（2026-10-08 她判 66 条）落地执行：删框 / 修框 / 补框。

- 只改 data/lab_ppe/{train,valid} 标签 txt；剔除图已单独走 reject_by_list
- 修改前整文件备份到 data/_label_backup/missing_verdict_20261008/（幂等，已有则不覆盖）
- 删/修用归一化坐标匹配（容差 2e-4），匹配不到即报错退出，不静默跳过
- 幂等：set/add 均按当前行状态判断
"""
from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "lab_ppe"
BACKUP = ROOT / "data" / "_label_backup" / "missing_verdict_20261008"
INDEX = ROOT / "outputs" / "missing_labels_20261008" / "index_full.csv"
CLS = {"mask": 0, "gloves": 1, "lab_coat": 2, "goggles": 3}
TOL = 2e-4

# ---- 动作表（编号 → 动作列表）----
# del:  匹配 cls+归一化(cx,cy,w,h) 后删行
# set:  匹配后改坐标；new_xyxy 为原图像素
# add:  新增行；xyxy 原图像素，或 cand=<序号> 取该图第 n 个候选框(0-based)
V = {
    18: [dict(t="del", cls="gloves", n=(0.6695, 0.5024, 0.2668, 0.1442)),
         dict(t="del", cls="gloves", n=(0.7139, 0.4195, 0.1202, 0.0986)),
         dict(t="add", cls="goggles", cand=0)],
    20: [dict(t="del", cls="gloves", n=(0.2500, 0.5541, 0.1058, 0.1755)),
         dict(t="del", cls="gloves", n=(0.4868, 0.6635, 0.2356, 0.1322))],
    40: [dict(t="del", cls="gloves", n=(0.4399, 0.2488, 0.2909, 0.2668))],
    47: [dict(t="del", cls="lab_coat", n=(0.7285, 0.7104, 0.3735, 0.5144)),
         dict(t="del", cls="lab_coat", n=(0.2992, 0.3009, 0.0583, 0.0352)),
         dict(t="del", cls="lab_coat", n=(0.3519, 0.6300, 0.3164, 0.6513))],
    53: [dict(t="del", cls="lab_coat", n=(0.7211, 0.7118, 0.3679, 0.5153)),
         dict(t="del", cls="lab_coat", n=(0.2942, 0.6373, 0.4146, 0.5542))],
    59: [dict(t="del", cls="gloves", n=(0.5132, 0.1406, 0.4712, 0.2812))],
    67: [dict(t="del", cls="gloves", n=(0.2224, 0.2548, 0.1827, 0.1779))],
    73: [dict(t="del", cls="lab_coat", n=(0.6730, 0.7186, 0.3179, 0.5629))],
    77: [dict(t="del", cls="gloves", n=(0.6526, 0.4231, 0.1611, 0.1899)),
         dict(t="del", cls="gloves", n=(0.5457, 0.3738, 0.0962, 0.1010))],
    87: [dict(t="del", cls="lab_coat", n=(0.2571, 0.7465, 0.5058, 0.5071))],
    92: [dict(t="del", cls="gloves", n=(0.4868, 0.5252, 0.0577, 0.0216)),
         dict(t="del", cls="gloves", n=(0.4483, 0.5577, 0.1034, 0.0433))],
    106: [dict(t="del", cls="goggles", n=(0.4567, 0.2752, 0.1250, 0.0697))],
    107: [dict(t="del", cls="goggles", n=(0.5421, 0.2740, 0.1226, 0.0481))],
    140: [dict(t="del", cls="gloves", n=(0.4363, 0.3498, 0.1923, 0.1731))],
    145: [dict(t="del", cls="gloves", n=(0.3702, 0.5240, 0.0481, 0.0769)),
          dict(t="del", cls="gloves", n=(0.4363, 0.5709, 0.1082, 0.0625))],
    17: [dict(t="set", cls="lab_coat", n=(0.4880, 0.7644, 0.0721, 0.1346), xyxy=(140, 238, 236, 431))],
    24: [dict(t="set", cls="lab_coat", n=(0.4651, 0.7332, 0.3029, 0.3101), xyxy=(113, 265, 330, 410)),
         dict(t="set", cls="mask", n=(0.5012, 0.4483, 0.0601, 0.0721), xyxy=(178, 193, 231, 233))],
    29: [dict(t="set", cls="lab_coat", n=(0.6647, 0.6502, 0.6562, 0.6995), xyxy=(0, 148, 395, 416))],
    31: [dict(t="set", cls="lab_coat", n=(0.6094, 0.6274, 0.3365, 0.7188), xyxy=(91, 140, 408, 412)),
         dict(t="set", cls="mask", n=(0.5108, 0.3053, 0.1971, 0.1298), xyxy=(151, 110, 227, 159))],
    32: [dict(t="set", cls="lab_coat", n=(0.2656, 0.5793, 0.3606, 0.6971), xyxy=(23, 106, 303, 383)),
         dict(t="set", cls="mask", n=(0.4243, 0.2740, 0.1851, 0.1731), xyxy=(144, 102, 197, 159))],
    61: [dict(t="set", cls="lab_coat", n=(0.6100, 0.7600, 0.4200, 0.4800), xyxy=(157, 113, 410, 416))],
    91: [dict(t="set", cls="lab_coat", n=(0.6755, 0.6106, 0.2404, 0.4904), xyxy=(147, 185, 315, 410))],
    102: [dict(t="set", cls="lab_coat", n=(0.3882, 0.8305, 0.1226, 0.3341), xyxy=(182, 182, 314, 410))],
    104: [dict(t="set", cls="lab_coat", n=(0.2031, 0.7151, 0.4062, 0.5649), xyxy=(0, 90, 189, 410)),
          dict(t="set", cls="gloves", n=(0.3966, 0.6959, 0.3870, 0.3365), xyxy=(96, 216, 212, 333))],
    56: [dict(t="set", cls="gloves", n=(0.7356, 0.5361, 0.0913, 0.0865), xyxy=(265, 208, 305, 243))],
    57: [dict(t="del", cls="lab_coat", n=(0.2767, 0.7662, 0.3034, 0.4630)),   # 与 old2 重复
         dict(t="del", cls="gloves", n=(0.4316, 0.7546, 0.0690, 0.1481)),     # 左手套拆成两框
         dict(t="set", cls="gloves", n=(0.4551, 0.9294, 0.1029, 0.1412), xyxy=(305, 294, 389, 432))],
    42: [dict(t="add", cls="gloves", cand=0), dict(t="add", cls="gloves", cand=1),
         dict(t="add", cls="goggles", cand=2),
         dict(t="add", cls="goggles", xyxy=(341, 75, 465, 112))],  # 右女生护目镜（她指出的漏标）
    49: [dict(t="add", cls="mask", cand=0),
         dict(t="add", cls="goggles", xyxy=(280, 87, 407, 111)),
         dict(t="add", cls="gloves", xyxy=(148, 76, 201, 140)),
         dict(t="add", cls="gloves", xyxy=(186, 256, 256, 297))],
    76: [dict(t="add", cls="gloves", cand=0),
         dict(t="add", cls="goggles", xyxy=(221, 35, 302, 81))],
}


def imread_size(p: Path):
    img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"读不到图 {p}")
    return img.shape[1], img.shape[0]


def main() -> int:
    rows = {int(r["编号"]): r for r in csv.DictReader(open(INDEX, encoding="utf-8-sig"))}
    apply_it = "--apply" in sys.argv
    n_del = n_set = n_add = 0
    errs = []
    for no, actions in V.items():
        r = rows[no]
        img_path = DATASET / "images" / r["split"] / r["文件名"]
        lbl_path = DATASET / "labels" / r["split"] / (Path(r["文件名"]).stem + ".txt")
        if not lbl_path.exists():
            errs.append(f"#{no} 标签缺失 {lbl_path}")
            continue
        w, h = imread_size(img_path)
        cands = []
        if r["候选框(x1,y1,x2,y2)"]:
            for seg in r["候选框(x1,y1,x2,y2)"].split(";"):
                seg = seg.strip()
                if seg and ":" in seg:
                    name, coords = seg.split(":", 1)
                    try:
                        cands.append((name, tuple(float(v) for v in coords.strip("() ").split(","))))
                    except ValueError:
                        pass
        lines = lbl_path.read_text(encoding="utf-8").splitlines()
        parsed = []
        for ln in lines:
            p = ln.split()
            if len(p) >= 5:
                parsed.append([int(float(p[0]))] + [float(v) for v in p[1:5]])
            else:
                parsed.append(None)

        changed = False
        for a in actions:
            c = CLS[a["cls"]]
            if a["t"] == "add":
                if "cand" in a:
                    name, (x1, y1, x2, y2) = cands[a["cand"]]
                    assert name == a["cls"], f"#{no} 候选类别不符 {name}!={a['cls']}"
                else:
                    x1, y1, x2, y2 = a["xyxy"]
                cx, cy, bw, bh = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h
                dup = any(p and p[0] == c and abs(p[1] - cx) < 5e-3 and abs(p[2] - cy) < 5e-3
                          for p in parsed if p)
                if dup:
                    print(f"#{no} add {a['cls']} 已存在(近重复) 跳过")
                    continue
                parsed.append([c, cx, cy, bw, bh])
                n_add += 1
                changed = True
                print(f"#{no} add {a['cls']} ({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
            elif a["t"] == "del":
                cx, cy, bw, bh = a["n"]
                hit = [i for i, p in enumerate(parsed) if p and p[0] == c
                       and abs(p[1] - cx) < TOL and abs(p[2] - cy) < TOL
                       and abs(p[3] - bw) < TOL and abs(p[4] - bh) < TOL]
                if len(hit) != 1:
                    errs.append(f"#{no} del {a['cls']} 匹配 {len(hit)} 行")
                    continue
                parsed[hit[0]] = None
                n_del += 1
                changed = True
                print(f"#{no} del {a['cls']} cx={cx:.4f}")
            else:  # set
                cx, cy, bw, bh = a["n"]
                hit = [i for i, p in enumerate(parsed) if p and p[0] == c
                       and abs(p[1] - cx) < TOL and abs(p[2] - cy) < TOL
                       and abs(p[3] - bw) < TOL and abs(p[4] - bh) < TOL]
                if len(hit) != 1:
                    errs.append(f"#{no} set {a['cls']} 匹配 {len(hit)} 行")
                    continue
                x1, y1, x2, y2 = a["xyxy"]
                parsed[hit[0]] = [c, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h,
                                  (x2 - x1) / w, (y2 - y1) / h]
                n_set += 1
                changed = True
                print(f"#{no} set {a['cls']} → ({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")

        if changed and apply_it:
            BACKUP.mkdir(parents=True, exist_ok=True)
            bak = BACKUP / lbl_path.relative_to(DATASET / "labels")
            if not bak.exists():
                bak.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(lbl_path, bak)
            out = "\n".join(f"{p[0]} " + " ".join(f"{v:.6f}" for v in p[1:5])
                            for p in parsed if p is not None)
            lbl_path.write_text((out + "\n") if out else "", encoding="utf-8")

    print(f"\n[汇总] del={n_del} set={n_set} add={n_add} apply={apply_it}")
    if errs:
        print("[错误]")
        for e in errs:
            print(" ", e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
