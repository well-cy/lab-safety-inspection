# -*- coding: utf-8 -*-
"""E2-3 best.pt 评估：dataset1 valid / test 逐类指标 + 推理验证。"""
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
BEST = ROOT / "runs" / "e2" / "labppe_v8s" / "weights" / "best.pt"


def report(r, title):
    print(f"\n===== {title} =====")
    print(f"{'class':10}{'P':>8}{'R':>8}{'F1':>8}{'mAP50':>8}{'mAP50-95':>10}")
    for i in sorted(r.names):
        c = r.names[i]
        P, R, m50, m5095 = r.box.p[i], r.box.r[i], r.box.ap50[i], r.box.ap[i]
        print(f"{c:10}{P:>8.3f}{R:>8.3f}{2*P*R/(P+R):>8.3f}{m50:>8.3f}{m5095:>10.3f}")
    P, R = r.box.mp, r.box.mr
    print(f"{'all':10}{P:>8.3f}{R:>8.3f}{2*P*R/(P+R):>8.3f}{r.box.map50:>8.3f}{r.box.map:>10.3f}")


def main():
    model = YOLO(str(BEST))
    print("best.pt 加载成功，类别:", model.names)

    r = model.val(data=str(ROOT / "data/lab_ppe/data.yaml"), split="val", device=0, verbose=False)
    report(r, "dataset1 valid (best.pt)")

    r2 = model.val(data=str(ROOT / "data/lab_ppe/data.yaml"), split="test", device=0, verbose=False)
    report(r2, "dataset1 test (best.pt)")

    print("\n===== best.pt 单图推理验证 =====")
    img = sorted((ROOT / "data/lab_ppe/images/test").glob("*.jpg"))[0]
    res = model.predict(str(img), device=0, verbose=False)
    d = res[0]
    print(f"输入: {img.name}, 检出 {len(d.boxes)} 个目标")
    for b in d.boxes:
        print(f"  {d.names[int(b.cls)]} conf={float(b.conf):.3f}")
    print("推理验证通过 ✓")


if __name__ == "__main__":
    main()
