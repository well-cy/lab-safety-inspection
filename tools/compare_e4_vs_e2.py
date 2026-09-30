"""E4 重训模型 vs E2：demo_images 批量离线对比（不依赖后端）"""
import random
from pathlib import Path

from ultralytics import YOLO

ROOT = Path("D:/ZuoYe/lab-safety-inspection")
DEMOS = sorted((ROOT / "outputs/demo_images").glob("*.jpg"))
random.seed(42)
sample = random.sample(DEMOS, 40)

MODELS = {
    "E4_new": str(ROOT / "ai/model/labppe_v8s.pt"),
    "E2_old": str(ROOT / "ai/model/backup/labppe_v8s_E2_20260928.pt"),
}
CONF = 0.25

results = {k: {} for k in MODELS}
for tag, mp in MODELS.items():
    model = YOLO(mp)
    for img in sample:
        r = model.predict(str(img), conf=CONF, imgsz=640, device="cpu", verbose=False)[0]
        det = {}
        for b in r.boxes:
            cls = model.names[int(b.cls)]
            det.setdefault(cls, []).append(round(float(b.conf), 2))
        results[tag][img.name] = det

lines = []
better = worse = same = 0
for img in sample:
    e4 = results["E4_new"][img.name]
    e2 = results["E2_old"][img.name]
    cls4, cls2 = set(e4), set(e2)
    if cls4 > cls2:
        tag = "E4多检" + str(cls4 - cls2)
        better += 1
    elif cls2 > cls4:
        tag = "E4漏检" + str(cls2 - cls4)
        worse += 1
    else:
        tag = "一致"
        same += 1
    lines.append(f"[{tag}] {img.name[:40]}\n  E4_new: {e4}\n  E2_old: {e2}")

summary = [
    "=" * 60,
    f"样本: {len(sample)} 张 demo_images | conf={CONF} | imgsz=640 | CPU",
    f"E4 检出更多类: {better} 张 | E4 少检: {worse} 张 | 完全一致: {same} 张",
    "=" * 60,
]
out = ROOT / "outputs/e4_retrain_unpack/demo_compare.txt"
out.write_text("\n".join(summary + lines), encoding="utf-8")
print("\n".join(summary))
print("明细已写入", out)
