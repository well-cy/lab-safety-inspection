# -*- coding: utf-8 -*-
"""翻车图复测：用户重发的 6 张历史翻车场景（带旧 overlay 烧录），用 E4 新模型复跑。
注意：图上已烧录旧模型检测框，E2_old 结果仅供参考（可能被 overlay 干扰），以 E4_new 为准。
"""
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path("D:/ZuoYe/lab-safety-inspection")
SRC = Path(r"C:/Users/24518/.workbuddy/clipboard-images")
OUT = ROOT / "outputs/flip_retest_20260928"
OUT.mkdir(parents=True, exist_ok=True)

# 用户消息中 @image#1..#6 对应的剪贴板文件（按顺序）
IMAGES = [
    ("flip1", "clipboard-2026-09-28T13-33-05-063Z-3b861fc9.jpg", "实验室-白大褂+手套+口罩"),
    ("flip2", "clipboard-2026-09-28T13-33-05-067Z-a5c46c54.jpg", "台钻-白大褂+黑手套"),
    ("flip3", "clipboard-2026-09-28T13-33-05-070Z-7ceeeaea.jpg", "白色连体防护服棚拍(旧P1翻车)"),
    ("flip4", "clipboard-2026-09-28T13-33-05-073Z-b2199ef0.jpg", "大厅白大褂生活照"),
    ("flip5", "clipboard-2026-09-28T13-33-05-076Z-54b6d686.jpg", "护目镜特写网图"),
    ("flip6", "clipboard-2026-09-28T13-33-05-079Z-0b49a368.jpg", "白色防护服+面罩+黑手套"),
]
# 各图预期应有的 PPE（人工判读）
EXPECT = {
    "flip1": {"lab_coat", "gloves", "mask"},
    "flip2": {"lab_coat", "gloves"},
    "flip3": {"lab_coat", "gloves", "goggles", "mask"},  # 连体服按 Coverall 并入 lab_coat 口径
    "flip4": {"lab_coat", "mask"},                        # 大厅照，无手套属真实违规
    "flip5": {"goggles"},
    "flip6": {"lab_coat", "gloves", "mask"},              # 面罩属 goggles 大类视口径
}

MODELS = {
    "E4_new": str(ROOT / "ai/model/labppe_v8s.pt"),
    "E2_old": str(ROOT / "ai/model/backup/labppe_v8s_E2_20260928.pt"),
}
CONF = 0.25

# 1) 归档原图（只复制，不动剪贴板源文件）
for tag, fname, _ in IMAGES:
    shutil.copy2(SRC / fname, OUT / f"{tag}.jpg")

# 2) 双模型推理 + 保存 E4 标注图
report = ["=" * 64, "翻车图复测 | conf=0.25 | imgsz=640 | CPU", "=" * 64]
for tag, fname, desc in IMAGES:
    img_path = OUT / f"{tag}.jpg"
    report.append(f"\n[{tag}] {desc}")
    for mtag, mpath in MODELS.items():
        model = YOLO(mpath)
        r = model.predict(str(img_path), conf=CONF, imgsz=640, device="cpu", verbose=False)[0]
        det = {}
        for b in r.boxes:
            cls = model.names[int(b.cls)]
            det.setdefault(cls, []).append(round(float(b.conf), 2))
        report.append(f"  {mtag}: {det}")
        if mtag == "E4_new":
            exp = EXPECT[tag]
            hit = set(det) & exp
            miss = exp - set(det)
            extra = set(det) - exp
            verdict = "PASS" if not miss else f"缺 {sorted(miss)}"
            report.append(f"  预期 {sorted(exp)} -> 命中 {sorted(hit)} | 多检 {sorted(extra)} | 判定: {verdict}")
            ann = img_path.with_name(f"{tag}_E4.jpg")
            model.predict(str(img_path), conf=CONF, imgsz=640, device="cpu", verbose=False)[0].save(str(ann))

out_txt = OUT / "flip_retest_report.txt"
out_txt.write_text("\n".join(report), encoding="utf-8")
print("\n".join(report))
print("\n明细与标注图目录:", OUT)
