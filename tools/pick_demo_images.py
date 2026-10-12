# -*- coding: utf-8 -*-
"""
E4 演示图扩大筛选：用 E2 模型扫数据集，挑出"实验服+其他PPE"齐全的保底演示图。
放宽口径（相比首版）：lab_coat conf>=0.4，mask/gloves/goggles 至少 1 类。
同源家族（.rf. 前的原始名）只留 lab_coat 置信度最高的一张。
"""
import os, random, shutil, sys
from collections import defaultdict
from ultralytics import YOLO
import cv2

ROOT = r"D:\ZuoYe\lab-safety-inspection"
OUT = os.path.join(ROOT, "outputs", "demo_images")
os.makedirs(OUT, exist_ok=True)

CONF_LABCOAT = float(sys.argv[1]) if len(sys.argv) > 1 else 0.40
N_TRAIN = int(sys.argv[2]) if len(sys.argv) > 2 else 1200
N_VALID = int(sys.argv[3]) if len(sys.argv) > 3 else 800

model = YOLO(os.path.join(ROOT, "ai", "model", "labppe_v8s.pt"))

def family(name: str) -> str:
    base = name.split(".rf.")[0]
    for suf in ("_jpg", "_jpeg", "_png"):
        if base.lower().endswith(suf):
            base = base[: -len(suf)]
            break
    return base

pools = {}
for split in ("test", "train", "valid"):
    d = os.path.join(ROOT, "data", "lab_ppe", "images", split)
    fs = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    pools[split] = fs
print(f"pool: test={len(pools['test'])} train={len(pools['train'])} valid={len(pools['valid'])}")

random.seed(42)
scan = list(pools["test"])
scan += random.sample(pools["train"], min(N_TRAIN, len(pools["train"])))
scan += random.sample(pools["valid"], min(N_VALID, len(pools["valid"])))
# test 集可能已在 outputs，仍重扫以统一按最高置信度择优

best = {}   # family -> (labcoat_conf, split, filename, cls_dict)
n_seen = 0
for split in ("test", "train", "valid"):
    subset = [f for f in scan if f in set(pools[split])]
    # 按目录逐个处理（test 全量在前）
for split in ("test", "train", "valid"):
    d = os.path.join(ROOT, "data", "lab_ppe", "images", split)
    files = pools[split]
    if split == "test":
        use = files
    elif split == "train":
        use = [f for f in files if f in set(scan)]
    else:
        use = [f for f in files if f in set(scan)]
    for f in use:
        p = os.path.join(d, f)
        img = cv2.imread(p)
        if img is None:
            continue
        r = model.predict(img, conf=0.30, verbose=False)[0]
        cls = {}
        for b in r.boxes:
            n = model.names[int(b.cls)]
            cls.setdefault(n, 0.0)
            cls[n] = max(cls[n], float(b.conf))
        n_seen += 1
        lc = cls.get("lab_coat", 0.0)
        others = sum(1 for k in ("mask", "gloves", "goggles") if k in cls)
        if lc >= CONF_LABCOAT and others >= 1:
            fam = family(f)
            if fam not in best or lc > best[fam][0]:
                best[fam] = (lc, split, f, cls)
    print(f"[{split}] scanned, cum good families: {len(best)}")

print(f"total scanned={n_seen}, qualified families={len(best)}")
copied = 0
for fam, (lc, split, f, cls) in sorted(best.items(), key=lambda x: -x[1][0]):
    src = os.path.join(ROOT, "data", "lab_ppe", "images", split, f)
    dst = os.path.join(OUT, f)
    if not os.path.exists(dst):
        shutil.copy(src, dst)
        copied += 1
print(f"copied {copied} new images -> {OUT} (total files now: {len(os.listdir(OUT))})")
for fam, (lc, split, f, cls) in list(sorted(best.items(), key=lambda x: -x[1][0]))[:15]:
    print(f"  {f[:46]} lab_coat={lc:.2f} " +
          " ".join(f"{k}:{v:.2f}" for k, v in sorted(cls.items()) if k != 'lab_coat'))
