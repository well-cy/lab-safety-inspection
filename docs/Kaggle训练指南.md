# Lab-PPE YOLOv8s 训练 Notebook（Kaggle 版）

**用途**：在 Kaggle 免费 GPU 上训练 4 类 PPE 检测模型（mask / gloves / lab_coat / goggles），产出 `best.pt`。

**使用前提**：已把数据集上传为 Kaggle Dataset，slug = `lab-ppe`（若是别的名字，改下面 Cell 1 的 `DATA` 路径）。

---

## Cell 1 — 环境与路径检查

```python
# 1. 装依赖（Kaggle 镜像已带 torch，只需 ultralytics）
!pip install -q ultralytics

# 2. 定位数据集
from pathlib import Path
DATA = Path("/kaggle/input/lab-ppe/lab_ppe_kaggle")

# 若上面路径不存在，自动探测真实挂载点
if not (DATA / "data.yaml").exists():
    hits = list(Path("/kaggle/input").rglob("data.yaml"))
    print("自动探测到:", hits)
    DATA = hits[0].parent
print("数据集:", DATA)
print((DATA / "data.yaml").read_text())

# 3. 核对数量
for sp in ("train", "valid", "test"):
    n = len(list((DATA / "images" / sp).glob("*")))
    l = len(list((DATA / "labels" / sp).glob("*.txt")))
    print(f"{sp:<6} 图片 {n:>6}  标签 {l:>6}")
```

---

## Cell 2 — 修正 data.yaml 路径（关键）

Kaggle 挂载点与本地不同，必须重写 `path`：

```python
import yaml
from pathlib import Path

cfg = yaml.safe_load((DATA / "data.yaml").read_text())
cfg["path"] = str(DATA)                 # 指向实际挂载目录
yaml.safe_dump(cfg, open(DATA / "data.yaml", "w"), sort_keys=False)
print((DATA / "data.yaml").read_text())
```

---

## Cell 3 — 训练

```python
from ultralytics import YOLO
import torch, time

print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无（Kaggle 需开 GPU 开关）")

t0 = time.time()
model = YOLO("yolov8s.pt")              # 官方预训练；Kaggle 能直连 github，自动下载

results = model.train(
    data=str(DATA / "data.yaml"),
    epochs=50,
    imgsz=640,
    batch=16,          # T4 16GB 可上 16~32；爆显存就降到 8
    device=0,
    workers=4,
    project="/kaggle/working/runs",
    name="labppe_v8s",
    exist_ok=True,
    seed=42,
    verbose=True,
    plots=True,
)
print(f"\n训练耗时 {(time.time()-t0)/60:.1f} 分钟")
print("save_dir:", results.save_dir)
```

---

## Cell 4 — 验证集评估

```python
from ultralytics import YOLO
from pathlib import Path

BEST = Path(results.save_dir) / "weights" / "best.pt"
print("best.pt:", BEST, BEST.stat().st_size / 1048576, "MB")

m = YOLO(str(BEST))
val = m.val(data=str(DATA / "data.yaml"), split="val", imgsz=640, device=0)

print(f"\nmAP50    = {val.box.map50:.4f}")
print(f"mAP50-95 = {val.box.map:.4f}")
print("\n各类 AP50:")
for i, name in m.names.items():
    try:
        print(f"  {name:<10} {val.box.ap50[i]:.4f}")
    except Exception:
        pass
```

---

## Cell 5 — 关键类 lab_coat 专项

```python
# lab_coat = 2，是本次补齐的核心目标类
import numpy as np
i = list(m.names.keys())[list(m.names.values()).index("lab_coat")]
print(f"lab_coat (class {i})  AP50 = {val.box.ap50[i]:.4f}")
print("对照：E3 时该模型在 dataset2 external 上 lab_coat F1 = 0.34（跨域）")
```

---

## Cell 6 — 导出权重（下载用）

```python
import shutil
from pathlib import Path

out = Path("/kaggle/working/export")
out.mkdir(exist_ok=True)
shutil.copy2(BEST, out / "best.pt")
shutil.copy2(Path(results.save_dir) / "weights" / "last.pt", out / "last.pt")

# 训练曲线与混淆矩阵（便于写报告）
for f in ("results.csv", "results.png", "confusion_matrix.png",
          "confusion_matrix_normalized.png", "BoxF1_curve.png", "BoxPR_curve.png"):
    p = Path(results.save_dir) / f
    if p.exists():
        shutil.copy2(p, out / f)

# 打包成一个 zip，在 Kaggle 右侧 Output 面板下载
shutil.make_archive("/kaggle/working/labppe_weights", "zip", out)
print("已打包: /kaggle/working/labppe_weights.zip")
print("→ 在 Kaggle 页面右侧 Output / 输出 面板点下载")
```

---

## Cell 7 — 快速出效果图（可选，看新数据表现）

```python
# 用训练好的权重对 valid 抽样推理，出带框拼版
import cv2, numpy as np
from pathlib import Path

m2 = YOLO(str(BEST))
CLR = {0:(255,150,40), 1:(0,150,255), 2:(90,220,60), 3:(200,60,220)}
NAMES = ["mask","gloves","lab_coat","goggles"]

imgs = sorted((DATA/"images"/"valid").glob("*"))[:24]
tiles = []
for p in imgs:
    r = m2.predict(str(p), conf=0.35, imgsz=640, device=0, verbose=False)[0]
    img = cv2.imread(str(p))
    h, w = img.shape[:2]
    if r.boxes is not None:
        for b in r.boxes:
            c = int(b.cls.item()); cf = float(b.conf.item())
            x1,y1,x2,y2 = [int(v) for v in b.xyxy[0]]
            cv2.rectangle(img,(x1,y1),(x2,y2),CLR.get(c,(0,0,255)),max(2,w//300))
            cv2.putText(img,f"{NAMES[c]} {cf:.2f}",(x1+3,max(14,y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX,max(0.5,w/1400),CLR.get(c,(0,0,255)),2)
    tw=320; tiles.append(cv2.resize(img,(tw,int(h*tw/w))))

cols=4; H=max(t.shape[0] for t in tiles)
canvas=np.full(((len(tiles)+cols-1)//cols*(H+8), cols*324,3),24,np.uint8)
for k,t in enumerate(tiles):
    rr,cc=divmod(k,cols); canvas[rr*(H+8):rr*(H+8)+t.shape[0], cc*324:cc*324+t.shape[1]]=t
cv2.imwrite("/kaggle/working/valid_preview.jpg", canvas)
print("已输出 /kaggle/working/valid_preview.jpg")
```

---

## 注意事项

1. **必须开 GPU**：Notebook 右侧 Settings → Accelerator → **GPU T4 x2**（免费额度每周 30 小时）
2. **必须开联网**：Settings → Internet → **On**（否则 `yolov8s.pt` 下不来）
3. **训练时长**：13507 张、50 epochs、T4 约 **2–4 小时**；Kaggle 单次会话上限 12 小时，够用
4. **会话中断会丢产物**：训完**立刻**跑 Cell 6 打包下载；也可在 Cell 6 后把 `/kaggle/working` 存为 Kaggle Dataset
5. **test 集不参与训练与调参**：Kaggle 包里带的 test 仅供训完自查，禁止据此调参（项目红线）
6. **显存不足时**：`batch=16` → `8`，或 `imgsz=640` → `512`
7. **数据只用 train/valid 训练**：`data.yaml` 的 `val` 已指向 `images/valid`，训练过程不会碰 test
