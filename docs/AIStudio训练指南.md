# Lab-PPE YOLOv8s 训练（百度 AI Studio 版，国内直连）

**适用**：Kaggle 注册被 Google reCAPTCHA 挡住的场景。AI Studio 国内直连、中文、百度账号即可。

**总流程**：注册 → 上传数据集 zip → 建 Notebook 项目（挂 GPU + 挂数据集）→ 逐 Cell 粘贴运行 → 下载 `labppe_weights.zip` 发回。

---

## 第 0 步：准备（已替你做好）

- **数据集已上传完毕**（2026-09-24，命令行直传，1.5GB 完整）：
  - 数据集名称：`lab-ppe-kaggle`
  - 数据集 ID：**400147**
  - 链接：https://aistudio.baidu.com/dataset/detail/400147/intro
  - 内含：`lab_ppe_kaggle.zip`（1553.4 MB，13507 图 + 13507 标签 + data.yaml）
- **所以你不用再上传了**，直接从第 3 步「建 Notebook 项目」开始，在启动环境时勾选这个数据集即可。

> 网页端拖拽 1.5GB 会卡死，已改为命令行直传完成。工具：`tools/aistudio_upload.py`

---

## 第 1 步：注册 / 登录（若还没账号）

1. 打开 https://aistudio.baidu.com
2. 用**百度账号**登录（没有就注册，手机号即可，无需 VPN）
3. 右上角头像 → **算力点**：新用户自带赠送点数；**每天签到也送**（免费额度足够训 3–4 小时 V100）

> 训练用 V100 32G 约 1 点/小时级别，50 epochs 总计约 3–4 点，免费额度完全够。

---

## 第 2 步：确认数据集（已代传，跳过上传）

数据集 **400147** 已上传完成，**这一步你不用做任何事**。

如果你想自己核对，打开 https://aistudio.baidu.com/dataset/detail/400147/intro
应该能看到 `lab_ppe_kaggle.zip`（1553.4 MB）。

---

## 第 3 步：建 Notebook 项目

1. 顶部菜单 → **项目** → **创建项目**（名称随意，如 `labppe-train`）
2. 进入项目 → **启动环境**：
   - **AIStudio Classic → Notebook**（选「免费版」即可）
   - **运行环境基础镜像**：选 Python 3.10 / 带CUDA 的镜像（如 `PaddlePaddle 2.6 + Python 3.10`）
   - **算力卡**：选 **V100 32G**
   - **数据集**：勾选 `lab-ppe-kaggle`（ID 400147）
3. 启动后进入 Notebook，**先顶栏保存一份副本**（默认环境是只读的，会提示「保存个人副本」，点确认）

---

## Cell 1 — 安装依赖 + 定位数据集

```python
# 国内有镜像，直连即可
!pip install -q ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple

import zipfile, glob
from pathlib import Path

# 找到挂载的数据集里的 zip 并解压到工作目录
zips = glob.glob("/home/aistudio/data/**/lab_ppe_kaggle.zip", recursive=True)
assert zips, "没找到 lab_ppe_kaggle.zip —— 检查数据集是否挂载"
print("找到:", zips[0])

WORK = Path("/home/aistudio/work")
if not (WORK / "lab_ppe_kaggle" / "data.yaml").exists():
    !unzip -q -o {zips[0]} -d {WORK}
DATA = WORK / "lab_ppe_kaggle"
print("数据集:", DATA)

for sp in ("train", "valid", "test"):
    n = len(list((DATA / "images" / sp).glob("*")))
    l = len(list((DATA / "labels" / sp).glob("*.txt")))
    print(f"{sp:<6} 图片 {n:>6}  标签 {l:>6}")
```

---

## Cell 2 — 重写 data.yaml 路径（关键，不能跳）

```python
import yaml

cfg = yaml.safe_load((DATA / "data.yaml").read_text())
cfg["path"] = str(DATA)          # 指向 AI Studio 实际解压目录
yaml.safe_dump(cfg, open(DATA / "data.yaml", "w", encoding="utf-8"),
               allow_unicode=True, sort_keys=False)
print((DATA / "data.yaml").read_text())
```

---

## Cell 3 — 下载预训练权重（走 hf-mirror，国内可连）

```python
# AI Studio 的服务器访问 github 不稳定，用 hf-mirror 下载官方 yolov8s.pt
import subprocess
from pathlib import Path

PT = Path("/home/aistudio/work/yolov8s.pt")
if not PT.exists():
    r = subprocess.run(
        ["curl", "-sL", "-o", str(PT),
         "https://hf-mirror.com/Ultralytics/YOLOv8/resolve/main/yolov8s.pt"],
        capture_output=True, text=True)
    print("curl:", r.returncode)
print("大小 MB:", PT.stat().st_size / 1048576 if PT.exists() else "下载失败")
assert PT.exists() and PT.stat().st_size > 1e6
```

---

## Cell 4 — 训练（50 epochs，V100 约 2–4 小时）

```python
from ultralytics import YOLO
import torch, time

print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无! 请检查算力卡选择")

t0 = time.time()
model = YOLO(str(PT))            # 用 Cell 3 下载的本地权重
results = model.train(
    data=str(DATA / "data.yaml"),
    epochs=50,
    imgsz=640,
    batch=32,                    # V100 32G 可到 32；若报显存不足降到 16
    device=0,
    workers=8,
    project="/home/aistudio/work/runs",
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

## Cell 5 — 验证集评估 + lab_coat 专项

```python
from ultralytics import YOLO
from pathlib import Path

BEST = Path(results.save_dir) / "weights" / "best.pt"
print("best.pt:", BEST, round(BEST.stat().st_size / 1048576, 1), "MB")

m = YOLO(str(BEST))
val = m.val(data=str(DATA / "data.yaml"), split="val", imgsz=640, device=0)

print(f"\nmAP50    = {val.box.map50:.4f}")
print(f"mAP50-95 = {val.box.map:.4f}")
for i, name in m.names.items():
    try:
        print(f"  {name:<10} AP50 = {val.box.ap50[i]:.4f}")
    except Exception:
        pass

i = list(m.names.values()).index("lab_coat")
print(f"\n[关键] lab_coat AP50 = {val.box.ap50[i]:.4f}")
print("对照：E3 基线在 dataset2 external 上 lab_coat F1 = 0.34（跨域）")
```

---

## Cell 6 — 打包下载（训完立刻跑！）

```python
import shutil
from pathlib import Path

out = Path("/home/aistudio/work/export")
out.mkdir(exist_ok=True)
shutil.copy2(BEST, out / "best.pt")
shutil.copy2(Path(results.save_dir) / "weights" / "last.pt", out / "last.pt")
for f in ("results.csv", "results.png", "confusion_matrix.png",
          "confusion_matrix_normalized.png", "BoxF1_curve.png", "BoxPR_curve.png"):
    p = Path(results.save_dir) / f
    if p.exists():
        shutil.copy2(p, out / f)

shutil.make_archive("/home/aistudio/work/labppe_weights", "zip", out)
print("已打包: /home/aistudio/work/labppe_weights.zip")
```

**下载方法**：Notebook 左侧文件树 → 找到 `/home/aistudio/work/labppe_weights.zip` → 右键 **下载**。
（环境关停后文件会保留一段时间，但**一定训完就下**，最稳。）

---

## Cell 7 — 效果预览（可选）

```python
import cv2, numpy as np
from pathlib import Path

m2 = YOLO(str(BEST))
CLR = {0:(255,150,40), 1:(0,150,255), 2:(90,220,60), 3:(200,60,220)}
NAMES = ["mask","gloves","lab_coat","goggles"]

imgs = sorted((DATA/"images"/"valid").glob("*"))[:24]
tiles = []
for p in imgs:
    r = m2.predict(str(p), conf=0.35, imgsz=640, device=0, verbose=False)[0]
    img = cv2.imread(str(p)); h, w = img.shape[:2]
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
cv2.imwrite("/home/aistudio/work/valid_preview.jpg", canvas)
print("已输出 /home/aistudio/work/valid_preview.jpg（左树右键下载）")
```

---

## 常见问题

| 症状 | 处理 |
|---|---|
| 启动时选不到 V100 | 换时段（高峰可能排队），或先选「免费单卡 16G」也能训，batch 降到 16 |
| 训练中断（点数不足/超时） | 环境里 `last.pt` 还在，把 Cell 4 的起点改为 `YOLO('/home/aistudio/work/runs/labppe_v8s/weights/last.pt')` 并把 epochs 顺延（resume 逻辑：`model.train(resume=True)`） |
| pip 装不上 | 换镜像 `-i https://mirrors.aliyun.com/pypi/simple/` |
| 数据集没挂载 | 项目启动页「数据集」一栏勾选 `lab-ppe` 后重新启动环境 |
| 环境是只读 | 首次进入点「保存个人副本」再运行 |
| 显存不足 | `batch=32 → 16 → 8` |

## 红线提醒

- test 集（732 张）在包里**仅供训完自查**，禁止参与训练/调参（项目 external_test 红线）
- data.yaml 的 `val` 已指向 `images/valid`，训练不会碰 test
