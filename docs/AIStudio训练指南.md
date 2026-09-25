# Lab-PPE YOLOv8s 训练（AI Studio · PyTorch 框架版）

> **更新 2026-09-25**：旧版「PaddlePaddle 镜像 + 手动装 torch」方案**作废**——AI Studio 在 PaddlePaddle 镜像里于 import 层锁死 torch（`Cannot run import torch because of system compatibility`），pip/wheel 改名下载等绕法全部无效。
> **唯一正路：创建 Notebook 时项目框架选 PyTorch。**

---

## 三个必知（全是踩过的坑）

| # | 坑 | 正确做法 |
|---|---|---|
| ① | PaddlePaddle 镜像在 import 层锁死 torch | 创建 Notebook 时**框架选 PyTorch** |
| ② | 显卡默认选中 **BI-V150S**（天数智芯国产卡，torch CUDA 用不了） | 手动改选 **V100 16GB**（2 点/小时） |
| ③ | pip 装 2GB 级大包几分钟无输出，像死机 | 一键格已带实时滚动输出；看到 `$ pip install ...` 就耐心等 |

---

## 第 0 步：数据集（已代传，跳过）

- 数据集 `lab-ppe-kaggle`，ID **400147**：https://aistudio.baidu.com/dataset/detail/400147/intro
- 内含 `lab_ppe_kaggle.zip`（1553.4 MB：13507 图 + 13507 标签 + data.yaml；train/valid/test 俱全，mask/gloves/lab_coat/goggles 四类）
- 命令行直传工具：`tools/aistudio_upload.py`（网页拖 1.5GB 会卡死）

---

## 第 1 步：停掉旧环境（省点数）

个人头像 → 算力点/项目 → 找到旧的 `labppe`（PaddlePaddle 那个）**运行中环境 → 停止**。

---

## 第 2 步：创建 Notebook（框架必须选 PyTorch）

按 `outputs/创建notebook指引.png` 打开「创建Notebook」弹窗，共改 4 处：

1. **项目名称**：`labppe-pt`（与旧的区分）
2. **项目框架：选 PyTorch** ← 关键新增步骤。选项一般在弹窗内或展开「高级配置」后的「框架 / 基础镜像」一栏；**如果展开后找不到，截图发我**
3. **数据集配置**：添加数据集 → 搜索勾选 `lab-ppe-kaggle`
4. **高级配置 → 显卡：V100 16GB**
   - IDE 保持 JupyterLab，其余不动 → 点「创建」

---

## 第 3 步：跑一键格

1. 打开 `outputs/AIStudio_PyTorch_照着抄.txt`
2. 把**格 1（主一键格）整段**复制进一个 Cell → Shift+Enter
3. 预计 **2~3 小时**（50 epochs，V100 16GB，batch 16；32GB 卡可改 batch=32），全程实时输出
4. 中断/断线/点数用完都不怕：**重跑同一格**，自动跳过已完成步骤或从 last.pt 续训

txt 里还有：

- **格 2 · 备用 CUDA 修复格**：仅当报 `NVIDIA driver is too old (found version 12000)` 时跑；跑完 Kernel → Restart Kernel，再重跑格 1
- **格 3 · 效果预览格（可选）**：训完出 24 张验证图检测拼版 `valid_preview.jpg`

---

## 训完之后

- 左侧文件树 → `/home/aistudio/work/labppe_weights.zip` → 右键**下载**（best.pt / last.pt / results.csv / 混淆矩阵 / PR 曲线）
- 把 zip 发给 AI → 本地跑 `infer_check.py` 出 external_test 效果拼版，对比 E3 基线（lab_coat external F1 = 0.34）

---

## 常见问题

| 症状 | 处理 |
|---|---|
| `import torch` 报 system compatibility | 创建时框架选成 PaddlePaddle 了 → 停环境重建，框架选 **PyTorch** |
| CUDA 不可用（格 1 assert 报错） | 显卡选错 → 必须 **V100 16GB**，不能用 BI-V150S |
| `NVIDIA driver is too old (found version 12000)` | 跑 txt 里的格 2 修 CUDA → 重启内核 → 重跑格 1 |
| 训练中断（点数不足/断线） | 重跑格 1，自动续训 |
| 没找到 lab_ppe_kaggle.zip | 创建时没挂数据集 → 停环境后编辑配置，勾上 `lab-ppe-kaggle` |
| 显存不足 OOM | batch 16 → 8 |
| 创建弹窗里找不到框架选项 | 展开「高级配置」找「框架/基础镜像」；还没有就截图发我 |

---

## 红线提醒

- 包内 test 集（732 张，源自 dataset2 external_test）**仅供训完自查，禁止参与训练/调参/模型挑选**
- data.yaml 的 `val` 已指向 `images/valid`，训练不会碰 test
- 新增数据只进 train/valid，不进 test（E4 规划）
