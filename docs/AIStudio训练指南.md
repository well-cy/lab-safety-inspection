# Lab-PPE YOLOv8s 训练（AI Studio · 经典版 + 手动装 torch）

> **更新 2026-09-25（终版路线）**：实测新版 BML CodeLab 的镜像**写死 PaddlePaddle 3.3.0**——
> 创建弹窗、切换环境弹窗（只有 CPU/异构/BI-V150S/V100 16GB 算力卡）都**没有框架选项**，
> 且 `import torch` 被 import 层锁死（`Cannot run import torch because of system compatibility`）。
> **唯一免费正路 = AI Studio 经典版项目**（无 import 锁，手动装 cu118 torch，近期实测文章证实可行）。

---

## 三个必知（全是踩过的坑）

| # | 坑 | 正确做法 |
|---|---|---|
| ① | BML CodeLab（新版）锁死 torch 且无法换框架 | **创建经典版项目**，手动装 cu118 torch |
| ② | 显卡默认 CPU / BI-V150S（国产卡，torch 用不了） | 启动/切换时选 **V100 16GB**（2 点/小时） |
| ③ | pip 装 2GB 级大包几分钟无输出，像死机 | 一键格已带实时滚动输出；看到 `$ pip install ...` 就耐心等 |

---

## 第 0 步：数据集（已代传，跳过）

- 数据集 `lab-ppe-kaggle`，ID **400147**：https://aistudio.baidu.com/dataset/detail/400147/intro
- 内含 `lab_ppe_kaggle.zip`（1553.4 MB：13507 图 + 13507 标签 + data.yaml；train/valid/test 俱全，mask/gloves/lab_coat/goggles 四类）
- 命令行直传工具：`tools/aistudio_upload.py`（网页拖 1.5GB 会卡死）
- 创建经典版项目时在「数据集」一栏挂上它；挂载路径固定在 `/home/aistudio/data/data400147/`

---

## 第 1 步：停掉旧环境（省点数）

两个旧项目（`labppe`、`labppe-pt`）都是 BML CodeLab，**都用不了，别再纠缠**：
项目详情页 → **停止环境**（V100 挂着每小时烧 2~3 点）。

---

## 第 2 步：创建「AI Studio 经典版」项目（关键步骤）

1. AI Studio 首页 → **项目大厅 → 创建项目** → 类型选 **Notebook**
2. **Notebook 版本选「AI Studio 经典版」**（与 BML Codelab 并列的选项）
3. **框架版本：选最新的 PaddlePaddle（2.6+，对应 Python 3.10）**——选它没关系，经典版没有 import 锁，一键格会手动装 torch
4. 数据集：挂 `lab-ppe-kaggle`；名称 `labppe-classic`
5. 创建 → 详情页点「**启动环境**」→ 算力卡选 **V100 16GB** → 启动

> 创建页找不到「经典版」选项，或启动弹窗里没有 V100 → 截图发我。

---

## 第 3 步：跑一键格（经典版专用）

1. 打开 `outputs/AIStudio_经典版_照着抄.txt`
2. 把**格 1 整段**复制进一个 Cell → Shift+Enter
3. 流程：打印 Python/nvidia-smi 自检 → 装 torch 2.4.1+cu118（约 2GB）→ 装 ultralytics → CUDA 检查 → 解压数据集 → 重写 data.yaml → 下载 yolov8s.pt → 训练 50 epochs（V100 约 2~3 小时）→ 验证 + lab_coat 专项 → 打包 `labppe_weights.zip`
4. 中断/断线/点数用完都不怕：**重跑同一格**，自动跳过已完成步骤或从 last.pt 续训

txt 里还有 **格 2 · 效果预览格**（可选）：训完出 24 张验证图检测拼版 `valid_preview.jpg`

---

## 训完之后

- 左侧文件树 → `/home/aistudio/work/labppe_weights.zip` → 右键**下载**（best.pt / last.pt / results.csv / 混淆矩阵 / PR 曲线）
- 把 zip 发给 AI → 本地跑 `infer_check.py` 出 external_test 效果拼版，对比 E3 基线（lab_coat external F1 = 0.34）

---

## 常见问题

| 症状 | 处理 |
|---|---|
| `import torch` 报 system compatibility | 说明进了 BML CodeLab——必须用**经典版**项目 |
| CUDA 不可用（格 1 报错） | 把格 1 从头开始的输出发 AI：多半这台 V100 驱动 CUDA < 11.8，需换 cu116/cu113 版 torch |
| 镜像 Python < 3.8 | 创建项目时框架版本选**最新 PaddlePaddle**（2.6+）后重建 |
| 切换/启动报「文件过多过大」 | work 里解压的数据集文件太多 → 跑清理格（或干脆换新项目，数据 zip 在挂载里不会丢） |
| 训练中断（点数不足/断线） | 重跑格 1，自动续训 |
| 没找到 lab_ppe_kaggle.zip | 创建时没挂数据集 → 停环境后编辑项目配置重新挂 |
| 显存不足 OOM | batch 16 → 8 |

---

## 红线提醒

- 包内 test 集（732 张，源自 dataset2 external_test）**仅供训完自查，禁止参与训练/调参/模型挑选**
- data.yaml 的 `val` 已指向 `images/valid`，训练不会碰 test
- 新增数据只进 train/valid，不进 test（E4 规划）
