# DEVELOPMENT_GUIDE — 开发协作说明

> 面向：全部 5 名项目成员（新成员入职必读）
> 基线：v0.1.0-baseline（2026-09-06 冻结）
> 配套文档：`docs/INTERFACE_CONTRACT.md`（接口契约）、`docs/PROJECT_STATUS.md`（项目现状）、
> `docs/TEAM_DEVELOPMENT_PLAN.md`（任务分解与排期）、`docs/PROJECT_HANDOFF_REPORT.md`（交接审计）

---

## 1. 环境搭建（从零开始）

**要求**：Windows + Python 3.10+（项目基线为 3.13）；NVIDIA GPU 可选（无 GPU 自动回退 CPU）。

```bash
cd labsafety

# 1. 创建虚拟环境
python -m venv venv

# 2. 安装 GPU 版 PyTorch（约 2.5GB；离线安装见 1.1）
venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
#   仅 CPU 环境：venv\Scripts\pip install torch torchvision

# 3. 安装其余依赖
venv\Scripts\pip install -r requirements.txt
```

### 1.1 离线安装（无外网/网速慢）
项目 `wheels/` 目录含 torch/torchvision cu124 离线 whl（**不入 git**，向组长索取）：
```bash
venv\Scripts\pip install --no-index --find-links=wheels torch torchvision
```

### 1.2 获取模型权重（git 不含 .pt 文件）
`ai/model/sh17_yolov8s.pt`（22.5MB，运行必需）**不在 git 仓库中**，两种获取方式：
```bash
# 方式一：官方下载
curl -L -o ai/model/sh17_yolov8s.pt https://github.com/ahmadmughees/SH17dataset/releases/download/v1/yolo8s.pt
# 方式二：向组长/AI 成员索取（FINAL_MODEL.zip）
```
下载后放到 `ai/model/` 下即可。缺少该文件时服务可启动，但首次检测请求会报错。

## 2. 启动项目

```bash
# 方式一：双击 start.bat
# 方式二：
venv\Scripts\python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
访问 http://127.0.0.1:8000 （接口文档 http://127.0.0.1:8000/docs）。
首次检测请求会触发模型加载（数秒延迟属正常，懒加载设计）。
数据库 `backend/labsafety.db` 首次启动自动创建并写入演示种子配置（实验室A/实验操作区，要求 mask+gloves+lab_coat）。

## 3. 运行测试

```bash
venv\Scripts\python tests\run_tests.py
```
自动执行 T01~T14 用例 + GPU/CPU 性能实测，生成 `tests/test_report.md`。
**基线结果：14/14 全部通过（2026-09-05 实测）**。改完代码必须重跑，报告不通过不得提 PR。

## 4. 运行图片 / 视频检测

**推荐走 Web**：启动后"智能巡检"页上传 `data/test/t01...t10` 图片或 `data/test/demo_lab_video.mp4`。

**命令行方式**（调试用，无需起服务）：
```bash
# 图片：调用 API（需服务已启动）
curl -X POST http://127.0.0.1:8000/api/detect/image -F "file=@data/test/t05_pcr_scientist.jpg" -F "lab_id=1"

# 视频（stride=2）
curl -X POST http://127.0.0.1:8000/api/detect/video -F "file=@data/test/demo_lab_video.mp4" -F "stride=2"
```
请求/响应字段详见 `docs/INTERFACE_CONTRACT.md` 第 6 节。

## 5. Git 分支规范

```
main            稳定版本；只有组长可 push；每次合并打 tag（当前：v0.1.0-baseline）
develop         集成分支；组长从 feature 分支合入；每日同步
feature/ai      AI 成员（数据/训练/评估/模型）
feature/backend 后端成员（backend/ + video/）
feature/frontend 前端成员（backend/static/ + tests/）
feature/docs    文档与 PPT 成员
```

规则：
1. **只在自己的 feature 分支提交**；develop/main 只能由组长合并。
2. feature 分支每 1~2 天同步一次 develop（`git merge develop` 或 rebase），避免长期漂移。
3. **只改自己 owner 的文件**（见第 8 节）；确需改他人文件（如 main.py 挂载一行），在 PR 描述里显式说明，由该文件 owner 评审。
4. 禁止 force push、禁止 reset/rebase 已推送的共享分支。
5. 提 PR 前必须本地跑通 tests/run_tests.py。

## 6. Commit message 规范

格式：`<type>(<scope>): <说明>`

| type | 用途 | 示例 |
|------|------|------|
| feat | 新功能 | `feat(export): 违规记录 CSV 导出` |
| fix | 修 bug | `fix(video): 修复 stride 复用帧时间戳未刷新` |
| refactor | 重构（不改行为） | `refactor(api): main.py 路由拆分到 api/` |
| test | 测试 | `test(detect): 新增多人遮挡用例` |
| docs | 文档 | `docs(contract): 冻结 ViolationEvent 字段` |
| chore | 配置/构建 | `chore(deps): 锁定 ultralytics 版本` |

- 说明必须写清"做了什么"，禁止 `update`、`fix bug` 这类空 commit。
- 一次 commit 一个意图；**禁止把他人 owner 文件的改动混进自己的 commit**。

## 7. 哪些目录/文件不能随便改（红线）

| 路径 | 红线 |
|------|------|
| `data/raw/dataset2/` | **永久 external_test，全员只读**。禁止进训练/验证/调参/挑选。 |
| `data/raw/`、`data/lab_ppe/` | 训练数据。只有 AI 成员经审计流程后可动；他人只读。 |
| `rule_engine/engine.py` 默认规则 | **必需 PPE=mask+gloves+lab_coat、缺1一般/缺≥2严重**，任何人不得改默认值。规则配置走数据库/设置页。 |
| `ai/detector.py` CLASS_ALIAS | 类别映射改动必须走契约评审（E4 定型时统一修订），不得单独改。 |
| `docs/E1~E4 报告、tests/test_report.md` | 已有实验结果，**只追加不覆盖不删除**。 |
| `runs/`、`outputs/` | 实验产物/运行输出，不入 git，不删除。 |
| `backend/labsafety.db` | 运行时库，不入 git，删了会自动重建（演示数据丢失）。 |
| 模型 `.pt` 文件 | 一律不入 git（.gitignore 已排除），经 FINAL_MODEL/AI_TRAINING 包分发。 |

## 8. 成员边界（推荐分工）

| 成员 | Owner 目录/文件 | 说明 |
|------|----------------|------|
| AI | `ai/`、`tools/`、训练数据、`runs/`、模型评估/转换脚本、AI 相关测试 | 生产代码实验不改；模型路径现为 main.py:49 硬编码，切换模型需与后端成员走 PR |
| 后端/视频 | `backend/main.py`、`backend/database.py`、`backend/api/`、`video/processor.py` | API 分层、视频后台任务、数据库接口 |
| 规则 | `rule_engine/engine.py` | 业务规则维护（默认值不改，见第 7 节） |
| 前端 | `backend/static/`（index.html 及新增 JS/CSS） | ECharts 本地化、ROI 可视化、进度显示 |
| 测试 | `tests/` | 用例补充、回归、验收测试 |
| 文档/PPT | `docs/` 新增文档、PPT 工程文件 | 正式文档终稿由组长定稿 |

**跨边界协议**：一切接口语义以 `docs/INTERFACE_CONTRACT.md` 为准；要改契约先提评审（流程见该文档第 0 节）。

## 9. AI ↔ 后端 ↔ 前端 边界速查

```
前端（static/） ──HTTP──► 后端（backend/） ──Python调用──► 管线（video/processor.py）
                                                      ├─► ai/detector.py（模型推理）
                                                      ├─► ai/ppe_matcher.py（人-PPE匹配）
                                                      ├─► ai/roi.py（区域判断）
                                                      └─► rule_engine/engine.py（规则判断）
后端 ──SQL──► backend/database.py ──► labsafety.db
```
- 前端**只**依赖后端 HTTP API（见契约第 6 节），不得 import 后端 Python 模块、不得依赖后端内部变量。
- 后端**只**通过 `InspectionPipeline.analyze_frame(frame, areas)` 使用 AI 能力，不直接调 ultralytics。
- AI 成员换模型/改映射，不改 `video/processor.py`；影响管线时提契约评审。

## 10. 常见坑（Windows 环境实测）

1. **中文路径**：`cv2.imread/imwrite` 读不了中文路径。上传文件已被后端重命名为 ASCII 安全名规避；新写的脚本请用 `cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)` 模式（tools/ 脚本有现成示例）。
2. **训练脚本后台运行**：Windows 下长任务建议前台跑或用 `Start-Process`；孤儿 dataloader 进程用 PowerShell `Stop-Process -Id <pid> -Force` 清理（bash 的 taskkill 无效）。
3. **ECharts CDN**：当前前端走 jsdelivr CDN，断网图表不显示——前端成员优先本地化。
4. **首次请求慢**：模型懒加载设计，属正常。
5. **端口占用**：8000 被占时换 `--port 8001`，前端 fetch 是相对路径无需改代码。
