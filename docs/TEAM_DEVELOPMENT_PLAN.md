# TEAM_DEVELOPMENT_PLAN — 多人协作开发方案

> 编制日期：2026-09-06
> 编制依据：`docs/PROJECT_HANDOFF_REPORT.md`（2026-09-06 只读交接审计，全部结论经代码核实）
> 项目名称：基于计算机视觉的实验室安全智能巡检与违规预警系统
> 当前基线：MVP v0.1（14/14 测试通过）+ E1~E3 完成、E4 暂停待 B1
> 本文档性质：**协作方案与代码切分设计**。本文档不含任何已执行的训练/下载/代码修改动作。

---

## 1. 五人职责总览

| 成员 | 角色 | 一句话定位 | 核心产出 |
|------|------|-----------|---------|
| 成员1 | 项目负责人 | 架构 + 集成 + 发布 + Git 管理 + 正式文档 | Git 仓库与基线、接口冻结、集成联调记录、三套交付包、部署手册 |
| 成员2 | AI算法 | 数据 + 训练 + 评估 + 模型定型 | E4 增训模型、三模型对比报告、MODEL_CARD、生产切换方案 |
| 成员3 | 后端+视频 | API 分层 + 视频后台任务 + 模型接入联调 | api/ 分层路由、视频后台任务+进度查询、配置化模型加载 |
| 成员4 | 前端+测试+Demo | ECharts 本地化 + ROI 可视化 + 全量测试 + 录屏 | 本地化前端、ROI 拖拽编辑、进度条、回归测试报告、Demo 视频 |
| 成员5 | PPT+演讲+独立功能 | 违规记录导出功能 + PPT + 答辩 | backend/export.py、CSV/Excel 导出 API 与前端入口、PPT 成稿、演讲稿 |

**边界重申**（防止扯皮）：
- 成员1 **不是**纯管理/文档人员——负责跨模块接口设计、最终集成与验收。
- 成员4 **不负责** PPT。
- 成员5 **不负责** 正式项目文档、Demo 视频、AI 训练。
- 成员2 **严禁**：dataset2 进入训练、Coverall/medical-suit/safety-suit 映射为 lab_coat、未经审计混合数据、以修改生产代码作为 AI 实验手段。

---

## 2. 文件 Ownership 表（按当前真实目录）

> 原则：**每个文件只有一个 owner；跨模块修改一律走 PR，由 owner 审核合入。**

### 2.1 成员1（架构/集成/文档/配置/发布）

| 文件/目录 | 说明 |
|-----------|------|
| `README.md` | 终稿维护（启动方式、双模型说明更新） |
| `requirements.txt` / 新增 `requirements.lock` | 依赖锁定（torch 2.6.0+cu124、Python 3.13 基准） |
| `start.bat` | 启动脚本（如 lifespan 升级后同步） |
| `docs/` 中正式文档（部署手册、最终验收报告、MODEL_CARD 终审） | 成员2 起草 MODEL_CARD，成员1 终审 |
| `.gitignore` / `.git/` / tag / release | 版本控制唯一管理员 |
| 打包脚本（新增 `scripts/package.py` 或文档化流程） | 三套 zip 的唯一制作人 |

### 2.2 成员2（AI 算法）

| 文件/目录 | 说明 |
|-----------|------|
| `ai/detector.py` | 含 CLASS_ALIAS 修订（生产切换前必须补 lab_coat/goggles 键、处理 suit/coat 兜底映射口径） |
| `ai/ppe_matcher.py` | 稳定维护；若做双模型融合，**新建** `ai/pipeline_dual.py`，不改 processor.py |
| `ai/roi.py` | 稳定维护 |
| `ai/model/` | 生产权重目录（E4 定型后新权重经 PR 流程放入） |
| `tools/` 全部 13 个脚本 | E1~E4 实验线 + 新增 `audit_b1.py`、`train_e4.py`、`eval_e4_compare.py` |
| `runs/`、`data/lab_ppe/`、`data/raw/`（**dataset2 只读！**）、根目录 `yolov8s.pt` | 训练数据与产物（不进 git，见第 6 节） |

### 2.3 成员3（后端+视频）

| 文件/目录 | 说明 |
|-----------|------|
| `backend/main.py` | **核心交点文件，成员3 独占**（见 2.6 交点消解方案） |
| `backend/database.py` | 设置接口 SQL 收敛到此 |
| `backend/api/` | 分层路由改造（空壳已预留）：`detect.py` / `stats.py` / `settings.py` |
| `backend/config.py`（新增） | 模型路径配置化，消除 main.py:49 硬编码 |
| `video/processor.py` | 视频后台任务、进度查询；不改 AI 管线四步逻辑 |

### 2.4 成员4（前端+测试+Demo）

| 文件/目录 | 说明 |
|-----------|------|
| `backend/static/index.html` | ECharts 本地化、Dashboard、ROI 拖拽、进度条、异常处理 |
| `backend/static/vendor/`（新增） | 本地化 JS 库（echarts.min.js 等） |
| `backend/static/export.js`（挂载点预留，内容由成员5填充） | 见 2.6 |
| `tests/` 全部 | run_tests.py 扩展、E4 后全量回归、验收测试 |
| Demo 视频工程文件 | 录屏/剪辑工程（成果放 outputs/，不进 git） |

### 2.5 成员5（PPT+演讲+独立功能）

| 文件/目录 | 说明 |
|-----------|------|
| `backend/export.py`（新增） | 独立功能核心：CSV/Excel 导出、筛选逻辑 |
| `backend/api/export.py`（新增） | 导出路由（挂载见 2.6） |
| `backend/static/export.js`（新增） | 导出按钮与前端交互（成员5 自己的新文件） |
| `docs/PPT_大纲.md`、`docs/演讲稿.md`、PPT 工程文件 | PPT 与答辩 |
| `rule_engine/extension.py`（可选扩展，**不改 engine.py**） | 若做分区分级策略扩展 |

### 2.6 ⚠️ 三个交点文件的冲突消解方案（本方案核心设计）

审计确认 `backend/main.py` 是所有改动的汇聚点。设计目标是：**把"必改他人文件"的动作压缩到接近零**。

**交点①：模型加载（main.py:49 硬编码 `sh17_yolov8s.pt`）**
- 方案：Phase 1 由成员3 落地 `backend/config.py`——模型路径从环境变量 `LABSAFETY_MODEL_PATH` 读取，缺省回退 SH17。此后**切换模型 = 改配置/环境变量，零代码修改**。
- 成员2 的动作变为：交付新权重文件 + 在 PR 里给出"权重路径 + CLASS_ALIAS 修订"两个信息；main.py **不动**。
- 兜底：若 Phase 3 仍需动 main.py，diff 上限 3 行，PR 给成员3。

**交点②：成员5 新增导出路由（需要 main.py 挂载）**
- 方案：成员5 在 `backend/api/export.py` 写标准 `router = APIRouter()`；成员3 在 api 分层改造时于 main.py 加**一行** `app.include_router(export_router, prefix="/api")`。
- 时序：成员3 的 T3-1（分层改造）排在成员5 合入之前；成员5 在 develop 上以桩路由先行开发，无需等 main.py。

**交点③：成员5 的导出按钮要进 index.html（成员4 独占文件）**
- 方案：Phase 1 成员4 在"违规记录"Tab 预留 `<div id="export-panel"></div>` 与一行 `<script src="/static/export.js"></script>`（成员4 自己写，共 2 行）；成员5 的全部 UI 代码放在**自己的新文件 `export.js`** 内，动态渲染进该容器。
- 结果：index.html 的 diff 恒为成员4 自己的 2 行，成员5 永不改 index.html。

> 结论：三交点全部消解为 **"新文件 + ≤2 行挂载"** 模式，main.py/index.html 的非 owner diff 合计不超过 5 行。

---

## 3. 任务分解（可执行任务表）

> 格式：成员 → 任务 → 具体文件 → 具体修改内容 → 输入 → 输出 → 依赖 → 验收标准 → 交付物。
> 每项任务单独建 branch、单独 PR、单独验收。

### 3.1 成员1 任务表

**T1-1 Git 仓库初始化与基线冻结**
- 文件：项目根目录 `.git/`、`.gitignore`（内容见第 6 节）
- 修改：`git init` → 提交当前全部生产代码/测试/工具/文档 → 打 tag `v0.1.0-baseline`
- 输入：交接审计报告第 1 节的目录分类
- 输出：远端仓库（建议校内 GitLab / Gitee 私仓）+ main 分支 + tag
- 依赖：无（**阻塞所有人的 Phase 0 任务**）
- 验收：五个成员均能 clone 并 `start.bat` 跑通 MVP；`git status` 干净；1.6GB data/ 与 2.4GB wheels/ 未入库
- 交付：仓库 URL + 基线 tag + 分支策略公告

**T1-2 依赖锁定**
- 文件：新增 `requirements.lock`（`pip freeze` 产物），原 `requirements.txt` 保留为语义清单
- 修改：锁定 fastapi/uvicorn/ultralytics/torch 2.6.0+cu124/opencv-python/numpy 版本
- 验收：新机器按 lock 安装后 14/14 测试通过
- 交付：requirements.lock + 一条 README 安装说明 PR

**T1-3 跨模块接口冻结（架构裁决）**
- 文件：新增 `docs/INTERFACE_CONTRACT.md`
- 修改：冻结 `Detection` dataclass 字段、`InspectionPipeline.analyze_frame(frame, areas)` 签名、`ViolationEvent` 字段、`match_person_ppe` 输入输出、API 返回 JSON schema
- 验收：文档列出全部 5 个契约 + 变更流程（需成员1 主持评审才可变更）
- 交付：接口契约文档 v1
- 说明：这是成员2（模型切换）、成员3（后台任务）、成员5（导出）三方并行不打架的法律基础

**T1-4 最终集成与联调（Phase 3）**
- 文件：`docs/INTEGRATION_TEST_LOG.md`（新增）
- 修改：模型切换后主流程走查（图片/视频/统计/设置/导出五链路）、冲突裁决记录
- 依赖：T2-6、T3-5、T4-4、T5-1
- 验收：五链路全部通过且记录在案
- 交付：联调日志

**T1-5 正式文档定稿（Phase 4）**
- 文件：`README.md`、新增 `docs/部署手册.md`、新增 `docs/最终验收报告.md`
- 修改：README 补双模型/配置化启动；部署手册含离线安装（wheels/ 路径）与 License 合规说明
- 交付：终稿文档

**T1-6 三套交付包**
- 文件：新增 `scripts/package.py`（或文档化打包清单）
- 修改：按交接报告第 9 节打包 PROJECT_SOURCE.zip（<100MB）/ AI_TRAINING.zip（约 1.5~2GB）/ FINAL_MODEL.zip（<80MB）
- 验收：SOURCE 包解压后全新环境可跑通 MVP；FINAL_MODEL 含 MODEL_CARD 与 License 说明
- 交付：三个 zip + 校验（体积、MD5）

### 3.2 成员2 任务表

**T2-1 B1 自采数据审计（B1 到位后 T0 启动）**
- 文件：新增 `tools/audit_b1.py`（复用 dedup_b3.py 的文件名/MD5+SHA256/pHash 三查框架）
- 修改：B1↔dataset1 / B1↔dataset2 / B1↔B3 三向去重；标注质量抽检（复用 visualize_b3.py 逻辑出抽检图）
- 输入：用户提供的 B1 图片（约 150 张）+ 标注
- 输出：`data/b1_dedup_report.txt` + `outputs/b1_label_check/` 抽检图
- 验收：与 dataset2 零泄漏（铁律）；泄漏图像剔除清单明确
- 交付：审计报告
- ⚠️ 红线：任何 B1↔dataset2 重复图**剔除 B1 侧**，dataset2 永不动

**T2-2 B3 清洗决策执行**
- 文件：新增 `tools/clean_b3.py`（生成剔除清单，不物理删除原始数据，用清单文件驱动训练集组装）
- 修改：剔除 20 张 getty/istock 水印图 + 5 张与 dataset1 近重复图（此前已向用户报备，建议默认执行，执行前在群里贴清单二次确认）
- 输出：`data/b3_keep_list.txt`（约 450 张白名单）
- 验收：剔除清单与 dedup_b3.py 审计报告逐条对得上
- 交付：白名单文件

**T2-3 E4 训练集组装与训练**
- 文件：新增 `tools/make_e4_split.py`、`tools/train_e4.py`（基于 train_e2.py 改：`YOLO('runs/e2/labppe_v8s/weights/best.pt')` 续训，epochs=30、lr0=0.001、imgsz=640、batch=16、seed=42）
- 修改：B3(≈450)+B1(≈150) 按 90/10 加入 train/valid；**原 dataset1 split 一行不动**；dataset2 不出现在任何 split
- 输出：`runs/e4/labppe_v8s_e4/`（best.pt/last.pt/results.csv）
- 依赖：T2-1、T2-2、用户 B1 数据到位
- 验收（已与用户约定的门槛）：dataset1 test lab_coat F1≥0.85；mask/gloves/goggles 各回退≤3pp；无 NaN；best.pt 可推理
- 交付：训练产物 + `docs/E4_train_report.md`

**T2-4 三模型重评（完整 E3 重跑）**
- 文件：新增 `tools/eval_e4_compare.py`（复用 eval_e3_dataset2.py / eval_e3_t01t10.py 双脚本骨架，SH17 vs LAB_PPE-E2 vs LAB_PPE-E4 三模型同口径）
- 修改：dataset2 external_test + t01~t10 + dataset1 valid/test 三个维度全部重跑（conf=0.35、IoU@0.5、11点AP50，口径与 E3 完全一致）
- 输出：`docs/E4_eval_report.md` 三模型对比表
- 验收：dataset2 lab_coat F1≥0.50（最低线，≥0.70 为优秀，<0.50 暂停并上报）；goggles≥0.327；t01/t02/t07 lab_coat 漏检改善情况如实记录
- 交付：对比报告 + 原始预测 JSON

**T2-5 MODEL_CARD**
- 文件：新增 `docs/MODEL_CARD.md`
- 修改：每个模型（sh17_yolov8s.pt / labppe_v8s.pt / e4 增训版）的类别、训练数据来源与 License、指标（E2/E3/E4 三轮）、加载方式、已知局限（lab_coat 域偏移、goggles 跨域回退、连体服不映射的设计取舍）
- 交付：MODEL_CARD（成员1 终审）

**T2-6 生产切换方案（Phase 3，唯一触碰生产链路的任务）**
- 文件：`ai/detector.py`（自己的文件）、新权重落入 `ai/model/`
- 修改：① CLASS_ALIAS 补 LAB_PPE 原生类别键；② 修订 suit/coat 兜底映射（Coverall → 丢弃而非 lab_coat，与 E4 决策对齐）；③ 按 config 方案登记新权重路径
- 输入：T2-4 验收通过 + 用户批准切换
- 输出：PR（ai/detector.py + ai/model/ 新文件 + config 示例），main.py 零 diff 或 ≤3 行
- 验收：切换后全量回归 14/14 + dataset2 指标不劣于 E4 报告值 ±1pp
- 交付：切换 PR + 切换前后指标对照表

### 3.3 成员3 任务表

**T3-1 API 分层改造（main.py 瘦身）**
- 文件：`backend/api/detect.py`、`backend/api/stats.py`、`backend/api/settings.py`（新增）；`backend/main.py`（改）
- 修改：13 个端点按域拆入三个 APIRouter；main.py 只保留 app 创建、startup、静态挂载、include_router（预计从 247 行降到 ~80 行）；**所有路由行为保持字节级等价**（仅搬移不改逻辑）
- 输入：交接报告第 2 节端点清单
- 输出：分层后的 backend
- 验收：改造前后 `curl` 13 个端点返回 JSON 逐字段一致（成员4 协助核对）；前端零改动可运行
- 交付：PR + 端点对照核对表
- ⚠️ 同时预留 export 路由挂载行（含注释 `# reserved for member5 export router`）

**T3-2 lifespan 升级**
- 文件：`backend/main.py`
- 修改：`@app.on_event("startup")`（已弃用写法）→ `async with lifespan(app)` 上下文管理器；建库/种子逻辑不变
- 验收：启动无 DeprecationWarning；数据库初始化行为与现在一致
- 交付：PR

**T3-3 database 接口收敛**
- 文件：`backend/database.py`、`backend/api/settings.py`
- 修改：main.py 中设置类端点的裸 SQL 迁入 database.py 函数（`update_area`/`update_rule`/`get_areas_for_edit` 等）；api 层只调函数
- 验收：设置页四类操作（区域增删改/规则改/阈值改）回归通过；database.py 无 api 层概念混入
- 交付：PR

**T3-4 视频后台任务 + 进度查询**
- 文件：`video/processor.py`（新增任务调度部分）、`backend/api/detect.py`（新增 status 端点）、新增 `backend/jobs.py`（内存任务表 job_id→{status,progress,result}）
- 修改：`POST /api/detect/video` 改为提交任务立即返回 `job_id`；新增 `GET /api/video/status/{job_id}` 返回进度（已处理帧/总帧/当前违规计数）；处理线程完成后落库
- 输出：异步视频接口 + 进度 API（接口契约先按 T1-3 冻结文档对齐）
- 验收：上传 60s 视频期间其他 API 正常响应（演示同步阻塞已消除）；进度百分比单调递增；任务结果与同步版一致（同一视频同一 stride 结果 diff 为空）
- 交付：PR + 自测记录

**T3-5 模型路径配置化**
- 文件：新增 `backend/config.py`
- 修改：`MODEL_PATH = os.getenv("LABSAFETY_MODEL_PATH", "ai/model/sh17_yolov8s.pt")`；main.py:49 改读 config（**这是消解交点①的关键任务，优先做**）
- 验收：不设环境变量行为与现在完全一致；设置环境变量后加载对应模型（用 tools/ 的 best.pt 验证一次即可）
- 交付：PR + 使用说明（写进 README 由成员1 合入）

**T3-6 摄像头实时巡检（时间允许才做，Phase 3+）**
- 文件：新增 `backend/api/camera.py`（不动现有文件）
- 修改：`VideoCapture(0)` 拉流 + 定帧巡检 + MJPEG 或 WebSocket 推送
- 验收：本机摄像头实时画面叠加违规标注，延迟 <500ms
- 交付：独立 PR（可整体放弃不影响主线）

**T3-7 模型接入联调（Phase 3）**
- 文件：`backend/main.py`（如 T2-6 需 ≤3 行 diff）
- 修改：合入成员2 切换 PR；组织切换后冒烟
- 交付：合入记录

### 3.4 成员4 任务表

**T4-1 ECharts 本地化（最高优先级——答辩断网风险）**
- 文件：新增 `backend/static/vendor/echarts.min.js`；改 `backend/static/index.html`
- 修改：删除 `https://cdn.jsdelivr.net/npm/echarts@5.5.0/...` CDN 引用，改为本地 `/static/vendor/echarts.min.js`
- 输入：echarts 5.5.0 官方 dist（约 1MB，**这是下载 JS 库文件，不是下载数据集，属允许范围**）
- 验收：**断网状态下** Dashboard 与数据统计 Tab 图表正常渲染
- 交付：代码 + 断网截图（前后对照）+ 操作记录

**T4-2 前端异常处理**
- 文件：`backend/static/index.html`
- 修改：fetch 统一封装超时/非 200/JSON 解析错误的 try-catch 与 Toast 提示；加载态；ECharts 容器空数据兜底文案
- 验收：后端停止时前端显示"服务不可达"提示而非白屏；每个 Tab 无 console 报错
- 交付：PR + 异常场景截图集

**T4-3 Dashboard 完善**
- 文件：`backend/static/index.html`
- 修改：违规等级/区域/PPE 类型分布图美化；时间范围筛选器（联动 /api/statistics 已有参数）
- 验收：四类图表数据与 API 返回一致（抽查 3 组）；视觉走查通过
- 交付：PR

**T4-4 ROI 可视化编辑**
- 文件：`backend/static/index.html`（可拆出 `backend/static/roi_editor.js`——仍是成员4 自己的文件）
- 修改：设置页新增"画布模式"：在巡检截图/示例图上鼠标拖拽绘制矩形 ROI → 自动换算归一化坐标回填表单 → 保存走现有设置 API
- 输入：现有手填坐标交互 + `/api/settings` 端点（不改后端）
- 验收：拖拽画出 ROI 保存后，巡检结果中 `area_name` 标注正确（用 t01~t10 图回归核对）；坐标保存后刷新不丢失
- 交付：PR + 操作录屏片段

**T4-5 视频进度显示**
- 文件：`backend/static/index.html`
- 修改：视频检测 Tab 改造——轮询/长连接 `GET /api/video/status/{job_id}`，渲染进度条 + 实时违规计数；完成后展示结果
- 依赖：**T3-4**（接口契约可先按冻结文档写桩并行开发，联调时接真）
- 验收：上传视频后进度条从 0→100%；期间页面可切 Tab 操作不卡死
- 交付：PR + 录屏

**T4-6 功能测试与回归测试**
- 文件：`tests/run_tests.py`（扩展）、新增 `tests/test_export.py`、`tests/test_video_async.py`
- 修改：① Phase 1 为成员5 的导出功能补 API 测试（CSV/Excel、三类筛选各一例）；② Phase 2 为视频后台任务补测试；③ **Phase 3 模型切换后全量回归**：T01~T14 + t01~t10 + dataset2 三模型指标核对 + 新增双模型用例
- 依赖：T2-6（回归）、T5-1、T3-4
- 验收：回归 14/14+N 全绿；test_report.md 重新生成并含模型切换前后对照
- 交付：`tests/test_report.md`（新版）+ 回归清单

**T4-7 Demo 视频**
- 文件：新增 `docs/DEMO_SCRIPT.md`（脚本）+ 视频工程；成果 `outputs/demo_final.mp4`
- 修改：端到端演示——启动→图片巡检→ROI 编辑→视频巡检（带进度）→Dashboard→违规导出→（若 T3-6 完成）摄像头实时；配字幕与解说
- 依赖：Phase 3 全部完成后录制（素材用最终版系统）
- 验收：3~5 分钟成片，画质 1080p，字幕与操作同步，全程**本地资源**（验证过断网可播）
- 交付：mp4 + 字幕文件 + 录制工程

### 3.5 成员5 任务表

**T5-1 违规记录导出功能（独立软件功能，核心交付）**
- 文件：新增 `backend/export.py`（纯逻辑层）+ `backend/api/export.py`（路由层）+ `backend/static/export.js`（前端入口）
- 修改：
  - `export.py`：`export_violations(fmt, date_from, date_to, severity, ppe_type) -> (filename, bytes)`；CSV 用标准库 csv；Excel 用 openpyxl（需成员1 批准后加入 requirements）；复用 `database.py` 查询函数（只读，不改 database.py——如需新查询函数，提 PR 给成员3 合入，diff 限新增函数）
  - `api/export.py`：`GET /api/export/violations?fmt=csv|xlsx&...` → `StreamingResponse` 附件下载；入参校验（日期格式/枚举值）
  - `export.js`：在成员4 预留的 `#export-panel` 容器渲染"格式选择 + 日期/等级/PPE 筛选 + 导出按钮"，点击即下载
- 输入：violation_events 表结构（交接报告 4.6 节）+ `/api/violations` 现有返回格式
- 输出：三个新文件，**不改任何现有生产代码**（main.py/index.html 挂载行由 owner 各自完成）
- 依赖：T3-1 预留挂载行、T4-1 同期的 `#export-panel` 预留（两行代码，开工前在群里@对方确认即可，无阻塞）
- 验收：① 导出 CSV 与 Excel 各一例，内容与前端"违规记录"列表一致；② 日期/等级/PPE 三类筛选各命中 1 组用例；③ 空结果导出返回仅表头文件不报错；④ 中文内容无乱码（CSV 需 UTF-8 BOM，Excel 注意编码）；⑤ tests/test_export.py 通过
- 交付：代码 + 自测截图（Excel 打开效果）+ 功能说明一页

**T5-2 PPT 大纲与成稿**
- 文件：`docs/PPT_大纲.md` → PPT 工程文件（新建 `ppt/` 目录）
- 修改：结构建议——背景与意义 / 系统架构（复用交接报告第 2 节调用链图）/ 技术方案（双模型+匹配+ROI+规则引擎）/ 实验体系 E1~E4（复用 E2/E3/E4 报告曲线与指标表）/ 系统演示截图 / 分工与工程实践 / 总结展望
- 输入：docs/ 全部 12 份报告、outputs/ 截图与曲线、ui_dashboard.png、roi_demo.mp4（素材已齐备）
- 验收：≥25 页；实验数据页与 E 报告数值一致（成员1 抽查）；图示不使用截图模糊拉伸
- 交付：PPT 初稿（Phase 2）→ 终稿（Phase 4，含最终指标）

**T5-3 演讲稿与答辩准备**
- 文件：`docs/演讲稿.md`、`docs/答辩QA.md`
- 修改：8~10 分钟讲稿；预判 15 个答辩问题（含：为什么连体服不算 lab_coat、dataset2 为何永不进训练、误报怎么处理、AGPL License、与市面方案差异）
- 验收：讲稿时长实测达标；QA 覆盖技术决策依据（答案须引用 E1~E4 报告结论，不得臆造）
- 交付：两份文档 + 试讲一次

**T5-4 演示逻辑设计（与成员4 协作）**
- 文件：`docs/DEMO_SCRIPT.md`（与 T4-7 同一份文档，成员5 写脚本、成员4 按脚本录制）
- 修改：演示动线、时间轴、每步要讲的话术要点
- 交付：脚本终稿

---

## 4. 任务依赖关系图

```
Phase 0 ──────────────────────────────────────────────────────────
T1-1 git init ──► 阻塞一切（半天内完成）

Phase 1 ──────────────────────────────────────────────────────────
T1-2 锁依赖 ─┐
T1-3 接口冻结 ─┤
              ├─► 并行开工：
T3-5 config ──┤    T2-2 B3清洗 ┐
T3-1 api分层 ──┤    (T2-1 待B1) ─┴─► T2-3 E4训练（Phase 2）
T3-2 lifespan ─┤    T4-1 ECharts本地化（独立）
T3-3 db收敛 ──┤    T4-2 前端异常处理（独立）
T4-1#预留挂载点┤    T5-1 导出功能（依赖 T3-1 挂载行 + T4-1 容器预留，均为2行）
              ┘

Phase 2 ──────────────────────────────────────────────────────────
T2-3 E4训练 ──► T2-4 三模型重评 ──┐
T3-4 视频后台任务 ──► T4-5 进度条 UI │
T4-3 Dashboard ──► T4-4 ROI编辑    ├─► Phase 3
T5-1 完成 ──► T4-6 补测试          │
T5-2 PPT初稿（并行）               ┘

Phase 3 ──────────────────────────────────────────────────────────
T2-4 通过+用户批准 ──► T2-6 生产切换PR ──► T3-7 合入联调 ──► T1-4 集成验收
                                                      └─► T4-6 全量回归

Phase 4 ──────────────────────────────────────────────────────────
T4-6 回归报告 ─┐
T1-5 文档定稿 ──┤
T5-2 PPT终稿 ──┼──► T1-6 三套交付包 ──► T5-3 答辩 ──► T4-7 Demo成片（可提前录制素材）
T5-4 演示脚本 ─┘
```

**关键路径**：T1-1 → T2-3 → T2-4 → T2-6 → T3-7 → T4-6 → T1-6（AI 线驱动，且 T2-3 被 B1 数据到位时间 gating——**B1 未到位期间，成员2 优先做 T2-2 与 T2-5 骨架，其余成员不受影响**）。

---

## 5. Git 分支方案

### 5.1 分支结构

```
main        仅存放经过验收的稳定版本；只有成员1 可 push；打 tag（v0.1.0-baseline → v0.2.0-rc → v1.0.0）
develop     集成分支；成员1 从各 feature 分支 merge；每日同步
feature/member2-ai-e4       成员2：B1审计/E4训练/评估/切换方案
feature/member3-backend     成员3：api分层/lifespan/config/视频后台任务
feature/member4-frontend    成员4：ECharts/ROI/进度条/异常处理
feature/member5-export      成员5：导出功能三件套
release/v1.0.0              最终验收冻结分支（Phase 4）
hotfix/*                    仅 main 出现阻塞性 bug 时使用
```

### 5.2 规则

1. **每人只在 `feature/自己-*` 分支提交**；develop 只能由成员1 merge 进入（PR 制）。
2. feature 分支每 1~2 天 rebase/merge develop 一次，避免长期漂移。
3. **只改自己 owner 的文件**——PR 若触碰他人 owner 文件且超出 2.6 节允许的挂载行，直接打回。
4. **不允许 force push**；不允许直接覆盖他人修改（冲突找 owner 商量，谁的区域听谁的）。
5. merge 进 develop 前必须：本地自测通过 + PR 描述含验收证据（截图/测试输出）。
6. main 的每次 merge 由成员1 执行并附验收记录；release tag 附 zip 校验值。

### 5.3 .gitignore（Phase 0 由成员1 落盘，内容如下）

```gitignore
# ===== Python =====
venv/
__pycache__/
*.pyc
*.pyo
*.egg-info/
.pytest_cache/

# ===== 运行时数据（启动自动重建） =====
*.db
backend/labsafety.db
data/uploads/
outputs/
uploads/

# ===== 日志与临时 =====
pip*.log
*.log
.DS_Store
Thumbs.db

# ===== 训练数据（体积 1.6GB，走网盘/AI_TRAINING 包分发） =====
data/raw/
data/lab_ppe/

# ===== 训练产物 =====
runs/

# ===== 大模型权重（策略见下） =====
# 策略：sh17_yolov8s.pt(22.5MB) 为运行必需 → 例外放行；
#       预训练底座与训练中间产物 → 排除
yolov8s.pt
weights/
runs/**/*.pt
# 例外：生产模型入库（LFS 可选，GitLab/Gitee 私仓 22.5MB 可直接推）
!ai/model/sh17_yolov8s.pt
!ai/model/*.pt

# ===== 办公与工程临时 =====
ppt/~$*
*.tmp
~$*
```

**模型入库策略（单独决策项，建议）**：
- `ai/model/*.pt`（≤100MB）→ 入 git（或开 Git LFS），保证 PROJECT_SOURCE 克隆即跑；
- `yolov8s.pt`（根目录预训练底座）、`runs/**`（训练中间产物）、`weights/yolo26n.pt`（零引用残留）→ 不入库，走 AI_TRAINING 包；
- 若最终选用远端 git 服务单文件 <100MB 限制，则 `ai/model/` 改走 LFS。

---

## 6. Commit 规范

格式：`<type>(<scope>): <subject>`

| type | 用途 |
|------|------|
| feat | 新功能（feat(export): 支持按日期筛选导出 Excel） |
| fix | 缺陷修复（fix(video): 修复 stride 复用绘制错位） |
| refactor | 重构不改行为（refactor(api): main.py 路由拆分至 api/ 三模块） |
| test | 测试（test(export): 新增 CSV/Excel 导出用例） |
| docs | 文档（docs(model): 新增 MODEL_CARD） |
| chore | 构建/配置（chore(deps): 锁定 torch 2.6.0） |
| exp | AI 实验产物登记（exp(e4): E4 训练 results.csv 与曲线入档说明） |

附加规则：
- subject 用中文或英文均可，但必须说明"做了什么"，禁止 "update"、"fix bug" 这类空话。
- 一次 commit 一个意图；禁止把他人文件改动混进自己的 commit。
- 涉及验收标准的 commit，正文附验收证据（测试输出摘要/截图路径）。
- 实验大产物（.pt/.csv/图）不进 commit，只在 commit message 里登记路径。

---

## 7. PR 规则

1. **发起**：标题 = `[成员N] 任务编号 + 一句话`（例：`[成员5] T5-1 违规记录 CSV/Excel 导出功能`）。
2. **描述必填四项**：改了什么 / 为什么这样改 / 自测证据（截图或测试输出）/ 是否触碰他人 owner 文件（触碰了几行、依据 2.6 节哪条）。
3. **评审人 = 被 touch 文件的 owner**：
   - 只改自己文件 → 成员1 评审；
   - 触碰 main.py 挂载行 → 成员3 评审；
   - 触碰 index.html → 成员4 评审；
   - 触碰 ai/ → 成员2 评审。
4. **体量红线**：单 PR ≤400 行 diff（不含新增独立文件与测试）；`backend/main.py` 的他人 diff ≤5 行；`backend/static/index.html` 的他人 diff ≤2 行。超限拆分。
5. **行为等价类 PR**（如 T3-1 分层搬移）必须附"前后端点返回对照表"。
6. 评审 24h 未响应可 @升级到成员1 裁决。
7. **merge 责任**：develop 由成员1 merge；PR 作者不得自行 merge 自己的 PR。
8. 冲突处理：feature 分支冲突由该分支作者 rebase，冲突涉及他人区域时必须拉 owner 一起看。

---

## 8. 开发阶段计划（Phase 0~4）

> 排期单位为"工作日"，按课余投入估算；B1 数据到位日记为 AI 线 T0。

### Phase 0：Git 初始化 + 基线冻结（0.5 天，成员1 独占）

| 项 | 内容 |
|----|------|
| 负责人 | 成员1 |
| 输入 | 交接审计报告、当前可运行基线（14/14） |
| 任务 | T1-1（init/.gitignore/tag/远端）+ 分支创建 + 分工公告会（30 分钟，全员） |
| 输出 | main + tag v0.1.0-baseline + 4 条 feature 分支 + 仓库全员可访问 |
| 出口条件 | 5 人 clone 后本地跑通 MVP；.gitignore 生效（git status 不出现 data/runs/venv） |
| 风险 | 无 git 经验的成员：Phase 0 会上 15 分钟速训 |

### Phase 1：并行启动（约 1 周，四人同时开工）

| 成员 | 任务 | 依赖 |
|------|------|------|
| 成员1 | T1-2 锁依赖、T1-3 接口冻结、打包脚本骨架 | 无 |
| 成员2 | T2-2 B3 清洗（不等 B1）；T2-5 MODEL_CARD 骨架；B1 标注方案准备 | 无 |
| 成员3 | **T3-5 config（最先，消解交点①）** → T3-1 api 分层 → T3-2 lifespan → T3-3 db 收敛 | T1-1 |
| 成员4 | **T4-1 ECharts 本地化（最先，消除断网风险）** → T4-2 异常处理 → T4-3 Dashboard；预留 `#export-panel` | T1-1 |
| 成员5 | T5-1 导出功能三件套（桩路由先行）→ T5-2 PPT 大纲 | T3-1 挂载行（可先用桩） |

- **输入**：Phase 0 产物 + 接口契约 v1
- **输出**：develop 上合入 5~7 个 PR；导出功能可用；前端断网可跑
- **出口条件**：T3-1/T3-5/T4-1/T5-1 验收全过；14/14 回归不破

### Phase 2：核心推进（约 1~1.5 周，视 B1 到位时间）

| 成员 | 任务 | 依赖 |
|------|------|------|
| 成员1 | 联调协调、接口契约变更裁决、打包预演 | 被动响应 |
| 成员2 | T2-1 B1 审计 → T2-3 E4 训练 → T2-4 三模型重评 | **B1 数据到位（外部依赖！）** |
| 成员3 | T3-4 视频后台任务 + 进度 API | T1-3 契约 |
| 成员4 | T4-4 ROI 编辑、T4-5 进度条 UI、T4-6 补导出/异步测试 | T3-4 |
| 成员5 | T5-1 收尾打磨、T5-2 PPT 初稿、T5-4 演示脚本 | 素材已备 |

- **输出**：E4 模型与三模型对比报告；异步视频链路端到端可用；PPT 初稿
- **出口条件**：T2-4 达验收门槛（lab_coat F1≥0.50）；视频异步联调通过
- **降级预案**：B1 持续未到位 → 成员2 提交"B3-only 增训方案"供用户决策（此前 E4 方案报告中已有预案），其余成员进度不受影响

### Phase 3：模型定型 + 集成联调（约 3~4 天）

| 成员 | 任务 | 依赖 |
|------|------|------|
| 成员1 | T1-4 集成验收（主持） | T2-6/T3-7 |
| 成员2 | T2-5 MODEL_CARD 终稿、T2-6 生产切换 PR | T2-4 通过 + **用户批准** |
| 成员3 | T3-7 合入切换、冒烟；（可选）T3-6 摄像头 | T2-6 |
| 成员4 | T4-6 **全量回归**（T01~T14 + t01~t10 + dataset2 指标核对 + 新功能用例） | T2-6 |
| 成员5 | 演示脚本定稿、PPT 数据页更新 | T2-4 报告 |

- **输出**：生产系统运行 E4 定型模型；新版 test_report.md；切换前后指标对照
- **出口条件**：回归全绿 + dataset1 test lab_coat F1≥0.85 且其余类回退≤3pp（生产口径复核）

### Phase 4：收尾交付（约 1 周）

| 成员 | 任务 |
|------|------|
| 成员1 | T1-5 文档定稿、T1-6 三套交付包、release tag v1.0.0 |
| 成员4 | T4-7 Demo 视频录制剪辑（最终系统） |
| 成员5 | T5-2 PPT 终稿、T5-3 演讲稿 + 答辩 QA + 试讲 |
| 成员2 | 配合答辩技术 QA 素材（指标口径、License 说明） |
| 成员3 | 配合部署手册（离线安装路径、常见故障） |

- **输出**：v1.0.0 全套交付（代码包/AI包/模型包/文档/PPT/Demo）
- **出口条件**：交付清单逐项验收签字

---

## 9. 验收标准汇总（跨任务总口径）

| 类别 | 标准 | 验收人 |
|------|------|--------|
| 行为等价 | T3-1/T3-2/T3-5 重构后 13 端点返回与基线逐字段一致 | 成员4 协测 |
| 前端可用性 | 断网状态全 Tab 无白屏、图表正常、有错误提示 | 成员1 |
| 导出功能 | CSV/Excel 内容=前端列表；三类筛选各 1 用例；空结果不报错；中文无乱码 | 成员1+成员4 |
| 视频异步 | 长视频处理期间其他 API 可用；进度单调；结果与同步版一致 | 成员3+成员4 |
| AI 红线 | dataset2 全程零入训练（以 split 文件审计）；无 Coverall→lab_coat 映射 | 成员1（抽查） |
| 模型门槛 | dataset1 test lab_coat F1≥0.85；其余类回退≤3pp；dataset2 lab_coat F1≥0.50（≥0.70 优秀）；goggles≥0.327 | 成员1+用户 |
| 回归 | 全量测试 100% 通过，test_report.md 重新生成 | 成员4 |
| 工程规范 | 无 .db/outputs/venv/runs 入 git；PR 全部留痕；commit 信息合规 | 成员1 |
| 交付包 | SOURCE 包新环境跑通；FINAL_MODEL 含 MODEL_CARD+License；体积达标 | 成员1+全员 |

---

## 10. 最终交付物清单

| # | 交付物 | 责任人 | 载体 |
|---|--------|--------|------|
| 1 | 可运行系统源码 v1.0.0 | 全员（成员1 集成） | git tag + PROJECT_SOURCE.zip |
| 2 | E4 定型模型 + MODEL_CARD | 成员2 | FINAL_MODEL.zip |
| 3 | AI 训练可复现包（数据+脚本+报告） | 成员2 | AI_TRAINING.zip |
| 4 | 回归测试报告（终版） | 成员4 | tests/test_report.md |
| 5 | Demo 视频成片 | 成员4 | outputs/demo_final.mp4 |
| 6 | PPT 终稿 | 成员5 | ppt/ |
| 7 | 演讲稿 + 答辩 QA | 成员5 | docs/ |
| 8 | 部署手册 + 最终验收报告 + README 终版 | 成员1 | docs/ |
| 9 | 接口契约文档 | 成员1 | docs/INTERFACE_CONTRACT.md |
| 10 | 违规导出功能（含使用说明） | 成员5 | 代码 + 一页说明 |

---

## 11. 风险与协作注意事项（承接审计报告第 10 节）

1. **B1 数据是唯一外部依赖**：AI 线 Phase 2 被 B1 gating。负责人应在 Phase 0 会上明确 B1 采集/标注的最后期限（建议 Phase 1 结束时），逾期走 B3-only 降级预案。
2. **main.py 交点已消解但有时序要求**：T3-5（config）必须排在 T2-6（切换）之前；T3-1 的挂载行排在 T5-1 合入之前——两条已在 Phase 1/3 排期中显式排序。
3. **dataset2 铁律写进 PR 模板**：任何触及 data/ 的 PR 必须在描述里声明"未使用 dataset2"，评审人核对。
4. **License 纪律**：ECharts（Apache-2.0）本地化后保留版权注释；openpyxl（MIT）加入 requirements 前经成员1 批准；PPT 中 SH17/CC BY 4.0/AGPL-3.0 引用由成员5 落实、成员1 终审。
5. **Windows/中文路径**：成员5 的导出功能下载文件名含中文时注意 `Content-Disposition` 编码（RFC 5987）；新脚本读图一律沿用 `np.fromfile+cv2.imdecode` 模式。
6. **不建议做的事**（本阶段明确排除）：前端整体换框架（Vue 化）、rule_engine 重写、数据库换 PostgreSQL——均属"为优化而重构"，课程交付收益低、回归风险高。若有余力优先做 T3-6 摄像头实时（演示加分项）。
7. **每阶段结束由成员1 发"阶段验收纪要"**，含 PR 清单、遗留问题、下阶段入口条件——对应本项目既有"每阶段完成后停止等确认"的纪律。

---

## 附：任务→文件→负责人速查表

| 任务 | 新建文件 | 修改文件 | 负责人 | Phase |
|------|---------|---------|--------|-------|
| T1-1 git 基线 | .gitignore | — | 成员1 | 0 |
| T1-2 锁依赖 | requirements.lock | requirements.txt | 成员1 | 1 |
| T1-3 接口冻结 | docs/INTERFACE_CONTRACT.md | — | 成员1 | 1 |
| T1-4 集成联调 | docs/INTEGRATION_TEST_LOG.md | — | 成员1 | 3 |
| T1-5 文档定稿 | docs/部署手册.md 等 | README.md | 成员1 | 4 |
| T1-6 交付包 | scripts/package.py | — | 成员1 | 4 |
| T2-1 B1 审计 | tools/audit_b1.py | — | 成员2 | 2 |
| T2-2 B3 清洗 | tools/clean_b3.py | — | 成员2 | 1 |
| T2-3 E4 训练 | tools/make_e4_split.py, tools/train_e4.py | — | 成员2 | 2 |
| T2-4 三模型重评 | tools/eval_e4_compare.py | — | 成员2 | 2 |
| T2-5 MODEL_CARD | docs/MODEL_CARD.md | — | 成员2 | 2→3 |
| T2-6 生产切换 | ai/model/ 新权重 | ai/detector.py（owner 自改）；main.py ≤3行 PR | 成员2 | 3 |
| T3-1 api 分层 | backend/api/{detect,stats,settings}.py | backend/main.py | 成员3 | 1 |
| T3-2 lifespan | — | backend/main.py | 成员3 | 1 |
| T3-3 db 收敛 | — | backend/database.py, api/settings.py | 成员3 | 1 |
| T3-4 视频后台任务 | backend/jobs.py | video/processor.py, api/detect.py | 成员3 | 2 |
| T3-5 config | backend/config.py | backend/main.py:49 | 成员3 | 1 |
| T3-6 摄像头 | backend/api/camera.py | — | 成员3 | 3+（可选） |
| T3-7 切换合入 | — | backend/main.py | 成员3 | 3 |
| T4-1 ECharts 本地化 | static/vendor/echarts.min.js | static/index.html | 成员4 | 1 |
| T4-2 前端异常处理 | — | static/index.html | 成员4 | 1 |
| T4-3 Dashboard | — | static/index.html | 成员4 | 2 |
| T4-4 ROI 编辑 | static/roi_editor.js | static/index.html | 成员4 | 2 |
| T4-5 进度条 | — | static/index.html | 成员4 | 2 |
| T4-6 回归测试 | tests/test_export.py, tests/test_video_async.py | tests/run_tests.py | 成员4 | 1~3 |
| T4-7 Demo 视频 | docs/DEMO_SCRIPT.md | — | 成员4 | 4 |
| T5-1 导出功能 | backend/export.py, backend/api/export.py, static/export.js | （零现有文件修改） | 成员5 | 1 |
| T5-2 PPT | docs/PPT_大纲.md, ppt/ | — | 成员5 | 2→4 |
| T5-3 演讲答辩 | docs/演讲稿.md, docs/答辩QA.md | — | 成员5 | 4 |
| T5-4 演示脚本 | docs/DEMO_SCRIPT.md（与T4-7共文） | — | 成员5 | 2→4 |
