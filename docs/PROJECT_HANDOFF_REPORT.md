# PROJECT_HANDOFF_REPORT — 项目交接审计报告

> 审计日期：2026-09-06
> 审计方式：**只读扫描**（未修改任何生产代码、数据、模型；未训练、未下载、未删除任何文件）
> 审计范围：`D:/课程相关/大三上/软件工程实训/labsafety/` 全目录（venv/wheels 仅统计体积，未逐文件分析）
> 项目名称：基于计算机视觉的实验室安全智能巡检与违规预警系统
> 当前版本：MVP v0.1（生产系统）+ E1~E4 模型升级实验（进行中，E4 暂停待 B1 自采数据）

---

## 1. 项目目录树与分类

```
labsafety/
├── README.md                    [文档]      项目总说明：定位/运行方法/测试结果/已知问题 ✅提交
├── requirements.txt             [生产]      依赖清单（8项，未锁版本）            ✅提交
├── start.bat                    [生产]      Windows 一键启动 uvicorn             ✅提交
├── yolov8s.pt                   [训练]      22.6MB COCO 预训练权重，E2 增训底座    ❌不进源码包（AI包）
├── pip_install.log / pip_rf.log [临时]      pip 安装日志                          ❌排除
│
├── ai/                          [生产]      AI 检测层                             ✅提交
│   ├── detector.py              [生产]      PPEDetector 封装 + 17类→6业务类映射
│   ├── ppe_matcher.py           [生产]      Person-PPE 空间匹配
│   ├── roi.py                   [生产]      归一化矩形 ROI 区域判断
│   ├── model/sh17_yolov8s.pt    [生产模型]  22.5MB 当前生产系统唯一加载的权重     ✅提交（运行必需）
│   └── __pycache__/             [临时]      ❌排除
│
├── rule_engine/                 [生产]      ✅提交
│   └── engine.py                [生产]      SafetyRuleEngine 违规分级规则引擎
│
├── video/                       [生产]      ✅提交
│   └── processor.py             [生产]      图片/视频统一管线 InspectionPipeline
│
├── backend/                     [生产]      ✅提交
│   ├── main.py                  [生产]      FastAPI 应用（13个API端点，模型加载点）
│   ├── database.py              [生产]      SQLite 数据层（5张表，含建库种子）
│   ├── labsafety.db             [数据]      运行时生成的 SQLite 库（含演示数据）  ❌排除（启动自动重建）
│   ├── api/__init__.py          [空壳]      预留的分层 API 包，0字节未开发        ✅提交（占位）
│   └── static/index.html        [生产-前端] 335行单页应用（5个Tab + ECharts）
│
├── tests/                       [测试]      ✅提交
│   ├── run_tests.py             [测试]      T01~T14 系统测试 + 性能实测（14/14通过）
│   ├── evaluate_model.py        [测试]      模型 P/R/AP50 基线评估脚本
│   ├── model_check.py           [测试]      10张公开图定性检查
│   ├── acceptance_check.py      [测试]      MVP 验收自检
│   ├── make_test_video.py       [测试]      合成演示视频生成器
│   ├── roi_demo.py              [测试]      ROI 功能演示（生成 outputs/roi_demo.mp4）
│   └── test_report.md           [文档]      2026-09-05 实测报告
│
├── tools/                       [AI训练]    数据/训练/评估脚本（E1~E4 全部产出）  ✅提交
│   ├── download_datasets.py / download_dataset2_api.py / download_b3.py   数据下载
│   ├── convert_lab_ppe.py / dedup_check.py / dedup_b3.py / audit_labcoat.py  数据处理审计
│   ├── visualize_labels.py / visualize_b3.py                             标注可视化
│   ├── train_e2.py / eval_e2.py                                          E2 训练与评估
│   └── eval_e3_dataset2.py / eval_e3_t01t10.py                           E3 双模型独立测试
│
├── data/                        [数据]      共 1.6GB                             ❌不进源码包
│   ├── test/                    [测试数据]  t01~t10 公开测试图 + 合成演示视频     ✅随源码包提交（约几MB）
│   ├── uploads/                 [临时]      上传暂存（API 运行时写入）            ❌排除
│   ├── lab_ppe/                 [训练数据]  dataset1 转换后 YOLO 格式（7401图）   仅进 AI_TRAINING 包
│   └── raw/                     [原始数据]  dataset1(7401)/dataset2(251)/b3(477)  仅进 AI_TRAINING 包
│       └── dataset2/            ⚠️ 永久锁定 external_test，任何情况下禁止入训练
│
├── runs/                        [训练产物]  70MB                                  ❌不进源码包
│   ├── detect/val~val-6/        [历史]      早期验证运行（6组曲线/混淆矩阵）
│   └── e2/labppe_v8s/           [E2产物]    best.pt / last.pt / results.csv / 曲线 / 17000+训练batch图
│
├── outputs/                     [运行/实验输出] 41MB                              ❌排除（按需摘录）
│   ├── screenshots/ annotated/ videos/   [生产运行输出] 违规截图/标注图/标注视频
│   ├── e2_train.log / e3_*.json|txt / b3_*  [实验日志与预测结果]
│   └── roi_demo.mp4 / ui_dashboard.png / label_check/  [演示与抽检产物]
│
├── weights/yolo26n.pt           [历史残留]  5.5MB，全项目无任何代码引用（疑为
│                                            ultralytics AMP 检查或手动实验下载）   ❌排除
│
├── docs/                        [文档]      12 份报告，全部 ✅提交
│   MVP验收报告 / 系统架构与模块说明 / API说明 / 数据来源与开源项目说明 /
│   模型升级分析与实施方案 / 错误分析与真实场景验证方案 / 自采数据集与模型评估方案 /
│   E1_data_report / E2_train_report / E3_eval_report / E4_lab_coat_plan / E4_data_prep_report
│
├── venv/                        [环境]      Python 3.13 虚拟环境（torch 2.6.0+cu124 等，数GB）❌排除
└── wheels/                      [环境]      2.4GB torch/torchvision cu124 离线whl   ❌排除
```

**结论**：生产代码 = `ai/ + rule_engine/ + video/ + backend/`（不含 labsafety.db）；测试 = `tests/ + data/test/`；AI 训练 = `tools/ + runs/ + data/lab_ppe + data/raw + 根目录 yolov8s.pt`；文档 = `docs/ + README`；临时/排除 = `__pycache__、pip 日志、outputs/、uploads/、labsafety.db、weights/yolo26n.pt、venv/、wheels/`。

---

## 2. 当前系统真实架构与调用链（全部经代码核实）

```
浏览器 backend/static/index.html (单页, 5 Tab)
   │ fetch
   ▼
FastAPI  backend/main.py  (uvicorn, start.bat → 127.0.0.1:8000)
   │
   ├─ POST /api/detect/image ──► video/processor.process_image(pipeline, img, areas)
   ├─ POST /api/detect/video ──► video/processor.process_video(pipeline, vid, areas, stride)
   │        两个入口都使用 get_pipeline() 懒加载的同一个 InspectionPipeline
   ▼
InspectionPipeline.analyze_frame(frame, areas)      [video/processor.py:97]
   │ ① PPEDetector.detect(frame)                    [ai/detector.py:132] ← 模型在此加载推理
   │      └─ YOLO(ai/model/sh17_yolov8s.pt)  ← 生产权重，main.py:49 硬编码传入
   │      └─ CLASS_ALIAS: SH17 17类 → person/mask/gloves/lab_coat/goggles/helmet
   │ ② match_person_ppe(detections)                 [ai/ppe_matcher.py:50] 中心点归属+最近距离兜底
   │ ③ annotate_person_states(states, areas, w, h)  [rule_engine/engine.py:107]
   │      └─ locate_person()                        [ai/roi.py:40] 归一化矩形命中
   │ ④ SafetyRuleEngine.evaluate(states)            [rule_engine/engine.py:72]
   │      └─ 缺0种=正常 / 缺1种=一般违规 / 缺≥2种=严重违规；非ROI区域不评估
   ▼
ViolationEvent（含 severity/missing_ppe/frame_time/screenshot）
   │ draw_annotations() 画 ROI/人框/PPE框/违规文字 [video/processor.py:37]
   │ 违规截图（视频模式带 10 秒冷却去重）→ outputs/screenshots/
   ▼
backend/database.py  SQLite (labsafety.db, 5张表)
   │  detection_records / violation_events 落库
   ▼
GET /api/dashboard · /api/statistics · /api/violations · /api/settings  →  前端 ECharts 呈现
```

### 十个关键位置问答

| # | 问题 | 答案（文件:行号） |
|---|------|------------------|
| 1 | 程序入口 | `start.bat` → `venv\Scripts\python.exe -m uvicorn backend.main:app --port 8000` |
| 2 | 后端入口 | `backend/main.py`（`app = FastAPI(...)` L37；`if __name__ == "__main__"` L244） |
| 3 | AI 模型加载 | `backend/main.py:49` `PPEDetector(ROOT/"ai"/"model"/"sh17_yolov8s.pt")`，懒加载于 `get_pipeline()`，首次请求触发；实际 YOLO 实例化在 `ai/detector.py:114` |
| 4 | 图片检测进入 | `backend/main.py:108` `POST /api/detect/image` → `video/processor.py:109` `process_image()` |
| 5 | 视频检测进入 | `backend/main.py:153` `POST /api/detect/video` → `video/processor.py:141` `process_video()`（stride 抽帧） |
| 6 | PPE 匹配 | `ai/ppe_matcher.py:50` `match_person_ppe()`，由 `InspectionPipeline.analyze_frame` 第②步调用 |
| 7 | ROI 判断 | `ai/roi.py:40` `locate_person()`，经 `rule_engine/engine.py:107` `annotate_person_states()` 回填 `in_roi/area_name/area_required` |
| 8 | 规则判断 | `rule_engine/engine.py:72` `SafetyRuleEngine.evaluate()` |
| 9 | 数据库 | `backend/database.py`（schema L19-67；写入 L124/L139/L157；查询统计 L172-247；启动时 `init_db(seed=True)` main.py:58） |
| 10 | 前端入口 | `backend/static/index.html`（main.py:68 `GET /` 返回；静态挂载 `/static`、`/media`） |

**重要架构事实**：生产系统当前**只加载 SH17 单模型**；E2 训练的 LAB_PPE 模型（best.pt）与"SH17+LAB_PPE 双模型方案"**仅存在于 tools/ 评估脚本中，尚未接入 backend/main.py**。切换模型需改 main.py:49 一处硬编码。

---

## 3. 模型文件审计（全项目 5 个 .pt，无 .pth/.onnx/.engine）

| 文件 | 大小 | 用途 | 生产使用？ | 判定 |
|------|------|------|-----------|------|
| `ai/model/sh17_yolov8s.pt` | 22.5MB | SH17 官方 YOLOv8s，17类 PPE 检测 | **是（唯一生产模型）** | 保留，随源码包分发 |
| `runs/e2/labppe_v8s/weights/best.pt` | 22.5MB | E2 自训练 LAB_PPE 4类模型（valid mAP50=0.924） | 否（仅 tools/eval_e2、eval_e3_* 引用） | 保留，进 FINAL_MODEL 包与 AI 包 |
| `runs/e2/labppe_v8s/weights/last.pt` | 22.5MB | E2 断点续训产物（best=epoch49，已封板） | 否 | 训练档案，AI 包可选 |
| `yolov8s.pt`（根目录） | 22.6MB | COCO 预训练权重，E2/E4 增训底座（tools/train_e2.py:21） | 否 | 仅进 AI_TRAINING 包 |
| `weights/yolo26n.pt` | 5.5MB | **全项目零引用**（无任何代码/脚本/文档引用） | 否 | 历史残留，不打包（不删除） |

### 模型路径引用全景（项目代码，排除 venv）

| 位置 | 行号 | 引用模型 | 性质 |
|------|------|---------|------|
| `backend/main.py` | 49 | `ai/model/sh17_yolov8s.pt` | **生产加载点（硬编码）** |
| `tests/run_tests.py` | 55, 275 | 同上（GPU+CPU 两次实例化） | 测试 |
| `tests/model_check.py` | 21 | 同上 | 测试 |
| `tests/acceptance_check.py` | 22 | 同上 | 验收 |
| `tests/evaluate_model.py` | 121 | 同上（`--model` 默认参数） | 评估 |
| `tools/eval_e2.py` | 8 | `runs/e2/labppe_v8s/weights/best.pt` | E2 评估 |
| `tools/eval_e3_dataset2.py` | 34-35 | SH17 + LAB_PPE best.pt（双模型） | E3 评估 |
| `tools/eval_e3_t01t10.py` | 28-29 | 同上 | E3 评估 |
| `tools/train_e2.py` | 21 | `yolov8s.pt`（自动下载底座） | 训练 |

**硬编码评估**：所有引用均为基于 `ROOT` 的相对路径拼接，无绝对盘符路径，可移植性良好；但生产模型选择硬编码在 main.py:49，无配置项/环境变量开关——模型升级切换时这是**唯一需要动的生产代码行**，建议后续由 AI 成员与后端负责人共同评审后再改。

---

## 4. 代码模块逐个审计

### 4.1 ai/detector.py（162行）
- **功能**：PPEDetector 封装 ultralytics YOLO；Detection 统一数据结构；CLASS_ALIAS 将模型原始类别映射到 6 个业务类（person/mask/gloves/lab_coat/goggles/helmet），非业务类返回 None 丢弃。
- **输入**：BGR ndarray 单帧 + conf 阈值（默认 0.35）｜**输出**：`list[Detection]`（xyxy 像素坐标）。
- **被调用**：video/processor.py、tests/run_tests.py、model_check.py、acceptance_check.py、evaluate_model.py、tools/eval_e3_*。**调用了**：ultralytics.YOLO、torch（设备探测）。
- **状态**：稳定，MVP 验收 14/14 通过。
- **已知问题**：① `map_class` 兜底子串匹配把含 "suit/coat" 的类别一律映射 lab_coat——与 E4 决策"连体防护服（Coverall/medical-suit）不视为实验服"存在口径冲突，切换双模型时必须同步修订；② CLASS_ALIAS 未包含 LAB_PPE 模型的原生类别键（E3 已发现），生产切换前需补充；③ conf 阈值为构造参数，生产未暴露配置。
- **建议负责**：成员2（AI算法）。

### 4.2 ai/ppe_matcher.py（83行）
- **功能**：Person-PPE 空间匹配。规则1：PPE 中心点落在 person 框内归属之（多人重叠取更小框）；规则2：兜底最近距离 ≤ 0.5×person框对角线；同类 PPE 保留高置信度。
- **输入**：`list[Detection]`｜**输出**：`list[PersonState]`（含 ppe dict、in_roi、area_name）。
- **被调用**：video/processor（管线第②步）、tests/run_tests.py T08。**调用了**：ai.detector。
- **状态**：稳定、可解释（README 明示为设计取舍）。
- **已知问题**：远处 PPE（如桌上/他人手持）可能被距离兜底错误归属 → 误报；无遮挡/背对处理（已知局限）。
- **建议负责**：成员2（AI算法）。

### 4.3 ai/roi.py（51行）
- **功能**：AreaROI 归一化矩形（0~1）；`locate_person` 判断 person 框中心点落入哪个区域，第一个命中即返回。
- **输入**：person Detection + 帧尺寸 + areas｜**输出**：(in_roi, AreaROI|None)。
- **被调用**：rule_engine.annotate_person_states、database.load_areas（构造 AreaROI）、tests/roi_demo.py。
- **状态**：稳定。**已知问题**：仅支持矩形、仅中心点判定（人跨区/框大部分在区内但中心在外会漏判）；多区域无优先级配置。
- **建议负责**：成员2（AI算法），第二阶段扩展多边形 ROI 时再议。

### 4.4 rule_engine/engine.py（118行）
- **功能**：SafetyRuleEngine 按"区域要求 PPE 列表（数据库可配置）− 实际佩戴"求缺失集；缺0=正常、缺1=一般违规、缺≥2=严重违规（阈值可配）；非 ROI 人员不评估；ViolationEvent 事件结构。
- **输入**：`list[PersonState]`（需已回填 in_roi/area_required）｜**输出**：`list[ViolationEvent]`。
- **被调用**：video/processor（管线第④步）、backend/main.py（重建事件对象入库）、tests。**调用了**：ai.detector.CLASS_CN、ai.roi。
- **状态**：稳定，业务规则错误经 E3 分析为 0。
- **已知问题**：`evaluate` 用 `hasattr(st, "area_required")` 动态属性做防御（PersonState 未声明该字段，属隐式契约，易踩坑）；违规等级仅按缺失数量，无分区域/分 PPE 危险等级。
- **建议负责**：成员2（AI算法）维护；业务规则扩展（如分区分级）可交成员5 立项新文件，不直接改本文件。

### 4.5 video/processor.py（216行）
- **功能**：核心管线。`InspectionPipeline.analyze_frame`（检测→匹配→ROI→规则）；`process_image`（读图+标注+截图+返回 dict）；`process_video`（抽帧 stride 复用结果、10秒冷却去重截图、写标注 mp4、统计 FPS）；`draw_annotations` 统一绘制。
- **输入**：pipeline + 图片/视频路径 + areas｜**输出**：结果 dict（person_states/events/detections/耗时/标注文件路径）。
- **被调用**：backend/main.py 两个检测端点。**调用了**：ai 全部三模块 + rule_engine + cv2。
- **状态**：稳定；性能实测图片 48.8ms、视频 89.3 FPS（RTX 4060）。
- **已知问题**：① 视频接口同步阻塞（README 已知问题4）；② stride 间帧复用结果导致绘制错位风险（已用 last_result 缓存缓解但变量未实际参与绘制，见 L165 注释与实现差异）；③ 违规冷却按"(person_id, 缺失组合)"全局去重，跨帧 person_id 变化时可能漏记（无 tracking）；④ `cv2.imread` 直读中文路径会失败——当前上传文件已重命名为 ASCII 安全名规避，但新增功能直接传中文路径会炸（tools/ 下已有 `np.fromfile+imdecode` 解法可借鉴）。
- **建议负责**：成员3（后端+视频）。

### 4.6 backend（main.py 247行 + database.py 248行 + api/ 空壳）
- **main.py**：FastAPI 应用，13 个端点（页面1 + 模型信息1 + 统计2 + 记录1 + 检测2 + 设置4 + 挂载2）；懒加载管线；startup 建库。**输入**：HTTP 请求/上传文件｜**输出**：JSON/静态文件。
- **database.py**：SQLite 5 张表（laboratories/areas/safety_rules/detection_records/violation_events），建库+种子数据、区域与规则加载、记录与事件写入、dashboard/statistics 聚合查询。纯 sqlite3，无 ORM。
- **api/**：`__init__.py` 0 字节——**预留分层结构，未开发**。
- **被调用**：浏览器前端（fetch）。**调用了**：video.processor、ai.detector、rule_engine、database。
- **状态**：MVP 可运行、闭环完整。**已知问题**：① 模型路径硬编码（见第3节）；② 视频检测同步阻塞、无进度查询；③ 全部业务逻辑在单文件 main.py，无路由分层（api/ 空壳即为此预留）；④ `@app.on_event("startup")` 是 FastAPI 已弃用写法；⑤ 设置接口直接裸 SQL，与 database.py 职责重复。
- **建议负责**：成员3（后端+视频）。

### 4.7 前端 backend/static/index.html（335行单页）
- **功能**：原生 JS 单页应用，5 个 Tab（Dashboard/智能巡检/违规记录/数据统计/系统设置），ECharts 5.5.0（**CDN 引入**）渲染图表，调用后端全部 API。
- **输入**：后端 API JSON｜**输出**：Web UI。
- **状态**：MVP 可运行，ui_dashboard.png 有界面截图佐证。**已知问题**：① ECharts 走 jsdelivr CDN——**断网演示即白板图表，Demo 大风险**；② 单文件 335 行已接近可维护上限，无组件化；③ 无 ROI 可视化拖拽绘制（设置页手填坐标数字）。
- **建议负责**：成员4（前端+测试+Demo）。

### 4.8 tests（6 个脚本 + 1 报告）
- **run_tests.py**：T01~T14 系统测试（检测类依赖真实模型输出，如实记录）+ GPU/CPU 性能实测，自动生成 test_report.md。**2026-09-05 实测 14/14 通过**。
- **evaluate_model.py**：11点插值 AP50 独立评估器（E3 脚本的前身基座）。
- **model_check.py**：10 张公开图定性检查；**acceptance_check.py**：MVP 验收自检；**make_test_video.py**：合成演示视频；**roi_demo.py**：ROI 演示。
- **状态**：功能完整、有真实报告。**已知问题**：非 pytest 框架（自研 record/RESULTS 机制），无 CI、无回归触发；T01-T08 为构造用例，真实场景覆盖靠 E3 的 t01-t10（不在 tests/ 内）。
- **建议负责**：成员4（前端+测试+Demo）。

### 4.9 tools/（13 个脚本，E1~E4 实验线）
- 全部为**离线实验工具，不属于生产链路**，但承载了模型升级的全部可复现工作（下载/转换/去重/审计/训练/评估/可视化）。与生产代码零耦合（只 import ai/rule_engine，不 import backend）。
- **建议负责**：成员2（AI算法）。

---

## 5. 项目负责人已完成的工作（均有实证依据）

| 领域 | 完成内容 | 依据 |
|------|---------|------|
| 系统架构 | 四层架构（AI检测→匹配/ROI→规则引擎→Web）+ SQLite + FastAPI 闭环 | docs/系统架构与模块说明.md；代码实测调用链（本报告第2节） |
| 后端 | FastAPI 13 端点：检测×2、统计×2、记录、设置×4、页面、模型信息、双静态挂载 | backend/main.py（37-241行） |
| AI 检测 | PPEDetector 封装 + SH17 17类→6业务类映射 + 统一 Detection 结构 | ai/detector.py；tests/model_check.py 输出 |
| PPE 匹配 | 中心点归属 + 最近距离兜底 + 高置信度保留，多人场景验证 | ai/ppe_matcher.py；T08 用例通过 |
| ROI | 归一化多矩形区域、可配置、演示视频 | ai/roi.py；outputs/roi_demo.mp4；tests/roi_demo.py |
| 规则引擎 | 区域级可配置 PPE 要求 + 三级违规分级 + 事件结构 | rule_engine/engine.py；T01-T07 通过 |
| 数据库 | SQLite 5 张表 + 种子配置 + dashboard/statistics 聚合 | backend/database.py；T12/T13/T14 通过 |
| 前端 | 5 Tab 单页应用（Dashboard/巡检/记录/统计/设置）+ ECharts | backend/static/index.html；outputs/ui_dashboard.png |
| 图片检测 | 上传→检测→匹配→ROI→规则→标注图+人员明细+入库 | main.py:108-149；T09（48.8ms）通过 |
| 视频检测 | 抽帧 stride、标注视频输出、10s 冷却截图去重、FPS 统计 | video/processor.py:141-216；T10（89.3FPS）通过 |
| 测试 | T01~T14 用例框架 + 性能实测 + 测试报告自动生成 | tests/run_tests.py；tests/test_report.md（14/14，2026-09-05） |
| 性能测试 | 图片延迟、视频 FPS（GPU 89.3 / CPU 12.9）、推理耗时 | test_report.md；run_tests.py:275 CPU 实测分支 |
| MVP 验收 | 验收自检 + 验收报告 | tests/acceptance_check.py；docs/MVP验收报告.md |
| 数据集处理 | dataset1 7401张 13类→4类转换；dataset2 251张 API 下载；去重零泄漏 | data/lab_ppe/；docs/E1_data_report.md |
| 模型实验 | E2 训练（YOLOv8s，valid mAP50=0.924）；E3 双模型独立测试；E4 lab_coat 域差异审计与 B3 数据准备（476张下载+去重审计） | runs/e2/；docs/E2~E4 报告；outputs/e3_dataset2_preds.json |

---

## 6. 未完成工作清单（按 A~H 分类）

| 类别 | 当前状态 | 已完成 | 未完成 | 建议负责 | 依赖 |
|------|---------|--------|--------|---------|------|
| A. AI模型 | **E4 增训暂停**，待 B1 自采 150 张 | E2 模型训练、E3 双模型评估、E4 方案设计+B3 数据(≈450张可用) | ① B1 数据采集标注；② E4 lab_coat 定向增训（best.pt 续训 30ep）；③ 重跑 E3 三模型对比；④ **生产切换双模型**（改 main.py:49 + detector 类别映射）；⑤ dataset2 external_test 结果达标验证（lab_coat F1≥0.50/0.70） | 成员2 | 依赖用户提供 B1 数据；切换生产依赖成员3 评审 |
| B. 后端 | MVP 可运行 | 13 端点、数据库、懒加载管线 | ① api/ 分层路由（空壳已预留）；② 视频后台任务+进度查询；③ 摄像头实时巡检端点（`VideoCapture(0)`）；④ 启动事件写法升级（lifespan）；⑤ 设置接口 SQL 收敛到 database.py | 成员3 | ②③ 依赖 A 完成模型定型后再做性能调优更划算 |
| C. 前端 | MVP 可运行 | 5 Tab 单页 + ECharts | ① **ECharts 本地化（去 CDN，Demo 断网风险）**；② ROI 拖拽绘制；③ 视频处理进度条（依赖 B-②）；④ 可选：Vue3 + Element Plus 升级 | 成员4 | ③ 依赖 B-② |
| D. 测试 | 14/14 通过 | T01~T14 + 性能实测 | ① E4 后回归重跑（模型切换后 t01-t10 + dataset2 + dataset1 全量）；② 双模型/新功能用例补充；③（可选）pytest 化 + CI | 成员4 | 依赖 A、B 完成度 |
| E. 项目集成 | 单机手动 start.bat | — | ① **建 git 仓库 + .gitignore（当前完全没有版本控制！）**；② requirements 锁版本；③ 一键环境脚本（README 已有步骤，需固化）；④ 模型路径配置化；⑤ 最终联调 + 验收 | 成员1（负责人） | 阻塞所有并行开发，**最高优先级** |
| F. 文档 | 12 份已交付 | 架构/API/数据来源/验收/E1~E4 | ① README 更新（双模型说明、启动方式变更）；② 最终交付文档（部署手册）；③ 接口文档与实现同步核对 | 成员1 | 随 A/B/C 定稿更新 |
| G. PPT | 未开始 | — | ① 项目背景/架构/创新点/实验数据/演示流程成稿；② 素材（已有大量可用：架构图、E2/E3 曲线、ui_dashboard.png、roi_demo.mp4、test_report.md） | 成员5 | 素材已备，随时可启动；终稿依赖 A/B 定稿 |
| H. Demo 视频 | 部分素材 | roi_demo.mp4、合成演示视频、标注视频样例 | ① 完整端到端演示录屏（真实实验服场景优先）；② 若 B1 到位用真实实验室素材 | 成员4（录制）+ 成员5（脚本） | 建议在模型定型后录制 |

---

## 7. 五人分工建议（按现有代码切分，无重叠核心文件）

| 成员 | 角色 | 独占文件/模块 | 可验收产出 | 依赖 |
|------|------|--------------|-----------|------|
| 成员1（负责人） | 集成与发布 | `README.md`、`requirements.txt`、`start.bat`、`docs/`（终稿）、打包脚本 | git 仓库+分支策略+.gitignore、requirements 锁定、集成联调记录、PROJECT_SOURCE/FINAL_MODEL 打包、部署手册 | 被依赖方：所有人 |
| 成员2（AI算法） | 模型与算法 | `ai/`（detector/ppe_matcher/roi）、`tools/` 全部、`runs/`、训练数据 | E4 增训+三模型对比报告、生产模型切换方案（main.py:49 修改以 PR 交成员3 合入）、CLASS_ALIAS 修订、模型卡 | B1 数据（用户提供）；与成员3 在"模型路径配置化"接口上协作 |
| 成员3（后端+视频） | 服务端 | `backend/main.py`、`backend/database.py`、`video/processor.py` | api/ 分层路由改造、视频后台任务+进度 API、摄像头实时端点、lifespan 升级 | 模型定型（成员2）后再做性能版；合入成员5 的路由 |
| 成员4（前端+测试+Demo） | 前端与质量 | `backend/static/index.html`、`tests/` 全部 | ECharts 本地化、ROI 可视化编辑、进度条 UI、E4 后全量回归测试报告、Demo 录屏 | 进度条依赖成员3 的 API；回归依赖成员2 定型 |
| 成员5（PPT+演讲+功能开发） | 呈现与外围功能 | `backend/api/`（新路由文件）、**新建** `backend/export.py`（报表导出）、**新建** `docs/PPT_大纲.md` | PPT 成稿+演讲稿、违规记录 Excel/PDF 导出功能（独立新文件）、api/ 路由骨架 | 路由挂载需成员3 在 main.py 加一行 `include_router`（唯一交点，约定接口即可） |

**防冲突规则**：
1. 每个核心文件只有一个 owner；跨模块改动一律走 PR/补丁由 owner 合入（因此**先建 git 仓库是第 0 步**）。
2. `backend/main.py` 归成员3 独占——成员2 的模型切换、成员5 的路由挂载都以 ≤5 行 diff 的 PR 提交，成员3 审核合入。
3. `video/processor.py` 归成员3；成员2 若需调整管线内 AI 调用（如双模型融合），新建 `ai/pipeline_dual.py` 类文件，不改 processor.py 原逻辑，由成员3 选择接入点。
4. `rule_engine/engine.py` 归成员2；业务规则扩展（分级策略等）由成员5 新建 `rule_engine/extension.py`，engine.py 仅暴露必要接口。
5. `data/raw/dataset2/` 任何人**只读**（external_test 锁定，铁律）。

---

## 8. 模块依赖关系图

```
                    成员4            成员3                成员1
              backend/static ──► backend/main.py ──► start.bat / requirements
                     │  fetch         │       │
                     ▼                ▼       ▼
              （HTTP API）      video/processor.py   backend/database.py ──► labsafety.db
                                      │  InspectionPipeline        ▲
                          ┌───────────┼────────────┐               │ load_areas()
                          ▼           ▼            ▼               │
                    ai/detector  ai/ppe_matcher  rule_engine/engine.py
                          │           │               │  annotate_person_states()
                          ▼           │               ▼
                  ai/model/*.pt       │           ai/roi.py ◄──────┘
                  （生产权重）        │
                                      │  成员2 独占 ai/ + tools/
              tools/* ──► runs/e2/best.pt ──（E4 定型后经 PR 切换）──► ai/model/
                     └──► data/lab_ppe + data/raw（dataset2 只读！）
```

纵向依赖：前端 → API → 管线 → AI/规则/ROI → 模型/数据库。**改动上游（detector 接口、Detection 结构）会波及全部下游**，因此 Detection dataclass 与 `analyze_frame` 签名在第二阶段冻结，如需变更由成员1 主持评审。

---

## 9. 打包建议（三个包）

### 9.1 PROJECT_SOURCE.zip（发全体组员，目标 <100MB）
**包含**：
- `ai/`（含 `ai/model/sh17_yolov8s.pt`——运行必需）、`rule_engine/`、`video/`、`backend/`（**排除 labsafety.db 与 __pycache__**，api/ 空壳保留）
- `backend/static/index.html`、`tests/` 全部、`tools/` 全部（脚本很小，便于复现）
- `docs/` 全部、`README.md`、`requirements.txt`、`start.bat`
- `data/test/`（t01~t10 测试图 + demo_lab_video.mp4，仅几 MB）

**排除**：venv/、wheels/（2.4GB whl！）、data/lab_ppe、data/raw、data/test/uploads、runs/、outputs/、pip 日志、全部 `__pycache__`、labsafety.db、weights/yolo26n.pt、根目录 yolov8s.pt。

### 9.2 AI_TRAINING.zip（只发成员2，目标约 1.5~2GB）
在 9.1 基础上**追加**：
- `data/lab_ppe/`（转换后 YOLO 格式训练集，7401 图）
- `data/raw/b3/`（476 图 + 审计清单）、`data/raw/dataset2/`（**附只读警告 README**，仅用于 external_test 评估）
- `data/raw/dataset1/` 体积大，建议**另出 DATA_RAW.zip 单独传**或给 Roboflow 下载脚本 + API key 说明（tools/download_datasets.py 已可复现）
- 根目录 `yolov8s.pt`（预训练底座）
- `runs/e2/labppe_v8s/`：**精简版**——best.pt、last.pt、args.yaml、results.csv、曲线/混淆矩阵 PNG；**剔除 17000+ 张 train_batch*.jpg**（占大头、无复用价值）
- docs/E1~E4 报告（复现依据）

**排除**：同 9.1 的环境与临时项；wheels/（成员2 已有 venv，whl 按需再传）。

### 9.3 FINAL_MODEL.zip（最终模型交付包，目标 <80MB）
- `ai/model/sh17_yolov8s.pt`（person 检测用，注明来源与 License）
- `runs/e2/labppe_v8s/weights/best.pt` → 重命名为 `labppe_v8s.pt`（4 类 PPE 模型）
- （若 E4 完成增训）增训后 best.pt → `labppe_v8s_e4.pt`
- `MODEL_CARD.md`（新建）：每个模型的类别、训练数据、指标（E2: valid mAP50=0.924 / dataset2 外测 F1 见 E3 报告）、加载方式、已知局限（lab_coat 域偏移、goggles 跨域回退）
- `docs/E2_train_report.md + E3_eval_report.md`（指标佐证）
- License 说明：SH17 权重来源、Ultralytics AGPL-3.0、数据集 CC BY 4.0

**排除**：一切数据、代码、runs 曲线图以外的大文件、last.pt（非必要不交付）。

---

## 10. 风险与注意事项

1. **无版本控制（最高风险）**：项目根目录**没有 .git**。五人并行开发前必须先 `git init` + `.gitignore`（至少排除 venv/、wheels/、data/raw/、data/lab_ppe/、runs/、outputs/、__pycache__/、*.db、pip*.log、*.pt 大文件视策略），否则协作即灾难。
2. **生产模型与实验模型脱节**：backend 实际用的是 SH17（lab_coat 经 medical-suit 近似映射，E3 实测 dataset2 lab_coat F1 仅 34%），E2/E4 更优模型未接入。切换动作很小（main.py:49），但必须先完成 E4 验收门槛（dataset1 test lab_coat F1≥0.85、其余类回退≤3pp、dataset2 lab_coat F1≥0.50）再切。
3. **dataset2 铁律**：`data/raw/dataset2/` 永久 external_test，禁止入 train/valid、禁止据此调参。已写入协作规则，打包时在包内放只读警告。
4. **前端 CDN 依赖**：index.html 的 ECharts 来自 jsdelivr，**答辩现场断网图表全挂**——成员4 优先本地化（下载 echarts.min.js 入 static/）。
5. **License 合规**：Ultralytics AGPL-3.0（课程演示可接受，商用需评估）；SH17/CPPE-5/Roboflow 数据各自 License 已在 docs/数据来源与开源项目说明.md 核实，PPT 与交付文档需引用。
6. **Windows/中文路径坑**：生产代码 `cv2.imread/imwrite` 遇中文路径失败（当前上传重命名规避）；E4 增训后若引入中文文件名 B1 数据，训练/评估脚本须沿用 tools/ 的 `np.fromfile+cv2.imdecode` 方案。
7. **同步阻塞视频接口**：长视频上传期间 API 无响应，Demo 时建议控制视频时长或提前完成 B-② 后台任务。
8. **历史残留**：`weights/yolo26n.pt`（零引用）与 `runs/detect/val~val-6`（早期验证）不影响运行，建议打包排除但**不删除**（审计约束）。
9. **requirements 未锁版本**：多人环境不一致隐患，成员1 负责生成 `requirements.lock`（当前实测环境：torch 2.6.0+cu124、Python 3.13）。
10. **E4 处于暂停态**：B1 自采 150 张未到位前，AI 线所有排期以"B1 到位日"为 T0 重新计算；B3 已就绪（≈450 张可用，待确认剔除 20 张水印图与 5 张近重复图）。

---

## 附：审计方法说明

本报告全部结论来自：目录树扫描（find/du）、9 个核心源文件逐行阅读（ai×3、rule_engine、video、backend×2、前端头部）、全项目模型文件定位（5 个 .pt）、项目代码内 `.pt` 引用逐条 grep（排除 venv 噪声）、tests/test_report.md 与 README 交叉核对。未运行任何模型、未修改任何文件（本报告自身除外）。
