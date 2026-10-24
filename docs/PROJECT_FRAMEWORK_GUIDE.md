# 实验室安全智能巡检系统——现有代码框架与开发说明

> 适用对象：第一次进入仓库的组员（Python / Git 经验一般的大学生）
> 文档目标：看完这份文档后，你能回答——
> - 项目目前已经搭建完成了什么？
> - 从图片/视频输入到网页显示，整条链是怎么跑起来的？
> - 每个目录和文件分别负责什么？
> - 模块之间怎么调用、怎么传数据？
> - 五人分别应该在哪些文件里写？
> - 哪些可以自己改，哪些是公共接口、必须先沟通？
> - 我要新增一个功能，应该先看哪些文件？

---

## 〇、用一句话回答"项目现在已经能做什么"

**输入一张 JPG/PNG 图片，或一段 MP4 视频 → 系统自动用 YOLO 检测出人和 PPE → 把 PPE 匹配给具体的人 → 判断这个人是否在受控区域内 → 对照该区域要求的 PPE → 缺失就记一次违规 → 截图保存到 outputs/screenshots → 数据写入 SQLite → 前端 Dashboard 显示统计、违规记录可查询。**

整个过程已经跑通（基线 14/14 测试通过），但仍有部分能力属于"第一版 MVP 范围"而非"最终交付"——区分两者是阅读本文档的关键。

---

## 一、项目当前"已实现"和"暂未实现"的区分

### 1.1 当前已经实现（均可在仓库代码中找到依据）

| 能力 | 实现位置 | 验证方式 |
|---|---|---|
| YOLO PPE 检测（SH17 权重） | `ai/detector.py` | `tests/run_tests.py` T01~T03 |
| 17 个原始类别 → 6 个业务类别映射 | `ai/detector.py:26-43` | T01 |
| Person-PPE 空间匹配 | `ai/ppe_matcher.py` | T02、T03 |
| 归一化矩形 ROI 区域判断 | `ai/roi.py` | T04 |
| 安全规则引擎（3 档严重等级） | `rule_engine/engine.py` | T05、T06 |
| 单张图片处理 + 标注 | `video/processor.py:107-136` | T07 |
| 视频文件处理 + 帧间隔推理 | `video/processor.py:139-216` | T08 |
| 违规截图（10 秒冷却去重） | `video/processor.py:30, 187-199` | T11 |
| SQLite 5 张表 | `backend/database.py` | T12 |
| FastAPI 11 个 HTTP 端点 | `backend/main.py` | T13 |
| 首页 Dashboard / 巡检 / 违规 / 统计 / 设置 5 个页面 | `backend/static/index.html` | T09、T10 |
| 历史违规筛选（类型/严重度/日期） | `backend/database.py:172-196` + `index.html` | T13 |
| 14 个端到端测试用例 | `tests/run_tests.py` | T01~T14 |

### 1.2 已存在但属于技术债 / 当前限制

> 这些**不是 bug**，是**MVP 阶段留下的取舍**，开发前必须清楚。

- **单文件后端**：所有 API 都在 `backend/main.py` 一个文件（246 行，11 个端点），没有按业务拆分 router
- **模型路径硬编码**：`backend/main.py:49` 直接写 `ai/model/sh17_yolov8s.pt`，没有走配置文件
- **前端图表走 CDN**：`index.html:7` 通过 `cdn.jsdelivr.net` 加载 ECharts，**断网环境下图表显示空白**
- **数据库查询裸 SQL**：所有 SQL 都直接写在 `backend/database.py`，没有 ORM
- **未做异常统一处理**：上传超大文件、模型加载失败等场景可能 500
- **requirements 未锁版本**：`requirements.txt` 用 `>=`，不同机器可能装出不同版本
- **存在孤立权重**：`weights/yolo26n.pt`（5.3 MB）在仓库里但**全项目零引用**，不是当前生产模型
- **空目录预留**：`backend/api/` 目前只有一个空 `__init__.py`，没有实际子模块
- **测试 T14 跨日失败**：T14 断言"今天"的检测次数 > 0；如果跨日运行且当天未做检测，会失败（已有记录在 `PROJECT_STATUS.md` 已知问题 13）

### 1.3 后续计划（不要当成"已实现"）

这些是 TODO 状态，**不是当前能跑的功能**：

- **AI 模型升级**：当前生产仍是 SH17（lab_coat F1 仅 34%，E3 实测）。E4 训练已暂停，等待 B1 自采数据
- **后端 API 分层**：将 `backend/main.py` 拆为 `backend/api/detect.py`、`backend/api/settings.py`、`backend/api/stats.py`
- **视频后台任务 + 进度查询**：当前视频检测是同步阻塞，长视频会卡住请求
- **摄像头实时巡检**：接口未实现
- **ROI 前端可视化编辑**：当前只能在系统设置页面通过表单输入归一化坐标，无拖拽画框
- **违规记录导出**（CSV / Excel）：API 未实现
- **前端异常处理 + 断网兜底**：CDN 降级方案
- **更多 AI 能力**：人脸识别、火焰检测、烟雾检测、化学品泄漏、行为识别等——**本项目明确不做**（见 `PROJECT_STATUS.md`）

---

## 二、系统完整运行链路（基于真实代码）

下面这张链路图，**每一个箭头都能在代码里找到对应行**。请配合代码一起读。

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. 用户打开 http://127.0.0.1:8000                                │
│    backend/main.py:66  GET /                                   │
│    → 返回 backend/static/index.html（前端单文件，335 行）       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. 浏览器请求 Dashboard 数据                                     │
│    index.html:179  loadDashboard()                              │
│    → fetch('/api/dashboard')                                    │
│    → backend/main.py:87  GET /api/dashboard                     │
│    → backend/database.py:199  dashboard_stats()                 │
│    → 返回 {今日检测次数, 今日违规次数, 今日违规类型分布, ...}     │
│    → ECharts 渲染图表（index.html:7 加载 echarts@5.5.0）        │
└─────────────────────────────────────────────────────────────────┘

用户点击 "🔍 智能巡检" → 上传图片/视频
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. 图片检测流程                                                  │
│    index.html:215  fetch('/api/detect/image', POST, form)        │
│    → backend/main.py:108 POST /api/detect/image                 │
│    → 保存到 data/test/uploads/img_xxxxxxxx.jpg                  │
│    → 从数据库加载 ROI 区域 + 规则                                │
│    → video/processor.py:109  process_image()                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. 视频处理管线 InspectionPipeline（核心）                       │
│    video/processor.py:86  class InspectionPipeline               │
│    每帧执行 4 步：                                               │
│    ① ai/detector.py:132  detector.detect(frame)                 │
│       → YOLO 推理 → Detection 列表（含 person / PPE）           │
│    ② ai/ppe_matcher.py:50  match_person_ppe(dets)              │
│       → 把 PPE 框按"中心点是否落在 person 框内"归属给具体人员    │
│       → 输出 PersonState 列表                                   │
│    ③ ai/roi.py:40  locate_person()                             │
│       → 判断 PersonState 是否在受控区域内                       │
│       → 区域内 → 标记 required_ppe                              │
│    ④ rule_engine/engine.py:72  engine.evaluate(states)          │
│       → 缺 PPE → 输出 ViolationEvent（一般/严重违规）           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. 标注 + 截图 + 写库                                            │
│    video/processor.py:37  draw_annotations()                    │
│       → OpenCV 在画面上画：ROI 矩形、person 框、PPE 框、违规文字│
│    video/processor.py:121 或 :187                               │
│       → 违规时把当前帧保存到 outputs/screenshots/img_xxx.jpg    │
│       → 视频模式按 (person_id, 缺失 PPE 组合) 10 秒冷却去重     │
│    backend/main.py:126  db.save_detection_record()              │
│       → 写入 detection_records 表                               │
│    backend/main.py:138  db.save_violation_events()              │
│       → 写入 violation_events 表                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. 返回前端                                                      │
│    backend/main.py:140  返回 JSON {                              │
│       person_states, events, detections,                        │
│       annotated_url: "/media/annotated/xxx.jpg"                 │
│    }                                                            │
│    index.html:224  renderImageResult(r)                         │
│       → 显示标注图 + 人员表格                                    │
└─────────────────────────────────────────────────────────────────┘

视频流程几乎一致，区别在：
- 后端接口：backend/main.py:153 POST /api/detect/video
- 处理函数：video/processor.py:141 process_video()
  - 每 stride=2 帧做一次推理，中间帧复用上次结果
  - 生成标注视频写入 outputs/videos/
  - 返回字段：total_frames / inferred_frames / video_fps / events
- 前端：index.html:237 renderVideoResult(r) 显示 video 标签
```

**关键调用关系一句话版**：
- 前端 → `backend/main.py`（HTTP）→ `video/processor.py`（管线）→ `ai/detector.py`（模型）→ `ai/ppe_matcher.py`（匹配）→ `ai/roi.py`（区域）→ `rule_engine/engine.py`（判断）→ `backend/database.py`（持久化）→ JSON 回前端

---

## 三、目录与文件说明表

### 3.1 顶层目录（打开项目根目录看到的）

| 路径 | 作用 | 当前状态 | 负责人 | 修改风险 |
|---|---|---|---|---|
| `ai/` | AI 模型封装（检测 + 匹配 + ROI） | 已生产 | AI 成员 | 中（多文件被多个上层调用） |
| `backend/` | 后端：FastAPI 入口 + 数据库 + 前端静态目录 | 已生产 | 后端成员 | 高（main.py 是汇聚点） |
| `backend/api/` | 后端 API 分层预留目录 | **空壳**（只有 `__init__.py`） | 后端成员 | — |
| `backend/static/` | 前端 HTML/JS/CSS（单文件 index.html） | 已生产 | 前端成员 | 中（业务逻辑全在一个文件） |
| `rule_engine/` | 安全规则引擎（违规等级判断） | 已生产 | 公共模块 | **高（所有人都会调用）** |
| `video/` | 视频/图片处理管线 | 已生产 | 后端成员 | 中 |
| `tests/` | 端到端测试用例（T01~T14） | 已生产 | 测试成员 | 低 |
| `tools/` | AI 实验工具（数据下载/转换/评估） | **实验脚本**，非生产代码 | AI 成员 | 极低（与生产解耦） |
| `docs/` | 项目文档（交接审计、计划、契约、报告） | 已生产 | 组长 | 低 |
| `data/test/` | 测试夹具（t01~t10 + 演示视频） | 已入库 | 测试成员 | 低 |
| `data/raw/` | AI 实验原始数据集 | **已 git ignore** | AI 成员 | — |
| `data/lab_ppe/` | 转换后的 YOLO 训练集 | **已 git ignore** | AI 成员 | — |
| `ai/model/sh17_yolov8s.pt` | **当前生产模型**（22 MB，YOLOv8s） | 已生产 | AI 成员 | **最高（碰错就生产崩）** |
| `weights/yolo26n.pt` | 孤立旧权重，零引用 | **可删除**（询问组长） | AI 成员 | — |
| `outputs/` | 标注图 / 截图 / 标注视频 | 运行时产物 | — | — |
| `runs/` | YOLO 训练过程产物（PR 曲线等） | **已 git ignore** | AI 成员 | — |
| `venv/` | Python 虚拟环境 | **已 git ignore** | — | — |
| `wheels/` | 离线 pip 安装包（2.4 GB） | **已 git ignore** | — | — |
| `start.bat` | Windows 一键启动脚本 | 已生产 | 后端成员 | 低 |
| `requirements.txt` | Python 依赖列表 | 已生产 | 后端成员 | 低（建议锁版本） |
| `README.md` | 项目入口说明 | 已生产 | 组长 | 低 |
| `.gitignore` | Git 排除规则 | 已生产 | 组长 | 中 |

### 3.2 关键源码文件详解

#### `ai/detector.py`（108 行）
- **作用**：加载 YOLO 模型，把 SH17 的 17 个原始类别映射成 6 个系统业务类别
- **核心数据**：
  - `CLASS_ALIAS`（行 26-43）：类别映射表，把 `face-mask-medical` → `mask`、`medical-suit` → `lab_coat` 等
  - `SYSTEM_CLASSES`（行 46）：`["person", "mask", "gloves", "lab_coat", "goggles", "helmet"]`
  - `CLASS_CN`（行 49-52）：中文标签
- **核心类**：
  - `Detection`：单个检测结果（cls_name / raw_cls / conf / bbox xyxy）
  - `PPEDetector`：检测器封装，含 `detect(frame)` 方法
- **被谁调用**：`video/processor.py:100` `pipeline.analyze_frame()` → `self.detector.detect(frame)`
- **调用了谁**：`ultralytics.YOLO`
- **修改后影响**：所有检测结果都会变化，前端表格、违规判断、数据库全部跟着变

> ⚠️ **业务语义边界**（必读）：
> 当前代码 `CLASS_ALIAS` 把 SH17 的 `medical-suit` 和 `safety-suit` 都映射为 `lab_coat`（行 40-41）。
> **项目已确认的业务决策是**：连体防护服 / 安全服 / coverall **不应**视为实验服。
> 这是一个**当前代码与业务语义不一致的点**，AI 成员在后续模型升级时需要专门处理。
> **不要在不明确的情况下修改这一行**——这会影响所有检测结果。

#### `ai/ppe_matcher.py`（83 行）
- **作用**：把画面里的 PPE 检测框"分给"具体某个人员
- **核心算法**：
  - 规则 1：PPE 框中心点 → 落在哪个 person 框内 → 归那个人
  - 兜底：中心点不在任何 person 框内 → 找最近的人（距离 < person 对角线 × 0.5）
- **核心数据**：`PersonState`（person_id / bbox / ppe dict / in_roi / area_name）
- **被谁调用**：`video/processor.py:101`
- **修改后影响**：PPE 归属错位会直接改变"哪些人违规"的结果

#### `ai/roi.py`（51 行）
- **作用**：判断人员是否在受控区域内
- **核心数据**：`AreaROI`（area_id / name / x1~y2 归一化矩形 / required_ppe 列表）
- **核心函数**：`locate_person(person_det, frame_w, frame_h, areas) → (in_roi, area)`
- **被谁调用**：`rule_engine/engine.py:112 annotate_person_states()` 在每个 PersonState 上回填 `in_roi` / `area_name` / `area_required`
- **修改后影响**：ROI 判断错位 → 所有违规事件都可能错位

#### `rule_engine/engine.py`（118 行）
- **作用**：判断每个在 ROI 内的人是否违规
- **核心常量**：
  - `SEVERITY_NONE = "正常"`
  - `SEVERITY_MINOR = "一般违规"`
  - `SEVERITY_MAJOR = "严重违规"`
  - `VIOLATION_LABEL`：英文 PPE → 中文违规标签
- **核心数据**：`ViolationEvent`（person_id / area_name / violation_types / missing_ppe / severity / timestamp / frame_time / screenshot / person_bbox）
- **核心类**：`SafetyRuleEngine.evaluate(person_states, frame_time)` → 对每个 in_roi 的人对照 required_ppe，缺失 0=正常、缺失 1=一般违规、缺失 ≥2=严重违规
- **被谁调用**：`video/processor.py:103` + `backend/main.py:124-138`（结果入库）
- **调用了谁**：`ai/roi.locate_person()`
- **修改后影响**：违规等级标准变了，整个系统的判定逻辑都变

> ⚠️ **业务硬规则（不能轻易改）**：
> - 严重等级阈值：`minor_threshold` 默认 = 1（即"缺失 1 个 = 一般违规，缺失 ≥2 个 = 严重违规"）
> - 默认必需 PPE：`mask` + `gloves` + `lab_coat`（见 `backend/database.py:96-99`）
> - 默认可选 PPE：`goggles` + `helmet`
> - 普通区域（非受控区域）：不做 PPE 强制判断

#### `video/processor.py`（216 行）
- **作用**：把"AI 检测 + 匹配 + ROI + 规则"四步串成一根管线，并提供图片/视频入口
- **核心类**：`InspectionPipeline`（detector + engine + screenshot_dir）
- **核心函数**：
  - `process_image()`（行 109-136）：单张图片 → 标注图 + 截图 + 返回 dict
  - `process_video()`（行 141-216）：视频逐帧 → 标注视频 + 截图（10 秒冷却去重）
- **被谁调用**：`backend/main.py:118, 163`
- **调用了谁**：所有 `ai/*` + `rule_engine/engine.py` + OpenCV
- **修改后影响**：影响所有检测结果的可视化与持久化

#### `backend/database.py`（247 行）
- **作用**：SQLite 数据库的所有 CRUD
- **5 张表**：
  - `laboratories`：实验室
  - `areas`：检测区域（矩形 ROI，归一化坐标）
  - `safety_rules`：区域的 PPE 要求（可配置）
  - `detection_records`：每次巡检记录
  - `violation_events`：违规事件
- **核心函数**：
  - `init_db(seed=True)`（行 77-101）：建表 + 首次启动注入演示配置（"实验室A" + "实验操作区" + mask/gloves/lab_coat 必需规则）
  - `load_areas(lab_id)`（行 106-121）：读 ROI + 规则
  - `save_detection_record()` / `save_violation_events()` / `save_event_dicts()`：写入
  - `query_violations()`（行 172-196）：支持按类型/严重度/日期筛选
  - `dashboard_stats()`（行 199-232）：首页统计
  - `statistics()`（行 235-247）：数据统计页
- **被谁调用**：`backend/main.py` 所有端点
- **修改后影响**：改表结构 / 改字段名 → 所有 API 和前端会坏

> ⚠️ **数据库硬规则**：
> - 数据库文件路径：`backend/labsafety.db`（行 17），**已经 git ignore**
> - 启动时 `init_db(seed=True)` 会自动建表 + 注入默认数据，重复执行不会冲突
> - 如果改了表结构：必须先备份 `backend/labsafety.db`，再删库重启

#### `backend/main.py`（246 行）—— ⚠️ **项目汇聚点**
- **作用**：FastAPI 入口，所有 HTTP 端点
- **11 个端点**（行号即代码位置）：

| 方法 | URL | 行号 | 用途 |
|---|---|---|---|
| GET | `/` | 66 | 返回前端页面 |
| GET | `/api/model/info` | 72 | 模型信息（设备、类别、置信度） |
| GET | `/api/dashboard` | 87 | 首页今日统计 |
| GET | `/api/statistics` | 92 | 数据统计页 |
| GET | `/api/violations` | 97 | 违规记录查询（支持筛选） |
| POST | `/api/detect/image` | 108 | 上传图片 → 检测 |
| POST | `/api/detect/video` | 153 | 上传视频 → 检测 |
| GET | `/api/settings` | 187 | 实验室/区域/规则配置 |
| POST | `/api/settings/lab` | 202 | 新建实验室 |
| POST | `/api/settings/area` | 216 | 新建区域（含 PPE 规则） |
| DELETE | `/api/settings/area/{area_id}` | 234 | 删除区域 |
| startup | — | 56 | 启动时初始化数据库 |
| mount | `/static` | 62 | 前端静态文件 |
| mount | `/media` | 63 | outputs 目录对外可访问 |

- **关键代码**：
  - `get_pipeline()`（行 43-53）：懒加载模型，首次请求时加载 `ai/model/sh17_yolov8s.pt`
  - **硬编码模型路径**：`ROOT / "ai" / "model" / "sh17_yolov8s.pt"`（行 49）
- **被谁调用**：前端 `index.html` 的所有 fetch
- **调用了谁**：`backend/database.py` + `video/processor.py` + `ai/detector.py`
- **修改后影响**：改了 URL / 改了返回字段 → 前端立刻 404 / 显示空白

> ⚠️ **修改 main.py 的协作规则**：
> 这是所有改动的汇聚点，**5 个人都可能需要碰**：
> - AI 成员：换模型（应改为读 `config.py`，但目前是硬编码）
> - 后端成员：拆分 router、修复 bug
> - 前端成员：通常不碰，但如果 API 变化需要协调
> - 测试成员：通过 TestClient 调用，但不改代码
> - **如果你修改了 main.py，请提 PR 并@组长**

#### `backend/static/index.html`（335 行）
- **作用**：整个前端，单文件包含 HTML + CSS + JS + ECharts 调用
- **5 个 Tab**：Dashboard / 智能巡检 / 违规记录 / 数据统计 / 系统设置
- **关键代码位置**：
  - 行 7：`<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/...">`  ← 断网会失效
  - 行 86：`onclick="runInspect()"` 上传检测入口
  - 行 178：`loadDashboard()` 加载首页
  - 行 215：上传文件 fetch
  - 行 224 / 237：图片 / 视频结果显示
  - 行 251 / 271 / 292：违规 / 统计 / 设置加载
  - 行 313 / 322 / 327：新建实验室 / 新建区域 / 删除区域
- **被谁调用**：用户浏览器
- **调用了谁**：`backend/main.py` 所有 `/api/*` 端点
- **修改后影响**：影响所有用户界面

### 3.3 测试与实验脚本

| 文件 | 作用 | 谁能改 |
|---|---|---|
| `tests/run_tests.py`（314 行） | T01~T14 端到端测试 | 测试成员 |
| `tests/evaluate_model.py`（212 行） | 模型评估脚本（独立，不进测试套件） | AI 成员 |
| `tests/acceptance_check.py`（63 行） | 验收检查（独立） | 测试成员 |
| `tests/model_check.py`（49 行） | 模型加载检查 | AI 成员 |
| `tests/make_test_video.py`（73 行） | 合成测试视频 | 测试成员 |
| `tests/roi_demo.py`（144 行） | ROI 可视化演示 | 测试成员 |
| `tools/` 全部 13 个脚本 | 数据下载 / 转换 / 去重 / 训练 / 评估 | AI 成员（实验脚本，与生产解耦） |

---

## 四、五人开发边界

> ⚠️ **这不是最终分工文档**，最终分工以 `docs/TEAM_DEVELOPMENT_PLAN.md` 为准。本节只告诉你"我的代码应该在哪个目录"。

| 成员 | 角色 | 主要负责目录 | 偶尔会碰 | 绝对不要碰 |
|---|---|---|---|---|
| **组长（你）** | 架构 / 集成 / 公共接口 / 最终审核 | `docs/`、`backend/main.py`（仲裁）、release 流程 | `backend/database.py` 表结构（需评审）、`rule_engine/engine.py` 业务常量 | 生产模型直接换 |
| **AI 成员（许婧雯）** | 模型 / 数据集 / AI 实验 | `ai/`、`tools/`、`data/raw/`、`data/lab_ppe/`、`ai/model/`、`weights/`、`runs/` | `rule_engine/engine.py`（只读，确认硬编码业务常量）、`docs/` AI 报告 | 前端、后端路由 |
| **后端+视频成员（马军）** | 后端 / 视频 / 数据库 | `backend/`（不含前端静态目录）、`video/`、`start.bat` | `rule_engine/engine.py`（只读） | 前端 HTML/JS、AI 模型文件 |
| **前端+测试+Demo 成员（吴森灵）** | 前端 / 系统功能 / 主文档 / Demo 视频 | `backend/static/`、`tests/`、`outputs/`、`docs/` 用户级 | `backend/main.py` 仅限新增挂载点 | 数据库 schema、AI 模型、后端业务 |
| **测试+PPT+独立功能成员（艾柯代）** | 测试 / PPT / 演讲 / 独立功能 | `tests/`、PPT 目录（新增）、新功能文件（建议放 `backend/export.py` 等独立模块） | 必要时可在 `backend/main.py` 新增 `include_router` 一行（极小 diff） | 核心业务逻辑、AI 训练 |

**边界硬规则**：
1. **AI 成员**不要改 `backend/main.py` 的业务逻辑（可以建议改模型加载方式给后端，由后端实施）
2. **后端成员**不要改 `ai/` 下的 AI 逻辑（可以建议改 API 字段给 AI，由 AI 评估）
3. **前端成员**不要改 SQL（需要新字段请提 issue，由后端改 DB + API + 前端）
4. **测试成员**不要为了测试通过而改业务代码（应改测试用例或向成员提 issue）
5. **PPT/独立功能成员**的独立功能**必须有独立文件**，避免和主代码冲突

---

## 五、禁止随意修改清单（公共接口 / 核心结构）

### 5.1 绝对不能直接改的（必须先沟通）

| 项目 | 在哪里 | 为什么不能动 |
|---|---|---|
| **生产模型文件** | `ai/model/sh17_yolov8s.pt` | 改了就崩生产 |
| **类别映射表** | `ai/detector.py:26-43` `CLASS_ALIAS` | 改了所有检测结果变；尤其是 medical-suit/safety-suit 是否映射 lab_coat 是**业务决策** |
| **系统类别列表** | `ai/detector.py:46` `SYSTEM_CLASSES` | 前端、数据库、UI 都依赖这 6 个值 |
| **违规等级常量** | `rule_engine/engine.py:23-25` | 改了前端标签显示、数据库查询都错 |
| **数据库表结构** | `backend/database.py:19-67` | 改了所有 API 和历史数据可能丢 |
| **默认必需 PPE** | `backend/database.py:96-99` | 这是**项目业务硬规则**：mask + gloves + lab_coat |
| **main.py 11 个 API 路径** | `backend/main.py` 行 66/72/87/92/97/108/153/187/202/216/234 | 前端全部 hardcoded |
| **ViolationEvent 字段** | `rule_engine/engine.py:36-48` | 改了数据库写入、前端显示都坏 |
| **Detection 字段** | `ai/detector.py:55-65` | 整个管线的契约 |
| **PersonState 字段** | `ai/ppe_matcher.py:18-25` | 改了 ROI 回填会失败 |

### 5.2 可以修改但必须小范围 / 必须测试

| 项目 | 在哪里 | 修改建议 |
|---|---|---|
| 后端 SQL 查询逻辑 | `backend/database.py:172-247` | 加新字段、加新筛选时**先备份 `backend/labsafety.db`** |
| 前端样式 | `backend/static/index.html:8-49` CSS 部分 | 不影响逻辑 |
| 测试用例 | `tests/run_tests.py` | 加新用例时不要改原 14 个用例 |
| 文档 | `docs/` | 任何时候都可以改 |

### 5.3 可以自由修改

- 你自己的新文件（`backend/export.py`、`docs/AI_REPORT_xxx.md` 等）
- 实验性脚本（`tools/`）
- `requirements.txt`（但建议锁版本，避免装出新版本出兼容问题）

---

## 六、当前必须遵守的"语义边界"

> 这些是项目已经确定的设计决策，**不要因为"模型能识别"就擅自改变**。

1. **连体防护服 ≠ 实验服**
   - SH17 模型的 `medical-suit`、`safety-suit`、`coverall` 等类别，**业务上不视为 lab_coat**
   - 当前代码 `CLASS_ALIAS` 把 `medical-suit`/`safety-suit` 映射为 `lab_coat`（detector.py:40-41），**这是一个已知的代码与业务不一致**，AI 成员后续需专门处理
   - 不要因为模型能检测到，就直接当作 lab_coat 计入违规

2. **默认必需 PPE**（不可改）
   - `mask`（口罩）
   - `gloves`（手套）
   - `lab_coat`（实验服）

3. **默认可选 PPE**（不强制）
   - `goggles`（护目镜）
   - `helmet`（安全帽）

4. **违规等级规则**（不可改）
   - 缺失 0 个 → 正常
   - 缺失 1 个 → 一般违规
   - 缺失 ≥2 个 → 严重违规

5. **当前生产模型就是 SH17**
   - 即使 E2/E4 训练出了更好的模型，**也不要直接替换生产**
   - 必须先经过评估（dataset1 + dataset2 + t01-t10），通过验收门槛（详见 `docs/E4_data_prep_report.md`），再由组长确认切换

6. **dataset2 是外部验证集**
   - **永远不进入训练 / 验证**
   - 不要根据 dataset2 调参或数据增强

---

## 七、给"小白"的实际例子

### Q1：我要修改"什么算违规"的规则，应该先打开哪个文件？

**先打开 `rule_engine/engine.py`**，重点看：
- 行 23-25：违规等级常量
- 行 27-33：违规标签（英文 → 中文）
- 行 65-104：`SafetyRuleEngine.evaluate()` 方法——这就是规则的核心

如果你想改：
- **缺失几个算严重违规** → 改 `SafetyRuleEngine.__init__()` 的 `default_severity_minor` 参数
- **某种 PPE 算哪种违规** → 改 `VIOLATION_LABEL`
- **加一种新的 PPE 类型** → 要同时改 `ai/detector.py:46` 的 `SYSTEM_CLASSES` + 数据库的 `safety_rules` 表 + 前端 UI（**这是公共接口变更，需要先和组长讨论**）

### Q2：我要修改视频处理，应该看哪个文件？

**主文件 `video/processor.py`**，重点看：
- 行 86-104：`InspectionPipeline` 类——这是"检测 + 匹配 + ROI + 规则"四步合一的封装
- 行 107-136：`process_image()` 单张图片处理
- 行 139-216：`process_video()` 视频处理
- 行 30：`VIOLATION_COOLDOWN_S = 10.0`——视频模式违规截图冷却时间

如果你想改：
- **推理频率** → 改 `backend/main.py:155` 的 `stride: int = Form(2)`，stride 越大越快但越粗糙
- **违规截图去重逻辑** → 改 `process_video()` 行 167-199
- **标注图样式** → 改 `draw_annotations()` 函数（行 37-83）

### Q3：我要修改前端统计页面，应该看哪里？

**`backend/static/index.html`**，重点看：
- 行 117-128："数据统计" Tab 的 HTML 结构
- 行 271：`loadStats()` 函数——从 `/api/statistics` 拿数据
- 行 76：`ECharts` 图表渲染

ECharts 用的 `echarts@5.5.0`，从 CDN 加载（行 7），**断网会失败**。

### Q4：我要换 AI 模型，应该改哪里，不能改哪里？

**当前生产模型**：`ai/model/sh17_yolov8s.pt`（22 MB，YOLOv8s 框架）

**只能改一处**：`backend/main.py:49`
```python
_detector = PPEDetector(ROOT / "ai" / "model" / "sh17_yolov8s.pt")
```

**不要改**：
- `ai/model/` 下的文件名（除非同步修改 `main.py:49`）
- `weights/yolo26n.pt`（这是孤立旧权重，不要拿来用）
- `ai/detector.py` 的 `PPEDetector` 类（除非新模型类别数和 SH17 不同）

**换模型的完整流程**：
1. 把新模型 `.pt` 文件放到 `ai/model/` 下
2. 改 `backend/main.py:49` 指向新模型
3. 重启服务
4. 跑 `tests/run_tests.py` 验证
5. 如果新模型的类别名不同 → 需要同步更新 `ai/detector.py:26-43` 的 `CLASS_ALIAS`

### Q5：我要新增一个 API，应该先检查什么？

1. **先看 `backend/main.py` 是否已有类似端点**——不要重复实现
2. **检查 URL 命名是否一致**——`/api/xxx/yyy` 风格
3. **检查返回格式**——是否返回 `{ok: true, ...}` 或 `{items: [...], total: N}`
4. **如果涉及数据库**——先看 `backend/database.py` 是否有现成的查询函数
5. **如果涉及 AI 推理**——先看 `video/processor.py` 的 `InspectionPipeline` 是否能复用

### Q6：我要看懂一张违规记录是怎么从图片变成数据库里的？

看 `backend/main.py:108-149` 的 `detect_image` 函数，**逐行读**：
- 行 109：接 `file` + `lab_id`
- 行 112-114：保存到 `data/test/uploads/`
- 行 116：加载 ROI 配置
- 行 119-120：调用 `process_image` 处理
- 行 124-125：分离违规 / 正常
- 行 126-128：写 `detection_records` 表
- 行 130-138：写 `violation_events` 表
- 行 140-149：返回 JSON 给前端

---

## 八、开发前必读规则

### 8.1 Git 协作

1. **不要直接在 `main` 上开发**。所有改动先在 `develop` 或 `feature/*` 上做
2. **每个人使用自己的 feature 分支**：
   - `feature/ai`（AI 成员）
   - `feature/backend`（后端成员）
   - `feature/frontend`（前端成员）
   - `feature/docs`（文档协作）
3. **修改前先确认自己负责的范围**——参考第四节"五人开发边界"
4. **不要随意修改公共 API、数据库表结构、核心业务类别**——参考第五节
5. **不把实验代码（`tools/`）当生产代码（`ai/` `backend/` `video/` `rule_engine/`）**
6. **提交前必须自己测试**：`venv\Scripts\python.exe tests\run_tests.py`，至少不破坏原有 14 个用例
7. **Push 后通过 Pull Request 提交**，PR 描述写明：改了什么、为什么改、影响范围、如何测试
8. **合并前由组长（成员 1）进行功能验收**

### 8.2 分支命名

```
main                    生产分支（仅组长可合并）
develop                 集成分支（feature/* 合到这里测试）
feature/ai              AI 成员工作分支
feature/backend         后端成员工作分支
feature/frontend        前端成员工作分支
feature/docs            文档协作分支
feature/member5-export  PPT 成员的独立功能分支（示例）
```

### 8.3 Commit 建议格式

```
<type>: <description>

[可选正文：详细说明]
[可选脚注：关联 issue / 验收依据]
```

`<type>` 选一个：`feat`（新功能）/ `fix`（修 bug）/ `docs`（文档）/ `refactor`（重构）/ `test`（测试）/ `chore`（杂项）

**示例**：
- `feat(api): 新增违规记录导出 CSV 接口`
- `fix(frontend): ECharts 断网时图表空白`
- `docs: 补充 PROJECT_FRAMEWORK_GUIDE.md`

### 8.4 修改后的自测清单

每改一处，建议跑：

1. **静态检查**：`python -c "import backend.main"` 不报错
2. **单元测试**：`python tests/run_tests.py`，原有 14 个用例至少通过 13 个（T14 是已知跨日失败）
3. **冒烟测试**：启动 `start.bat`，浏览器打开 http://127.0.0.1:8000 能看到页面

### 8.5 紧急情况

- **生产模型加载失败** → 检查 `ai/model/sh17_yolov8s.pt` 是否存在、是否被损坏；可临时回退到 E2 训练的 best.pt
- **数据库表不存在** → 启动时 `db.init_db(seed=True)` 会自动建表；如果已存在会跳过
- **前端图表空白** → ECharts CDN 加载失败，需本地化（见"暂未实现"部分）
- **视频上传 500** → 检查 `outputs/` 是否可写、磁盘是否满

---

## 九、附录：当前默认配置速查

### 9.1 端口与启动

```
启动：start.bat → venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
访问：http://127.0.0.1:8000
数据库：backend/labsafety.db（启动时自动建表 + 注入演示数据）
```

### 9.2 默认 PPE 配置（不可改）

| 区域 | 必需 PPE | 可选 PPE |
|---|---|---|
| 实验操作区（默认） | mask, gloves, lab_coat | goggles, helmet |

### 9.3 系统支持的全部业务类别（不可改）

```
person, mask, gloves, lab_coat, goggles, helmet
```

### 9.4 违规等级（不可改）

| 缺失数量 | 等级 | 中文标签 |
|---|---|---|
| 0 | SEVERITY_NONE | 正常 |
| 1 | SEVERITY_MINOR | 一般违规 |
| ≥2 | SEVERITY_MAJOR | 严重违规 |

### 9.5 测试用例列表

| 用例 ID | 测试什么 |
|---|---|
| T01 | YOLO 模型加载与类别映射 |
| T02 | 单人 PPE 检测 |
| T03 | 多人 PPE 检测 |
| T04 | ROI 区域判断 |
| T05 | 一般违规判定 |
| T06 | 严重违规判定 |
| T07 | 图片处理完整流程 |
| T08 | 视频处理完整流程 |
| T09 | 数据库初始化 |
| T10 | 实验室 + 区域 + 规则 CRUD |
| T11 | 违规截图保存 |
| T12 | 违规记录写入数据库 |
| T13 | 历史记录查询（筛选） |
| T14 | 数据统计（**已知跨日失败**，见 `PROJECT_STATUS.md` 已知问题 13） |

### 9.6 一行命令速查

```bash
# 启动服务
start.bat

# 跑测试
venv\Scripts\python.exe tests\run_tests.py

# 单独加载模型检查
venv\Scripts\python.exe tests\model_check.py

# 合成测试视频
venv\Scripts\python.exe tests\make_test_video.py

# ROI 可视化
venv\Scripts\python.exe tests\roi_demo.py
```

---

## 十、自检清单（写这份文档的人留下的最后核对记录）

本文档中所有"当前已实现"的描述都能在以下位置找到代码依据：

| 描述 | 代码位置 |
|---|---|
| YOLO PPE 检测 | `ai/detector.py:108-157` |
| 17 → 6 类别映射 | `ai/detector.py:26-43` |
| Person-PPE 匹配 | `ai/ppe_matcher.py:50-82` |
| 矩形 ROI 判断 | `ai/roi.py:13-50` |
| 3 档严重等级规则 | `rule_engine/engine.py:65-104` |
| 单图处理 | `video/processor.py:109-136` |
| 视频处理 + 10 秒冷却 | `video/processor.py:139-216` |
| SQLite 5 张表 | `backend/database.py:19-67` |
| FastAPI 11 个端点 | `backend/main.py` 行 66/72/87/92/97/108/153/187/202/216/234 |
| 默认 PPE 配置 | `backend/database.py:96-99` |
| 模型硬编码 | `backend/main.py:49` |
| 5 个前端 Tab | `backend/static/index.html:59-65` |
| ECharts CDN | `backend/static/index.html:7` |
| 14 个测试用例 | `tests/run_tests.py` |

**不一致点**（文档已标记）：
- 当前 `ai/detector.py:40-41` 把 `medical-suit` / `safety-suit` 映射为 `lab_coat`，与项目业务决策"连体防护服 ≠ 实验服"不一致——AI 成员后续需专门处理（详见 1.2 节与第六节）

---

> 文档版本：v1.0  
> 适用代码基线：v0.1.0-baseline（commit 3aa0599）  
> 最后更新：基于真实代码逐行验证  
> 维护者：组长（成员 1）