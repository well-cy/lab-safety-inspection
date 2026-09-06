# INTERFACE_CONTRACT — 接口契约文档 v1.0

> 冻结日期：2026-09-06（基于 v0.1.0-baseline 代码逐行核实，非猜测）
> 作用：五人并行开发的"法律基础"。任何跨模块改动若触碰本文档定义的结构/签名，
> 必须先提契约变更评审（组长主持），否则 PR 直接打回。
> 依据文件：`ai/detector.py`、`ai/ppe_matcher.py`、`ai/roi.py`、`rule_engine/engine.py`、
> `video/processor.py`、`backend/main.py`、`backend/database.py`
> 行号以 baseline commit 为准，后续代码演进时以"结构定义"为准、行号仅供参考。

---

## 0. 契约变更流程

1. 提出人在自己的 feature 分支上写清：改哪个契约、为什么、影响哪些下游模块。
2. @组长 + 受影响模块 owner 评审（≥2 人同意）。
3. 通过后：先改本文档（单独 PR），再改代码 PR。
4. **默认规则：只增不改不删**——新增可选字段可以；改字段名/类型/语义必须走评审。

---

## 1. Detection（检测统一结构）

**定义位置**：`ai/detector.py:55-80`（`@dataclass Detection`）

### 1.1 输入
- `PPEDetector.detect(frame, conf=None)`：`frame` 为 **BGR ndarray**（OpenCV 读取的原生格式，未做颜色转换）
- 推理参数：`imgsz=640`、`device` 构造时决定（auto→CUDA 可用则 GPU）
- 置信度阈值：构造参数 `conf_threshold=0.35`（生产默认值），`detect()` 可临时覆盖

### 1.2 输出字段（冻结）

| 字段 | 类型 | 含义 |
|------|------|------|
| `cls_name` | str | **系统业务类别**（6 选 1，见 1.4） |
| `raw_cls` | str | 模型原始类别名（如 `face-mask-medical`） |
| `conf` | float | 置信度 0~1 |
| `x1, y1, x2, y2` | float | bbox，**像素绝对坐标 xyxy**（相对输入帧尺寸） |

派生属性：`center`（框中心点）、`area`（框面积）。

### 1.3 坐标与尺寸约定（重要）
- Detection 坐标 = **像素绝对值**（相对当前输入帧的 w/h）
- ROI 坐标 = **归一化值 0~1**（两者在 `ai/roi.py:normalize_detection` 转换，转换需传 `frame_w, frame_h`）
- 下游模块**禁止假设帧尺寸**，一律以 `frame.shape[:2]` / 传参为准

### 1.4 类别映射（`CLASS_ALIAS`，`ai/detector.py:26-43`）

系统业务类别共 6 个：`person / mask / gloves / lab_coat / goggles / helmet`。

SH17 原始类别 → 业务类别映射（当前生产实现）：

| SH17 原始类别 | 映射结果 |
|--------------|---------|
| person | person |
| face-mask / face-mask-medical / facemask / mask | mask |
| gloves / glove | gloves |
| glasses / face-guard | goggles |
| helmet | helmet |
| **medical-suit / safety-suit** | **lab_coat**（近似映射，见 1.5 ⚠） |
| head / face / ear / ear-mufs / hands / foot / shoes / safety-vest / tools | None（丢弃，不进入下游） |

兜底规则（`map_class`，字典未命中时按子串匹配）：含 `person/mask/glove/glass/goggle/helmet/hardhat` → 对应业务类；**含 `suit/coat` → lab_coat**。

### 1.5 ⚠ 已知口径冲突（记录在案，暂不改）
E4 实验已决策"连体防护服（Coverall/medical-suit/safety-suit）**不应**视为 lab_coat"，但当前生产代码仍将其映射为 lab_coat（MVP 近似取舍）。**此差异在 E4 定型模型接入生产时一并修订**（AI 成员负责），在此之前任何成员不得单独改动此映射。

### 1.6 序列化格式（`to_dict()`，对外 API 使用）
```json
{"cls": "mask", "cls_cn": "口罩", "conf": 0.87, "bbox": [x1, y1, x2, y2]}
```
`cls_cn` 来自 `CLASS_CN` 中文表；bbox 数值 round 1 位小数；conf round 3 位。

---

## 2. Person-PPE 匹配（`ai/ppe_matcher.py`）

### 2.1 输入 / 输出
- 输入：`match_person_ppe(detections: list[Detection], max_dist_ratio=0.5)`
- 输出：`list[PersonState]`

### 2.2 PersonState 结构（冻结，`ppe_matcher.py:18-39`）

| 字段 | 类型 | 含义 |
|------|------|------|
| `person_id` | int | **帧内**编号（从 1 递增；无 tracking，跨帧不稳定——已知局限） |
| `bbox` | Detection | person 检测框 |
| `ppe` | dict[str, Detection] | 已绑定到该人员的 PPE，键为业务类别 |
| `in_roi` | bool | 是否在受控区域（由 ROI 环节回填） |
| `area_name` | str | 所在区域名（回填） |

隐式扩展字段（**注意**）：`area_required: list[str]` 由 `rule_engine.annotate_person_states` 动态回填（`PersonState` dataclass 本身**未声明**此字段）。此为已知技术债，契约要求下游用 `hasattr` 防御（当前 engine.py:84 即如此），重构需走评审。

### 2.3 匹配规则（两步，保持可解释）
1. **中心点归属**：PPE 检测框中心点落入哪个 person 框 → 归属之；多人框重叠时归属**面积更小**的框（更可能是本人）。
2. **距离兜底**：中心点不在任何 person 框内时，归属"person 中心点距离最近"的人员，但仅当距离 ≤ `0.5 × 该 person 框对角线长度`（`max_dist_ratio=0.5`）。
3. 同类 PPE 冲突：保留**置信度更高**的一个。

### 2.4 已知局限（不改，记录在案）
远处/他人手持的 PPE 可能被距离兜底错误归属 → 误报；背对镜头、遮挡的 PPE 无法检出 → 漏报。

---

## 3. ROI（`ai/roi.py`）

### 3.1 AreaROI 结构（冻结）

| 字段 | 类型 | 含义 |
|------|------|------|
| `area_id` | int | 数据库 areas.id |
| `name` | str | 区域名，如"实验操作区" |
| `x1, y1, x2, y2` | float | **归一化矩形坐标 0~1** |
| `required_ppe` | list[str] | 该区域要求的 PPE 业务类别列表 |

### 3.2 判定规则
- **形状：仅矩形**（多边形 ROI 属后续扩展，需契约变更）
- **判定点：person 检测框的中心点**（非框重叠面积）
- `locate_person(person_det, frame_w, frame_h, areas) -> (in_roi, AreaROI|None)`
- **多区域时按数据库返回顺序，第一个命中即生效**（无优先级配置）
- `in_roi=False` → 该人员"不评估"（普通区域）

---

## 4. Rule Engine（`rule_engine/engine.py`）

### 4.1 默认规则（**任何人不得擅自修改**）

数据库种子配置（`database.py:77-99`，实验室A/实验操作区）：

- **必需 PPE（required=1）**：`mask`、`gloves`、`lab_coat`
- **可选 PPE（默认不要求）**：`goggles`、`helmet`
- 区域要求存储在 `safety_rules` 表，**按区域可配置**，非硬编码

### 4.2 严重程度定义（`SafetyRuleEngine.evaluate`，`minor_threshold=1`）

对**每个在 ROI 内的人员**：`missing = required − 已佩戴`

| 情况 | severity（中文字符串常量） |
|------|--------------------------|
| 缺失 0 种 | `正常`（SEVERITY_NONE） |
| 缺失 1 种 | `一般违规`（SEVERITY_MINOR） |
| 缺失 ≥2 种 | `严重违规`（SEVERITY_MAJOR） |
| 不在 ROI（普通区域） | **不产生事件**（不评估） |

### 4.3 违规类型标签（`VIOLATION_LABEL`）
`mask→未佩戴口罩`、`gloves→未佩戴手套`、`lab_coat→未穿实验服`、`goggles→未佩戴护目镜`、`helmet→未佩戴安全帽`。`violation_types` 输出**中文标签**；`missing_ppe` 输出**英文类别**。

### 4.4 处理顺序（管线契约，`video/processor.py:97-104 analyze_frame`）
```
detect(frame) → match_person_ppe(dets) → annotate_person_states(states, areas, w, h) → engine.evaluate(states, frame_time)
```
`analyze_frame` 返回三元组 `(person_states, events, detections)`——此签名冻结。

---

## 5. ViolationEvent（`rule_engine/engine.py:36-62`）

### 5.1 字段（冻结）

| 字段 | 类型 | 含义 |
|------|------|------|
| `person_id` | int | 帧内人员编号 |
| `area_name` | str | 所在区域名（"普通区域"或区域名） |
| `violation_types` | list[str] | 违规中文标签列表 |
| `missing_ppe` | list[str] | 缺失 PPE 英文类别列表 |
| `severity` | str | 正常 / 一般违规 / 严重违规 |
| `timestamp` | str | 系统时间 `%Y-%m-%d %H:%M:%S` |
| `frame_time` | float | 视频内时间（秒）；图片模式恒 0 |
| `screenshot` | str | 截图文件名（相对 `outputs/screenshots/`）；无则为空串 |
| `person_bbox` | list[float] | 人员框像素坐标 xyxy |

派生：`is_violation`（severity ∈ {一般违规, 严重违规}）。

### 5.2 序列化格式（`to_dict()`）
```json
{"person_id": 1, "area": "实验操作区", "violation_types": ["未佩戴口罩"],
 "severity": "一般违规", "timestamp": "2026-09-06 12:00:00",
 "frame_time": 3.52, "screenshot": "vid_20260906_..._p1.jpg"}
```
注意：**API 返回的事件不含 missing_ppe / person_bbox**（仅内部使用）。

### 5.3 数据库落地（`violation_events` 表）
`violation_types` 落库为**逗号分隔字符串**；`created_at` 由 SQLite `datetime('now','localtime')` 生成。图片模式正常评估也入库（`include_normal=True`，供 dashboard normal_count 统计）；视频模式只入违规事件。

---

## 6. API 契约（`backend/main.py`，FastAPI，uvicorn 127.0.0.1:8000）

> 全部端点实测于 baseline；交互式文档 `/docs`（FastAPI 自动生成）。
> 静态挂载：`/static` → `backend/static/`；`/media` → `outputs/`（截图/标注图/视频的访问前缀）。

### 6.1 `GET /` — 前端页面
返回 `backend/static/index.html`。

### 6.2 `GET /api/model/info` — 模型信息
首次调用触发模型懒加载（可能耗时数秒）。
```json
{"model_file": "sh17_yolov8s.pt",
 "base": "SH17 dataset (official YOLOv8s weights, ultralytics)",
 "device": "NVIDIA GeForce RTX 4060 Laptop GPU",
 "conf_threshold": 0.35,
 "raw_classes": {"0": "person", "...": "..."},     // 模型原始类别 dict
 "system_classes": ["person", "mask", "gloves", "lab_coat", "goggles", "helmet"]}
```

### 6.3 `GET /api/dashboard` — 今日统计
无参数。
```json
{"date": "2026-09-06",
 "detection_count": 5,          // 今日检测次数（detection_records）
 "violation_count": 3,          // 今日违规事件数（severity != 正常）
 "normal_count": 4,             // 今日正常评估数（severity = 正常）
 "violation_rate": 42.9,        // 违规数/(违规+正常)×100，round 1 位
 "by_type": [{"t": "未佩戴口罩", "c": 2}, ...],   // 今日违规类型 Top10
 "by_day": [{"d": "2026-08-31", "c": 5}, ...]}   // 近 7 天违规趋势
```

### 6.4 `GET /api/statistics` — 历史统计
无参数。返回 `{"by_day": [{d,c}], "by_type": [{t,c}], "by_severity": [{s,c}]}`（by_day 近 30 天；by_severity 含"正常"）。

### 6.5 `GET /api/violations` — 违规记录查询
Query 参数（全部可选）：

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `vtype` | str | 无 | 违规类型模糊匹配（LIKE，传中文标签如"未佩戴口罩"） |
| `severity` | str | 无 | 精确匹配；**不传时默认排除"正常"记录** |
| `date` | str | 无 | `YYYY-MM-DD`，按 created_at 日期过滤 |
| `limit` | int | 200 | 分页 |
| `offset` | int | 0 | 分页 |

响应：`{"total": <本次返回条数>, "items": [<violation_events 行 + lab_name + screenshot_url>]}`
`items` 元素含 `id, record_id, lab_id, area_name, person_id, violation_types(逗号串), severity, screenshot_path, frame_time, created_at, lab_name, screenshot_url`；`screenshot_url` = `/media/screenshots/<文件名>`，无截图为空串。
错误：参数类型错误 → FastAPI 自动 422。

### 6.6 `POST /api/detect/image` — 图片检测
请求：`multipart/form-data`
| 字段 | 类型 | 必填 | 默认 |
|------|------|------|------|
| `file` | file | ✓ | — |
| `lab_id` | form int | | 1 |

响应：
```json
{"record_id": 12, "elapsed_ms": 48.8, "person_count": 2,
 "persons": [PersonState.to_dict()],       // {person_id, bbox, in_roi, area, ppe:{cls:Detection.to_dict()}}
 "events": [ViolationEvent.to_dict()],     // 含正常评估事件
 "violation_count": 1,
 "annotated_url": "/media/annotated/annotated_xxx.jpg",
 "detections": [Detection.to_dict()]}
```
错误：非图片/损坏文件 → `ValueError`（当前未捕获 → 500，**已知问题**，记录于 PROJECT_STATUS）；`lab_id` 不存在 → 无区域配置，全员按"普通区域"处理（不报错）。
副作用：写 detection_records + violation_events（含正常）；生成标注图与违规截图。

### 6.7 `POST /api/detect/video` — 视频检测（同步阻塞）
请求：`multipart/form-data`
| 字段 | 类型 | 必填 | 默认 |
|------|------|------|------|
| `file` | file | ✓ | — |
| `lab_id` | form int | | 1 |
| `stride` | form int | | 2（每 stride 帧推理 1 次，中间帧复用结果） |

响应：
```json
{"record_id": 13, "total_frames": 300, "inferred_frames": 151,
 "video_fps": 89.3, "elapsed_s": 3.4, "violation_count": 2,
 "events": [ViolationEvent.to_dict()],       // 已按 10 秒冷却去重（key=person_id+缺失组合）
 "output_video_url": "/media/videos/annotated_xxx.mp4"}
```
错误：非视频 → `ValueError`（同上，500）。⚠ 处理期间**同步阻塞**整个请求（已知问题，第二阶段改后台任务）。

### 6.8 `GET /api/settings` — 配置查询
响应：`{"labs": [{"id", "name", "description", "created_at", "areas": [{"id","lab_id","name","x1","y1","x2","y2","created_at","required_ppe":["mask","gloves","lab_coat"]}]}]}`

### 6.9 `POST /api/settings/lab` — 新建实验室
请求：form `name`（必填）、`description`（可选）。
响应：`{"ok": true}`；重名 → **400** `{"error": "实验室名称已存在"}`。

### 6.10 `POST /api/settings/area` — 新建区域
请求：form `lab_id`、`name`、`x1,y1,x2,y2`（float，归一化）、`required_ppe`（**默认字符串 `"mask,gloves,lab_coat"`**，逗号分隔）。
响应：`{"ok": true, "area_id": <int>}`。⚠ x/y 不做 0~1 范围校验（已知问题）。

### 6.11 `DELETE /api/settings/area/{area_id}` — 删除区域
级联删除其 safety_rules。响应：`{"ok": true}`；area_id 不存在也返回 ok（幂等，已知行为）。

---

## 7. 数据库 Schema 契约（`backend/database.py:19-67`）

5 张表（SQLite，`backend/labsafety.db`，**该文件不入 git**，启动自动建库+种子）：
`laboratories(id,name UNIQUE,description,created_at)` / `areas(id,lab_id→laboratories,name,x1,y1,x2,y2,created_at)` / `safety_rules(id,area_id→areas,ppe_type,required,UNIQUE(area_id,ppe_type))` / `detection_records(id,lab_id,source_type image|video,source_name,person_count,violation_count,normal_count,process_time_ms,created_at)` / `violation_events(id,record_id→detection_records,lab_id,area_name,person_id,violation_types逗号串,severity,screenshot_path,frame_time,created_at)`。

⚠ 已知问题：`get_settings` 端点在 main.py 直接裸 SQL（绕过 database.py），第二阶段收敛。

---

## 8. 契约级技术债清单（只记录，不擅自改）

| # | 债 | 位置 | 处置计划 |
|---|-----|------|---------|
| 1 | medical-suit/safety-suit→lab_coat 映射与 E4 决策冲突 | detector.py CLASS_ALIAS | E4 定型接入生产时由 AI 成员修订 |
| 2 | map_class 子串兜底会把未知 suit/coat 类映射为 lab_coat | detector.py:103 | 同上 |
| 3 | PersonState.area_required 隐式动态属性 | engine.py:84 hasattr | 后端重构时补 dataclass 字段 |
| 4 | 检测端点未捕获 ValueError → 500 | main.py:108/153 | 后端成员 Phase 1 统一异常处理 |
| 5 | settings 裸 SQL 绕过 database 层 | main.py:188-241 | 后端成员收敛 |
| 6 | area 坐标无 0~1 校验 | main.py:216 | 后端成员补校验 |
| 7 | person_id 帧内编号、无 tracking | ppe_matcher | 明确暂不做（PROJECT_STATUS） |
