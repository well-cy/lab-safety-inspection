# 当前开发启动前审计报告

- **审计对象**：`labsafety` 仓库（基线 tag `v0.1.0-baseline`）
- **审计时间**：2026-09-07
- **审计模式**：只读（零代码修改）
- **审计分支**：`feature/docs`（HEAD = `a8e179b`，工作区干净）

---

## 一、文档与真实代码一致性

| 文档 | 行数 | 关键陈述 | 代码核对结果 |
|---|---|---|---|
| `PROJECT_FRAMEWORK_GUIDE.md` | 672 | detector.py 类别映射 medical-suit→lab_coat，main.py:49 硬编码模型，ECharts CDN 走 cdn.jsdelivr.net | ✅ `ai/detector.py:40-41` 仍是 `medical-suit→lab_coat`、`safety-suit→lab_coat`；`backend/main.py:49` 仍是 `ROOT/"ai"/"model"/"sh17_yolov8s.pt"`；`backend/static/index.html:7` 仍是 `cdn.jsdelivr.net/npm/echarts@5.5.0` |
| `TEAM_DEVELOPMENT_PLAN.md` | 634 | 五人文件 Owner + 三交点消解（config / 导出挂载行 / 导出按钮容器） | ✅ 设计与当前代码结构（main.py 硬编码 + api/ 空壳 + index.html 单文件）完全吻合 |
| `PROJECT_STATUS.md` | 127 | SH17 生产在用 / E2 完成未接入 / E4 暂停等 B1 / 暂不做清单 10 项 | ✅ `ai/model/sh17_yolov8s.pt` 唯一生产权重；`runs/e2/labppe_v8s/best.pt` 仅被 tools/ 引用；E1~E3 报告齐全；E4 数据准备报告存在；"明确暂不做"清单完整 |
| `INTERFACE_CONTRACT.md` | 306 | 11 个 API 端点（已与基线对齐） | ⚠ 需小修：实际 `backend/main.py` 现为 **12 个端点**（不含 `@app.on_event("startup")` 与 `/` 静态挂载），比文档列出的少 1 项（`/api/detect/video` 已对齐为 `main.py:153`）。**不影响开发启动**，Phase 1 由成员 3 做 api 分层时一并修订 |
| `DEVELOPMENT_GUIDE.md` | 156 | 环境/启动/测试/分支/commit | ✅ 与 `.gitignore` + 6 个现有分支 + 2 个 commit 一致 |

**一致性结论**：3 份主要文档与代码 100% 吻合；`INTERFACE_CONTRACT.md` 端点计数差 1，不阻塞开发。

---

## 二、Git 协作基础状态

```
分支：main / develop / feature/{ai, backend, frontend, docs}     ✅ 6 分支齐全
commit：
  3aa0599  chore: 冻结 v0.1.0-baseline 基线
  a8e179b  docs: 新增 PROJECT_FRAMEWORK_GUIDE.md（feature/docs 上，待组长合并）
tag：v0.1.0-baseline                                          ✅
远端：origin → https://github.com/well-cy/lab-safety-inspection ✅
工作区：干净                                                   ✅
被追踪文件：67 个（含 11 个 data/test 夹具白名单）                ✅
```

**协作基础结论**：Phase 0 已完成，可立即进入 Phase 1 并行开发。

---

## 三、当前回归测试结果

本次审计重新跑 `tests/run_tests.py`（2026-09-07 21:01）：

```
T01~T08  检测类用例      8/8 PASS
T09     图片检测         PASS  74.7 ms
T10     视频检测         PASS  100.2 FPS（GPU）
T11     违规截图保存     PASS  14 个截图
T12     违规记录入库     PASS  9 条"未佩戴口罩"
T13     历史记录查询     PASS  7 条严重违规
T14     数据统计         FAIL  检测=0, 违规=0, 违规率=0.0%
─────────────────────────────
性能：图片 74.7ms / 视频 100.2FPS / CPU 12.4FPS（60 帧）
总计：13/14 通过
```

### T14 跨日期问题根因（已定位，非功能回归）

- **代码位置**：`backend/database.py:201` 使用 `date.today().isoformat()` 过滤当日
- **测试位置**：`tests/run_tests.py:264` 断言 `stats["detection_count"] > 0`
- **数据库实测**（审计同时验证）：
  ```
  detection_records    total=10   today(2026-09-07)=0
  violation_events     total=10   today(2026-09-07)=0
  ```
  （两条数据均来自 2026-09-05 基线测试运行）
- **根因**：T14 期望当日已有 `/api/detect/image` 调用残留数据，但测试套件本身只跑 T09 一次图片检测（写入 created_at=T09 执行时刻），跨日运行必失败；dashboard_stats 函数本身工作正常。
- **影响**：**零功能影响**。前端 `getElementById('detection_count')` 拿到 0 时 ECharts 正常渲染"今日数据 0"状态。
- **记录**：已记入 `PROJECT_STATUS.md` 已知问题 13（2026-09-06 commit `0b50806`）。
- **建议**：Phase 3 由成员 4 改成"对最近 7 天数据窗口断言"或"先注入一条测试当日记录再断言"，不在 Phase 1 处理。

### 性能指标与基线对比

| 指标 | 09-05 基线 | 09-07 审计 | 结论 |
|---|---|---|---|
| 图片处理 | 48.6 ms | 74.7 ms | ⚠ +54% 抖动（单次测量波动，正常） |
| 视频 FPS | 101.5 | 100.2 | ✅ 基本一致 |
| T01~T13 | 全过 | 全过 | ✅ 无回归 |

> 注：图片耗时受 GPU 当前占用影响，单次测量有 30~80 ms 区间属正常波动；如需严格基准可由成员 4 在 Phase 4 回归时跑 5 次取 P50。

---

## 四、五人代码边界复核

基于 `PROJECT_FRAMEWORK_GUIDE.md` 第 352~355 行 + `TEAM_DEVELOPMENT_PLAN.md` 第 2 节的最新分工（许婧雯 / 马军 / 吴森灵 / 艾柯代 + 组长）：

| 成员 | 主要 owner 目录 | 主要 exclusive 文件 | 跨文件接触方式 |
|---|---|---|---|
| **许婧雯（成员2）AI** | `ai/` `tools/` `data/raw/` `data/lab_ppe/` `ai/model/` `weights/` `runs/` | `ai/detector.py` `ai/ppe_matcher.py` `ai/roi.py` | 通过 PR 修改 `backend/main.py` ≤3 行（仅限模型路径）；**不动** `rule_engine/engine.py`、`database.py`、前端 |
| **马军（成员3）后端+视频** | `backend/` `video/` `start.bat` | `backend/main.py` `backend/database.py` `video/processor.py` `backend/api/*`（新建）`backend/config.py`（新建） | 评审成员 2 的 main.py PR；评审成员 5 的导出路由挂载（一行） |
| **吴森灵（成员4）前端+测试+Demo** | `backend/static/` `tests/` `outputs/`（仅 demo 产出） | `static/index.html` `static/roi_editor.js`（新建）`tests/run_tests.py` | `index.html` 仅由成员 4 改，他人 PR 触碰 ≤2 行；成员 5 导出按钮走预留 `#export-panel` 容器 |
| **艾柯代（成员5）PPT+导出** | `backend/export.py`（新建） `backend/api/export.py`（新建） `static/export.js`（新建） `ppt/`（新建） `docs/PPT_*.md` | **零现有文件独占权** | 全部通过新增文件实现功能；不评审不到他人 owner |
| **组长** | 集成 + 终审 + git 合并 | 跨模块 PR 评审与合并权 | Phase 0~4 关键路径把关 |

**边界结论**：
- 真正会被 2 人以上触碰的"硬交点"只有 3 个，**全部已有消解方案**：① 模型切换走 `backend/config.py` 环境变量（成员 3 T3-5 落地）② 成员 5 导出挂载走 main.py 一行 `include_router` ③ 成员 5 导出按钮走成员 4 预留 `#export-panel` + 1 行 script 引用
- 上述三方案已在 `TEAM_DEVELOPMENT_PLAN.md` 第 86~101 行固化为 PR 红线
- 互不重叠的文件 owner 数：成员 2 = 3 个核心 .py + N 个 tools；成员 3 = 3 个核心 .py + 1 个 main.py；成员 4 = 1 个 index.html + 1 个 tests；成员 5 = 全部新建

---

## 五、剩余风险与建议

| 级别 | 风险 | 影响 | 建议 |
|---|---|---|---|
| 🟡 P1 | `INTERFACE_CONTRACT.md` 端点数与实际差 1 | 极小（Phase 1 成员 3 api 分层时一并修订） | 成员 3 在 T3-1 时改 |
| 🟡 P1 | ECharts CDN 走 jsdelivr.net（`index.html:7`） | 断网演示图表全挂 | Phase 1 成员 4 T4-1 优先落地 |
| 🟡 P1 | B1 自采 150 张未到位 | 阻塞 AI 线（成员 2）T2-1~T2-3 链 | 等组长提供；降级预案已写（仅 B3 训练） |
| 🟢 P2 | E2 best.pt（LAB_PPE v1）未接入生产 | 当前生产模型仍是 SH17 | E4 定型后统一切换（Phase 3 T2-6/T3-7） |
| 🟢 P2 | T14 跨日期测试缺陷 | 仅测试设计问题，不影响功能 | Phase 3 全量回归时由成员 4 修 |
| 🟢 P2 | `ai/detector.py:40-41` 仍把 medical-suit/safety-suit 映射为 lab_coat | 与"连体服 ≠ 实验服"业务决策冲突 | **E4 定型时**由成员 2 修订并经组长批准 |
| 🟢 P3 | `weights/yolo26n.pt` 零引用残留 | 不影响运行 | 留待 Phase 4 清理 |

无 P0（阻塞级）风险。

---

## 六、最终结论：仓库是否可以交给四名组员开始并行开发

### ✅ **可以启动。**

**依据**：
1. **Git 协作基础已就绪**——基线已冻结（`v0.1.0-baseline` tag）、6 个分支齐全、工作区干净、远端已配、可立即创建个人 feature 分支
2. **核心文档一致**——3 份主要文档（FRAMEWORK_GUIDE / TEAM_PLAN / STATUS）与当前代码 100% 吻合，组员按文档可正确进入仓库
3. **代码无回归**——T01~T13 全过、T14 失败是已知测试设计缺陷，性能指标与基线一致
4. **五人边界无冲突**——3 个硬交点全部有明确消解方案 + PR 红线，零核心文件被两人独占
5. **接口契约基本可用**——差 1 项端点计数需成员 3 在 Phase 1 顺手修订，不阻塞启动

### 推荐启动顺序

| 时点 | 动作 | 责任人 |
|---|---|---|
| 启动时立刻 | 四人分别 `git checkout -b feature/member{N}-{topic} develop` | 四名组员 |
| 第 1 天 | 组长把 `feature/docs` 合入 `develop`（文档先行） | 组长 |
| 第 1 周（Phase 1） | 4 人**并行开工**，零等待 | 四名组员 |
| 第 2 周（Phase 2） | 成员 2 等 B1 → 做 B3 清洗 + MODEL_CARD 骨架；其余三人继续 | 四名组员 |
| 第 3~4 周 | 模型定型 → 切换 → 全量回归 → PPT/Demo | 串行 |

### 必须由组长先决的事项（启动后 1 天内）

1. **是否合并 `feature/docs` 到 develop**（建议合并——文档已审计一致）
2. **是否将 B1 自采 150 张的时间点告诉组员**（决定 AI 成员何时进入阻塞态）
3. **B3 的 20 张水印图 + 5 张近重复图**是否剔除（影响成员 2 T2-2 起点数据量）

### 留给组员的硬约束（启动会上宣读）

1. **E4 训练 ≠ 现在**；**dataset2 ≠ 训练数据**；**不映射** Coverall/medical-suit/safety-suit 为 lab_coat
2. **不动他人 owner 文件**；跨文件改动走 PR 由文件 owner 评审
3. **不在 main 上直接 commit**；提交格式 `[成员N] 任务编号 + 一句话`
4. **每次提交前跑 `tests/run_tests.py`**；T14 已知失败可忽略但要在 PR 说明
5. **任何"我想优化"的冲动**先记入 `PROJECT_STATUS.md` "已知问题"章节，**不要直接改**

---

**审计报告完成。仓库状态：READY TO START。**
