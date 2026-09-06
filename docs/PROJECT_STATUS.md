# PROJECT_STATUS — 项目当前状态

> 更新日期：2026-09-06（组长前置工作完成时点）
> 基线：v0.1.0-baseline（git tag）
> 数据来源：真实代码与实验产物（docs/E1~E4 报告、tests/test_report.md、runs/、outputs/）
> 姊妹文档：`docs/PROJECT_HANDOFF_REPORT.md`（完整审计）、`docs/TEAM_DEVELOPMENT_PLAN.md`（任务分解）

---

## 一、当前已完成

**生产系统（MVP v0.1，可运行、闭环完整）**

| 领域 | 内容 | 依据 |
|------|------|------|
| 系统架构 | 检测→匹配→ROI→规则→事件→落库→Web 四层闭环 | docs/系统架构与模块说明.md；代码调用链 |
| 后端 | FastAPI 13 端点（页面/模型信息/统计×2/记录/检测×2/设置×4/挂载×2） | backend/main.py |
| AI 检测 | PPEDetector 封装 SH17 YOLOv8s，17类→6业务类映射 | ai/detector.py |
| PPE 匹配 | 中心点归属 + 0.5×对角线距离兜底 + 高置信度保留 | ai/ppe_matcher.py（T08 通过） |
| ROI | 归一化矩形多区域、数据库可配置、演示视频 | ai/roi.py；outputs/roi_demo.mp4 |
| 规则引擎 | 区域级可配置 PPE 要求 + 三级违规分级 | rule_engine/engine.py（T01-T07 通过） |
| 数据库 | SQLite 5 表 + 种子配置 + dashboard/statistics 聚合 | backend/database.py（T12-T14 通过） |
| 前端 | 5 Tab 单页（Dashboard/智能巡检/违规记录/数据统计/系统设置）+ ECharts | backend/static/index.html |
| 图片检测 | 上传→检测→标注图→入库，实测 48.8ms | main.py:108（T09 通过） |
| 视频检测 | stride 抽帧、标注视频、10s 冷却截图去重、89.3 FPS | video/processor.py（T10/T11 通过） |

**模型实验体系（E1~E3 完成，E4 暂停）**

| 实验 | 状态 | 关键结果 |
|------|------|---------|
| E1 数据准备 | ✅ 完成 | dataset1 7401 张（13类→4类）+ dataset2 251 张；去重零泄漏；dataset2 锁定 external_test |
| E2 LAB_PPE 训练 | ✅ 完成 | YOLOv8s 训练 50ep；valid mAP50=**0.924**；test lab_coat F1=**0.952**；产物 runs/e2/labppe_v8s/best.pt |
| E3 独立测试 | ✅ 完成 | dataset2 外测：LAB_PPE vs SH17 —— mask F1 75.1 vs 56.5（改善）/ gloves 75.7 vs 78.5（持平）/ lab_coat 34.0 vs 15.9（2.1x 改善但未达 0.70 目标）/ goggles 32.7 vs 49.8（回退）；t01-t10 判对 2 vs 3 |
| E4 lab_coat 定向增训 | ⏸ **暂停（禁止启动）** | 方案已批准（B3+B1 数据、best.pt 续训 30ep）；B3 476 张已下载+去重审计完成（≈450 可用）；**B1 自采 150 张未到位，等组长提供** |

**协作基础（2026-09-06 新建）**
- git 仓库 + main/develop/feature 分支 + .gitignore（tag v0.1.0-baseline）
- docs/INTERFACE_CONTRACT.md（接口契约 v1.0，冻结 Detection/PersonState/AreaROI/ViolationEvent/13 个 API）
- docs/DEVELOPMENT_GUIDE.md（开发协作说明）
- docs/TEAM_DEVELOPMENT_PLAN.md（五人任务分解 29 项 + Phase 0~4 排期）
- docs/PROJECT_HANDOFF_REPORT.md（只读交接审计）

## 二、当前已知问题

**生产代码级（第二阶段处理，见 INTERFACE_CONTRACT 第 8 节技术债清单）**
1. 模型路径硬编码 `backend/main.py:49`（sh17_yolov8s.pt）——换模型需改代码，计划配置化。
2. `medical-suit/safety-suit → lab_coat` 映射与 E4 决策（连体服不算实验服）冲突，E4 定型时修订。
3. 视频检测接口同步阻塞，长视频期间整个 API 无响应；无进度查询。
4. 检测端点未捕获 ValueError（非图片/视频上传 → 500 而非 4xx）。
5. 设置端点裸 SQL 绕过 database 层；区域坐标无 0~1 范围校验。
6. `@app.on_event("startup")` 为 FastAPI 弃用写法。
7. 前端 ECharts 走 jsdelivr CDN——**断网演示图表全挂（答辩风险）**。
8. person_id 为帧内编号，无 tracking；跨帧编号变化可能导致事件漏记/重复。
9. requirements.txt 未锁版本（实测环境：Python 3.13 / torch 2.6.0+cu124 / ultralytics）。
10. 中文路径 cv2.imread 失败（上传已用 ASCII 重命名规避；新代码需注意）。

**数据/模型级**
11. 生产模型仍是 SH17：dataset2 外测 lab_coat F1 仅 34.0%（E3 实测），E2/E4 更优模型**尚未接入生产**。
12. E4 增训被 B1 数据 gating；B1 未到位不得启动训练（组长约束）。

**测试框架级**
13. **T14 存在跨日数据依赖**（2026-09-06 回归验证发现）：T12 注入的测试数据会被用例自身清理（run_tests.py:242-247），而 T14 断言 `detection_count > 0` 统计的是"当天"记录——若当日尚未通过 API 跑过任何检测，T14 必然失败。2026-09-05 通过 14/14 是因为当天有 API 检测残留数据；2026-09-06 复跑 13/14（仅 T14 失败，T01-T13 全过，性能 48.6ms/101.5FPS 与基线一致），**非功能回归**。修复建议（测试成员 Phase 1 处理）：T14 改为断言返回结构字段齐全 + by_day 含历史数据，或注入数据不清理。今日回归记录：`tests/test_report_2026-09-06_regression.md`。

## 三、当前模型情况

| 模型 | 位置 | 状态 | 指标 |
|------|------|------|------|
| SH17 YOLOv8s | `ai/model/sh17_yolov8s.pt`（22.5MB） | **生产在用（唯一）**，main.py:49 加载 | SH17 官方验证集 mAP50=63.7（非本项目实测） |
| LAB_PPE v1（E2） | `runs/e2/labppe_v8s/weights/best.pt` | 训练完成，**未接入生产**，仅 tools/ 评估引用 | dataset1 valid mAP50=**0.924**；test lab_coat F1=**0.952** |
| E4 增训版 | 无 | 未启动 | 目标：dataset2 lab_coat F1≥0.50（最低）/≥0.70（优秀） |

**模型红线**：dataset2 永久 external_test；不映射 Coverall/medical-suit/safety-suit 为 lab_coat（E4 定型接入时统一修订映射）。

## 四、当前测试情况

- `tests/run_tests.py`：T01~T14 系统测试 + GPU/CPU 性能实测，**2026-09-05 实测 14/14 全部通过**，报告 `tests/test_report.md`。
- 辅助脚本：evaluate_model.py（11点AP50评估器）、model_check.py（10张公开图定性检查）、acceptance_check.py（验收自检）、make_test_video.py、roi_demo.py。
- 外测基准：E3 的 t01~t10 判定对比 + dataset2 三模型指标（见 docs/E3_eval_report.md）。
- 不足：非 pytest 框架、无 CI、回归靠手动触发。

## 五、当前性能情况（RTX 4060 Laptop 8GB 实测，2026-09-05）

| 指标 | 数值 |
|------|------|
| 单张图片全流程（检测+匹配+规则+绘制） | 48.8 ms |
| 视频处理（GPU, stride=2, 含写盘） | 89.3 FPS |
| CPU 纯推理（无 GPU） | 12.9 FPS |
| 违规事件生成成功率 | 100%（T10~T12） |

## 六、尚未实现的功能

1. E4 lab_coat 定向增训与三模型对比（等 B1 数据）
2. 生产切换到 E4 定型模型（含 CLASS_ALIAS 修订）
3. 视频后台任务 + 进度查询 API
4. 摄像头实时巡检（管线已就绪，仅差接口）
5. 多目标跟踪（person_id 稳定化）
6. 前端 ECharts 本地化、ROI 拖拽绘制、视频进度条
7. 违规记录导出（CSV/Excel）
8. API 分层（backend/api/ 目前是空壳）+ lifespan 升级
9. requirements 版本锁定、模型路径配置化

## 七、明确暂不做的功能（范围控制，任何人不得擅自加入）

以下功能**明确排除在项目范围外**，不因"功能看起来更多"而加入：

- 人脸识别
- 登录/权限系统（users 表）
- 火焰检测
- 烟雾检测
- 化学品泄漏检测
- 温度/气体传感器接入
- 打架/异常行为识别
- 门禁
- 多摄像头复杂管理（单摄像头实时巡检属"时间允许才做"的可选项）
- 大规模云端部署（本课程交付以单机演示为准）

## 八、下一阶段工作

按 `docs/TEAM_DEVELOPMENT_PLAN.md` 执行：

1. **Phase 0（已完成）**：git 基线冻结 ✅
2. **Phase 1**：并行启动——前端 ECharts 本地化（消除断网风险）、后端 config 化+API 分层、违规导出功能、B3 清洗、PPT 大纲
3. **Phase 2**：E4 训练（**等组长提供 B1 自采 150 张**；到位前 AI 成员做 B3 清洗与 MODEL_CARD 骨架）、视频后台任务、ROI 前端
4. **Phase 3**：模型定型 → 生产切换 → 全量回归
5. **Phase 4**：文档定稿、PPT、Demo 视频、三套交付包

**待组长决定的事项**：
- B1 自采数据（约 150 张）的采集/标注时间点（AI 线唯一外部依赖）
- B3 中 20 张图库水印图与 5 张近重复图是否剔除（建议剔除，等待确认）
- 生产模型 sh17_yolov8s.pt 是否随 git 仓库分发（当前策略：不入库，走 FINAL_MODEL 包；如改主意取消 .gitignore 中一行动画释即可）
