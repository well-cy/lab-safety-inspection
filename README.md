# 基于计算机视觉的实验室安全智能巡检与违规预警系统

Laboratory Safety Intelligent Inspection and Violation Warning System —— MVP v0.1

> 「基于计算机视觉对实验室人员安全防护装备进行自动检测，并结合实验区域和
> 安全规则，实现实验室安全违规行为的智能巡检、自动预警和记录统计。」
>
> 核心创新不是新的 YOLO 算法，而是：**将开源计算机视觉模型与实验室安全
> 业务规则结合，形成从视觉检测到违规事件管理的完整闭环。**

## 1. 项目定位

传统实验室安全检查依靠人工巡查：频率有限、无法持续监测、违规发现不及时。
本系统利用计算机视觉对实验室视频/图片中的人员及安全防护装备（PPE）进行
检测，结合可配置的实验区域（ROI）与安全规则，自动判断防护装备佩戴违规，
生成违规事件、保存截图、落库并提供 Web 查询与统计。

**系统闭环：**

```
图片/视频输入 → YOLO 检测（人员+PPE）→ Person-PPE 空间匹配
→ ROI 区域判断 → 安全规则引擎 → 违规事件（截图+分级）
→ SQLite 存储 → FastAPI + Web 页面（Dashboard/巡检/记录/统计/设置）
```

## 2. 当前完成情况（MVP）

| 功能 | 状态 |
| ---- | ---- |
| 图片上传检测（人员+PPE+违规判断+标注） | ✅ 可运行 |
| 视频上传检测（标注视频输出+违规事件） | ✅ 可运行 |
| person / mask / gloves / lab_coat / goggles / helmet 检测 | ✅ 可运行（lab_coat 为近似映射，见已知问题） |
| 矩形 ROI 区域判断（可配置） | ✅ 可运行 |
| Person-PPE 空间匹配（多人场景） | ✅ 可运行 |
| 可配置安全规则引擎（区域级 PPE 要求、违规分级） | ✅ 可运行 |
| 违规截图保存 + 冷却去重 | ✅ 可运行 |
| SQLite 违规记录 / 检测记录 | ✅ 可运行 |
| Web 页面：Dashboard/智能巡检/违规记录/数据统计/系统设置 | ✅ 可运行 |
| T01~T14 测试用例 + 性能实测 | ✅ 见 tests/test_report.md |
| 摄像头实时检测 | ⏸ 第二阶段 |
| 多目标跟踪（tracking） | ⏸ 第二阶段 |
| 模型微调（自采数据） | ⏸ 第二阶段 |
| 用户登录/权限 | ⏸ 暂缓（按范围控制要求） |

## 3. 如何运行（从零开始）

环境要求：Windows + Python 3.10+（本项目使用 3.13）；NVIDIA GPU 可选
（无 GPU 自动回退 CPU）。

```bash
cd labsafety

# 1. 创建虚拟环境
python -m venv venv

# 2. 安装依赖（GPU 版 PyTorch，约 2.5GB）
venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
#   仅 CPU 环境用: venv\Scripts\pip install torch torchvision
venv\Scripts\pip install -r requirements.txt

# 3. （可选）生成演示测试视频
venv\Scripts\python tests\make_test_video.py

# 4. 运行测试用例（生成 tests/test_report.md）
venv\Scripts\python tests\run_tests.py

# 5. 启动 Web 系统
venv\Scripts\python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
#   或直接双击 start.bat
```

访问：**http://127.0.0.1:8000**（接口文档 http://127.0.0.1:8000/docs）

模型权重 `ai/model/sh17_yolov8s.pt`（22MB）**不随 git 仓库分发**（.gitignore 已排除全部
.pt）。获取方式二选一：

```
# 方式一：官方下载
curl -L -o ai/model/sh17_yolov8s.pt https://github.com/ahmadmughees/SH17dataset/releases/download/v1/yolo8s.pt
# 方式二：向项目负责人索取（FINAL_MODEL.zip）
```

## 4. 项目目录结构

```
labsafety/
├── ai/                        # AI 检测层
│   ├── model/sh17_yolov8s.pt  # SH17 官方 YOLOv8s 权重
│   ├── detector.py            # 检测器封装 + 类别映射
│   ├── ppe_matcher.py         # Person-PPE 空间匹配
│   └── roi.py                 # ROI 区域判断
├── rule_engine/
│   └── engine.py              # 安全规则引擎 + ViolationEvent
├── video/
│   └── processor.py           # 图片/视频处理管线 + 标注绘制
├── backend/
│   ├── main.py                # FastAPI 应用
│   ├── database.py            # SQLite（5 张表）
│   └── static/index.html      # Web 前端（单页）
├── data/test/                 # 测试图片(t01~t10) + 演示视频
├── outputs/
│   ├── screenshots/           # 违规截图
│   ├── annotated/             # 图片模式标注结果
│   └── videos/                # 视频模式标注结果
├── tests/
│   ├── run_tests.py           # T01~T14 + 性能实测
│   ├── make_test_video.py     # 演示视频生成
│   └── test_report.md         # 测试报告（运行后生成）
├── docs/
│   ├── 数据来源与开源项目说明.md
│   ├── 系统架构与模块说明.md
│   └── API说明.md
├── requirements.txt
├── start.bat                  # Windows 一键启动
└── README.md
```

## 5. 演示流程（Demo 操作步骤）

1. 双击 `start.bat`（或运行 uvicorn 命令），浏览器打开 http://127.0.0.1:8000
2. **系统设置**页：确认"实验室A → 实验操作区"已配置（默认要求口罩+手套+
   实验服）；可现场新建区域/修改必戴项，立即生效
3. **智能巡检**页：选择实验室A → 上传 `data/test/t05_pcr_scientist.jpg`
   → 点击"开始检测" → 查看标注图（蓝框=操作区，绿/橙/红框=人员状态）与
   下方人员明细表
4. 再上传视频 `data/test/demo_lab_video.mp4` → 等待处理 → 播放标注视频，
   观察"人员从普通区域走入操作区后才开始违规判断"
5. **违规记录**页：按类型/严重程度/日期筛选，点击缩略图放大查看截图
6. **数据统计**页：查看每日违规趋势、各类违规占比、正常/违规比例
7. **首页 Dashboard**：查看今日检测次数、违规次数、违规率

## 6. 开源资源与数据集

（详见 docs/数据来源与开源项目说明.md，此处摘要）

| 资源 | 来源 | License | 用途 |
| ---- | ---- | ------- | ---- |
| SH17 YOLOv8s 权重 | github.com/ahmadmughees/SH17dataset | 数据集开放（论文声明）；Ultralytics AGPL-3.0 | PPE 检测 |
| Ultralytics | github.com/ultralytics/ultralytics | AGPL-3.0 | 推理框架 |
| SH17 数据集 | 同上（8,099 图/17 类） | 开放 | 模型训练（官方） |
| 测试图片 ×10 | Wikimedia Commons（FDA/NIH/CDC 公开图片） | 公有领域/CC | MVP 测试 |
| 演示视频 | tests/make_test_video.py 合成 | 自制 | 演示 |

**模型原始指标（SH17 官方验证集，非本项目测试结果）**：
P=81.5 / R=55.7 / mAP@50=63.7 / mAP@50-95=41.7。
本项目自测结果见第 8 节与 tests/test_report.md。

## 7. 测试方案

14 个测试用例（T01 正常佩戴 / T02 未戴口罩 / T03 未戴手套 / T04 未穿实验服 /
T05 缺两种装备 / T06 普通区域 / T07 进入操作区 / T08 多人 / T09 图片检测 /
T10 视频检测 / T11 截图保存 / T12 数据库写入 / T13 历史查询 / T14 统计），
运行 `venv\Scripts\python tests\run_tests.py` 自动执行并生成
`tests/test_report.md`（含条件/期望/实际/是否通过 + 性能实测）。

## 8. 模型实际效果与性能（本机实测）

> 以下为真实测试结果，未编造。详见 tests/test_report.md。

- 测试设备：NVIDIA GeForce RTX 4060 Laptop GPU (8GB) + PyTorch 2.6.0+cu124
- 测试时间：2026-09-05（`venv\Scripts\python tests\run_tests.py` 实测输出）

| 指标 | 实测结果 |
| ---- | ---- |
| T01~T14 测试用例 | **14/14 全部通过** |
| 单张图片处理时间（检测+匹配+规则+绘制） | **48.8 ms** |
| 视频处理速度（GPU, stride=2, 含绘制写盘） | **89.3 FPS** |
| CPU 纯推理速度（无 GPU） | **12.9 FPS** |
| 违规判断准确率（T01~T08 构造用例） | 8/8 正确 |
| 违规事件生成成功率（T10~T12） | 100%（事件+截图+入库） |
| 数据库存储成功率（T12） | 100% |

**定性观察（10 张公开实验室图片）**：person 置信度 0.54~0.92、gloves
0.48~0.90、lab_coat(medical-suit 映射) 0.78、mask 0.52、goggles
0.44~0.78，主要类别均可稳定检出（详见 tests/model_check.py 输出记录）。

**注意**：模型在公开工业场景图上的检测表现 ≠ 在真实实验室场景的表现。
正式验证阶段将使用项目组自采实验室场景数据重新评估
Precision/Recall/mAP@50/F1。

## 9. 已知问题与局限

1. **lab_coat 为近似映射**：SH17 无独立白大褂类，medical-suit/safety-suit
   映射而来，对宽松白大褂识别可能不稳定 → 建议第二阶段用 Roboflow
   ppe-airbf（CC BY 4.0，含 labcoat 类）微调。
2. **无多目标跟踪**：person_id 是帧内编号，同一人跨帧编号会变化；违规
   冷却按"缺失组合"全局去重，可能漏记个别人员的独立事件。
3. **遮挡/背对/小目标**：PPE 检测依赖可见性，背对镜头的手套无法识别，
   可能产生误报（可用置信度阈值调节权衡）。
4. **视频同步处理**：上传接口为同步阻塞，长视频需等待；未做上传进度
   与后台任务队列。
5. **口罩/手套误检**：图片中他人手中的 PPE 可能通过距离兜底被错误归属。
6. **演示视频为合成**：第一阶段无自采数据，人员移动为模拟。

## 10. 后续开发建议（第二阶段）

1. 加 tracking（ultralytics 自带 ByteTrack 一行接入），person_id 稳定后
   做"持续违规时长"统计与更精准的事件去重。
2. 自采实验室数据（建议 300~500 张标注图）微调模型，重点补 lab_coat。
3. 摄像头实时巡检（`VideoCapture(0)` + 抽帧推理，管线已就绪，仅差接口）。
4. 后台任务队列处理视频（FastAPI BackgroundTasks / Celery）+ 进度查询。
5. 前端升级 Vue 3 + Element Plus + ECharts（当前单页 HTML 已预留 API 结构）。
6. 邮件/企业微信预警推送、导出报表（PDF/Excel）。
7. users 表与登录（当前按范围控制省略）。

## 11. License 说明

- 本项目自有代码：用于课程教学演示（项目组原创部分见
  docs/数据来源与开源项目说明.md 第 4 节）。
- 依赖组件遵循各自 License：Ultralytics AGPL-3.0（教学使用可接受，
  商用需评估）、SH17 数据集开放、FastAPI/MIT、OpenCV/Apache-2.0。

## 12. 多人协作开发（2026-09-06 起）

本项目已进入 5 人协作开发阶段，基线版本 `v0.1.0-baseline`。新成员必读：

- `docs/DEVELOPMENT_GUIDE.md` —— 环境搭建、启动、测试、Git 分支与 commit 规范、禁区目录
- `docs/INTERFACE_CONTRACT.md` —— Detection / PPE 匹配 / ROI / 规则引擎 / ViolationEvent / 全部 API 契约（跨模块改动的法律基础）
- `docs/PROJECT_STATUS.md` —— 当前进展、已知问题、模型情况、明确暂不做的功能
- `docs/TEAM_DEVELOPMENT_PLAN.md` —— 五人任务分解与排期
- `docs/PROJECT_HANDOFF_REPORT.md` —— 项目交接审计报告
