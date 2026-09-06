# 系统测试报告

执行时间: 2026-09-05 15:57:31  
模型: sh17_yolov8s.pt @ NVIDIA GeForce RTX 4060 Laptop GPU

**通过: 14/14**

| 用例 | 名称 | 条件 | 期望结果 | 实际结果 | 是否通过 |
| ---- | ---- | ---- | -------- | -------- | -------- |
| T01 | 正常佩戴全部装备 | 构造: person 拥有 mask+gloves+lab_coat 且在操作区 | severity=正常 | 构造用例: severity=正常, missing=[]; (参考: 真图t05实际检出PPE=['gloves']) | ✅ |
| T02 | 未戴口罩 | 构造: person+gloves+lab_coat, 缺 mask, 在操作区 | 违规类型=[未佩戴口罩], severity=一般违规 | ['未佩戴口罩'], 一般违规 | ✅ |
| T03 | 未戴手套 | 构造: person+mask+lab_coat, 缺 gloves, 在操作区 | 违规类型=[未佩戴手套], severity=一般违规 | ['未佩戴手套'], 一般违规 | ✅ |
| T04 | 未穿实验服 | 构造: person+mask+gloves, 缺 lab_coat, 在操作区 | 违规类型=[未穿实验服], severity=一般违规 | ['未穿实验服'], 一般违规 | ✅ |
| T05 | 同时缺少两种装备 | 构造: person+mask, 缺 gloves+lab_coat, 在操作区 | 缺2种→severity=严重违规 | 缺 ['gloves', 'lab_coat'], 严重违规 | ✅ |
| T06 | 人员位于普通区域 | 构造: person 中心点在 ROI 外(左侧) | 不生成违规事件（不评估） | 事件数=0, in_roi=False | ✅ |
| T07 | 人员进入操作区域 | 构造: person 中心点在 ROI 内 | 生成评估事件（缺3种→严重违规） | 事件数=1, in_roi=True | ✅ |
| T08 | 多人同时出现 | 构造: 2 个 person 空间分离，PPE 全部属于 P1 | 2 个评估事件；P1 检出3种PPE，P2 检出0种（不串人） | 人数=2, 事件数=2, P1 PPE=['gloves', 'lab_coat', 'mask'], P2 PPE=[] | ✅ |
| T09 | 图片检测 | 输入: t05_pcr_scientist.jpg | 返回标注图/人员状态/事件，耗时记录 | persons=1, events=1, 耗时=48.2ms | ✅ |
| T10 | 视频检测 | 输入: demo_lab_video.mp4 (399帧) | 输出标注视频+处理FPS | inferred=200, fps=88.3, events=2, 输出=test_annotated.mp4 | ✅ |
| T11 | 违规截图保存 | 检查 outputs/screenshots 中的视频违规截图 | 违规时生成 jpg 截图文件 | 截图文件数=8 | ✅ |
| T12 | 违规记录写入数据库 | 插入一条测试违规事件后查询 | 能查询到该记录 | 查询到 4 条'未佩戴口罩'记录 | ✅ |
| T13 | 历史记录查询 | 按 severity=严重违规 筛选 | 返回过滤后的记录列表 | 严重违规记录数=2 | ✅ |
| T14 | 数据统计 | 调用 dashboard_stats() | 返回检测数/违规数/违规率/类型分布 | 检测=5, 违规=3, 违规率=75.0% | ✅ |

## 性能指标

- 单张图片处理时间: 48.2 ms
- 视频处理速度（GPU, stride=2, 含绘制/写盘）: 88.3 FPS
- CPU 纯推理速度: 12.9 FPS
- 违规判断准确率 / 事件生成率 / 存储成功率: 见 T02-T12 结果
