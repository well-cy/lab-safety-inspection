# E2 训练报告 —— YOLOv8s Lab_PPE Baseline

日期：2026-09-06
状态：**E2 完成，等待确认后进入 E3（独立测试）。未修改 detector.py，未触碰 dataset2。**

---

## 1. 训练命令与参数

```
命令: ./venv/Scripts/python.exe tools/train_e2.py
权重: yolov8s.pt（官方 COCO 80 类预训练，ultralytics.com 官方站下载，已验证）
```

| 参数 | 值 | 说明 |
|---|---|---|
| epochs | 50 | |
| imgsz | 640 | |
| batch | 16 | RTX 4060 Laptop 8GB 实测占用 3.4GB，无 OOM |
| device / seed | 0 / 42 | |
| 其他 | 全部 Ultralytics 默认 | 未额外调参 |

## 2. 训练过程

- **耗时**：约 4 小时（epoch 1-40 约 3h15m + 断点续训 epoch 41-50 约 45m）
- **显存**：峰值约 3.4GB / 8GB；GPU 利用率 95-100%
- **中断与恢复**：epoch 40 后进程被会话切换杀掉（非训练问题），清理孤儿进程后 `resume=True` 从 last.pt 无损续训，超参/随机状态完整保留
- **loss 收敛**：train box 1.403→0.641，train cls 2.008→0.371；val box 1.437→0.840，val cls 1.558→0.490。全程**无 NaN**，下降平滑收敛
- **产物**（`runs/e2/labppe_v8s/`）：best.pt、last.pt、results.csv、results.png（loss 曲线）、BoxPR/BoxF1/BoxP/BoxR 曲线、confusion_matrix.png（原始+归一化）、labels.jpg、训练/验证批次可视化

## 3. 数据（训练前确认，未使用 dataset2）

| split | 图片 | mask | gloves | lab_coat | goggles |
|---|---|---|---|---|---|
| train | 6864 | 1789 | 8048 | 1530 | 2850 |
| valid | 369 | 60 | 458 | 124 | 143 |
| test | 168 | 58 | 188 | 42 | 80 |

lab_coat 占 11.5%，少数类但非极端少数类，未做特殊处理（保持默认）。

## 4. 核心指标

**Best epoch = 49**（按 valid mAP50-95 选出）；best.pt 已成功加载并完成单图推理验证（mask conf=0.875 ✓）。

### dataset1 valid（训练过程参考，best.pt）

| 类别 | P | R | F1 | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| mask | 0.843 | 0.900 | 0.871 | 0.898 | 0.600 |
| gloves | 0.936 | 0.910 | 0.923 | 0.946 | 0.748 |
| lab_coat | 0.909 | 0.888 | **0.898** | 0.914 | 0.710 |
| goggles | 0.949 | 0.916 | 0.932 | 0.937 | 0.693 |
| **all** | 0.909 | 0.904 | 0.906 | **0.924** | **0.688** |

### dataset1 test（同分布测试，best.pt）

| 类别 | P | R | F1 | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| mask | 0.871 | 0.879 | 0.875 | 0.838 | 0.549 |
| gloves | 0.935 | 0.819 | 0.873 | 0.922 | 0.693 |
| lab_coat | 0.929 | 0.976 | **0.952** | 0.988 | 0.690 |
| goggles | 0.985 | 0.844 | 0.910 | 0.965 | 0.666 |
| **all** | 0.930 | 0.880 | 0.904 | 0.928 | 0.650 |

> dataset2 = external_test（251 张，零泄漏已锁定），按计划在 E3 作为**主要泛化判断依据**，本阶段未读取。

## 5. 分析

- **最好的类**：valid 上 goggles（F1 0.932 / mAP50-95 0.693）；test 上 lab_coat 表现突出（F1 0.952 / mAP50 0.988）
- **最差的类**：**mask**（valid F1 0.871、mAP50-95 0.600；test mAP50-95 仅 0.549）。P 偏低（valid 0.843）说明存在误报；框定位质量也偏弱（mAP50→mAP50-95 衰减大）
- **lab_coat（重点类）**：valid F1 = 0.898，test F1 = 0.952，**远超 E4 决策门 F1≥0.70**。证明调整1（丢弃 Coverall）后语义纯净的 lab_coat 完全可学
- **过拟合判断**：无明显过拟合——valid 指标在最后 15 个 epoch 持续上升/平稳（epoch 49 达峰，无回落）；train/val loss 差距中等（box 0.641 vs 0.840，cls 0.371 vs 0.490）属正常水平；test 与 valid 指标一致（all mAP50 0.928 vs 0.924），同分布泛化良好
- test 上 gloves/goggles Recall 偏低（0.819/0.844）但 P 很高，类别阈值权衡问题，留待 E3/E4 观察

## 6. E2 完成条件核对

| 条件 | 状态 |
|---|---|
| 训练无 NaN | ✅ |
| loss 正常下降/收敛 | ✅ |
| valid 指标正常输出 | ✅ mAP50 0.924 / mAP50-95 0.688 |
| best.pt 正常生成 | ✅ runs/e2/labppe_v8s/weights/best.pt（epoch 49） |
| best.pt 成功加载推理 | ✅ 单图推理验证通过 |

## 7. 结论

**建议进入 E3**：用 dataset2（external_test）+ t01-t10 视频帧做独立测试，与 SH17 模型对比，重点考察 lab_coat/gloves/mask 的 Recall 与跨数据集泛化能力。特别关注：mask 是当前短板，若 dataset2 上 mask 指标同样偏弱，E4 需权衡是否接受或补数据。
