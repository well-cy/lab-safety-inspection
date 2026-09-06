# E1 数据准备阶段报告

日期：2026-09-05
状态：**E1 完成，等待确认后进入 E2（训练）**

---

## 1. 数据来源

| 数据集 | Roboflow 项目 | 图片数 | 用途 |
|---|---|---|---|
| dataset1 (LAB_PPE) | vitor-ferraz-marini/lab_ppe-g8hja v1 | 7401 | 训练（4类 PPE 检测） |
| dataset2 (LAB PPE 251) | masters-ppe/lab-ppe | 251 | external_test（锁死，严禁进训练/验证） |

- dataset1 通过 Roboflow SDK 下载（YOLOv8 格式，13 类原始标注）
- dataset2 因项目无发布版本，改用 REST API 逐张下载（251/251 成功），保留其原生 10 类标注于 `data/raw/dataset2/`，不参与训练

## 2. 目录结构

```
data/
├── raw/                      # 原始数据（保留，不动）
│   ├── dataset1/{train,valid,test}/{images,labels}   # 13类原始
│   └── dataset2/{train,valid,test}/{images,labels}   # 10类原始（external_test）
├── lab_ppe/                  # 转换后训练数据集（标准 YOLO）
│   ├── data.yaml
│   ├── manifest.csv          # 17157 行（逐标注框记录）
│   ├── images/{train:6864, valid:369, test:168}
│   └── labels/{train:6864, valid:369, test:168}
├── dedup_report.txt          # 去重检查报告
└── dedup_pairs.csv           # 重复对明细
outputs/label_check/          # 20 张可视化抽检图
tools/                        # 全部脚本（download/convert/dedup/visualize）
```

## 3. data.yaml

```yaml
path: D:/课程相关/大三上/软件工程实训/labsafety/data/lab_ppe
train: images/train
val: images/valid
test: images/test
nc: 4
names: {0: mask, 1: gloves, 2: lab_coat, 3: goggles}
```

## 4. 类别转换（13 类 → 4 类）

**类别 ID 分布（标注框总数 15370）：**

| 新类别 | ID | 标注框数 | train / valid / test |
|---|---|---|---|
| gloves | 1 | 8694 | 8048 / 458 / 188 |
| goggles | 3 | 3073 | 2850 / 143 / 80 |
| lab_coat | 2 | 1696 | 1530 / 124 / 42 |
| mask | 0 | 1907 | 1789 / 60 / 58 |

**保留映射（含脏类名清理）：**
- Mask/mask → 0；Glove/Glove/Gloves → 1；lab coat/Lab coat → 2；Goggles/googles/safety goggles/`safety goggles - v1 2023-08-02 12-26pm`（作者误把版本号敲进类名，语义即 safety goggles）→ 3

**丢弃类（共 1787 框，按调整1执行——Coverall 不进 lab_coat）：**
- Coverall 1156 / Face_Shield 430 / protective head cap 179 / lab shoe 22
- 图片保留，仅丢弃对应标注行；空标注文件 353 个（成为背景负样本，对训练有益）

**多边形处理：** 892 个 YOLO-seg 多边形标注（多为手套类）取外接矩形转 bbox，全部成功转换。

**验证：** 转换后无任何 id≥4 残留（grep 全量扫描为空）。

## 5. 去重检查报告（调整2）

检查 7652 张图（dataset1 全部 7401 + dataset2 全部 251），三项检查结果：

| 检查项 | 结果 |
|---|---|
| 文件名重复 | 0 组 |
| MD5+SHA256 完全重复 | **dataset1↔dataset2：0 对**；dataset1 内部 train↔valid：1 对（同一张高尔夫手套图，作者拆分所致） |
| pHash 近重复（汉明距离≤10，64bit） | **dataset1↔dataset2：0 对** |

**结论：dataset2 与 dataset1 之间零泄漏，251 张图全部独立 → dataset2 正式锁定为 external_test，后续严禁进入训练与验证。**

注意事项：
- dataset1 内部跨 split 近重复 3137 对，绝大多数是同一视频的相邻帧（如 `...mp4-0000~0009`）被作者分到不同 split，属数据集原始拆分特性，无法干预。影响：valid 指标可能略偏乐观，但不影响 dataset2 独立测试的客观性。
- dataset1 内部那 1 对完全重复（train/valid 各一份）建议保持原样（尊重原始数据集划分）。

## 6. 可视化抽检（20 张）

`outputs/label_check/check_01 ~ check_20`，随机抽样，覆盖 4 类共 43 个框：
- mask 5 框（红）/ gloves 26 框（绿）/ lab_coat 7 框（橙）/ goggles 5 框（紫）
- 抽检文件名与 lab_coat/goggles 类别内容相符（如 `check_11_safety_goggles_39`、`check_17_gozluk_215` 护目镜类文件名）
- 说明：本会话的图片读取通道不可用，无法逐张目检图像内容；抽检图已保存，可人工打开 `outputs/label_check/` 目录复核

## 7. E1 交付物清单

- [x] 数据集目录：`data/lab_ppe/`（标准 YOLO 结构，7401 张图）
- [x] `data/lab_ppe/data.yaml`（nc=4）
- [x] `data/lab_ppe/manifest.csv`（17157 行：source_dataset/split/原文件名/新文件名/原类别/新类别/action）
- [x] 去重报告：`data/dedup_report.txt` + `data/dedup_pairs.csv`
- [x] 20 张可视化：`outputs/label_check/`
- [x] 类别统计：见上表
- [x] 原始数据保留：`data/raw/dataset1/`、`data/raw/dataset2/`

## 8. 下一阶段（待确认）

E2：训练 baseline —— YOLOv8s 官方预训练权重，epochs=50，imgsz=640，基于 `data/lab_ppe/data.yaml`。**等待用户确认后才启动。**
