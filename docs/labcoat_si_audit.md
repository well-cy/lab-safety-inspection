# labcoat_si 数据集入库审计报告

- 审计人：许婧雯（AI 线）
- 日期：2026-09-21
- 数据集：`jeiotech/labcoat_si`（Roboflow Universe，v1）
- 许可证：CC BY 4.0（以项目页为准；数据集 data.yaml 内未标注 license 字段）
- 审计工具：`tools/dedup_new_dataset.py`（文件名 / MD5+SHA256 / pHash≤10 三级去重 + 自身完整性检查）
- 产出文件：`data/labcoat_si_dedup_report.txt`、`data/labcoat_si_leak_images.txt`

## 1. 数据集概况

| 项 | 数值 |
|---|---|
| 图片总数 | 566（train 399 / valid 111 / test 56） |
| 标注框 | 596（lab_coat 543 + face 53） |
| 空标注（故意负样本） | 80 张（含风衣等非实验服图，可降低误检） |
| 缺标注 / 有标注无图 / 越界框 / 内部重复 | 全部为 0 |

类别说明：该数据集只有 `face` 与 `lab_coat` 两类；`face` 与本项目无关，合并时仅取 lab_coat 框（映射到项目 4 类体系的 lab_coat）。

## 2. 去重 / 泄漏审计结果

| 检查项 | 结果 | 结论 |
|---|---|---|
| 与 dataset2（external_test）命中 | **0 张** | ✅ 红线通过，无测试集泄漏 |
| 与 dataset1 完全/近重复 | **0 张** | ✅ 完全互补，无训练集重复 |
| 与 b3 近重复 | **129 张**（距离 0~10） | ⚠ 见第 3 节分析 |
| 自身完全重复 | 0 组 | ✅ |

**结论：剔除 129 张与 b3 同源的图片后，437 张可安全入库。**
（129 张的 split 分布：train 103 / valid 18 / test 8；剔除集含 lab_coat 框 138 个）

## 3. 重要发现：labcoat_si 与 B3 同源，且标注正确

剔除清单中的 129 张与 B3 高度重叠，且文件名呈现明显同源特征
（`014_xxxx_crop_margin_0-1`、`KakaoTalk_*`、`frame_XXXX.png`、`LC*`、`labcoat-<uuid>`）——
**B3 的原始图片来源与 labcoat_si 相同（或来自同一上游图池）。**

关键对比证据（pHash 距离 0 的同一张图 `014_0271_005121_crop_margin_0-1`）：

| 来源 | 标注质量 |
|---|---|
| B3 版本 | lab_coat 框中心 x=0.948、宽 0.899，右边界越出图外约 40%（系统性右偏坏标注） |
| labcoat_si 版本 | 框完整贴合实验服，位置正确（见 `outputs/labcoat_si_b3_overlap_check/` 对比图） |

即：**B3 想提供的图像内容，labcoat_si 已经带着正确标注提供了**。这进一步支持
`docs/B3_data_audit.md` 的结论——放弃修复 B3、转向公开数据集是正确决策，且损失极小。

## 4. 入库方案（待组长确认）

1. **剔除** `data/labcoat_si_leak_images.txt` 中 129 张（与 b3 同源；其中距离 8~10 的多为
   同一视频的相邻帧，信息冗余，实际信息损失有限）
2. **类别转换**：仅保留 lab_coat 框，丢弃 face 框（项目 4 类体系：mask/gloves/lab_coat/goggles）
3. **只并入 train split**：437 张全部进入训练集；dataset1 的 valid/test 与 dataset2
   保持不动，保证 E4 与 E1~E3 的指标口径可比
4. **来源追溯**：合并清单记入 manifest（来源=labcoat_si，保留原始文件名）

## 5. 预期收益

| 数据源 | lab_coat 相关增量 |
|---|---|
| dataset1（7401 张，13→4 类转换后） | 已含 lab coat 类，但真实实验服样本偏少 |
| labcoat_si 入库 437 张 | **lab_coat 框 405 个**（剔除集中另有 138 个随同源图剔除），**负样本 80 张** |
| B3 | 废弃（22/456 可信，见 B3_data_audit.md） |
