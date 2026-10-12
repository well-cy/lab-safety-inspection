# E6 §3 整库 5 类化方案（lab_ppe 四类 → 五类原子手术）

> 2026-10-09 制定。对应 `E6_AI_优化与评测方案.md` §3："训练集 data/lab_ppe（5 类化：person 需补标）"。
> 授权：她选定方向"E6 §3 整库五类化"；test 纳入与否的技术判断见 §4。

## 1. 目标与类别口径

`data/lab_ppe` 从四类升为五类（E6 业务口径）：

| cid | 旧（4 类） | 新（5 类） |
|---|---|---|
| 0 | mask | **person** |
| 1 | gloves | mask |
| 2 | lab_coat | gloves |
| 3 | goggles | lab_coat |
| 4 | — | goggles |

**原子性**：类号迁移与 person 补标必须作为一个整体推进（E6 原文："避免库内类别口径中途混杂"）。中间态（已迁移但 person 未补全）**禁止用于训练**，文档与 commit 均需标注。

## 2. 现状盘点（2026-10-09，commit 6dfa0f9 后）

- 主库 2,154 图 / 5,920 框（train 1,200/4,480 · valid 222/856 · test 732/584），0 坏行 / 0 重名 / 0 孤儿。
- **网图批 person 草稿**：`data/raw/web_lab_scene/labels_5cls/` 123 文件 / 166 person 框
  （r1 47 图 66 框 + r2 19 图 31 框 + up0 21 图 21 框 + r3 36 图 48 框）。
  **校验通过**：123 张全部命中主库（train 105 / valid 18），且其非 person 框与主库标签逐字段一致
  → 该存档含她全部手工修正，**可直接采信**。
- **COCO person 抽样评估**（`tools/survey_person.py`，215 张分层抽样，conf=0.25 imgsz=1280）：
  - 检出率 94% 图 / 1.99 框图，MFZ·digit·up0·webs* 全部 96~100%；
  - **lcsi_crop 有假人模特误检**（橱窗模特、挂拍风衣，conf 0.3±），另有同人多框现象；
  - 结论：COCO 可作预标注底稿，但**必须分级人工兜底**。

## 3. 手术步骤

1. **快照备份**：`data/_label_backup/lab_ppe_5cls_<date>/`（labels 三 split + data.yaml 全量副本，可一键还原）。
2. **类号迁移 +1**（train/valid/test 全库 5,920 框）：`c → c+1`，幂等脚本，写前校验旧值域 0..3。
3. **网图批 person 回补**：对 123 图，直接以 `labels_5cls` 内容为新标签（非 person 框已验证与主库一致）。
4. **存量 person 补标**（2,031 图 = 2,154 − 123）：
   - 底稿：`tools/prelabel_person_all.py` 全库 COCO 预标注 → `data/_prelabel/person_backfill/`；
   - **分级采纳**：
     - `conf ≥ 0.55` 且不与既有框病态重叠 → 自动采纳；
     - `conf < 0.55` → 进人工审清单（核查页按 conf 升序 = 最可疑优先）；
     - 无检出图 → 抽验 10%（漏检兜底）；
     - **lcsi_crop 145 张全审**（已知假人误检）。
5. **data.yaml** → `nc: 5` + 五类 names。
6. **复检**：0 坏行 / 类别对账 / person 覆盖统计 / 跨 split 重名 / 孤儿标签。

## 4. test 是否纳入（技术判断，她可否决）

**纳入**。理由：
- 红线本意是"防训练数据泄漏进测试集 / 不往 test 添加新图"，本次**零新增图、零新增来源**；
- 不迁移则 test 类号错位（0=mask 会被 5 类训练的模型当 person），`lab_ppe/test` 从此不可用；
- 不补 person 则 test 的 person GT 为空，5 类评测在 test 上完全失真；
- 4 类快照完整备份，`data.yaml` 历史版本在 git，可回滚。
- 评测正式基线已在 dataset2（external_test，本次**绝不触碰**）。

## 5. 风险与回滚

| 风险 | 缓解 |
|---|---|
| COCO 误检（假人/海报/屏幕小人） | 分级人工审 + lcsi 全审 |
| COCO 漏检（遮挡/半身） | 无检出图抽验 10% |
| 同人多框 | 预标注 iou=0.45 收紧 NMS + 审核时合并 |
| 中间态被误用于训练 | 文档与 commit 明确标注"迁移完成≠可训，须 person 补标完成" |
| E1~E3 历史指标口径断裂 | 已由 E6 切换 dataset2 作正式基线，历史报告注明 4 类口径 |

**回滚**：`data/_label_backup/lab_ppe_5cls_<date>/` 整体拷回 labels/ 与 data.yaml 即可。

## 6. 交付物

- `tools/migrate_5cls.py`（类号迁移 + 网图批回补，默认干跑）
- `tools/prelabel_person_all.py`（全库 person 预标注）
- 人工审核查页 `outputs/person_backfill/` + 逐图 CSV
- 本文档执行记录小节 + `E4_lab_coat_plan.md` B+14 + `E6_AI_优化与评测方案.md` §3 状态更新

---

## 7. 执行记录（2026-10-09 晚）

### 7.1 类号迁移 + 网图批回补 ✅ 已完成

- 备份：`data/_label_backup/lab_ppe_5cls_20261009.tar.gz`（2,155 文件，md5 抽验一致）。
- 全库 5,920 框类号 +1（1,582 文件断点续跑补齐 + 572 首轮完成）。
- 网图批 123 图以 labels_5cls 整体替换，person 166 框并入。
- data.yaml → nc:5（person/mask/gloves/lab_coat/goggles）。
- 复检：2,154 文件 / **6,086 框** / 0 坏行；分布 person=166 / mask=652 / gloves=2,279 / lab_coat=1,862 / goggles=1,127（与原 4 类分布严格对账）。
- ⚠️ **中间态警告：存量图 person 未补标（仅网图批 166 框），此状态禁止用于训练**——未标注的人会成为 person 负样本毒点，恰恰毒害 E6 要治的"多人场景漏检"。

### 7.2 person 自动预标注校准 ❌ 自动采纳被否决

用网图批 123 图 / 166 person GT（她手工精修）做天然校准集：

| 引擎 | A 档(≥0.55) P | R（总） | 结论 |
|---|---|---|---|
| yolov8n_coco（conf≥0.25, iou=0.45, imgsz=1280） | 85.2% | 84.3% | 不可自动采纳 |
| sh17_yolov8s（person 通道，同参） | 84.4% | 78.9% | 更差 |

目检（`outputs/person_backfill/_calib_fp.jpg` / `_calib_miss.jpg`）：
- "FP"约一半是**框口径差**（模型框部分人体 vs GT 全身框/未标），一半是真误检（试剂盒、仪器被框成 person）；
- **漏检全部是残缺人体**（手部/手臂特写、背面、被仪器遮挡）——这在 PPE 数据集大量存在；
- 致命组合：自动采纳会把"手套/白大褂特写图"的人变成未标注背景 → 毒害 person-PPE 关联。

### 7.3 存量 person 补标 = 人工分批（编辑器流程）

- 工具：复用 `tools/make_box_editor.py`，工作区预载「现有 PPE 框（五类）+ COCO person 候选框」；
- 她逐图确认（删误框/补漏框/确认），单图 10~20 秒，按批导出贴回，我导入落盘；
- 优先级：**P0** train 含 lab_coat/mask/goggles 框的图（person-PPE 关联，~1,100 图）→ **P1** train 其余 → **P2** valid 222 → **P3** test 732（评测 GT，最后）；
- 全部完成后 5 类训练才具备条件。

### 7.4 环境坑（新增）

- 沙箱对**批量创建文件**有阈值（~300-400 触发 SIGTERM）：`shutil.copytree`、shell `cp -r`、tar 解压全中招；**tar 打包（读多写一）安全**；Python 分批 write_text（150/批 + sleep 2.5s）安全。
- 大备份用单 tar 归档 + 流式 md5 抽验，绝不落盘解压。
