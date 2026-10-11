# E4 前置：lab_coat 定向增训方案设计与数据审计报告

> 状态：**方案设计，未执行任何训练、未修改任何生产代码、未动 dataset2**
> 日期：2026-09-06
> 审计脚本：`tools/audit_labcoat.py`（只读）

---

## A. 数据审计结果

### A1. dataset1 中 lab_coat 的实际构成（训练侧）

| 指标 | train | valid | test |
|---|---|---|---|
| 含 lab_coat 图片 | 1162 | 102 | 42 |
| lab_coat 框数 | 1530 | 124 | 42 |
| 框相对面积 中位数 | 0.251 | 0.303 | 0.346 |
| 每图框数 | 863 张仅 1 框 | 84 张仅 1 框 | 全部 1 框 |

**来源聚类（train 1162 张，按文件名前缀）**：

| 来源批次 | 张数 | 性质推断 |
|---|---|---|
| 纯数字命名（100_jpeg_jpg.rf...） | 358 | 人物摆拍图库 |
| g*（g100_png_jpg...） | 179 | 人物摆拍图库（PNG 转存） |
| M*（M10_jpeg_jpg...） | 166 | 人物摆拍图库（M/F=男/女编号） |
| **istockphoto-*（612x612 网络图）** | 178 | **网络图库照片** |
| w*（w-10-_jpeg...） | 117 | 人物摆拍图库 |
| z*（z10_jpeg...） | 66 | 人物摆拍图库 |
| F*（F10_jpeg...） | 89 | 人物摆拍图库 |
| k*（k1_jpeg...） | 9 | 人物摆拍图库 |

**颜色构成（HSV 代理指标，中心 70% 区域）**：
- dataset1 train lab_coat 框：**白大褂（低饱和+亮）47%**（约 720 框）、彩色/中亮 49%、暗色 4%
- 纯白大褂样本实际只有约 **720 框**，且集中在摆拍图

### A2. dataset2 中 Lab Coat 的域特征（external_test，只读未动）

| 指标 | dataset2 (127 框) | dataset1 train (1530 框) | 差异 |
|---|---|---|---|
| 框内亮度中位 | **134–142** | 175 | dataset2 明显更暗 |
| 白大褂比例 | **34%** | 47% | dataset2 彩色实验服更多 |
| 高亮像素比例中位 | 0.04–0.09 | 0.28 | dataset2 白得更"脏"（光照弱） |
| 框相对面积中位 | 0.219 | 0.251 | dataset2 中远距离 |
| 框相对面积 p75 | 0.267 | 0.385 | dataset1 大量近景特写 |
| 图片性质 | 文件名为视频帧编号（018_0030_021185）+ 抓取 ID（6264...） | 摆拍/网络图 | **dataset2 = 真实实验室视频帧域** |

### A3. 域差异结论（t01/t02/t07 白大褂漏检的根因）

训练域与目标域在**四个维度同时偏移**：

1. **场景**：dataset1 的 lab_coat 近 100% 来自摆拍图库 + istockphoto 网络图，**没有真实实验室监控/新闻现场帧**；dataset2 与 t01/t02/t07 全是真实实验室场景
2. **尺度**：dataset1 以单人近景为主（863/1162 图仅 1 人，框面积中位 25%，最大 86%）；真实场景是中远距离多人
3. **光照**：dataset1 平均亮度 175（打光充足），真实场景 134–142（实验室顶光/背光）
4. **颜色**：纯白大褂仅 47%，训练信号里"白大褂"本体就偏少

**把 conf 降到 0.10 都完全不激活（E3 实测）→ 这是域窄问题，不是阈值问题**。这也解释了 dataset1 同分布 test F1=0.952 与 dataset2 F1=0.34 的巨大落差。

### A4. 标注质量与类别定义

- 标注质量无明显问题：最小框面积 0.001（仅 2 个，可忽略）；892 个多边形已在 E1 统一转外接矩形；无 id 残留
- **类别定义偏窄是真实存在的**：dataset1 的 "lab coat" 语义≈"摆拍图库里的实验服"，未覆盖真实实验室内、中远距离、弱光下的白大褂形态。但**不建议扩类别定义**（不映射 Coverall，遵守调整1），正确做法是**扩充真实场景样本**而不是改定义

---

## B. 建议增加的数据（候选清单，均未加入训练）

| 候选 | 来源 | 许可证 | 数量 | 类别 | 评估 |
|---|---|---|---|---|---|
| **B1 自采集**（推荐主力） | 用手机在实验室/教学楼自拍：不同人×光照×距离×角度 | 自有版权，零风险 | 100–200 张 | 标注 lab_coat（可顺带 mask/gloves/goggles） | **域匹配度最高**，直击真实场景短板；半天采集+标注（Roboflow 免费标注） |
| **B2 Roboflow: chintan/labcoat** | universe.roboflow.com/chintan/labcoat | CC BY 4.0 | 约 2000 张 | LabCoat / No Labcoat（bbox） | 图片为 getty/istock 缩略图与视频帧混杂，**部分带水印、版权存疑**；只取 LabCoat 类，需逐张审计+去重后预计可用几百张 |
| **B3 Roboflow: ez-spn/lab-coat-v8qin** | universe.roboflow.com/ez-spn/lab-coat-v8qin | CC BY 4.0 | 479 张 | Goggles/Gloves/Lab_coat | 类名与 LAB_PPE 家族一致，**疑似同源**——使用前必须先与 dataset1+dataset2 做 MD5/pHash 去重，若与 dataset2 重叠则**弃用**（防止泄漏） |
| ~~SH17~~ | github.com/ahmadmughees/SH17dataset | CC BY-NC-SA 4.0 | 8099 张 | 17 类，**无 lab coat**（只有 Medical-suit/Safety-suit） | ❌ 已核实不适用 |
| ~~CPPE-5~~ | arXiv 2112.09569 | OpenRAIL | ~1000 张 | Coverall/Face_Shield/Gloves/Goggles/Mask，**论文明确排除 lab coat** | ❌ 不适用 |

**最小增训数据集建议**：B1（150 张左右）+ B3 去重通过的子集（预计 300–400 张）+ B2 审计过滤后取 200–300 张，合计目标 **新增 650–850 张、lab_coat 新增框 800–1200**。不为凑数无限扩充——重点是"真实场景占比"而非总量。

---

## B+. 数据准备执行情况（2026-09-21 更新，负责人：许婧雯）

原方案 B/C 节写于 09-06，以下是实际执行结果：

| 原计划 | 实际执行 | 结论 |
|---|---|---|
| B3（ez-spn/lab-coat-v8qin，479 张）去重后使用 | 全量审计发现**系统性标注右偏**：456 张中完全可信仅 22 张（328 张框越界、106 张可疑偏移/漏标），根因为 crop 后坐标未重映射 | ❌ **废弃**，详见 `docs/B3_data_audit.md` |
| （原无此项） | 改用 **labcoat_si**（Roboflow jeiotech，566 张，CC BY 4.0）：与 dataset1/dataset2 **零重复**（MD5+pHash），与 B3 同源的 129 张剔除，**437 张已并入 `data/lab_ppe/train`**（lab_coat(1)→项目 lab_coat(2)，face 框丢弃，文件名加 `lcsi_` 前缀） | ✅ **已入库**，详见 `docs/labcoat_si_audit.md`、`data/labcoat_si_merge_report.txt` |
| B1 自采集 150 张 | **未到位**（组长负责） | ⏳ E4 训练的唯一硬前置 |
| B2（chintan/labcoat，含水印图） | 未启用 | 待决策（见文末） |

**入库后的数据现状（`data/lab_ppe/train`）**：

| 指标 | 合并前 | 合并后 |
|---|---|---|
| train 图片 | 6864 | **7301** |
| lab_coat 框 | 1530 | **1935**（+405） |
| mask / gloves / goggles 框 | 1789 / 8048 / 2850 | 不变（本次只补 lab_coat） |

两点说明：

1. **水印问题**：labcoat_si 部分样本（LC 系列、`Lab_coat_*` 系列）带 CanStock/iStock 图库水印。与风险 3 对 B2 的口径一致——dataset1 本身已含 178 张 istockphoto 网络图，课程项目内部使用可接受，报告中注明来源即可；如需更干净可后续按 `outputs/merged_labcoat_check_all/`（437 张画框图）人工剔除。
2. **域差距仍在**：labcoat_si 以商品图/棚拍/近景为主，**不解决 A3 节的"真实实验室监控视角"域差**——这一缺口只能靠 B1 自采集补，这也是 B1 仍是硬前置的原因。

**E4 训练启动条件**：B1 到位 → 标注 → 对 dataset1/dataset2 去重（复用 `tools/dedup_new_dataset.py`）→ 并入 train → 按方案Ⅰ/Ⅱ开训。

### B+2. Coverall 口径变更与补框（2026-09-28，负责人：许婧雯）

> **⚠️ 本小节的「并入」决定已被 2026-10-08 口径推翻**：她当日明确 **coverall 不收作 lab_coat**（以 E6 §2.1 口径 v2 为准）。
> 已执行清理：`cov_*` 29 图的 35 个 lab_coat 框删除（图保留作负样本，备份 `data/_label_backup/cov_coverall_20261008/`）；
> 当年补进 279 张旧图的 331 个 coverall 框**零残留**——宿主图（Demonstration-of-Donning 等）已随 C2/CCTV 来源域手术全部移出。

E2 模型实测对连体防护服零识别（conf 降到 0.05 仍无输出）。当日用户拍板：**Coverall 并入 lab_coat**（此前为防语义混淆特意剔除，现按真实使用口径放宽）。数据源：`data/raw/ppedetect`（saba/ppedetection v7，CC BY 4.0）train 364 张，其中 329 张含 Coverall 标注。执行（`tools/merge_coverall.py`，报告 `data/coverall_merge_report.txt`）：

| 处理 | 数量 | 说明 |
|---|---|---|
| MD5 同图（早年手套合并已入库） | 300 | 逐框比对确认与 ppedetect 同一套标注，但**缺 Coverall 框**（教模型"穿防护服=没穿"，必须补） |
| └ 其中在 train，补 lab_coat 框 | 279 张 / +331 框 | 逐行精确比对幂等追加；改前原文件备份至 `data/_label_backup_coverall_2026-09-28/` |
| └ 其中在 valid/test | 21 | **一律不碰**（口径可比 + external_test 红线） |
| 全新图片并入 train | 29（`cov_` 前缀） | 映射 Coverall→lab_coat、Gloves→gloves、Goggles→goggles、Mask→mask，Face_Shield 14 框丢弃 |

**合并后 `data/lab_ppe/train`：图片 10812→10841；四类框 mask 1630 / gloves 9331 / lab_coat 2380 / goggles 5199**（lab_coat 净增 366，均为防护服形态）。manifest.csv 已追加 keep/patch/drop 全部记录。

### B+3. ppes_* 家族削减（2026-09-30，负责人：许婧雯）

增广抽检牵出的家族级标签审计（详见 `docs/E5_补数据与重训方案.md` §2.5）：`ppes_*`（源 `data/raw/ppes_kaxsi`）占 train 的 **47%（5,096 张）**，经 30 张随机抽检确认**全部来自同一间仓库的同一台 CCTV 机位**，且源数据只标 glove/goggles 两类——图里的白大褂/口罩一律未标注（=背景），是 lab_coat/mask 短板与"眼镜→护目镜"毒标签的最大来源。

处理（`tools/reduce_family_sample.py`，剔除清单 `data/_ppes_reduce_drop.txt`）：按 9 个抽帧批次等间隔抽样，保留 **1,401 张（27.5%）**，其余 **3,695 张**连同标签移入 `data/_rejected/ppes_reduce_20260930/`（可整体移回）。

**削减后 `data/lab_ppe/train`：图片 10841→7146；四类框 mask 1630 / gloves 7474 / lab_coat 2380 / goggles 3019，合计 14,503**。

> gloves 占比削减后仍达 51.5%（不降反微升），说明手套样本大量来自 `ppes_*` 之外的数据源——类别失衡还需靠"补 mask/lab_coat + 控制 gloves"解决，削减只解决场景单一与领域错配。valid/test 未做任何改动。

### B+4. 来源域分诊图版标注复核（2026-10-07，负责人：许婧雯）

E6 前置手术的分诊图版（`outputs/phase2_provenance/sheet_*`）人工复核修正，逐条均先备份再改（备份 `data/_label_backup/c1_fix_20261007/`）：

| 图版 | 编号 | 判定（她报） | 执行 |
|---|---|---|---|
| C1（~穿戴演示-含白大褂） | 33 | 剔除 | 图+标签移入 `data/_rejected/c1_review_20261007/` |
| C1 | 1/4/22/47 | 框选偏差 | 修框（备份后改标签） |
| C1 | 2/7 等 | 其余确认 | 保留 |
| labcoat_si | 20 | 「没有被框选」 | **误报**：标签里有 lab_coat 框且位置准确（xyxy 120,0,514,638），只是框贴图片上下边缘、缩略图上只剩两条竖线不易察觉。**未改动**，核实图 `outputs/phase2_provenance/c1_fix/verify_1.jpg` |
| labcoat_si | 42（lcsi_LC29） | 是风衣不是白大褂，错误框选 | 删除该 lab_coat 框，标签清空作背景图（图保留）；改前备份同上目录 |
| web_star（实验室-网页图） | 34（web_0196） | 没有穿白大褂 | 核实为蓝色一次性隔离衣 → 删 lab_coat 框，保留 2 个 gloves 框 |
| web_star | 32（web_0170） | 没有穿白大褂 | 核实为蓝色洗手衣 → 删 lab_coat 框（唯一框，标签清空作背景图） |
| web_star | 11（web_p84856588） | 没有穿白大褂 | 核实为白色刷手服 → 删 lab_coat 框，保留 gloves 框 |

**复核后 `data/lab_ppe/train`：标签文件 7,147；四类框 mask 1635 / gloves 7477 / lab_coat 2381 / goggles 2894，合计 14,387**。

> 待办：#42 删框后成为「非实验室时装图背景样本」，按 §2.1 口径 v2（负样本也须为实验室环境）其实不宜留 train，待她定夺是否一并移出。
> **（2026-10-07 22:43 更新：她回「剔」，#42 已移入 `data/_rejected/lcsi_lc29_notlabcoat_20261007/`）**

### B+5. E6 来源域手术与 valid 重建（2026-10-07 22:43，负责人：许婧雯）

她三连拍板「同意手术 / 可以（C1 抽查模式）/ 剔（#42）」后执行完毕，详见 `docs/E6_AI_优化与评测方案.md` §2.0 手术执行进度表：

- 7 个来源域组共 **7,279 组（14,558 文件）**移入 `data/_rejected/domain_audit_20261007_*/`（只移动不删除，分组可回滚）；
- valid 从保留池整家族补足重建（`tools/rebuild_valid_20261007.py`），**train∩valid 家族交集 = 0**；
- **手术后 `data/lab_ppe`：train 1,547 图 / 4,971 框（mask 503 / gloves 1,955 / lab_coat 1,864 / goggles 649，另空标签负样本 135）；
  valid 273 图 / 915 框；test 732 图未动（红线）**；
- lab_coat 现在是保留池第一大类（37.5%），mask 只剩 503——第二阶段补数优先级：**mask ≫ goggles > gloves**。

### B+6. C1 嫌疑图版判定 + coverall 视频整组剔除（2026-10-08，负责人：许婧雯）

C1 续批抽查图版（`outputs/phase2_provenance/c1_suspects/sheet/`，top100）她按编号判定，AI 先逐图核实再执行：

| 编号 | 图 | 判定（她报） | 执行 |
|---|---|---|---|
| 2 | istockphoto-809958214-0012 | 多了个实验服框 | 删重复 lab_coat 框（保留更完整的 1 个） |
| 4 | istockphoto-810129952-0015 | 实验服框过大过小均有 | 3 个 lab_coat 框重画为 2 个（中央人物含举臂袖 + 右侧人物） |
| 7-19 | 8533991 视频帧 13 张 | 剔除 | **按家族跨 split 扩展**：8533991 全组 16 张（train 11 + valid 5）整图移出 |
| 43 | istockphoto-809879018-0025 | 框选过小 | 右侧背景女子 + 左侧男子两个过小 lab_coat 框重画 |
| 44 | istockphoto-810129952-0003 | 框选过大 | 过大 lab_coat 框重画（收窄到主人物含袖） |
| 63/64 | 4267250 视频帧 2 张 | 剔除 | **按家族跨 split 扩展**：4267250 全组 16 张（train 14 + valid 2）整图移出 |
| 80 | F30 | 框选错位 | lab_coat 框重画（含双臂袖，去除左侧背景） |
| 85/86 | F4 两帧 | 框选错位 | 胸前误置小框移到白袖/裙摆；#86 底部条修正到裙摆 |
| 92/93 | g105 两帧 | 框选错位 | lab_coat 框重画（含右/左伸臂袖） |
| 97 | g39 | 无法判断是否戴手套 | 删该 gloves 框（不可验证的标注不留） |

- 剔除落盘 `data/_rejected/coverall_video_20261008/`（32 组 / 64 文件），零残留可回滚；两家族均为连体防护服（coverall）演示视频，与「coverall 不收 + 负样本须实验室场景」口径一致。
- 修框备份 `data/_label_backup/c1_suspects_fix_20261008/train/`；修正后核实图 `outputs/c1_batch_verdict_20261008/after/`。
- **执行后 `data/lab_ppe`：train 1,494 图（空标签 136）/ 4,728 框（mask 462 / gloves 1,874 / lab_coat 1,782 / goggles 610）；valid 262 图 / 870 框（mask 76 / gloves 348 / lab_coat 307 / goggles 136）；test 未动（红线）**。（勘误 2026-10-08 晚：原记 4,727/609 为当时统计误差 -1，已按 B+7 执行前反推修正；valid 分项原记 350/308 亦按 348/307 修正。）

### B+7. 漏标候选复核执行（2026-10-08 晚，负责人：许婧雯；审批至编号 152）

底座漏标复查（21c773f）出的候选清单她完成逐条判定（AI 先用网格放大图逐条核实再执行），覆盖 66 个编号：

**① 整图剔除（4 张）**：#4 web_0196、#5 web_0170（蓝 PPE/蓝色实验服场景）、#52（灰度杂乱图）、#152 istockphoto-1170463165-0005（水印+与前重复）→ `data/_rejected/missing_review_20261008/`（4 组/8 文件，可回滚）。

**② 删已有框（25 个）**：劳保手套冒充 gloves（#18×2 #20×2 #40 #59 #67 #77×2 #92×2 #140 #145×2 = 14）；洁净服/非实验服冒充 lab_coat（#47×3 #53×2 #73 #87，即 8771134 系列连帽洁净服 + web_p155720339 罩衫 + web_0155 袖口误标）；普通眼镜冒充 goggles（#106 #107）；重复框/拆框（#57 lab_coat 重复 1 + gloves 拆两框删 1）。

**③ 修已有框（15 个）**：#17 #24×2 #29 #31×2 #32×2 #56 #57 #61 #91 #102 #104×2（lab_coat 未框全身/口罩护目镜偏移/手套框交叉）。

**④ 补框（11 个）**：#42 左男护目镜（用候选）+ **右女生护目镜（她指出的漏标）** + 2 前景蓝手套；#49 口罩 + 额前护目镜 + 双手乳胶手套；#76 护目镜 + 写字板蓝手套；#18 透明护目镜（候选正确转正）。

**⑤ 候选否决（约 45 项，不改数据只记录）**：眼镜冒充护目镜（#19 #36 #50 #72 #78 #80 #88 #97 #105 #110）、劳保手套候选（#40 #66 #77 等）、框歪/框背景（#11 #23 #34 #35 #39 #43 #44 #46 #47 #84 #85 #87 #114 #116 #118 #119 #146 #151 等）。

**⑥ 口径补充确认**：劳保手套（机加工厚手套）不算 gloves；连帽洁净服（8771134 系列）不算 lab_coat；额头横带式护目镜与透明面罩式护目镜均收。

- 修改备份 `data/_label_backup/missing_verdict_20261008/`（29 文件）；after 核实图 `outputs/missing_labels_20261008/verify_after/`（51 图）；网格定位图 `grid/`；原图查看器 `view.html` + `index_full.csv`（编号↔绝对路径）。

### B+8. 车间系列整批剔除 + 漏标候选收尾（2026-10-08 深夜，负责人：许婧雯）

她审批候选编号 153–192 时发现 **#174 为机加工车间（drill press）场景而非实验室**，拍板"这一系列图都剔除"。AI 全库排查后确认：**纯数字命名家族（`N_jpeg/jpg/png_jpg.rf.*`，共 77 族 / 191 张）中 67 族为同一机加工车间拍摄系列**（钻床/铣床/CNC，白大褂+护目镜，含劳保手套——正是 B+6 确认"不算 gloves"的那类），仅 10 族（编号 1/4/5/6/7/8/9/10/11/12）为真实的显微镜实验室场景。E4 §A 表中该系列原记为"人物摆拍图库"，实拍判定为**车间域**，与本项目实验室域不符。

**① 整批剔除（167 张：train 151 + valid 16，67 族，含 #154/#158/#163/#174/#179 所在家族）**→ `data/_rejected/workshop_series_20261008/`（334 文件，可回滚）。保留 24 张显微镜实验室家族图。

**② 逐条判定执行**：#156/#160/#161 候选否决（lab_coat 候选框错，无数据改动）；#157 两人均穿连体防护服（coverall 不收口径）→ 删误标 lab_coat 框 1 个；#191 候选 goggles 实为边缘半遮挡口罩 → 改类补 mask 框 1 个（下移至可见口罩带区域，after 图 `outputs/missing_labels_20261008/after_191.jpg`）；#152 已在 B+7 剔除。

**③ 她的治理决策（记入 Phase-2）**：图片重复率过高 → 剔除坏图 + **引入新的实验室场景测试集**（详见 `docs/E6_Phase2_采集清单.md` §10）。

- 修改备份 `data/_label_backup/missing_verdict2_20261008/`（2 文件）；执行后全库 0 坏行、图文配对 0 缺失。
- **执行后底座（10-08 深夜）**：**train 1,339 图 / 4,291 框（mask 463 / gloves 1,658 / lab_coat 1,614 / goggles 556）；valid 246 图 / 820 框（mask 76 / gloves 325 / lab_coat 291 / goggles 128）；test 未动**。本轮车间系列剔除 + B+6/B+7 累计使数据量从 10-07 的 train 1,547/valid 273 降至当前值，其中 valid 剔除的 16 张属车间家族（跨 split 同源整组划分口径的落实）。

- **执行后 `data/lab_ppe`：train 1,490 图 / 4,702 框（mask 462 / gloves 1,859 / lab_coat 1,771 / goggles 610）；valid 262 图 / 867 框（mask 76 / gloves 348 / lab_coat 307 / goggles 136）；test 未动。**
- **事故记录**：首次写回时标签行丢失类别位（85 行变 4 字段），当轮发现 → 从备份恢复 29 文件 → 修正 `tools/apply_missing_verdict.py` 写回格式后重放 → 全库 0 坏行，逐 split 对账精确（教训：批量写标签后必须跑格式校验+框数对账，不能只对行数）。
- **待她决策**：① 候选图内 pHash 查重出 **17 个近重复组**（如 74/90/153/162/182/190 六图一组、147/177/180/181 四图一组，图版外 121-192 编号占多数，验证了她"后面大部分有重复"的观察）——等她定保留策略；② #151 疑似人体模特图是否剔除；③ #91 右下角桌上黑厚手套（old0 gloves）疑似劳保；④ 剩余编号 153-192 未审批。
- 新工具：`tools/make_candidate_viewer.py`（候选→HTML 查看器+路径清单）、`make_verify_sheet.py`（逐条核实图版）、`make_grid_view.py`（网格定位图）、`apply_missing_verdict.py`（判定落地，坐标匹配+备份+幂等）。

### B+9. 网图（小红书真实实验室场景）48 张入库（2026-10-08 深夜，负责人：许婧雯）

承接 `docs/E6_AI_优化与评测方案.md` 第二阶段"补真实实验室场景"目标：本批为**小红书收集的 48 张真实实验室/检验科实拍图**（非图库摆拍、非棚拍），来源与编号见 `outputs/web_lab_scene_review/index.txt`，原始 55 张经她分诊剔除 6 张残图 + 1 张（#41 病理包埋场景）后定稿。

**① 预标注三轮修框（她逐张抽查回报，AI 执行）**：`tools/apply_webscene_fix1.py`（DEL30/SET32/改类3/ADD45）、`fix2.py`（DEL8/SET2/ADD15，另剔 #41）、`fix3.py`（DEL2/SET3/ADD3）。核实图 49→48 张（`outputs/web_lab_scene_review/grid_fixed/`，`make_webscene_grid.py --no-grid`），抽查页 `verify.html`（`make_webscene_verify.py`）。她的口径确认：**普通框架眼镜不标 goggles**（#21/#22/#28/#33/#34/#36/#44 全部删）、**劳保/台面杂物不标 gloves**（#4/#33）、**一框一人**（#1/#3/#13/#23/#35/#47 拆分或合并）。备份 `data/_label_backup/webscene_fix{1,2,3}_20261008/`。

**② 四关审计（`tools/audit_webscene_batch.py`，报告 `data/webscene_batch_audit.txt`）**：
- ① 红线去重：与 dataset1（7,401 张）、dataset2（红线）、lab_ppe（2,317 张）pHash 比对，**命中 ≤8 为 0**（最近邻 12–14，无泄漏）；
- ② 批内近重复：48 张两两 pHash 全 > 8，**无重复组**；
- ③ 脏图筛查（`screen_dirty_graphics.py --dir`）：47 ok，1 张 `grayscale_like`（#22 朱师兄_2，暗光实拍照，非脏图，保留）；
- ④ 标注几何：48/48 有标签，223 框，**0 越界/0 超小/0 超大/0 怪比例/0 重复框**。
- 说明：`crosslib_phash.py` 的 `list_images` 会跳过 `_` 开头文件名，本批恰好全部以 `_` 或中文开头，**不能直接复用**，故单独实现列图。

**③ 入库执行（`tools/import_webscene_batch.py`）**：
- **类别口径**：主库当前为**四类**（person 未纳入），入库时 person 框丢弃、四类左移（mask→0 / gloves→1 / lab_coat→2 / goggles→3）；**含 person 的 5 类完整草稿另存 `data/raw/web_lab_scene/labels_5cls/`**，待主库做 5 类化（person 补标）时按 stem 一键回补，信息不丢；
- **划分**：按**系列（同一博主同一篇笔记）整组进同一 split** 防泄漏，小系列优先给 valid（6 个系列 / 7 张），**不进 test**；
- **命名**：中文/emoji 原名改 ASCII `webs_<编号>_<md5[:8]>.jpg`，原名/series/split 落 `data/raw/web_lab_scene/import_manifest.csv`；与主库撞名 0；
- 预留 person 框 66 个（未入库，在 5 类存档内）。

- **执行后底座（B+9 落地）**：**train 1,380 图 / 4,417 框（mask 474 / gloves 1,714 / lab_coat 1,662 / goggles 567）；valid 253 图 / 851 框（mask 84 / gloves 336 / lab_coat 303 / goggles 128）；test 732 图未动**。全库 2,365 图 / 5,852 框，0 坏行、0 孤儿标签；新入库 `webs_*` 在 `tools/audit_lab_ppe_boxes.py` 全库复检中**零 flags**（flags 目录内所有命中均属入库前既有来源）。
- 批次增强：mask +19、gloves +67、lab_coat +60、goggles +11（四类），外加 66 个 person 框存档待 5 类化。

### B+10. 网图 r2 批次（stock 视频截图）19 张入库（2026-10-09，负责人：许婧雯）

承接 B+9 同一目标：本批为 **19 张 stock 视频截图**（真实实验室场景，PNG 转 JPG，源归档 `data/raw/web_lab_scene/_src_screenshots_r2/`），编号见 `outputs/web_lab_scene_review_r2/index.txt`。

**① 预标注修框（AI 三轮 + 她手工框选定稿）**：`tools/apply_webscene_fix_r2.py`（DEL18/SET14/CLS1/ADD26，含 #9 裁底部 22 行视频播放条并重缩放标签 y）→ `fix_r2b.py`（SET4/ADD7；**口径拍板：护目镜实际戴在眼上、白口罩要标**，推翻 AI "未佩戴不标" 的误判）→ `fix_r2c.py`（SET4，#12 手套框改用 HSV 蓝色阈值分割实测坐标）。她仍不满意 9/12/13 后，**交付手工框选页**（`tools/make_box_editor.py` → `outputs/web_lab_scene_review_r2/editor.html`，离线自包含：拖框/缩放/增删/切类/导出标签），**#9/#12/#13 由她亲自拖框定稿**，`tools/import_manual_boxes.py` 落盘（带备份 `data/_label_backup/webscene_r2_manual_20261009/`）。定稿 19 图 / 137 框（person 31 / mask 20 / gloves 39 / lab_coat 20 / goggles 27），0 坏行。

**② 四关审计（`tools/audit_webscene_batch.py --batch r2`，报告 `data/webscene_batch_audit_r2.txt`）**：① 红线去重 0 命中（dataset1 7,401 / dataset2 251 / lab_ppe 2,365，最近邻 12–14）；② 批内近重复 0（19 张两两 pHash 均 >8，**故按张独立划分不构成同源泄漏**）；③ 脏图 19/19 ok；④ 几何 137 框 0 问题。

**③ 入库执行（`tools/import_webscene_batch.py --batch r2`，已参数化支持批次）**：person 框 31 个丢弃（5 类完整草稿照例存档 `labels_5cls/`，累计待 5 类化回补）；**train +16 / valid +3**（#1/#2/#3，每张自成系列按 ~15% 比例划分），不进 test；命名 `websr2_<编号>_<md5[:8]>.jpg`，溯源 `import_manifest_r2.csv`，撞名 0。

- **执行后底座（B+10 落地）**：**train 1,396 图 / 4,510 框（mask 493 / gloves 1,748 / lab_coat 1,679 / goggles 590）；valid 256 图 / 864 框（mask 85 / gloves 341 / lab_coat 306 / goggles 132）；test 732 图未动**。全库 **2,384 图 / 5,958 框**，0 坏行、0 孤儿标签。
- 批次增强：mask +20、gloves +39、lab_coat +20、goggles +27（四类共 106 框），外加 31 个 person 框存档。

### B+11. uploads 第 0 批 95→21 张入库（2026-10-09，负责人：许婧雯）

upload 原始 95 张，2026-10-08 分诊保留 43 张（草稿 `data/_prelabel/uploads_20261008/labels/`，剔除 52 张存 `data/_rejected/uploads_reject_20261008/`），本日完成入库前审计与入库。

**① 四关审计（`tools/audit_webscene_batch.py --src data/test/uploads --lbl ... --index ... --tag uploads0`，报告 `data/webscene_batch_audit_uploads0{,b,c}.txt`）**：
- **⚠️ 首轮即爆雷**：43 张中 **14 张与主库 pHash dist=0/2（像素级同图，13 张撞 `images/test/`、1 张撞 valid）** —— uploads 本质含 dataset1 的重命名子集，直接入库即 **train/test 泄漏**，剔 14 张。
- 另剔 5 张域/质量不符（影棚灰底、蓝底摆拍、白底电商产品图 ×2、暗竖屏）。
- **复检又补剔 3 张**（首轮 `tail` 截断致漏看 dataset1 段）：`img_8fbb49f6`（与 dataset1 92 张 `screenshot*` 系列互 dist≤8）、`img_8fda114e` / `img_f34aa3f7`（dist=0 同图）。
- 她拍板「全剔」，共 **22 张**移入 `data/_rejected/uploads_20261008_final/`。**三检（`uploads0c`）全过**：dataset1 / 红线 / 主库 0 命中、批内 0 重复、脏图 0、几何 0 问题（原第④关 2 个超大 person 框 #37/#41 恰在被剔名单内）。
- **教训固化**：外部批次入库前**必须先跑第①关对 dataset1/红线/主库三库去重**，且审计报告**必须读完整文件**（不能只看 tail）。

**② 入库执行（`tools/import_webscene_batch.py --src data/test/uploads --prefix up0`）**：剩 **21 图 / 61 框**（person 21 / lab_coat 20 / gloves 13 / mask 3 / goggles 4）→ person 21 框丢弃（5 类草稿存档 `labels_5cls/`），**train +18 / valid +3**（每张自成系列按 ~15% 划分），不进 test；命名 `up0_<编号>_<md5[:8]>.jpg`，溯源 `data/raw/uploads0/import_manifest.csv`。

- **执行后底座（B+11 落地）**：**train 1,414 图 / 4,543 框（mask 496 / gloves 1,757 / lab_coat 1,696 / goggles 594）；valid 259 图 / 871 框（mask 85 / gloves 345 / lab_coat 309 / goggles 132）；test 732 图未动（584 框）**。全库 **2,405 图 / 5,998 框**，0 坏行、0 跨 split 重名、0 孤儿标签。
- 批次增强：mask +3、gloves +13、lab_coat +20、goggles +4（四类共 40 框），外加 21 个 person 框存档。
- 工具增强：`import_webscene_batch.py` 增 `--src/--lbl/--index/--arch/--manifest/--prefix`（可入任意批次）+ 写 manifest 前自动建父目录。

### B+12. 网图 r3 批次（40 张收图 → 35 张入库，全人工标注）（2026-10-09，负责人：许婧雯）

她放入 `data/raw/web_lab_scene/images/`：**17 张 unsplash 高清图**（NIH / NCI / NIAID 官方图，2811~9032px，另含 kelly-chiang、diana-hauan、hush-naidoo、muhammad-syafi 等摄影师图）+ `images/Screenshots/` **23 张 png 截图**（~500×350）。`tools/stage_webscene_r3.py` 收进 `images_r3/`（PNG→JPG q92、ASCII 改名 `r3u_NN_slug` / `r3s_NN_hhmmss`，**复制不移动**，溯源 `stage_manifest_r3.csv`）。

**① 分诊剔除**：#6（影棚纯灰背景人像）/ #18（人像棚拍风）/ #19（黑白照）→ 她拍板剔除；修框轮 1 又剔 #4（错误过多）。**编号保留留洞**（1-5、7-17、20-40）。

**② 预标注失败 → 全转人工**：`prelabel_generic.py --tag webscene_r3` 出 40 图/191 框（person62/mask10/gloves61/lab_coat38/goggles20），但质量差（模型=labppe_v8s + yolov8n 轻量半成品权重，域外+500px 小目标基本看不见）。两轮 AI 修框（`apply_webscene_fix_r3a/r3b/r3c.py`）后她判定"错误太多"，**拍板 36 张全部人工框选**。`make_box_editor.py` 为此升级：`--max-side` 自动生成 2400px 预览缩图（9000px 原图不再卡浏览器，坐标仍按原图归一化）、上/下一张按钮 + ←→ 快捷键、下拉显示框数。她用 `editor.html` 一次交付 **36 张 / 201 框**，`import_manual_boxes.py --batch r3` 落盘（备份 `data/_label_backup/webscene_r3_manual_20261009/`）。**口径确认：38/39 的普通眼镜不算 goggles，不标**（维持既有口径，避免污染）。

**③ 四关审计（`tools/audit_webscene_batch.py --tag r3`，报告 `data/webscene_batch_audit_r3.txt`）**：
- ① dataset1（7,401）/ dataset2 红线（251）**0 命中**；主库命中 **1 对：#14 r3s_14_130938 ↔ train/lcsi_27 dist=6**（同一张 stock 照片，持写字板蓝手套男生）。
- ② 批内近重复 **1 对：#7 r3s_07_130425 ↔ #27 r3u_04_chiang-f（dist=0，同一照片的 513px 截图 vs 7360px 原图）**。
- ③ 脏图 35/35 ok；④ 几何 196 框 0 问题。
- **她拍板方案 B**：剔 #7（保留高清 #27，标注相同）；**保留 #14**（标注更全：2 person + 2 lab_coat + gloves + goggles，优于 lcsi 单类版），把 `lcsi_27_jpg.rf.f79da049…` 记入 lcsi 专项审计待剔清单（`outputs/lcsi_audit_pending.txt`）。#7 图+标签移入 `data/_rejected/web_lab_scene_r3_20261009/`。

**④ 入库执行（`tools/import_webscene_batch.py --prefix websr3`）**：剩 **35 图 / 196 框**（person 48 / mask 12 / gloves 56 / lab_coat 44 / goggles 36）→ person 48 框丢弃（5 类草稿存档 `labels_5cls/`）；**train +30 / valid +5**，不进 test；命名 `websr3_<编号>_<md5[:8]>.jpg`，溯源 `import_manifest_r3.csv`，撞名 0。
- **同源分组**：r3 新增 `series_of` 分支——截图 `r3s_*` 按张独立（审计②证明 0 近重复）；高清 `r3u_*` 按摄影师/机构 slug 归组（`chiang-8`/`chiang-f` 同组），组内整组同 split 防泄漏。
- **划分算法改进**：valid 选择改为**按来源族轮转**（此前按字母序取前 N，r3 会导致 valid 全是 500px 截图）；现在 valid = 3 截图 + 2 高清，跨两个域。r2/uploads 回归无变化。

- **执行后底座（B+12 落地）**：**train 1,444 图 / 4,674 框（mask 506 / gloves 1,805 / lab_coat 1,736 / goggles 627）；valid 264 图 / 888 框（mask 87 / gloves 353 / lab_coat 313 / goggles 135）；test 732 图 / 584 框未动**。全库 **2,440 图 / 6,146 框**，0 坏行、0 跨 split 重名、0 孤儿标签。
- 批次增强：mask +12、gloves +56、lab_coat +44、goggles +36（四类共 148 框），外加 48 个 person 框存档。

### B++. 其余三类标注审计结果（2026-09-21，`tools/audit_lab_ppe_boxes.py`）

对 `data/lab_ppe` 全部四类做几何审计（越界/超小/超大/极端宽高比/重复框/非法值），并按类别抽检画框：

| 类别 | 框数 | 越界 | 超小 | 超大 | 怪比例 | 重复 | 几何结论 |
|---|---|---|---|---|---|---|---|
| mask | 1907 | 0 | 7 | 0 | 0 | 0 | 干净 |
| gloves | 8694 | 0 | 17 | 135 | 10 | 5 | 几何干净 |
| lab_coat | 2101 | 0 | 0 | 3 | 6 | 4 | 干净 |
| **goggles** | 3073 | 0 | 1 | 24 | 3 | 4 | 几何干净，**但有类别污染（见下）** |

**几何层面全部干净（0 越界框）**，与 B3 形成鲜明对比。真正的问题在**域与类别定义**：

1. **goggles 类别污染（重点）**：train 2204 张含 goggles 图中，**918 张（42%）来自 `gozluk_*` 批次**（土耳其语"眼镜"）。抽检显示该批次混杂：部分是防冲击护目镜商品图（正确），**部分是普通时尚眼镜/眼镜广告图被标成 goggles（错误，如 `gozluk_062` Ray-Ban 广告）**。这与 E2 在 dataset2 上 goggles F1 仅 32.7 的现象吻合——模型被教了"普通眼镜≈护目镜"。处置：全量画框图在 `outputs/audit_gozluk_all/`（918 张），需人工筛查误标图；若污染率高（>30%）建议整批剔除 gozluk 批次（goggles 剩约 1286 张框，量仍充足），E4 训练前完成。
2. **gloves 人工复查与剔除（2026-09-21 完成）**：对几何可疑的 165 张图拼版人工复查（`tools/make_contact_sheet.py` 拼版 + 编号回报），判定标准：实验室场景 / 一次性丁腈·乳胶手套 = 合格；JUBA 工作手套、园艺手套、皮手套、婚礼手套等**商品特写图及裸手误标 = 不合格**。共剔除 **142 张图**（人工判定 107 + 同原图家族扩展副本 35，来自 33 个原始照片），gloves 框 8694→8537（-1.8%），连带 lab_coat -5、goggles -3 框。剔除文件**移动**至 `data/_rejected/gloves_review_20260921/`（可整体移回恢复），清单见 `data/gloves_review_remove_report.txt`。复核发现：此前记录的"超大框多为正常特写"结论有误——其中相当部分是**裸手特写被误标为 gloves**（正是 E2 手套误检的毒数据）。剔除后 train 7165 / valid 364 / test 167，图文一一对应已校验。
3. **gloves 批次级清理（2026-09-22 完成）**：几何标记只命中坏批次零头，升级为按"原始照片家族/批次"清点后锁定 5 个可疑批次（`outputs/gloves_triage/01~05`），人工逐批复查结论：**01 screenshot*（721 张）、02 数字IMG_（285 张，实锤裸手/手模误标）、03 IMG-2022\*WA\*（351 张）、04 images-2022\*（134 张）整批不合格；05 十三位时间戳批次部分合格**（保留 4-7、9-28、32 号丁腈/乳胶一次性手套，剔除家用橡胶/毛线/骑行手套及裸手共 14 张）。两轮共剔 **1505 张**，移至 `data/_rejected/gloves_batch0104_20260922/` 与 `gloves_ts13_20260922/`，报告见 `data/gloves_batch0104_remove_report.txt`、`data/gloves_ts13_remove_report.txt`。剔除后 **train 5799 / valid 262 / test 130（共 6191 张）**，四类框数 mask 1907 / gloves 6153 / lab_coat 2096 / goggles 3070，图文一一对应已校验。gloves 框 8694→6153（累计 -29%），剩余均为真实佩戴场景。
4. **新增手套公开数据集（2026-09-22 已入库）**：① `safety-gloves-xbnf8` v2（Roboflow 官方，3373 张，Gloves/NO-Gloves，CC BY 4.0）：去重审计零重复、与 dataset2 红线零命中，全量入库；② `ppes-kaxsi` v7（11978 张，10 类含 no_glove 负类，CC BY 4.0）：剔除 5 张近重复后入库 11973。入库映射（`tools/merge_glove_datasets.py`，带 `sgv_`/`ppes_` 来源前缀，幂等可重跑）：Gloves/glove→gloves(1)；同图内 goggles→goggles(3)、mask→mask(0) 一并映射（避免无标注正样本压制自有类别）；NO-Gloves/no_glove 等 no_* 及 helmet/shoes 框丢弃、图保留为**背景负样本**（治裸手误检，共 7751 张，占 36%）。入库后经同源去重（safety-gloves 实为 ppes 源数据的子集，95% 家族重名，剔除 3217 张 sgv 副本保留 ppes 版，见 `data/sgv_dedup_remove_report.txt`）与人工抽查确认（`outputs/gloves_check/`），**最终 train 12553 / valid 3867 / test 1900，共 18320 张**；四类框 mask 2176 / gloves 10816 / lab_coat 2096 / goggles 7251，背景负样本 6646（36%），图文一一对应校验通过，越界框 0。工厂 CCTV 域差异已通过来源前缀保留消融实验可能；报告见 `data/safety_gloves_dedup_report.txt`、`data/ppes_kaxsi_dedup_report.txt`，抽检图版见 `outputs/safety_gloves_review/`、`outputs/ppes_review/`。
5. **lab_coat 与 mask 收尾清理（2026-09-22 完成）**：lab_coat 几何复查（`outputs/labcoat_review/flags_v2/`）判定 **1 号（g50 框过大）、4 号（g103 框过小）不合格**，剔 **4 张**（`data/_rejected/labcoat_bad_20260922/`）。mask 改为「先自动去脏、再人工审」：用图像特征筛查（`tools/screen_dirty_graphics.py`：白底占比/主色调占比/低边缘密度平坦区）从 1719 张筛出 249 张可疑，其中**白底目录图 90 张高置信已剔**（`data/_rejected/mask_dirty_white_20260922/`），纯色棚拍 155 张保留待复核。本步后数据集 **18170 张**（原文档"mask 无需处理"的结论已由本条取代）。
6. **全库 pHash 近重复去重（2026-09-22 完成，方案 B）**：人工抽检发现"**大量重复样本但框选正确**"——同一视频连帧/同机位多拍被反复计入，且存在 **586 组跨 split 泄漏**（同图同时落在 train 与 valid/test）。新增 `tools/dedup_phash.py` 全库扫描：64 位 DCT pHash + LSH 候选分桶 + **严格代表聚类**（只与已保留代表比对，杜绝 A~B~C 链式误并）+ **标注感知二次分桶**（同组内须框 IoU≥0.85 且匹配率≥0.9 才算真冗余，避免"同机位但人已位移"的相邻帧被误杀）。扫描 18170 张得 2416 组 / 10156 张卷入。**采用方案 B：每组最多保留 3 张**，剔除 **4619 张** 至 `data/_rejected/phash_dedup_20260922/`（含 3982 张无框背景连帧），报告 `data/phash_dedup_remove_report.txt`，清单 `data/index_phash_dedup_k3.txt`。因代表聚类是贪心式的，剔掉一批后剩余样本的邻接关系会重排，故重扫收敛：**第二轮再剔 64 张、第三轮 13 张**（`data/_rejected/phash_dedup_r2/r3_20260922/`），逐轮 64→13 递减，已到噪声量级，判定收敛。**去重后数据集 13474 张**（原 18320 → 剔 lab_coat 4 + mask 90 + 去重 4696）：四类框 mask 1974 / gloves 10601 / lab_coat 2086 / goggles 6589（共 21250），背景负样本 2593（19.2%，去重前 6646 张里冗余背景占 61%），孤立图片 0，图文一一对应已校验。
   **⚠️ 未完全消除的残留风险——跨 split 同源**：方案 B 有意每组留 3 张，这些互为近重复的图仍可能分处 train/valid/test。首轮扫描时跨 split 组 586 组，末轮仍有 **390 组 / 900 张**（绝大多数是同一视频的相邻连帧，如 `ppes_helmet*`、`ppes_frame*`）。**若 E4 训练要求严格无泄漏，正确做法不是在图片粒度继续剔，而是在划分阶段按「原图家族／近重复连通分量」整组划分**（同一连通分量整体进 train 或整体进 valid/test）。当前 split 沿用原有划分，此项留待 C1 划分环节处理，需在 E4 报告中如实标注。
7. **mask 类专项收紧（2026-09-22，方案 B 后的定向补刀）**：人工复查 `outputs/mask_review/` 仍觉 mask 观感重复，分两层诊断——① **来源层**：含 mask 框的 1522 张图来自 **1273 个原图家族**，平均 **1.20 张/家族**、1145 个（90%）家族仅 1 张，**家族级重复已清干净**；② **内容层**：pHash 仍剩 124 组（全为 2–3 张小组），典型是**同一视频连帧**（如 `4121322-uhd_3840_2160_25fps_mp4-0019/0020/0021` 三帧连续、且跨 train/valid）。根因是方案 B 允许每组留 3 张，连帧因此幸存。处置：**对 mask 类单独收紧为 keep=1**，再剔 **168 张**（全部带框，非背景负样本）至 `data/_rejected/mask_dedup_k1_20260922/`，复扫 **重复组 0 / 跨 split 泄漏 0**，完全收敛。另剔超小框复查判不合格的 2 个家族 4 张（`data/_rejected/mask_bad_20260922/`：26_png、w-12-）。**mask 现状 1354 张图 / 1799 框**，为四类中最薄，需外部补充。
8. **mask 补充数据候选（2026-09-22 调研，未下载）**：mask 是当前唯一"量偏少且域单一"的类别。入库前必须过三关：**与 dataset1/dataset2 红线去重 → `tools/dedup_phash.py` 同源检测 → 水印/图库图筛查**（B3 的 crop 坐标未重映射事故为前车之鉴）。

| 候选 | 量 | 类 | 许可 | 域匹配 | 备注 |
|---|---|---|---|---|---|
| `epidetect/ppe-wlllw` | 1.8k | HELMET/MASK/GLASS/VEST/GLOVE/PERSON/BOOTS | CC BY 4.0 | 高（工业安全） | 独立来源，与 ppes 系无同源嫌疑，**首选** |
| `saba/ppedetection` | 364 | Coverall/Face_Shield/Gloves/Goggles/Mask | CC BY 4.0 | 高（PPE 齐全） | 量小但类映射度最佳，可与 coverall 一并补 |
| Kaggle `aiotthien/personal-protection-equipment-datasets` | 8086 | 16 类含 Face-mask | 页面标 MIT，源 Roboflow 标 Private | 中高 | 类定义（Face-mask/Glasses/Gloves/Safety-suit/Person + 暴露部位）与本项目极契合，**许可存疑，须先核实** |
| `maskfacedatasetpublic/maskfacedataset-mswcx` | 5.4k | mask/face/incorrect | CC BY 4.0 | 低（公共场合人脸特写） | 量最大、含"不规范佩戴"类；域差大，宜作**口罩外观多样性**补充而非主力 |
| `personal-protective-equipment/ppes-kaxsi` v8 | 24924 | 同 v7 | CC BY 4.0 | 高 | **与已入库 v7 同源**（v7 的 11978 张已入库 11973），仅在有新增独立帧时取用，防泄漏 |
| B1 组长自采 | 150 | 全类 | 自有 | **最高** | 唯一能治"实验室域缺失"的选项，仍是硬前置 |

> 判断：外部 PPE 数据集普遍是"工厂/工地"域，与实验台场景仍有距离，**引入外部的边际收益低于把 B1 自采做扎实**；上表宜作补充而非替代。

9. **外部 mask 数据集引入线结案（2026-09-23，结论：全部否决，不入库）**：实际下载两个候选并跑完审计，**双双证伪**：

   - **`saba/ppedetection` v7（364 张）**：`tools/dedup_new_dataset.py` 审计显示 **362/364 与 dataset1 MD5 级完全重复**（pHash 距离 0）——本质是 dataset1 的派生发布版，**同源，否决**。报告 `data/ppedetect_dedup_report.txt`。
   - **`epidetect/ppe-wlllw` v1（1882 张 / 19 类 / 8981 框）**：三关过后剩 1340 张（同源 304、内部近重复 237、宽阈值 1），但**第 4 关"标注一致性"全线崩坏，否决**：
     - **同源多副本标注冲突**：317 个多副本家族中 **233 个（73%）手套标注不一致**（同一张源图，一份标了 GLOVE、另一份未标；`47_JPG` 甚至一份 GLOVE+MASK、一份全空），mask 不一致 87 个。入库即等于"同图既是正样本又是背景负样本"。
     - **相邻帧标注翻转率**：手套 Video1 **38%** / Video2 **47%** / Video3 **56%** / Video4 **60%** / image 系列 **43%**；mask Video1 **40%**。工人全程佩戴，标注者只在约半数帧画框。
     - **类别定义不符**：screenshot 系列 33 张 100% 为**针织劳保线手套**（黑底商品特写 + 人物比划），按项目手套判据（实验室一次性丁腈/乳胶）不合格；该数据集整体为工地/工厂（安全帽）域，mask 框多落在远景头部，口罩本身存疑。
   - 处置：**两数据集均不入库**；`data/raw/ppe_wlllw/` 保留原始素材并加 `_REJECTED_不入库.md` 标记，审计清单与报告全部留档，否决报告见 `data/ppe_wlllw_reject_report.txt`。
   - **新增第 4 关（此后任何外部数据入库前必查）**：**标注一致性审计**——① 同源副本的标注一致率；② 同一视频/连拍系列相邻帧的标注翻转率；③ 类别的域匹配（如手套须为实验室一次性手套）。前三关（红线去重 / pHash 同源 / 脏图筛查）**只能抓图片级问题，抓不到标签级毒数据**；`ppes-kaxsi`、`safety-gloves` 入库时未过此关，属**已知残留风险**，后续抽检需留意。
   - **结论：外部引入线就此收束。** mask / gloves 的补强主力仍为 B1 自采 150 张——唯一能治"实验室域缺失"的数据源（E3 已证 lab_coat 跨域 Recall 仅 20.5%）。

10. **B1 自采集执行方案（2026-09-23，去阻塞）**：外部线结案后 B1 成为唯一活路，而它已挂"未到位"逾两周。复核发现 **B1 根本不是"等人给"的依赖**，而是**一个可以立刻执行的技术任务**：

   - **性质澄清**：B1 的原始定义（本文 §B 表）是「**用手机在实验室/教学楼自拍**，100–200 张」，**自有版权零风险**。它被标注为"组长负责"是**分工标签，不是技术门槛**——手机 + 白大褂 + 实验室，本项目任何成员都能完成。
   - **唯一的真实门槛是"人"**：需 ≥3 人出镜（理想 5 人）以覆盖多样化要求，需拉同学配合；其余全是流程。
   - **已产出**：`docs/B1_采集SOP.md` —— 一页纸拍摄规范（拍摄矩阵 5 人×3 距离×3 光照×3 角度、场景要求、手机端技术规范、命名与交付、隐私合规、采集偏差硬指标）。**任何成员拿手机照此执行即可，半天完成。**
   - **已备工具链**（零依赖，`--apply` 前先干跑）：
     - `tools/prep_b1_photos.py` —— 手机照片预处理：**EXIF 方向校正**（不做则画框整体旋转 90°，标注全废）＋ **HEIC→JPG**（iPhone 默认格式，OpenCV 读不了）＋ 统一 ASCII 重命名 `b1_0001.jpg`（规避项目已知中文路径坑）＋ 按拍摄时间排序 ＋ 输出 `rename_map.csv` 映射表。
     - `tools/dedup_new_dataset.py` —— **T2-1 原计划新写的 `audit_b1.py` 不必再写**：该工具已是通用的入库前三向审计（B1↔dataset1 / **dataset2（铁律零泄漏）** / labcoat_si），用法 `--src data/raw/b1 --name b1 --refs dataset1,dataset2,labcoat_si`（本轮已用它审掉 ppedetect 与 ppe-wlllw 两个候选）。
     - `tools/make_contact_sheet.py --box-color` —— 出抽检拼版供人工复核。
   - **前置障碍（需向组长索要）**：本地**没有任何模型权重**（`*.pt` 全被 `.gitignore` 排除，经 `FINAL_MODEL.zip`/`AI_TRAINING.zip` 单独分发，`ultralytics` 亦未安装），因此"模型预标注供人工校正"目前**做不到**。建议向组长索取 `runs/e2/labppe_v8s/weights/best.pt`——既解锁预标注（校正比从零画框快 5–10 倍），也顺带验证权重分发渠道是否通畅。否则标注退回 Roboflow 在线手画。
   - **流程合规**：`PROJECT_STATUS.md`「B1 未到位不得启动训练（**组长约束**）」与 `E4_data_prep_report.md`「不伪造、**不替代**」——后者指**不得用公开数据集冒充 B1**，**自采并不构成"替代"**（自采即"到位"）。惟文档将 B1 绑定于"组长负责"，最稳做法是**先与组长确认一句**（"B1 由我组织自采"），避免流程争议。
   - **降级预案（同文档既有的解套机制）**：`TEAM_DEVELOPMENT_PLAN.md` 第 526/591 条已写明「B1 持续未到位 → 成员2 提交 **B3-only 增训方案**供用户决策，其余成员进度不受影响」且「负责人应明确 B1 采集/标注的**最后期限**（建议 Phase 1 结束时），逾期走降级预案」。因 B3 线已废弃并由 labcoat_si（437 张已入库）替代，该预案现实际为 **labcoat_si-only 增训**——**数据侧已完全就绪**（当前 `data/lab_ppe` 13298 张即含该批）。
   - **行动建议（三条并行，互不阻塞）**：① 组织自采（半天，治本）；② 向组长要一个明确的 B1 截止日期（给 gating 一个出口）；③ 逾期即启动 labcoat_si-only 降级预案，先让 E4 跑起来、B1 到位后再增量。

> ⚠️ 工具坑位记录：本机 WorkBuddy 沙箱把 `os.unlink` 劫持为「移入回收站」，回收站操作被中止时触发 `SAFE_DELETE_FAIL_CLOSED` **直接杀进程**（表现为莫名其妙的 SIGTERM）；且 Windows 下 `os.rename` 到已存在目标必失败，`shutil.move` 会回退到 `copy + unlink` 而踩中该雷。故 `remove_bad_images.py` 已改为**只用 `os.replace`（原子可覆盖）+ 失败换备用名 + 单条失败不中断**，绝不回退到任何 unlink 路径。

工具：几何审计 `tools/audit_lab_ppe_boxes.py`（报告 `data/lab_ppe_audit_report.txt`）；按类别画框抽检 `tools/draw_labels.py --only-class <类>`。

### B+13. lcsi（labcoat_si）431 张专项审计 + 域瘦身（2026-10-09，负责人：许婧雯）

**审计**（只读，`tools/audit_lcsi_subset.py`，报告 `docs/lcsi_audit_20261009.md`）：①红线 vs dataset1/dataset2 **0 命中**；②内部近重复 41 组/104 张（**d=0 完全冗余 21 组/42 张**，标签核对一致；**跨 split 泄漏 8 组/valid 侧 10 张**——labcoat_si 原方案"只进 train"，valid 里的 62 张本身违规）；③脏图 126 张，图版目检证实负样本几乎全是**电商服装目录图**；④几何干净（2 超大框 + 1 重复框）。**结构性失真**：431 图占全库 17.7% 仅 414 框（6.7%），128 空标签，99.3% 单类图，全族 640×640 方块，54% 框贴边；漏标毒点 3 张（口罩/护目镜/手套未标）。

**处置**（她授权"做你觉得最好的方案"，`tools/lcsi_prune_apply.py --apply`，全部移动到 `data/_rejected/lcsi_prune_20261009/` 可逆，manifest `_prune_manifest.csv`）：保留**真实拍摄场景**、剔除**电商目录域与冗余**——剔 **286 张 / 225 框**（R1 d=0 冗余 21 + R2 挂账 lcsi_27 + R3 目录负样本 99 + R4 目录族 LC##/Lab_coat_*/suggested 40 + R5/R6 目录正样本 26 + R7 20250228 同一摆拍抽稀 49 + R8 crop_margin 抽稀 24 + R9 监控连续帧抽稀 24 + R10 漏标毒点 2）；保留 **145 张 / 189 框**（KakaoTalk 实拍 19、frame_* 监控 32、png-2024-05-20 真实场景 30、其他真实有框 43、CoatGoggle 4、灰图负样本 10、crop_margin/摆拍代表 7）。含 mask/gloves/goggles 框的图全部保护保留（除 lcsi_27，其内容由标注更全的 websr3_14 覆盖）。另：修复 1 处重复框（`-2024-05-20-143341` IoU 0.94，删 #1，备份 `_label_backup/lcsi_prune_20261009/`）；剩余 lcsi 重检跨 split 泄漏 **归零**（valid 侧 2 张移回 train）。

**执行后底座（B+13 落地）**：**train 1,200 图 / 4,480 框（mask 506 / gloves 1,804 / lab_coat 1,544 / goggles 626）；valid 222 图 / 856 框（mask 87 / gloves 353 / lab_coat 281 / goggles 135）；test 732 图 / 584 框未动**。全库 **2,154 图 / 5,920 框**（mask 652 / gloves 2,279 / lab_coat 1,862 / goggles 1,127），0 坏行、0 跨 split 重名、0 孤儿。lcsi 占比从 17.7% 图/6.7% 框降到 6.7% 图/3.2% 框，域失真显著缓解。可回收：`_rejected/lcsi_prune_20261009/` 整体移回即可（毒点 3 张如需保留须先补标）。

### B+14. 同源增强副本去重（2026-10-10，负责人：许婧雯）

**背景**：全库五类化迁移 + 增强数据审查时发现大量"同一张图换色 / 左右翻转"被 Roboflow 增强当作新样本（特征：同基础名 + 不同 `.rf.<hash>` 后缀），既虚增数据量、又让 valid 与 train 隐性同图（泄漏），还会污染 person 补标审查队列。

**工具**（`tools/dedup_aug_dups.py`，判决报告 `data/aug_dup_report.txt`、对照图版 `tools/make_dup_conflict_sheet.py`）：三路感知哈希（灰度 / 梯度幅值 × 原图 / 水平镜像，取 min 汉明距离）**只在同 `.rf.` 基础名组内**判冗余（避免 ppes_* CCTV 连续帧被误串成大组），配合 `best_kind()` 对 identity / fliph / flipv / rot180 变换后的 IoU 覆盖率与**灰度像素平均绝对差闸门**（防哈希把灰度副本误判为不同图），输出 8 类判决：`keep / drop_subset`（标签被代表完全覆盖=纯冗余）/ `drop_empty`（同源空标签副本=背景毒点）/ `drop_broken`（图像被镜像但标签未镜像=标注错位毒数据）/ `review_extra / review_class_conflict / review_pixel_diff`（两版标注矛盾，人工拍板）/ `keep_far`。

**结果**：全库 1,845 张扫描 → 唯一基础名 1,776、同源多副本基础名 65。**剔除 309 张（train）→ 移动到 `data/_rejected/aug_dup_20261010/`（manifest `_prune_manifest.csv` 可逆回）**；**test 侧 44 张冗余副本挂起**于 `data/_aug_dup_held.txt`（动 test = 改评测口径，待单独拍板，未动）；**19 项标注冲突**（18 `review_extra` + 1 `review_class_conflict`）生成 7 版对照图版 `outputs/aug_dup_review/conflict/` 供人工拍板，**未自动剔**。去重后 person 审查队列由 1,095 降到 786（-28%）。

**执行后底座（B+14 落地）**：**train 891 图 / 3,538 框（person 141 / mask 360 / gloves 1,305 / lab_coat 1,187 / goggles 545）；valid 222 图 / 881 框（person 25 / mask 87 / gloves 353 / lab_coat 281 / goggles 135）；test 732 图 / 584 框未动（mask 59 / gloves 122 / lab_coat 37 / goggles 366）**。全库 **1,845 图 / 5,003 框**（person 166 / mask 506 / gloves 1,780 / lab_coat 1,505 / goggles 1,046），0 坏行、0 跨 split 重名、0 孤儿。

### B+15. person 补标第一批落库（2026-10-10，负责人：许婧雯）

**流程**：`make_person_review.py` 生成 P0 批 p0_a（train 有 PPE 框队列前 60 图，现有五类框实线 + COCO person 候选 conf≥0.40 虚线）→ 编辑器修复两处（① 库内 cxcywh 直接喂左上角 xywh 编辑器导致**所有框右下偏移半框**，已加转换；② 候选框虚线 + "候选"前缀与旧标注区分）→ 她逐图确认，36 图有改动导出（编辑器只导出 dirty 图）→ 新工具 **`tools/apply_person_review.py`** 落库（整图覆盖、越界裁剪、全库校验）。

**处置**：36 图整图覆盖入库；其中 10 张（F 系列）她误删的护目镜框**按库内原框恢复**，M15 的台面普通眼镜框按"未佩戴不标"口径**保持删除**；合并净效果 = 新增 person 40（候选人工确认）、goggles -11+10、mask -1、gloves -8（重复合并）、lab_coat 若干框拉正。**重复副本一律不剔**（含 test 44 张挂起，维持现状）。

**执行后底座（B+15 落地）**：**train 891 图 / 3,568 框（person 181 / mask 359 / gloves 1,297 / lab_coat 1,187 / goggles 544）；valid 222/881、test 732/584 不变**。全库 **1,845 图 / 5,033 框**（person 206 / mask 505 / gloves 1,772 / lab_coat 1,505 / goggles 1,045），0 坏行、0 孤儿。落库前备份 `data/lab_ppe/_label_backup/labels_train_person_20261010.tar.gz`（891 文件）。person 审查队列 786→750，下一批 p0_b 已生成（`--exclude-export` 自动剔除已完成图）。

### B+16. person 补标第二批落库 + 口径变更（2026-10-10，负责人：许婧雯）

**口径变更（她 22:22 明确）**：「剩下没有改动的图片加上候选都是正确的」——即**未改动图 = 通过**，
其 person 候选整批采纳，不再要求逐图确认。此后每批只需回报**需要修改的图**（编辑器导出），
未出现在导出里的图按「库内框 + 候选」直接落库。

**批次**：p0_b 共 60 图（`outputs/person_review_p0_b/`）＝ 导出 22 图（她逐图改过）+ 未改动 38 图（采纳候选）。

**工具增强**（落在 `tools/`，供后续批次复用）：
- `apply_person_review.py`：`--keep-classes`（库内该类框被删则以原框保留，防误删；IoU<0.5 视为被删）、
  `--accept-from-ws` + `--batch-index`（未改动图直接采用工作区标签＝库内+候选）、
  `--done-file`（落库后追加"已完成清单"）、备份文件名改为**带时分秒**（原先只带日期，同日两批会互相覆盖）。
- `make_person_review.py`：`--exclude-all`（自动排除所有 `outputs/person_review_*/_her_export.txt` 的已完成图）、
  `--done-file`（读已完成清单）。已回填 `data/_person_review_done.txt` = p0_a 36 + p0_b 60 = **96 图**。

**处置**：
- 22 图整图覆盖入库，38 图采纳候选（每图各 +1 person 框）；
- **防误删保留 7 个 lab_coat 框**（M61 / M67 / M77 / M78 / M82 / M85 / M8，导出中消失且与导出框 IoU<0.5）
  —— 按 B+15 口径以库内原框保留，核实图版 `outputs/person_review_p0_b/_kept_labcoat_check.jpg`，
  **待她复核：确认是误删（保留）还是有意删（再移除）**；
- z13 两个高度重叠的 goggles 框（IoU 0.69）被她合并为 1 个，属实质修正，已落库；
- 其余"删除+新增"实为坐标微调（删了又加回），非实质改动。

**执行后底座（B+16 落地）**：**train 891 图 / 3,633 框（person 246 / mask 359 / gloves 1,297 / lab_coat 1,188 / goggles 543）；valid 222/881、test 732/584 不变**。全库 **1,845 图 / 5,098 框**（person 271 / mask 505 / gloves 1,772 / lab_coat 1,506 / goggles 1,044），0 坏行、0 孤儿。
净效果 = **person +65**、lab_coat +1（保留的 7 个 − 6 个待复核删除 + M8 新增 1）、goggles −1（z13 合并）。
落库前备份 `data/lab_ppe/_label_backup/labels_train_person_pre_p0b_20261010_2225.tar.gz`（891 文件）。
**person 审查队列 786 → 690**，下一批 `p0_c` 已生成（60 图，起始为 z 系列）。

---

## C. 增训方案

### C1. 数据划分
- **原有 dataset1 split 一律不动**（保持可比性）
- 新增数据单独按 **90/10 划分进 train/valid**（新增 valid 样本让 best.pt 选择更可靠）；新增数据不进 test（dataset1 test 保持纯净）
- **dataset2 全程零接触**
- 所有新增数据先过 `dedup_check.py`（对 dataset1 + dataset2 双向去重），再入库

### C2. 训练配置（两个选项，二选一）

| 项 | 方案Ⅰ：从 best.pt 续训（推荐） | 方案Ⅱ：从官方权重重训 |
|---|---|---|
| 起点 | `runs/e2/labppe_v8s/weights/best.pt` | `yolov8s.pt`（官方 COCO） |
| epochs | 30 | 50 |
| batch / imgsz | 16 / 640（实测显存 3.4GB，安全） | 同左 |
| lr0 | **0.001**（降低，防灾难性遗忘） | 默认 0.01 |
| 其他 | 全默认 | 全默认（与 E2 完全一致） |
| 预计耗时 | ~2.5 小时 | ~4.5 小时 |
| 优点 | 快、原有能力保持好 | 不受 E2 已收敛的局部最优限制 |
| 风险 | 新旧数据不平衡时学不动新域 | mask/goggles 等可能轻微波动 |

**推荐方案Ⅰ**：E2 模型在 dataset1 上已经很成熟（mAP50 0.924），短板只在域覆盖；低学习率 + 新增真实场景数据正好补域。若方案Ⅰ增训后 lab_coat 跨域提升 <10pp 再考虑方案Ⅱ。

### C3. 训练后动作
重跑 E3 全套口径：dataset2 external_test（251 张）+ t01-t10 + dataset1 valid/test 回归测试，输出对比报告（SH17 / E2 / E4 三列）。

---

## D. 风险

1. **灾难性遗忘**：新增数据分布与原训练集差异大，可能拉低 mask/gloves/goggles → 用低 lr0 + 回归测试兜底；任何一类 dataset1 test F1 回退 >3pp 即判不合格
2. **数据泄漏**：B3 疑与 dataset1/dataset2 同源 → 强制先去重；与 dataset2 有任何 MD5/pHash 命中的整批弃用
3. **B2 版权瑕疵**：getty/istock 水印图商用/公开发布有风险 → 课程项目内部使用可接受，报告中注明来源；也可以只用 B1+B3 规避
4. **跨域 F1 仍达不到 0.70**：新增 800–1200 框对域差距的修复是概率性的，若真实场景白大褂形态多样（侧身/半遮挡/白大褂敞开），可能只到 0.5–0.6 → 验收门槛分级（见 E）
5. **自采集偏差**：全是同一实验室/同类人的照片会让模型过拟合到新小域 → 采集时强制多样化（≥5 人、≥3 种光照、近/中/远三种距离）

## E. 验收门槛（建议）

| 检验 | 门槛 |
|---|---|
| dataset1 test lab_coat F1 | ≥ 0.85（E2 为 0.952，允许 ≤10pp 回退） |
| dataset1 test mask/gloves/goggles F1 | 回退 ≤ 3pp |
| **dataset2 external_test lab_coat F1** | **≥ 0.50 达标；≥ 0.70 直接过 E4 决策门；0.50–0.70 接入并文档说明局限；< 0.50 停止，重新评估路线** |
| dataset2 mask/gloves F1 | 回退 ≤ 5pp（E2：75.1 / 75.7） |
| dataset2 goggles F1 | 不低于 E2 的 32.7 |
| t01-t10 | t01/t02/t07 白大褂至少检出 2 张；不新增误检 |
| 训练健康性 | 无 NaN、loss 正常收敛、best.pt 推理通过（同 E2 标准） |

---

## 待用户决策事项（2026-09-21 更新）

1. ~~B3 下载后先跑去重审计~~ → **已完成**：审计结论为废弃（标注系统性右偏），改用 labcoat_si 并已入库
2. labcoat_si 的水印子集剔不剔（建议：不剔，与 dataset1 已含 istockphoto 图的口径一致，报告注明；如要剔，翻 `outputs/merged_labcoat_check_all/` 记文件名即可）
3. B2（chintan/labcoat，约 2000 张）还用不用——labcoat_si 入库后 lab_coat 框已达 1935，B2 的边际价值下降，**建议暂缓**，等 E4 第一轮结果再定
4. 训练选方案Ⅰ（best.pt 续训 30 epochs, lr0=0.001）还是方案Ⅱ（重训 50 epochs）——可先定下来，等 B1 到位即开训
5. **B1 落地**（唯一硬前置，执行规范见 `docs/B1_采集SOP.md`）：150 张自采（≥3 人、≥3 种光照、近/中/远三种距离，**远距离+背面为重点**）。已明确**不必等组长亲自动手**——任何成员可组织采集；若走降级，则由成员2 提交 labcoat_si-only 增训方案供决策（见 §B++ 第 10 条）。

### B+17. 7 张 lab_coat 待定图复核重做批（2026-10-10 深夜，负责人：许婧雯）

她反馈：B+16 保留的 7 个 lab_coat 框所在图她**确实都改过**，但希望重新框选这 7 张再定。
为此把 `make_person_review.py` 扩了三个能力（复核重做通用）：

- `--stems`：逗号分隔图名强制成批（忽略已完成剔除与队列切片）；
- `--pending-vs-export`：库内框在旧导出中消失（IoU<0.5）→ 编辑器里标 **★待确认**（橙点线框，cand=2，导出格式不变）；
- 候选去重：候选与库内 person 框 IoU≥0.85（或任意类 IoU≥0.95、候选间 IoU≥0.85）时丢弃，
  避免编辑器叠画双框、落库写出重复行（本批 4 图共丢弃 5 个重复候选）。

**批 p0_b2**：7 图（M61 / M67 / M77 / M78 / M82 / M85 / M8），
`outputs/person_review_p0_b2/editor.html`（需本地浏览器打开），对照图版 `_pending_check.jpg`。
图面观察：7 个 ★框多为**搭在椅背/桌面上的白大褂**，与她"未佩戴不标"口径（B+15 M15 台面眼镜）一致，
最终以她重框结果为准。落库时**不带 --keep-classes**（她的删除/保留直接生效），
未改动图仍按"未改动即通过"采纳工作区（库内+候选）。

### B+18. person 补标第三批落库（p0_c，2026-10-10 深夜，负责人：许婧雯）

她 22:46 贴回导出（20 图有改动），口径同前：「剩下没有改动的图片算上候选框都是正确的」→ 未改动即通过。

- 60 图 = 导出 20 + 未改动采纳 40（库内框 + person 候选）；
- **person +66**（导出净 +21：19 图各 +1、3191292-…-0015 +2；未改动图候选净 +45）；
- z5 除 person 外她另补 **gloves +1**；z42 保留库内 3 个手套框不变（其中 2 个几乎重合，IoU 0.86，属**历史遗留重复标注**，后续单独处理）；
- `12_jpeg` 的 lab_coat 框被她**改形状**（0.1995×0.5192 → 0.4634×0.5438，即同一件白大褂重框，IoU 0.41）→ 判定为**修正而非误删**，本批**未启用 `--keep-classes`**（核对后仅此一处非 person 变化）；
- 落库前逐图核对：**无误删**、批次外导出 0、未改动图工作区重复行 0。

**执行后底座（B+18 落地）**：**train 891 图 / 3,700 框（person 312 / mask 359 / gloves 1,298 / lab_coat 1,188 / goggles 543）**；valid 222/881、test 732/584 不变。全库 **1,845 图 / 5,165 框**（person 337 / mask 505 / gloves 1,773 / lab_coat 1,506 / goggles 1,044），0 坏行、0 孤儿。
落库前备份 `data/lab_ppe/_label_backup/labels_train_person_20261010_224716.tar.gz`（891 文件）；已完成清单 96 → **156 图**。

### B+19. person 补标第四批落库（p0_d，2026-10-10 深夜，负责人：许婧雯）

她 23:00 贴回导出（36 图有改动），口径同前：「剩下没有改动的图片算上候选框都是正确的」→ 未改动即通过。

- 60 图 = 导出 36 + 未改动采纳 24；**person +91**（train 312→403，本批为视频帧域，候选密集，人均 +2~3）；
- **19 个非 person 框被她删除，逐框放大核实后全部判定为合理修正、直接落库**（未启用 `--keep-classes`）：
  - 3191292 帧：goggles 框在未戴护目镜的脸上、gloves 框在裸手上（错误标注）；
  - 3196600 / 4121322 的 gloves 小框：框在显微镜机身上，或为被保留大框套住的小重复框（合并去重）；
  - 4121322-0000 的 lab_coat 框在戴口罩的脸部（错误标注）；
  - 8381407 系列 8 帧：lab_coat 大框实为「人+显微镜」混框，她改标 person（跨帧一致的修正）；
- 核实图版：`outputs/person_review_p0_d/_deleted_nonperson_check.jpg`；
- 历史遗留重复框仍为 6 处（F16、k1、z42、3 张 istockphoto 帧），本批零新增，待统一清理。

**执行后底座（B+19 落地）**：**train 891 图 / 3,777 框（person 403 / mask 359 / gloves 1,290 / lab_coat 1,181 / goggles 544）**；valid 222/881、test 732/584 不变。全库 **1,845 图 / 5,242 框**（person 428 / mask 505 / gloves 1,765 / lab_coat 1,499 / goggles 1,045），0 坏行、0 孤儿。
落库前备份 `data/lab_ppe/_label_backup/labels_train_person_20261010_230422.tar.gz`（891 文件）；已完成清单 156 → **216 图**，队列 570 → **510**。

**下一批 p0_e 已生成**（2026-10-10 23:10）：60 图 / 现有框 304 / person 候选 84 → `outputs/person_review_p0_e/editor.html`；`--exclude-all` 剔除已完成 216 图，本批与已完成清单**零交集**；族分布以视频帧系列为主（8381454 十一帧、8771134 九帧、8381407 六帧）。队列 570 → 510。

### B+20. 同族套用首航：8381407 剩余 6 帧自动处理（2026-10-10 深夜，负责人：许婧雯）

她指出 p0_e 开头与 p0_d 画面雷同——核对后确认是**同一段视频的后续帧**（0008/0009/0011/0012/0014/0015），
且每帧都带着与 p0_d 完全相同的错误 lab_coat 混框。经她确认采用「同族批量套用」：

- 新工具 `tools/family_transfer.py`：从参照批的**落库前备份**与**她的导出**里 diff 出删除操作
  （按类 + IoU 合并跨帧副本，本例 9 条 → 2 类），按坐标映射（IoU≥0.5）套到同族目标帧；
  **person 不跨帧复制坐标**，改为采纳各帧自己的 COCO 候选（逐帧位置更准）；
  结果回写工作区标签，保证后续「未改动即采纳」幂等；
- 执行：6 帧共删 lab_coat 8 个（0008/0009/0014/0015 各 1、0011/0012 各 2），person +12（各 +2 候选）；
  核对图版 `outputs/person_review_p0_e/_family_8381407_check.jpg`（黑框 DEL=已删、红=person）；
- **5 个目标帧新框保守保留待她看**：0014/0015 的 mask/gloves 位置随人俯身移动（正常）、
  0009 一枚 0.017×0.051 的小 gloves 框（参照帧没有，可能是噪声，她可后续删）；
- 同段 8381407 全 14 帧现已口径统一：无 lab_coat、每人 person+mask+gloves 标齐。

**执行后底座（B+20 落地）**：**train 891 图 / 3,781 框（person 415 / mask 359 / gloves 1,290 / lab_coat 1,173 / goggles 544）**；
全库 **5,246 框**（person 440），0 坏行。已完成清单 216 → **222**；p0_e 剩余 **54 帧**待她编辑器审查。
落库前备份 `labels_train_person_20261010_231145.tar.gz`。
后续视频帧族（3196600 / 4121322 / 8328042 / 8381408 / 8381416 / 8381451 等）可复用此工具加速。

**B+20 追记（她看图版后确认）**：8381407 六帧里两人的白色服装是**连帽防护服（头发全包）而非实验服**，
所以删除 8 个 lab_coat 框的口径正确；该片段保持「无 lab_coat、person+mask+gloves 标齐」。

### B+21. lab_coat 复核重做落库（p0_b2，2026-10-10 深夜，负责人：许婧雯）

B+17 那批「★待确认」lab_coat 框的终审落地。她在 p0_b2 编辑器里**二次确认删除的正是
p0_b 时被 `--keep-classes 3` 防误删保护保留下来的 7 个 lab_coat 框**（M61/M67/M77/M78/M82/M85/M8
各 1 个）——对照图版核实全部是**搭在椅背上/铺在桌面上的空白大褂**，按「未穿戴不标」口径删除正确。
她的 p0_b2 导出与 p0_b 原导出逐框一致，证明当初删意明确。

- 落库**不带 `--keep-classes`**（复核重做批铁律：导出即最终态）；person 候选未采纳
  （M61 的 3 个、M77 的 1 个候选她没有保留）；
- 执行后底座：**train 891 图 / 3,774 框（person 415 / mask 359 / gloves 1,290 / lab_coat 1,166 / goggles 544）**；
  全库 **5,239 框**（person 440），0 坏行、0 孤儿；
- 落库前备份 `data/lab_ppe/_label_backup/labels_train_person_20261010_233138.tar.gz`（891 文件）；
- 已完成清单保持 **222**（这 7 图在 p0_b 已计入，复核不重复计数）。

**流程教训**：`--keep-classes` 防误删保护与「她真要删」的场景冲突时，会形成"删→保护→复核→再删"的返工环。
后续凡带 `--pending-vs-export` 的复核重做批，落库一律不加保护类；真误删由落库后的逐图差异报告兜底。

### B+22. person 补标第五批落库（p0_e，2026-10-10 深夜，负责人：许婧雯）

她 23:29 贴回导出（50 图有改动，含 8381407 六帧——与 B+20 同族套用结果一致，幂等覆盖无副作用），口径同前。

- 60 图 = 导出 50 + 未改动采纳 10；**person +73**（train 415→488，本批 g 系列实验服场景 + 视频帧域混合）；
- **45 个非 person 框被删，按族去重后 15 类，逐类放大核实（`_deleted_nonperson_check.jpg`）全部合理、直接落库**（未启用 `--keep-classes`）：
  - 视频帧族 lab_coat（8381454×11 / 8771134×12 / 8381451×4 / 8381408×3 / 9244231×2）：连帽防护服误标 lab_coat（延续 B+20「防护服≠实验服」口径）或被 person 大框套住的小重复框；
  - g1 / g22 的 lab_coat：挂在墙上/搭着的白大褂（未佩戴不标，同 B+15 M15 口径）；
  - g3 / g11 / g12 / g22 / 8381416 的 gloves：框在桌面蓝色物件、黑色物品、画面模糊处（非手戴手套）；
  - g107 / g17 的 lab_coat：**改形状**（IoU 0.41~0.46，同类框数不变），非删除；
- mask +4（未改动帧采纳）、goggles +1；gloves −7（去噪）。

**执行后底座（B+22 落地，接 B+21 p0_b2 之后）**：**train 891 图 / 3,811 框（person 488 / mask 363 / gloves 1,283 / lab_coat 1,132 / goggles 545）**；
valid 222/881、test 732/584 不变。全库 **1,845 图 / 5,276 框**（person 513 / mask 509 / gloves 1,758 / lab_coat 1,450 / goggles 1,046），0 坏行、0 孤儿。
落库前备份 `labels_train_person_20261010_233238.tar.gz`（891 文件）；已完成清单 222 → **276 图**，队列 510 → **450**。
历史遗留重复框仍 6 处，本批零新增。

### B+23. person 补标第六批落库（p0_f，2026-10-10 深夜，负责人：许婧雯）

她 23:41 贴回导出（37 图有改动），口径同前。

- 60 图 = 导出 37 + 未改动采纳 23；**person +71**（train 488→559；含 lcsi_ 系列 +8：该族 5 图里 4 图补齐 person）；
- **14 个非 person 框被删**，逐类放大核实（`_deleted_nonperson_check.jpg`）后**就 g53/g54/g55/g56/g57/g63/g65 温室图的白衣性质专门与她确认**：
  她判定为**防护服/误标**（温室农业场景非实验服），删除正确 → 全部直接落库（未启用 `--keep-classes`）：
  - 温室族 lab_coat ×7（g53~g65）；g4 / g98 为**重框替换**（旧小框删、新框保留）；g82 两个小误标框；g73 小框；
  - gloves ×2（g72 / g74，桌面物件误标）；
- 口径沉淀：**温室/农业白大褂一律不标 lab_coat**（与 B+20「防护服≠实验服」同源——lab_coat 仅限实验室/科研/医疗操作场景）。

**执行后底座（B+23 落地）**：**train 891 图 / 3,871 框（person 559 / mask 363 / gloves 1,283 / lab_coat 1,121 / goggles 545）**；
valid 222/881、test 732/584 不变。全库 **1,845 图 / 5,336 框**（person 584 / mask 509 / gloves 1,758 / lab_coat 1,439 / goggles 1,046），0 坏行、0 孤儿。
落库前备份 `labels_train_person_20261010_234416.tar.gz`（891 文件）；已完成清单 276 → **336 图**，队列 450 → **390**。
历史遗留重复框仍 6 处，本批零新增。

### B+24. CPPE-5 终审清单入库：401 图 / 2,441 框（2026-10-10 深夜，负责人：许婧雯）

她先拍「keep_sheets 16 版全部可以采用」（468 张），审计揭示三处需清洗后她拍「按你说的剔」。完整拍板链：

1. **泄漏剔除 39**：第①关发现 dataset1 本身就含 CPPE-5（`N_png_png.rf.*` 家族 986 张，当初建主库只收了 24 张进 test），
   468 张与 dataset1 全部命中、其中 **39 张与主库 test 同图（pHash d=0~8）→ 必剔防泄漏**；净新增实为"捡回当初未选图"。
2. **批内同图冗余 33**：第②关 18 组 dist=0 同图（如 661/709/768/882/928/931 八张互为同图），每组留标注最全 1 张。
3. **空挂装备 8**（她拍板「按你说的剔」，AI 拟 15 张，其中 7 张已被第 2 步顺带剔除）：
   教学视频里挂在椅背/站立的空防护服被 CPPE-5 标了 person+mask+gloves，违反「未穿戴不标」口径。
   AI 目检补正：#734/#630 放大核实是**真人穿戴**（虚惊，保留）。
4. **毒标注 3**（AI 画框核实）：#60 mask/gloves 框锚在 (1,1) 的巨型越界废框；
   #503/#770 的 goggles/mask 框框在**桌面未佩戴装备**上（同「未佩戴不标」口径）。
5. **几何清洗**：第④关 63 条问题 → 越界裁剪、大半出图丢弃、同类 IoU≥0.95 去重，共丢 3 框（55/487/578 各 1）；
   暗光帧（74 张脏图嫌疑大多为此）经三版拼版目检为**真实穿戴场景，保留**。

**入库细节**：COCO→YOLO 五类映射（Mask→1、Gloves→2、Goggles→4；Coverall/Face_Shield 弃）；
person 框来自 `yolov8n_coco` 自动检测（conf≥0.40，共 881 个，**非人工标注**——训练后 person 指标若异常，可凭 manifest 整体回退本批）；
划分按第②关 dist≤8 近重复**连通分量整组同 split**（359 组），train +361 / valid +40，不进 test；
PNG→JPG(q95)，改名 `cppe5_<编号>_<md5[:8]>.jpg`，溯源 `data/raw/cppe5/import_manifest_cppe5.csv`（401 行）；
被剔 67 张仍留在 `data/raw/cppe5/picked/` 未删。工具 `tools/import_cppe5_batch.py`。
落库前备份 `data/lab_ppe/_label_backup/labels_tv_cppe5_20261010_234941.tar.gz`（train+valid 1,113 文件）。

**执行后底座（B+24 落地）**：**train 1,252 图 / 5,937 框（person 1,306 / mask 918 / gloves 1,865 / lab_coat 1,121 / goggles 727）**；
valid **262 图 / 1,256 框**（person 159 / mask 184 / gloves 471 / lab_coat 281 / goggles 161）；test 732/584 不变。
全库 **2,246 图 / 7,777 框**（person 1,465 / mask 1,161 / gloves 2,458 / lab_coat 1,439 / goggles 1,254），0 坏行、0 孤儿。
**mask 短板大幅缓解：509 → 1,161（+652）**，不再是唯一真短板；person 亦 513 → 1,465。
遗留：person 自动框质量未逐框人工过（881 个），重训前建议抽查；queue p0_g 待她审查。

### B+25. person 补标第七批落库（p0_g，2026-10-11 凌晨，负责人：许婧雯）

批次 60 图（lcsi_ 系列），她导出 36 图 + 未改动即通过 23 图 = 59 图落库，1 张剔除。

1. **剔除 `lcsi_-2024-05-20-143254`**：dreamstime 图库水印铺满全图（螺旋水印 + 底部水印条），她指出
   「太不清晰并且满是水印」且在导出中已将该图清零 → 图 + 标签移入
   `data/_rejected/p0g_watermark_20261011/`（只移不删）。
2. **迷你废框拦截 3 个**：导出中 3 个 0.02×0.02 的编辑器最小尺寸废框未入库——
   143254 的 goggles 1 个（随图剔除）；lcsi_39 天花板处 lab_coat 2 个（系误拖，丢弃，图内真实标注不受影响）。
3. **同批澄清 p0_b2 遗留**：逐框比对确认磁盘 M61/M67/M77/M78/M82/M85/M8 与她的 p0_b2 导出完全一致
   （仅小数尾零差异）——各图保留的 1 个 lab_coat 是她重框后的**修正框**而非漏删，「7 个未删 lab_coat」遗留项关闭。

**增量（59 图 vs 落库前备份）**：person +126、mask +6、gloves +18、lab_coat +5、goggles +23；剔除图另 -5 lab_coat。
备份：`data/lab_ppe/_label_backup/labels_train_person_20261011_001404.tar.gz`（1,251 文件）。
done 清单 336 → 395；剩余队列（p0_h 生成时统计）：751 图（P0 有PPE框 704 / P1 空标签 47）。

**执行后底座（B+25 落地）**：**train 1,251 图 / 6,110 框（person 1,432 / mask 924 / gloves 1,883 / lab_coat 1,121 / goggles 750）**；
valid 262/1,256、test 732/584 不变。全库 **2,245 图 / 7,950 框**（person 1,591 / mask 1,167 / gloves 2,476 /
lab_coat 1,439 / goggles 1,277），0 坏行、0 孤儿图、0 孤儿标签。
下一批：p0_h（`outputs/person_review_p0_h/editor.html`）。

### B+26. p0_g 同族追剔：电商商品/抠图 11 张（2026-10-11 上午，负责人：许婧雯）

B+25 落库后复核场景红线，`lcsi_-2024-05-20-` 视频族实为图库/商品素材混入：

1. **她点名剔除 143141/143213**（透明棋盘底电商抠图），并确认 143254 水印剔除（B+25 已执行）。
2. AI 对该族剩余 22 帧出拼版排查（`outputs/lvis_mine/_family_20240520_sheet.jpg`），她拍板再剔 9 张：
   无人挂拍 **141826**（人台模特）/ **142916**（墙上挂白大褂）；透明底抠图 **141740 / 142128 / 142506 / 142443 / 143131**；影棚商品图 **142424 / 142713**。
3. 共 11 张图+标签移入 `data/_rejected/p0g_ecom_20261011/`（只移不删），-16 person / -18 lab_coat。
   族内剩余 13 帧（带实验/书架等真实背景的摆拍图库照）保留。

**执行后底座（B+26 后）**：**train 1,240 图 / 6,076 框（person 1,416 / mask 924 / gloves 1,883 / lab_coat 1,103 / goggles 750）**；
valid 262/1,256、test 732/584 不变；全库 **2,234 图 / 7,916 框**。p0_h 批次不含该族帧，不受影响。

### B+27. person 补标第八批落库（p0_h）+ 剔除 cppe5_153（2026-10-11 上午，负责人：许婧雯）

她 11:15 贴回 p0_h 导出（45 图有改动），口径同前，并点名**剔除批次最后一张 `cppe5_153_a1394779`**。

- 60 图 = 导出 45 + 未改动采纳 15；**person +30**（1,416→1,446）、mask +2、gloves +23、goggles +10、lab_coat 0；
- **亮点：lcsi_KakaoTalk 族 9 图主动补标 goggles ×9 / gloves ×18**（编辑器 1–5 键切类手画）——该族原只有 lab_coat，
  属于"标注补全"；cppe5 若干图亦补 mask/gloves/goggles。导出与库内逐类 diff **无非 person 误删**，未启用 `--keep-classes`；
- **剔除 cppe5_153_a1394779**（她点名）：火葬场混乱人群场景，person 框大面积互相重叠、gloves/mask 框乱贴，
  且不符合「实验室/科研/医疗操作」场景红线 → 图+标签移入 `data/_rejected/cppe5_review_20261011/`（只移不删），
  -11 person / -6 mask / -5 gloves；import_manifest_cppe5.csv 中该行编号 153 作废；
- 落库前备份 `data/lab_ppe/_label_backup/labels_train_person_20261011_111759.tar.gz`（1,240 文件）；
- done 清单 395 → **455 图**；队列 751 → **691**。

**追记（11:31）：剔除已做成永久登记。** 她把同一份 p0_h 导出第三次贴回、仍要求「这张图片剔除」——核实为重复投递
（该导出 45 图与库内数值完全一致，唯一差异就是 153 已不在库）。为防 `import_cppe5_batch.py` 重跑把 153 复活，
已把该编号写入脚本常量 **`EXCLUDE_MANUAL = {153: …}`**（与 EXCLUDE_HANGING / EXCLUDE_POISON 同级），
并在 `data/raw/cppe5/manual_exclusions.txt` 留存带原因的登记表。另：**p0_h 的编辑器是旧版**（无「剔除此图」按钮），
且其 153 的图片路径已因剔除而失效——该批已收尾，**新审请用 p0_i 的编辑器**（含一键剔除 + X 键）。

**执行后底座（B+27 落地）**：**train 1,239 图 / 6,118 框（person 1,434 / mask 920 / gloves 1,901 / lab_coat 1,103 / goggles 760）**；
valid 262/1,256、test 732/584 不变。全库 **2,233 图 / 7,958 框**（person 1,593 / mask 1,163 / gloves 2,494 /
lab_coat 1,421 / goggles 1,287），0 坏行、0 孤儿。

### 工具更新：审核编辑器支持「整图剔除」（2026-10-11，负责人：许婧雯）

此前剔除整图必须口头告诉我、由我在落库时手工 `mv`，她无法在审核现场决定；现内置到编辑器：

1. **编辑器**（`tools/make_box_editor.py`，person 批次共用）新增：
   - 顶部 **「剔除此图」** 按钮 + **X 键**（再按一次撤销）；剔除后画面套红条纹遮罩 + 居中「✕ 已剔除」字样，
     框层淡化 18%，防止误标；旁边可填**剔除原因**（水印/电商图/场景不符…）。
   - 导出格式新增标记行（旧导出无此行，完全向后兼容）：
     ```
     ### <文件名>
     !REJECT 水印
     0 0.6 0.5 0.2 0.2        ← 框照旧导出（便于反悔恢复）
     ```
2. **落库端**（`tools/apply_person_review.py`）识别 `!REJECT`：
   - 该图**不写标签**，把 `images/<split>/<name>.*` 与 `labels/<split>/<name>.txt` **一并移动**到
     `data/_rejected/<批名>_reject_<日期>/`（只移不删，可回滚），并删除工作区标签；
   - 剔除图照常计入 done 清单（不再进队列）；干跑会先列出「将移动哪些图」。
   - 新增可选参数 `--reject-tag <目录名>`，默认 `<批名>_reject_<YYYYMMDD>`。
3. 已用一次性沙盒 split（`_smoketest`，测试后即清理）跑通：图+标签移动、工作区清理、未剔除图正常落库、
   全库校验 0 坏行 / 0 孤儿；编辑器内嵌 JS 通过 `node --check`。
4. `docs/E4_lab_coat_plan.md` 与技能文档同步更新；**p0_i 批次编辑器已重生成**，可直接用新功能。

### B+28. person 补标第九批落库（p0_i，2026-10-11 中午，负责人：许婧雯）

她 11:34 贴回 p0_i 导出（37 图有改动，**首次使用新版编辑器**，无 `!REJECT` 标记 = 本批零整图剔除）。

- 60 图 = 导出 37 + 未改动采纳 23；**净增 27 框：person +16 / mask +4 / gloves +5 / goggles +1 / lab_coat +1**；
- 非 person 删除仅 2 框、集中在 `cppe5_161_e93014b4`（mask+goggles）：画框核实该图是**全面罩/面罩式防护**，
  按「面罩不收」口径删除合理；同图她另把 person 框重画到头部面罩整体；
- 其余改动以 **person 补框/挪框**为主（自动候选框人工修正确认），另补 mask ×5 / gloves ×5 / goggles ×2 / lab_coat ×1；
- 落库前备份 `data/lab_ppe/_label_backup/labels_train_person_20261011_113602.tar.gz`（1,239 文件）；
- done 清单 455 → **515 图**；队列 691 → **631**（P0 584 / P1 47）；下一批 p0_j 已生成。

**执行后底座（B+28 落地）**：**train 1,239 图 / 6,145 框（person 1,450 / mask 924 / gloves 1,906 / lab_coat 1,104 / goggles 761）**；
valid 262/1,256、test 732/584 不变。全库 **2,233 图 / 7,985 框**（person 1,609 / mask 1,167 / gloves 2,499 /
lab_coat 1,422 / goggles 1,288），0 坏行、0 孤儿。
