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

> **编号规约（2026-10-11 起）**：B+ 编号全文件**唯一且不复用**。若并行会话同时写入导致撞号，
> 原则上**后写的一方改号**，且**不同工作流使用不重叠号段**：
> - B+1 ~ B+99：person 补标线 / 通用数据治理
> - **B+101 ~ B+199：lc13（lab_coat 单类人工画框）线**
> 正文内的交叉引用（如「lc13 系累计（B+101~105）」）须与标题同步更新。

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
- 本会话落库后逐框复核：37 张导出图与磁盘标签归一化比对**全部一致**，未导出 23 图工作区与库内一致，无需重放；
- **废框拦截 1 个**：cppe5_328 中央床栏处 `0.02×0.02` person（编辑器最小拖拽尺寸、该处无人）→ 落库后剔除；
- done 清单 455 → **515 图**；队列 691 → **631**（P0 584 / P1 47）；下一批 p0_j 已生成。

**执行后底座（B+28 落地，含废框剔除）**：**train 1,239 图 / 6,144 框（person 1,449 / mask 924 / gloves 1,906 / lab_coat 1,104 / goggles 761）**；
valid 262/1,256、test 732/584 不变。全库 **2,233 图 / 7,984 框**（person 1,608 / mask 1,167 / gloves 2,499 /
lab_coat 1,422 / goggles 1,288），0 坏行、0 孤儿。
### B+29. p0_i 复核点名整图剔除 ×2：cppe5_188 / cppe5_328（2026-10-11 下午，负责人：许婧雯）

她截图点名两张 p0_i 批内图整图剔除（截图无文件名，按画面匹配 p0_i 批次确认编号）：

1. **cppe5_188_31ada0d1**（8 框：person4/mask2/gloves2）：马路/摩托车街景，
   违反场景红线（非实验室/科研/医疗操作），剔除；
2. **cppe5_328_305277ee**（10 框：person2/mask2/gloves4）：ICU 抢救场景为**双重曝光重影**合成，
   框无法标稳，剔除。

- 图+标签移入 `data/_rejected/cppe5_review_20261011/`（与 cppe5_153 同批）；
- 登记 `data/raw/cppe5/manual_exclusions.txt`，`tools/import_cppe5_batch.py` EXCLUDE_MANUAL 增补
  188/328，保证脚本重跑不复活；p0_j 批次不含这两张，无需改队列。

**执行后底座（B+29 落地，实测盘点）**：**train 1,237 图 / 6,127 框（person 1,442 / mask 920 / gloves 1,900 / lab_coat 1,104 / goggles 761）**；
valid 262/1,256、test 732/584 不变。全库 **2,231 图 / 7,967 框**（person 1,601 / mask 1,163 / gloves 2,493 /
lab_coat 1,422 / goggles 1,288）。done 清单 515、队列 631 不变。

### B+30. 外部 PPE 数据包接入评估（4 份桌面数据，2026-10-11 下午，负责人：许婧雯）

组长委托下载 4 份公开数据包（PPE.zip / ppe_public.zip / protective-equipment-detection / e58z9vu3fcusg7wzjk91n.zip）
并转来说明，本线做了来源核查与接入评估，结论**未入库任何新数据**（评估全文见 `docs/E7_外部PPE数据集接入评估_20261011.md`）：

- **PPE.zip 与 ppe_public 内 PPE_Detection 逐字节同源**（`tanish-y7iqo/ppe-detection_data-adcya` v3，6 类）→ 同数据两副本；
  该数据集**无 lab_coat / 无 person**、工地/工业域、空标签 31%、goggles 疑混普通眼镜，**建议不投入**；
  抽 400 张 MD5 vs 库内 2,231 张 **命中 0**，确为独立来源（非既有派生物）。
- **CPPE-5 早已入库 358 张 `cppe5_*`** 并经 p0_c…p0_i 共 7 轮审核（E4 既定口径：整包不入库、挂起为 mask 应急备胎）→ 不新增。
- **LVIS lab_coat 子集仅 10 张 / 46 框**（官方 category_id=620）→ 维持"当测试/回归样本"。
- **`e58z9vu3fcusg7wzjk91n.zip` = images.cv 1,299 张 lab coat 分类图（无 bbox）**：
  唯一对症 lab_coat 的新素材，需 ① 场景红线筛选 ② 人工标 lab_coat 检测框 ③ 按前缀+只移不删+落库前备份的既有口径接入；
  人工标框需向组长索取 `best.pt` 做预标注加速（本地无任何 `.pt`，已被 `.gitignore` 排除）。
- **`protective-equipment-detection/`（2.3 GB）来源待确认**：须核对 Roboflow project 与已知三条线是否重合；
  `no_*` 类不入正类（本项目无"未佩戴"类）、`suit` 不映射 lab_coat（Coverall→lab_coat 是用户特例，不可外推）。

**底座不变（B+30 无数据变更）**：train 1,237 图 / 6,127 框（person 1,442 / mask 920 / gloves 1,900 / lab_coat 1,104 / goggles 761）；
valid 262/1,256、test 732/584；全库 **2,231 图 / 7,967 框**。done 515、队列余 631（P0 584 / P1 47），下一批 `p0_j` 不变。

### B+31. person 补标第十批落库（p0_j，2026-10-11 中午，负责人：许婧雯）

批次 60 图（CPPE-5 尾段 cppe5_337~509 + cppe5_50）。她导出 45 图，其中 **6 张整图剔除（首次实战新功能）**，
39 张有效导出 + 15 张未改动即通过 = 54 图落库；备份 `labels_train_person_20261011_120348.tar.gz`（1,237 文件）；
done 清单 515 → **575 图**（含剔除图）。

1. **整图剔除 6 张 → `data/_rejected/p0j_reject_20261011/`**（图+标签一并移动，框保留在 _rejected 内可回溯）：
   - `cppe5_385`（防疫通道排队人群，框碎到不可用）`cppe5_388`（医疗援助站，多人杂乱）
     `cppe5_394`（路边车窗检测点，街景）`cppe5_434`（街头人群，虚焦）
     `cppe5_452`（Scripps 感谢照，生活照非医疗操作）`cppe5_493`（ICU 隔离病房，防护服+标签难辨）
   - 均 violate 场景红线/标注可用性，她在编辑器里用 X 键现场标记，导出带 `!REJECT`；
     这 6 张不在 `EXCLUDE_MANUAL` 登记范围内（后续如重跑 import_cppe5_batch 会复活为 412 内的图，
     需要再登记 385/388/394/434/452/493——见遗留）。
2. **净增量（120348 备份 → 落库后）**：person -29、mask -22、gloves -31、goggles +3、lab_coat 0，合计 -79 框
   （剔除图带走大头：6 张合计 person-21/mask-15/gloves-23/goggles-3；批内有效修正以删冗余候选为主）。
3. 干跑→正式→复跑三段验证：第二次按 index 重跑「剔除 0 张、落库 54 图」幂等 ✅；
   全库校验 0 坏行 / 0 孤儿图 / 0 孤儿标签。

**执行后底座（B+31 后）**：**train 1,231 图 / 6,048 框（person 1,413 / mask 898 / gloves 1,869 / lab_coat 1,104 / goggles 764）**；
valid 262/1,256、test 732/584 不变；全库 **2,225 图 / 7,888 框**（person 1,572 / mask 1,141 / gloves 2,462 /
lab_coat 1,422 / goggles 1,291）。
**遗留**：`tools/import_cppe5_batch.py` 的 `EXCLUDE_MANUAL` 需增补 {385,388,394,434,452,493}，保证重跑导入脚本不复活这 6 张。

### B+32. labcoat_imagenetcv 1,299 张分类图解压 + 自动初筛（2026-10-11 下午，负责人：许婧雯）

`e58z9vu3fcusg7wzjk91n.zip`（images.cv，lab_coat 单类**分类**集，无 bbox）解压至
`data/raw/labcoat_imagenetcv/data/{train,val,test}/lab_coat/`（1,299 张，MD5 抽样 300 vs 库内命中 0，零重叠）。

新增 `tools/screen_labcoat_imagenetcv.py`（复用 screen_dirty_graphics 判据 + 拼图网格 Hough 检测 +
拉普拉斯模糊度 + 水印边带密度 + 尺寸下限），自动初筛：

- **suspect 355 张（27.3%）**：黑白 118 / 过小 111 / 广告排版 92 / 白底平铺 57 / 拼图 49 / 糊图 18 / 文档 1；
- **clean 944 张** —— 机器判不了"是否实验室场景"，需人工过拼版；
- 产物在 `outputs/labcoat_imagenetcv_screen/`（suspect 8 版 + clean 20 版拼版、两档清单、逐图 TSV）。

人工抽看拼版估计：clean 里真实实验室/医疗场景约 3–4 成，其余为影棚模特照/生活照（场景红线剔）。
**预计最终可用 250–400 张**——超过 LVIS（46 框）一个量级，是 lab_coat 线最大的新增候选来源。
详细分析与人审指引见 `docs/E8_labcoat_imagenetcv_筛查报告_20261011.md`。

**底座不变（B+32 无数据变更，仅 raw 层解压+初筛；底座数字取 B+31 p0_j 落库后实测）**：
train 1,231 图 / 6,048 框（person 1,413 / mask 898 / gloves 1,869 / lab_coat 1,104 / goggles 764）；
valid 262/1,256、test 732/584；全库 **2,225 图 / 7,888 框**。done 清单 575；下一批 p0_k 于本小节后由 B+33 落库（见下）
（队列剩余按并行会话 p0_j 落库后口径，生成时以 `data/_person_review_done.txt` 实时为准）。
待办：① 她过 `_clean_p01–20.jpg` 判场景红线；② 按批生成 lab_coat 画框工具（复用审核编辑器）；
③ 向组长索 `best.pt` 做预标注。

### B+33. person 补标第十一批落库（p0_k，2026-10-11 下午，负责人：许婧雯）

**批次构成**：`outputs/person_review_p0_k/`，index 共 57 图（19 张在她导出中出现 + 38 张未改动按「库内框 + 候选」直接采纳）。
她提交的 `_her_export.txt` 含 22 个图段，其中 **3 张整图剔除**（新上的编辑器 X 键「剔除此图」首次批量实战）。

**落库命令**：

```bash
.venv/Scripts/python.exe tools/apply_person_review.py \
  --export outputs/person_review_p0_k/_her_export.txt \
  --keep-classes 3 \
  --accept-from-ws data/_prelabel/person_p0_k/labels \
  --batch-index outputs/person_review_p0_k/index.txt \
  --reject-tag p0k_reject_20261011
```

**整图剔除明细（3 张，`!REJECT`）**：

| 图 | 剔除原因 | 带走框数（person / mask / gloves） |
|---|---|---|
| `cppe5_51_cbc32941` | 影棚摆拍两人（白幕前摆姿势，非实验室/医疗操作场景，场景红线） | 6 / 2 / 0 |
| `cppe5_52_3bf55944` | 商场人群密集，无任何实验室背景（场景红线） | 14 / 9 / 4 |
| `cppe5_58_740ad7aa` | 街边排队消毒人员（户外街景，非实验室操作场景） | 14 / 9 / 4 |

剔除图 images + labels 已移入 `data/_rejected/p0k_reject_20261011/`（3 组 jpg+txt），并计入 done 清单（不会在后续批次复活）。

**净增量（相对 B+31 底座）**：person **-40** / mask **-20** / gloves **-7** / goggles **+1** = **-66**。
其中 3 张剔除图合计带走 person 34 / mask 20 / gloves 8；剩余 54 图的人工精修另有小幅负向调整（删碎片框、修正误标）。

**执行后底座**：**train 1,228 图 / 5,982 框（person 1,373 / mask 878 / gloves 1,862 / lab_coat 1,104 / goggles 765）**；
valid 262/1,256、test 732/584 不变。
备份 `data/lab_ppe/_label_backup/labels_train_person_20261011_122620.tar.gz`（1,231 文件）。
done 清单 **575 → 635**；全库校验 **7,822 框，0 坏行 / 0 孤儿图 / 0 孤儿标签**。

**验证**：按 index 重跑幂等（剔除 0 张、落库 57 图，`!REJECT` 图因库内已不存在而安全跳过）✅；
剔除拼版 `outputs/person_review_p0_k/_rejected_sheet.jpg` 已生成并目视确认 3 张确为场景红线图。

**队列状态与下一批**：done 635，现存残留未审核图按 `data/_person_review_done.txt` 实时口径继续生成 **p0_l**。
`EXCLUDE_MANUAL` 待增补 {51, 52, 58}（防 `import_cppe5_batch.py` 重跑复活）。

### B+34. labcoat_imagenetcv suspect 355 张整批剔除（2026-10-11 12:27，负责人：许婧雯）

用户拍板「所有怀疑的直接剔除」。`tools/reject_labcoat_imagenetcv.py`（shutil.move 只移不删）把
B+32 筛出的 suspect 355 张（too_small 96 / grayscale 95 / 广告 60 / 平铺 52 / 拼图 33 / 糊 18 / 文档 1，
单图可多因）全部移入 `data/_rejected/labcoat_imagenetcv_screen_20261011/`（保留 split 层级，可回滚）。

核对：raw 池 1,299→944；train 662+247=909、val 135+62=197、test 147+46=193 ✅；
剔除登记 `data/_rejected/labcoat_imagenetcv_rejected_355.txt`；复跑筛查剩余 944 张 0 flags。

⚠ clean 944 张仍含影棚模特照/生活照（机器无法判场景），**保留待人工过 `_clean_p01–20.jpg`**——不是漏剔。
此批只动 `data/raw/` 候选池，**`data/lab_ppe` 底座不变**（取并行会话 p0_k 落库后实测：train 1,228 图 / 5,982 框）。

### B+35. person 补标第十二批落库（p0_l，2026-10-11 下午，负责人：许婧雯）

**批次构成**：`outputs/person_review_p0_l/`，index 共 60 图（25 张出现在她的导出中 + 35 张未改动按「库内框 + 候选」直接采纳）。
本批 **无整图剔除**（`_her_export.txt` 25 个图段全部为精修标签，无 `!REJECT`）——p0_k 连剔 3 张后，本批候选池质量已明显回升。

**落库命令**：

```bash
.venv/Scripts/python.exe tools/apply_person_review.py \
  --export outputs/person_review_p0_l/_her_export.txt \
  --keep-classes 3 \
  --accept-from-ws data/_prelabel/person_p0_l/labels \
  --batch-index outputs/person_review_p0_l/index.txt \
  --reject-tag p0l_reject_20261011
```

**净增量（相对 B+33 底座）**：person **+1** / mask **-3** / gloves **0** / lab_coat **0** / goggles **0** = **-2**。
精修以微调为主（删掉 cppe5_59/609/610/614 等图上重复或误标的 mask 碎片，person 补上少量漏标）。

**执行后底座**：**train 1,228 图 / 5,980 框（person 1,374 / mask 875 / gloves 1,862 / lab_coat 1,104 / goggles 765）**；
valid 262/1,256（person 159 / mask 184 / gloves 471 / lab_coat 281 / goggles 161）、test 732/584 不变。
备份 `data/lab_ppe/_label_backup/labels_train_person_20261011_123834.tar.gz`（1,228 文件）。
done 清单 **635 → 695**；全库校验 **7,820 框，0 坏行 / 0 孤儿图 / 0 孤儿标签**。

**验证**：按 index 干跑幂等（仍为「导出 25 + 未改动采纳 35 = 60 图」，剔除 0 张、防误删保留 0 个）✅。

**队列状态与下一批**：done 695，现存残留未审核图按 `data/_person_review_done.txt` 实时口径继续生成 **p0_m**。
`EXCLUDE_MANUAL` 本批无新增（无剔除图）。

### B+36. person 补标第十三批落库（p0_m，2026-10-11 下午，负责人：许婧雯）

**批次构成**：`outputs/person_review_p0_m/`，index 共 60 图（32 张出现在她的导出中 + 28 张未改动按「库内框 + 候选」直接采纳）。
本批 **无整图剔除**（32 个图段全部为精修标签，无 `!REJECT`），连续第二批零剔除。

**落库命令**：

```bash
.venv/Scripts/python.exe tools/apply_person_review.py \
  --export outputs/person_review_p0_m/_her_export.txt \
  --keep-classes 3 \
  --accept-from-ws data/_prelabel/person_p0_m/labels \
  --batch-index outputs/person_review_p0_m/index.txt \
  --reject-tag p0m_reject_20261011
```

**净增量（相对 B+35 底座）**：person **+6** / mask **-1** / gloves **+1** / lab_coat **0** / goggles **+1** = **+7**。
本批以「补漏标」为主（cppe5_703/724/762/765/807/809 等密集人群图上补齐了漏标 person），并清掉个别 mask 碎片。

**执行后底座**：**train 1,228 图 / 5,987 框（person 1,380 / mask 874 / gloves 1,863 / lab_coat 1,104 / goggles 766）**；
valid 262/1,256、test 732/584 不变。
备份 `data/lab_ppe/_label_backup/labels_train_person_20261011_124720.tar.gz`（1,228 文件）。
done 清单 **695 → 755**；全库校验 **7,827 框，0 坏行 / 0 孤儿图 / 0 孤儿标签**。

**验证**：按 index 干跑幂等（仍为「导出 32 + 未改动采纳 28 = 60 图」，剔除 0 张、防误删保留 0 个）✅。

**队列状态与下一批**：done 755，现存残留未审核图按 `data/_person_review_done.txt` 实时口径继续生成 **p0_n**。
`EXCLUDE_MANUAL` 本批无新增（无剔除图）。

---

### B+37. lab_coat 人工画框工具链搭建（lc13 批，2026-10-11 下午，负责人：许婧雯）

**背景**：B+34 剔除 355 张后，`data/raw/labcoat_imagenetcv/data/{train,val,test}/lab_coat/`
剩 **944 张**（train 662 / val 135 / test 147）通过自动筛查。经她确认（「不用管场景」），
944 张全部采纳为候选。但这批是 **images.cv 分类图**，**无任何 bbox**，不能直接混训 ——
必须先人工画 `lab_coat` 框。

**新增工具（2 个）**：

1. `tools/make_labcoat_label_review.py` —— 批次生成端（复用 `make_box_editor.HTML` 编辑器）
   - 读 `outputs/labcoat_imagenetcv_screen/clean.txt` → 按 `--split` / `--limit` / `--start` 切片
   - 从 raw 池生成编辑器工作区 `outputs/labcoat_label_review_<tag>/editor.html`
   - 产出 `_stem2rel.tsv`（stem → raw 相对路径）供落库端定位原图
   - `--done-file data/_labcoat_label_done.txt` 自动跳过已完成图（防批次回退）
   - 超长边（>1600）生成缩图的坐标仍按原图归一化，标签精度不受影响

2. `tools/apply_labcoat_label.py` —— 落库端
   - 读编辑器导出（`### stem` + 5 列 `cls cx cy w h`），**整图覆盖**工作区标签
   - `!REJECT` 图不写标签，原图**移动**到 `data/_rejected/<tag>/`（只移不删铁律）
   - 正常图：`lc13_` 前缀**复制**原图到 `data/lab_ppe/images/<split>/`，
     标签直写 `data/lab_ppe/labels/<split>/lc13_<stem>.txt`（类别 **3=lab_coat**）
   - split 映射：raw `train→train` / `val→valid` / `test→test`（与库内口径一致）
   - 写前自动 tar 备份 `labels/`；`--dry` 干跑；坐标/类别异常拦截
   - 校验：只允许 lab_coat(3)，坐标须落 `[0,1]`

**首批工作区**：`outputs/labcoat_label_review_lc13a/editor.html`（60 张，队列位置 0~59）。
编辑器类别固定用法：**按 4 键切到 lab_coat**（或直接拖拽，默认 cls 也是 4 需手动选），
一张图多件白大褂 → 画多个框；无白大褂/非实验场景 → **X 键整图剔除**。

**验证**：`parse_export` 单测（多框 + `!REJECT` 解析）✅；
干跑落库链路（3 图合成：2 入 test、1 剔除）split 路由/框计数/剔除去向均正确 ✅。

**队列状态**：944 张待标注，建议 **60 张/批**，首批 `lc13a` 已生成。标注完成后按既有口径
落库（`lc13_` 前缀、三向审计、落库前 tar 备份、E4 记 B+NN）。

### B+38. person 补标第十四批落库（p0_n，2026-10-11 中午，负责人：许婧雯）

**批次构成**：`outputs/person_review_p0_n/`，index 共 60 图（30 张出现在她的导出中 + 30 张未改动按「库内框 + 候选」直接采纳）。
本批 **6 张整图剔除**（`!REJECT`），并首次出现 **istockphoto 视频抽帧图**（10 张 `istockphoto-*-mp4-*` 系列，医护穿脱 PPE 剧照，含 lab_coat 大框，全部正常落库）。

**落库命令**：

```bash
.venv/Scripts/python.exe tools/apply_person_review.py \
  --export outputs/person_review_p0_n/_her_export.txt \
  --keep-classes 3 \
  --accept-from-ws data/_prelabel/person_p0_n/labels \
  --batch-index outputs/person_review_p0_n/index.txt \
  --reject-tag p0n_reject_20261011
```

**整图剔除明细（6 张）**：剔除拼版 `outputs/person_review_p0_n/_rejected_sheet.jpg` 已目视复核，均为「密集人群/混乱场景，框碎到不可用或主体非实验室操作」：

| 图 | 情况 |
|---|---|
| `cppe5_82_9b3c1c33` | 服装厂/仓库杂乱全景，人群细碎、非实验室操作 |
| `cppe5_84_6e3a3d51` | 户外多人防护服休息场景，构图混乱 |
| `cppe5_85_562acd11` | 夜间户外隔离转移人群，框碎 |
| `cppe5_87_1789a26c` | 暗光特写，主体遮挡严重 |
| `cppe5_90_f85dd284` | 帐篷内杂乱，水印（PARTHA PAUL） |
| `cppe5_93_702078e3` | 病房多人杂乱，框碎到不可用 |

剔除图（jpg+txt 扁平结构）已移入 `data/_rejected/p0n_reject_20261011/`，并计入 done 清单。

**净增量（相对 B+36 底座）**：person **+51** / mask **-9** / gloves **-14** / lab_coat **-1** / goggles **+3** = **+30**。
person 大增主因 istockphoto 系列 10 图每张 2–4 个 person 大框入库；剔除 6 张带走部分 mask/gloves 碎框。
「防误删保留框 4 个」——`--keep-classes 3` 生效：4 个库内 lab_coat 框在她导出中消失（IoU<0.5）但按原框保留。

**执行后底座**：**train 1,222 图 / 6,017 框（person 1,431 / mask 865 / gloves 1,849 / lab_coat 1,103 / goggles 769）**；
valid 262/1,256、test 732/584 不变。
备份 `data/lab_ppe/_label_backup/labels_train_person_20261011_125621.tar.gz` 为落库后快照（1,222 文件，person 1,431 / sum 6,017，与下方底座一致）。
done 清单 **755 → 815**；全库校验 **7,857 框，0 坏行 / 0 孤儿图 / 0 孤儿标签**。

**验证**：按 index 复跑幂等（剔除 0 张、底座数字逐项一致）✅。

**队列状态与下一批**：done 815，剩余 331（P0 284 / P1 47），继续生成 **p0_o**。
`EXCLUDE_MANUAL` 待评估：p0_n 剔除 6 张的 cppe5 原编号 {82, 84, 85, 87, 90, 93}（需按 md5 后缀核对唯一性后再登记）。

---

### B+101. lc13a 首批画框落库（60 张 → test +28 / 剔除 31 / 跳过 1，2026-10-11 下午，负责人：许婧雯）

**她贴回编辑器导出**（60 图全覆盖：29 带框 / 31 剔除）。**重要发现：她不止画了 lab_coat，
而是把可见的五类全标了**（person 48 / lab_coat 41 / gloves 11 / goggles 8 / mask 1）——
相当于免费补了其他类标注。落库端校验随之从「只许 cls=3」放宽为五类 0-4，并新增
「保留图必须 ≥1 个 lab_coat」警告与 `--skip-stems` 参数。

**质检**（自动 + 拼图人工复核 `_review_kept_29.jpg` / `_review_rejected_18.jpg`）：
- 41 个 lab_coat 框 100% 被对应 person 框包含（覆盖率>50%），几何一致 ✅
- 剔除 31 张 judgment 合理：穿白大褂的婴孩、万圣节妆、钢笔商品图、航拍人群等
- 2 图剔除标记与框行并存（0BPPMLZZAGEY / 0WLMZ85M5N7U）→ 按剔除处理、框丢弃；
  parse_export 已修 bug（原版 pop 键后遇框行 KeyError 崩溃）并加冲突告警
- `2MOXBGMXQ21M` 为 **Tyvek coverall**（非白大褂，她未标 lab_coat 是正确判断），
  其 mask/goggles/gloves 框有效 → 保留入库
- `193WAVBL1GB6` 医生白大褂漏标 lab_coat（只标 person）→ `--skip-stems` 扣下待补

**落库**：**test +28 图 / +108 框（lab_coat +41）**，前缀 `lc13_`；
剔除 31 张原图移入 `data/_rejected/lc13a_reject_20261011/`（只移不删，可回滚）。
备份 `data/lab_ppe/_label_backup/labels_labcoat_lc13_20261011_130948.tar.gz`。
全库校验 **7,965 框，0 坏行 / 0 孤儿图 / 0 孤儿标签** ✅。
源池剩余 **913 张**（944 − 31 剔除）。

**注意**：本批全部落在 **test** split（源即 raw/test/）。后续 E3 口径的 test 评测
将自动纳入这批更难的实拍图（非图库摆拍），对 lab_coat 泛化评估更有意义。

**下一步**：她补 `193WAVBL1GB6` 的 lab_coat 框（1 个框即可，可用编辑器重开单图批）；
继续 `lc13b`（队列 60~119）。

### B+39. person 补标第十五批落库（p0_o，2026-10-11 下午，负责人：许婧雯）

**批次构成**：`outputs/person_review_p0_o/`，index 共 60 图（20 张出现在她的导出中 + 40 张未改动按「库内框 + 候选」直接采纳）。
本批 **无整图剔除**，且 **20 张精修图全部是 istockphoto 视频抽帧**（`istockphoto-{1389456266,1418389265,1431802973,1496646047,2151031450}` 五个片段，医护穿脱 PPE 连续帧）。

**落库命令**：

```bash
.venv/Scripts/python.exe tools/apply_person_review.py \
  --export outputs/person_review_p0_o/_her_export.txt \
  --keep-classes 3 \
  --accept-from-ws data/_prelabel/person_p0_o/labels \
  --batch-index outputs/person_review_p0_o/index.txt \
  --reject-tag p0o_reject_20261011
```

**净增量（备份差分实测，相对 B+38 底座）**：person **+113** / mask **0** / gloves **-3** / lab_coat **+1** / goggles **-1** = **+110**。
（口径说明：本轮改用 **tar 备份与磁盘逐图差分**，而非类别总数相减——因为并行会话的 lc13 lab_coat 线正在同库写入，
总数相减会被污染。差分结果 60 图变化，与本批 index 完全吻合，可信。）

**⚠️ 跨线影响记录**：`data/lab_ppe` 的 **test split 已由并行会话 lc13 线写入 28 张 lab_coat 图**
（`lc13_*`，12:03 落盘）→ test 由 732 图/584 框变为 **760 图/692 框**。**这不是 person 线的动作**，
person 线只写 train。后续 person 批次对比 base 时，test 数字请以 lab_coat 线口径为准。

**执行后底座**：**train 1,222 图 / 6,127 框（person 1,544 / mask 865 / gloves 1,846 / lab_coat 1,104 / goggles 768）**；
valid 262/1,256 不变；test 760/692（含 lc13）。
备份 `data/lab_ppe/_label_backup/labels_train_person_20261011_131051.tar.gz`（1,222 文件，落库前 = B+38 底座）。
done 清单 **815 → 875**；train 全库校验 0 坏行 / 0 孤儿。
「防误删保留框 11 个」——istockphoto 系列的 lab_coat 大框在她导出中未出现（IoU<0.5）但按原框保留。

**验证**：按 index 干跑幂等（60 图、剔除 0 张、底座数字不变）✅。

**队列状态与下一批**：done 875，剩余 271（P0 224 / P1 47），继续生成 **p0_p**。
`EXCLUDE_MANUAL` 本批无新增（无剔除图）。

### B+40. person 补标第十六批落库（p0_o 之后 · p0_p，2026-10-11 下午，负责人：许婧雯）

**批次构成**：`outputs/person_review_p0_p/`，index 共 60 图（17 张出现在她的导出中 + 42 张未改动按「库内框 + 候选」直接采纳，另 1 张整图剔除）。
本批以 **istockphoto 视频抽帧**为主（2169298710 / 472933717 / 619165530 / 809879018 / 809904308 / 809958214 / 810129952 等
多组连续帧，医护穿脱 PPE），另含 3 张单类补标图（`k2` / `k3` / `w-11-`）。

**落库命令**：

```bash
.venv/Scripts/python.exe tools/apply_person_review.py \
  --export outputs/person_review_p0_p/_her_export.txt \
  --keep-classes 3 \
  --accept-from-ws data/_prelabel/person_p0_p/labels \
  --batch-index outputs/person_review_p0_p/index.txt \
  --reject-tag p0p_reject_20261011
```

**整图剔除（1 张）**：`istockphoto-810129952-640_adpp_is_mp4-0015`（该帧只剩局部手臂/半身，lab_coat 与 mask 均无法标稳）。
剔除图已移入 `data/_rejected/p0p_reject_20261011/`。

**净增量（备份差分实测，相对 B+39 底座）**：person **+125** / mask **-6** / gloves **-5** / lab_coat **+4** / goggles **-1** = **+117**。
（`--keep-classes 3` 防误删保留 5 个 lab_coat 框。）

**执行后底座**：**train 1,221 图 / 6,244 框（person 1,669 / mask 859 / gloves 1,841 / lab_coat 1,108 / goggles 767）**；
valid 262/1,256 不变；test 760/692（含并行会话 lc13 写入的 28 张，见 B+102）。
备份 `data/lab_ppe/_label_backup/labels_train_person_20261011_131929.tar.gz`（1,222 文件，落库前 = B+39 底座）。
done 清单 **875 → 935**；全库校验 **0 坏行 / 0 孤儿图 / 0 孤儿标签**。

**验证**：按 index 干跑幂等（59 图、剔除 1 张、底座数字不变）✅。

**队列状态与下一批**：done 935，剩余 211（P0 164 / P1 47），继续生成 **p0_q**。
`EXCLUDE_MANUAL` 待评估：剔除图 `istockphoto-810129952-...-0015`（非 cppe5 编号，走 workfile 排除逻辑，非 EXCLUDE_MANUAL）。

---

### B+102. lc13b 第二批画框落库（60 张 → test +31 / 剔除 29，2026-10-11 下午，负责人：许婧雯）

**导出 60 图全覆盖**（31 带框 / 29 剔除），0 漏标（每张保留图都有 lab_coat 框）、0 框行冲突。

**落库**：**test +31 图 / +108 框（lab_coat 49 / person 52 / goggles 6 / gloves 1）**；
剔除 29 张（含 193WAVBL1GB6——她复核后判定为图库摆拍，直接剔除，B+38 的待补项就此关闭）
→ `data/_rejected/lc13b_reject_20261011/`。
备份 `data/lab_ppe/_label_backup/labels_labcoat_lc13_20261011_132957.tar.gz`。
全库校验 **8,300 框，0 坏行 / 0 孤儿图 / 0 孤儿标签** ✅。源池剩余 **884 张**。

**lc13 系累计（B+101+B+102）**：test **+59 图 / +216 框（lab_coat +90）**，剔除 60 张。

**下一步**：`lc13c`（队列继续，全 val/test 段将陆续过渡到 train 段 662 张主战场）。

### B+41. person 补标第十七批落库（p0_q，2026-10-11 下午，负责人：许婧雯）

**批次构成**：`outputs/person_review_p0_q/`，index 共 60 图（37 张出现在她的导出中 + 23 张未改动按「库内框 + 候选」直接采纳）。
本批 **无整图剔除**，构成为 `w-*` 系列（w-2/w-3/w-5/w-8/w-13/w-22…w-62，多为 lab_coat + goggles 半身像）
与 `web_00xx` 系列（web_0032/0038/0040/0055/0057/0058/0071/0075 等小尺寸网络图）。

**落库命令**：

```bash
.venv/Scripts/python.exe tools/apply_person_review.py \
  --export outputs/person_review_p0_q/_her_export.txt \
  --keep-classes 3 \
  --accept-from-ws data/_prelabel/person_p0_q/labels \
  --batch-index outputs/person_review_p0_q/index.txt \
  --reject-tag p0q_reject_20261011
```

**净增量（备份差分实测，相对 B+40 底座）**：person **+74** / mask **-1** / gloves **-1** / lab_coat **+2** / goggles **0** = **+74**。
（`--keep-classes 3` 防误删保留 11 个 lab_coat 框。）

**执行后底座**：**train 1,221 图 / 6,318 框（person 1,743 / mask 858 / gloves 1,840 / lab_coat 1,110 / goggles 767）**；
valid 262/1,256 不变。
⚠️ test 已由并行 lc13 线继续扩至 **791 图 / 800 框**（较 B+103 又增 31 图，非 person 线动作）。
备份 `data/lab_ppe/_label_backup/labels_train_person_20261011_133529.tar.gz`（1,221 文件，落库前 = B+40 底座）。
done 清单 **935 → 995**；train 全库校验 **0 坏行 / 0 孤儿图 / 0 孤儿标签**。

**验证**：按 index 干跑幂等（60 图、剔除 0 张、底座数字不变）✅。

**队列状态与下一批**：done 995，剩余 151（P0 104 / P1 47），继续生成 **p0_r**——person 补标队列预计 3 批内清空。
`EXCLUDE_MANUAL` 本批无新增（无剔除图）。

---

### B+103. lc13c 第三批画框落库（60 张 → train +19 / test +16 / 剔除 25，2026-10-11 下午，负责人：许婧雯）

**导出 60 全覆盖**（35 带框 / 25 剔除），0 漏标 0 冲突。
**落库**：**train +19 图（lc13 首批进 train）** / test +16 图，合计 **35 图 / 132 框**
（lab_coat 55 / person 54 / gloves 13 / goggles 6 / mask 4）。
剔除 25 → `data/_rejected/lc13c_reject_20261011/`。
全库校验 **8,506 框，0 坏行 / 0 孤儿** ✅。done 清单 **179**；源池剩余 **859 张**。

**lc13 系累计（B+101~103）**：**+94 图 / +356 框（lab_coat +145）**，剔除 84 张。
test 段 194 张已全部过完（剩 0）；后续批次纯 train 段（662 张）+ val 段（135 张）。

**质检备注**：本批保留图中 stock 摆拍占比偏高（RYBV5THN1TS9 / UCU4HUPJOW0C /
V58Q1Q634UPD / YVBZCHAPRP3N 等为图库人像风格）。均在 train 段，作多样性样本可接受；
若后续 E3 复测仍见「只认摆拍」倾向，可考虑对 lc13_* 摆拍子集做降采样。

---

### B+42. person 补标第十八批落库（p0_r，2026-10-11 下午，负责人：许婧雯）

**批次构成**：`outputs/person_review_p0_r/`，index 共 60 图（22 张出现在她的导出中 + 38 张未改动按「库内框 + 候选」直接采纳）。
本批 **无整图剔除**，构成为 `web_0133`…`web_0387` 小尺寸网络图与 `web_p106918230` / `web_p110847673` / `web_p116301490` 等图库人像系列，
以 person 补漏标为主（多张由 1 框补至 3~5 框）。

**落库命令**：

```bash
.venv/Scripts/python.exe tools/apply_person_review.py \
  --export outputs/person_review_p0_r/_her_export.txt \
  --keep-classes 3 \
  --accept-from-ws data/_prelabel/person_p0_r/labels \
  --batch-index outputs/person_review_p0_r/index.txt \
  --reject-tag p0r_reject_20261011
```

**净增量（备份差分实测，相对 B+41 底座）**：**+84 框**。
备份 vs 磁盘逐图差分：1240 → 1240 图（无新增/无删除图），common 段 **+84 框**，全部落在 `web_*` 系列
（`web_p116301490` +5、`web_p110847673` +5、`web_0188` +5、`web_0269` +3、`web_0267` +3…），与导出图段完全吻合。
（`--keep-classes 3` 防误删保留 **5** 个 lab_coat 框：`web_0145` ×2、`web_0215` ×1、`web_0292` ×2。）

**执行后底座**：**train 1,240 图 / 6,483 框（person 1,856 / mask 862 / gloves 1,850 / lab_coat 1,144 / goggles 771）**；
valid 262/1,256 不变。
⚠️ **train 图数 1,221 → 1,240（+19）非本批 person 线动作**：并行 lc13 线（B+103 lc13c）已将首批 19 张 `lc13_*` 写入 train 段，
故本批净增量**必须**用备份差分（按 stem 逐图比）而非类别总数相减——差分结果与 index 图数完全一致，person +84 为真实值。
⚠️ test 807 图 / 851 框同样含 lc13 线写入，非 person 线动作。
备份 `data/lab_ppe/_label_backup/labels_train_person_20261011_135239.tar.gz`（1,240 文件，落库前 = B+41 底座 + lc13c 首批 19 图）。
done 清单 **995 → 1055**；train 全库校验 **0 坏行 / 0 孤儿图 / 0 孤儿标签**。

**验证**：按 index 干跑幂等（60 图、剔除 0 张、防误删保留 5 框、底座数字不变）✅。

**队列状态与下一批**：done 1055，剩余约 91（P0 44 / P1 47），继续生成 **p0_s**——person 补标队列预计 2 批内清空。
`EXCLUDE_MANUAL` 本批无新增（无剔除图）。

---

### B+104. lc13d 第四批画框落库（60 张 → train +34 / 剔除 26，2026-10-11 下午，负责人：许婧雯）

**导出 60 全覆盖**（34 带框 / 26 剔除），0 漏标 0 冲突。首批**纯 train 段**批次。
**落库**：**train +34 图 / +158 框（lab_coat 65 / person 75 / goggles 10 / gloves 7 / mask 1）**；
剔除 26 → `data/_rejected/lc13d_reject_20261011/`。
全库校验 **8,748 框，0 坏行 / 0 孤儿** ✅。done 清单 **239**；源池剩余 **833 张**。

**lc13 系累计（B+101~104）**：**+128 图 / +514 框（lab_coat +210）**，剔除 110 张。
train 段进度：**128/662（约 19%）**；val 段 135 张未动。

**质检备注**：本批含较多白大褂人像/半身照（非操作场景），框几何正确；密集人群图
（277NA1SYM6AG 6人 / 27TE9YPM9V40 7人 / 3TCLLM9AVCQA 7人 / 4CJQX0AIO14I 5人）
她均逐人分开标注，质量好。

---

### B+43. person 补标第十九批落库（p0_s，2026-10-11 下午，负责人：许婧雯）

**批次构成**：`outputs/person_review_p0_s/`，index 共 60 图（**12 张**出现在她的导出中 + 48 张未改动按「库内框 + 候选」直接采纳）。
本批 **无整图剔除**，构成为 `web_p1xxxxxx` 系列（web_p155720380 / p156651720 / p157472853 / p183952725 / p184778185 /
p184996318 / p184996331 / p184996348 等医护场景）与 `web_p5xxxxxxx` 系列（web_p51263811 / p51264941 / p51266085 / p51927941 等）。
其中 `web_p183952725` 为群体肖像（15 框：person3 + mask5 + lab_coat3 + goggles1 + gloves... 前段补标的密集小框），精修后保持完整。

**落库命令**：

```bash
.venv/Scripts/python.exe tools/apply_person_review.py \
  --export outputs/person_review_p0_s/_her_export.txt \
  --keep-classes 3 \
  --accept-from-ws data/_prelabel/person_p0_s/labels \
  --batch-index outputs/person_review_p0_s/index.txt \
  --reject-tag p0s_reject_20261011
```

**净增量（备份差分实测，相对 B+42 底座）**：**+61 框**。
备份 vs 磁盘逐图差分：1274 → 1274 图（无新增/无删除图），common 段 **+61 框**，
集中落在 `web_p*` 系列（`web_p64233535` +3、`web_p51263872` +3、`web_p183952725` +3、`web_p64234578` +2、`web_p51265753` +2…），与导出图段一致。
（`--keep-classes 3` 防误删保留 **8** 个 lab_coat 框：web_p155720380 ×1、web_p184996318 ×1、web_p184996331 ×2、web_p184996348 ×2、web_p51266085 ×2。）

**执行后底座**：**train 1,274 图 / 6,702 框（person 1,991 / mask 863 / gloves 1,857 / lab_coat 1,210 / goggles 781）**；
valid 262/1,256 不变。
⚠️ **train 图数 1,240 → 1,274（+34）非本批 person 线动作**：并行 lc13 线（**B+104 lc13d 第四批**，纯 train 段）已将其 34 张 `lc13_*` 写入 train，
故本批净增量**必须**用备份差分（按 stem 逐图比）而非类别总数相减——差分结果与 index 图数完全吻合，person +61 为真实值。
⚠️ test 807 图 / 851 框同样含 lc13 线写入，非 person 线动作。
备份 `data/lab_ppe/_label_backup/labels_train_person_20261011_140717.tar.gz`（1,274 文件，落库前 = B+42 底座 + lc13d 34 图）。
done 清单 **1,055 → 1,115**；train 全库校验 **0 坏行 / 0 孤儿图 / 0 孤儿标签**。

**验证**：按 index 干跑幂等（60 图、剔除 0 张、防误删保留 8 框、底座数字不变）✅。

**队列状态与下一批**：done 1,115，剩余 **84**（P0 37 / P1 47），继续生成 **p0_t**（60 图）——person 补标队列预计 2 批内清空。
`EXCLUDE_MANUAL` 本批无新增（无剔除图）。

---

### B+105. lc13e 第五批画框落库（60 张 → train +32 / 剔除 28，2026-10-11 下午，负责人：许婧雯）

**导出 60 全覆盖**（32 带框 / 28 剔除），0 漏标。
1 处剔除+框行冲突（76P4XHJRJZJR：先画整图框后反悔剔除）→ 按剔除优先处理。
**落库**：**train +32 图 / +130 框（lab_coat 46 / person 65 / gloves 11 / goggles 8）**；
剔除 28 → `data/_rejected/lc13e_reject_20261011/`。
全库校验 **8,939 框，0 坏行 / 0 孤儿** ✅。done 清单 **299**；源池剩余 **805 张**。

**lc13 系累计（B+101~105）**：**+160 图 / +644 框（lab_coat +256）**，剔除 138 张。
train 段进度：**160/662（约 24%）**；val 段 135 张未动。

**质检备注**：本批实验室/医疗操作场景占比回升（5OPGXIXC4MI9 五人实验台、
5QMQ3AQ739CS 五人、7YLPUMC91383 等均为真实操作图）；亦含少量人像与
婴孩穿白大褂（77PQ2HVJSNVQ）、户外灰大衣（77RM8EDS7301）等边缘样本，
框几何正确，留作难度样本。

---

### B+44. person 补标第二十批落库（p0_t，2026-10-11 下午，负责人：许婧雯）

**批次构成**：`outputs/person_review_p0_t/`，index 共 60 图（**25 张**出现在她的导出中 + 35 张未改动按「库内框 + 候选」直接采纳）。
本批 **无整图剔除**，构成为 3 张 `web_p904xxxxx` 系列（web_p90407325 / 90407328 / 90407329，多人穿脱 PPE 场景）+ **22 张 `lcsi_crop` 族**
（`lcsi_03/07/10/17/27/34/46/48` + `lcsi_frame_00003…00100`，均为单人 person 全身/半身裁剪图，多数由空标签补至 1 框）。

**落库命令**：

```bash
.venv/Scripts/python.exe tools/apply_person_review.py \
  --export outputs/person_review_p0_t/_her_export.txt \
  --keep-classes 3 \
  --accept-from-ws data/_prelabel/person_p0_t/labels \
  --batch-index outputs/person_review_p0_t/index.txt \
  --reject-tag p0t_reject_20261011
```

**净增量（备份差分实测，相对 B+43 底座）**：**+34 框**。
备份 vs 磁盘逐图差分：1306 → 1306 图（无新增/无删除图），common 段 **+34 框**，
完全对应导出图段：`web_p90407325` +4、`web_p90407329` +3、`web_p90407328` +3，以及 22 张 `lcsi_*` 各 +1~+2（多数 0→1）。
（`--keep-classes 3` 防误删保留 **0** 个框：本批 lcsi 裁剪图无 lab_coat 标注。）

**执行后底座**：**train 1,306 图 / 6,866 框（person 2,090 / mask 863 / gloves 1,868 / lab_coat 1,256 / goggles 789）**；
valid 262/1,256 不变。
⚠️ **train 图数 1,274 → 1,306（+32）非本批 person 线动作**：并行 lc13 线（**B+105 lc13e 第五批**，纯 train 段）已将其 32 张 `lc13_*` 写入 train，
故本批净增量**必须**用备份差分（按 stem 逐图比）而非类别总数相减——差分结果与 index 图数完全吻合，person +34 为真实值。
⚠️ test 807 图 / 851 框同样含 lc13 线写入，非 person 线动作。
备份 `data/lab_ppe/_label_backup/labels_train_person_20261011_142034.tar.gz`（1,306 文件，落库前 = B+43 底座 + lc13e 32 图）。
done 清单 **1,115 → 1,175**；train 全库校验 **0 坏行 / 0 孤儿图 / 0 孤儿标签**。

**验证**：按 index 干跑幂等（60 图、剔除 0 张、防误删保留 0 框、底座数字不变）✅。

**队列状态与下一批**：done 1,175，剩余 **56**（P0 32 / P1 24），继续生成 **p0_u**（56 图，即**最后一批**）——person 补标队列将在 p0_u 后清空。
`EXCLUDE_MANUAL` 本批无新增（无剔除图）。

---

### B+45. person 补标第二十一批落库（p0_u，最后一批核心，2026-10-11 下午，负责人：许婧雯）

**批次构成**：`outputs/person_review_p0_u/`，index 共 56 图（**5 张**出现在她的导出中 + 51 张未改动按「库内框 + 候选」直接采纳）。
本批为 person 补标队列的**收尾批**，导出精修仅 5 张（`lcsi_frame_00101` + `cppe5_461_3dc58375` + `web_0355` + `web_0386` + `web_p110877089`），
其余 51 张经她确认「未改动即通过」，**无整图剔除**。

**落库命令**：

```bash
.venv/Scripts/python.exe tools/apply_person_review.py \
  --export outputs/person_review_p0_u/_her_export.txt \
  --keep-classes 3 \
  --accept-from-ws data/_prelabel/person_p0_u/labels \
  --batch-index outputs/person_review_p0_u/index.txt \
  --reject-tag p0u_reject_20261011
```

**净增量（备份差分实测，相对 B+44 底座）**：**+23 框**。
备份 vs 磁盘逐图差分：1347 → 1347 图（无新增/无删除图），common 段 **+23 框**，
对应导出：`cppe5_461_3dc58375` +3（原空标签→person+mask+goggles）、`web_p110877089` +2，及 `web_*` 系列多张 0→1~2。
（`--keep-classes 3` 防误删保留 **0** 个框。）

**执行后底座**：**train 1,347 图 / 7,063 框（person 2,200 / mask 865 / gloves 1,878 / lab_coat 1,328 / goggles 792）**；
valid 262/1,256 不变。
⚠️ **train 图数 1,306 → 1,347（+41）非本批 person 线动作**：并行 lc13 线继续往 train 写 `lc13_*`。
故本批净增量**必须**用备份差分（按 stem 逐图比）而非类别总数相减——差分结果与 index 图数完全吻合，person +23 为真实值。
⚠️ test 807 图 / 851 框同样含 lc13 线写入，非 person 线动作。
备份 `data/lab_ppe/_label_backup/labels_train_person_20261011_143317.tar.gz`（1,347 文件，落库前 = B+44 底座 + lc13 新增）。
done 清单 **1,175 → 1,231**；train 全库校验 **0 坏行 / 0 孤儿图 / 0 孤儿标签**。

**验证**：按 index 干跑幂等（56 图、剔除 0 张、防误删保留 0 框、底座数字不变）✅。

**⚠️ 队列与 lc13 线交互（重要）**：p0_u 落库后重扫队列，**仍有 41 图待审，全部为 `lc13_*` 系列**
（`lc13_8ISUS0WL1POW` 等，即并行 lc13 线新写入 train 段的 lab_coat 画框图）。原因：person 审核工具的队列来源是
「train 段中尚未确认 person 标签的图」，lc13 线每写一批新图，person 队列就会自动纳入这些新图。
**这不是遗漏，而是两线共库的必然结果**——结论：**person 补标队列不会真正「清空」，只要 lc13 线继续往 train 写图**。

**person 补标线累计（B+35~45，共 10 批）**：各批净增 person 框依次为
p0_l -2 / p0_m +7 / p0_n +30 / p0_o +110 / p0_p +117 / p0_q +74 / p0_r +84 / p0_s +61 / p0_t +34 / p0_u +23 = **本线贡献 person 框净 +538**；
落库图累计 **约 616 图**（每批 60 图 ×9 + 最后一批 56）。场景红线整图剔除累计 **10 张**（p0_k 3 + p0_n 6 + p0_p 1）。
（注：train 段 person 总数同期 1,376 → 2,200 的增幅还包含 lc13 线画框图自带的 person 标注，**不能**与本线 +538 直接相抵。）

**后续建议**：
1. person 补标主体工作**已收口**；剩余 `lc13_*` 新图的 person 复核可作为**常态化尾随任务**，随 lc13 线批次滚动处理即可，无需当作独立大批次。
2. 若 lc13 线继续大规模写 train，建议在 person 审核工具中**默认排除 `lc13_*`**（或单独标记「lc13 图 person 复核」子队列），避免两线队列互相干扰。
→ **已于 B+47 实施**（`make_person_review.py` 新增 `--exclude-prefix`，默认 `lc13_`）。
`EXCLUDE_MANUAL` 本批无新增（无剔除图）。

---

### B+106. lc13f 第六批画框落库（60 张 → train +41 / 剔除 18 / 待补 1，2026-10-11 下午，负责人：许婧雯）

**导出 60 全覆盖**（42 保留 / 18 剔除）。
**落库**：**train +41 图 / +175 框（person 90 / lab_coat 72 / gloves 10 / goggles 2 / mask 1）**；
剔除 18 → `data/_rejected/lc13f_reject_20261011/`。
**跳过 1 图待补标**：`9B0ZP1V0MDNT`（显微镜前研究员，典型实验室场景但漏标 lab_coat，
图保留在源池 `train/lab_coat/`，done 清单未登记，下批补框后入库）。
全库校验 **9,170 框（train 7,063 / valid 1,256 / test 851），0 坏行 / 0 孤儿** ✅。
done 清单 **299 → 358**；源池剩余 **787 张**（train 577 / val 135 / test 75）。

**lc13 系累计（B+101~106）**：**+201 图 / +810 框（lab_coat +328）**，剔除 156 张。
train 段进度：**192/662（约 29%）**；test 段 75 张全部完成；val 段 135 张未动。
复核拼图：`outputs/labcoat_label_review_lc13f/_review_kept_42.jpg` / `_review_rejected_18.jpg`。
新增通用复核工具 `tools/make_labcoat_review_sheet.py`（此前各批为内联脚本，本批固化复用）。

**质检备注**：剔除合理——影棚人像摆拍（8TMKI19HSJ5A、9F6F0YSVWK7 水印）、
万圣节妆（9GSX9X7L96BX）、人台商品图（9PGOZ0LQXM2P）、生活照（8ZOBYFW71CVW）等；
保留图中 9HXZ825Y1G32（户外玉米地灰大衣）、A7M5D2TS8Q3D（影棚）等边缘样本
框几何正确，留作难度样本。

---

### B+47. person 审核工具加固：默认排除 `lc13_*`（2026-10-11 下午，负责人：许婧雯）

**背景**：见 B+45——person 补标主体收口后重扫队列仍剩 41 图，全部为并行 lc13 线写入 train 的 `lc13_*`。
根因：`make_person_review.py` 的队列来源是「train 段中尚未确认 person 标签的图」，lc13 线每写一批新图即被 person 队列自动纳入，
导致 **person 队列永不真正清空**、两线队列互相干扰。

**改动**（`tools/make_person_review.py`）：新增 `--exclude-prefix` 参数，**默认值 `lc13_`**。
- 默认行为：生成 person 批次时按前缀跳过 `lc13_*`，person 队列从此可真正收敛。
- 逃生开关：传 `--exclude-prefix ""` 可关闭排除（或传其他前缀集合，逗号分隔），用于需要专门复核 lc13 图片 person 的场景。

**验证**：
- 默认（排除 lc13_）：`[train] 存量图 0（P0 0 / P1 0），总队列 0` → **空批**，person 补标队列正式清空 ✅。
- 关闭排除（`--exclude-prefix ""`）：`存量图 41（P0 41），总队列 41` → 与原行为一致，41 张 `lc13_*` 仍在 ✅。

**结论**：person 补标主体工作**完全收口**（默认队列 0）；`lc13_*` 图的 person 复核转为**按需触发**的独立任务
（需要时用 `--exclude-prefix ""` 或指定 `--stems` 单独拉批），不再阻塞 person 主队列。

---

### B+107. lc13g 第七批画框落库（60 张 → train +40 / 剔除 19 / 待补 1，2026-10-11 下午，负责人：许婧雯）

**导出 60 全覆盖**（41 保留 / 19 剔除），0 坏行 / 0 越界。
**上批待补项关闭后新增一件**：
- ✅ `9B0ZP1V0MDNT`（显微镜前研究员）本批队首补标 lab_coat + person，正常入库；
- ⚠️ **`BU4ZPSTGQ7ST` 漏标 lab_coat**（真实实验室白大褂+蓝手套场景，只标了 gloves+person）
  → `--skip-stems` 扣下，图留在源池，done 未登记，下批队首补框。
1 处剔除+框冲突（DXOA55NEGIC0：生活照白夹克，剔除正确，1 框丢弃）→ 按剔除优先处理。
**落库**：**train +40 图 / +169 框（person 76 / lab_coat 68 / gloves 16 / goggles 12，本批无 mask）**；
剔除 19 → `data/_rejected/lc13g_reject_20261011/`。
全库校验 **9,339 框（train 7,232 / valid 1,256 / test 851），0 坏行 / 0 孤儿** ✅。
done 清单 **358 → 417**；源池剩余 **768 张**（train 558 / val 135 / test 75）。

**lc13 系累计（B+101~107）**：**+241 图 / +979 框（lab_coat +396 / person +456 / gloves +68 / goggles +52 / mask +7）**，剔除 175 张。
train 段进度：**232/662（约 35%）**；val 段 135 张未动。
复核拼图：`outputs/labcoat_label_review_lc13g/_review_kept_41.jpg` / `_review_rejected_19.jpg`。

**质检备注**：保留图中含若干边缘样本——骷髅穿白大褂（DYRNJEQ9C7QF）、蓝发人偶（DMFYCQIJMIWS）、
户外航拍远景（C3TC3WHRMX5K）、番茄手持特写（CHZ4M336DN5F）等，框几何正确，留作难度样本；
若 E3 复测 lab_coat 误检上升，可优先对本类边缘子集降采样。

---

### B+108. lc13h 第八批画框落库（60 张 → train +44 / 剔除 15 / 待定 1，2026-10-11 下午，负责人：许婧雯）

**导出 60 全覆盖**（45 保留 / 15 剔除），0 坏行 / 0 越界。
1 处剔除+框冲突（GFHUX0IU3Z3E：`!REJECT` 后跟 3 个 lab_coat 框）→ 按剔除优先，框丢弃。
**⚠️ `BU4ZPSTGQ7ST` 连续两批未标 lab_coat**（真实实验室白大褂+蓝手套研究员，本批只标
gloves×2 + person；上批同样漏）→ 再次 `--skip-stems` 扣下，**待用户明确意图**：
若判为 coverall/不收 → 整图剔除；若应收 → 补 1 个 lab_coat 框。图留源池，done 未登记。
**落库**：**train +44 图 / +185 框（person 97 / lab_coat 69 / gloves 16 / goggles 5 / mask 1）**；
剔除 15 → `data/_rejected/lc13h_reject_20261011/`。
全库校验 **9,524 框（train 7,417 / valid 1,256 / test 851），0 坏行 / 0 孤儿** ✅
（落库脚本重跑幂等：第二次执行剔除源图缺失但不影响已入库数据）。
done 清单 **417 → 476**；源池剩余 **753 张**（train 543 / val 135 / test 75）。

**lc13 系累计（B+101~108）**：**+285 图 / +1,164 框（lab_coat +465 / person +552）**，剔除 190 张。
train 段进度：**276/662（约 42%）**；val 段 135 张未动。
复核拼图：`outputs/labcoat_label_review_lc13h/_review_kept_45.jpg` / `_review_rejected_15.jpg`。

**质检备注**：本批影棚/图库摆拍占比回升（工作证照、圣诞帽摆拍、婚礼白礼服边缘样本等），
亦有 FFYHC2LMWGUZ 集市人群（16 person 逐个标）等高密度样本，框几何正确。

---

### B+109. lc13i 第九批画框落库（60 张 → train +42 / 剔除 18，2026-10-11 下午，负责人：许婧雯）

**导出 60 全覆盖**（42 保留 / 18 剔除），0 坏行 / 0 越界 / 0 剔除冲突。
**✅ `BU4ZPSTGQ7ST` 悬案关闭**：她最终按 X 整图剔除（连续三批不标 → 明确不收），已移入 `_rejected`。
**⚠️ 新口径问题——coverall 两张保留但无 lab_coat**：
`HYXBKF2PA0HW`（实验室穿白色连体防护服+蓝手套，标 gloves+person）、
`KIF16IEVN1QV`（工业 coverall+全面罩+长胶手套，标 mask+goggles+person）。
两图均为 **coverall 连体服**，她保留并标了其他 PPE 类但**未标 lab_coat**——
与并行会话 route A 决策（coverall 也算 lab_coat）存在口径分歧，待统一：
若按 route A，这两张需补 lab_coat 框；若维持她的区分口径，则维持现状
（其他类框有效已入库，对 lab_coat 训练无贡献）。
**落库**：**train +42 图 / +215 框（person 93 / lab_coat 80 / gloves 21 / goggles 15 / mask 6）**；
剔除 18 → `data/_rejected/lc13i_reject_20261011/`。
全库校验 **9,739 框（train 7,632 / valid 1,256 / test 851），0 坏行 / 0 孤儿** ✅。
done 清单 **476 → 536**；源池剩余 **735 张**（train 525 / val 135 / test 75）。
**待补标队列清零**（BU4Z 已剔除，无悬留图）。

**lc13 系累计（B+101~109）**：**+327 图 / +1,379 框（lab_coat +545 / person +645）**，剔除 208 张。
train 段进度：**318/662（约 48%）**；val 段 135 张未动。
复核拼图：`outputs/labcoat_label_review_lc13i/_review_kept_42.jpg` / `_review_rejected_18.jpg`。

**质检备注**：本批高密度样本多（HW1LZAE8IMZ0 13 白大褂+6 goggles、JZO923R26UB5 7+8、
K34ROO7M8LR2 9 白大褂逐件分开标），标注质量好；含 J2X7IMYJSQQH 歌手白外套、
HVYEMUIXNQRU 壁画前摆拍等边缘样本，框几何正确留作难度样本。

---

### B+110. lc13j 第十批画框落库（60 张 → train +44 / 剔除 15 / 待补 1，2026-10-11 下午，负责人：许婧雯）

**导出 60 全覆盖**（45 保留 / 15 剔除），0 坏行 / 0 越界 / 0 剔除冲突。
**⚠️ 新待补 1 图**：`MLMW7NETUH4T`（诊室写字白大褂女医生，典型 lab_coat 场景但只标了
person，明显漏标非 coverall 口径问题）→ `--skip-stems` 扣下，图留源池，done 未登记，下批队首补框。
**B+49 遗留确认**：coverall 两张（HYXBKF2PA0HW/KIF16IEVN1QV）维持她的区分口径现状，
若她后续按 route A 补框另行处理。
**落库**：**train +44 图 / +147 框（person 72 / lab_coat 59 / gloves 8 / goggles 8 / mask 1）**；
剔除 15 → `data/_rejected/lc13j_reject_20261011/`。
全库校验 **9,886 框（train 7,779 / valid 1,256 / test 851），0 坏行 / 0 孤儿** ✅。
done 清单 **536 → 595**；源池剩余 **720 张**（train 510 / val 135 / test 75）。

**lc13 系累计（B+101~110）**：**+371 图 / +1,526 框（lab_coat +604 / person +716）**，剔除 223 张。
train 段进度：**362/662（约 55%）**；val 段 135 张未动。
复核拼图：`outputs/labcoat_label_review_lc13j/_review_kept_45.jpg` / `_review_rejected_15.jpg`。

**质检备注**：本批图库人像摆拍占比高（白大褂职业照），真实实验室操作场景偏少；
KMJEABN5KEOC 首次出现 mask(1) 框（外科口罩医生）。框几何均正确。

---

### B+111. lc13k 第十一批画框落库（300 张 → train +169 / valid +64 / 剔除 66 / 待补 1，2026-10-11 傍晚，负责人：许婧雯）

**本批为一次性 300 张大批次**（应她要求），导出 300 张中 299 张有内容、**1 张漏标**。
**⚠️ 新待补 1 图**：`ZSMH2IW3X5Z8`（序位 211/300，train/lab_coat）导出中未出现，
判定为她导出时漏掉（非剔除）→ `--skip-stems` 扣下，图留源池，done 未登记，下批队首补框。
**落库**：**train +169 图 / valid +64 图**，共 **+899 框**
（person 445 / lab_coat 365 / gloves 46 / goggles 35 / mask 8）；
剔除 66 → `data/_rejected/lc13k_reject_20261011/`（全为 val 段，队列已跨入 val）。
保留但无 lab_coat 框 3 图（警告保留）：`Q6OQ3CFL3R9E`(0,1)、`VV7ISCWXHC89`(0)、`YYLDEL5TZR5N`(0,2)。

**里程碑：首次带入 valid 段**（+64 图），train 段队列接近尾声，本轮起队列混入 val 图。
全库校验 **10,785 框（train 8,419 / valid 1,515 / test 851），0 坏行 / 0 孤儿** ✅。
done 清单 **595 → 894**；源池剩余 **421 张**（train 约 286 / val 71 / test 75）。

**lc13 系累计（B+101~111）**：**+604 图 / +2,425 框（lab_coat +969 / person +1,161）**，剔除 289 张。
复核拼图：`outputs/labcoat_label_review_lc13k/_review_kept_233.jpg` / `_review_rejected_66.jpg`。

**质检备注**：本批高质量实验室/医疗实拍比例上升，goggles(35)/mask(8) 明显增多，
P7QRUE1S3DC3/SLXRD2M32X7W/H9G30BLJXKGP 等大群体密集图逐件分开标注，质量好；
4 类（goggles）标注显著增加是 E3 跨域问题的对症改善。

---

### B+112. lc13l 收官批落库（50 张 → valid +38 / 剔除 11 / 悬留 1，2026-10-11 傍晚，负责人：许婧雯）

**images.cv 白大褂源池（944 张）至此全部处理完毕** —— done 清单 **894 → 943**，
唯一未处理的是悬留图 `ZSMH2IW3X5Z8`。
**落库**：**valid +38 图 / +151 框（person 79 / lab_coat 65 / goggles 6 / gloves 1）**；
剔除 11（全 val 段）→ `data/_rejected/lc13l_reject_20261011/`。
`VA056PO2PB6W`/`M1F2698SF8NM` 导出格式异常（!REJECT 后带框行），按剔除优先正确处理 ✅。

**悬留图确认**：`ZSMH2IW3X5Z8`（空实验室场景，白大褂挂在椅背上、无人）连续两批未被标也未剔除。
看图判断：实为有效难度样本（挂着的白大褂可标 lab_coat 无 person），但不宜为一图拖队列 →
继续扣下，图留源池 `train/lab_coat/ZSMH2IW3X5Z8.jpg`，待用户拍板：补标 / 剔除 / 或留作负样本。
全库校验 **10,936 框（train 8,419 / valid 1,666 / test 851），0 坏行 / 0 孤儿** ✅。
复核拼图：`outputs/labcoat_label_review_lc13l/_review_kept_38.jpg` / `_review_rejected_11.jpg`。

---

## 🏁 lc13 系最终统计（B+101~112，12 批画框全部落库）

| 指标 | 数值 |
|---|---|
| 源池 | images.cv 白大褂 944 张（自动筛查 clean） |
| 人工画框入库 | **+642 图 / +2,576 框** |
| 框类别 | lab_coat +1,034 / person +1,240 / gloves +112 / goggles +86 / mask +15（近似） |
| 整图剔除 | 300 张（32%） |
| 悬留 | 1（ZSMH2IW3X5Z8 空实验室） |
| 本线带来全库规模 | **10,936 框**（train 8,419 / valid 1,666 / test 851） |

**后续动作清单**：
1. `ZSMH2IW3X5Z8` 悬留图拍板（补标/剔除/负样本）
2. route A 口径回补：lc13i 的 `HYXBKF2PA0HW`、`KIF16IEVN1QV` 连体服补 lab_coat
3. 全量标完 → 跑 E3 复测验证 lab_coat 跨域改善（上次 F1 0.34→0.891）
4. 向组长索取 `best.pt` 做预标注，支撑下一数据源（E5 P0' 手套/护目镜补数据）

---

## 🔒 lc13 数据线正式封存（2026-10-11 傍晚，用户指令）

**用户指令**：停止向 `data/lab_ppe` 添加图片，lc13 白大褂线到此封存。

**封存时快照**（冻结基线，供后续训练/复测引用）：

| split | 图 | 标签 | 框数 |
|---|---|---|---|
| train | 1,686 | 1,686 | 8,419 |
| valid | 364 | 364 | 1,666 |
| test | 807 | 807 | 851 |
| **合计** | **2,857** | **2,857** | **10,936** |

- 格式校验：0 坏行 / 0 越界 / 0 孤儿 ✅
- 未决项（不影响封存，待用户指示）：`ZSMH2IW3X5Z8` 悬留图、lc13i 两例 route A 回补
- **此后 `data/lab_ppe` 进入只读状态**，除非用户另有明确指令。

**可用训练/评估资产**（已盘点）：
- 现网基线权重：`ai/model/labppe_v8s.pt`（另有 `backup/labppe_v8s_E2_20260928.pt`）
- 训练脚本：`tools/train_e2.py`（YOLOv8s / 50ep / 640 / b16）
- 复测脚本：`tools/run_e3_retest.py`（零侵入包装，可 `--model` 指定权重）
- 数据配置：`data/lab_ppe/data.yaml`（5 类 nc=5）

---

### B+113. A档369 预标注审核落库（369 张 → train +331 / valid +37，2026-10-16 晚，负责人：许婧雯）

**来源**：`A档_6类预标注包`（251020 + SafetyLab 系，模型预标注 + 人工编辑器逐批审核）。
tools：`make_A369_editor.py`（生成 4 批审核编辑器）/ `check_A369_export.py`（导出校验）/ `land_A369.py`（本节落库）。

**四批处置**：

| 批 | 张数 | 处置 |
|---|---|---|
| u8 | 8 | 23:29 导出贴回（8/8 全改），终稿覆写已落库标签：纯框位修正，类别数零变化（含 2 张 UV 紫光图的 person 候选复核） |
| c35 | 31 | 23:24 导出贴回（31/31 全改），终稿覆写已落库标签：净 −8 gloves（有意删除）/ +1 person / +1 lab_coat |
| m2 | 165 | 应用导出 44 图改动 |
| m1 | 165 | 应用导出 58 图改动；30 个原标 gloves 框删除经复核为**有意口径**（用户拍板「手套统统不恢复」，2026-10-16 深夜；此前 AI 曾按硬约束④恢复，已用 `outputs/_A369/undo_gloves_restore.py` 对称撤回，只动恢复集零波及） |

**去重**：1 张与库内 MD5 重复（`SafetyLab__g97` ≡ `g97_png_jpg`，保库内版、弃 A369 副本，实际落 368 张）。
**导出 2 个极小 goggles 框**（251020__56 顶部 h≈0.026）目检为远景人脸护目镜，非误拖，保留。

**框类别增量**：person +531 / mask +135 / gloves +386 / lab_coat +458 / goggles +123 / coverall 0（合计 +1,633）。
**备份**：`data/lab_ppe/_label_backup/labels_pre_A369_20261016_225809.tar.gz`。
**切分**：seed=20261016 随机 90/10。

**执行后底座（B+113 落地）**：

| split | 图 | person | mask | gloves | lab_coat | goggles | coverall | 框合计 |
|---|---|---|---|---|---|---|---|---|
| train | 2,338 | 3,331 | 1,360 | 2,744 | 2,280 | 1,118 | 360 | 11,193 |
| valid | 444 | 417 | 234 | 569 | 493 | 208 | 42 | 1,963 |
| test | 807 | 123 | 60 | 138 | 148 | 382 | 0 | 851 |

- test 红线未动 ✅；img=lbl 对账全 ✅
- 待办：下一轮训练前用 `tools/eval_dataset2_6cls.py` / `eval_t01t10_6cls.py` 复测，重点看 gloves（本批 +386）
