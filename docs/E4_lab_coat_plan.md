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
   - **结论：外部引入线就此收束。** mask / gloves 的补强主力仍为 B1 组长自采 150 张——唯一能治"实验室域缺失"的数据源（E3 已证 lab_coat 跨域 Recall 仅 20.5%）。

> ⚠️ 工具坑位记录：本机 WorkBuddy 沙箱把 `os.unlink` 劫持为「移入回收站」，回收站操作被中止时触发 `SAFE_DELETE_FAIL_CLOSED` **直接杀进程**（表现为莫名其妙的 SIGTERM）；且 Windows 下 `os.rename` 到已存在目标必失败，`shutil.move` 会回退到 `copy + unlink` 而踩中该雷。故 `remove_bad_images.py` 已改为**只用 `os.replace`（原子可覆盖）+ 失败换备用名 + 单条失败不中断**，绝不回退到任何 unlink 路径。

工具：几何审计 `tools/audit_lab_ppe_boxes.py`（报告 `data/lab_ppe_audit_report.txt`）；按类别画框抽检 `tools/draw_labels.py --only-class <类>`。

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
5. **催 B1**：唯一硬前置，组长自采 150 张（≥5 人、≥3 种光照、近/中/远三种距离）
