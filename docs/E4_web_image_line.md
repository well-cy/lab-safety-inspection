# E4 补充线：网页真实实验室场景图引入（web_lab）

> 状态：**采集中，未入库，未标注**
> 发起：2026-09-23，许婧雯拍板"不用版权核实，能用就行"
> 工具：`tools/fetch_web_images.py`（采集）、`tools/labeler.html`（标注）
> 关联：`docs/E4_lab_coat_plan.md` §A3（域差根因）、§B 表（B1/B2/B3）

---

## 0. 这条线和 B2 的本质区别（为什么现在做）

E3 已证明 dataset1 的 lab_coat 跨域 Recall 只有 20.5%，A3 把根因定在**域窄**：
训练侧近 100% 是"摆拍图库 + istockphoto 缩略图"，亮度中位 175、单人近景、打光充足；
目标侧（dataset2 / t01-t07）是真实实验室视频帧，亮度 134-142、多人中远距离、自然顶光。

**关键澄清**：此前"上网找图无效"的结论只对**图库缩略图**成立（dataset1 已有 178 张
istockphoto，612×612、带水印、单人近景——和它同类的再抓多少都不治域差）。
本线抓的是**另一类东西**：Commons 上科研机构/高校的实拍原图，常见 2000-6000px、
真实实验室、多人、中远距离、自然光。这类图**不是**在往同一个窄域里灌水。

| | dataset1 已有的网络图 | 本线（Commons 实拍） |
|---|---|---|
| 分辨率 | 612×612 / 768×432 | 2000-6000px |
| 场景 | 棚拍 / 摆拍 | 真实实验室 |
| 人数 | 多为单人近景 | 常见 2-5 人 |
| 光照 | 打光充足（亮度 ~175） | 自然顶光（接近目标域） |
| 许可 | 无（水印图） | CC / PD，来源落盘 |

## 1. 许可口径

按既有先例执行（`docs/E4_lab_coat_plan.md` 风险 3）：**课程项目内部使用可接受，
报告注明来源**。dataset1 本身含 178 张 istockphoto、labcoat_si 含 CanStock/iStock
水印图，均已在库。本线只采集 Commons 上**本来就带自由许可**（CC/PD）的条目，
license 与作者自动写入 `data/raw/web_lab/manifest.csv` 备查，**不做逐条清权**。

## 2. 采集口径（fetch_web_images.py 内置）

**只要「真实实验室里有人」**，检索式必须同时命中人（scientist / researcher /
technician / student / worker）与实验室语义（laboratory / cleanroom / fume hood）。

硬过滤：bitmap、JPEG/PNG、宽 ≥800、高 ≥600、长宽比 0.4-2.8、
标题黑名单（svg/logo/diagram/chart/glassware/building/portrait/painting…）。

18 组检索式 + 8 个分类目录，枚举结果落盘 `candidates.json`，
按分辨率降序下载，许可/作者/来源 URL 写入 `manifest.csv`。

```bash
# 枚举（受 Commons API 限流，一次 2-4 分钟；结果落盘，可中断）
python tools/fetch_web_images.py --collect-only
# 下载（走 upload.wikimedia.org，不受 API 限流）
python tools/fetch_web_images.py --from-json data/raw/web_lab/candidates.json
```

## 3. 筛选漏斗（入库前）

| 关 | 工具 | 抓什么 |
|---|---|---|
| ① 红线去重 | `tools/dedup_new_dataset.py --src data/raw/web_lab --refs dataset1,dataset2,labcoat_si` | 与在库数据 MD5/pHash 命中 → 剔 |
| ② 脏图筛查 | `tools/screen_dirty_graphics.py --dir data/raw/web_lab/images --tsv …` | 排版图/白底商品图/黑白老照片 |
| ③ 场景人工筛 | 拼版（`make_contact_sheet.py`）+ 人工圈选 | 无人的器材图 / 棚拍摆拍 / 与实验室无关 |
| ④ 标注一致性 | 人工标注 + 复核 | 见 §4 口径 |

①② 是脚本，③ 由 AI 先做粗筛（拼版逐页看，砍掉明显不合格的），再出**精选拼版**
给负责人终筛，把人工量压到最低。

## 4. 标注口径（2026-09-23 从在库数据实拍核实）

对 `data/lab_ppe` 现有 1972 个 lab_coat 框抽样画框核实，**在库口径是：
「罩住白大褂本体」的盒，一人一框，肩部到大腿中部；未穿白大褂的人不框**。
（面积中位 0.257，宽高比中位 0.74——高>宽，符合"大衣盒"而非"整个人盒"。）

标注铁律（与 ppes-kaxsi 入库时同样的理由）：

1. **图里出现的四类 PPE 必须全部标出**。YOLO 里没标 = 背景负样本，
   漏标一个戴口罩的人 = 教模型"这张脸没有口罩"。
2. 只标**实验室判据内**的：手套 = 一次性丁腈/乳胶（线手套/皮手套不标不框）；
   护目镜 = 防冲击护目镜（普通框架眼镜不标）。
3. 分辨率低到看不清是否佩戴的，**整图剔除**（宁缺毋滥，ppes 的教训）。
4. 黑白历史照片、明显摆拍、器材特写 → 标为不合格（A 键）。

## 5. 标注工具（tools/labeler.html）

单文件 HTML，**零依赖、零联网、不开服务器**，浏览器直接打开。浏览器选
`data/raw/web_lab/images` 目录即可开始。要点：

- 类别热键 `1` mask / `2` gloves / `3` lab_coat / `4` goggles
- 拖拽画框，拖角把手改尺寸，`D` 删框，`A` 标不合格，`Enter` 存并下一张
- `载入已有标注` 按钮可读入 YOLO txt 做**预标注校正**
  （拿到 E2 `best.pt` 后即可先模型预标再人工校正，速度提升 5-10 倍）
- 标注进度自动存 localStorage，刷新不丢
- `导出 ZIP` 产出 `labels/*.txt` + `classes.txt` + `_skipped.txt` +
  `_export_report.txt`（含各类框数统计）
- **同一套工具直接服务 B1 自采 150 张**——这是它真正的战略价值：
  本地没有 `*.pt` 导致"模型预标注"这条最快路走不通，人工从零画框是 B1
  卡了两周的隐性原因；有了这个工具，人工标注 1-3 框/张的图约 10 秒/张。

## 6. 入库（✅ 2026-09-24 已完成）

1. ZIP 解包 → CRC 全通过；96 张有框 / 16 张未标 / 0 张不合格
2. **发现并修复标注器导出口径 bug**：`fmtBox` 把框写成了「左上角+宽高」，
   而 YOLO 要求「中心点+宽高」——249 框经确定性反推（中心=左上+宽高/2）全部修复
   （`labels/` 为原始导出，`labels_yolo/` 为修正版；81 个越界框裁剪到图像边界，0 框剔除）。
   `tools/labeler.html` 已修，B1 标注不受影响。复核拼版确认框全部对齐目标。
3. 16 张未标注逐张核验：全部为无人/无 PPE 场景（空台面/仪器/便装人员），作背景负样本合法
4. 四关审计：红线去重 0 命中（`data/web_lab_keep_dedup_report.txt`）；
   内部 pHash 0 组（`data/web_keep_internal_phash_report.txt`）；
   脏图筛查在粗筛已过；标注一致性经渲染复核通过
5. **入库**：112 张（96 有框 + 16 负样本）按 90/10 进 train(100)/valid(12)，**不进 test**；
   随机种子 42；顺带清除 1 个历史遗留空标签孤儿
   （`labels/valid/ppes_frame558_*.txt`，0B 无图）

**入库后全库：13410 张 / 20966 框（mask 1807 / gloves 10562 / lab_coat 2078 / goggles 6519）**
图文配对全库一致。E4 训练启动条件不变：B1 到位；本线数据为并行增量。

## 7. 边界与不做的事

- **不替代 B1**。真实实验室**现场**照片（本项目自己的实验楼）域匹配度仍最高，
  本线是"真实域的旁证"，不是替身。
- 不抓取需要登录/反爬的站点，不绕过任何访问限制。
- 不做逐条清权（§1 口径），如未来要公开发布模型或数据集，需回头补。

---

## 8. 执行进展（2026-09-23，AI 自动完成部分）

| 步骤 | 结果 |
|---|---|
| 枚举（18 检索式 + 8 分类目录） | **478 张候选**，分辨率中位 **3236×2712**，许可全为 PD/CC0/CC 系 |
| 下载 | **382 张**（1920px 服务端缩略图，252MB）；其余 96 张为分辨率最低的一档，因 upload.wikimedia.org 限流+SSL 中断暂停，可随时 `--from-json` 续跑（编号已改为按清单位置，断点续跑不撞名） |
| ① 红线去重（dataset1/dataset2/labcoat_si） | **0 命中**（MD5 与 pHash≤10 均为 0）——完全独立来源，`data/web_lab_dedup_report.txt` |
| 内部 pHash 近重复（thr=6, keep=2） | 4 组各 2 张，**无需剔除**，`data/web_lab_internal_phash_report.txt` |
| ② 脏图筛查 | **74 张命中（19.4%）**：黑白历史照片 66、排版/商品图 8（`data/raw/web_lab/stats.tsv`、`flags_dirty_raw.txt`）——**建议整批剔除**（黑白老照片与目标域严重不符） |
| ③ AI 场景粗筛（9 页拼版逐页看） | 308 张待审 → **保留 112 / 剔除 196**。剔除主因：Navajo Tech 兽医教学连拍里的牧场/教室/建筑外观、黑白历史照漏网、棚拍人像、工业焊接/ hazmat（非实验室域）。清单 `data/raw/web_lab/list_keep_ai.txt` / `list_reject_ai.txt` |
| 保留 112 张构成 | 其中 **93 张有佩戴 PPE 的人员**（lab_coat 为主，含 mask/gloves/goggles 镜头），**19 张为无人的真实实验室场景**（作背景负样本，治误检） |

**产出物**：

- 精选拼版（终审用）：`outputs/web_review_keep/sheet_01..04.jpg`（112 张，每格左上角为序号，`outputs/web_review_keep/index.txt` 是序号↔文件名映射）
- 全量粗筛拼版（留档）：`outputs/web_review/sheet_01..09.jpg`
- 标注暂存目录（**只含 112 张精选**，硬链接不占额外空间）：`data/raw/web_lab_keep/images/`
- 标注器：`tools/labeler.html`（浏览器打开 → 选 `data/raw/web_lab_keep/images`）

**下一步（需人工）**：

1. ~~终审 4 页精选拼版~~ **已通过（2026-09-23 用户终审：112 张全保，按"真实实验室场景"口径无剔除）**
2. 打开 `tools/labeler.html` 标注（口径见 §4；热键 1-4，Enter 下一张）
3. 导出 ZIP → 四关复核 → 合并脚本入库（未做，见 §6）

**终审结论（2026-09-23）**：用户查看 4 页精选拼版后确认，按「真实实验室场景」口径 112 张全部合格，无剔除。精选清单即为最终清单：`data/raw/web_lab/list_keep_ai.txt`（112 张）。当前卡点：**人工标注**（步骤 2）。

