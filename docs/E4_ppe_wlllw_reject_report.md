ppe-wlllw 外部数据集 —— 审计否决报告（不入库）
日期: 2026-09-23    负责人: 许婧雯（AI/数据集）    结论: 否决，不入库

一、基本信息
  来源: Roboflow Universe / epidetect / ppe-wlllw  v1  (许可 CC BY 4.0)
  素材: data/raw/ppe_wlllw   1882 张 / 19 类 / 8981 框 / 212MB   —— 保留归档，禁止入库
  审计工具: tools/dedup_new_dataset.py（红线）、tools/dedup_phash.py（近重复，--root/--exclude）、
            tools/make_contact_sheet.py（人工复查拼版）、tools/draw_labels.py（画框）

二、逐关结果
  [关1] 红线去重（对 dataset1 / dataset2）
    - 与 dataset2(external_test, 251 张) 命中 0        → 无红线泄漏 ✅
    - 与 dataset1 同源 304 张（16%；MD5/pHash 命中，以 screenshot* 系列为主）
  [关2] 内部近重复（pHash，标注感知 IoU>=0.85）
    - thr=6 keep=1 剔 236 张 + 收敛 1 张；thr=10 keep=2 补刀 +1 → 剩 1340 张
    - 同域集中：Video1–4 四段视频占 637 张（48%），同一批工人/同一楼顶
  [关3] 脏图/截图筛查
    - screenshot 系列 33 张（屏幕截图）
  [关4] 标注一致性（本轮新增，决定性）
    - 同源多副本标注冲突：317 个多副本家族中，手套标注不一致 233 个（73%）、mask 不一致 87 个（27%）
      例：47_JPG 两份副本，一份 GLOVE+MASK，另一份全空（同图既是正样本又是背景负样本）
    - 相邻帧标注翻转率（手套）：Video1 38% / Video2 47% / Video3 56% / Video4 60% / image 系列 43%
    - 相邻帧标注翻转率（mask）：Video1 40% / Video2 20% / Video3 4% / Video4 0%
    - 类别定义不符：screenshot 33 张 100% 为针织劳保线手套（非实验室一次性丁腈/乳胶手套），按判据不合格

三、否决理由（决定性）
  1. 标签级毒数据：同一张图的两份副本标注互相矛盾（73%），且同一场景相邻帧的手套标注有 38%~60% 在
     「有/无」之间翻转 → 训练后模型会学到「戴手套的手有时是目标、有时是背景」，
     恰好复制本项目已花两天清除的「裸手误标」问题。
  2. 类别定义域不符：整体为工地/工厂（安全帽）域，手套为针织劳保线手套；mask 框多落在远景头部，口罩本身存疑。
  3. 有效增量被大幅侵蚀：名义增益（mask +146 / gloves +1416 / goggles +563 框）在一致性过滤后所剩无几，
     性价比低于投入 B1 自采。

四、处置
  - 不入库。原始素材保留于 data/raw/ppe_wlllw/（含 _REJECTED.md 标记，禁止误用/误入库）
  - 审计产物留档：data/ppe_wlllw_leak_images.txt（同源 304）、data/ppe_wlllw_internal_index*.txt（内部近重复）、
    data/ppe_wlllw_exclude_all.txt（合并排除）、data/ppe_wlllw_keep_all.txt（1340 保留清单）
  - 人工复查拼版：outputs/wlllw_review/{mask(蓝框,146张,5版), screenshot(33张,2版), sample(48张,2版)}
  - 项目文档登记：docs/E4_lab_coat_plan.md 第 9 条「外部 mask 数据集引入线结案」

五、沉淀（流程改进，务必沿用）
  外部数据入库前新增第 4 关「标注一致性审计」：
    ① 同源副本的标注一致率
    ② 同一视频/连拍系列相邻帧的标注翻转率
    ③ 类别定义与项目判据的域匹配（如手套须为实验室一次性手套）
  前三关（红线去重 / pHash 同源 / 脏图筛查）只能抓图片级问题，抓不到标签级毒数据。
  已知残留风险：ppes-kaxsi、safety-gloves 入库时未过此关（后续抽检需留意）。

六、关联结论
  saba/ppedetection v7（364 张）同期被审计证伪：362/364 与 dataset1 MD5 级完全重复（dataset1 的派生发布版），
  同样否决。报告 data/ppedetect_dedup_report.txt。
  → 本轮外部引入线（两个候选）全部否决，mask/gloves 补强主力回归 B1 组长自采 150 张。
