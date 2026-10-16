# 回复补充材料：结果表、解释与修复代码

这里将已公开的冻结审查包展开为可直接浏览的 CSV、JSON 和代码入口，便于按审稿问题引用。实验基准仍为 `20261011_closed`；以下两项基础端点来源、图稿核查和聚合摘要均使用既有结果，没有新增 GPU 实验。完整审查包和 SHA256 保持不变。

## 按问题取用

| 回复问题 | 可直接查看的证据 | 应补充的说明 |
| --- | --- | --- |
| Z6dd Q3 / Mfwd Q6：效率比较 | [18 行时间概览](evidence/reply_support_20261012/e4_sample/latency_summary.csv) · [计时协议](evidence/reply_support_20261012/e4_sample/protocol.json) · [范围例外](evidence/reply_support_20261012/e4_sample/scope_exceptions.json) · [实际配置](evidence/reply_support_20261012/e4_sample/settings_all.json) · [当前 5 配置质量/时间配对](evidence/reply_support_20261012/derived/current_quality_latency_pairs.csv) | 18 行是 17 个方法加 online CorrCLIP 变体；历史 DINO 时间不与新质量混配。当前配对表的质量取全量原生 compact 目标，不能误用时间子集的质量。 |
| Z6dd Q1：固定 GT 与旧起点 | [136 行完整曲线](evidence/reply_support_20261012/fixed_gt/curves.csv) · [34 个端点](evidence/reply_support_20261012/fixed_gt/endpoints.csv) · [4 个局部上升步骤](evidence/reply_support_20261012/fixed_gt/nonmonotonic_steps.csv) · [CorrCLIP ADE150 基础运行](evidence/reply_support_20261012/baseline_sources/corrclip_ade150_base.json) · [Context59 基础运行](evidence/reply_support_20261012/baseline_sources/corrclip_context59_base.json) | 26.2356/48.4803 是新控制起点，26.96/48.68 是原投稿值；下降使用同一控制内的基础端点。不能据此分解原生下降的因果比例。 |
| oBoy Q2：绝对精度与差值 | [固定 GT 端点](evidence/reply_support_20261012/fixed_gt/endpoints.csv) · [投稿数值快照](evidence/reply_support_20261012/submitted_pdf_baseline.json) · [ZIoU 拆分](evidence/reply_support_20261012/diagnostics/ziou_split.csv) | 承认原 Table 2 的 Δ=4.16 加粗会误导；将绝对两端和差值联合报告，取消差值的独立稳健性排名。 |
| oBoy Q3：共同分母与 MCMR 更新 | [56 行共同 GT 计数](evidence/reply_support_20261012/diagnostics/common_gt_counts.csv) · [SAM+SigLIP 聚合对照](evidence/reply_support_20261012/derived/sam_siglip_mcmr_aggregation.csv) · [实际指标实现](../../src/tf_ovos/metrics.py) | Context459=0.693331、ADE847=0.656280；等权平均 0.674805≈0.675，替代历史 0.713。合并计数为 0.678451，是不同统计量；版本变化不只是分母展示变化。 |
| oBoy Q4：同图 thing/stuff | [8 方法、16 行结果](evidence/reply_support_20261012/diagnostics/coco_thing_stuff_intersection.csv) · [共同图像与来源](evidence/reply_support_20261012/diagnostics/coco_intersection_provenance.json) · [修正图表](evidence/reply_support_20261012/diagnostics/coco_thing_stuff_intersection.pdf) | 4,894 张共同图像；8/8 方法 stuff 更低。旧跨数据集 retention 比值与家族排序不能沿用。 |
| oBoy Q5 / Z6dd Q2：Mean/Worst | [四种 proposal 当前原生 compact 汇总](evidence/reply_support_20261012/diagnostics/compact_mean_worst.csv) · [24 项原生新旧对照](evidence/reply_support_20261012/diagnostics/submitted_comparison.csv) | Mean/Worst 仅描述声明的目标集合；这是当前四种 proposal 的表，不是全部方法的统一新质量表。 |
| 小区域指标更正 | [面积阈值 128](evidence/reply_support_20261012/component_min128/component_metric_audit.csv) · [阈值 64](evidence/reply_support_20261012/component_min64/component_metric_audit.csv) · [完美预测控制](evidence/reply_support_20261012/component_perfect_prediction_control.json) | 在 ADE150 全量五方法上核验；不能与旧跨目标数字拼成同设置前后变化。 |
| FreeDA / DINO 修复 | [FreeDA 5,000 张验证](evidence/reply_support_20261012/closure/freeda_full/verification.json) · [DINO 12 项全量完成](evidence/reply_support_20261012/closure/dino_geometry/full/complete.json) · [修正前后表](evidence/reply_support_20261012/dino_geometry_changes.csv) · [200 张效率配对](evidence/reply_support_20261012/closure/dino_e4/verification.json) | 原生 FreeDA=28.7963%；修正 DINO 的质量与旧空间实现时间分开标识。 |

这些文件均可在 GitHub 直接打开。子集[清单](evidence/reply_support_20261012/manifest.json)区分冻结包原文件、既有运行来源和新算术汇总。历史协议中的绝对路径用于追溯，不代表浏览者必须拥有这些路径。完整逐图哈希、原生外部代码等仍见[冻结审查包](artifacts/review_20261011_closed.tar.gz)。

## 可补入回复的解释

### 固定 GT 的新旧起点

> The base endpoint is recomputed within this control rather than copied from the submitted compact table. For CorrCLIP, the new ADE-150 and Context-59 baselines are 26.2356% and 48.4803%, compared with 26.96% and 48.68% in the original table. All control drops use the new within-run baselines. The resulting drops of 7.4260 and 3.1806 percentage points show that degradation persists with GT taxonomy fixed; they do not identify a causal fraction of the historical native compact-to-large drop.

基础运行记录包含 TF32-off、词表和配置哈希；尚无单因素对照证明起点差异全部来自某一个设置，因此不应补这样的归因。

### MCMR 版本与聚合

> For SAM-AMG + SigLIP, the current MCMR values are 0.693331 on Context-459 and 0.656280 on ADE-847. Their equal-target mean is 0.674805 (0.675 rounded), replacing the historical 0.713; pooling the counts instead gives 0.678451, a different statistic. The update uses the corrected prediction and evaluation pipeline, including evaluation-time naming preprocessing and verified GT/metric handling, rather than merely changing the displayed denominator.

### 完整计时的可比范围

> The supplementary timing overview covers all 17 benchmark methods plus the online CorrCLIP variant. It uses a common sample, hardware, and timer design while retaining native model settings. The historical DINO rows are not paired with corrected quality; cached CorrCLIP excludes mask construction; FreeDA excludes offline corpus construction; and CLIPtrase retains center-crop geometry. The configuration-matched table is therefore reported separately from this broader timing overview.

### Mfwd Q4 / oBoy Q2：原表高亮

原 Table 2 的下划线意图为家族内第二好，但图注未解释；从投稿 PDF 第 8 页确认了以下问题：

- MaskCLIP 的 Δ_vocab=4.16 确实加粗，须承认这会把低绝对精度下的小差值误读为稳健性。
- ADE847 ZIoU 的 MaskCLIP-Attn-Slide 38.5 多标下划线；原表第二低为 SCLIP 37.4。
- ADE847 mIoU 的 NACLIP 6.01 不应标第二；原表第二高为 ResCLIP 6.49。

最终必须按更正后的整张表重新计算标记，不能仅把这两个旧数字挪动。差值和条件 MCMR 不再作为独立稳健性排名高亮。

### oBoy Q6：指标与评价问题的对应

| 指标/控制 | 直接回答的问题 | 解释边界 |
| --- | --- | --- |
| 每目标 mIoU | 整体语义正确重叠怎样 | 不能独自解释失败原因 |
| ZIoU + never-predicted | 没有正确重叠，还是从未预测该类 | 声明 GT-supported / union-supported 分母 |
| 固定 GT 候选曲线 | 对本次候选扩张有多敏感 | 单 candidate seed，包含语义重叠，不是纯数量因果效应 |
| C/N、W/N、U/N | 共同 GT 单位上的正确、错名、未达阈值各占多少 | 最终类别图诊断，不是独立 native proposal recall |
| 条件 MCMR | 已达阈值子集中的错名比例 | 条件分母依方法而变 |
| 同类别 component IoU | 小/碎连通区域的错误怎样变化 | 受尺寸、场景等因素共同影响 |
| 同图 thing/stuff | 同一批图上的组别精度差异 | 未独立控制频率、尺寸、形状 |
| Mean/Worst 与配对质量/成本 | 声明目标上的稳定性、具体配置的成本权衡 | 不证明任意未知域泛化或全方法统一端到端最优 |

## 图稿核查材料

[旧案例与当前 SAM+SigLIP 的逐像素比较](evidence/reply_support_20261012/figure_audit/prediction_comparison.json)和[CPU 核查脚本](../../scripts/audit_rebuttal_figure_predictions.py)现可直接查看。

| 案例 | 新旧整图类别变化 | 选中预测名 | 选中掩码变化像素 |
| --- | ---: | --- | ---: |
| person / 2008_002366 | 42.8888% | helmet→helmet | 20,823 |
| scarf / ADE_val_00001111 | 57.7036% | cup→shoe | 1,101 |
| signboard / ADE_val_00000068 | 31.8486% | barrel→barrel | 0 |
| bus / ADE_val_00001803 | 15.9438% | light→light | 7,484 |

归档 signboard 和 bus 全景图中 SAM+SigLIP 的实际覆盖像素与旧预测重采样结果完全一致，确认旧图来源。signboard 选中的覆盖层和标签可保留；另外三个案例应更新变化的覆盖层，scarf 同时更新名称。本地投稿 PDF 没有图 6–8，故其最终附录图号对应仍待对应版本确认。

Figure 5 的数据集标题错误已由投稿第 11 页图像与[生成脚本](../../TMLR26_Bench/scripts/make_independent_case_studies.py)确认：脚本将 ADE150 的 windowpane/pot/computer 分别标为 VOC-20/Ctx-59/COCO，将 ADE847 shelf 标为 Ctx-459。它们应按实际来源重命名，或重新选择已有对应目标预测；正文的跨数据集解释也需同步修改。ADE847 类名解码还须按其 0-based 协议处理，不能无条件沿用 compact 导出预测的减 1 规则。本次发布的是核查材料，不代表已经重画或替换论文中的图。

## 修复代码与核验

仓库顶层现同步了冻结包中的实际运行代码，可直接阅读：

- [DINO mask/RGB 空间对应](../../src/tf_ovos/dino_geometry.py)与[适配器调用](../../src/tf_ovos/adapters/dinov2_sam.py)。
- [SAM+SigLIP 的 eval 预处理](../../src/tf_ovos/adapters/sam_amg_siglip.py)、[共同 GT 指标](../../src/tf_ovos/metrics.py)。
- [DINO 定向全量修复](../../scripts/repair_dino_geometry_from_cache.py)、[FreeDA 原生标签恢复](../../scripts/restore_freeda_coco_native.py)、[DINO 效率配对](../../scripts/measure_dino_geometry_e4.py)。

42 个发布的源码/测试/配置文件与冻结包逐字节一致；本次仅同步代码，没有改变已冻结的实验结果。相关 36 项测试通过（有一个 Transformers 旧接口弃用警告）。在仓库根目录执行以下命令可核对浏览版证据、源码和原压缩包：

```bash
python scripts/verify_reply_support.py
```

该命令不需要 GPU、模型或原始数据集；它验证发布产物的完整性，不重新推理。需要逐图验证时使用原实验工作区与冻结包记载的输入。

## 引用与发布状态

本页提供实际 GitHub 文件链接，便于作者写回复；逐条回复中应给读者可访问的证据索引，不能只在“不复制”的作者说明中放链接。正式采用匿名入口前仍需核实匿名服务已同步这些文件并能下载；GitHub 发布成功不等于匿名入口已经同步。这里不把实名 GitHub 包宣称为匿名材料，也未向 OpenReview 发布回复。

**2026-10-12 关机交接补充：** [16 案例离线素材与 MaskCLIP 缺项补齐](server_handoff_20261012.md)。三个 MaskCLIP 变体已完成保存预测的全量 CPU 重评分，补充表取代旧 n.a.；原冻结包不改写。完整预测、数据集和权重仍需保留原硬盘。

**2026-10-13 配置来源核查：** [Trident SAM、官方预设及回复措辞修正](release_preset_audit_20261013.md)。Trident 的 SAM ViT-B 与所核对官方提交一致；必须补列 SAM，并区分官方按目标预设、项目扩展及 ResCLIP VOC 滑窗偏离。按最新决定不启动统一超参重跑，E3 收窄为已声明配置下的跨数据集表现。

**2026-10-13 ResCLIP 定向修正：** [VOC 官方滑窗设置恢复及全量成对证据](resclip_voc_release_fix_20261013.md)。VOC由78.51更新为82.13；同环境只改slide_crop提升3.621861个百分点。更新Table 1与Mean/Worst，旧E4抽样行须标为历史配置；大词表与fixed-GT结果未因本次修正改变。

**2026-10-13 Figure 4 聚合更正：** [统一四目标权重及计时范围说明](figure4_timing_scope_20261013.md)。online CorrCLIP 改用等权 0.172468 秒/张，四种 proposal 数值不变；后者仍为每目标 25 张抽样，前者为全量，不能称为统一全量效率复测。新表替代旧配对表的聚合展示，未新增 GPU 推理。

**2026-10-15 CLIPtrase 缺项补齐：** [原生中心裁剪重评分与两组差值](cliptrase_native_crop_20261015.md)。已用保存预测完成14,210张CPU重评分；官方GT下原生ADE-847=5.8915，不能称旧5.89已证实标签错误。按统一union-supported口径，Context-459=7.5418、ADE-847=5.8214、平均下降=19.1274pp。四种proposal全量效率队列按最新指示取消，未启动GPU推理。

**2026-10-15 E4 新授权：** [每目标1,000张扩大抽样协议](proposal_e4_expanded_20261015.md)。四种proposal合计16,000张次，单卡串行，时间加95%区间并核对旧预测；不替换Table 1精度。完成前继续使用已核验旧时间并披露其样本数。

**同批图像补充：** online CorrCLIP SAM2-8 也已加入[扩大抽样计划](proposal_e4_expanded_20261015.md#同批图像的-online-corrclip-补测)，在相同4,000张图像上重新计时并核对自己的全量预测，串行等待proposal结束。五配置共20,000张次，完成后统一给出时间及95%区间。

**2026-10-16 E4扩大抽样完成：** [五配置同图计时最终结果](e4_expanded_results_20261016.md)。每目标1,000张、每配置4,000张，共20,000张全部与旧预测逐像素一致。五行等权时间为3.2454/3.5138/3.2937/3.5662/0.1770秒，含95%区间；保留原全量质量，不更改Table 1。
