# Figure 4：统一目标权重与仍存在的抽样差异

本次仅复算已有计时，没有新增 GPU 推理，也没有替换论文 Figure 4。

## 核对结果

原配对表的四种 proposal 使用每目标 25 张、共 100 张的效率复测；质量使用全量原生 compact 目标。online CorrCLIP SAM2-8 的质量和计时来自其自身的 13,554 张全量运行，但原时间按图像数加权。将它们称为全配置统一全量效率复测不准确。

新表统一采用四目标等权：T = (T_VOC + T_Context + T_ADE + T_COCO) / 4。其中每个 T 是对应目标的逐图平均在线时间。质量继续使用四目标全量 mIoU 的等权平均。

| 配置 | 全量质量 Mean mIoU (%) | 等权在线时间 (秒/张) | 计时样本 |
| --- | ---: | ---: | --- |
| SAM-AMG + CLIP | 26.9584 | 3.140006 | 每目标 25 张 |
| SAM-AMG + SigLIP | 30.1283 | 3.410393 | 每目标 25 张 |
| DINOv2 + SAM + CLIP | 25.2297 | 3.183577 | 每目标 25 张；修正空间对应 |
| DINOv2 + SAM + SigLIP | 28.4272 | 3.458969 | 每目标 25 张；修正空间对应 |
| online CorrCLIP SAM2-8 | 48.3549 | 0.172468 | 四目标全量，共 13,554 张 |

CorrCLIP 原图像加权时间为 0.173753 秒/张，等权后为 0.172468。其他四行不变，五个配置中已测的质量与时间关系不变。

## 为什么有抽样与全量之别

proposal 的效率记录来自独立的配置配对复测；online CorrCLIP 是另一个在线变体，其全量质量运行同时记录了逐图耗时。这解释了现有记录的来源，但不能消除测量设计差异。按现有平均速度估算，四种 proposal 各自完成 13,554 张计时需约 12–13 小时，合计约 50 GPU 小时；这不是已完成运行，也不是承诺工期。online CorrCLIP 对应纯在线推理累计约 39 分钟。以上不包括初始化、结果保存和验证。

统一目标权重不等于统一样本：proposal 仍是每目标 25 张的估计，CorrCLIP 仍为全量平均。报告应保留样本数，不声称同图全量计时，不给出未经估计的不确定区间，不推导普遍的端到端 Pareto 最优或精确部署预算保证。离线准备、具体配置和环境的边界仍须披露。是否进一步全量复测属于新增 GPU 工作，本次没有启动。

## 可用于 Z6dd A3 / Mfwd A6 / Figure 4 图注的说明

> We now aggregate online latency with equal weight for each of the four compact targets, matching the target weighting used for mean quality. This changes the online CorrCLIP SAM2-8 value from 0.173753 to 0.172468 seconds per image; the four proposal values are unchanged. The timing sample sizes remain different: each proposal configuration was measured on 25 images per target, whereas the online CorrCLIP variant was timed over its full 13,554-image quality run. We therefore report these as configuration-specific latency estimates with explicit sampling scopes, not as a uniformly full-dataset timing benchmark or a universal end-to-end Pareto ranking. Offline preparation costs are outside these online measurements.

## 证据与复算

- [统一权重配对表](evidence/figure4_equal_target_20261013/quality_latency_pairs.csv)
- [20 行逐数据集计时](evidence/figure4_equal_target_20261013/latency_by_dataset.csv)
- [来源哈希及核验记录](evidence/figure4_equal_target_20261013/verification.json)
- [无需 GPU 的复算脚本](../../scripts/reaggregate_figure4_latency.py)：在仓库根目录执行 `python scripts/reaggregate_figure4_latency.py`。

脚本使用既有 CSV 和冻结包中的修正 DINO 记录，逐图重算 DINO 的 200 条时间，并复现原五行聚合。没有使用旧空间实现的 DINO 时间。历史冻结包与旧配对表保留原值，此新表作为聚合口径更正。

截图的第二条只显示“oBoy A6 的表头仍有一点误导”，未包含具体解释；现有本地回复稿的 A6 没有对应的新汇总表，故不能声称该项已核验或已修正。
