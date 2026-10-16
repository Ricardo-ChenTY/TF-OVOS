# E4 扩大抽样完成：五配置同图计时与95%区间

**2026-10-16 10:22:43 UTC 全部完成。** 四种 proposal 的16项任务和 online CorrCLIP 的4项任务，共20项、20,000张次。每种配置均在同一批4,000张图像上计时（VOC、Context-59、ADE-150、COCO各1,000张）；全部20,000张新预测与各自已有的全量质量运行逐像素一致。GPU队列已退出。

## 用于论文及回复的最终数字

时间是四目标等权在线时间，单位秒/张。95%区间来自按目标分层的10,000次image bootstrap。

| 配置 | 保留的全量Mean mIoU (%) | 平均时间 | 95% CI |
| --- | ---: | ---: | --- |
| SAM-AMG + CLIP | 26.96 | **3.2454** | [3.2236, 3.2671] |
| SAM-AMG + SigLIP | 30.13 | **3.5138** | [3.4878, 3.5398] |
| DINOv2 + SAM + CLIP | 25.23 | **3.2937** | [3.2712, 3.3159] |
| DINOv2 + SAM + SigLIP | 28.43 | **3.5662** | [3.5396, 3.5927] |
| online CorrCLIP SAM2-8 | 48.35 | **0.1770** | [0.1763, 0.1776] |

精度没有替换成这次抽样精度；online CorrCLIP 的48.35来自该在线变体自身的全量运行，不是缓存版本的质量。以上五行作为当前Figure 4的同图配对比较，替代之前25张/目标的四种proposal时间及CorrCLIP全量加权时间的混合展示。历史结果仍保留供追溯。

## DINO约0.05秒的增量

按相同image ID配对，计算DINO−SAM时间差，再在四目标上等权平均：

| 命名骨干 | 平均增量 (秒/张) | 配对95% CI |
| --- | ---: | --- |
| CLIP | **0.048242** | [0.046873, 0.049610] |
| SigLIP | **0.052418** | [0.050948, 0.053900] |

在这批图像与这些计时会话下，两个增量均为正，数值都约0.05秒。每目标的增量并不相等，不能写成固定常数。区间仅覆盖图像差异，未估计跨运行、温度、方法执行顺序及硬件变化，不能据此断言所有环境都保持相同增量。

online CorrCLIP在这五个声明配置中仍具有更高的全量平均质量和更低的实测在线时间；这不是所有17种方法的同图扩大抽样结果，也不是普遍的端到端Pareto或方法家族优劣结论。

## 计时边界必须保留

- 每目标1,000张，五配置相同图像及顺序，目标等权；**扩大抽样，不是全量验证集**。
- 同一张A100串行计时，不复用视觉特征或预测。竞争时的测量丢弃重测；已接受记录没有观察到其他GPU进程。
- 从解码后的RGB输入到CPU类别输出，CUDA同步；模型/文本初始化、文件I/O、结果写盘和离线准备不计入在线时间。另存读取至输出时间。
- 保留各自已核验配置：四种proposal cuDNN TF32 on，online CorrCLIP off；所有配置matmul TF32 off。DINO空间对应与SigLIP eval预处理均为修正版本。
- 不能将五配置的1,000张/目标协议扩写为全部17方法的协议；原18行概览仍需披露其原抽样数量、版本和范围例外。

## 可粘贴到Z6dd A3的英文补充

> We have completed an expanded timing comparison on the same 4,000 images for all five configuration-matched entries: 1,000 images from each of VOC-20, Context-59, ADE-150, and COCO-Stuff-171. Measurements were run serially on one A100 with synchronized timers, and all 20,000 timed predictions were pixel-identical to the corresponding saved full-quality predictions. We retain those full-quality scores rather than replace them with sample mIoU. Equal-target mean online latencies are 3.2454, 3.5138, 3.2937, 3.5662, and 0.1770 seconds per image for SAM-AMG+CLIP, SAM-AMG+SigLIP, DINOv2+SAM+CLIP, DINOv2+SAM+SigLIP, and online CorrCLIP SAM2-8, respectively; the table reports their stratified image-bootstrap 95% intervals. Paired DINO-minus-SAM overheads are 0.04824 seconds with CLIP and 0.05242 seconds with SigLIP. These intervals describe image variability conditional on the measured sessions, not between-run or hardware variability. Native settings are retained, including cuDNN TF32 enabled for the four proposal configurations and disabled for online CorrCLIP; matmul TF32 is disabled for all. Online timing excludes initialization, file I/O, and offline preparation. This five-configuration comparison is distinct from the broader supplementary timing overview.

## 改稿清单

1. Figure 4及相关计时表：用本页五行时间和区间；质量沿用各自已核验全量结果。
2. Observation 7：把DINO开销写成本次配对测量约0.048/0.052秒，保留会话与配置边界。
3. 附录计时协议：这五配置从每目标25张更新为每目标1,000张，补统一抽样、等权聚合、bootstrap与像素核验；旧全方法概览的样本量不随之修改。
4. Z6dd A3及Mfwd有关Figure 4可读性的回复：补本页表格、区间和证据入口。

本次完成实验、核验和报告，不代表已经编辑用户最新版论文PDF/LaTeX或向OpenReview发布回复。

## 证据

- [最终五行质量/时间配对表](evidence/e4_expanded_results_20261016/quality_latency_pairs.csv)
- [25行逐目标及等权时间/区间](evidence/e4_expanded_results_20261016/latency_ci_five_configurations.csv)
- [配对DINO开销及区间](evidence/e4_expanded_results_20261016/paired_dino_overhead_ci.csv)
- [20项任务与20,000张一致性核验](evidence/e4_expanded_results_20261016/verification.json)
- [结果审查包](artifacts/e4_expanded_results_20261016.tar.gz)及[SHA256](artifacts/e4_expanded_results_20261016.tar.gz.sha256)：逐图计时、逐图预测对照、配置、输入清单、统计来源及汇总，不含模型权重或预测PNG。
- [冻结运行协议与代码入口](proposal_e4_expanded_20261015.md)、[导出和数字复核脚本](../../scripts/export_expanded_e4_results.py)。

导出时重新核对20项任务状态、逐图记录与清单顺序、配对报告零差异、统计来源哈希，并从逐图时间复算目标均值及五个等权均值。全部通过。


## 仓库顶层代码补齐（2026-10-16）

已补入本轮入口实际依赖的 [measure_e4_full.py](../../scripts/measure_e4_full.py) 和 [run_e4_full.py](../../scripts/run_e4_full.py)，以及历史配对核验的 [run_e4_targeted.py](../../scripts/run_e4_targeted.py) 与 [断点恢复/评分/预测差异测试](../../tests/test_e4_full.py)。这两项底层计时文件此前已在冻结审查包中，现在可以直接浏览和导入。

本轮使用的11个脚本/启动器及25个包内Python文件，共36个文件，均与两份冻结运行代码包逐字节一致。无需GPU可检查：

```bash
python scripts/verify_expanded_e4_code.py
```

另在只复制Git已跟踪文件的独立目录中验证了入口导入和7项评分、恢复、配对及bootstrap测试；未依赖工作区未跟踪脚本。本次仅补公开代码，不重跑GPU任务或改变实验结果。实际重新推理仍需要协议指定的数据、模型、第三方环境和本地运行目录。
