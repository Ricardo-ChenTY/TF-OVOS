# E4 扩大抽样：每个数据集 1,000 张

用户确认的规模为 **每个 compact 数据集 1,000 张**。VOC-20、Context-59、ADE-150、COCO-Stuff-171 四目标，每种配置共 4,000 张；SAM-AMG+CLIP、SAM-AMG+SigLIP、DINOv2+SAM+CLIP、DINOv2+SAM+SigLIP 四种配置合计 **16,000 张次、16 个任务**。这是扩大抽样，不是全量验证集复测。

## 固定协议

- 在各目标完整清单中排除原先固定的 5 张 warmup，再按 image ID 排序，以 seed=20261015 均匀无放回抽取 1,000 张。四种配置共用同一批图像及顺序。抽样在看到本轮计时前固定。
- 单张 A100 上只运行一个计时进程。NVML 在推理前后及过程中监测其他 GPU 进程，竞争期间的测量丢弃重测，不停止别人的任务。
- 采用当前修正的 SigLIP eval 预处理和 DINO mask/token 空间对应；保留原生参数，matmul TF32 off、cuDNN TF32 on、4 CPU threads。
- 在线计时从已解码 RGB 输入到 CPU 类别输出，CUDA 同步；不复用预测或跨图视觉特征。模型/文本初始化、图片读取和结果写盘不计入在线时间；读取至输出时间另存。
- 每进程先 warmup。逐图计时、预测 PNG、输入与输出哈希、配置、进度进入可恢复记录。新预测与已冻结的全量质量运行逐像素对照。少量预检失败或完整任务出现不一致时停止队列排查。
- **不替换 Table 1 精度**。本次抽样 mIoU 仅作诊断；脚本输出中的任何 sample-quality 汇总都不是新的全量质量表。

## 统计与解释

输出每目标及四目标等权平均在线时间。使用固定 seed=20261015 的 10,000 次分层 image bootstrap，分别在每个目标内重采样图像，报告 percentile 95% 区间；四目标权重各为 1/4。

DINO−SAM 差值在相同命名骨干下按相同 image ID 配对，报告配对差值及区间。这个区间刻画已测会话下的图像差异，不覆盖跨运行、温度、方法顺序或硬件变化，因此不能直接宣布“DINO 固定增加 0.05 秒”。

耗时估计约 15 小时，基于历史约 3.1–3.5 秒/图；实际受初始化、GPU 等待、核验和图像组成影响。不是完成承诺。

## 运行与交付

运行目录：`/data/sdd/tianyi/TF-OVOS-rebuttal/runs/proposal_e4_1000_per_target_20261015`。

运行入口：[队列及清单冻结](../../scripts/run_proposal_e4_expanded.py)、[置信区间汇总](../../scripts/summarize_proposal_e4_expanded.py)。实际运行使用目录内冻结的代码与适配器，避免后续仓库编辑改变运行配置。

- `queue_status.json`、`progress.json`、`coverage.csv`：实际进度。
- `jobs/<method>/<dataset>/per_image.jsonl`：逐图时间。
- `jobs/<method>/<dataset>/prediction_pairing.json`：与已有全量预测的逐像素核验。
- `latency_ci.csv`：全部通过后生成的时间及 95% 区间。
- `paired_dino_overhead_ci.csv`：两组命名骨干下的 DINO−SAM 配对时间差。
- `interval_verification.json`：统计来源与适用边界。

完成后需用最终数值更新论文相关计时表、Figure 4、Observation 7、附录计时样本数，以及 Z6dd A3 的计时说明。本文记录协议，不将尚未完成的测量写成结果。

旧全量准备目录与误解成每目标 250 张的准备目录均已标记取消，未启动 GPU 推理；本次是用户重新授权的每目标 1,000 张任务。

冻结证据：[协议](evidence/proposal_e4_expanded_20261015/protocol.json)、[每目标1,000张的ID与输入哈希](evidence/proposal_e4_expanded_20261015/sample_ids.csv)、[运行代码及输入/参考预测清单包](artifacts/proposal_e4_expanded_protocol_20261015.tar.gz)。协议包不含权重或原始图像。队列于2026-10-15 18:36 UTC在tmux `tfovos-e4-1000`启动；完成状态以运行目录中的实际记录为准。
