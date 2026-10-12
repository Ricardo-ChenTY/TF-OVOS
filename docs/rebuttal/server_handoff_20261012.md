# 关机前材料核查与补充（2026-10-12）

此次补充发布在 `rebuttal/reply-materials-20261012` 分支。原 `20261011_closed` 冻结包不改写；下述 MaskCLIP 补充结果覆盖旧文档中的相应 n.a.，不能继续把它解释为缺预测。未启动模型推理或 GPU 实验，未关闭服务器。

## 1. 离线改图材料已导出

下载 [16 案例素材包](artifacts/figure_assets_20261012.tar.gz)及其 [SHA256](artifacts/figure_assets_20261012.tar.gz.sha256)，解压后运行：

```bash
python render.py . --verify-only
python render.py . --out rendered
```

仅需 NumPy、Pillow。包含混合数据集旧清单的 8 个案例，以及投稿 Figure 5 当前全 ADE 版本的另外 8 个案例；每个案例有输入 JPEG、当前语义 GT、目标 mask、词表、CorrCLIP/NACLIP/ProxyCLIP/SCLIP 与修正后 seed-0 SAM+SigLIP 的原始预测。共 80 张预测 PNG。原始文件逐字节拷贝；偏移量、void、目标选择方法、SHA256 均记录。解压后无需原服务器、权重、HF 或数据集目录。

[完整案例及解码结果](evidence/server_handoff_20261012/figure_cases.json) · [逐文件哈希](evidence/server_handoff_20261012/figure_manifest.json) · [导出脚本](../../scripts/export_rebuttal_figure_assets.py) · [离线核验/预览脚本](../../scripts/render_rebuttal_figure_assets.py)。

`/private/tmp/ade_hf_rows.jsonl` 未找到，不能称为已备份。已从现存 ADE 原始 ZIP 提取 8 个案例的多边形 JSON，按照旧脚本的“同 raw_name 最大多边形”规则生成目标，替代对该临时文件和 HF 在线接口的依赖。不声称与遗失的 HF 缓存逐字节相同。

当前全 ADE Figure 5 可以使用这套素材修正实际数据集标题和类名，但发布素材不等于论文 PDF 已替换。ADE-847 两例必须按 0-based 解码：

| 案例 | CorrCLIP | NACLIP | ProxyCLIP | SCLIP |
| --- | --- | --- | --- | --- |
| shelf / ADE_val_00000456 | bookcase (74) | drawer (343) | bookcase (74) | pantry (288) |
| jacket / ADE_val_00000255 | coat (484) | coat (484) | coat (484) | coat (484) |

另有一处旧清单限制：COCO `000000033707` 的历史 flower 目标 ID=107 在当前修正 GT 中不存在；原来的 42,213 个目标像素现在是 ID=108（fog）。包内同时保留新旧 GT 并在预览标红，不能把历史 flower 标题直接当作当前 GT。若改用混合数据集 Figure 5，需要另核实目标/挑选 COCO 案例；保留当前全 ADE Figure 5 并纠正标题则不依赖该旧候选。

对旧附录候选的结论仍是：signboard 选中的 SAM+SigLIP 覆盖层和名称可保留，person/scarf/bus 的覆盖层需更新，scarf 名称 cup→shoe。本地投稿 PDF 没有图 6–8，最终附录的图号与面板对应尚需最终 PDF 确认，不可说“所有附录图已改完”。

## 2. MaskCLIP 的 n.a. 已查明并补齐

实际原因是**目录入口遗漏**：旧重评分只扫描 `runs/artifacts/official_predictions/<method>/<dataset>`；三个 MaskCLIP 变体存放在 `runs/<method>/<manifest>/predictions.jsonl` 所索引的 shards 下。因此它们没有进入此前的官方 ID 重评分/共同 GT 表，并非未保存、无法解码或 uint8 截断。

逐文件检查了三个方法各 5,105 张 Context-459 和 2,000 张 ADE-847，共 **21,315 张 0-based、16 位 PNG**，无缺图、ID 越界或重复图。用已发布的 `dense_class_region_max_iou_v2` 与当前 GT 完成六组全量 CPU 重评分；GT 分母严格为 33,026 / 20,369。每张预测与 GT 均记录 SHA256，无新推理。

| 方法 | ADE-847 mIoU 原稿→当前（%） | ADE-847 union ZIoU 原稿→当前（%） | 两目标等权 MCMR@0.5 原稿→当前 |
| --- | ---: | ---: | ---: |
| MaskCLIP | 0.49→0.526749 | 64.7→62.927981 | 0.843→0.844013 |
| MaskCLIP-Attn | 0.50→0.548424 | 64.7→63.282172 | 0.858→0.858422 |
| MaskCLIP-Attn-Slide | 1.44→1.522365 | 38.5→32.939787 | 0.541→0.542860 |

这些是原稿对应历史适配器的保存输出；并不把 lite/attention/slide 实现重新宣称为另一个官方实现。重评分采用输出整图最近邻对齐当前 GT，适配器本身使用整图 resize/滑窗而非中心裁剪。

- [六组完整评分](evidence/server_handoff_20261012/maskclip_summary.csv)
- [两个阈值共同 GT 计数](evidence/server_handoff_20261012/maskclip_common_gt_counts.csv)：12 行，可追加旧共同 GT 表；原 14 方法扩大为 17 条目（含 SAN 参照）。
- [原稿对照及聚合](evidence/server_handoff_20261012/maskclip_submitted_comparison.csv)：Δvocab 一列明确仅用其他原稿单元格的两位小数计算，不能充当全精度最终表。
- [遗漏原因结构化记录](evidence/server_handoff_20261012/maskclip_reason.json)
- [逐图哈希与评分审计包](artifacts/maskclip_saved_audit_20261012.tar.gz) / [SHA256](artifacts/maskclip_saved_audit_20261012.tar.gz.sha256)
- [CPU 重评分脚本](../../scripts/audit_maskclip_saved_outputs.py)

可在回复中补充：

> The three MaskCLIP variants were omitted from our initial corrected tables because their saved predictions were indexed under adapter-specific shard directories rather than the directory scanned by the rescoring script. We have now verified and rescored all 5,105 Context-459 and 2,000 ADE-847 predictions per variant under the corrected protocol, and provide their common-denominator counts and updated metrics in the supplementary evidence.

上面的证据包含逐图哈希，不含全部 MaskCLIP PNG；不能用它替代完整预测备份。

## 3. 是否可以关 server

**停止 GPU 计算、但保留并可重新挂载原硬盘：可以按这个条件规划。释放实例并删除/丢失本地盘：目前不能把 GitHub 当作完整备份。**

- 写回复/改表所需旧证据与本次新增数字已发布；16 个候选案例可在其他电脑离线检查/画预览。
- 全部实验预测、数据集、模型权重、缓存仍不在 GitHub。讨论期若需换案例、扩展核验或新推理，需要这些文件。
- 实验文件跨 `/data` 与 `/data/sdd` 两个挂载点，必须分别确认保留；目录名不能证明关机后云盘会自动保留。此次未更改实例或存储设置。
- 最终论文图稿替换、最终附录 PDF 对应以及匿名证据入口同步/访问还未完成。实名 GitHub 分支可下载不等于匿名服务已同步。

若要释放实例而不保留磁盘，应先完成完整预测的外部备份并验证可下载；本次只导出了小型改稿素材，没有执行全量存储迁移。
