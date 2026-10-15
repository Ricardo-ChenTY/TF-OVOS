# CLIPtrase：补齐 ADE-847，并统一两组原生词表比较的指标分母

**已完成保存预测的 CPU 重评分，不需要新增 GPU 推理。** ADE-847 的 2,000 张预测并未缺失。之前的通用脚本 `rescore_ade847_official_gt.py` 明确拒绝 CLIPtrase，因为该脚本按全图几何重评分，不能将中心裁剪预测拉伸到全图。缺的是专门的裁剪评分流程，并非 center crop 无法获得 ADE-847 mIoU。

## 补表时采用什么数字

保持 CLIPtrase 原生的 `Resize(336) + CenterCrop(336)` 图像范围；GT 使用原生 `Resize(336, nearest) + CenterCrop(336)`。保存预测为 336×336，不拉伸、不改预测类别。ADE-847 GT 使用已恢复的官方类别编号，void 为 65535；零起始预测不减 1。

为与当前仓库 `metrics.miou_from_confusion` 的定义一致，补充表采用 **union-supported mIoU**；原生官方评分的 GT-supported 数字单列用于追溯。

| 目标 | 完整图像数 | 官方原生 GT-supported mIoU (%) | 统一 union-supported mIoU (%) | 裁剪内 GT / union 支持类别数 |
| --- | ---: | ---: | ---: | ---: |
| Context-59 | 5,105 | 34.580399 | 34.580399 | 59 / 59 |
| Context-459 | 5,105 | 9.952560 | **7.541829** | 341 / 450 |
| ADE-150 | 2,000 | 17.037592 | 17.037593 | 150 / 150 |
| ADE-847 | 2,000 | 5.891525 | **5.821388** | 830 / 840 |

Context-459 的原生 9.95 只平均 GT 中出现的 341 类；union-supported 还纳入仅在预测中出现的类别，共 450 类，得到 7.54。ADE-847 对应 830 与 840 类，得到 5.89 与 5.82。原生实现还在每类 IoU 分母加 0.01；本次同时输出不加 epsilon 的 GT-supported 数值，确认主要差异来自支持集合而非这个很小的平滑项。

**不能只把 ADE-847 填成 5.82，却继续使用 Context-459 的原生 9.95 来算同一条差值。** 两端及两组比较均需注明一致的分母。

| 口径 | Context 下降 (pp) | ADE 下降 (pp) | 两组平均下降 Δ_vocab (pp) | Context 相对下降 | ADE 相对下降 | 两组相对下降的算术平均 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 官方原生 GT-supported | 24.627839 | 11.146068 | 17.886953 | 71.2191% | 65.4204% | 68.3198% |
| **统一 union-supported** | **27.038570** | **11.216204** | **19.127387** | **78.1905%** | **65.8321%** | **72.0113%** |

使用全精度端点先计算再四舍五入。若表格采用当前统一口径，显示 Context-459 **7.54**、ADE-847 **5.82**、Δ_vocab **19.13**。如果 relative drop 列只描述 Context，填 **78.19%**；如果改为两组比例的等权平均，填 **72.01%** 并明确列名，不能将二者混写。

对应 union-supported ZIoU：Context-459 **54.22%**、ADE-847 **59.76%**。这是裁剪范围内的诊断，不能当成共同全图 GT 审计的结果。

## 原稿 5.89 是否已经证明错误

**没有。** 历史 `official_cliptrase_ADEfull.log` 明确记录 `image_size=336, eps=1.1, min=7` 的 mIoU 为 5.89。本次用官方类别编号 GT 和保存的原生裁剪预测重算，按同一官方指标得到 5.891524577%，再次四舍五入为 5.89；Context-459 的原生 9.95 也得到复现。

因此不应在回复中写“CLIPtrase 的旧 5.89 已知由错误类别偏移造成”。其他方法的 ADE-847 标签修复，不能直接证明这一个原生日志分数也错。此次应说明：此前未实现裁剪专用重评分，因此暂时缺项；现在已完成，并同时统一指标分母。

历史配置指向的旧 ADEfull `.tif` GT 路径目前不存在，故这里没有声称旧 GT 与新 GT 字节一致，也没有对旧标签错误做逐像素因果归因。可确认的是当前官方编号下的分数，以及它与原生日志显示值的一致性。ADE847 的 847 个类别名按顺序核验，只有第 300 类的撇号书写差异，未改动候选词或重跑模型。

## 可直接补入回复的英文

> We have now rescored all 2,000 saved CLIPtrase ADE-847 predictions using the official class IDs and the method's native 336-pixel resize-and-center-crop geometry, applying the corresponding nearest-neighbor transform to GT. The earlier omission reflected the lack of a crop-aware rescoring path, not an inability to evaluate ADE-847 or missing predictions. The native GT-supported evaluator yields 5.8915%, reproducing the submitted rounded value of 5.89%; we therefore do not attribute that value to an established class-ID error. Under the union-supported mIoU convention used in our corrected evaluation, ADE-847 is 5.8214% and Context-459 is 7.5418%. With both compact-to-large pairs rescored consistently, their mean absolute drop is 19.1274 percentage points. We will label CLIPtrase's center-crop geometry explicitly and keep it outside the common full-image C/W/U audit.

## 证据与修改范围

- [四目标完整分数、支持集合与 ZIoU](evidence/cliptrase_native_crop_20261015/scores.csv)
- [全精度两组差值与相对下降](evidence/cliptrase_native_crop_20261015/pair_drops.json)
- [核查结论及来源](evidence/cliptrase_native_crop_20261015/audit.json)
- [四个混淆矩阵](evidence/cliptrase_native_crop_20261015/confusion_matrices.npz)
- [原生裁剪及保存函数](evidence/cliptrase_native_crop_20261015/native_crop_and_export.txt)、[原生指标公式](evidence/cliptrase_native_crop_20261015/native_metric.py)
- [CPU 重评分脚本](../../scripts/rescore_cliptrase_native_crop.py)
- [轻量审查包](artifacts/cliptrase_native_crop_20261015.tar.gz)及 [SHA256](artifacts/cliptrase_native_crop_20261015.tar.gz.sha256)：含 14,210 张预测/GT 的哈希记录、汇总、混淆矩阵和脚本，不含模型权重。

四个混淆矩阵均通过仓库正式 `miou_from_confusion` 独立复核；每目标图像 ID 唯一，共重评分 14,210 张保存预测。审查包可查看和复核汇总；重新从 PNG 重评分仍需原服务器保存的预测及 GT。

修改 Table 2、oBoy A2 的端点、差值、relative-drop 列名及解释，并保留中心裁剪脚注。共同全图 C/W/U 和 MCMR 不因此补值，不能以本次 mIoU 核验替代那些诊断。固定 GT 曲线没有改变。本次已补证据与建议措辞，未直接编辑最终论文 PDF 或用户的最新回复全文。

四种 proposal 全量效率任务已按用户最新指示取消：只准备了部分清单，没有启动 GPU 队列。本报告的 CPU 重评分与该取消任务分开。
