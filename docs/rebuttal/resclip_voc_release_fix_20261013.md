# ResCLIP VOC 官方滑窗修正：全量成对复测（2026-10-13）

**已完成。** 只把遗漏的官方 VOC `slide_crop=0` 恢复，未更换模型权重、阈值、RCS/SFR 权重、336 分辨率或 benchmark 类别名称。用同一当前环境完成旧设置与修正设置各 1,449 张，共 2,898 张预测，并从每张保存的 PNG 独立重算混淆矩阵和 mIoU。

| 版本 | 图片数 | VOC mIoU（%） | 本轮平均在线推理（秒/张） |
| --- | ---: | ---: | ---: |
| 历史 PNG 重评分（原稿 78.51） | 1,449 | 78.511226 | — |
| 同环境对照：slide_crop=224 | 1,449 | 78.511350 | 0.093682 |
| 修正：官方 slide_crop=0 | 1,449 | **82.133211** | **0.037545** |

同环境下只改变滑窗设置，mIoU **+3.621861 个百分点**。新环境旧设置与历史 PNG 的 mIoU 仅差 +0.000124 个百分点，但逐像素不完全一致，因此没有把历史到当前的全部变化直接归因于滑窗。旧、新设置纯推理分别约 135.75 秒和 54.40 秒；加载、保存、评分和核验另计。

## 修正范围与来源

官方 ResCLIP 提交 `0e085f7f52f06098b383599e9653cd28cc665a8c` 的 [cfg_voc20.py](https://github.com/yvhangyang/ResCLIP/blob/0e085f7f52f06098b383599e9653cd28cc665a8c/configs/cfg_voc20.py) 指定 `slide_crop=0`。项目生成器此前未传此项，落到构造默认224。本次只恢复这一推理参数；其他方法和数据集没有重新推理。

本次追溯原稿 **78.51** 对应的 `official_resclip_voc20_datafix.log` 和 `cls_tfovos_voc20.txt`。另有历史官方别名运行得到82.31，但那份使用 `cls_voc20.txt`，不是本次对照来源。保留 benchmark 名称是为了隔离滑窗修正，不能把本结果描述为“包括官方同义词名称在内的整个官方实验原样复现”。

此前的来源审计按每个目标最新成功日志选择样本，已经注明不证明所有投稿单元格的唯一来源；本次进一步锁定了实际投稿 VOC 单元格，记录了来源配置和词表哈希。

配置生成器以及发布的两份 VOC 配置副本已修正为0，避免今后再次遗漏。历史日志、旧预测、冻结包和实验服务器中的旧配置保持原样，由定向运行在实例化后设置0，完整记录在 metadata 中。

## 论文与回复更新

- **Table 1：** ResCLIP 的 VOC 单元格改为 **82.13**。
- **E3 / Mean、Worst：** 其余三个 compact 单元格保持36.85、18.07、23.59时，Mean 为 **40.16**、Worst 仍为 **18.07**。这是按原稿其余三格两位小数计算；全精度最终汇总应使用原始完整数值。E3 仍按之前决定收窄为所声明配置下的跨目标表现。
- **E4：** 本页0.037545为修正后 VOC 全量在线时间。旧18行抽样表的 ResCLIP 行仍对应旧 VOC 滑窗配置，不能把它与新质量值直接宣称为同配方配对；也不能把全量VOC均值直接混入其他三个25张抽样均值冒充新的统一四目标计时表。
- **Table 2 / fixed-GT：** 此次没有改变 Context/ADE 结果、Δvocab、共同 GT 或候选扩张曲线。它不是跨所有目标统一一套超参数的实验。

可补入回复：

> Our audit identified an omitted VOC-specific inference setting in the ResCLIP wrapper: the released VOC configuration disables sliding-window inference, whereas our wrapper had used the constructor default of 224. We restored the released setting and evaluated all 1,449 VOC images, retaining the benchmark class names and all other settings. In a paired rerun under the same software environment, VOC mIoU increased from 78.5114% to 82.1332%. We will replace the VOC entry and update the corresponding aggregate, and explicitly distinguish historical timing measurements from the corrected configuration.

## 大词表扩展的处理

本次没有把“项目扩展”自动判成错误，也没有再任意指定一套新超参数：

1. 官方没有 Context459/ADE847 配置时，增加类别表、数据集接口和相应 GT 读取属于必要适配；CASS、ResCLIP 沿用对应 compact 目标的权重/阈值已经查明来源。
2. SCLIP、NACLIP、ResCLIP 等项目 E2 配置使用448，是额外的已记录预处理选择，须明确披露；不能冒称对应官方大词表预设。仅凭它不同于 compact 的336，不能断定预测无效，更不能称把它改成336就是恢复某个不存在的官方 E2 配置。
3. 现有大词表标签与 GT 重评分已有独立核验。此次未发现像 VOC 漏传已知官方参数那样明确、可以定向恢复的另一项参数错误。

因此本轮仅修正有明确官方依据的 VOC 漏项。大词表保持原结果并披露适配；若未来要检验336/448敏感性，应单列为控制实验，并完整报告两种设置，不依据哪一个成绩高来决定保留哪个。

## 证据与复核

- [成对结果 CSV](evidence/resclip_voc_release_20261013/comparison.csv)、[完整比较 JSON](evidence/resclip_voc_release_20261013/comparison.json)
- [全部2,898张预测核验和哈希](evidence/resclip_voc_release_20261013/prediction_verification.json)、[输入/GT哈希](evidence/resclip_voc_release_20261013/input_hashes.json)
- [两组预测 PNG 压缩包](artifacts/resclip_voc_release_predictions_20261013.tar.gz)、[SHA256](artifacts/resclip_voc_release_predictions_20261013.tar.gz.sha256)。包内 slide224/ 与 slide0/ 的000000.png等按 input_hashes.json 顺序对应 image_id；均为0-based预测。包不含全部VOC原图/GT，重新计算mIoU需要对应GT。
- [执行计划](evidence/resclip_voc_release_20261013/plan.json)、[协议](evidence/resclip_voc_release_20261013/protocol.json)、[冻结运行代码](evidence/resclip_voc_release_20261013/code/measure_resclip_voc_release.py)、[来源文件哈希](evidence/resclip_voc_release_20261013/native_source_hashes.json)
- [成对核验与汇总程序](../../scripts/summarize_resclip_voc_release.py)、[本轮启动程序](../../scripts/run_resclip_voc_release.py)

此页为新增修正版，原20261011冻结包不被静默覆盖。本轮任务已结束；未进行服务器关机或存储释放。
