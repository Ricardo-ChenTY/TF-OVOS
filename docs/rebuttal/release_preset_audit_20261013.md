# 官方预设与 Trident SAM 核验（2026-10-13）

按最新决定，未启动统一超参数重跑，也未修改任何模型超参数、既有预测或冻结结果。此前提出的 VOC 配方迁移与第二套敏感性实验均未启动。

## Trident：ViT-B 与所核对的官方代码一致

核对的是官方仓库提交 `7feea10222bf15d5aabdeabefde5902a62584d94`，而非凭本项目 `cfg_tfovos` 中的字段推断：

- [官方 base_config.py](https://github.com/YuHengsss/Trident/blob/7feea10222bf15d5aabdeabefde5902a62584d94/configs/base_config.py) 明确写 `sam_model_type='vit_b'`、`sam_vit_b_01ec64.pth`、`sam_refinement=False`。
- 本项目六个目标的历史质量运行配置均记录同样的 SAM 型号、文件名和 refinement 开关。项目生成器将 checkpoint 改为本机路径并重复指定 `vit_b`；这不是把官方 ViT-H 换成 ViT-B。
- [官方 trident.py](https://github.com/YuHengsss/Trident/blob/7feea10222bf15d5aabdeabefde5902a62584d94/trident.py) 的 `get_trident_seg` 仍调用 `get_sam_feat`，取得 SAM encoder features、attention 和 value tensors。关闭末端 refinement 不等于没有使用 SAM。
- 实际应列为 **CLIP ViT-B/16 (OpenAI) + DINO ViT-B/16 + SAM ViT-B；optional final SAM refinement off**。只写 CLIP+DINO 会漏掉实际使用的模型与成本。
- [官方 eval.py](https://github.com/YuHengsss/Trident/blob/7feea10222bf15d5aabdeabefde5902a62584d94/eval.py) 的 `--sam_refine` 默认为 false，但 README 示例命令带该开关。应写清运行的 off 变体，不能笼统宣称与 README 示例或所有论文结果完全相同。

当前缓存 checkpoint 为 375,042,383 字节，SHA256：

```text
ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912
```

它与 2026-10-09 的权重哈希记录相符。历史 5 月日志中的旧 checkpoint 路径现已不存在，因此能确认历史运行的型号、文件名和开关，但不能据当前缓存证明已丢失旧文件的逐字节身份。

[Trident 结构化证据及六份质量配置哈希](evidence/release_presets_20261013/trident_sam_audit.json)。

## “全部设置都来自官方发布”不能原样使用

本次直接读取五个 dense 方法的源码提交、构造默认值、配置继承，以及六个目标各自最新成功的历史质量日志，共 30 份配置；另比较 FreeDA 四个 compact 目标的完整官方 `model` 段。日志选择规则不等于已逐一证明所有论文单元格与日志的唯一对应关系，审计范围在 JSON 中明确记载。

| 方法 | 已确认来源与设置 | 必须披露的区别 |
| --- | --- | --- |
| Trident | 六个目标继承对应官方配置；SAM ViT-B 型号一致；分辨率和滑窗设置随目标变化 | 末端 SAM refinement off；不应遗漏 SAM |
| SCLIP | compact 目标官方 resize：VOC/Context59/ADE150 为 336，COCO 为 448 | Context459 **和 ADE847 都是 448**；这两份为项目大词表扩展，所核对官方提交没有对应原生配置 |
| ResCLIP | compact 的 temp_thd 与 RCS/SFR 权重吻合官方对应预设 | VOC 的官方 `slide_crop=0`，项目配置遗漏此项，实际构造默认 **224**；这是推理差异，不是路径替换。大词表配置也为项目扩展，resize=448 |
| CASS | compact 的权重/阈值吻合对应官方配置 | 大词表配置由项目分别沿用 Context59/ADE150 的参数；不能称为官方原生大词表配置。PAMR=0 符合官方 eval.py 默认 `--pamr off`，不是新发现的偏离 |
| FreeDA | 四个 compact 目标官方 model 段的所有字段均与质量配置一致 | superpixel (min_size, sigma)：VOC=(20,0.7)、Context59/ADE150=(20,1.0)、COCO=(100,1.0)；不能称跨目标同一配方；本比较不证明全部评估/数据处理从未修改 |
| NACLIP | compact 的 VOC `slide_crop=0`、其余默认224，符合官方对应设置 | 也有跨目标配方差异，不能只标此前的“五个方法”。Context459/ADE847 是项目扩展，resize=448 |

CASS 比较时必须合并官方 eval.py 的 CLI 默认覆盖，否则只看构造函数 `pamr_steps=10` 会误报本项目 PAMR=0 的差异。本审计已处理这一点。

材料：

- [30 份质量设置 CSV](evidence/release_presets_20261013/quality_settings.csv)
- [配置继承、默认值、来源关系及逐项差异](evidence/release_presets_20261013/dense_config_audit.json)
- [FreeDA 四个 model 段比较](evidence/release_presets_20261013/freeda_config_audit.json)
- [只读核查脚本](../../scripts/audit_release_presets.py)

源码中还存在此前已记录的输出适配、标签修复等补丁；参数取值吻合不代表源码逐字节未改。这里核对的是具体来源提交，不证明原作者如何选择参数，也不证明我方在所有历史阶段均未看过结果后修正配置。尤其不能由现存配置推出“从未改过任何设置”。

## 建议的 Z6dd A2 替换段落

> We agree that E3 should be described as cross-dataset stability rather than domain generalization. Our configuration audit shows that the evaluated implementations used target-dependent presets and documented project adaptations, rather than one invariant inference recipe across targets. Thus, the submission's fixed-recipe description was inaccurate. We will retain the measured results, identify the evaluated configurations in Tables 1 and 2, and list their per-target settings, source commits, implementation deviations, and inference variants in the appendix. E3 will describe variation in performance across the evaluated targets under those declared configurations; it will not establish generalization of a single configuration to unseen targets. We cannot verify how the original authors selected their presets, so possible benefits from dataset-specific preset selection will be acknowledged. The Introduction, Section 3.1, E3, and the Conclusion will be revised consistently.

这一段有意不写：

- “Every configuration was taken from the official release”：大词表项目扩展和 ResCLIP VOC 滑窗偏离使这个全称判断不成立。
- “We did not adjust any setting after seeing results”：现有证据不足以支持这段历史行为断言。
- “Any shared setting chosen now would itself be target-specific tuning”：事先锁定、完整披露的新对照不自动等于调参；本次选择不补跑是收窄研究主张与保留已有结果，而非证明新对照在方法上不可做。
- 重新把“每个目标分别固定”叫作“跨目标 fixed recipe”：这两个实验条件必须区分，不能只重新定义术语。

Trident 设置表与解释可直接使用：

> Trident uses CLIP ViT-B/16 (OpenAI), DINO ViT-B/16, and SAM ViT-B, with the optional final SAM refinement disabled. The SAM backbone agrees with the audited upstream configuration. SAM encoder features and attention are still used when final refinement is disabled.

## 改稿动作与边界

Introduction、3.1、E1 的固定配方表述、Table 1/2 设置标注、E3/Table 3 图表标题及 Conclusion 均需一致收窄。先标已核实的六个方法，不能据此宣称其余全部方法已经通过配置不变审计。附录分别列“官方对应目标预设”“项目大词表扩展”“已确认实现偏离”。

ResCLIP 的旧 VOC 结果可保留为明确标注的项目实现结果，但不可写为严格原样复现官方 VOC 配置。本次按用户决定不补跑此项，也不静默把配置改成0，以免配置与现有结果再次不对应。

不启动新实验不代表原跨目标固定配方主张已验证。保留原始数据盘时可以停止计算；本核查没有执行服务器关机或磁盘释放。
