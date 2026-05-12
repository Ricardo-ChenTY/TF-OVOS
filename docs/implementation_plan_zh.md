# TF-OVCOS Benchmark 代码搭建方案

## 目标

这个项目不要一开始写成“某个模型的代码库”。proposal 的核心是评测框架，所以代码应先固定四件事：

1. 数据 manifest 格式
2. 方法统一输出格式
3. E1/E2/E3/E4 的评测入口
4. 每类方法的 adapter 接口

这样后面接 MaskCLIP、SCLIP、GroundingDINO+SAM、SAM-AMG+CLIP 等方法时，不会把每个 repo 的环境和评测脚本混在一起。

## 推荐目录

```text
TF-OVCOS/
  configs/
    benchmark.yaml
    vocab/
      ovcamo_75.txt
      ovcamo_61_unseen.txt
  data/
    manifests/
      ovcamo_te.jsonl
      camo_te.jsonl
      cod10k_te_camo.jsonl
      nc4k.jsonl
  external/
    MaskCLIP/
    SCLIP/
    NACLIP/
    ProxyCLIP/
    Grounded-Segment-Anything/
  runs/
    <method>/<dataset>/predictions.jsonl
    <method>/<dataset>/metrics.json
  src/tf_ovcos/
    data.py
    metrics.py
    eval.py
    adapters/
```

`external/` 里放第三方方法代码，`src/tf_ovcos/` 只保留 benchmark 自己的稳定逻辑。

## 统一数据格式

每个 split 一个 JSONL manifest：

```json
{"image_id": "0001", "image_path": "data/OVCamo-TE/images/0001.jpg", "mask_path": "data/OVCamo-TE/masks/0001.png", "label": "frog"}
```

外部 E3 数据集没有 OVCOS 类别时：

```json
{"image_id": "NC4K_0001", "image_path": "data/NC4K/images/0001.jpg", "mask_path": "data/NC4K/masks/0001.png", "label": null}
```

## 统一预测格式

所有方法最后必须输出：

```json
{"image_id": "0001", "mask_path": "pred_masks/0001.png", "label": "frog", "score": 0.73}
```

注意：每张图只允许一个 mask-category pair。这对应 proposal 的
`I + C -> (M_hat, y_hat)`。

## E1 / E2 / E3 实现

E1：

- 输入：OVCamo-TE + 官方 61 unseen labels
- 输出：class-aware metrics
- 当前 evaluator 已支持 `cIoU`, `cF_beta`, `cE_m`, `cMAE`, `cBIoU`
- label 错误时 overlap 类指标记 0，`cMAE` 记 1

E2：

- 复用 E1 predictions
- 统计 `Loc@0.5`, `Exact@0.5`, `ClsErr@Loc`
- 收集 localized-but-wrong 的 `gt -> pred` confusion edges

E3：

- OVCamo-TE：继续 class-aware
- CAMO-TE / COD10K-TE-Camo / NC4K：mask-only，忽略预测 label
- 固定输入 vocab 为 OVCamo-75，避免外部数据集 prompt engineering

E4：

- 单独记录运行成本，不和 mask evaluator 混在一起
- 建议每个 adapter 运行时写 `runtime.json`
- 字段包括：GPU 名称、输入尺寸、平均每图秒数、峰值显存、模型调用次数、权重路径

## 方法接入优先级

第一阶段，先做最快能闭环的 baseline：

1. `GroundingDINO + SAM`
2. `SAM-AMG + CLIP`
3. `SCLIP` 或 `NACLIP`
4. `ProxyCLIP`

原因：

- GroundingDINO+SAM 天然输出 box-label-mask，最接近 benchmark 格式
- SAM-AMG+CLIP 能测试 proposal+naming 路线
- SCLIP/NACLIP 覆盖 dense CLIP map 路线
- ProxyCLIP 覆盖 CLIP+VFM 路线

第二阶段再接：

- MaskCLIP / CLIP-DIY / ResCLIP
- GroundingDINO + SAM2
- DINOv2 + SAM + CLIP/SigLIP + MCC
- FreeDA / OVDiff

FreeDA 和 OVDiff 依赖更重，还有 offline prototype/reference artifacts，建议最后做。

## Adapter 设计

每个 adapter 做三件事：

1. 调第三方 repo 或模型
2. 把原始输出转换成 candidate masks
3. 选出一个最高分 `(mask, label)` 写入 predictions.jsonl

不要让 adapter 直接算论文指标。所有指标统一走 `python -m tf_ovcos.eval`。

## 下一步

1. 补齐 `configs/vocab/ovcamo_75.txt` 和 `configs/vocab/ovcamo_61_unseen.txt`
2. 写脚本把 OVCamo-TE、CAMO-TE、COD10K-TE-Camo、NC4K 转成 manifest
3. 先接 `GroundingDINO + SAM` adapter，跑 20 张图 smoke test
4. 再跑完整 OVCamo-TE，生成 E1/E2 第一条 baseline
