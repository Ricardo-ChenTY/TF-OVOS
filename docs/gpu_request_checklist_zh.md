# 申请 GPU 前准备清单

这份清单用于在正式申请大显存机器前，确认 TF-OVCOS benchmark 已经具备可执行输入。目标是拿到 GPU 后可以直接进入 baseline adapter 调试，而不是先花时间整理数据、标签和路径。

## 当前仓库状态

已具备：

- benchmark 核心骨架：manifest、shard、method runner、prediction merge、evaluator
- debug adapters：`debug_copy_gt`、`debug_empty`
- E1/E2/E3 统一预测格式：一张图输出一个 `mask_path + label + score`
- 并行执行设计：按 dataset/method/shard 拆分，合并后统一评测
- 基础测试：可用 `PYTHONPATH=src python -m pytest tests` 验证

尚未具备：

- 官方 OVCamo 75 类 vocab 和 61 unseen vocab
- OVCamo-TE、CAMO-TE、COD10K-TE-Camo、NC4K 的本地数据
- `data/manifests/*.jsonl`
- 真实方法 adapter，例如 `groundingdino_sam`、`sam_amg_clip`
- 第三方方法代码 repo 和模型权重

## 申请 GPU 前必须准备

### 1. 数据集

建议先准备这些目录：

```text
data/raw/
  OVCamo/
    TE/
      images/
      masks/
    class_info.json
    sample_info.json
  CAMO/
    TE/
      images/
      masks/
  COD10K/
    TE-Camo/
      images/
      masks/
  NC4K/
    images/
    masks/
```

必做检查：

- 图片和 mask 数量一致
- mask 是单通道或可稳定转成二值 mask
- OVCamo-TE 每张图有官方类别 label
- CAMO/COD10K/NC4K 外部目标不需要 OVCOS 类别 label
- NC4K 和 OVCamo 做 filename/MD5/pHash 去重记录

### 2. Vocabulary

补齐：

```text
configs/vocab/ovcamo_75.txt
configs/vocab/ovcamo_61_unseen.txt
```

要求：

- 每行一个类别名
- `ovcamo_75.txt` 包含官方 14 seen + 61 unseen
- `ovcamo_61_unseen.txt` 只包含官方 unseen 类
- 所有 manifest 里的 OVCamo label 必须能在对应 vocab 中找到

### 3. Manifest

生成并检查：

```text
data/manifests/ovcamo_te.jsonl
data/manifests/camo_te.jsonl
data/manifests/cod10k_te_camo.jsonl
data/manifests/nc4k.jsonl
```

示例命令：

```bash
PYTHONPATH=src python -m tf_ovcos.make_manifest \
  --image-dir data/raw/OVCamo/TE/images \
  --mask-dir data/raw/OVCamo/TE/masks \
  --label-source data/raw/OVCamo/sample_info.json \
  --out data/manifests/ovcamo_te.jsonl \
  --require-labels
```

外部 mask-only 数据：

```bash
PYTHONPATH=src python -m tf_ovcos.make_manifest \
  --image-dir data/raw/NC4K/images \
  --mask-dir data/raw/NC4K/masks \
  --out data/manifests/nc4k.jsonl
```

### 4. 本地 smoke test

在申请 GPU 前，至少跑通 debug pipeline：

```bash
PYTHONPATH=src python -m pytest tests
PYTHONPATH=src python -m tf_ovcos.run_method --list-methods
```

如果已经有 `ovcamo_te.jsonl`，再跑：

```bash
PYTHONPATH=src python -m tf_ovcos.make_shards \
  --manifest data/manifests/ovcamo_te.jsonl \
  --num-shards 8 \
  --out-dir data/manifests/shards/ovcamo_te

PYTHONPATH=src python -m tf_ovcos.run_method \
  --method debug_copy_gt \
  --manifest data/manifests/shards/ovcamo_te/part-000.jsonl \
  --vocab configs/vocab/ovcamo_61_unseen.txt \
  --out runs/debug_copy_gt/ovcamo_te/shards/part-000
```

### 5. 第一批方法范围

申请 GPU 时先承诺第一批 baseline，而不是一次性跑完所有 rows：

1. `GroundingDINO + SAM ViT-B`
2. `SAM-AMG + CLIP ViT-B/16`
3. `SCLIP` 或 `NACLIP`
4. `ProxyCLIP`

第二批再补：

- `GroundingDINO + SAM2`
- `SAM-AMG + SigLIP`
- `DINOv2 + SAM + CLIP/SigLIP`
- `MCC` re-ranking ablations

扩散和 reference-based 方法放最后：

- `FreeDA`
- `OVDiff`
- compact trained references：`OVCoser`、`SuCLIP`

## 申请机器配置

推荐申请：

```text
OS: Ubuntu/Linux
GPU: A100 80GB preferred; 24GB minimum for first baselines
CPU RAM: 64GB preferred; 32GB minimum
Disk: 300GB preferred; 150GB minimum
CUDA: match PyTorch CUDA wheel
Network: required for model weights and third-party repos
```

如果只能申请普通单卡，优先级：

1. 24GB 显存：可以跑第一批 baseline，但 batch/worker 要保守
2. 12GB 显存：可以做小规模 smoke test，不适合完整主表
3. 4GB 显存：只适合 evaluator、manifest、debug adapter 和小模型检查

## 申请说明模板

可以直接使用下面这段说明：

```text
I need a Linux CUDA GPU machine to run a reproducible benchmark for training-free open-vocabulary camouflaged object segmentation. The benchmark evaluates frozen public foundation-model pipelines, including GroundingDINO+SAM, SAM-AMG+CLIP, SCLIP/NACLIP, and ProxyCLIP, on OVCamo-TE plus CAMO-TE, COD10K-TE-Camo, and NC4K. The code is already structured around dataset manifests, sharded inference, unified prediction JSONL, and a shared evaluator. I need at least 24GB GPU memory for first baselines, with A100 80GB preferred for full method coverage and stable E4 cost measurement. Expected disk need is 150-300GB for datasets, model weights, third-party repos, and cached predictions.
```

## 拿到 GPU 后第一天执行顺序

1. 创建环境并安装 benchmark：

```bash
conda create -n tf-ovcos python=3.11 -y
conda activate tf-ovcos
pip install -e ".[dev]"
```

2. 安装匹配 CUDA 的 PyTorch。
3. 下载 SAM ViT-B/L、GroundingDINO Swin-T、CLIP ViT-B/16 权重。
4. 跑 `PYTHONPATH=src python -m pytest tests`。
5. 跑 `debug_copy_gt` shard smoke test。
6. 实现并验证 `groundingdino_sam` adapter 的 20 张图输出。
7. 确认 mask 尺寸、label vocab、每图一条 prediction、overlay 可视化。
8. 再跑完整 OVCamo-TE E1/E2。
