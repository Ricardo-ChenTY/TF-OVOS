# TF-OVCOS 后续执行计划

## 总目标

把 proposal 里的 E1-E4 做成可复现 benchmark：

```text
数据集 -> manifest -> method adapter -> predictions.jsonl -> evaluator -> tables
```

所有方法都统一输出一张图一个结果：

```json
{"image_id": "...", "mask_path": "...png", "label": "...", "score": 0.0}
```

## Phase 0：机器和目录准备

推荐机器：

- A100 80G 优先
- 至少 200-300GB 空闲磁盘
- Linux/Ubuntu 优先；Windows 可以做核心评测，但第三方视觉 repo 更容易踩坑

建议目录：

```text
TF-OVCOS/
  data/
    raw/
      OVCamo/
      CAMO/
      COD10K/
      NC4K/
    manifests/
  external/
    segment-anything/
    GroundingDINO/
    sam2/
    SCLIP/
    NACLIP/
    ProxyCLIP/
  weights/
    sam/
    groundingdino/
    sam2/
    clip/
    siglip/
    dinov2/
  runs/
```

## Phase 1：下载数据集

### 必下

1. OVCamo
   - 用途：E1/E2 主 benchmark，E3 class-aware target
   - 需要：OVCamo-TE images, masks, `class_info.json`, `sample_info.json`, 官方 61 unseen labels
   - 训练 split 只给 trained references 用；strict-TF 不用训练集调参

2. CAMO-TE
   - 用途：E3 external mask-only target
   - 需要：test images + masks

3. COD10K-TE-Camo
   - 用途：E3 external mask-only target
   - 需要：COD10K test camouflaged subset images + masks

4. NC4K
   - 用途：E3 external mask-only target
   - 需要：all images + masks

### 暂不必下

- CHAMELEON：proposal 里只作为 appendix sanity check，不进主 E3
- SA-1B / 大规模 SAM 训练数据：我们只推理，不训练 SAM
- COCO / Object365：除非某个第三方 repo 强制跑它自己的 evaluation，不需要

## Phase 2：生成 manifest

不要重新处理原始数据集。只生成轻量 JSONL：

```json
{"image_id": "xxx", "image_path": "data/raw/OVCamo/TE/images/xxx.jpg", "mask_path": "data/raw/OVCamo/TE/masks/xxx.png", "label": "frog"}
```

输出：

```text
data/manifests/ovcamo_te.jsonl
data/manifests/camo_te.jsonl
data/manifests/cod10k_te_camo.jsonl
data/manifests/nc4k.jsonl
configs/vocab/ovcamo_75.txt
configs/vocab/ovcamo_61_unseen.txt
```

需要做的检查：

- 每个 image_path 存在
- 每个 mask_path 存在
- OVCamo-TE 每个样本有 label
- label 全部属于官方 OVCamo vocab
- 外部 E3 数据 label 可为 null

## Phase 3：安装基础推理环境

当前本机已经有 benchmark 核心环境：

```powershell
.\.conda\tf-ovcos\python.exe -m pytest tests
```

大 GPU 机器上建议新建同名环境：

```bash
conda create -n tf-ovcos python=3.11 -y
conda activate tf-ovcos
pip install -e ".[dev]"
```

然后装 PyTorch CUDA：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

如果服务器 CUDA/PyTorch 版本不匹配，就按 PyTorch 官网 selector 改成对应 CUDA wheel。

## Phase 4：下载第一批模型权重

第一批只下能跑通主线 baseline 的模型。

### 1. SAM

建议先下：

- `sam_vit_b_01ec64.pth`
- `sam_vit_l_0b3195.pth`

先不要默认用 `vit_h`。ViT-H 质量高，但慢且占显存。benchmark pipeline 先用 ViT-B/L 跑通，再决定是否全量换 ViT-H。

目标路径：

```text
weights/sam/sam_vit_b_01ec64.pth
weights/sam/sam_vit_l_0b3195.pth
```

### 2. GroundingDINO

先下 Swin-T：

```text
weights/groundingdino/groundingdino_swint_ogc.pth
```

后续如果 Swin-T 效果太弱，再补 Swin-B。

### 3. CLIP / open_clip

先用：

- `ViT-B-16`
- `ViT-L-14`

CLIP 权重可以由 `open_clip` 自动缓存，也可以统一设置缓存目录到 `weights/clip`。

### 4. SigLIP

第二批再下，先不挡第一条 baseline。建议用 Hugging Face Transformers：

- `google/siglip-base-patch16-224`
- 或更大版本作为 later ablation

### 5. DINOv2

第二批再下：

- `dinov2_vitb14`
- `dinov2_vitl14`

不要一开始上 `dinov2_vitg14`，显存和下载体量都更高。

### 6. SAM2

第二批再下：

- `sam2.1_hiera_tiny`
- `sam2.1_hiera_base_plus`
- `sam2.1_hiera_large` 只在 A100 80G 上做正式版本

## Phase 5：第一条可跑 baseline

优先实现：

```text
GroundingDINO + SAM ViT-B
```

原因：

- 输入是 image + vocabulary
- GroundingDINO 输出 boxes + class scores
- SAM 接 boxes 输出 masks
- 很自然转成 one mask-category pair

目标命令：

```bash
python -m tf_ovcos.run_method \
  --method groundingdino_sam \
  --manifest data/manifests/ovcamo_te.jsonl \
  --vocab configs/vocab/ovcamo_61_unseen.txt \
  --out runs/groundingdino_sam/ovcamo_te
```

然后评测：

```bash
python -m tf_ovcos.eval \
  --manifest data/manifests/ovcamo_te.jsonl \
  --predictions runs/groundingdino_sam/ovcamo_te/predictions.jsonl \
  --task class-aware \
  --out runs/groundingdino_sam/ovcamo_te/metrics.json
```

先只跑 20 张图 smoke test：

- 检查 mask 是否对齐原图尺寸
- 检查 label 是否来自 vocab
- 检查每张图只有一个 prediction
- 可视化 overlay 20 张

## Phase 6：第二条 baseline

实现：

```text
SAM-AMG + CLIP
```

流程：

1. SAM AMG 生成 class-agnostic proposals
2. 对每个 proposal 做 alpha crop 或 masked crop
3. CLIP 对 crop 和 vocabulary 打分
4. 选最高分 proposal-label pair
5. 输出 final mask + label

注意：

- 不要默认缓存所有 AMG 候选 mask
- 只缓存最终 mask；debug 时再开 `--save-candidates`

## Phase 7：扩展主表方法

按这个顺序接：

1. SCLIP 或 NACLIP
2. ProxyCLIP
3. SAM-AMG + SigLIP
4. DINOv2 + SAM + CLIP
5. DINOv2 + SAM + SigLIP
6. MCC reranking
7. GroundingDINO + SAM2

最后再接：

- MaskCLIP
- CLIP-DIY
- ResCLIP
- CorrCLIP
- Trident
- FreeDA / OVDiff
- OVCoser / SuCLIP trained references

## Phase 8：并行和断点续跑

实现 shard 机制：

```bash
python -m tf_ovcos.make_shards \
  --manifest data/manifests/ovcamo_te.jsonl \
  --num-shards 8 \
  --out data/manifests/shards/ovcamo_te
```

每个 worker 独立输出：

```text
runs/<method>/<dataset>/shards/part-000/predictions.jsonl
```

合并：

```bash
python -m tf_ovcos.merge_predictions \
  --manifest data/manifests/ovcamo_te.jsonl \
  --shards runs/<method>/<dataset>/shards \
  --out runs/<method>/<dataset>/predictions.jsonl
```

这样可以：

- 多 GPU 并行
- 单 GPU 多 worker
- 中断后只重跑失败 shard

## Phase 9：E3 和 E4

E3：

```text
OVCamo-TE: class-aware
CAMO-TE: mask-only
COD10K-TE-Camo: mask-only
NC4K: mask-only
```

外部数据统一输入 OVCamo-75 vocab，预测 label 忽略，只评 mask。

E4：

分两套记录：

1. `isolated`: 单方法、单 worker、固定 GPU，写论文主表
2. `throughput`: 并行跑完整 benchmark 的 wall-clock time，写工程附录或说明

## Milestone

### M1：核心闭环

- manifest 生成完成
- `GroundingDINO + SAM` 跑 20 张
- evaluator 输出 E1/E2 metrics

### M2：第一版主表

- `GroundingDINO + SAM`
- `SAM-AMG + CLIP`
- `SCLIP/NACLIP`
- `ProxyCLIP`
- OVCamo-TE 全量 E1/E2

### M3：E3 完成

- 四个 dataset manifest
- 外部 mask-only metrics
- dedup index

### M4：并行完成

- shard
- merge
- resume
- 多 GPU 调度

### M5：论文完整实验

- 主方法补齐
- E4 isolated cost
- ablation
- LaTeX table 自动填充
