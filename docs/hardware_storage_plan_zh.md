# TF-OVCOS 硬件和磁盘估算

## 结论

如果目标是把 proposal 里的主线方法接起来并能稳定跑 E1/E2/E3，建议机器配置：

- GPU：优先 24GB 显存；最低可用 12GB；4GB 只能做小模型 smoke test
- 系统内存：32GB 起步，64GB 更稳
- 磁盘：建议预留 300GB；最低不要低于 150GB
- 系统：Linux + CUDA 环境最省事；Windows 可以跑 benchmark 核心，但 GroundingDINO/SAM2/部分 OVSS repo 会更折腾

当前本机 RTX 3050 Laptop 4GB 显存可以用于：

- 跑 evaluator
- 生成 manifest
- 小规模 CLIP/SAM ViT-B smoke test
- 检查 20 张图的输出格式

不适合完整跑所有方法，尤其是 SAM ViT-H、GroundingDINO + SAM 批量推理、DINOv2/SAM2/扩散类方法。

## 磁盘预算

### 推荐预留

| 项目 | 保守估计 |
|---|---:|
| conda / pip 环境，含 PyTorch CUDA | 10-20GB |
| 第三方方法代码 repo | 5-15GB |
| 模型权重基础包 | 10-30GB |
| OVCamo + CAMO + COD10K + NC4K 解压后 | 10-30GB |
| 每个方法的预测 mask / JSON / 日志 | 1-10GB |
| 20 个方法全量输出缓存 | 30-100GB |
| 临时文件、下载缓存、失败重跑余量 | 50GB+ |

最低能跑版本：约 80-120GB。

比较舒服的 benchmark 开发版本：约 200-300GB。

如果要保留每个方法的中间 heatmap、proposal、SAM AMG 全候选 mask、可视化图片，建议 500GB。

## 主要权重大小

| 模块 | 估计大小 | 备注 |
|---|---:|---|
| SAM ViT-B | 375MB | 当前 4GB 显存机器优先用这个 smoke test |
| SAM ViT-L | 1.2-1.3GB | 质量和成本折中 |
| SAM ViT-H | 2.4-2.6GB | 论文常用，但显存压力大 |
| GroundingDINO Swin-T | 694MB | 第一条 detector+SAM baseline |
| CLIP ViT-B/L | 0.4-1GB | open_clip/transformers 缓存还会多占一些 |
| SigLIP | 1-3GB | 取决于 backbone |
| DINOv2 ViT-B/L/G | 0.4-4GB+ | ViT-G 不建议先上 |
| SAM2 | 0.2-1GB+ | 取决于 tiny/base/large |
| diffusion rows | 10GB+ | FreeDA/OVDiff 相关依赖和缓存重，放最后 |

## 数据集是否需要重新处理

原则上不需要重新处理数据集。

我们只需要做三类轻量操作：

1. 生成 JSONL manifest：记录 `image_id`, `image_path`, `mask_path`, `label`
2. 对 E3 外部数据做去重记录：filename/MD5/pHash，避免和 OVCamo 重叠
3. 运行时按模型需要 resize，尽量不保存永久 resize 版本

不要把图片和 mask 全部重写成另一套格式。这样可以保持数据可追溯，也避免多占几十 GB。

例外：

- 某些第三方 repo 强制要求固定目录名或 list 文件，可以生成 adapter 专用 list/cache，但不要替换原始数据
- SAM AMG 全候选 mask 如果缓存下来会很大，建议只缓存最终 one-pair mask；需要 debug 时再单独开候选缓存
- 如果做 pHash 去重，可以保存一个很小的 `dedup_index.jsonl`

## 方法接入优先级

第一批，目标是最快跑通论文主表雏形：

1. GroundingDINO + SAM ViT-B
2. SAM-AMG + CLIP ViT-B/16
3. SCLIP 或 NACLIP
4. ProxyCLIP

第二批，补齐 proposal 的代表性：

1. GroundingDINO + SAM2
2. SAM-AMG + SigLIP
3. DINOv2 + SAM + CLIP/SigLIP
4. MCC reranking

第三批，最后接：

1. MaskCLIP / CLIP-DIY / ResCLIP / CorrCLIP / Trident
2. FreeDA / OVDiff
3. OVCoser / SuCLIP trained references

## 能直接跑的目标形态

每个方法最终都应该支持同一种命令形态：

```powershell
.\.conda\tf-ovcos\python.exe -m tf_ovcos.run_method `
  --method groundingdino_sam `
  --manifest data/manifests/ovcamo_te.jsonl `
  --vocab configs/vocab/ovcamo_61_unseen.txt `
  --out runs/groundingdino_sam/ovcamo_te
```

然后统一评测：

```powershell
.\.conda\tf-ovcos\python.exe -m tf_ovcos.eval `
  --manifest data/manifests/ovcamo_te.jsonl `
  --predictions runs/groundingdino_sam/ovcamo_te/predictions.jsonl `
  --task class-aware `
  --out runs/groundingdino_sam/ovcamo_te/metrics.json
```

也就是说，第三方方法可以各用各的 repo，但输出必须回到统一的 `predictions.jsonl`。
