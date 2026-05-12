# TF-OVCOS 并行运行设计

## 结论

可以设计成并行跑，而且应该一开始就支持：

1. 多方法并行
2. 多数据集并行
3. 单方法按 image shard 并行
4. 断点续跑和 shard 合并

但 E4 efficiency 指标要单独隔离跑。并行模式适合产出预测和主表指标，不适合直接记录公平的单方法耗时和峰值显存。

## 推荐并行层级

### 1. 多 GPU：方法/数据集级并行

这是最干净的并行方式。

例子：

```text
GPU 0: groundingdino_sam on OVCamo-TE
GPU 1: sam_amg_clip on OVCamo-TE
GPU 2: sclip on OVCamo-TE
GPU 3: proxyclip on OVCamo-TE
```

优点：

- 每个 worker 独占一张 GPU，OOM 风险低
- 运行日志清楚
- 方法之间互不影响
- 结果容易复现

### 2. 多 GPU：同一方法按图片 shard 并行

当只有一个方法但数据集很大时，把 manifest 切成多个 shard：

```text
OVCamo-TE shard 0 -> GPU 0
OVCamo-TE shard 1 -> GPU 1
OVCamo-TE shard 2 -> GPU 2
OVCamo-TE shard 3 -> GPU 3
```

每个 shard 输出：

```text
runs/<method>/<dataset>/shards/part-000/predictions.jsonl
runs/<method>/<dataset>/shards/part-001/predictions.jsonl
```

最后按 `image_id` 合并成：

```text
runs/<method>/<dataset>/predictions.jsonl
```

### 3. 单 GPU：多个轻量 worker 共享一张卡

这个可以做，但要保守。

A100 80G 上可以同时跑多个轻量 worker，例如：

- SAM ViT-B + CLIP：4-8 个 worker 可能可行
- GroundingDINO + SAM ViT-B：2-4 个 worker 更稳
- DINOv2 large / SAM2 large / diffusion：建议 1 个 worker

5090 32G 上建议：

- 轻量方法：2-4 个 worker
- detector+SAM：1-2 个 worker
- 大模型/扩散：1 个 worker

不要只看模型静态显存。真实峰值还包括：

- 输入分辨率
- SAM image embedding
- AMG 候选 mask 数量
- CLIP/SigLIP crop batch
- PyTorch CUDA allocator 缓存
- 第三方 repo 中没有及时释放的 tensor

## 调度器设计

建议给每个方法配置一个显存预算：

```yaml
methods:
  groundingdino_sam_vit_b:
    gpu_mem_gb: 14
    max_workers_per_gpu: 2
  sam_amg_clip_vit_b:
    gpu_mem_gb: 8
    max_workers_per_gpu: 4
  sclip:
    gpu_mem_gb: 10
    max_workers_per_gpu: 3
  dinov2_sam_siglip:
    gpu_mem_gb: 24
    max_workers_per_gpu: 1
  freeda:
    gpu_mem_gb: 40
    max_workers_per_gpu: 1
```

调度器只做资源分配，不改方法内部逻辑：

1. 读 benchmark jobs
2. 按 GPU 空闲显存和方法预算排队
3. 启动 worker
4. worker 写 shard output
5. 全部完成后 merge
6. 调 evaluator

## 输出格式

每个 worker 必须只写自己的 shard 目录，避免并发写同一个 JSONL：

```text
runs/<method>/<dataset>/shards/part-000/predictions.jsonl
runs/<method>/<dataset>/shards/part-000/runtime.json
runs/<method>/<dataset>/shards/part-000/log.txt
```

合并阶段做检查：

- 是否每个 image_id 都有预测
- 是否存在重复 image_id
- mask 文件是否存在
- label 是否在指定 vocab 内

## 断点续跑

worker 启动时先检查 shard output：

- 如果 `predictions.jsonl` 完整，跳过
- 如果不完整，重新跑该 shard
- 不在单个 JSONL 里边跑边 append 到总文件

这样某个方法中途 OOM 或断电，只需要重跑失败 shard。

## E4 注意事项

并行跑出来的吞吐量可以记录，但不能直接作为 E4 单方法成本。

E4 应该有两套数字：

1. `isolated`: 单 worker、单方法、固定 GPU，记录公平 per-image time 和 peak memory
2. `throughput`: 多 worker 并行模式，记录整批 benchmark wall-clock time

论文主表里的 E4 更适合用 `isolated`。

## 推荐实现顺序

1. 先实现 manifest shard 切分和 predictions merge
2. 再实现 `run_method` 单 worker
3. 再实现本机多 GPU 调度
4. 最后实现单 GPU 多 worker 的显存预算调度

这样即使没有大集群，也能先在一张卡上验证完整流程。
