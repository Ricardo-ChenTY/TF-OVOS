# GPU 前可完成事项

这份文档记录不依赖 GPU 就能先完成的工作。目标是让新机器上 `git clone`
以后，先确认 benchmark harness 能跑，再补 CUDA/PyTorch、权重和真实 adapter。

## 已经在仓库里准备好的部分

- conda 环境入口：`environment.yml`
- 一键 setup：`scripts/setup_conda.sh`
- workspace 目录准备：`scripts/prepare_workspace.sh`
- toy smoke test：`scripts/smoke_test.py`
- readiness 检查：`python -m tf_ovcos.check_ready`
- adapter registry：`python -m tf_ovcos.run_method --list-methods`
- 方法配置：`configs/methods/*.yaml`
- 本地路径示例：`configs/local_paths.example.yaml`

## 现在可以继续补的内容

1. 补官方 vocab：

```text
configs/vocab/ovcamo_75.txt
configs/vocab/ovcamo_61_unseen.txt
```

每行一个类别名。`ovcamo_75.txt` 应该有 75 行，`ovcamo_61_unseen.txt`
应该有 61 行。

2. 如果本机已有数据，生成 manifests：

```bash
python -m tf_ovcos.make_manifest \
  --image-dir data/raw/OVCamo/TE/images \
  --mask-dir data/raw/OVCamo/TE/masks \
  --label-source data/raw/OVCamo/sample_info.json \
  --out data/manifests/ovcamo_te.jsonl \
  --require-labels
```

3. 检查准备状态：

```bash
python -m tf_ovcos.check_ready
```

正式搬到 GPU 前可以用严格模式：

```bash
python -m tf_ovcos.check_ready --strict
```

## 需要 GPU/server 后完成的内容

- 安装与 server CUDA 匹配的 PyTorch。
- 拉第三方 repo，例如 Grounded-Segment-Anything、segment-anything、SCLIP、ProxyCLIP。
- 下载模型权重到 `weights/`。
- 实现并验证真实 adapter：优先 `groundingdino_sam`，然后 `sam_amg_clip`。
- 跑 20 张图的小规模输出检查，再跑完整 shard。
