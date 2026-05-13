# 常用命令

下面的命令默认在项目根目录运行。

## 1. 检查环境

Windows 本机当前环境：

```powershell
.\.conda\tf-ovcos\python.exe -m pytest tests
```

Linux/A100 机器上如果环境名是 `tf-ovcos`：

```bash
conda activate tf-ovcos
python -m pytest tests
```

## 2. 生成 Manifest

通用目录格式：

```bash
python -m tf_ovcos.make_manifest \
  --image-dir data/raw/OVCamo/TE/images \
  --mask-dir data/raw/OVCamo/TE/masks \
  --label-source data/raw/OVCamo/labels.json \
  --out data/manifests/ovcamo_te.jsonl \
  --require-labels
```

如果外部 E3 数据集没有类别：

```bash
python -m tf_ovcos.make_manifest \
  --image-dir data/raw/NC4K/images \
  --mask-dir data/raw/NC4K/masks \
  --out data/manifests/nc4k.jsonl
```

`--label-source` 支持：

- `.json`: `{ "image_id": "label" }` 或 list of objects
- `.jsonl`: 每行包含 `image_id` 和 `label`
- `.csv`: 表头包含 `image_id,label`
- `.txt`: 每行 `image_id label`

如果 OVCamo 官方 json 字段不是 `image_id/label`，用：

```bash
--id-field <字段名> --label-field <字段名>
```

## 3. 切 Shard

```bash
python -m tf_ovcos.make_shards \
  --manifest data/manifests/ovcamo_te.jsonl \
  --num-shards 8 \
  --out-dir data/manifests/shards/ovcamo_te
```

输出：

```text
data/manifests/shards/ovcamo_te/part-000.jsonl
...
data/manifests/shards/ovcamo_te/part-007.jsonl
```

## 4. 跑一个 Method Adapter

先用 debug adapter 检查 pipeline：

```bash
python -m tf_ovcos.run_method \
  --method debug_copy_gt \
  --manifest data/manifests/shards/ovcamo_te/part-000.jsonl \
  --vocab configs/vocab/ovcamo_61_unseen.txt \
  --out runs/debug_copy_gt/ovcamo_te/shards/part-000
```

列出当前可用 adapter：

```bash
python -m tf_ovcos.run_method --list-methods
```

当前只有：

- `debug_copy_gt`: 复制 GT mask，检查上限和路径
- `debug_empty`: 输出空 mask，检查失败路径

真实方法后面会注册为：

- `groundingdino_sam`
- `sam_amg_clip`
- `sclip`
- `naclip`
- `proxyclip`

## 5. 合并 Predictions

```bash
python -m tf_ovcos.merge_predictions \
  --manifest data/manifests/ovcamo_te.jsonl \
  --shards-dir runs/debug_copy_gt/ovcamo_te/shards \
  --out runs/debug_copy_gt/ovcamo_te/predictions.jsonl \
  --vocab configs/vocab/ovcamo_61_unseen.txt
```

合并时会检查：

- 是否有重复 `image_id`
- 是否缺预测
- 是否有 manifest 之外的预测
- label 是否在 vocab 内
- mask 路径是否按合并后的目录重写

## 6. 评测

E1/E2 class-aware：

```bash
python -m tf_ovcos.eval \
  --manifest data/manifests/ovcamo_te.jsonl \
  --predictions runs/debug_copy_gt/ovcamo_te/predictions.jsonl \
  --task class-aware \
  --out runs/debug_copy_gt/ovcamo_te/metrics.json
```

E3 external mask-only：

```bash
python -m tf_ovcos.eval \
  --manifest data/manifests/nc4k.jsonl \
  --predictions runs/sam_amg_clip/nc4k/predictions.jsonl \
  --task mask-only \
  --out runs/sam_amg_clip/nc4k/metrics.json
```

## 7. 一键 Pipeline

单个方法、单个数据集：

```bash
python -m tf_ovcos.run_benchmark \
  --method debug_copy_gt \
  --dataset ovcamo_te \
  --limit 20 \
  --num-shards 8 \
  --skip-existing
```

单个方法跑 `configs/benchmark.yaml` 里的所有数据集：

```bash
python -m tf_ovcos.run_benchmark \
  --method debug_copy_gt \
  --num-shards 8 \
  --skip-existing
```

小样本 smoke test 通过后，去掉 `--limit` 就是全量。

汇总 E1/E2/E3/E4 风格结果表：

```bash
python -m tf_ovcos.summarize_results \
  --method debug_copy_gt \
  --out-dir runs/tables
```

每个 shard 会写：

```text
runs/<method>/<dataset>/shards/part-000/predictions.jsonl
runs/<method>/<dataset>/shards/part-000/runtime.json
```
