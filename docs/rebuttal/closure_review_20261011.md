# 20261011_closed 修复与证据审查入口

实验结果冻结为 `20261011_closed`。本次收尾提供可搬走、无需 GPU 即可核验的审查包；不启动新实验。后续若必须更正数值，应建立新版本并记录原因，不覆盖这份快照。

## 交付内容

下载：[审查包（约 17 MB）](artifacts/review_20261011_closed.tar.gz) · [SHA256](artifacts/review_20261011_closed.tar.gz.sha256)。

本地审查包：`runs/rebuttal/review_20261011_closed/`；压缩包：`runs/rebuttal/review_20261011_closed.tar.gz`，旁边的 `.sha256` 文件用于检查压缩包完整性。完整保留现有约 52 MB 证据快照，包括逐图哈希、配置、结果 CSV、验证记录和实际执行代码；原始预测 PNG、数据集及模型权重仍保存在实验工作区。

| 审查事项 | 包内入口 | 可检查的内容 |
| --- | --- | --- |
| 所有文件及版本 | `manifest.json`、`evidence/manifest.json` | 基础 Git 提交、相对路径、大小、SHA256；外层清单也覆盖原清单未列出的嵌套 manifest |
| 实际修复代码 | `changes.patch`、`code/src/tf_ovos/dino_geometry.py`、`code/src/tf_ovos/adapters/dinov2_sam.py` | RGB 与 mask 使用相同 resize/crop，再映射 token；相对公开基础版本的完整差异 |
| 全量 DINO 修正 | `code/scripts/repair_dino_geometry_from_cache.py`、`evidence/closure/dino_geometry/full/` | 12 项任务、24 行指标；每张旧预测重建一致后才采用修正输出 |
| FreeDA 标签修复 | `code/scripts/restore_freeda_coco_native.py`、`evidence/closure/freeda_label_corruption_audit.json`、`evidence/closure/freeda_full/` | 5,000 张原生 0–170 标签输出；旧非 255 像素无变化；28.7963% 复现 28.80% |
| FreeDA 来源和环境 | `evidence/closure/freeda_reproduction/`、`evidence/closure/code_snapshot/manifest.json` | 检索原型、索引、权重及环境版本；历史转换命令未找到，不把归因推断写成已证实执行链 |
| 效率与质量配对 | `code/scripts/measure_dino_geometry_e4.py`、`evidence/closure/dino_e4/` | 8 项任务、200/200 个预测配对一致；配置、同步、warmup 和计时范围 |
| 统一结果 | `evidence/fixed_gt/`、`evidence/diagnostics/` | 136 行曲线、34 个端点、56 行共同 GT、16 行 COCO 分组及更新后的原生质量和 Mean/Worst |
| 审稿意见与数据 | `report/reviewer_experiment_summary_20261009.md` | 每条意见下的新数据、解释和改稿事项 |
| 本机完整核验 | `local_report_validation.json` | 打包前原始来源文件及报告表格的核验记录 |

`code/` 基于清单中的 Git 提交展开，再覆盖快照保存的实际执行源码，并补入汇总与审查工具依赖。`evidence/closure/code_snapshot/` 保留未经改动的运行时来源。`changes.patch` 包含新文件及必要依赖，既有指标、标签协议等工作区修复也可能出现在差异中，不能全部解释为最后一次 DINO 修复。原生 FreeDA 外部源码保留于证据中的 `native_freeda/`，其许可证与归属按原项目处理。

## 无 GPU 核验

先将压缩包与校验文件下载到同一目录：

```bash
sha256sum -c review_20261011_closed.tar.gz.sha256
tar -xzf review_20261011_closed.tar.gz
cd review_20261011_closed
```

在解压后的包目录运行，Python 3.10 及以上即可，无需安装第三方包：

```bash
python verify.py .
```

核验会检查文件清单和全部哈希、固定 GT 端点与曲线的算术一致性、4 个局部上升步骤、共同 GT 的 `N=C+W+U`、COCO 的 4,894 张/8 方法范围、DINO 全量重建记录、200 张效率配对记录，以及报告中各张主要数据表与 CSV 是否一致。命令不会加载模型、启动队列或修改冻结产物。

这项核验验证交付完整性与已存证据的一致性；它不能替代拿到原始图像、GT、权重和逐图预测后的独立全量复现。证据中的绝对路径是历史来源标识，不影响包内核验，但运行历史推理脚本前需要恢复对应输入或调整路径。

## 修复回归测试

在具有记录中依赖版本的环境中，从包目录执行：

```bash
PYTHONPATH=code/src python -m pytest -q code/tests/test_dino_geometry.py code/tests/test_rebuttal_metrics.py
```

这 15 项测试覆盖空间对应和指标行为，不需要 GPU 或模型下载。完整版本见 `evidence/closure/code_snapshot/manifest.json`。FreeDA 和 DINO 使用不同环境，不能把两者的依赖表合并为一个已验证环境。

## 在原工作区复建审查包

```bash
python scripts/validate_geometry_closure_report.py
python scripts/build_closure_review.py --output /tmp/tf-ovos-closure-review
python scripts/verify_closure_review.py /tmp/tf-ovos-closure-review
```

构建器先验证冻结证据，再复制；目标已存在即拒绝覆盖。这些命令不会运行历史推理脚本。全量复现实验入口已随代码提供，但此次收尾无需再次执行。

## 结论与适用范围

修正后仍有 34/34 个最大候选端点下降，COCO 8/8 方法的 stuff mIoU 低于 thing。单 candidate seed、同图中的尺寸/频率差异、最终语义图诊断不能独立分离定位与命名因果等边界保持不变。E4 是既定抽样与定向配对范围，不是恢复全方法全量计划。未加入多 seed、新数据集或额外 dense 控制。

审查包现随仓库公开，包含修复代码、差异、轻量证据与核验入口。压缩包字节保持冻结时的版本，其内部说明中的“尚未公开”记录的是打包时状态；当前发布状态以本页为准。论文正文、LaTeX 和 rebuttal 草稿仍待按报告修改。
