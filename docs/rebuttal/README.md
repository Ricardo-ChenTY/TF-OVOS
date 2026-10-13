# Rebuttal 材料入口

请使用[按审稿意见逐条整理的说明](reviewer_experiment_summary_20261009.md)：每条意见下列出新数据、解释和改稿事项。

**固定 GT 的 17 种方法、136 行曲线全部完成；本次三项技术收尾也已完成。** FreeDA/COCO 恢复原生标签并对全 5,000 张重新评分；DINO 修正空间对应、完成 12 项全量对照，并补测 200 张与新质量对应的效率；相关表格已统一替换。34/34 固定 GT 最大候选端点仍下降，COCO 的 8 种方法仍为 stuff 低于 thing。

主分析统一 candidate seed=0。未增加多 seed 队列；当前修复范围内无待跑实验。指标文献、正文图表和逐条回复仍需据此修改。

修复代码及轻量证据随[审查包](artifacts/review_20261011_closed.tar.gz)一起公开，附[SHA256 校验文件](artifacts/review_20261011_closed.tar.gz.sha256)。当前本地证据快照为 `docs/rebuttal/evidence/20261011_closed/`，含修正表格、配置、来源和 SHA256；`20261011_final/` 与 `20261011/` 保留作历史版本。原审稿转录位于 `docs/rebuttal/reviewer_comments.txt`。原始预测 PNG、数据集和模型权重保留在实验工作区。

修复代码与证据现已整理为可下载的审查包，使用方法和范围见[修复与证据审查入口](closure_review_20261011.md)。包内提供无需 GPU 的完整性、表格一致性核验及回归测试入口。结果冻结为 `20261011_closed`，后续变更另建版本。

**2026-10-12 补充发布：** [回复材料直达索引](reply_materials_20261012.md)提供可浏览的完整计时、固定 GT、共同 GT、同图 COCO、修复核验和图稿检查材料；实际运行修复代码也已同步至仓库顶层，不再需要先解压才能阅读。

**2026-10-12 关机交接补充：** [16 案例离线素材与 MaskCLIP 缺项补齐](server_handoff_20261012.md)。三个 MaskCLIP 变体已完成保存预测的全量 CPU 重评分，补充表取代旧 n.a.；原冻结包不改写。完整预测、数据集和权重仍需保留原硬盘。

**2026-10-13 配置来源核查：** [Trident SAM、官方预设及回复措辞修正](release_preset_audit_20261013.md)。Trident 的 SAM ViT-B 与所核对官方提交一致；必须补列 SAM，并区分官方按目标预设、项目扩展及 ResCLIP VOC 滑窗偏离。按最新决定不启动统一超参重跑，E3 收窄为已声明配置下的跨数据集表现。

**2026-10-13 ResCLIP 定向修正：** [VOC 官方滑窗设置恢复及全量成对证据](resclip_voc_release_fix_20261013.md)。VOC由78.51更新为82.13；同环境只改slide_crop提升3.621861个百分点。更新Table 1与Mean/Worst，旧E4抽样行须标为历史配置；大词表与fixed-GT结果未因本次修正改变。
