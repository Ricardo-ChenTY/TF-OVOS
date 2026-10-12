"""Render verified closure values into the existing reviewer-by-reviewer report."""
import csv,json,re,hashlib,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[1];B=R/'runs/rebuttal/closure_20261011';U=B/'unified';P=R/'docs/rebuttal/reviewer_experiment_summary_20261009.md'
def read(p):return json.loads(Path(p).read_text())
def rows(p):return list(csv.DictReader(Path(p).open()))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
NAMES={'maskclip':'MaskCLIP','maskclip_attn':'MaskCLIP-Attn','maskclip_attn_slide':'MaskCLIP-Attn-Slide','sclip':'SCLIP','naclip':'NACLIP','resclip':'ResCLIP','scclip':'SC-CLIP','proxyclip':'ProxyCLIP','corrclip':'CorrCLIP','trident':'Trident','cass':'CASS','freeda':'FreeDA','sam_amg_clip':'SAM + CLIP','sam_amg_siglip':'SAM + SigLIP','dinov2_sam_clip':'DINOv2 + SAM + CLIP','dinov2_sam_siglip':'DINOv2 + SAM + SigLIP','cliptrase':'CLIPtrase †','san':'SAN（训练参考）'}
def main():
 v=read(U/'verification.json');e4=read(B/'dino_e4/verification.json');f=read(B/'freeda_full/verification.json');a=read(B/'freeda_label_corruption_audit.json')
 closed_utc=max([v['updated_utc']]+[read(p)['completed_utc'] for p in (B/'dino_e4').glob('*/*/result.json')])
 assert v['status']==e4['status']=='complete' and e4['paired_images']==e4['exact_matches']==200
 assert v['descending_endpoints']==34 and v['stuff_lower_methods']==8
 text=P.read_text();backup=B/'report_before_closure.md'
 if not backup.exists():backup.write_text(text)
 text=backup.read_text()
 def sub(old,new):
  nonlocal text
  assert old in text,old[:100];text=text.replace(old,new)
 def table(marker,header,data):
  nonlocal text
  pos=text.index(marker);start=text.index('| ',pos);end=text.find('\n\n',start)
  body='| '+' | '.join(header)+' |\n| '+' | '.join(['---']*len(header))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in data)
  text=text[:start]+body+text[end:]
 ends=rows(U/'fixed_gt/endpoints.csv');curves=rows(U/'fixed_gt/curves.csv');steps=rows(U/'fixed_gt/nonmonotonic_steps.csv');common=rows(U/'diagnostics/common_gt_counts.csv');coco=rows(U/'diagnostics/coco_thing_stuff_intersection.csv');comp=rows(U/'diagnostics/submitted_comparison.csv');changes=rows(U/'dino_geometry_changes.csv')

 proposal_counts=[r for r in common if r['method'] in ['sam_amg_clip','sam_amg_siglip','dinov2_sam_clip','dinov2_sam_siglip'] and float(r['iou_threshold'])==.5]
 unloc=[100*float(r['unlocalized_fraction']) for r in proposal_counts];wrong=[100*float(r['mcmr']) for r in proposal_counts]
 text=text.replace('约 53%–59%',f'约 {min(unloc):.1f}%–{max(unloc):.1f}%').replace('约 65%–82%',f'约 {min(wrong):.1f}%–{max(wrong):.1f}%')
 endidx={(r['method'],r['dataset']):r for r in ends}
 order=['maskclip','maskclip_attn','maskclip_attn_slide','sclip','naclip','resclip','scclip','proxyclip','corrclip','trident','cass','freeda','sam_amg_clip','sam_amg_siglip','dinov2_sam_clip','dinov2_sam_siglip','cliptrase']
 data=[]
 for method in order:
  rr=[endidx[(method,ds)] for ds in ['ade150','context59']];data.append([NAMES[method]]+[f'{float(r[k]):.4f}' for r in rr for k in ['base_miou_percent','expanded_miou_percent','drop_pp']])
 # Select the numeric table after O2 heading, not the reviewer's introductory prose.
 table('### O2：',['方法','ADE150：150 候选 %','ADE150：847 候选 %','下降 pp','Context59：59 候选 %','Context59：459 候选 %','下降 pp'],data)
 sub('上表 ADE150+847 候选仍用 ADE150 GT','上表 847 个总候选仍用 ADE150 GT')
 # O3 has one count table per target.
 for ds,label in [('context459','Context459'),('ade847','ADE847')]:
  candidates=[r for r in common if r['dataset']==ds and float(r['iou_threshold'])==.5]
  # Locate existing tables by their distinctive header; operate only within O3.
  section=text.index('### O3：');target=text.index('| 方法 |',section) if ds=='context459' else text.index('| 方法 |',text.index('| 方法 |',section)+10)
  end=text.find('\n\n',target)
  tab='| 方法 | N | C | W | U | Coverage % | MCMR % |\n| --- | --- | --- | --- | --- | --- | --- |\n'
  tab+='\n'.join('| '+' | '.join([NAMES[r['method']],str(r['gt_pairs']),str(r['localized_correct']),str(r['localized_wrong']),str(r['unlocalized']),f"{100*float(r['coverage']):.2f}",f"{100*float(r['mcmr']):.2f}"])+' |' for r in candidates)
  text=text[:target]+tab+text[end:]
 cocoorder=list(dict.fromkeys(r['method'] for r in coco));data=[]
 for method in cocoorder:
  t=float(next(r['miou_percent'] for r in coco if r['method']==method and r['group']=='thing'));s=float(next(r['miou_percent'] for r in coco if r['method']==method and r['group']=='stuff'));data.append([NAMES[method],f'{t:.4f}',f'{s:.4f}',f'{t-s:.4f}'])
 table('### O4：',['方法','Thing mIoU %','Stuff mIoU %','Thing−Stuff pp'],data)
 # Replace the four explicitly listed non-monotonic steps wherever their header starts.
 start=text.index('| 方法 | 目标 | 候选变化');end=text.find('\n\n',start)
 text=text[:start]+'| 方法 | 目标 | 相邻候选 | 增加 pp |\n| --- | --- | --- | --- |\n'+'\n'.join(f"| {NAMES[r['method']]} | {r['dataset']} | {r['from_candidates']}→{r['to_candidates']} | {float(r['increase_pp']):.4f} |" for r in steps)+text[end:]
 table('### 原生数据集重跑与固定 GT 控制分别引用',['目标','方法','投稿 mIoU %','新 mIoU %','新−旧 pp'],[[r['dataset'],NAMES[r['method']],f"{float(r['submitted_miou']):.2f}",f"{float(r['new_miou']):.4f}",f"{float(r['change_pp']):.4f}"] for r in comp])
 count=sum(abs(float(r['change_pp']))<=2 for r in comp);larger='；'.join(f"{NAMES[r['method']]} / {r['dataset']} 为 {float(r['change_pp']):+.4f} pp" for r in comp if abs(float(r['change_pp']))>2)
 sub('24 项中 23 项在 ±2 pp 内；DINOv2+SAM+SigLIP / VOC20 为 +2.8579 pp。',f'24 项中 {count} 项在 ±2 pp 内；{larger}。')
 sub('旧值为投稿数字，新值为本轮完成结果。','旧值为投稿数字，新值为本轮完成结果；两种 DINO 行全部采用空间对齐修正后的单 seed 全量结果。')
 sub('原先 40 项任务、共同 GT 诊断、COCO 同图对照及约定范围内的 E4 抽样与定向补测均已完成。**当前约定范围内已无待跑 GPU 实验，17 种方法的单 seed 固定 GT 控制已全部补齐。**',f'原先 40 项任务与固定 GT 队列完成后，本次又完成三项技术收尾：**FreeDA/COCO 全 5,000 张原生标签恢复、DINO 空间对应修正及 12 项全量对照、受影响指标与效率配对统一**。DINO 新配置追加 8 项效率抽样任务，200/200 张与修正后的全量结果逐像素一致。当前这三项均已核验通过，无待跑修复任务。')
 sub('**最终结果记录：2026-10-11 11:35 UTC 完成，11:35:54 UTC 汇总核验完成。**','**固定 GT 原队列记录：2026-10-11 11:35 UTC 完成，11:35:54 UTC 汇总核验完成；技术收尾完成时间见下文。**')
 sub('docs/rebuttal/evidence/20261011_final/','docs/rebuttal/evidence/20261011_closed/')
 sub('本次已补入 CASS 全部最终结果；早期 `20261011/` 快照保留为历史记录，不覆盖本次最终版本。','本次快照合入修正后的 DINO、FreeDA/COCO 与相应效率结果；`20261011/`、`20261011_final/` 保留为历史版本。')
 sub('本次最终快照重新核对了 353 项归档源文件哈希；端点、逐级曲线和完成范围相互一致。','本次快照记录新旧来源及 SHA256，并将历史 DINO 多 seed 结果与本次修正的 seed 0 主表区分；端点、逐级曲线和完成范围相互一致。')
 sub('18 行各 100 张效率抽样；400 张 SAM 原配方输出一致；在线 CorrCLIP 独立变体 13,554 张全量配对','18 行历史效率抽样；两种 SAM 原配方各 100 张及两种修正后 DINO 各 100 张输出配对一致；在线 CorrCLIP 独立变体 13,554 张全量配对')
 sub('约定范围内 GPU 补实验已完成；M1','约定范围内 GPU 补实验及本次三项技术收尾均已完成；M1')
 sub('上述 4 个局部上升步骤',f'上述 {len(steps)} 个局部上升步骤')
 # Keep historical timings explicitly versioned; replace the active native pairing table.
 sub('| DINOv2 + SAM + CLIP | 3.2745 | 4.2135 | 8.811 | 在线推理 |','| DINOv2 + SAM + CLIP（旧空间实现） | 3.2745 | 4.2135 | 8.811 | 历史 TF32-off；不配修正后质量 |')
 sub('| DINOv2 + SAM + SigLIP | 3.5448 | 4.7026 | 9.694 | 在线推理 |','| DINOv2 + SAM + SigLIP（旧空间实现） | 3.5448 | 4.7026 | 9.694 | 历史 TF32-off；不配修正后质量 |')
 data=[['SAM + CLIP','100','3.1400','8.504'],['SAM + SigLIP','100','3.4104','9.378']]+[[NAMES[r['method']]+'（已对齐）',str(r['images']),f"{r['mean_seconds']:.4f}",f"{r['peak_allocated_mib']/1024:.3f}"] for r in e4['methods']]
 table('#### Z3-b：',['方法','配对图像数','平均在线 秒/张','峰值分配 GiB'],data)
 sub('4 种 proposal × 4 目标，共 16 项，每种方法 100 张。恢复原 cuDNN TF32-on 设置后，**400/400 张预测与对应旧全量输出逐像素一致**。该一致性限于已测样本，不能写成重新跑过全验证集质量。','原 16 项配对核验为历史记录。本次保留两种 SAM 的原配方各 100 张，并对两种已修正 DINO 配置追加 2 × 4 项、共 200 张无视觉缓存计时；**200/200 张与空间对齐后的全量预测逐像素一致**。当前表的四行均使用 matmul TF32-off、cuDNN TF32-on；抽样配对一致性限于已测图像，DINO 的全量质量另由 12 项修正运行取得。')
 sub('配对表（本地 `e4_targeted/proposal_same_recipe/comparison.csv`）；400 张一致性与完成核验（本地 `e4_targeted/verification.json`）。','两种 SAM 的历史配对表见本地 `e4_targeted/proposal_same_recipe/comparison.csv`；当前 DINO 配对见 `closure/dino_e4/verification.json`。旧 DINO 3.1973/3.4605 秒不能继续绑定新的质量行。')
 sub('18/18 抽样行、20/20 定向任务均完成。','18/18 历史抽样行、20/20 原定向任务及本次 8/8 DINO 配对任务均完成。')
 sub('- COCO 已修正标签编号和评分重复偏移，中途标记为无效的旧结果不得引用。FreeDA 历史输出在修正后的全 5,000 张 COCO 上为 27.0244%，与投稿 28.80% 的来源差异尚未完全归因，不称为已复现投稿值；这与 O4 的 4,894 张交集分数不同。',f'- FreeDA/COCO 当前全量分数为 **{f["miou_percent"]:.4f}%**，来自恢复原生标签后的 5,000 张结果；历史 27.0244% 使用了有损导出的 PNG，已退出主分析。O4 仍是另外固定的 4,894 张同图交集，不能混用分数。')
 sub('- DINOv2+SAM 保留原 baseline 的特征裁剪/网格对应行为，无 proposal 覆盖的像素仍赋基础类 0。产物核验一致不代表已改进或验证该实现的全部空间对应，结论限于实际运行的 baseline。','- DINOv2+SAM 已修正 mask 与 RGB 的 resize/中心裁剪对应，主分析只引用修正后的 seed 0 结果；旧多 seed 产物仅作历史记录。DINO 仍使用中心视野特征作为 guidance，无 proposal 覆盖的像素仍赋基础类 0；本次没有改变这两项模型设计。')
 sub('实验队列和汇总服务均已正常完成退出；本次公开提交更新两份说明 Markdown，详细证据快照保存在本地，未修改论文 PDF、LaTeX 或原回复草稿。','上述三项技术收尾已完成并合入各审稿意见的数据表。本次公开提交更新两份说明 Markdown，修正代码、逐图预测及详细证据保存在本地；论文 PDF、LaTeX 和原回复草稿尚未修改。')
 sub('4 种已有 dense 输出重评分，4 种 proposal 为新推理并独立重评分。','3 种已有 dense 输出重评分、FreeDA 原生标签恢复后重新推理，以及 4 种 proposal（其中两种 DINO 已修正空间对应）并独立重评分。')

 compact=rows(U/'diagnostics/compact_mean_worst.csv')
 merged='**本次四种 proposal 的统一版本摘要（新数据）**：DINO 行已采用空间修正结果，四个分数均为对应原生 compact 目标，不使用固定 GT 扩张端点。\n\n'
 merged+='| 方法 | VOC20 % | Context59 % | ADE150 % | COCO171 % | Mean % | Worst % / 目标 |\n| --- | --- | --- | --- | --- | --- | --- |\n'
 merged+='\n'.join('| '+' | '.join([NAMES[r['method']]]+[f"{float(r[k]):.4f}" for k in ['voc20','context59','ade150','coco_stuff171','mean_miou_percent']]+[f"{float(r['worst_miou_percent']):.4f} / {r['worst_dataset']}"] )+' |' for r in compact)+'\n\n'
 merged+='本地 `diagnostics/compact_mean_worst.csv` 可用于 O5/Z2；只覆盖这四种方法的当前完整原生结果，不据此把未更新方法拼成统一排名。\n\n'
 text=text.replace('**改稿说明**：接受合并建议',merged+'**改稿说明**：接受合并建议')
 # Detailed, reviewable repair explanation next to the metrics version section.
 fullbase=[r for r in changes if int(r['candidates'])=={'ade150':150,'context59':59,'context459':459,'ade847':847,'voc20':20,'coco_stuff171':171}[r['dataset']]]
 repair='### 三项技术收尾：修正、全量验证和版本统一\n\n'
 repair+=f'**完成核验：{closed_utc}；效率配对见本地 `closure/dino_e4/verification.json`。** 本次仅修复核验失败的分支，不增加候选 seed 或扩大 dense 控制范围。\n\n'
 repair+=f'**FreeDA / COCO 可复现性。** 找到 2026-05-18 原生评估和 2026-05-23 导出运行的 28.80 日志。旧 PNG 并非模型原生拒识输出：模型为 171 类、`with_bg=False`，十个合法类 ID（0、11、24、26、40、60、61、62、73、80）在归档中被抹为 255。全 5,000 张重跑后，旧非 255 像素有 **{a["full_changed_non255_pixels"]:,}** 个不一致；原 255 的 **{a["full_old_255_pixels"]:,}** 个像素恢复为合法标签。损坏标签的类别集合与混淆 COCO category ID 和 PNG category ID−1 产生的空洞完全对应，但未找到历史破坏性转换命令，故具体转换步骤仍属于有依据的归因推断。新输出直接保存 0–170，无偏移、LUT 或 GT 掩码；读回全部 PNG 独立重评分为 **{f["miou_percent"]:.4f}%**。'+('按两位小数复现投稿 **28.80%**。' if f['reproduces_submitted_to_two_decimals'] else '未在两位小数复现投稿值，须使用本次实测值。')+'旧 27.0244% 不再作为当前方法性能。配置、代码、权重/索引、图像、GT 与逐图输出的哈希已存档。\n\n'
 repair+='**DINOv2 + SAM 空间对应。** 旧流程将 RGB 缩放后中心裁剪，却把整张 SAM mask 直接缩到 token 网格，导致 mask 选取了不对应原图位置的 DINO 特征。现让 mask 经过与 RGB 相同的空间 resize/crop，再用 nearest 映射至 token 网格；裁剪外 mask 不选 token。横图、竖图、恒等变换及裁剪外区域测试通过，连同指标测试共 15 项。复用原 SAM 掩码、命名特征和候选，仅重算 DINO guidance；每张图先以旧 guidance 重建原输出并逐像素核对，再生成修正预测。两种命名方法 × 六个原生目标共 12 项全量工作、24 行指标全部完成，含两套固定 GT 的四档候选。修正不会普遍提高分数，以下如实列出影响。\n\n'
 repair+='| 方法 | 目标 | 修正前 mIoU % | 修正后 mIoU % | 变化 pp |\n| --- | --- | --- | --- | --- |\n'
 repair+='\n'.join(f"| {NAMES[r['method']]} | {r['dataset']} | {float(r['old_full_miou_percent']):.4f} | {float(r['corrected_miou_percent']):.4f} | {float(r['change_pp']):+.4f} |" for r in fullbase)+'\n\n'
 repair+=f'**统一指标与证据。** O2/Z1 的 136 行固定 GT、O3 的 56 行共同 GT 计数、O4 的 16 行同图分组、24 行原生新旧对照均已统一到当前版本，另补齐修正 DINO 的 E4 配对。固定 GT 仍为 **34/34 端点下降**，存在 **{len(steps)} 个局部上升步骤**；COCO 仍为 **8/8 方法 stuff mIoU 低于 thing**。FreeDA 与 DINO 的 COCO 都重新从落盘 PNG 核对混淆矩阵和 4,894 张交集。旧结果保留以供追溯，不能与新行混合平均；本地 `closure/active_result_sources.json` 指明当前产物。\n\n'
 text=text.replace('### 仍须保留的解释边界',repair+'### 仍须保留的解释边界')
 P.write_text(text)
 readme=R/'docs/rebuttal/README.md'
 readme.write_text('# Rebuttal 材料入口\n\n请使用[按审稿意见逐条整理的说明](reviewer_experiment_summary_20261009.md)：每条意见下列出新数据、解释和改稿事项。\n\n**固定 GT 的 17 种方法、136 行曲线全部完成；本次三项技术收尾也已完成。** FreeDA/COCO 恢复原生标签并对全 5,000 张重新评分；DINO 修正空间对应、完成 12 项全量对照，并补测 200 张与新质量对应的效率；相关表格已统一替换。34/34 固定 GT 最大候选端点仍下降，COCO 的 8 种方法仍为 stuff 低于 thing。\n\n主分析统一 candidate seed=0。未增加多 seed 队列；当前修复范围内无待跑实验。指标文献、正文图表和逐条回复仍需据此修改。\n\n本次公开推送仅更新两份说明文档。当前本地证据快照为 `docs/rebuttal/evidence/20261011_closed/`，含修正表格、配置、来源和 SHA256；`20261011_final/` 与 `20261011/` 保留作历史版本。原审稿转录位于 `docs/rebuttal/reviewer_comments.txt`。详细证据及代码不随本次文档提交公开。\n')
 print('REPORT RENDERED',len(text.splitlines()),flush=True)
if __name__=='__main__':main()
