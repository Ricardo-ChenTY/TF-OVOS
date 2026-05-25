# TF-OVOS Proposal Completion Plan

Goal: finish every method row in the proposal main tables first, with a
reproducible route from method adapters / official-repo runs to E1-E4 tables
and diagnostic probes. Appendix hard-domain transfer runs are deferred until
the main tables are filled or every blocker is documented.

The proposal uses a two-tier rule:

- Standard tier: E1-E3 are ranked by mIoU on semantic targets.
- Exploratory tier: supporting diagnostics are recorded in parallel, but do not
  replace the leaderboard metric.

Strict TF rows must use frozen public models only: no target-dataset training,
no prompt tuning on validation data, no adapter training, no closed-source API
VLM calls, and no inference-time generative MLLM calls.

Full-scope override:

- The rows in the proposal tables are main-scope rows, not optional rows.
- E1, E2, E3, E4, and diagnostics should be filled for every row whenever the
  required code/artifacts exist.
- Rows that cannot be run must have a concrete blocker and source note.
- Appendix transfer starts only after E1-E4 and diagnostics are complete.

Main-scope rows:

| Family | Rows |
|--------|------|
| CLIP-only / attention-edited | MaskCLIP, CLIP-DIY, SCLIP, NACLIP, CLIPtrase, SC-CLIP, ResCLIP |
| CLIP + VFM | ProxyCLIP, CorrCLIP, Trident, CASS |
| Diffusion / reference-based | OVDiff, FreeDA |
| OV detection + SAM | GroundingDINO + SAM, GroundingDINO + SAM2 |
| Class-agnostic proposal + VLM naming | SAM-AMG + CLIP, SAM-AMG + SigLIP, DINOv2 + SAM + CLIP, DINOv2 + SAM + SigLIP |
| Compact training-based references | OVSeg, SAN, ODISE |

Main-table run waves:

1. Freeze current official E1 and finish official E2 for ProxyCLIP/CorrCLIP.
2. Implement/run proposal and detector rows: SAM-AMG + CLIP, SAM-AMG + SigLIP, GroundingDINO + SAM, GroundingDINO + SAM2, DINOv2 + SAM + CLIP, DINOv2 + SAM + SigLIP.
3. Implement/run missing CLIP/VFM rows: MaskCLIP, CLIP-DIY, CLIPtrase, SC-CLIP, Trident, CASS.
4. Implement/run diffusion rows: OVDiff, FreeDA, with offline synthesis/prototype time separated from test cost.
5. Fill trained references: OVSeg, SAN, ODISE, using official inference when practical or exact-protocol published values.
6. Run diagnostic probes: GT-region naming, GT-text localization, proposal recall, and MCC consistency.
7. Run isolated E4 timing on the same GPU after method settings are frozen.
8. Export E1/E2/E3/E4/diagnostic tables, then decide appendix scope.

---

## Current Snapshot

Already usable:

- Benchmark harness, manifests, sharding, evaluator, and runtime JSON path.
- MaskCLIP-lite adapter results:
  - E1: `voc20`, `context59`, `ade20k150`, `coco_stuff171`
  - E2: `context459`, `ade20k847`
- Official MMSeg-style E1 datafix results:
  - Complete: SCLIP, NACLIP, ResCLIP, ProxyCLIP on VOC20 / Context59 / ADE20K / COCO-Stuff171.
  - Complete: CorrCLIP on VOC20 / Context59 / ADE20K / COCO-Stuff171.
- Official E2 large-vocabulary queue:
  - Running in tmux session `official_e2_queue`.
  - Order: ProxyCLIP Context459, ProxyCLIP ADE847, CorrCLIP Context459, CorrCLIP ADE847.
  - ProxyCLIP uses its official E2 class files/config conventions.
  - CorrCLIP uses official CorrCLIP dataset classes and region masks, with the
    same official E2 class files copied from ProxyCLIP because CorrCLIP ships the
    dataset classes but not E2 cfg files.
- Analysis watcher writes:
  - `runs/analysis/official_best_metrics.csv`
  - `runs/analysis/official_log_metrics.csv`
  - `runs/analysis/proposal_novelty_metrics.csv`
  - `runs/analysis/candidate_signals.csv`

Important interpretation:

- Official-repo E1 numbers can fill proposal E1 standard rows immediately.
- Adapter-native rows are still needed for proposal methods that are not covered
  by official MMSeg logs and for diagnostics requiring unified prediction JSONL,
  top-k scores, proposal masks, or per-stage runtime.

---

## Completion Definition

The proposal is "run through" when the following artifacts exist.

1. E1 standard table:
   - VOC20, Context59, ADE20K-150, COCO-Stuff171 mIoU for every kept method row.

2. E2 vocabulary robustness table:
   - Context59 vs Context459 and ADE150 vs ADE847 mIoU.
   - Delta_vocab for every method where compact/large pairs are complete.
   - MCMR@0.5 and top mismatch pairs for methods with region or prediction artifacts.

3. E3 generalization table:
   - Target mIoU on the same standard targets.
   - Rank stability / worst-target / worst-class readouts from completed E1-style runs.
   - Hard-domain appendix runs only after the main semantic tables are stable.

4. E4 efficiency table:
   - seconds/image, peak GPU memory, model calls, and per-stage runtime where available.
   - Official logs may seed partial values, but final E4 should use isolated reruns.

5. Paper-ready CSV/Markdown outputs:
   - `runs/tables/e1_standard.csv`
   - `runs/tables/e2_vocab.csv`
   - `runs/tables/e3_generalization.csv`
   - `runs/tables/e4_efficiency.csv`
   - `runs/tables/exploratory_diagnostics.csv`

---

## Priority Order

### P0 - Freeze Current Official E1

Purpose: finish the standard table backbone before implementing more adapters.

Tasks:

1. Keep the completed CorrCLIP datafix logs as canonical:
   - Context59 uses the missing-region-mask fallback for one official archive miss.
   - COCO-Stuff171 uses the official labelTrainIds convention.
2. Refresh analysis:
   ```bash
   /data/tianyi/conda_envs/tf-ovos/bin/python scripts/analyze_official_results.py
   ```
3. Validate that `official_best_metrics.csv` has 20 rows:
   - methods: SCLIP, NACLIP, ResCLIP, ProxyCLIP, CorrCLIP
   - datasets: VOC20, Context59, ADE20K, COCO-Stuff171
4. Mark old bad COCO logs as invalid and keep datafix rows as canonical.

Exit criteria:

- E1 official standard table has 5 x 4 complete rows.
- `proposal_novelty_metrics.csv` has artifact-completeness rows for every saved prediction directory that exists.

### P0.5 - Official E2 Large-Vocab Overnight Pass

Purpose: fill the proposal E2 backbone while staying inside official MMSeg
method implementations.

Running command:

```bash
tmux new-session -d -s official_e2_queue \
  'cd /data/tianyi/TF-OVCOS && bash scripts/run_official_e2_queue.sh >> runs/logs/official_e2_queue_driver.log 2>&1'
```

Tasks:

1. Generate local official-style E2 data trees:
   - Context459: VOC2010 `JPEGImages`, `annotations_detectron2/pc459_val`, and `ImageSets/SegmentationContext/val.txt`.
   - ADE847: ADE validation image tree, `annotations_detectron2/validation`, and `validation.txt`.
2. Preserve official E2 prompt class files:
   - `cls_context459.txt`
   - `cls_ade20k847.txt`
3. Run only methods with official E2 support in code:
   - ProxyCLIP: official E2 cfg pattern.
   - CorrCLIP: official dataset classes + official region masks; generated cfg only wires paths and class files.
4. Refresh analysis after the queue finishes:
   ```bash
   /data/tianyi/conda_envs/tf-ovos/bin/python scripts/analyze_official_results.py
   ```

Exit criteria:

- `official_best_metrics.csv` includes ProxyCLIP/CorrCLIP rows for `context459` and `ade847`.
- Analysis marks these rows as `phase2_e2`.
- SCLIP/NACLIP/ResCLIP E2 are not added until their official repos have verified compatible E2 dataset support.

### P1 - Full Main-Table Row Inventory

Purpose: fill every row explicitly listed in the proposal main tables.

Strict-TF rows:

| Family | Rows | Source path |
|--------|------|-------------|
| CLIP-only / attention-edited | MaskCLIP, CLIP-DIY, SCLIP, NACLIP, CLIPtrase, SC-CLIP, ResCLIP | official repo when available; otherwise adapter |
| CLIP + VFM | ProxyCLIP, CorrCLIP, Trident, CASS | official repo when available; otherwise adapter |
| Diffusion / reference-based | OVDiff, FreeDA | official repo or adapter; offline stage noted in E4 |
| OV detection + SAM | GroundingDINO + SAM, GroundingDINO + SAM2 | adapter |
| Proposal + VLM naming | SAM-AMG + CLIP, SAM-AMG + SigLIP, DINOv2 + SAM + CLIP, DINOv2 + SAM + SigLIP | adapter |

Non-strict reference rows:

- OVSeg
- SAN
- ODISE

Exit criteria:

- Every main-table row has E1/E2/E3/E4/diagnostic slots.
- Every row has an implementation route or a documented blocker.
- No row above is treated as appendix-only.

### P2 - Adapter Contract Pass

Purpose: make remaining methods run through one benchmark interface.

For each adapter, implement:

1. `src/tf_ovcos/adapters/<method>.py`
2. registration in `tf_ovcos.run_method`
3. one smoke command on 20 images
4. full benchmark command
5. runtime fields:
   - wall seconds
   - seconds/image
   - peak GPU memory if sampled
   - model-call counts where natural
   - proposal count / selected mask count for proposal methods

Required output format:

```json
{"image_id": "xxx", "mask_path": "pred_masks/xxx.png", "label": "class name", "score": 0.73}
```

Additional optional artifacts for exploratory metrics:

- `proposal_masks/`
- `topk_labels.jsonl`
- `score_maps/` or compact per-region score arrays
- `runtime.json`
- `stage_runtime.json`

Smoke test rule:

```bash
python -m tf_ovcos.run_benchmark \
  --method <method> \
  --dataset voc20_val \
  --limit 20 \
  --num-shards 1 \
  --skip-existing
```

Full E1/E2 command:

```bash
python -m tf_ovcos.run_benchmark \
  --method <method> \
  --dataset voc20_val \
  --dataset context59_val \
  --dataset ade20k150_val \
  --dataset coco_stuff171_val \
  --dataset context459_val \
  --dataset ade20k847_val \
  --num-shards 4 \
  --skip-existing
```

Exit criteria:

- Every implemented adapter can run `--limit 20` and produce valid masks,
  labels, metrics, and runtime JSON.

### P3 - Run Proposal / Detector Rows

Purpose: fill the rows currently empty in the proposal table and generate the
diagnostic artifacts that dense official logs cannot provide.

Run order:

1. SAM-AMG + CLIP
   - simplest class-agnostic proposal + CLIP naming row.
   - produces proposal masks, top-k naming artifacts, and proposal-recall diagnostics.

2. SAM-AMG + SigLIP
   - same proposal backbone, stronger naming model.
   - compares directly against CLIP naming.

3. GroundingDINO + SAM
   - detector-driven row.
   - prompt with all vocabulary labels, use fixed thresholds, then SAM masks.

4. GroundingDINO + SAM2
   - detector-driven row with SAM2 masks.
   - compares SAM vs SAM2 mask refinement.

5. DINOv2 + SAM + CLIP
   - DINOv2-guided proposal/ranking plus CLIP naming.

6. DINOv2 + SAM + SigLIP
   - stronger proposal/naming row.
   - use DINOv2 features for proposal guidance or ranking, then SigLIP naming.

Exit criteria:

- E1/E2/E3 complete for all six proposal/detector rows.
- MCMR@0.5, proposal Recall@0.5/0.7, GT-region naming, and top mismatch pairs are computed where artifacts exist.

### P4 - Add Missing Dense / VFM / Diffusion Rows

Purpose: complete important proposal rows not covered by current official logs.

Run order:

1. MaskCLIP attention-surgery adapter
   - replaces MaskCLIP-lite baseline with the actual method row.

2. CLIP-DIY
   - CLIP patch inference plus unsupervised localization prior.

3. CLIPtrase
   - CLIP self-attention / dense inference modification.

4. SC-CLIP
   - verify exact official method/source before running.

5. Trident adapter or official-repo run
   - CLIP + DINO + SAM high-resolution training-free pipeline.

6. CASS
   - CLIP with VFM spectral object-context distillation at inference.

7. OVDiff
   - diffusion support/prototype stage; no target training.

8. FreeDA
   - offline diffusion-augmented prototype generation.

Exit criteria:

- E1/E2/E3 rows exist for every dense, VFM, and diffusion method in the table.
- E4 records offline synthesis/prototype time separately from test cost.
- Diagnostics are computed when a method emits masks, regions, top-k labels, or proposals.

### P5 - Trained References

Purpose: provide context rows without confusing them with strict TF methods.

Rows:

- OVSeg
- SAN
- ODISE

Allowed sources:

1. official inference with released weights, preferred when setup is practical.
2. published numbers, only when the exact target dataset/protocol matches.

Reporting rule:

- Mark `strict_tf = No`.
- Keep source/training data visible.
- Do not use trained references to define the main TF ranking.

Exit criteria:

- E1/E3/E4 table has compact trained-reference context rows or documented missing protocol.

### P6 - Main-Table Diagnostics

Purpose: fill the failure-source diagnostic probes before appendix.

Diagnostics:

1. GT-region naming diagnostic
   - classify ground-truth regions with CLIP/SigLIP.
   - localization is removed.
   - report Top-1 / Top-5 naming accuracy.

2. GT-text localization diagnostic
   - segment or localize using the ground-truth class prompt.
   - recognition is removed.
   - report IoU / BIoU / mask metrics.

3. Proposal-recall diagnostic
   - choose the proposal with highest GT IoU from SAM-AMG or DINOv2+SAM.
   - naming is removed.
   - report oracle IoU / BIoU.

4. MCC consistency diagnostic
   - compare the same proposal+naming pipeline with and without post-hoc mask-category consistency reranking.
   - report pair-consistency gain and representative failure cases.

Exit criteria:

- `runs/tables/exploratory_diagnostics.csv` has rows for every diagnostic/method combination with required artifacts.
- All diagnostic rows are marked `diagnostic_only`.

### P7 - Final E4 Reruns

Purpose: produce fair cost numbers after method settings are frozen.

Run each kept method on the same GPU with:

- fixed input resolution / resize policy
- fixed batch size
- no concurrent GPU jobs
- peak memory sampler
- stage timers where possible

Collect:

- inference seconds/image
- peak GPU memory
- trainable parameters
- training flag
- train source
- train time or report-only note
- offline preprocessing time
- model calls
- per-stage latency

Exit criteria:

- E4 table does not mix partial official-log timing with isolated final timing.
- Every method row has a populated E4 record or a concrete blocker.

### P8 - Appendix Transfer

Purpose: run transfer/stress targets only after the main E1-E4 and diagnostic
tables are stable.

Appendix hard-domain transfer:

- OVCamo-TE
- CAMO-TE
- COD10K-TE-Camo
- NC4K

Appendix metrics:

- IoU
- S_m
- F_beta^w
- E_m
- MAE
- OVCamo class-aware metrics when labels are aligned.

Exit criteria:

- hard-domain results are clearly separated from E1-E4 semantic mIoU tables.

---

## Dataset Matrix

### E1 Standard

| Dataset | Manifest / official name | Role |
|---------|--------------------------|------|
| VOC20 | `voc20_val` / `voc20` | compact object semantic target |
| Context59 | `context59_val` / `context59` | compact scene semantic target |
| ADE20K-150 | `ade20k150_val` / `ade20k` | compact scene semantic target |
| COCO-Stuff171 | `coco_stuff171_val` / `coco_stuff164k` | object + stuff semantic target |

### E2 Vocabulary Robustness

| Compact | Large | Required result |
|---------|-------|-----------------|
| Context59 | Context459 | mIoU compact, mIoU large, Delta_ctx |
| ADE20K-150 | ADE20K-847 | mIoU compact, mIoU large, Delta_ade |

### Appendix Transfer

| Dataset | Role |
|---------|------|
| OVCamo-TE | class-aware hard-domain target where labels align |
| CAMO-TE | mask-only hard-domain target |
| COD10K-TE-Camo | mask-only hard-domain target |
| NC4K | mask-only hard-domain target |

---

## Method Checklist

| Method | Priority | Current state | Next action |
|--------|----------|---------------|-------------|
| MaskCLIP-lite | done | E1/E2 adapter results exist | keep as baseline, not final MaskCLIP row |
| SCLIP | P0 | official E1 complete | use official E1; add E2 only if official config can scale vocab |
| NACLIP | P0 | official E1 complete | use official E1; add E2 only if official config can scale vocab |
| ResCLIP | P0 | official E1 complete | use official E1; add E2 only if official config can scale vocab |
| ProxyCLIP | P0.5 | official E1 complete; E2 queue running | collect Context459/ADE847 E2 |
| CorrCLIP | P0.5 | official E1 complete; E2 queue running | collect Context459/ADE847 E2 |
| MaskCLIP | P4 | not implemented | implement real attention-surgery adapter |
| CLIP-DIY | P4 | not implemented | implement or run official route |
| CLIPtrase | P4 | not implemented | implement or run official route |
| SC-CLIP | P4 | not implemented | verify source, then implement or run official route |
| SAM-AMG + CLIP | P3 | not implemented | first proposal/naming adapter |
| SAM-AMG + SigLIP | P3 | not implemented | second proposal/naming adapter |
| GroundingDINO + SAM | P3 | not implemented | detector/SAM adapter |
| GroundingDINO + SAM2 | P3 | not implemented | detector/SAM2 adapter |
| DINOv2 + SAM + CLIP | P3 | not implemented | DINO-guided proposal/naming adapter |
| DINOv2 + SAM + SigLIP | P3 | not implemented | DINO-guided proposal/naming adapter |
| Trident | P4 | not implemented | adapter or official-repo run |
| CASS | P4 | not implemented | adapter or official-repo run |
| OVDiff | P4 | not implemented | diffusion row; track offline synthesis separately |
| FreeDA | P4 | not implemented | diffusion row; track offline prototypes separately |
| OVSeg | P5 | not run | official inference or published aligned numbers |
| SAN | P5 | not run | official inference or published aligned numbers |
| ODISE | P5 | not run | official inference or published aligned numbers |

---

## Table-Building Commands

After each run batch:

```bash
python -m tf_ovcos.summarize_results --out-dir runs/tables
/data/tianyi/conda_envs/tf-ovos/bin/python scripts/analyze_official_results.py
```

Before paper table export:

```bash
python -m tf_ovcos.summarize_results \
  --out-dir runs/tables \
  --include-e1 \
  --include-e2 \
  --include-e3 \
  --include-e4
```

If a CLI flag above is not implemented yet, implement the exporter before
adding more method rows; table generation should not be a manual spreadsheet
step.

---

## Risk Controls

- Never tune thresholds on target validation metrics. Thresholds, prompts,
  SAM settings, and resize sizes are fixed before full evaluation.
- Keep official-repo results and adapter-native results separate in filenames
  and tables until protocols are verified.
- For COCO-Stuff official methods, use the official labelTrainIds convention
  already fixed in `scripts/prepare_official_mmseg_configs.py`.
- For CorrCLIP, preserve the downloaded official region masks and the missing
  Context59 zero-mask fallback note.
- For exploratory metrics, record broad readouts, but promote only patterns that
  are stable across at least two datasets and distinguish method families.

---

## Immediate Next Steps

1. Let `official_e2_queue` finish and confirm ProxyCLIP/CorrCLIP E2 logs.
2. Refresh analysis and freeze official E1 + available official E2 rows.
3. Implement/run `SAM-AMG + CLIP`, then `SAM-AMG + SigLIP`.
4. Implement/run `GroundingDINO + SAM`, then `GroundingDINO + SAM2`.
5. Implement/run `DINOv2 + SAM + CLIP`, then `DINOv2 + SAM + SigLIP`.
6. Implement/run `MaskCLIP`, `CLIP-DIY`, `CLIPtrase`, `SC-CLIP`, `Trident`, `CASS`.
7. Implement/run `OVDiff` and `FreeDA`, with offline stage cost separated.
8. Add trained references or document published-number compatibility.
9. Run diagnostic probes.
10. Run isolated E4 timing pass.
11. Export E1/E2/E3/E4/diagnostic tables.
12. Decide appendix transfer scope.
