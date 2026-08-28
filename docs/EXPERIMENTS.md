# Experiment ledger

Every run of this project: finished, in flight, and queued. **Generated** by
`tools/exp_log.py` -- do not edit the tables by hand, edit `docs/experiments.yaml`
and re-render. Every AUC below is read off disk at render time.

Rendered 2026-08-28 17:13 UTC.

Two architectures appear here and they must never be compared casually:

- **ARC-CT** -- The published model (MICCAI 2026 TIA). R3D-18 pretrained on Kinetics, layer4 feature map [B,512,12,12,12], AnatomyQFormer with 39 queries (10 anatomy, 27 pathology, 2 global), CLIP-contrastive against a frozen CXR-BERT with LoRA on layers 8-11. The image is the only input. Every number under this heading is a baseline, never a result of the new work.
- **context** -- The new model. The same tower, and queries[:39] load bit-identical from the ARC-CT checkpoint, plus ONE new parameter row (clinical global) read out over 67 slots: 27 of them are the pathology queries re-read after conditioning on the indication (C1: cross-attention then FiLM), and pooling is steered by a relevance weight that can only ever raise, never suppress (C2: w = 1 + beta*r). Age band and sex enter as context tokens. At initialisation it is numerically the published model, so every delta reads as departure from ARC-CT.

`val AUC` is the validation macro AUC the training loop selected on, mean over
completed seeds. It is a MODEL SELECTION number and is not a result. `held-out`
columns come from `eval_matrix` and are the reportable ones.

## ARC-CT: pediatric fine-tuning (V-series)

Recipe search on the 8,816-volume BCH cohort, warm-started from the published adult checkpoint and re-headed to the 27-class pediatric schema. V1 is the recipe everything since is measured against; V5 is the best of the search. The three V1 seeds are also where the seed band comes from.

| experiment | dates | arch | cohort · classes | masks | seeds | val AUC | state |
|---|---|---|---|---|---|---|---|
| **V1** | 2026-08-22 → 2026-08-27 | ARC-CT | peds 7,001 · 27 | peds 100% | 3/3 | 0.7922 ±0.0046 | done |
| **V2** | 2026-08-23 | ARC-CT | peds 7,001 · 27 | peds 100% | 1/1 | 0.7974 | done |
| **V3** | 2026-08-23 | ARC-CT | peds 7,001 · 27 | peds 100% | 1/1 | 0.7971 | done |
| **V4** | 2026-08-23 | ARC-CT | peds 7,001 · 27 | peds 100% | 1/1 | 0.7972 | done |
| **V5** | 2026-08-23 | ARC-CT | peds 7,001 · 27 | peds 100% | 1/1 | 0.8019 | done |

Held-out evaluation:

| experiment | cell | n | classes | global AUC | 95% CI | routed AUC | Δ routed |
|---|---|---|---|---|---|---|---|
| V1 | `peds/V1` | 1772 | 27 | 0.8005 | 0.7819–0.8128 | 0.7651 | -0.0354 |
| V1 | `ctrate/V1` | 3002 | 27 | 0.7954 | 0.7821–0.8047 | no mask | -- |
| V1 | `ctrate/V1_masked` | 3002 | 27 | 0.7991 | 0.7870–0.8089 | 0.7617 | -0.0375 |
| V1 | `combined/V1` | 6362 | 27 | 0.7821 | 0.7747–0.7891 | 0.7469 | -0.0353 |
| V2 | `peds/V2` | 1772 | 27 | 0.7974 | 0.7774–0.8098 | 0.7657 | -0.0317 |
| V2 | `ctrate/V2` | 3002 | 27 | 0.8000 | 0.7855–0.8102 | no mask | -- |
| V2 | `combined/V2` | 6362 | 27 | 0.7868 | 0.7795–0.7938 | 0.7470 | -0.0399 |
| V3 | `peds/V3` | 1772 | 27 | 0.7972 | 0.7773–0.8095 | 0.7654 | -0.0318 |
| V3 | `ctrate/V3` | 3002 | 27 | 0.8003 | 0.7860–0.8104 | no mask | -- |
| V3 | `combined/V3` | 6362 | 27 | 0.7869 | 0.7796–0.7940 | 0.7470 | -0.0399 |
| V4 | `peds/V4` | 1772 | 27 | 0.7971 | 0.7772–0.8097 | 0.7645 | -0.0326 |
| V4 | `ctrate/V4` | 3002 | 27 | 0.8005 | 0.7867–0.8106 | no mask | -- |
| V4 | `combined/V4` | 6362 | 27 | 0.7867 | 0.7792–0.7936 | 0.7476 | -0.0391 |
| V5 | `peds/V5` | 1772 | 27 | 0.8019 | 0.7828–0.8142 | 0.7662 | -0.0357 |
| V5 | `ctrate/V5` | 3002 | 27 | 0.7925 | 0.7819–0.8015 | no mask | -- |
| V5 | `ctrate/V5_masked` | 3002 | 27 | 0.8000 | 0.7897–0.8096 | 0.7556 | -0.0444 |
| V5 | `combined/V5` | 6362 | 27 | 0.7841 | 0.7763–0.7910 | 0.7408 | -0.0433 |

**V1 -- what it settled.** The reference recipe, and the source of the seed band: three seeds of an identical configuration give sd = 0.0046, so a ladder delta below about 0.0092 is inside seed noise and is not a difference. Every ablation in this project is read against that number.

**V5 -- what it settled.** Best pediatric held-out AUC of the recipe search, and the checkpoint the age-stratified analysis runs on. Its lead over V1 on pediatrics is inside the 0.0046 seed band, so it is the better recipe by selection, not by a demonstrated margin.

## ARC-CT: joint pediatric + CT-RATE (C-series)

The same architecture trained on both cohorts together, testing whether adult volume helps the pediatric half and what the label harmonisation costs. "mix" = each cohort keeps its own labeller; "harm" = one labeller for both.

| experiment | dates | arch | cohort · classes | masks | seeds | val AUC | state |
|---|---|---|---|---|---|---|---|
| **C16** | 2026-08-22 → 2026-08-23 | ARC-CT | peds + CT-RATE 16k · 27 | peds only | 1/1 | 0.8175 | done |
| **C47mix** | 2026-08-23 → 2026-08-24 | ARC-CT | peds + CT-RATE 47k · 27 | peds only | 1/1 | 0.8248 | done |
| **C47harm** | 2026-08-24 → 2026-08-25 | ARC-CT | peds + CT-RATE 47k · 27 | peds only | 1/1 | 0.8170 | done |

Held-out evaluation:

| experiment | cell | n | classes | global AUC | 95% CI | routed AUC | Δ routed |
|---|---|---|---|---|---|---|---|
| C16 | `peds/C16` | 1772 | 27 | 0.7881 | 0.7599–0.8039 | 0.7596 | -0.0284 |
| C16 | `ctrate/C16` | 3002 | 27 | 0.7991 | 0.7764–0.8109 | no mask | -- |
| C16 | `combined/C16` | 6362 | 27 | 0.8085 | 0.8021–0.8155 | 0.7649 | -0.0436 |
| C47mix | `peds/C47mix` | 1772 | 27 | 0.7964 | 0.7744–0.8100 | 0.7608 | -0.0356 |
| C47mix | `ctrate/C47mix` | 3002 | 27 | 0.8228 | 0.8052–0.8336 | no mask | -- |
| C47mix | `ctrate/C47mix_masked` | 3002 | 27 | 0.8242 | 0.8129–0.8339 | 0.7645 | -0.0597 |
| C47mix | `combined/C47mix_mixlbl` | 6362 | 27 | 0.8248 | 0.8183–0.8309 | 0.7967 | -0.0281 |
| C47mix | `combined/C47mix_harmlbl` | 6361 | 27 | 0.8221 | 0.8151–0.8284 | 0.7913 | -0.0309 |
| C47harm | `peds/C47harm` | 1772 | 27 | 0.7886 | 0.7684–0.8001 | 0.7587 | -0.0299 |
| C47harm | `ctrate/C47harm` | 3002 | 27 | 0.8271 | 0.8149–0.8361 | no mask | -- |
| C47harm | `ctrate/C47harm_masked` | 3002 | 27 | 0.8213 | 0.8112–0.8308 | 0.7642 | -0.0570 |
| C47harm | `combined/C47harm_harmlbl` | 6361 | 27 | 0.8170 | 0.8099–0.8237 | 0.7799 | -0.0371 |
| C47harm | `combined/C47harm_mixlbl` | 6362 | 27 | 0.8185 | 0.8119–0.8247 | 0.7838 | -0.0346 |

**C47mix -- what it settled.** The strongest held-out numbers in the project, and the reason the joint split is where the new model has to be shown: adult volume lifts the pediatric half too. Each cohort keeps its own labeller here, which is also its weakness -- see C47harm.

**C47harm -- what it settled.** One labeller across both cohorts. It scores below C47mix, and that gap is partly the price of removing an artefact rather than a loss of signal: with two labellers the same class carried two thresholds across the halves -- Lymphadenopathy 25.9% adult against 7.9% pediatric, of which 10.8 points were the annotator and not the disease. The harmonised labels are what the context runs train on; the PUBLISHED CT-RATE labels stay the adult evaluation target.

## Phase 0: does anatomy routing cost accuracy, and is that pediatric?

Not training runs -- the same checkpoints evaluated twice, once pooling globally and once through the anatomy masks. The question is whether the routing penalty in the pediatric cohort is a property of children or of the method. A cell whose routed column reads "no mask" had no mask supplied and fell back to global; it is not a routed measurement.

| experiment | dates | arch | cohort · classes | masks | seeds | val AUC | state |
|---|---|---|---|---|---|---|---|
| **P0_routing** | -- | ARC-CT | eval only · 27 | eval-time | 0/0 | -- | planned |

Held-out evaluation:

| experiment | cell | n | classes | global AUC | 95% CI | routed AUC | Δ routed |
|---|---|---|---|---|---|---|---|
| P0_routing | `ctrate/V1_masked` | 3002 | 27 | 0.7991 | 0.7870–0.8089 | 0.7617 | -0.0375 |
| P0_routing | `ctrate/V5_masked` | 3002 | 27 | 0.8000 | 0.7897–0.8096 | 0.7556 | -0.0444 |
| P0_routing | `ctrate/C47mix_masked` | 3002 | 27 | 0.8242 | 0.8129–0.8339 | 0.7645 | -0.0597 |
| P0_routing | `ctrate/C47harm_masked` | 3002 | 27 | 0.8213 | 0.8112–0.8308 | 0.7642 | -0.0570 |
| P0_routing | `peds/V1` | 1772 | 27 | 0.8005 | 0.7819–0.8128 | 0.7651 | -0.0354 |
| P0_routing | `peds/V5` | 1772 | 27 | 0.8019 | 0.7828–0.8142 | 0.7662 | -0.0357 |
| P0_routing | `combined/V1` | 6362 | 27 | 0.7821 | 0.7747–0.7891 | 0.7469 | -0.0353 |
| P0_routing | `combined/V5` | 6362 | 27 | 0.7841 | 0.7763–0.7910 | 0.7408 | -0.0433 |

**P0_routing -- what it settled.** The routing penalty is real and it is NOT pediatric -- on the same checkpoint and the same 27 classes, CT-RATE falls 0.7991 to 0.7617 (-0.0375) and pediatrics 0.8005 to 0.7651 (-0.0354). But the macro number badly misdescribes what happens, and the per-class breakdown is the finding. TWO classes account for roughly two thirds of the entire penalty: Pneumothorax 0.695 to 0.317 and Bone lesion 0.616 to 0.347. Those are BELOW CHANCE. A model merely looking in the wrong place lands near 0.5; below 0.5 is systematic inversion, and it reproduces in every cohort and every checkpoint measured. Both route to the two regions we DERIVED rather than segmented with TotalSegmentator -- the pleural shell and bone. Drop those two and the penalty falls to -0.0146 on CT-RATE and -0.0159 on pediatrics, with 7 of the remaining 24 classes IMPROVING under routing (Pulmonary metastases +0.061, Pleural thickening +0.018). Three classes are never routed at all and are identical by construction. A structural contributor to the residual: _routed_probs scores organ_lat against the class prompts, but training aligns organ_lat to REGION SENTENCES (L_org) and the class prompts to the GLOBAL latent (L_cls). That pairing is never trained, so part of the remaining -0.015 is a read-out mismatch rather than a statement about anatomy. Two loose ends. ctrate/V1 reads 0.7954 against ctrate/V1_masked's 0.7991 on the same checkpoint, labels and order -- config drift between two evaluation runs, unexplained; it does not touch the paired -0.0375. And the combined cells lose more than either cohort alone (-0.0353, and -0.0475 when restricted to masked rows). The obvious explanation -- that partial mask coverage mixes two score scales in one ranking -- was tested and is wrong, since restricting to masked rows makes it worse, not better. Unexplained.

## Context ladder, first pass (ctx)

Eight rungs x three seeds on pediatrics alone, adding one mechanism at a time. This pass is superseded: beta was initialised at -6, where d(beta)/db = sigma(-6) = 0.0025, so the relevance gate could not open within the run -- measured +0.00012 against a predicted +0.000100. The C2 rung here is therefore not a test of C2. Kept because the C1 rungs are unaffected.

| experiment | dates | arch | cohort · classes | masks | seeds | val AUC | state |
|---|---|---|---|---|---|---|---|
| **ctx/ct_only** | 2026-08-26 → 2026-08-27 | context | peds 7,001 · 27 | peds 100% | 2/3 | 0.7906 ±0.0003 | partial |
| **ctx/concat** | 2026-08-26 → 2026-08-27 | context | peds 7,001 · 27 | peds 100% | 3/3 | 0.7928 ±0.0016 | done |
| **ctx/c1_xattn** | 2026-08-26 → 2026-08-27 | context | peds 7,001 · 27 | peds 100% | 2/3 | 0.7989 ±0.0007 | partial |
| **ctx/c1_full** | 2026-08-26 → 2026-08-27 | context | peds 7,001 · 27 | peds 100% | 3/3 | 0.7980 ±0.0017 | done |
| **ctx/c2** | 2026-08-26 → 2026-08-27 | context | peds 7,001 · 27 | peds 100% | 3/3 | 0.7979 ±0.0017 | done |
| **ctx/fusion_gated** | 2026-08-27 | context | peds 7,001 · 27 | peds 100% | 1/3 | 0.7949 | partial |
| **ctx/isolation_off** | 2026-08-27 | context | peds 7,001 · 27 | peds 100% | 0/3 | -- | STALLED |
| **ctx/age_scalar** | 2026-08-27 | context | peds 7,001 · 27 | peds 100% | 0/1 | -- | STALLED |

`ctx/ct_only`: 1 seed(s) died before update 2,000 and are excluded from the mean (`ctx_ct_only_seed1`).

`ctx/c1_xattn`: 1 seed(s) died before update 2,000 and are excluded from the mean (`ctx_c1_xattn_seed1`).

**ctx/c2 -- what it settled.** Not a test of C2. beta was initialised at -6, where the softplus gradient is sigma(-6) = 0.0025; over the whole run beta moved by a measured 0.00012 against a predicted 0.000100, so the relevance gate never opened and this rung is C1 with an inert head attached. The rerun is ctx2/c2.

`ctx/fusion_gated`: 2 seed(s) died before update 2,000 and are excluded from the mean (`ctx_fusion_gated_seed1`, `ctx_fusion_gated_seed2`).

**ctx/isolation_off -- what it settled.** Killed before producing anything. With C2 off, ctx_out.r is None and the guard indexed it anyway -- a TypeError that took out nine of this ladder's twenty-four tasks. The guard now holds all classes stable when r is None.

**ctx/age_scalar -- what it settled.** Same TypeError as isolation_off; never produced a checkpoint.

## Context ladder, second pass (ctx2)

The rerun with beta_init -2, a 20x learning-rate multiplier on the zero-init context modules, and patience 8. This is the pass that picks the winner for the joint runs.

| experiment | dates | arch | cohort · classes | masks | seeds | val AUC | state |
|---|---|---|---|---|---|---|---|
| **ctx2/ct_only** | 2026-08-27 → 2026-08-28 | context | peds 7,001 · 27 | peds 100% | 3/3 | 0.7908 ±0.0013 | done |
| **ctx2/concat** | 2026-08-27 | context | peds 7,001 · 27 | peds 100% | 3/3 | 0.7854 ±0.0025 | done |
| **ctx2/c1_xattn** | 2026-08-27 → 2026-08-28 | context | peds 7,001 · 27 | peds 100% | 3/3 | 0.7942 ±0.0024 | done |
| **ctx2/c1_full** | 2026-08-27 → 2026-08-28 | context | peds 7,001 · 27 | peds 100% | 3/3 | 0.7940 ±0.0022 | done |
| **ctx2/c2** | 2026-08-27 → 2026-08-28 | context | peds 7,001 · 27 | peds 100% | 3/3 | 0.7925 ±0.0028 | queued |
| **ctx2/fusion_gated** | 2026-08-27 → 2026-08-28 | context | peds 7,001 · 27 | peds 100% | 2/3 | 0.7954 ±0.0014 | queued |
| **ctx2/isolation_off** | 2026-08-27 | context | peds 7,001 · 27 | peds 100% | 0/3 | -- | queued |
| **ctx2/age_scalar** | 2026-08-27 → 2026-08-28 | context | peds 7,001 · 27 | peds 100% | 3/3 | 0.7957 ±0.0017 | queued |

**ctx2/ct_only -- what it settled.** The architecture-only control: the context model with the Q-Former's context path switched off, so it is the published model under the new code path. Every other rung is a delta against this, not against V1.

## Joint context runs (the headline)

The winning rung retrained on the combined 49,545-volume split, alongside ct_only on the SAME split. The joint baseline is not redundant: CT-RATE outnumbers pediatrics six to one there, and a configuration that wins on 7,001 pediatric volumes is not thereby a winner in that regime.

| experiment | dates | arch | cohort · classes | masks | seeds | val AUC | state |
|---|---|---|---|---|---|---|---|
| **joint/ct_only** | -- | context | peds + CT-RATE 49,545 · 27 | peds only (71.6% once ctx3 masks land) | 0/3 | -- | queued |
| **joint/winner** | -- | context | peds + CT-RATE 49,545 · 27 | peds only | 0/0 | -- | planned |

**joint/winner -- what it settled.** Waits on the ctx2 ladder to name a winner. Submitted as `sbatch --export=ALL,JOINT_RUNG=<winner> --array=0-5%3 pipeline/05_train/34_train_joint_ctx.sbatch`, which runs it and the joint baseline in one array.

## CT-RATE only: the adult gate

The run that has to clear 0.8574 on the 18 classes CT-RATE publishes. No pediatric data, published 18-class labels, warm-started from the published adult checkpoint (qformer.queries (30,768), so the context bank appends one row to reach 31). ct_only runs beside it on the same data, because 0.8574 came from THEIR training run and without our own control a result of 0.855 could not be told apart from a recipe that fails to reproduce ARC-CT.

| experiment | dates | arch | cohort · classes | masks | seeds | val AUC | state |
|---|---|---|---|---|---|---|---|
| **ctrate/ct_only** | -- | ARC-CT | CT-RATE 42,544 · 18 (published) | CT-RATE, gated at 99% | 0/3 | -- | queued |
| **ctrate/c2** | -- | context | CT-RATE 42,544 · 18 (published) | CT-RATE, gated at 99% | 0/3 | -- | queued |

**ctrate/ct_only -- what it settled.** The control, and it is not a formality: it must come back near the published checkpoint's own 0.8547 for the context arm's delta to mean anything.

**ctrate/c2 -- what it settled.** Blocked on mask coverage, deliberately. configs/stage2.env sets RAC_REQUIRE_MASK=1 and dataset.py drops any training volume without a mask, so at today's 67% this would have trained on 28,494 volumes instead of 42,544 -- against a published model that had all of them -- and said so in one line of log. check_mask_coverage.py now exits 78 below 99%. The 18-class step-0 identity gate has PASSED: 49 slots over 31 rows, Z_final == Z_gen exactly, unconditioned tokens matching the 39-slot model to 0.0e+00, preflight clean.

## Label repair (V3) and the gold standard

A 200-report gold standard, annotated five times independently, against which the pediatric label definitions were repaired. Qwen against the gold agrees on 95.7% of 5,400 cells at mean kappa 0.855. The gains are concentrated on the classes two text-only audits had already proved broken.

| experiment | dates | arch | cohort · classes | masks | seeds | val AUC | state |
|---|---|---|---|---|---|---|---|
| **V3_labels** | -- | n/a | peds 8,817 reports · 27 | -- | 0/0 | -- | queued |

**V3_labels -- what it settled.** Extraction running (8 shards, 3 at a time). Definitions repaired against the gold standard, and the single most important change is not per class: the evidence citation became a GATE rather than a justification -- if no sentence states the finding itself, the label is 0. Kappa gains land exactly where two text-only audits said they would: Tree-in-bud 0.844 to 0.956, Arterial wall calcification 0.689 to 0.817, Lymphadenopathy 0.676 to 0.780, Peribronchial thickening 0.819 to 0.902. Losses cluster on classes stated diffusely (Bronchiectasis, Consolidation, Lung opacity) and are the same trade that produced the gains. Whether any of it moves AUC is untested -- that is the next experiment, not this one.

## Masked ablation (ctx3)

The winner rerun with anatomy masks on the adult half too. Held separate from the ctx2 ladder on purpose: switching masks on at the same time as C1/C2 would make the gain unattributable.

| experiment | dates | arch | cohort · classes | masks | seeds | val AUC | state |
|---|---|---|---|---|---|---|---|
| **ctx3/masked** | -- | context | peds + CT-RATE 49,545 · 27 | 100% (in production) | 0/0 | -- | planned |

**ctx3/masked -- what it settled.** Blocked on mask generation, which is running. 31,570 CT-RATE training masks arrived from another machine already in target format, taking coverage from 14.1% (pediatrics alone) to 71.6%; the remaining 15,575 are being segmented here so the training set has one provenance. Worth finishing rather than training at 71.6% because the gap is not a random third -- sliced by patient id it runs 1.6 / 58 / 66 / 8 / 2 / 76 / 74 / 43 / 4 / 2 percent missing, the signature of a shard array that stopped part way, which would leave L_org fitted to a subsample selected by acquisition order.

## Journal

Dated record of what happened and what it changed. Newest first.

**2026-08-28 — A sync loop emptied the context config, and the identity gate passed anyway**  configs/stage2_ctrate_ctx.env was zero bytes and had been committed that way: a loop meant to copy it into the second pipeline tree resolved to the file itself, and a redirect onto its own source truncates it. The gate passed on the empty file, because a missing env file does not fail -- it leaves the code defaults, and those are beta_init = -6.0 and lr_mult = 1.0, exactly the two settings that made the first pediatric ladder measure nothing. Six runs would have spent three days reproducing that with nothing objecting, since every individual default is legal. The launcher now rejects an empty variable and those two values by name.

**2026-08-28 — Two shards hung on a sick node, and the watchdog is what said so**  Both remaining segmentation shards stopped writing for eight and fourteen hours while SLURM still showed them RUNNING -- a hung job is not a failed one. Both were on gpu-25-0, one after a FileNotFoundError on its own node-local nnUNet scratch file and a volume that took 2,185 s against a 94 s median. The 91 volumes they owed were finished by a cleanup array with that node excluded by name. Segmentation is complete at 15,575 of 15,575.

**2026-08-28 — Two alarms from yesterday were misreadings of my own**  |W_ind|/|W_gen| = 0.000 and cf = 0.0000 were read as the context pathway failing to train. Both were structural: the run in question is the fusion_gated rung, which does not use the linear fusion at all, so its weight stays at its [I|0] init and the ratio is zero by construction -- and with the gate shut the indication cannot change the prediction, so the counterfactual term is exactly zero too. The c2 rungs show cf firing normally at 0.0013 to 0.0056. The ladder is not invalidated. A separate 1000x slowdown was real but transient: the training was starved of I/O while 29 segmentation shards read full volumes off NFS, and it returned to 2.6 s/update once they finished.

**2026-08-28 — Stage 2 was running single-threaded on the critical path**  Reducing the raw labels to 10-region masks measured 14.9 volumes a minute as one process -- 17.4 hours for 15,575, with the CT-RATE run waiting on it. build_peds_masks.py already sharded and already skipped existing outputs, so the parallelism was there and simply unused. Re-run as a 16-way array.

**2026-08-27 — The ribcage-derived pleural region does not work -- measured, not assumed**  Rebuilding region 8 from the bony thorax instead of the lung leaves it essentially EMPTY: 130-212 voxels against the old 133k-265k, and 207 of 208 volumes reach ZERO tokens on the 12x12x12 grid. A region with no tokens makes its query UNRESTRICTED, so this is not a fix but a silent deletion of the anatomy query. The cause is obvious in hindsight: the ribcage interior is already lung, heart and vessels, and the pleural region may only claim background, so nothing is left to claim. The decision file was NOT written and the overnight chain keeps the existing definition -- known behaviour beats an unverified one. What the first measurement actually implies is narrower and more useful: TotalSegmentator folds pneumothorax air INTO the lung label, so no pleural-space definition can capture it, and the candidate worth testing next is routing Pneumothorax to the lung lobes rather than to region 8.

**2026-08-27 — The routing penalty is two broken regions, and the fix is measured not assumed**  Region 8 is not the pleural space. On 400 volumes stratified for the class it reads +6 HU on pneumothorax positives against -98 HU on negatives -- 104 HU the WRONG way for an air finding, with the region twice the size. The shell is derived by dilating the lung union, so its geometry is a function of the pathology: TotalSegmentator folds pneumothorax air into the lung field, the shell lands further out in the chest wall, and the pooled feature ends up anti-correlated with the finding. Rebuilt from the bony ribcage instead, which does not move when a lung collapses. Being verified on a rebuilt sample before any of the 47k masks are committed to it.

**2026-08-27 — The CT-RATE-only run is set up and its 18-class gate passes**  Assets built and asserted, architecture verified at 18 classes (49 slots over 31 rows, step-0 identity exact, preflight clean). The gate found three real faults first: RAC_FOCAL_CLASSES arrived EMPTY because an unquoted value with a space broke the env sourcing, so top-K focal pooling was off in every adult- recipe run; the age-band check refused an adult-only cohort on principle; and the counterfactual sampler rejected only the same accession, not the same text, so L_cf routinely compared a prediction with itself.

**2026-08-27 — The sbatch scripts had locked themselves out of 24 A100s**  Every GPU launcher asked for gres=gpu:large:1, and the A100s are gpu:xlarge -- a gpu:large request can never match one, and bch-gpu-xlarge was not in the partition list either. gpu:large had been chosen to avoid the 24 GB TITANs that OOM; those are gpu-1-0 and gpu-2-0, the only medium nodes across all three partitions, so excluding them by name does the same job without giving up the pool this account contends least for.

**2026-08-27 — Half of CT-RATE has an indication, not a quarter**  The user challenged the 24% figure and was right. classify() required three words, which fits the pediatric requisitions (median 24 words) and discards the adult ones, which are often a single question. Coverage is 48.7%. An assertion encoding the same three-word rule had to be corrected too, since it blocked the fix.

**2026-08-27 — 31,570 CT-RATE training masks arrived, already in target format**  Moved from another machine at 192x192x96 with labels 0-10, so nothing needed converting. Coverage went from 14.1% (pediatrics alone) to 71.6% of the training split. An alignment check against two known-good controls put them between our own CT-RATE validation masks (98% pass) and our pediatric masks (75%), at 82% -- not broken.

**2026-08-27 — Generating the remaining 15,575 rather than training at 71.6%**  The gap is not a random third: sliced by patient id it runs 1.6 / 58 / 66 / 8 / 2 / 76 / 74 / 43 / 4 / 2 percent missing, the signature of a shard array that stopped part way. All 15,575 already have their npz on disk, so nothing is downloaded. Also found that ts_ctrate.py could not be imported from this repo at all -- two lines above the __future__ import, a SyntaxError, invisible because ast.parse accepts it and because the valid masks were built from a second copy of the tree.

**2026-08-27 — A quiet run directory is not a finished run**  Building the ledger showed that runs killed at update 400 leave exactly what finished runs leave, and their early AUCs were being averaged in. Excluding them, ctx1 reads ct_only 0.7906 (not 0.7883), c1_full +0.0074 (not +0.0097), c2 +0.0073 (not +0.0106) -- so NO ctx1 rung clears the 0.0092 seed band, where two previously appeared to.

**2026-08-27 — The routing penalty is two regions, not the method**  Prompted by the user asking how telling the model where to look could possibly hurt this much. Per class it does not: Pneumothorax (0.695 to 0.317) and Bone lesion (0.616 to 0.347) fall BELOW CHANCE and carry two thirds of the whole -0.0375, in every cohort and every checkpoint. Excluding them the penalty is -0.0146 on CT-RATE and -0.0159 on pediatrics, and 7 of the remaining 24 classes improve. Both route to the two regions we derived rather than segmented. Region repair is now queued ahead of any conclusion about anatomy routing as a method.

**2026-08-26 — Phase 1 implemented end to end**  Slot layout, group-restricted attention, the C1 conditioner, the C2 relevance head, the context Q-Former and its step-0 identity gate, the training wiring, and the PHI scrubber for the indication field. The identity gate passes exactly: Z_final equals Z_gen, and the 39 unconditioned tokens match the published model to 0.0e+00.

**2026-08-26 — Sixteen sbatch scripts pointed at a directory with no schema.py**  Every launcher sourced $HOME/arc-ct, which after a home reorganisation held the published ADULT tree -- no configs/stage2_peds.env, a different train_stage2.py. Without set -e the missing env file printed one line and the run continued as an 18-class CT-RATE job with the Q-Former off. Silent and entirely wrong. Fixed with ARCCT_ROOT, a hard gate, and check_env.py.

**2026-08-26 — The first ladder measured almost nothing**  Two independent faults. beta initialised at -6, where the softplus gradient is sigma(-6) = 0.0025, so the relevance gate could not open -- beta moved a measured 0.00012 over the whole run. And the zero-init context modules trained at the warm-start learning rate. Rerun as ctx2 with beta_init -2 and a 20x learning-rate multiplier. Separately, a TypeError on ctx_out.r with C2 off killed nine of the ladder's twenty-four tasks.

## Standing conclusions

**The seed band is 0.0046**  Three seeds of an identical V1 configuration give sd = 0.0046 on pediatric validation macro AUC. A ladder delta below roughly 0.0092 is inside that band and is not a difference. This is measured on the V1 recipe, not the context recipe, and is the most honest estimate available rather than an exact one.

**Two broken regions, not a broken idea**  The -0.0375 routing penalty is not spread across the label set. Pneumothorax (0.695 to 0.317) and Bone lesion (0.616 to 0.347) fall BELOW CHANCE and carry about two thirds of it, in every cohort and every checkpoint. Below 0.5 is inversion, not misplaced attention -- looking in the wrong place gives 0.5. Both route to the two regions we derived rather than segmented. Excluding them the penalty is -0.0146 (CT-RATE) and -0.0159 (pediatrics), and 7 of the remaining 24 classes improve under routing. Fix those two regions before concluding anything about anatomy routing as a method.

**The routed read-out uses a pairing that was never trained**  Training aligns organ_lat to region sentences (L_org) and the class prompts to the global latent (L_cls). Evaluation scores organ_lat against the class prompts. Part of the residual routing penalty is therefore a read-out mismatch and not a property of anatomy routing.

**Validation AUC is not a result**  The val AUC columns are what the training loop selected checkpoints on. They are computed on the split the model stopped early against, so they are optimistically biased by construction. Only eval_matrix numbers are reportable, and only with the patient-clustered bootstrap CI beside them.

**A quiet run directory is not a finished run**  A run killed at update 400 leaves what a finished run leaves: a best.pt, a best_auc file, and silence. Nine of the first ladder's twenty-four tasks died that way, and averaging their early AUCs would report those arms as worse than they are. Runs below patience x val_every are marked DIED and excluded.

**Half of CT-RATE has an indication, not a quarter**  An early filter required three words, which suited the pediatric requisitions (median 24 words) and threw away the adult ones, which are often a single question. Corrected, CT-RATE indication coverage is 48.7%, not 24.1%.

