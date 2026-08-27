# Experiment ledger

Every run of this project: finished, in flight, and queued. **Generated** by
`tools/exp_log.py` -- do not edit the tables by hand, edit `docs/experiments.yaml`
and re-render. Every AUC below is read off disk at render time.

Rendered 2026-08-27 19:02 UTC.

Two architectures appear here and they must never be compared casually:

- **ARC-CT** -- The published model (MICCAI 2026 TIA). R3D-18 pretrained on Kinetics, layer4 feature map [B,512,12,12,12], AnatomyQFormer with 39 queries (10 anatomy, 27 pathology, 2 global), CLIP-contrastive against a frozen CXR-BERT with LoRA on layers 8-11. The image is the only input. Every number under this heading is a baseline, never a result of the new work.
- **context** -- The new model. The same tower, and queries[:39] load bit-identical from the ARC-CT checkpoint, plus ONE new parameter row (clinical global) read out over 67 slots: 27 of them are the pathology queries re-read after conditioning on the indication (C1: cross-attention then FiLM), and pooling is steered by a relevance weight that can only ever raise, never suppress (C2: w = 1 + beta*r). Age band and sex enter as context tokens. At initialisation it is numerically the published model, so every delta reads as departure from ARC-CT.

`val AUC` is the validation macro AUC the training loop selected on, mean over
completed seeds. It is a MODEL SELECTION number and is not a result. `held-out`
columns come from `eval_matrix` and are the reportable ones.

## ARC-CT: pediatric fine-tuning (V-series)

Recipe search on the 8,816-volume BCH cohort, warm-started from the published adult checkpoint and re-headed to the 27-class pediatric schema. V1 is the recipe everything since is measured against; V5 is the best of the search. The three V1 seeds are also where the seed band comes from.

| experiment | arch | cohort · classes | masks | seeds | val AUC | state |
|---|---|---|---|---|---|---|
| **V1** | ARC-CT | peds 7,001 · 27 | peds 100% | 3/3 | 0.7922 ±0.0046 | done |
| **V2** | ARC-CT | peds 7,001 · 27 | peds 100% | 1/1 | 0.7974 | done |
| **V3** | ARC-CT | peds 7,001 · 27 | peds 100% | 1/1 | 0.7971 | done |
| **V4** | ARC-CT | peds 7,001 · 27 | peds 100% | 1/1 | 0.7972 | done |
| **V5** | ARC-CT | peds 7,001 · 27 | peds 100% | 1/1 | 0.8019 | done |

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

| experiment | arch | cohort · classes | masks | seeds | val AUC | state |
|---|---|---|---|---|---|---|
| **C16** | ARC-CT | peds + CT-RATE 16k · 27 | peds only | 1/1 | 0.8175 | done |
| **C47mix** | ARC-CT | peds + CT-RATE 47k · 27 | peds only | 1/1 | 0.8248 | done |
| **C47harm** | ARC-CT | peds + CT-RATE 47k · 27 | peds only | 1/1 | 0.8170 | done |

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

| experiment | arch | cohort · classes | masks | seeds | val AUC | state |
|---|---|---|---|---|---|---|
| **P0_routing** | ARC-CT | eval only · 27 | eval-time | 0/0 | -- | planned |

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

| experiment | arch | cohort · classes | masks | seeds | val AUC | state |
|---|---|---|---|---|---|---|
| **ctx/ct_only** | context | peds 7,001 · 27 | peds 100% | 2/3 | 0.7906 ±0.0003 | partial |
| **ctx/concat** | context | peds 7,001 · 27 | peds 100% | 3/3 | 0.7928 ±0.0016 | done |
| **ctx/c1_xattn** | context | peds 7,001 · 27 | peds 100% | 2/3 | 0.7989 ±0.0007 | partial |
| **ctx/c1_full** | context | peds 7,001 · 27 | peds 100% | 3/3 | 0.7980 ±0.0017 | done |
| **ctx/c2** | context | peds 7,001 · 27 | peds 100% | 3/3 | 0.7979 ±0.0017 | done |
| **ctx/fusion_gated** | context | peds 7,001 · 27 | peds 100% | 1/3 | 0.7949 | partial |
| **ctx/isolation_off** | context | peds 7,001 · 27 | peds 100% | 0/3 | -- | STALLED |
| **ctx/age_scalar** | context | peds 7,001 · 27 | peds 100% | 0/1 | -- | STALLED |

`ctx/ct_only`: 1 seed(s) died before update 2,000 and are excluded from the mean (`ctx_ct_only_seed1`).

`ctx/c1_xattn`: 1 seed(s) died before update 2,000 and are excluded from the mean (`ctx_c1_xattn_seed1`).

**ctx/c2 -- what it settled.** Not a test of C2. beta was initialised at -6, where the softplus gradient is sigma(-6) = 0.0025; over the whole run beta moved by a measured 0.00012 against a predicted 0.000100, so the relevance gate never opened and this rung is C1 with an inert head attached. The rerun is ctx2/c2.

`ctx/fusion_gated`: 2 seed(s) died before update 2,000 and are excluded from the mean (`ctx_fusion_gated_seed1`, `ctx_fusion_gated_seed2`).

**ctx/isolation_off -- what it settled.** Killed before producing anything. With C2 off, ctx_out.r is None and the guard indexed it anyway -- a TypeError that took out nine of this ladder's twenty-four tasks. The guard now holds all classes stable when r is None.

**ctx/age_scalar -- what it settled.** Same TypeError as isolation_off; never produced a checkpoint.

## Context ladder, second pass (ctx2)

The rerun with beta_init -2, a 20x learning-rate multiplier on the zero-init context modules, and patience 8. This is the pass that picks the winner for the joint runs.

| experiment | arch | cohort · classes | masks | seeds | val AUC | state |
|---|---|---|---|---|---|---|
| **ctx2/ct_only** | context | peds 7,001 · 27 | peds 100% | 2/3 | 0.7909 ±0.0018 | partial |
| **ctx2/concat** | context | peds 7,001 · 27 | peds 100% | 3/3 | 0.7854 ±0.0025 | done |
| **ctx2/c1_xattn** | context | peds 7,001 · 27 | peds 100% | 1/3 | 0.7962 | partial |
| **ctx2/c1_full** | context | peds 7,001 · 27 | peds 100% | 0/3 | -- | RUNNING |
| **ctx2/c2** | context | peds 7,001 · 27 | peds 100% | 0/3 | -- | queued |
| **ctx2/fusion_gated** | context | peds 7,001 · 27 | peds 100% | 0/3 | -- | queued |
| **ctx2/isolation_off** | context | peds 7,001 · 27 | peds 100% | 0/3 | -- | queued |
| **ctx2/age_scalar** | context | peds 7,001 · 27 | peds 100% | 0/3 | -- | queued |

`ctx2/ct_only`: 1 seed(s) died before update 3,200 and are excluded from the mean (`ctx2_ct_only_seed0`).

**ctx2/ct_only -- what it settled.** The architecture-only control: the context model with the Q-Former's context path switched off, so it is the published model under the new code path. Every other rung is a delta against this, not against V1.

`ctx2/c1_xattn`: 2 seed(s) died before update 3,200 and are excluded from the mean (`ctx2_c1_xattn_seed1`, `ctx2_c1_xattn_seed2`).

`ctx2/c1_full`: 1 seed(s) died before update 3,200 and are excluded from the mean (`ctx2_c1_full_seed2`).

## Joint context runs (the headline)

The winning rung retrained on the combined 49,545-volume split, alongside ct_only on the SAME split. The joint baseline is not redundant: CT-RATE outnumbers pediatrics six to one there, and a configuration that wins on 7,001 pediatric volumes is not thereby a winner in that regime.

| experiment | arch | cohort · classes | masks | seeds | val AUC | state |
|---|---|---|---|---|---|---|
| **joint/ct_only** | context | peds + CT-RATE 49,545 · 27 | peds only (71.6% once ctx3 masks land) | 0/3 | -- | queued |
| **joint/winner** | context | peds + CT-RATE 49,545 · 27 | peds only | 0/0 | -- | planned |

**joint/winner -- what it settled.** Waits on the ctx2 ladder to name a winner. Submitted as `sbatch --export=ALL,JOINT_RUNG=<winner> --array=0-5%3 pipeline/05_train/34_train_joint_ctx.sbatch`, which runs it and the joint baseline in one array.

## Masked ablation (ctx3)

The winner rerun with anatomy masks on the adult half too. Held separate from the ctx2 ladder on purpose: switching masks on at the same time as C1/C2 would make the gain unattributable.

| experiment | arch | cohort · classes | masks | seeds | val AUC | state |
|---|---|---|---|---|---|---|
| **ctx3/masked** | context | peds + CT-RATE 49,545 · 27 | 100% (in production) | 0/0 | -- | planned |

**ctx3/masked -- what it settled.** Blocked on mask generation, which is running. 31,570 CT-RATE training masks arrived from another machine already in target format, taking coverage from 14.1% (pediatrics alone) to 71.6%; the remaining 15,575 are being segmented here so the training set has one provenance. Worth finishing rather than training at 71.6% because the gap is not a random third -- sliced by patient id it runs 1.6 / 58 / 66 / 8 / 2 / 76 / 74 / 43 / 4 / 2 percent missing, the signature of a shard array that stopped part way, which would leave L_org fitted to a subsample selected by acquisition order.

## Standing conclusions

**The seed band is 0.0046**  Three seeds of an identical V1 configuration give sd = 0.0046 on pediatric validation macro AUC. A ladder delta below roughly 0.0092 is inside that band and is not a difference. This is measured on the V1 recipe, not the context recipe, and is the most honest estimate available rather than an exact one.

**Two broken regions, not a broken idea**  The -0.0375 routing penalty is not spread across the label set. Pneumothorax (0.695 to 0.317) and Bone lesion (0.616 to 0.347) fall BELOW CHANCE and carry about two thirds of it, in every cohort and every checkpoint. Below 0.5 is inversion, not misplaced attention -- looking in the wrong place gives 0.5. Both route to the two regions we derived rather than segmented. Excluding them the penalty is -0.0146 (CT-RATE) and -0.0159 (pediatrics), and 7 of the remaining 24 classes improve under routing. Fix those two regions before concluding anything about anatomy routing as a method.

**The routed read-out uses a pairing that was never trained**  Training aligns organ_lat to region sentences (L_org) and the class prompts to the global latent (L_cls). Evaluation scores organ_lat against the class prompts. Part of the residual routing penalty is therefore a read-out mismatch and not a property of anatomy routing.

**Validation AUC is not a result**  The val AUC columns are what the training loop selected checkpoints on. They are computed on the split the model stopped early against, so they are optimistically biased by construction. Only eval_matrix numbers are reportable, and only with the patient-clustered bootstrap CI beside them.

**A quiet run directory is not a finished run**  A run killed at update 400 leaves what a finished run leaves: a best.pt, a best_auc file, and silence. Nine of the first ladder's twenty-four tasks died that way, and averaging their early AUCs would report those arms as worse than they are. Runs below patience x val_every are marked DIED and excluded.

**Half of CT-RATE has an indication, not a quarter**  An early filter required three words, which suited the pediatric requisitions (median 24 words) and threw away the adult ones, which are often a single question. Corrected, CT-RATE indication coverage is 48.7%, not 24.1%.

