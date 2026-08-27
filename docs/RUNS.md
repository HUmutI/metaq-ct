# Which run is which

Run directories live in `/temp_work/ch278233/runs/` and are named
`<ladder>_<rung>_seed<n>`. The `<ladder>` prefix is a *generation*: one sweep of
the whole ablation ladder under one set of training conditions. When something
about the recipe changes in a way that makes old runs incomparable, the
generation increments and the old runs stay on disk, because a superseded sweep
is usually still a measurement of something.

Read a table with:

    python pipeline/08_analysis/ladder_report.py --tag ctx2

## Generations

### `ctx` — architecture only, conditioning inert
*Trained 2026-08-26 night. Peds-only. 14 of 24 cells completed before it was
stopped.*

Ran with `RAC_CTX_BETA_INIT=-6` and the context modules on the same learning rate
as the warm-started ResNet. Both were mistakes and the diagnostics show it:
`beta` moved from 0.00248 to 0.0026 over 3,600 updates, and `|W_ind|/|W_gen|`
plateaued at 0.027 — the conditioned half of the fusion carried under 3% of the
general half's norm. `c1_xattn`, `c1_full` and `c2` came out identical seed by
seed because everything separating them lives downstream of gates that never
opened.

**This is not a failed sweep, it is the control.** With the indication pathway
inert, what it measures is the *architecture on its own*: the 67-slot bank, the
extra clinical query row, and the learned fusion, with no indication effect
mixed in. `ctx2`'s gain has to beat this before any of it can be called an
indication effect.

    ct_only 0.7883 · concat +0.0045 · c1_xattn +0.0067 · c1_full +0.0097 · c2 +0.0106
    (seed band from Phase 0/8: sigma 0.0046, so a delta under 0.0092 is noise)

### `ctx2` — the real conditioning ladder
*Started 2026-08-27 morning. Peds-only, 8 rungs x 3 seeds.*

Same architecture, three changes: `RAC_CTX_BETA_INIT=-2` so beta's gradient is
alive (at -6 the softplus derivative is 0.0025 and beta cannot move),
`RAC_CTX_LR_MULT=20` on the zero-initialised context modules, and early-stopping
patience 8 instead of 5 so the gates have room to open. Applied to every rung
including `ct_only`, so the sweep is internally comparable.

Its job is to pick the configuration, not to be the model anyone is shown.

### `ctx3` — reserved: CT-RATE masks
Not started. When the 47k CT-RATE TotalSegmentator masks land, this is the sweep
that turns anatomy routing on for the adult half too. It is a separate
generation because it changes the training signal for 86% of the combined
corpus, and folding it in silently would make the gain unattributable.

### `joint_*` — the model that gets presented
Not started. The ladders above train on the 7,001-volume pediatric split, which
is the cheap way to choose an architecture. The model an advisor or a reviewer
sees has to hold BOTH cohorts, so the winning configuration is retrained on the
49,545-volume combined split and evaluated on each cohort separately.

## Rungs

| rung | what it is |
|---|---|
| `ct_only` | today's ARC-CT, context flag off — the baseline |
| `concat` | the wider bank and the fusion, no conditioning: the cheap fusion the cited study found could win at this data scale |
| `c1_xattn` | + cross-attention conditioning, FiLM disabled |
| `c1_full` | + FiLM channel gating — C1 complete |
| `c2` | + relevance weighting — C1+C2, the headline configuration |
| `fusion_gated` | `Z_gen + g*Z_ind` instead of the concat fusion |
| `isolation_off` | the self-attention group mask REMOVED. Deliberately unsafe: it measures the leak. Never a reportable model — if it leads, the reason is the leak |
| `age_scalar` | scalar age instead of the ordinal band |

## Reference runs (pre-existing)

| directory | what |
|---|---|
| `peds_finetune`, `peds_finetune_v1_seed1`, `_seed2` | the V1 recipe at three seeds — this is where the seed band sigma comes from |
| `peds_finetune_v2` … `v5` | earlier pediatric recipe iterations |
| `combined_16k`, `combined_47k`, `combined_47k_harmonised` | joint peds+CT-RATE training; `c47harm` is the current best joint model and the thing a new joint run must beat |
