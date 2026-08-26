# Ablations (Table 3)

Each row is the Stage-2 configuration with exactly one component disabled.
Everything else stays at `configs/stage2.env`.

| Table 3 row | Override |
|---|---|
| Full ARC-CT | none |
| − Stage-1 warm-start | `RAC_STAGE1_CKPT=` (empty: Kinetics init only) |
| − anatomy routing | `RAC_USE_ANATOMY_QFORMER=0 RAC_USE_QFORMER=1` |
| − label-Jaccard soft target | `RAC_FN_WEIGHT=0` (one-hot InfoNCE) |
| − learnable temperature | `RAC_USE_LEARNABLE_TEMP=0` |
| − per-organ alignment | `RAC_ALIGN_WEIGHT=0` |
| − per-token query supervision | `RAC_PERTOKEN_NCE_WEIGHT=0` |

Run each with `RAC_SEED=0,1,2` and report the mean ± s.d.
