# bch-arc-ct

Pediatric adaptation of [ARC-CT](https://github.com/arc-ct/arc-ct) — an
anatomy-routed vision-language model for 3D chest CT — to a Boston Children's
Hospital cohort of 8,817 studies from 4,003 patients, under the constraint that
adult ability must not regress.

This is a **separate project**, not a branch: the schema, the region routing, the
preprocessing and several loss paths all diverge from the published adult model.

## What is here

```
arcct/             the model package — this is the working copy, not a snapshot
  schema.py          27-class pediatric schema + 10-region routing (new)
  dataset.py         air padding, HU floor, NaN labels, empty-region verdicts
  anatomy_qformer.py role-mask construction
  image_encoder.py   multi-scale feature pyramid (RAC_USE_MULTISCALE)
tools/             train_stage2.py, evaluate.py, and the rest
configs/           stage2_peds.env (pediatric), stage2.env (adult reference)
pipeline/          the pediatric pipeline, one folder per stage
  lib/             shared modules (LLM extractor, region maps, report sectioniser)
  00_setup … 08_analysis
```

**This repository is where the pediatric code lives and runs.** The upstream
`arc-ct` clone on the cluster has been restored to its published state and is
kept only as the reference that reproduces the adult 0.8524 gate. Every job
script under `pipeline/` points at this repository, not at upstream.

Divergence from upstream at the time of the fork: 257 lines across 7 files, plus
two new files (`arcct/schema.py`, `configs/stage2_peds.env`).

## What is deliberately NOT here

**No data, no weights, no clinical text.** Volumes, reports, extracted labels,
masks and checkpoints stay on the hospital cluster under `/temp_work/ch278233/`.
The `.gitignore` blocks `*.npz`, `*.nii.gz`, `*.pt`, `*.csv`, `*.jsonl`, `*.tsv`
so they cannot be committed by accident.

**The written reports are not committed either.** They quote patient report
sentences verbatim as evidence, and that text should not leave the cluster even
into a private repository. They live in `~/reports/` on e3.

## Current state

Schema: 18 adult classes (byte-identical to CT-RATE) + 9 pediatric additions =
27; 10 anatomical regions re-derived for pediatric anatomy; 39 Q-Former queries.

Best results to date, one evaluator, three cohorts:

| model | pediatric n=1,773 | CT-RATE n=3,002 | adult loss¹ |
|---|---|---|---|
| pediatric-only (best) | **0.8019** | 0.7925 | −0.0488 |
| joint 47k, mixed labels | 0.7964 | 0.8228 | −0.0174 |
| joint 47k, one labeller | 0.7886 | **0.8271** | **−0.0132** |
| published adult model | — | — | (0.8574) |

¹ same 18 classes against the published adult model.

## Environments

Two, and they cannot be merged: `arcct` (torch 2.4.1, numpy 1.x, monai) for
everything except masks, and `totalseg` (numpy 2.x) for segmentation only —
TotalSegmentator requires numpy 2.x, which breaks torch's numpy 1.x C ABI.
