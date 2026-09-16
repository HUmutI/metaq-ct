# bch-arc-ct — MetaQ-CT

Metadata-conditioned vision-language modeling for pediatric chest CT, built on
an anatomy-routed 3-D encoder derived from
[ARC-CT](https://github.com/arc-ct/arc-ct), and developed on a Boston
Children's Hospital cohort.

The premise is that a radiologist reads the clinical indication before the
images. The indication names the question, and age and sex shift disease priors
before a single voxel is examined. Standard CT vision-language models ignore all
of it and score a volume against generic pathology prompts. **MetaQ-CT** feeds
that context into pathology-specific query slots while keeping a separate
metadata-free representation, so the indication can act on the pathologies it
concerns and leave the rest untouched.

This is a **separate project**, not a branch. The schema, the region routing,
the preprocessing and several loss paths diverge from the published adult model.

## Architecture

A general Q-Former bank holds 10 anatomy queries, `C` pathology queries, and 2
global queries. Its pooled output `z_gen` is deliberately isolated from every
metadata path, which keeps a canonical image and report embedding available for
prompt inference and retrieval. Three components add the clinical context:

- **C1, pathology-specific conditioning.** Each pathology query has a
  conditioned twin that attends to the clinical context through a gated
  cross-attention residual and is then modulated by FiLM, initialized near
  identity. The conditioned bank holds `13 + 2C` slots: 59 for the 23-class
  pediatric task, 49 for the 18-class adult task.
- **C2, class-specific clinical relevance.** For each pathology, C2 compares the
  indication embedding with a frozen class-prompt embedding and produces a
  non-negative pooling weight. Its bias starts at −2, so C2 begins as a modest
  correction rather than suppressing incidental findings.
- **Fusion.** `z_final = W_f[z_gen; z_ind]` with `W_f` initialized to `[I 0]`.
  Because every conditioning path is zero-gated or identity-initialized, the
  conditioned readout equals the general readout at step zero. Everything the
  conditioned branch learns is a measured departure from the CT-only parent.

Prompt inference is the primary readout. A supervised one-layer `ctx_cls` head
exists as a labelled auxiliary ablation and is never conflated with it.

## Results

Pediatric, frozen MRN-disjoint `<18` test split, 1,273 studies, 23 findings,
three seeds:

| arm | macro AUROC | macro AUPRC |
|---|---|---|
| CT-only control | 0.7687 | 0.3592 |
| **MetaQ-CT, conditioned prompt** | **0.8268** | **0.4616** |

Blanking or shuffling the indication removes most of that gain, which is what
separates genuine visual-clinical fusion from extra model capacity. The
historical age-≥5 cohort replicates it, 0.7933 → 0.8396 over three seeds.

Adults are the contrast case. On CT-RATE the image-only branch reaches 0.8733
and the conditioned branch 0.8781 on one final checkpoint, but blanking the
indication costs only 0.27 points. Adult indications are frequently missing or
generic, so most of the adult gain is class-specific residual adaptation rather
than indication semantics. The papers report that distinction rather than
hiding it.

The repository also carries the first locally measured pediatric benchmark of
released chest-CT foundation models. CT-CLIP, CT-CLIP VocabFine, H-LIP,
BiomedCLIP, Merlin, MPS-CT and GreenRFM were each run with their own released
code, unmodified architecture and published readout. Adult rankings do not
transfer: two of them land near chance, and a 2-D encoder with a linear probe
beats every volumetric model reproduced here.

## Layout

```
arcct/              the model package, the working copy and not a snapshot
  schema.py           pediatric schema and 10-region routing
  conditioning.py     C1 cross-attention and FiLM, C2 relevance gate
  context_qformer.py  conditioned query bank
  dataset.py          air padding, HU floor, NaN labels, empty-region verdicts
  image_encoder.py    multi-scale feature pyramid
tools/              train_stage1.py, train_stage2.py, evaluate.py, figure scripts
configs/            stage2_peds.env (pediatric), stage2.env (adult reference)
pipeline/           the pediatric pipeline, one folder per stage, 00_setup … 08_analysis
benchmark/harness/  the released-model benchmark: per-model training, evaluation,
                    checkpoint sweeps, retrieval latents and table generators
paper/              pediatric.tex, adult.tex, main.tex, figures and generated tables
```

`paper/generated/` holds LaTeX fragments written directly by the evaluation
scripts. A table in the paper is never typed by hand, so it cannot drift from
the artifact it came from.

## What is deliberately NOT here

**No data, no weights, no clinical text.** Volumes, reports, extracted labels,
masks and checkpoints stay on the hospital cluster under `/temp_work/ch278233/`.

**The written reports are not committed either.** They quote patient report
sentences verbatim as evidence, and that text should not leave the cluster even
into a private repository. They live in `~/reports/` on e3.

Figure 1 and Figure 2 of `paper/pediatric.tex` are currently blank placeholders
while the real diagrams are drawn. The original TikZ source is kept in
`paper/figures/_pipeline_figures_original.tex`.

## Environments

Two, and they cannot be merged: `arcct` (torch 2.4.1, numpy 1.x, monai) for
everything except masks, and `totalseg` (numpy 2.x) for segmentation only.
TotalSegmentator requires numpy 2.x, which breaks torch's numpy 1.x C ABI.

## The two pipeline trees

`pipeline/` here and `~/pipeline` on the cluster are now reconciled: every file
present in both is identical. This repository is the superset and the
authoritative copy. It carries 48 scripts the cluster tree does not have, which
is the newer work.

The cluster tree keeps three files that must never be committed, the report
spreadsheet and its two label derivatives under `pipeline/lib/ctrate_map/`.
They contain verbatim patient report text and stay on the cluster.
