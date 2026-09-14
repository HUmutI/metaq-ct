# CT-RATE retrieval audit — 7 September 2026

## Protocol

- Evaluation cohort: complete locally cached CT-RATE validation set (`n=3,002`).
- Checkpoints: three independently trained adult Refined-C1 seeds.
- Inference: mask-free; no retrieval-specific training or fine-tuning.
- `general`: canonical shared `z_gen` embedding. This is the fair readout for
  comparison with report-to-volume retrieval in ARC-CT and GreenRFM.
- `mixed`: exploratory `normalize(z_gen + z_ind)` readout. It uses each gallery
  examination's indication, age, and sex and is therefore not an equal-input
  comparison with published CT-only retrieval models.
- Values below are percentages and are mean ± sample SD over three seeds.

## Report-to-CT retrieval

| Method/readout | R@5 | R@10 | R@50 | R@100 |
|---|---:|---:|---:|---:|
| CT-CLIP (published) | 2.9 | 5.0 | 18.0 | 28.7 |
| Merlin (published) | 1.5 | 2.7 | 7.7 | 12.7 |
| BrgSA (published) | 5.8 | 10.1 | 28.6 | 42.0 |
| MPS-CT (published) | 10.0 | 16.3 | 39.0 | 52.2 |
| GreenRFM (published) | 9.5 | 16.0 | 39.9 | 54.3 |
| ARC-CT (published) | 11.4 | 18.0 | 40.4 | 52.1 |
| **Ours — Refined C1 checkpoint, canonical `z_gen`** | **13.07 ± 0.13** | **20.29 ± 0.30** | **44.88 ± 0.02** | **57.27 ± 0.11** |
| Ours — exploratory metadata-mixed `z_gen + z_ind` | 12.91 ± 0.16 | 20.28 ± 0.47 | 44.37 ± 0.22 | 56.53 ± 0.59 |

The canonical readout exceeds the strongest published value by 1.67 points at
R@5, 2.29 at R@10, 4.48 at R@50, and 2.97 at R@100. The metadata-mixed readout
does not improve retrieval: relative to `z_gen`, it changes these four metrics
by -0.16, -0.01, -0.51, and -0.74 points. This is consistent with Refined C1
having been trained for class-specific diagnostic correction rather than
global report alignment.

Because Refined C1 freezes its inherited base and enforces a one-way split that
keeps `z_gen` independent of the conditioned slots, the canonical retrieval
result demonstrates the strength of the inherited ARC-CT shared embedding. It
must not be claimed as evidence that C1 metadata conditioning itself improves
retrieval. A retrieval-specific metadata objective would be required for that
claim.

## CT-to-CT clinical-similarity audit

The released local evaluator computes mean label-set Jaccard among the top-K
retrieved scans:

| Readout | Mean Jaccard@5 | @10 | @50 |
|---|---:|---:|---:|
| Canonical `z_gen` | 42.883 ± 0.001 | 37.207 ± 0.030 | 31.349 ± 0.101 |
| Metadata-mixed | 42.925 ± 0.073 | 37.291 ± 0.062 | 31.386 ± 0.096 |

These values must not be placed directly beside ARC-CT Table 4's CT-to-CT
numbers yet. The paper labels that block `MAP@K`, while the released evaluator
calls and computes mean `Jaccard@K`; the definitions and reported magnitudes do
not reconcile. Report-to-CT Recall@K does not have this ambiguity and is the
current defensible headline retrieval comparison.

## Result artifacts

Metric JSON and latent archives are under:

`/temp_work/ch278233/eval_matrix/ctrate18_retrieval/`

The evaluation implementation is `tools/eval_retrieval.py`, and the submitted
array recipe is `pipeline/06_eval/48_eval_ctrate_retrieval_c1.sbatch`.

Three early array elements encountered a text-table writer error only after
their JSON and latent archive had been saved. The zero-bootstrap writer was
fixed during the run; the numerical artifacts were complete and unaffected.
