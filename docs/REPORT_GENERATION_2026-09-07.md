# CT report generation track — 2026-09-07

## Scientific question

Does the metadata-conditioned Q-Former produce more accurate and complete CT
reports than its exact CT-only parent, and does performance degrade when the
clinical indication is blank or assigned to the wrong scan?

## Matched experiment

Both arms use CT-RATE only, seed 1, the same frozen 3-D encoder/Q-Former parent,
the same reports and patient-level train/development split, Qwen3-8B, LoRA
configuration, 32-token resampler, optimizer, and decoding configuration.

- **CT-only:** first 30 invariant general Q-Former tokens.
- **Refined C1:** all 49 tokens (30 general, 18 indication-conditioned
  pathology twins, and one clinical/global token).

The language decoder is not given indication, age, or sex as text. Metadata can
affect the main arm only through C1. A four-layer trainable cross-attention
resampler maps either input bank to exactly 32 Qwen prefix tokens, preventing a
decoder-capacity advantage for C1.

Training combines report teacher forcing with an auxiliary, class-balanced
18-pathology BCE loss on the resampled representation. The auxiliary loss is
identical in both arms and is intended to reduce clinically important omissions
and hallucinations.

## Evaluation

Model selection uses development loss only. The official CT-RATE validation set
will be touched after selection. Report evaluation includes:

- F1-RadGraph entity, entity/relation, and complete-graph scores;
- CT-specific factual precision and factual recall (RadFact-CT, second pass);
- BLEU-1 through BLEU-4 and ROUGE-L as secondary language-overlap metrics;
- Refined-C1 inference with true, blank, and shuffled indications;
- paired per-study bootstrap confidence intervals for C1 minus CT-only;
- blinded radiologist review of a fixed error-stratified sample after automated
  selection.

Report text and generated samples remain in restricted local result paths; logs
and presentation files receive aggregate metrics only.

## Implementation and jobs

- Token cache: `tools/cache_report_tokens.py`
- Model: `arcct/report_generator.py`
- Training: `tools/train_report_generator.py`
- Generation/text metrics: `tools/eval_report_generator.py`
- Clinical graph metric: `tools/eval_report_radgraph.py`
- SLURM: `pipeline/05_train/50_cache_report_tokens.sbatch`,
  `pipeline/05_train/51_train_report_generator.sbatch`, and
  `pipeline/06_eval/50_eval_report_generator.sbatch`

Long production jobs are submitted only after the end-to-end smoke test loads
the decoder, completes backward/optimizer steps, reloads the saved adapter, and
generates valid report text.
