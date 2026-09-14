# Pediatric context heatmaps

`tools/visualize_peds_context_heatmaps.py` adapts the internal
`HUmutI/heatmap_codes` visualization approach (audited at commit
`cde75a956aa719c89cbbf3624df5dd95a497f402`) to the 23-class pediatric
`ContextQFormer`. The upstream scripts must not be run directly on this model:
they assume the legacy 18-class query indices and do not construct indication,
age, or sex context.

Each panel compares, on one common axial slice:

1. the CT;
2. metadata-isolated general pathology-query cross-attention;
3. conditioned attention with correct indication, age, and sex;
4. conditioned attention with the indication blanked while retaining age/sex;
5. correct-minus-blank attention; and
6. Grad-CAM of the final conditioned prompt margin.

The primary checkpoint uses hard anatomy routing. White contours show the
allowed region, and the attention panels therefore describe evidence within
that region. They are not an unguided localization benchmark. Cross-attention
comes from the last Q-Former block and is averaged across its eight heads.
Grad-CAM is differentiated through the exact final prompt readout. Displayed
probabilities use the ordinary evaluator path; the script also asserts that
requesting attention weights changes `z_final` by no more than `1e-4`.

The launcher is:

```bash
sbatch pipeline/08_analysis/60_visualize_peds_context_heatmaps.sbatch
```

An explicit case can be requested without editing code:

```bash
sbatch --export=ALL,HEATMAP_CASE=ped_XXXXX_1.nii.gz:Pneumothorax \
  pipeline/08_analysis/60_visualize_peds_context_heatmaps.sbatch
```

The default candidate gallery chooses one high-confidence ground-truth-positive
test case for each of five prespecified pathologies, using distinct patients.
This is qualitative selection and must be described as such; it is not evidence
of localization accuracy. Before publication, a radiologist should review the
chosen slice and whether the highlighted region corresponds to the finding.

Outputs live under `/temp_work/ch278233/PEDS_HEATMAPS`. They contain derived
pediatric imaging and must not be committed or copied outside the governed BCH
environment. Figures omit accession and clinical text by default. The private
manifest retains the de-identified accession needed for radiologist review but
never stores report or indication text.

## Completed validation

- Smoke job `24104963`: completed successfully on one pneumothorax case.
- Five-case job `24104964`: completed successfully for pneumothorax, pleural
  effusion, bronchiectasis, pulmonary metastases, and bone lesion/fracture.
- Numerically guarded rerun `24105330` / `24105331`: completed successfully;
  maximum attention-path versus evaluator-path embedding difference was
  `7.2e-7`, well below the registered `1e-4` failure threshold.
- Verified output:
  `/temp_work/ch278233/PEDS_HEATMAPS/context_qformer_seed0_verified`.
