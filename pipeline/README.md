# Pipeline stages

Each folder is one stage. Numbers are execution order, not strict dependencies.

| stage | what it does | key entry points |
|---|---|---|
| `00_setup` | conda/micromamba envs for arcct and TotalSegmentator | `setup_arcct_env.sh`, `ts_setup.sbatch` |
| `01_data` | pull volumes, resample to 1.5×1.5×3.0 mm, write npz, build splits | `00_download.sbatch`, `04_npz.sbatch`, `make_split.py`, `validate_npz.py` |
| `02_extract` | Qwen3-8B label + region extraction, fan-out, cross-cohort harmonisation | `10_extract.sbatch` (peds), `11_extract_ctrate.sbatch` (adult train), `14_extract_ctrate_valid.sbatch`, `harmonise_labels.py` |
| `03_masks` | TotalSegmentator, then the 10-region mask at 192×192×96 | `02_totalseg.sbatch`, `50_ts_ctrate_valid.sbatch`, `build_peds_masks.py` |
| `04_dataset` | assemble the joint dataset and gate it before training | `build_combined.py`, `13_prep_v2.sbatch` |
| `05_train` | the training runs | `20_train_peds.sbatch` … `32_train_combined47k_harm.sbatch` |
| `06_eval` | cross-cohort evaluation matrix and report generation | `41_eval_matrix.sbatch`, `60_eval_joint_pair.sbatch`, `morning_report.py` |
| `07_ops` | monitoring | `90_watchdog.sbatch`, `watchdog2.py`, `status.sh` |
| `08_analysis` | one-off measurements that informed design decisions | `region_survival.py`, `indication_analysis.py`, `age_analysis.py` |

## Two environments, deliberately separate

`micromamba/envs/arcct` — torch 2.4.1, numpy 1.x, monai. Everything except masks.

`/temp_work/ch278233/envs/totalseg` — TotalSegmentator, numpy 2.x. Mask
segmentation only. The two cannot be merged: TotalSegmentator needs numpy 2.x,
which breaks arcct's torch (built against the numpy 1.x C ABI). This is why mask
generation is two jobs — segmentation in one env, region building in the other.
