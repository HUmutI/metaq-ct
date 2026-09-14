#!/bin/bash
#SBATCH --job-name=pbRetrScore
#SBATCH --partition=bch-compute,bch-compute-pe
#SBATCH --array=0-18
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/temp_work/ch278233/PEDS_BENCH/logs/retr_score_%A_%a.out
# Scores every retrieval latent file with the SAME metric code and a 1000-draw
# patient-level bootstrap. One task per file; missing files are skipped and named.
set -uo pipefail
R=/temp_work/ch278233/PEDS_BENCH/retrieval; O=$R/results; mkdir -p $O
M=/home/ch278233/BENCHMARK/harness/peds_bench/retrieval_metrics.py
E=/temp_work/ch278233/eval_matrix/peds23_retrieval
FILES=(
  "$E/ct_only_seed0_general/retrieval_latents_ct_only_seed0_general.npz|ours_ct_only_seed0"
  "$E/ct_only_seed1_general/retrieval_latents_ct_only_seed1_general.npz|ours_ct_only_seed1"
  "$E/ct_only_seed2_general/retrieval_latents_ct_only_seed2_general.npz|ours_ct_only_seed2"
  "$E/refined_c1_seed0_general/retrieval_latents_refined_c1_seed0_general.npz|ours_refined_c1_seed0"
  "$E/refined_c1_seed1_general/retrieval_latents_refined_c1_seed1_general.npz|ours_refined_c1_seed1"
  "$E/refined_c1_seed2_general/retrieval_latents_refined_c1_seed2_general.npz|ours_refined_c1_seed2"
  "$R/ctclip_valid.npz|ctclip"
  "$R/ctclip_vocabfine_valid.npz|ctclip_vocabfine"
  "$R/greenrfm_valid.npz|greenrfm"
  "$R/mpsct_valid.npz|mpsct"
  "$R/hlip_valid.npz|hlip"
  "$R/biomedclip_valid.npz|biomedclip"
  "$R/merlin_valid.npz|merlin"
  "$E/refined_c1_seed0_mixed/retrieval_latents_refined_c1_seed0_mixed.npz|ours_refined_c1_mixed_seed0"
  "$E/refined_c1_seed1_mixed/retrieval_latents_refined_c1_seed1_mixed.npz|ours_refined_c1_mixed_seed1"
  "$E/refined_c1_seed2_mixed/retrieval_latents_refined_c1_seed2_mixed.npz|ours_refined_c1_mixed_seed2"
  "$E/refined_c1_seed0_conditioned/retrieval_latents_refined_c1_seed0_conditioned.npz|ours_refined_c1_conditioned_seed0"
  "$E/refined_c1_seed1_conditioned/retrieval_latents_refined_c1_seed1_conditioned.npz|ours_refined_c1_conditioned_seed1"
  "$E/refined_c1_seed2_conditioned/retrieval_latents_refined_c1_seed2_conditioned.npz|ours_refined_c1_conditioned_seed2"
)
IFS='|' read -r F NAME <<< "${FILES[$SLURM_ARRAY_TASK_ID]}"
[ -f "$F" ] || { echo "SKIP (henuz yok): $NAME -> $F"; exit 0; }
[ -f "$O/$NAME.json" ] && { echo "SKIP (zaten skorlanmis): $NAME"; exit 0; }
echo "=== $NAME  $(date) ==="
$M --latents "$F" --name "$NAME" --out "$O/$NAME.json" --bootstrap 1000
echo "=== rc=$? ==="
