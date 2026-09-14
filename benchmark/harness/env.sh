# Shared environment for the MPS-CT / GreenRFM benchmark runs.
# Sourced by every sbatch here so a path is fixed in exactly one place.
export BM_HARNESS=/home/ch278233/BENCHMARK/harness
export BM_REPOS=/home/ch278233/BENCHMARK
export BM_DATA=/temp_work/ch278233/BENCHMARK_DATA
export BM_RUNS=/temp_work/ch278233/BENCHMARK_RUNS

# Compute nodes are not guaranteed outbound network: every pretrained weight
# must already be on disk.  r3d_18 was fetched to TORCH_HOME on the login node;
# CXR-BERT already lives in the shared HF cache.
export TORCH_HOME=/temp_work/ch278233/torch_cache
export HF_HOME=/temp_work/ch278233/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# CTCLIPTrainer.py:211 and GreenRFM's trainers call wandb.init unconditionally.
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# CT-RATE cohort artefacts (18-class, adult only)
export CTR=/temp_work/ch278233/CTRATE_ONLY
