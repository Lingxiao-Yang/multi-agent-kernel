#!/bin/bash
#SBATCH -J kernel-source-pilot
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A ASC26098
#SBATCH -o logs/source-pilot-%j.out
#SBATCH -e logs/source-pilot-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module load python/3.12.11
module load cuda/12.8
source "$SCRATCH/kernel-study-venv/bin/activate"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1
export TRITON_CACHE_DIR="$SCRATCH/kernel-study-cache/triton"
: "${OPENAI_API_KEY:?Supply credential via environment}"
python -m kernel_study.cli doctor
python scripts/source_preflight.py --output "results/source-preflight-${SLURM_JOB_ID}"
python scripts/run_campaign.py --output "results/a100-source-pilot-${SLURM_JOB_ID}" \
  --search-space source-v1 --cases matmul_rectangular,softmax_irregular,layernorm_irregular,elementwise_irregular \
  --seeds 11 --attempts 2 --dollar-cap 1.25
