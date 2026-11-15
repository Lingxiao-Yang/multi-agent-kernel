#!/bin/bash
#SBATCH -J kernel-repair
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -A ASC26098
#SBATCH -o logs/repair-%j.out
#SBATCH -e logs/repair-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module load python/3.12.11
module load cuda/12.8
source "$SCRATCH/kernel-study-venv/bin/activate"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1
export TRITON_CACHE_DIR="$SCRATCH/kernel-study-cache/triton"
: "${OPENAI_API_KEY:?Supply credential through the environment}"
: "${ROUND_PHASE:?mini or strong required}"
python -m kernel_study.cli doctor
if [ "$ROUND_PHASE" = mini ]; then
  python scripts/source_preflight.py --output "results/repair-preflight-${SLURM_JOB_ID}" \
    --cases softmax_irregular,layernorm_irregular
  python scripts/run_repair_round.py --phase mini --output "results/repair-mini-${SLURM_JOB_ID}"
else
  : "${MINI_OUTPUT:?Set the prerequisite mini phase output}"
  python scripts/run_repair_round.py --phase strong --output "results/repair-strong-${SLURM_JOB_ID}" --mini-output "$MINI_OUTPUT"
fi
