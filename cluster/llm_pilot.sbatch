#!/bin/bash
#SBATCH -J kernel-llm-pilot
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A ASC26098
#SBATCH -o logs/llm-pilot-%j.out
#SBATCH -e logs/llm-pilot-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module load python/3.12.11
module load cuda/12.8
source "$SCRATCH/kernel-study-venv/bin/activate"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export TRITON_CACHE_DIR="$SCRATCH/kernel-study-cache/triton"
: "${OPENAI_API_KEY:?Supply credential in environment, never a script or command argument}"
python -m kernel_study.cli doctor
python scripts/probe_counters.py "logs/counters-${SLURM_JOB_ID}.json"
python scripts/run_campaign.py --output "results/a100-llm-pilot-${SLURM_JOB_ID}" --dollar-cap 5
