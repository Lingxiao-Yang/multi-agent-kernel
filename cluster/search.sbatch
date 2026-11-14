#!/bin/bash
#SBATCH -J kernel-study
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -A ASC26098
#SBATCH -o logs/search-%j.out
#SBATCH -e logs/search-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module load python/3.12.11
source "$SCRATCH/kernel-study-venv/bin/activate"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export TRITON_CACHE_DIR="$SCRATCH/kernel-study-cache/triton"
# Explicit manifest index, model endpoint and a per-job financial limit are required.
# Never place API keys in this script or the sbatch command line.
: "${MANIFEST_INDEX:?set a manifest index}"
: "${OPENAI_API_KEY:?configure endpoint credential in the environment}"
python scripts/run_manifest_entry.py --index "$MANIFEST_INDEX" --output "results/primary" \
  --model "${STUDY_MODEL:?set a pinned model}" --endpoint "${STUDY_ENDPOINT:?set endpoint}" \
  --dollar-cap "${STUDY_JOB_DOLLAR_CAP:?set job cap}" \
  --input-price "${STUDY_INPUT_PRICE:?set price}" --output-price "${STUDY_OUTPUT_PRICE:?set price}"
