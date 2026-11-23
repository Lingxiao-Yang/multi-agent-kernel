#!/bin/bash -l
#SBATCH -J matvec-idle
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -o idle-%j.out
#SBATCH -e idle-%j.err

set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
unset OPENAI_API_KEY PYTHONHOME PYTHONPATH
module load python/3.12.11
module load "$(cat "$MATVEC_ARCHIVE/cuda-module")"
source "$MATVEC_ARCHIVE/venv/bin/activate"
export KERNEL_BASELINE_UPSTREAM="$MATVEC_ARCHIVE/upstream"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 PYTHONPATH="$MATVEC_ARCHIVE/project"
python measure_matvec_idle.py "$MATVEC_ARCHIVE" "results-$SLURM_JOB_ID"
