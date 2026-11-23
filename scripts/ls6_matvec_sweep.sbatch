#!/bin/bash -l
#SBATCH -J matvec-launch
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH -o sweep-%j.out
#SBATCH -e sweep-%j.err
set -euo pipefail
umask 077
unset OPENAI_API_KEY PYTHONHOME PYTHONPATH
work="$SLURM_SUBMIT_DIR"
archive="$SCRATCH/kernel-study-campaign/a100-matvec-20261122"
module load python/3.12.11
module load "$(cat "$archive/cuda-module")"
source "$archive/venv/bin/activate"
cd "$work/project"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 KERNEL_BASELINE_UPSTREAM="$archive/upstream"
export PYTHONPATH="$PWD:$KERNEL_BASELINE_UPSTREAM/KernelAgent"
if ! timeout 8m python -m scripts.check_matvec_feedback "$work/feedback-$SLURM_JOB_ID"; then
    echo 'Feedback check did not complete. See the verification records and stderr.' >&2
fi
exec python -m scripts.sweep_matvec "$work/results-$SLURM_JOB_ID"
