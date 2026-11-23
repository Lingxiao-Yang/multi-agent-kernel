#!/bin/bash -l
#SBATCH -J row-sweep
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 16:00:00
#SBATCH -o rows-%j.out
#SBATCH -e rows-%j.err
set -euo pipefail
umask 077
unset OPENAI_API_KEY PYTHONHOME PYTHONPATH
work="$SLURM_SUBMIT_DIR"
case_id=${1:?public case number}
case "$case_id" in 23|40) ;; *) exit 2 ;; esac
archive="$SCRATCH/kernel-study-campaign/a100-matvec-20261122"
module load python/3.12.11
module load "$(cat "$archive/cuda-module")"
source "$archive/venv/bin/activate"
cd "$work/project"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 KERNEL_BASELINE_UPSTREAM="$archive/upstream"
export PYTHONPATH="$PWD:$KERNEL_BASELINE_UPSTREAM/KernelAgent"
out="$work/results-$SLURM_JOB_ID"
for phase in preflight stress calibrate hold; do
    python -c 'from kernel_study.cli import main; main()' preflight "$phase" --case "$case_id" --output "$out/preflight"
done
exec python -m scripts.sweep_rows --case "$case_id" "$out/sweep"
