#!/bin/bash -l
#SBATCH -J kernel-sweep
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
guard="$work/runtime-$SLURM_JOB_ID"
case_id=${1:?public case number}
case "$case_id" in
    2) sweep=(scripts.sweep_matmul) ;;
    4) sweep=(scripts.sweep_matvec) ;;
    23|40) sweep=(scripts.sweep_rows --case "$case_id") ;;
    *) exit 2 ;;
esac
source "$work/project/scripts/ls6_stage_runtime.sh"
out="$work/results-$SLURM_JOB_ID"
for stage in preflight stress calibrate hold; do
    python -c 'from kernel_study.cli import main; main()' preflight "$stage" --case "$case_id" --output "$out/preflight"
done
python -m "${sweep[@]}" "$out/sweep"
