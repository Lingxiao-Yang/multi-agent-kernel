#!/bin/bash -l
#SBATCH -J qualification
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o qualification-%j.out
#SBATCH -e qualification-%j.err
set -euo pipefail
set +x
umask 077
unset OPENAI_API_KEY PYTHONHOME PYTHONPATH
work="$SLURM_SUBMIT_DIR"
archive="$SCRATCH/kernel-study-campaign/a100-matvec-20261122"
case "${1:?job name}" in
    matvec-11) case_id=4; phase=matvec-recheck; jobs=(0); report_args=(--require-seed 11) ;;
    matvec-22) case_id=4; phase=matvec-recheck; jobs=(1); report_args=(--require-seed 22) ;;
    softmax) case_id=23; phase=row-qualification; jobs=(0 1); report_args=() ;;
    layernorm) case_id=40; phase=row-qualification; jobs=(2 3); report_args=() ;;
    *) exit 2 ;;
esac
module load python/3.12.11
module load "$(cat "$archive/cuda-module")"
source "$archive/venv/bin/activate"
cd "$work/project"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 KERNEL_BASELINE_UPSTREAM="$archive/upstream"
export PYTHONPATH="$PWD:$KERNEL_BASELINE_UPSTREAM/KernelAgent"
guard="$work/guard-$SLURM_JOB_ID-${SLURM_RESTART_COUNT:-0}"
status=0
python -m scripts.check_matvec_feedback "$guard" --start-deadline --case "$case_id" || status=$?
if [[ $status == 75 ]]; then
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
elif [[ $status != 0 ]]; then
    exit "$status"
fi
python scripts/probe_counters.py "$work/counters-$SLURM_JOB_ID.json"
source "$HOME/.openai_env"
export OPENAI_API_KEY
cli=(-c 'from kernel_study.cli import main; main()')
for job in "${jobs[@]}"; do
    python "${cli[@]}" run "$phase" --job "$job" --preflight "$work/preflight" --output "$work/campaign" \
        --counters "$work/counters-$SLURM_JOB_ID.json" --model gpt-4.1-2025-04-14 \
        --endpoint https://api.openai.com/v1 --input-price 2 --output-price 8
done
unset OPENAI_API_KEY
flock "$work/report.lock" python "${cli[@]}" report "$phase" "$work/campaign" "${report_args[@]}"
