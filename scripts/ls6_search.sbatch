#!/bin/bash -l
#SBATCH -J kernel-search
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o search-%j.out
#SBATCH -e search-%j.err
set -euo pipefail
set +x
umask 077
unset OPENAI_API_KEY PYTHONHOME PYTHONPATH
work="$SLURM_SUBMIT_DIR"
guard="$work/guard-$SLURM_JOB_ID-${SLURM_RESTART_COUNT:-0}"
phase=${1:?search phase}
shift
source "$work/project/scripts/ls6_stage_runtime.sh"
case_id=$(python -c 'import sys
from kernel_study.campaign.jobs import batch_case
from kernel_study.tasks.kernelbench import experiment
print(batch_case(experiment(),sys.argv[1],list(map(int,sys.argv[2:]))))' "$phase" "$@")
status=0
python -m scripts.verify_starts "$guard" --start-deadline --case "$case_id" || status=$?
if [[ $status == 75 ]]; then
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
elif [[ $status != 0 ]]; then
    exit "$status"
fi
python scripts/probe_counters.py "$work/counters-$SLURM_JOB_ID.json"
source "$HOME/.openai_env"
export OPENAI_API_KEY
for job in "$@"; do
    python -c 'from kernel_study.cli import main; main()' run "$phase" --job "$job" \
        --preflight "$work/preflight" --output "$work/campaign" --counters "$work/counters-$SLURM_JOB_ID.json" \
        --model gpt-4.1-2025-04-14 --endpoint https://api.openai.com/v1 --input-price 2 --output-price 8
done
