#!/bin/bash -l
#SBATCH -J matvec-dev
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o matvec-%j.out
#SBATCH -e matvec-%j.err
set -euo pipefail
set +x
umask 077
unset OPENAI_API_KEY PYTHONHOME PYTHONPATH
work="$SLURM_SUBMIT_DIR"
archive="$SCRATCH/kernel-study-campaign/a100-matvec-20261122"
preflight="$SCRATCH/kernel-study-campaign/a100-native-matvec-b68a1e8/preflight"
first=${1:?first job index}
case "$first" in 0|2) ;; *) exit 2 ;; esac
module load python/3.12.11
module load "$(cat "$archive/cuda-module")"
source "$archive/venv/bin/activate"
cd "$work/project"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 KERNEL_BASELINE_UPSTREAM="$archive/upstream"
export PYTHONPATH="$PWD:$KERNEL_BASELINE_UPSTREAM/KernelAgent"
guard="$work/guard-$SLURM_JOB_ID-${SLURM_RESTART_COUNT:-0}"
status=0
python -m scripts.check_matvec_feedback "$guard" --start-deadline || status=$?
if [[ $status == 75 ]]; then
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
elif [[ $status != 0 ]]; then
    exit "$status"
fi
python scripts/probe_counters.py "$work/counters-$SLURM_JOB_ID.json"
source "$HOME/.openai_env"
export OPENAI_API_KEY
for job in "$first" "$((first+1))"; do
    python -c 'from kernel_study.cli import main; main()' run matvec-development --job "$job" \
        --preflight "$preflight" --output "$work/campaign" --counters "$work/counters-$SLURM_JOB_ID.json" \
        --model gpt-4.1-2025-04-14 --endpoint https://api.openai.com/v1 --input-price 2 --output-price 8
done
unset OPENAI_API_KEY
flock "$work/report.lock" python -c 'from kernel_study.cli import main; main()' report matvec-development "$work/campaign"
