#!/bin/bash -l
#SBATCH -J matvec-single
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o matvec-single-%j.out
#SBATCH -e matvec-single-%j.err

set -euo pipefail
set +x
umask 077
cd "$SLURM_SUBMIT_DIR"
unset OPENAI_API_KEY PYTHONHOME PYTHONPATH
module load python/3.12.11
module load "$(cat cuda-module)"
source venv/bin/activate
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
export KERNEL_BASELINE_UPSTREAM="$PWD/upstream"
export PYTHONPATH="$PWD/project:$KERNEL_BASELINE_UPSTREAM/KernelAgent"

study() {
    python -c 'from kernel_study.cli import main; main()' "$@"
}

for phase in preflight stress; do
    study preflight "$phase" --case 4 --output "$PWD/preflight"
done
python "$PWD/project/scripts/probe_counters.py" "$PWD/counters.json"

source "$HOME/.openai_env"
export OPENAI_API_KEY
study run smoke --job 0 --preflight "$PWD/preflight" --output "$PWD/campaign-native" \
    --counters "$PWD/counters.json" --model gpt-4.1-2025-04-14 \
    --endpoint https://api.openai.com/v1 --input-price 2 --output-price 8
