#!/bin/bash -l
#SBATCH -J matvec-warm
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:35:00
#SBATCH -o warm-%j.out
#SBATCH -e warm-%j.err
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
unset OPENAI_API_KEY PYTHONPATH PYTHONHOME
archive="$SCRATCH/kernel-study-campaign/a100-matvec-20261122"
module load python/3.12.11
module load "$(cat "$archive/cuda-module")"
source "$archive/venv/bin/activate"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
export KERNEL_BASELINE_UPSTREAM="$archive/upstream"
export PYTHONPATH="$PWD/project:$KERNEL_BASELINE_UPSTREAM/KernelAgent"
out="$PWD/results-$SLURM_JOB_ID"
mkdir "$out"
cd project
git rev-parse HEAD > "$out/revision"
for phase in calibrate hold; do
    python -c 'from kernel_study.cli import main; main()' preflight "$phase" --case 4 --output "$out/preflight"
done
python -m scripts.measure_matvec_finalists "$archive" "$out/finalists.json"
