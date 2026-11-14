#!/bin/bash
#SBATCH -J kernel-pilot
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -A ASC26098
#SBATCH -o logs/pilot-%j.out
#SBATCH -e logs/pilot-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module load python/3.12.11
module load cuda/12.8
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export TRITON_CACHE_DIR="$SCRATCH/kernel-study-cache/triton"
export PIP_CACHE_DIR="$SCRATCH/kernel-study-cache/pip"
mkdir -p "$TRITON_CACHE_DIR" "$PIP_CACHE_DIR" logs
python3 -m venv "$SCRATCH/kernel-study-venv"
source "$SCRATCH/kernel-study-venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
# NumPy 2.5.x falsely rejects the EPYC 7763 compute nodes (X86_V2 check, job 3582290).
python -m pip install "numpy>=1.26,<2.3" matplotlib pytest
python -c "import numpy, torch; print('numpy', numpy.__version__, 'torch', torch.__version__, torch.cuda.get_device_name(0))"
python -m pip freeze > "logs/environment-${SLURM_JOB_ID}.txt"
nvidia-smi > "logs/gpu-${SLURM_JOB_ID}.txt"
python -m pytest -q
python -m kernel_study.cli doctor
python scripts/probe_counters.py "logs/counters-${SLURM_JOB_ID}.json"
python -m kernel_study.cli pilot --output "results/a100-pilot-${SLURM_JOB_ID}"
