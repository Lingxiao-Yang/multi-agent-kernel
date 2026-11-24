#!/bin/bash -l
#SBATCH -J verifier-timing
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:20:00
#SBATCH -o verifier-%j.out
#SBATCH -e verifier-%j.err
set -euo pipefail
umask 077
unset OPENAI_API_KEY PYTHONHOME PYTHONPATH
work="$SLURM_SUBMIT_DIR"
archive="$SCRATCH/kernel-study-campaign/a100-matvec-20261122"
out="$work/results-$SLURM_JOB_ID"
mkdir -p "$out"
module load python/3.12.11
module load "$(cat "$archive/cuda-module")"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
run_verifier() {
    local location=$1 venv=$2 project=$3 upstream=$4
    export KERNEL_BASELINE_UPSTREAM="$upstream"
    export PYTHONPATH="$project:$upstream/KernelAgent"
    PATH="$venv/bin:$PATH" "$venv/bin/python" "$work/tools/profile_start_verifier.py" "$out/$location"
}
run_verifier shared "$archive/venv" "$work/project" "$archive/upstream"
read -r required < "$work/runtime.bytes"
for parent in "${SLURM_TMPDIR:-/tmp}" "${TMPDIR:-/tmp}" /tmp; do
    [[ -d $parent && -w $parent ]] || continue
    fs=$(findmnt -n -T "$parent" -o FSTYPE)
    free=$(df -B1 --output=avail "$parent" | tail -1)
    printf '%s %s %d %d\n' "$parent" "$fs" "$free" "$required" >> "$out/storage.tsv"
    case "$fs" in xfs|ext[234]|btrfs|tmpfs) ;; *) continue ;; esac
    (( free > required )) || continue
    node_dir=$(mktemp -d "$parent/verifier-$SLURM_JOB_ID.XXXXXXXX")
    trap 'rm -rf -- "$node_dir"' EXIT
    /usr/bin/time -f %e -o "$out/unpack.seconds" tar --use-compress-program=zstd -xf "$work/runtime.tar.zst" -C "$node_dir"
    run_verifier local "$node_dir/venv" "$node_dir/project" "$node_dir/upstream"
    exit 0
done
echo 'No node-local directory with enough free space' > "$out/local-unavailable.txt"
