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
guard="$work/guard-$SLURM_JOB_ID-${SLURM_RESTART_COUNT:-0}"
mkdir -p "$guard"
read -r required < "$work/runtime.bytes"
node_dir=
for parent in "${SLURM_TMPDIR:-/tmp}" "${TMPDIR:-/tmp}" /tmp; do
    [[ -d $parent && -w $parent ]] || continue
    fs=$(findmnt -n -T "$parent" -o FSTYPE)
    available=$(df -B1 --output=avail "$parent" | tail -1)
    printf '%s %s %d %d\n' "$parent" "$fs" "$available" "$required" >> "$guard/storage.txt"
    case "$fs" in xfs|ext[234]|btrfs|tmpfs) ;; *) continue ;; esac
    (( available > required )) || continue
    node_dir=$(mktemp -d "$parent/qualification-$SLURM_JOB_ID.XXXXXXXX")
    break
done
if [[ -z $node_dir ]]; then echo "No node-local space for the runtime" >&2; exit 1; fi
trap 'rm -rf -- "$node_dir"' EXIT
/usr/bin/time -f %e -o "$guard/unpack.seconds" tar --use-compress-program=zstd -xf "$work/runtime.tar.zst" -C "$node_dir"
revision=$(git -C "$work/project" rev-parse HEAD)
cd "$node_dir/project"
git fetch -q "$work/project" "$revision"
git checkout -q --detach "$revision"
test "$(git rev-parse HEAD)" = "$revision"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 KERNEL_BASELINE_UPSTREAM="$node_dir/upstream"
export PATH="$node_dir/venv/bin:$PATH" PYTHONPATH="$PWD:$KERNEL_BASELINE_UPSTREAM/KernelAgent"
printf '%s\n' "$revision" "$(command -v python)" "$PWD" "$KERNEL_BASELINE_UPSTREAM" > "$guard/runtime.txt"
hostname >> "$guard/runtime.txt"
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
