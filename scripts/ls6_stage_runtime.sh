# Source this file from a batch job after setting work and guard.
module load python/3.12.11
module load "$(cat "$work/cuda-module")"
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
    node_dir=$(mktemp -d "$parent/kernel-study-$SLURM_JOB_ID.XXXXXXXX")
    break
done
if [[ -z $node_dir ]]; then echo 'No node-local space for the runtime' >&2; exit 1; fi
trap 'rm -rf -- "$node_dir"' EXIT
/usr/bin/time -f %e -o "$guard/unpack.seconds" tar --use-compress-program=zstd -xf "$work/runtime.tar.zst" -C "$node_dir" venv upstream
revision=$(git -C "$work/project" rev-parse HEAD)
git init -q "$node_dir/project"
cd "$node_dir/project"
git fetch -q --depth=1 "$work/project" "$revision"
git checkout -q --detach "$revision"
test "$(git rev-parse HEAD)" = "$revision"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 KERNEL_BASELINE_UPSTREAM="$node_dir/upstream"
export PATH="$node_dir/venv/bin:$PATH" PYTHONPATH="$PWD:$KERNEL_BASELINE_UPSTREAM/KernelAgent"
export TMPDIR="$node_dir/tmp" TRITON_CACHE_DIR="$node_dir/triton-cache"
export TORCHINDUCTOR_CACHE_DIR="$node_dir/inductor-cache" CUDA_CACHE_PATH="$node_dir/cuda-cache"
export TORCH_EXTENSIONS_DIR="$node_dir/torch-extensions" XDG_CACHE_HOME="$node_dir/cache"
mkdir -p "$TMPDIR"
printf '%s\n' "$revision" "$(command -v python)" "$PWD" "$KERNEL_BASELINE_UPSTREAM" > "$guard/runtime.txt"
hostname >> "$guard/runtime.txt"
