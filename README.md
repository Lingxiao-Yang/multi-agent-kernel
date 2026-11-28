# Budget-Matched Multi-Agent Kernel Optimization

This repository compares two ways to optimize a Triton kernel with the same model, starting program, hardware evidence and search budget. In `single`, one general optimizer analyzes, advises, implements and reviews. In `multi`, four specialist prompts perform those same steps in the same order. Only the implementation response contains source code. KernelAgent's single-worker `greedy` provides an external reference. [Study protocol](docs/PROTOCOL.md) · [Baseline qualification](docs/AGENT_BASELINE_PLAN.md)

The local search and the upstream adapter both submit a full Python module with a `kernel_function` entry point. Its wrapper reads tensor metadata, allocates output and workspace, launches Triton kernels and returns the result. A common judge screens the module, checks the entire output and times each eligible candidate against its incumbent. It replays proposals on fresh inputs before selecting the finalist. A search that retains the start contributes its measured latency to the comparison.

## Repository layout

| Path | Contents |
| --- | --- |
| `kernel_study/tasks/` | Pinned KernelBench tasks, inputs, verifier scripts and six frozen native starts in `baselines/` |
| `kernel_study/agents/` | Four-stage local search, stage prompts, hardware evidence, optional knowledge cards and KernelAgent adapter |
| `kernel_study/judge/` | Native source checks, GPU worker, correctness, paired timing and finalist selection |
| `kernel_study/campaign/` | Preflight, job schedules, shared starting evidence, launch sweeps and paired reports |
| `kernel_study/records.py`, `budget.py`, `providers.py` | Source hashes, experiment records, budget reservations and actual model usage |
| `configs/`, `knowledge/` | Case shapes and tolerances, device and runtime settings, search limits, greedy settings and knowledge cards |
| `scripts/` | Counter probe, frozen-start verifier, launch sweeps and Lonestar6 batch entry points |
| `tests/` | Component tests and integration tests that exercise the public search and upstream adapter |
| `docs/` | Study protocol, baseline qualification and [archived plans and experiment notes](docs/archive/) |

## Install and test

Use an editable checkout with Python 3.10 or later. Put the pinned KernelAgent, KernelBench and tritonbench repositories under `KERNEL_BASELINE_UPSTREAM` using those directory names. The A100 experiments use Lonestar6's NVIDIA A100 PCIe 40GB, driver 570.195.03, PyTorch 2.12.1+cu126, Triton 3.7.1 and CUDA 12.6. Paired searches and their finalist comparison use the same physical GPU. [Configuration and upstream revisions](configs/agent_baseline.json)

```sh
python -m pip install -e '.[test]'
python -m pytest -q -m 'not gpu and not upstream'
kernel-study --help
```

For the full suite, install the pinned GPU and upstream runtime dependencies, export the two paths below and point `KERNEL_BASELINE_PREFLIGHT` at the prepared case directories. The counter probe and the six-start verifier each take an output path.

```sh
export PYTHONPATH="$KERNEL_BASELINE_UPSTREAM/KernelAgent:$KERNEL_BASELINE_UPSTREAM/tritonbench${PYTHONPATH:+:$PYTHONPATH}"
python scripts/probe_counters.py results/counters.json
python -m scripts.verify_starts results/starts --all-starts
python -m pytest -q
```

## Prepare and compare searches

Prepare each case before running a search. The four preflight commands measure the start against eager and `torch.compile`, check pressure inputs, and count false wins in two sets of 20 self-comparisons at the fixed 1.01 threshold. The directories contain the starting source, its hash, the device and the raw timing samples.

```sh
for case_id in 4 23 40 2 26 36; do
    for phase in preflight stress calibrate hold; do
        kernel-study preflight "$phase" --case "$case_id" --output "$KERNEL_BASELINE_PREFLIGHT"
    done
done
python -m scripts.sweep_matmul results/matmul-sweep
python -m scripts.sweep_matvec results/matvec-sweep
python -m scripts.sweep_rows --case 23 results/softmax-sweep
```

The formal role study uses matvec (4), softmax (23), LayerNorm (40) and FP32 matrix multiplication (2), each with its original frozen start. Five seeds per variant give 20 matched `single`–`multi` pairs, or 40 searches without knowledge cards.

The role study reads `role-study.json` from a fresh private campaign directory. Register the full merged mainline commit in `revision`, `[4, 23, 40, 2]` in `cases`, the five new integer search seeds in `seeds` and the knowledge-card choice as a boolean in `knowledge`. One job index identifies one case, seed and variant. Successive pairs alternate which variant goes first. The same case and seed share one starting Nsight Compute measurement. Each search has four candidate slots, at most 16 calls and a $1 allowance. Check that the registered schedule fits the campaign balance before submitting it.

On Lonestar6, the submit directory contains the registered checkout as `project/`, prepared cases under `preflight/`, the registration under `campaign/`, the CUDA module name in `cuda-module`, and a zstd runtime archive `runtime.tar.zst` containing `venv/` and `upstream/`. Put its expanded size in bytes in `runtime.bytes`. The job stages the checkout and runtime on node-local storage, verifies the frozen start and probes counters before loading the model credential from the account's private `~/.openai_env` (mode 600). Submit the indices for a pair in the same allocation.

```sh
kernel-study check-budget roles campaign
sbatch -A "$SLURM_ACCOUNT" project/scripts/ls6_search.sbatch roles 0 1
sbatch -A "$SLURM_ACCOUNT" project/scripts/ls6_sweep.sbatch 2
kernel-study report roles campaign
```

For a direct invocation, set `KERNEL_BASELINE_MODEL`, `KERNEL_BASELINE_INPUT_PRICE` and `KERNEL_BASELINE_OUTPUT_PRICE` to the registered snapshot and token prices in USD per million. Supply `OPENAI_API_KEY` in the private search environment.

```sh
kernel-study run roles --job 0 --preflight "$KERNEL_BASELINE_PREFLIGHT" \
    --counters results/counters.json --output results/campaign
kernel-study report roles results/campaign
```

The report checks the requests, replies and settlements in each search directory. It records confirmed final gains and fallbacks per task, direct finalist latencies, performance against launch-sweep references, matched candidate and spend curves, and the costs of interrupted attempts. The Slurm job retries a transport interruption once using the unused part of the search's allowance. A direct invocation can use `--retry-transport` to retry a recorded transport interruption on the registered GPU. Use `report roles --measure-pairs` on that GPU to complete missing finalist timings without a model call. [Historical results](../../tree/results)
