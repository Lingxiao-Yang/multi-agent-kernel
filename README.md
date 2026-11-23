# Budget-Matched Multi-Agent Kernel Optimization

This study compares the local four-stage single agent with single-worker
KernelAgent greedy. Both receive the same native Triton module, problem and verification script on six KernelBench Level 1 tasks. Development uses matvec (4),
softmax (23) and LayerNorm (40). Holdout uses matmul (2), GELU (26) and
RMSNorm (36). The longer-term question is how role specialization and curated
hardware knowledge affect optimization under a matched budget.
[Baseline study](docs/AGENT_BASELINE_PLAN.md) · [Project proposal](docs/proposal_final.pdf)

Both submit a module with a `kernel_function` wrapper and Triton kernels.
The wrapper may inspect tensor metadata, allocate output and workspace, and
launch kernels. Both agents pass the same static check before execution.
The judge checks the resulting tensors and input integrity on
held-out inputs, confirms improvements in two rounds of paired timing, and
falls back to a validated start when needed. For each task and seed, it times
the two finalists against one another. The primary ratio is local latency
divided by greedy latency. Both agents charge the same campaign ledger for
model calls and candidate attempts.

## Repository layout

| Path | Responsibility |
| --- | --- |
| `kernel_study/records.py`, `budget.py`, `providers.py` | Provenance, code hashes, budgets and model requests |
| `kernel_study/tasks/` | Pinned upstream tasks and six frozen native starting modules |
| `kernel_study/judge/` | Native module checks, correctness, timing and finalist selection |
| `kernel_study/agents/` | Local search, guidance, knowledge and the KernelAgent adapter |
| `kernel_study/campaign/` | Preflight, job registration, retry accounting and paired reports |
| `kernel_study/cli.py` | The `kernel-study` command |
| `configs/` and `knowledge/` | Registered cases, environment, protocol and knowledge cards |
| `tests/` | Tests by component, with real search and worker runs in `integration/` |
| `scripts/` | Counter probe, packaging, remote operations and artifact utilities |
| `docs/archive/` | Historical experiments and development notes |

Raw experiment outputs and historical reports live on the
[results branch](../../tree/results).

## Install and test

From a source checkout, use Python 3.10 or later and a virtual environment
with pip. An editable install lets the commands find the registered
configuration and device sources in the checkout.

```sh
python -m pip install -e '.[test]'
python -m pytest -q -m "not gpu and not upstream"
kernel-study --help
```

The `gpu` tests require a CUDA device, PyTorch and Triton. The `upstream` tests
use the pinned KernelBench, KernelAgent and tritonbench checkouts and their
runtime dependencies. Put the three checkouts under `KERNEL_BASELINE_UPSTREAM`
with those exact directory names. The registered revisions, A100 model,
driver and package versions are in the [experiment configuration](configs/agent_baseline.json).

## Run the public study

On the study GPU, set `KERNEL_BASELINE_PREFLIGHT` to the preflight directory.
Make the upstream Python packages available to the adapter and its worker
processes. The counter probe warms one tile of the public GELU start and
profiles one launch.

```sh
export PYTHONPATH="$KERNEL_BASELINE_UPSTREAM/KernelAgent:$KERNEL_BASELINE_UPSTREAM/tritonbench${PYTHONPATH:+:$PYTHONPATH}"
python scripts/probe_counters.py results/counters.json
```

Prepare one case at a time on the registered GPU. Each phase writes its
measurements under the case ID. Calibration measures self-comparisons. The hold phase repeats them on an
independent seed. The campaign uses a registered improvement threshold of 1.01.

```sh
for id in 4 23 40 2 26 36; do
    for phase in preflight stress calibrate hold; do
        kernel-study preflight "$phase" --case "$id" --output "$KERNEL_BASELINE_PREFLIGHT"
    done
done
```

For the full test suite, point `KERNEL_BASELINE_PREFLIGHT` at those records.
On a counter-denied GPU, `KERNEL_BASELINE_DENIED_CONTROL=1` enables the
separate greedy-manager interruption test. It uses a zero-dollar ledger.

```sh
python -m pytest -q
```

A model search requires verified counter access and the registered device.
Set `OPENAI_API_KEY`, `KERNEL_BASELINE_MODEL`, `KERNEL_BASELINE_INPUT_PRICE`
and `KERNEL_BASELINE_OUTPUT_PRICE` for the chosen snapshot. Prices are USD
per million tokens. Set `OPENAI_BASE_URL` when using a custom endpoint.
Each invocation runs one zero-based job index and records its cost. After
validating A100 timing, run the native single smoke (one candidate, four
calls) and the independent eight-round greedy reference search. Start a fresh
campaign directory for this protocol. Its opening balance includes the
archived smoke charges. Development has 12 jobs and holdout has 18.

```sh
kernel-study run smoke --job 0 --counters results/counters.json \
    --preflight "$KERNEL_BASELINE_PREFLIGHT" --output results/campaign-native
```

A recorded transport interruption allows one `--retry-transport` for the same
job. Once both agents finish for a task and seed, the campaign measures their
finalists directly. The report includes raw paired samples, fallbacks and the
charges from both attempts of a retried job.

```sh
kernel-study report development results/campaign-native
```

Use `--measure-pairs` on the registered GPU to fill missing finalist
measurements. With complete development measurements, `--freeze` registers
the search and evaluation code for holdout. An empty campaign reports
`incomplete 0`.
