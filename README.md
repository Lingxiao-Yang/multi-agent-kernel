# Budget-Matched Multi-Agent Kernel Optimization

This repository implements the study described in the
[project proposal](docs/proposal_final.pdf). The study asks whether LLM role
specialization and curated hardware knowledge improve Triton kernel
optimization when every variant gets the same attempt and call budget.

The current baseline study compares the local four-stage single agent with
single-worker KernelAgent greedy on six KernelBench tasks. The
[baseline plan](docs/AGENT_BASELINE_PLAN.md) specifies the shared starts,
timing controls and acceptance criteria.

Engineering controls and local-GPU measurements are kept separate from the
planned A100 LLM study, and they are not pooled with it.

## Repository layout

| Branch | Contents |
| --- | --- |
| `main` | Implementation, configurations, tests, cluster launchers and documentation |
| [`results`](../../tree/results) | Raw experiment outputs, reports, preliminary plots and standalone plotting scripts |

Every result directory records source hashes and a manifest, so each run can
be traced to the code on `main`.

| Path | Purpose |
| --- | --- |
| `kernel_study/` | Search controller, LLM providers, benchmark harness, metrics and analysis |
| `configs/` | Case definitions, protocol parameters and the 144-search primary manifest |
| `knowledge/cards.json` | Twelve sourced hardware knowledge cards |
| `scripts/` | Pilot, campaign, probing and reporting entry points |
| `cluster/` | Slurm launchers for TACC Lonestar6 |
| `tests/` | CPU integrity tests and CUDA integration tests |
| `docs/` | Proposal, protocol and development log |

## What is implemented

- **Benchmark suite.** Twelve cases cover matrix multiplication, row softmax,
  layer normalization and bias + ReLU. Each family has three shapes, including
  irregular boundaries.
- **Search space.** Model runs default to agent-authored Triton source and
  declarative launch sequences, starting from baseline code only. Predefined
  optimized kernels are reserved for the explicit `parameter-control` mode.
  See [the source-generation revision](docs/SOURCE_EXPERIMENTS.md).
- **Four variants.** `single`, `single_knowledge`, `multi` and
  `multi_knowledge` share a four-call cadence and greedy retention. Only the
  multi-agent variants use specialized role prompts. The single agent receives
  equivalent analysis and revision calls with a general prompt.
- **Knowledge bank.** Twelve sourced cards with deterministic selection and
  category ablation.
- **Measurement harness.**
  - A trusted PyTorch reference and fresh inputs for final validation.
  - Edge-case inputs and checks that inputs are not modified.
  - Paired AB/BA timing, with a second measurement to confirm each gain.
- **Provenance.** Every run records:
  - prompts and responses as JSONL;
  - candidate source and compiler errors;
  - correctness results and latency samples;
  - budgets, hardware/software metadata and SHA-256 hashes.
- **Budget control.**
  - Fallback to earlier candidates when final validation fails.
  - Accounting for failed and repaired attempts.
  - A token safety cap, explicit model prices and a bounded inference spend.
- **Analysis.**
  - Paired factorial effects and case-cluster bootstrap confidence intervals.
  - Performance-versus-budget curves and CSV summaries.

## Quick start

GPU measurements require Linux with a CUDA GPU. The CPU-only integrity tests
also run on Windows. Do not benchmark on a cluster login node.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[gpu,test]'
python -m pytest -q
python -m kernel_study.cli doctor
python -m kernel_study.cli pilot --output results/my-pilot
```

Use `--quick` only during development. Each run must write to a new output
directory, so existing runs are never overwritten. The baseline/transform
pilot uses no LLM and cannot measure a role or knowledge effect.

A scripted-provider control run:

```bash
python -m kernel_study.cli run --case softmax_irregular \
  --variant multi_knowledge --seed 11 --attempts 2 \
  --provider scripted --output results/control-example
```

### Real-model runs

Set `OPENAI_API_KEY` in the shell environment. The endpoint must implement the
Chat Completions schema and support:

- reported token usage;
- JSON output;
- a seed and a sampling temperature.

Use one fixed model snapshot for a whole campaign.

```bash
python -m kernel_study.cli run --case softmax_irregular \
  --variant multi_knowledge --seed 11 --provider llm \
  --model gpt-4.1-mini-2025-04-14 --endpoint https://api.openai.com/v1 \
  --input-price 0.40 --output-price 1.60 --dollar-cap 1 \
  --output results/llm-example
python -m kernel_study.cli analyze --input results/llm-example --output reports/example
```

**Prices.** Prices are in USD per million tokens. Check them before starting a
new campaign.

**Model.** The pilot uses the pinned `gpt-4.1-mini-2025-04-14` snapshot.
It was available as of 2026-11-14. It is a deprecated snapshot, chosen for a
reproducible, low-cost pilot. The study assumes no model is scientifically
superior.

**Spending caps.** The `$5` limit in `scripts/llm_pilot.py` is shared across
all pilot jobs. A `--dollar-cap` passed to the CLI limits only that one
process. It does not limit other Slurm jobs.

## Running on Lonestar6

The project runs under allocation `ASC26098` from `$SCRATCH/kernel-study`.
Python environments and caches also live on `$SCRATCH`.

```bash
mkdir -p logs
sbatch cluster/pilot.sbatch
squeue -u "$USER"
```

`cluster/search.sbatch` runs a single manifest index, so the 144 searches
cannot be submitted all at once by accident. The script reads the following
from environment variables:

- the pinned model and endpoint;
- the token prices;
- the API credentials;
- the spending limit.

Review `configs/primary_manifest.json` before submitting repeated jobs.

Queue wait time is not part of optimization wall time. Record the Slurm
`sacct` submit, start, end and elapsed times with each run.

## Protocol status and interpretation

**Protocol status.** The protocol stays in development and is not frozen
until:

- the A100 pilot passes;
- its costs and caps have been reviewed.

**Final plan.** The proposal's final plan is 144 searches: 12 cases, three
seeds and four variants. Each search gets 20 attempts and 100 calls. Token use
is reported but is not a matched budget. An initial 500,000-token safety cap
will be reviewed after the pilot.

**Measurement mode.** The current mode uses common timing plus compiler
resource feedback.

- No Nsight hardware-counter data is used or inferred. The resource summary
  says nothing about bandwidth, occupancy, stall reasons or roofline bounds.
- Each new incumbent kernel is profiled, within a shared cap of 21
  invocations.
- Compile and first-launch time, process time and validation time are
  reported separately.
- Two things wait until hardware access and specifications are verified:
  hardware-counter profiling and A100 peak-based roofline fractions.

**Integrity limits.**

- Candidate modules may contain only approved Triton JIT functions. Launch,
  memory allocation, reference computation and timing stay in trusted code
  outside those modules.
- The AST gate and subprocess timeout guard integrity. They are **not a
  security sandbox** against adversarial GPU code.
- Model-generated code must keep the existing function interface. New
  host-side dispatch strategies are out of scope for this prototype.

**Matmul baseline.** As the proposal specifies, the matmul baseline does not
use tensor-core instructions. Library performance is reported separately so
that this headroom does not inflate claims about the agents.

See [docs/PROTOCOL.md](docs/PROTOCOL.md) for the full protocol and
[docs/WORK_LOG.md](docs/WORK_LOG.md) for the development record. For executed
experiments, the authority is the raw result directories and manifests on the
`results` branch. Unit-test fixtures and scripted controls are not LLM
findings.

## References

- [Project proposal](docs/proposal_final.pdf)
- [TACC Lonestar6 user guide](https://docs.tacc.utexas.edu/hpc/lonestar6/)
- [NVIDIA Ampere tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/)
- [CUDA C++ best practices guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/)
- [OpenAI Chat Completions API reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
- [`gpt-4.1-mini` model and pricing](https://developers.openai.com/api/docs/models/gpt-4.1-mini)
