# Budget-matched multi-agent kernel optimization

Implementation of **proposal_final.pdf**. The older `.tex` file is not the
protocol. This is an experiment repository: engineering controls and local
GPU measurements are kept separate from the planned A100 LLM study.

Branch policy: **`code`** contains implementation/configurations/tests;
**`results`** contains raw experiment logs, measurements, plots and reports.
Results carry source hashes and manifests. The initial bootstrap remains in
`main`; subsequent work uses the two dedicated branches.

## What is implemented

- Twelve cases: matmul, row softmax, layer normalization and bias + ReLU,
  three shapes each, including irregular boundaries.
- Readable Triton baselines; parameter and device-source transformations.
- Four variants: `single`, `single_knowledge`, `multi`, `multi_knowledge`.
  Identical four-call cadence and greedy search; specialized role prompts
  change only in multi variants. General agents get analysis/revision calls.
- Twelve sourced knowledge cards, deterministic selection and category ablation.
- A trusted PyTorch reference and timing harness, fresh final inputs, edge
  cases, immutable input checks, paired AB/BA timing and gain confirmation.
- JSONL prompts/responses, candidate source, compiler errors, correctness,
  latency samples, budgets, hardware/software metadata and SHA256 hashes.
- Final-validation fallback, accounting for failures/repairs, a token safety
  cap, explicit prices and bounded inference spending.
- Paired factorial effects, case-cluster bootstrap confidence intervals,
  performance-versus-budget curves, CSV summaries and Slurm launchers.

## Quick start

Linux with a CUDA GPU is required for measurements. CPU-only integrity tests
can run on Windows. Do not benchmark on a cluster login node.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[gpu,test]'
python -m pytest -q
python -m kernel_study.cli doctor
python -m kernel_study.cli pilot --output results/my-pilot
```

Use `--quick` only for development. Every output directory must be new;
existing runs are never silently overwritten. The baseline/transform pilot
uses no LLM and therefore cannot establish a role or knowledge effect.

```bash
python -m kernel_study.cli run --case softmax_irregular \
  --variant multi_knowledge --seed 11 --attempts 2 \
  --provider scripted --output results/control-example
```

For real model runs, configure `OPENAI_API_KEY` in the shell environment.
The endpoint implements the Chat Completions schema with reported usage,
JSON output, a seed and sampling temperature. Use one fixed model snapshot.

```bash
python -m kernel_study.cli run --case softmax_irregular \
  --variant multi_knowledge --seed 11 --provider llm \
  --model gpt-4.1-mini-2025-04-14 --endpoint https://api.openai.com/v1 \
  --input-price 0.40 --output-price 1.60 --dollar-cap 1 \
  --output results/llm-example
python -m kernel_study.cli analyze --input results/llm-example --output reports/example
```

Prices are USD per million tokens and must be checked before a new campaign.
The selected model is available in the supplied account as of 2026-11-14;
it is a deprecated snapshot, retained here for a reproducible low-cost pilot.
No scientific superiority is assumed. The `$5` bound in `scripts/llm_pilot.py`
is shared across all pilot jobs; individual CLI `--dollar-cap` limits apply
to one process, not to unrelated Slurm jobs.

## Lonestar6

The verified allocation is `ASC26098`. The project is deployed under
`$SCRATCH/kernel-study`. Python environments and caches use `$SCRATCH`.

```bash
mkdir -p logs
sbatch cluster/pilot.sbatch
squeue -u "$USER"
```

`cluster/search.sbatch` runs one explicit manifest index, so accidental
submission of all 144 searches cannot occur. It requires a pinned model,
endpoint, prices, credentials and a financial limit in environment variables.
Inspect `configs/primary_manifest.json` before submitting repeated jobs.
TACC queue wait is separate from optimization wall time; query Slurm
`sacct` for submit/start/end/elapsed and preserve that output with the run.

## Protocol status and interpretation

The protocol is **development, not frozen** until the A100 pilot passes and
costs/caps are inspected. The PDF's final plan is 20 attempts, 100 calls,
three seeds and 12 cases (144 searches). Tokens are a reported resource,
with an initial 500,000-token safety cap requiring pilot review.

The first measurement mode is common timing + compiler resource feedback.
No Nsight hardware-counter evidence is fabricated. The resource summary
does not establish bandwidth, occupancy, stall reasons or a roofline bound.
Each new incumbent receives profiling within the common 21-invocation cap.
Compilation/first-launch time, process time and validation time are reported
separately. Hardware-counter profiling and certified A100 peak-based
roofline fractions remain additional work until access/specifications are verified.

Candidate modules may contain only approved Triton JIT functions. Launching,
allocation, references and timers remain trusted code outside those modules.
The AST gate and subprocess timeout are integrity controls, **not a security
sandbox** against adversarial GPU code. Model-generated modules are limited
to the existing function ABI; arbitrary new host dispatch strategies are
outside this prototype. The baseline matmul deliberately avoids tensor-core
instructions per the PDF; report library speed separately so its headroom
does not inflate claims about agent quality.

Current evidence and unfinished work are in [docs/WORK_LOG.md](docs/WORK_LOG.md).
Raw result directories and their manifests are the authority for executed
experiments. Unit-test fixtures and scripted controls are not LLM findings.

## Sources

- [Final proposal](proposal_final.pdf), supplied by the user.
- [TACC Lonestar6 guide](https://docs.tacc.utexas.edu/hpc/lonestar6/).
- [NVIDIA Ampere tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/).
- [CUDA best practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/).
- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/).
- [Chat Completions reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).
- [Pinned model and pricing](https://developers.openai.com/api/docs/models/gpt-4.1-mini).
