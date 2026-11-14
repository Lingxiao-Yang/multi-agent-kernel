# Development Log

Dates are in America/Chicago time. Logs of machine events use UTC.
Credentials are never stored in this repository. Later experimental records
are on the `results` branch (`reports/WORK_LOG.md`).

## 2026-11-14 — Implementation and first validation

**Scope.** We took the final proposal PDF as authoritative over an earlier
LaTeX draft. Its key terms:

- 20 candidate attempts and 100 model calls per search;
- a token cap that is a safety limit, not a matched variable;
- profiling of the baseline and of every new incumbent;
- bootstrapped paired effects, with roofline/headroom reporting.

**Sources.** We consulted current documentation for TACC Lonestar6, NVIDIA
Ampere tuning and CUDA best practices, Triton, and the OpenAI API. The
references are in `knowledge/cards.json` and the README.

**Hardware.**

- Local development uses an RTX 4060 Laptop GPU under WSL Ubuntu. Local GPU
  results are never presented as A100 results.
- The first attempt to install the Linux GPU environment on the Windows-mounted
  drive failed. The environment was then rebuilt on Linux-native storage.
  Setup logs are on the `results` branch.

**Implemented.**

- *Benchmark and harness.*
  - Four kernel families, 12 cases and trusted references.
  - Edge-case, random and fresh-input validation.
  - Paired AB/BA CUDA-event timing with confirmation of gains.
  - Compiler-resource feedback, candidate process timeouts and an AST
    integrity gate.
- *Agent loops.*
  - General and specialized four-call loops with greedy retention.
  - JSONL, state and source logging, and accounting for every attempt.
  - Model token reporting and a scripted engineering control.
  - Fallback during final validation, and 12 hardware knowledge cards.
- *Analysis.*
  - Factorial paired analysis and a case-cluster bootstrap.
  - CSV output and performance/cost curves.
  - A balanced 144-search manifest.

**Tests.** The first suite: **19 passed**. These are software integrity
tests. They are not GPU correctness checks or scientific results.

## 2026-11-14 — Cluster access and local GPU pilot

**Lonestar6.**

- Access to allocation `ASC26098` and its scratch space was verified.
- A100 pilot job 3582290 was submitted to `gpu-a100-small` with a 30-minute
  limit. When this entry was written, it was still waiting in the queue.
- The cluster checks out the repository over HTTPS.

**Model API.**

- We selected the pinned `gpt-4.1-mini-2025-04-14` snapshot for a low-cost
  pilot. Its rates are $0.40 per million input tokens and $1.60 per million
  output tokens.
- Inference spending for the pilot is capped at **$5 total**.
- Sending kernel source, prompts and measurements to the API was approved
  before any requests were made.

**Local software.** PyTorch 2.11.0+cu128 and Triton 3.6.0 on WSL, installed in
a persistent `.venv-linux` environment.

**Pilot v2 (exploratory).**

- 24 RTX 4060 evaluations ran; 23 passed.
- `layernorm_irregular` failed the unchanged `2e-5` absolute tolerance, with a
  maximum difference of about `3.99e-5`. The cause was cancellation from
  approximate division in the mean on a constant row of irregular width.
- Fix: both layer-normalization kernels now use `tl.div_rn` for the mean.
  The tolerances were not loosened.
- Reporting instrumentation changed while v2 was running. It is kept as the
  record of the failure, not as the evaluation dataset.

**Pilot v3 (reproducible).** Records immutable candidate source and code
hashes, and uses the standard 31-sample timing.

**Timing noise.** Laptop timing is noisy: one baseline-versus-itself ratio
was about 0.59. Laptop timings are development evidence, not A100
performance claims.

**Tests.** The integrity suite now has **21 passed**.

**Additions.**

- Throughput accounting for logical traffic and FLOPs.
- An Nsight counter probe.

Peak-fraction fields stay null until the hardware's peak values are verified.
