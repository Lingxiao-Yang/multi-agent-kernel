# Development Log

## 2026-11-15 - Longer reduction repair rounds

1. Audited the downloaded 3582814 raw summaries: 16 complete searches, 32
   attempts, 5 correct, 10 numerically incorrect, 7 runtime/compiler errors
   and 10 policy rejections; $0.3224324. All retained candidates were first
   attempts. No reduction family produced a correct candidate.
2. User approved a $25 total inference ceiling. The fixed phase ceilings and
   existing spend reserve $21.3739192 at most; no unbounded account spending.
3. Added common API correctness guidance (versioned, all four arms), resolved
   logical/padded dimensions, clearer failure diagnostics, and a history view
   that removes duplicate generated code while retaining full raw logs.
4. Added an 8-attempt/32-call, 2-seed, 4-arm reduction round (16 searches).
   A matched larger-model phase runs only if the complete mini round misses
   the predeclared 6-of-8 final validated-incumbent threshold for either case.
   Incomplete matrices cannot trigger a selective comparison. Each phase has
   its own immutable output and durable event/decision logs.
5. Added prefix and recovery analysis; tests cover incomplete/duplicate
   matrices, threshold decisions, dimensions, context preservation and caps.
   Initial validation: 41 tests passed (logs/tests_repair_round.log).
6. Started a separate four-arm, four-attempt local softmax smoke with a $0.50
   ceiling to exercise guidance and repeated repairs before broadening runs.

Design and source citations: REPAIR_ROUND.md. The complete action journal,
machine outputs, budget updates and subsequent outcomes live on results.

## 2026-11-14 - Source-generation revision

Reviewed completed A100 jobs 3582405 and 3582416 and their raw traces. Both
completed successfully; all 48 model proposals were configuration-only.
Reclassified that pilot as a parameter-selection control. Added `source-v1`,
which exposes baseline-only source, accepts new JIT functions and declarative
launch sequences, and rejects config-only/null/cosmetic/constant-only proposals.
The trusted validator and paired timing remain unchanged. Failed proposals are
saved, reviewers receive the actual submitted code, and repair context includes
the previous proposal. Model CLI and campaign entry points default to source-v1.

Validation: 36 CPU integrity tests pass; all four irregular family baselines
pass the new ABI on the RTX 4060 Laptop GPU, including fresh and edge inputs.
The A100 pilot launcher validates all 12 cases before requesting inference.
Planned source pilot: 4 families x 4 variants x 1 seed x 2 attempts ($1.25 cap).
Separate local smoke: 2 families x 4 variants x 1 attempt ($0.50 cap).
These ceilings plus prior pilot spend remain below the approved $5 total.
See SOURCE_EXPERIMENTS.md for scope, interpretation and remaining gates.

Added a source-audit exporter that counts supplied, structurally changed,
correct and retained proposals and records static primitive/launch features.
Fixed cohort validation to allow distinct baseline source per family while
still rejecting baseline drift within a family or changed case definitions.
The original shared-module hash check incorrectly rejected the new family-only
modules. This is an analysis fix; search and measurement logic are unchanged.

The completed local source smoke has 8 searches, 32 real model calls and
$0.0695772 inference cost. Two generated tiled matmul kernels passed validation
and were retained; six proposals failed policy, compilation or correctness.
Three of those failures only rearranged temporaries and changed launch knobs.
Gate version 2 now rejects those three in offline replay; both validated matmul
sources still pass. There are 37 passing CPU tests. The running A100 job
3582814 is frozen at `ea03292` (original gate); no code was changed in its
checkout after it started. All 12 source-interface baseline checks passed on
A100 before inference. Full logs and the completed laptop report are on results.

A100 job 3582814 completed (exit 0:0, 27 min 51 s, about $0.322): 16/16
searches, 5 correct authored kernels (tiled `tl.dot` matmul in all four
variants, 1.48–3.03×; fused bias + ReLU, 1.86×), no correct softmax or
layernorm. Offline gate-v2 replay changes no outcome. Results and audit are on
the `results` branch (`reports/SOURCE_PILOT_REPORT.md`).

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

## 2026-11-14 (evening) — First A100 runs

**First attempt failed.** A100 job 3582290 failed during environment setup.
NumPy 2.5.4 raised a spurious `X86_V2` CPU-baseline error on the EPYC 7763
compute nodes.

**Fix (`0ef781a`).**

- NumPy is pinned to `>=1.26,<2.3`, in `pyproject.toml` and
  `cluster/pilot.sbatch`.
- `cluster/pilot.sbatch` now checks that NumPy and Torch import, so a broken
  environment fails within seconds.

**Reruns.**

- Engineering pilot, job 3582405: 24/24 evaluations correct. Nsight counter
  access verified.
- LLM pilot, job 3582416: 24 searches, $0.448.

Results, analysis and continuation notes are on the `results` branch:
`reports/A100_PILOT_REPORT.md`, `reports/WORK_LOG.md` and
`logs/lonestar6_jobs.json`.

**New tooling.** `scripts/ls6_remote.py` runs a local script on Lonestar6, or
downloads results as a tar.gz.

- It authenticates with the password and TACC token, read from environment
  variables or prompted for.
- It checks the pinned host key first.
- It stores no credentials.

**Cluster API key.** The OpenAI key lives in `~/.openai_env` (mode 600) on
Lonestar6. Source it before submitting LLM jobs.
