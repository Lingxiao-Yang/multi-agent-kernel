# Next experiments after the guided reduction pilots

Planned on 2026-11-15; **not submitted**. Code stays on `main`; immutable run
artifacts and analysis stay on `results`. Completed evidence is in
[`results:reports/REPAIR_ROUND_REPORT.md`](https://github.com/Lingxiao-Yang/multi-agent-kernel/blob/results/reports/REPAIR_ROUND_REPORT.md).

## Decision from the completed round

Use pinned GPT-4.1 for the next source-generation validation stage. With the
same shared guidance, two cases, two seeds and eight attempts, it retained
fresh-validated improvements in 14/16 searches, versus mini's 5/16. Both
practical case thresholds were met (softmax 8/8; layernorm 6/8). This is a
model-readiness decision, not evidence that multi-agent roles outperform a
single agent. Do not pool these reduction pilots with older configuration
selection or source pilots whose protocols differ.

## 1. Audit numerical robustness before more inference

Re-evaluate the existing final layernorm candidates and their baselines on
large-offset, small-variance inputs, near-constant rows and several random
seeds. Record reference precision and unchanged tolerances. Candidate 8 in
`layernorm_irregular_single_22` uses E[x^2]-E[x]^2, which can suffer
cancellation. It passed the frozen nine-input final suite, but that suite
does not establish robustness on this additional distribution. Preserve its
original result; label stress results separately. Test every final layernorm
candidate, not just the suspicious one. This needs GPU time, no API calls.

Inspect generated code at the actual constexpr sizes before attributing
speedups to mechanisms: this candidate's BLOCK=1024 and N=781 make its
claimed two-block unrolled main loop execute zero times. The active code
changes reduction arithmetic and uses the tail path. Profile representative
retained kernels and compare against the recorded library implementation.

Review the source allowlist and repeated diagnostics offline. A rejected
`tl.uint32` is a policy restriction, not proof of a nonexistent API. Verify
any proposed allowlist change against the installed Triton version, with
regression checks; do not relax source-edit or correctness requirements.
If validation, gate or shared guidance changes, version and freeze a new
cohort and rerun matched cells. Never silently combine it with this cohort.

## 2. Spend remaining inference budget on replication and breadth

Conditional on the audit, keep the frozen model, source gate, shared guidance,
eight-attempt/32-call caps and all four arms. Eight attempts expose repairs
even though retention plateaued by attempt 2; do not use this pilot to claim
that two attempts suffice generally. Keep every failure in the analysis.

| Stage | Cases | Seeds | Searches | Planning ceiling |
| --- | --- | --- | ---: | ---: |
| A: reduction replication | softmax_irregular, layernorm_irregular | 33 | 8 | $5.50 |
| B: family transfer | matmul_rectangular, elementwise_irregular | 11 | 8 | $5.50 |

Each stage's rough expected cost is $5.21, extrapolating the completed full
model's $10.4175/16 searches. Family-specific output length may differ. The
ceilings are hard limits, not a promise to complete all cells. Launch stages
sequentially; reconcile the ledger first, and never reset spending after a
partial run. If a stage is incomplete, report all completed and censored
cells and defer arm comparisons. Do not top up silently.

The approved total is $25. Completed estimated usage is $13.4122396,
leaving $11.5877604. These proposed $11 ceilings leave $0.5877604 unallocated.
No new reservations or purchases were made during the analysis turn.
If step 1 requires a protocol change, prioritize a complete new matched
reduction cohort within the remaining cap and defer Stage B; do not treat
old seeds as replications under the changed protocol.

## 3. Freeze the primary study separately

The proposal's 12 cases x four arms x three seeds, at 20 attempts/search,
requires 144 searches and 2,880 attempts, plus separate role/category
ablations. At the observed full-model cost per attempt, the primary matrix
alone extrapolates to **$234.39**. This is a planning estimate, not an approved
budget or a quote; growing contexts may increase it. Keep matched call and
candidate budgets, report tokens/dollars separately, randomize run order and
use case-level uncertainty with the full suite. The present two-case pilot
cannot support a reliable roles/knowledge effectiveness claim or a claim
that the primary experiment is complete.

Before that stage, resolve numerical stress findings, predeclare validation
and hardware controls, verify model availability, and obtain a sufficient
total budget. Do not automatically launch the primary matrix under $25.

## Handoff

Both Slurm jobs completed successfully. No experiment was submitted while
checking and analyzing these results. Start the next work session at step 1,
then recheck the ledger before requesting any inference. Log each audit,
protocol decision, submission, retrieval and analysis in the step journal.
