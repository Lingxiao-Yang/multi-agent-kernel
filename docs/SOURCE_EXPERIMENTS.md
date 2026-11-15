# Source-generation revision

The first A100 model pilot completed 24 searches and 48 attempts for $0.4477624.
All proposals left device source unchanged. Its strongest gains selected an
already-written tensor-core implementation; this does not address the main
research question about agents writing optimized kernels.

The next pilot uses `source-v1`: one irregular case from each of the four
families, all four variants, seed 11, two attempts and eight calls per search.
That is 16 searches and 32 attempts with a shared $1.25 inference ceiling.
The same model snapshot, sampling, correctness tolerances and timing procedure
are retained. All 12 baseline cases must pass the new launch ABI before any
model request. This is a development pilot, not the 144-search primary study.

## What agents can change

Agents receive baseline source only and author JIT kernels plus a data-only
launch plan. They may introduce new function names and signatures, change
parallel decomposition, replace scalar accumulation with a tiled algorithm,
and change launch boundaries. The interface exposes input/output tensors,
row-statistic scratch and one output-shaped scratch buffer. It does not execute
agent-written host Python. The plan may contain one to four launches and
bounded integer grids and arguments. This excludes arbitrary host algorithms,
dynamic allocations, atomics and helper JIT functions in this version.

Configuration-only and superficial source edits are rejected. The AST check
is a conservative structural filter, not a semantic equivalence checker.
Manual inspection must identify the transformation actually executed in every
retained candidate; comments and the agent's rationale are not sufficient.

The first local source pilot exposed a loophole: lifting expressions into
temporary variables passed the initial AST check. Gate version 2 additionally
inlines straight-line temporaries and normalizes associative grouping and
equivalent positional/keyword calls for comparison. It rejects all three such
saved softmax proposals while allowing both validated tiled matmul proposals.
This is an offline replay, not a new GPU result. New runs record
`source_gate_version=2`; the already-running A100 job 3582814 remains frozen
on the original gate at commit `ea03292`. The cohorts must stay separate.

## Review and next gates

Report authored, compiled, correct and retained counts separately; classify
actual source changes, including fusion, data reuse, tensor-core computation,
layout changes, and ineffective rewrites. Preserve errors and unsuccessful runs.
Compare source-generation arms only within their own cohort. Keep the earlier
configuration controls as a separate mechanism comparison with their smaller
case coverage stated. Four cases and one seed do not support definitive causal
claims or stable confidence intervals.

If authored code compiles and passes all families, extend seeds and attempts,
freeze the protocol and estimate full-study cost from actual source-generation
usage. The primary study remains 12 cases x 4 variants x 3 seeds, 20 attempts
and 100-call caps, followed by role and knowledge ablations. The existing $5
total pilot authorization is still binding; a monthly account limit is not an
experiment spending authorization.

## Reproduce

```bash
python scripts/source_preflight.py --output results/source-preflight
python scripts/run_campaign.py --output results/source-pilot \
  --search-space source-v1 \
  --cases matmul_rectangular,softmax_irregular,layernorm_irregular,elementwise_irregular \
  --seeds 11 --attempts 2 --dollar-cap 1.25
```

On Lonestar6 use `cluster/source_pilot.sbatch` after exporting the key through
the existing private environment file. Never put credentials in scripts or logs.
