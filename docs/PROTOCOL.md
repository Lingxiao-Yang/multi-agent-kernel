# Development protocol and completion criteria

The supplied **final PDF** defines the research question. This document
records implementation choices, deviations and gates; it does not silently
replace the full experiment with a smaller engineering exercise.

## Primary study

Twelve cases x four arms x three seeds = 144 searches. Each starts from an
identical source hash for its family. Match 20 attempts and 100 model-call
caps, report actual token use, and flag any token-limited runs separately.
The default token safety cap of 500,000 is provisional until pilot review.
All four arms use the same underlying model, four sequential calls per
candidate, context construction, sampling, greedy retention and evidence.

The single-agent prompt remains general across all four calls; stage labels
offer the same analysis, planning, implementation and review opportunities.
The multi-agent arm assigns four explicit specialist prompts. Review is
advisory. Retention is exclusively the automated harness's decision.

The prototype permits config changes and edits to existing Triton device
functions through a stable ABI. New host-level scheduling algorithms are
outside the current search space. That bounds generalization and should be
stated in any final report.

## Validation and timing

- Fixed per-case dtype/atol/rtol in configs/cases.json; contiguous tensors.
- Search seeds 101/202/303; final-only seeds 10001–10005. Each validation
  also tests zeros, constants, large values and negative inputs.
- Trusted reference on cloned inputs; outputs must overwrite poisoned
  buffers and inputs must be bitwise unchanged after execution.
- Warm-up and first launch outside timing; alternating AB/BA CUDA events,
  31 samples, 10 inner repetitions. Report median and IQR plus raw samples.
- Accept only a correct candidate that exceeds 1.01x in an initial paired
  comparison and a second paired confirmation. This threshold is a pilot
  choice, not a confidence test. Search curves use baseline-relative timings;
  final endpoints remeasure the selected candidate and baseline together.
- Fresh final-validation failure triggers fallback through prior retained
  candidates. Initial kernels remain eligible; unchanged runs score 1.0.
- Final evaluation allowance is at most 21 retained candidates, identical
  for every arm. Actual final attempts depend on failures and are logged.
- Compiler resources are collected initially and for every new incumbent
  with a 21-invocation cap. Nsight counters are currently unavailable/unverified;
  they must not be inferred from resource reports or speedup alone.

## Analysis

Pair case and seed across all four variants. Work in log speedups. Average
paired effects within each case and bootstrap complete case clusters 5,000
times with seed 42. Report role effect without knowledge, knowledge effect
for the single agent, role effect with knowledge and the multiplicative
interaction `(multi_knowledge * single)/(multi * single_knowledge)`.
Report missing/incomplete runs and avoid post-hoc selection. Token-limited
or endpoint-failed runs require separate accounting. Do not pool local RTX
and A100 results, different providers, source hashes or model versions.

## Gates before declaring the experiment complete

1. A100 GPU correctness and software compatibility verified on all cases.
2. Profiling capability checked; common reduced mode explicitly frozen if needed.
3. Model, prices, token safety cap, cases, knowledge bank and protocol hashes frozen.
4. Four-way pilot passes and affordability is reviewed.
5. Primary repeated searches complete, including unsuccessful runs.
6. Role/category ablations on two cases and mechanism examples inspected.
7. Confidence intervals, cost curves, library comparisons, measured throughput
   and justified roofline bounds generated; final report/presentation completed.

Completion percentages must be based on evidence against these milestones.
Unit tests and source files alone cannot establish completion of steps 4–7.
