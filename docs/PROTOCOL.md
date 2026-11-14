# Experimental Protocol and Completion Criteria

The [project proposal](proposal_final.pdf) defines the research question.
This document records the implementation choices, deviations from the
proposal, and the gates for calling the study complete. It does not swap the
full experiment for a smaller engineering exercise.

## Primary study

**Scale.** Twelve cases × four variants × three seeds give 144 searches.
Every search in a family starts from the same baseline source hash.

**Budgets.**

- Each search has matched caps of 20 candidate attempts and 100 model calls.
- Actual token use is reported, and token-limited runs are flagged separately.
- The default 500,000-token safety cap is provisional until after the pilot
  review.

**Shared conditions.** All four variants use:

- the same underlying model;
- four sequential calls per candidate;
- the same context construction and sampling settings;
- greedy retention and the same evidence.

**Prompts.** The single agent keeps a general prompt for all four calls. Stage
labels give it the same analysis, planning, implementation and review steps.
The multi-agent variants assign four specialist role prompts instead. Review
is advisory only; the automated harness alone decides which candidates to
keep.

**Search space.** The prototype allows two kinds of change:

- configuration changes;
- edits to existing Triton device functions behind a stable interface.

New host-side scheduling algorithms are out of scope. This limits how far the
results generalize, and any final report should state that limit.

## Validation and timing

**Correctness criteria.** Tolerances (dtype, `atol`, `rtol`) are fixed per
case in `configs/cases.json`. All tensors are contiguous.

**Inputs.**

- Search uses seeds 101, 202 and 303.
- Final validation uses seeds 10001–10005, which are never seen during search.
- Every validation also tests zero, constant, large-magnitude and negative
  inputs.

**Reference checks.**

- The trusted reference runs on cloned inputs.
- Each candidate must overwrite deliberately poisoned output buffers.
- Inputs must be bitwise unchanged after the candidate runs.

**Timing.**

- Warm-up and the first launch are excluded from timing.
- Candidate and baseline alternate in AB/BA order using CUDA events: 31
  samples of 10 inner repetitions each.
- The median and IQR are reported, and the raw samples are kept.

**Acceptance.** A candidate is kept only if it is correct and beats the
incumbent by more than 1.01× in two separate paired comparisons. This
threshold is a pilot choice, not a statistical test.

**Reported speedups.**

- Search curves use timings relative to the baseline.
- Final speedups come from remeasuring the selected candidate and the
  baseline together.

**Fallback.** If a candidate fails final validation on fresh inputs, the
harness falls back to the previously retained candidate, then the one before
it. The initial kernel is always eligible, so a run with no improvement scores
1.0.

**Final-evaluation budget.** Final evaluation covers at most 21 retained
candidates, the same for every variant. The actual number of attempts depends
on failures and is logged.

**Profiling.** Compiler resource usage is collected for the baseline and for
each new incumbent, up to 21 invocations. Nsight hardware counters are
currently unverified. They must not be inferred from resource reports or from
speedups.

## Analysis

**Effects.** Runs are paired by case and seed across all four variants, and
effects are computed on log speedups. Within each case, the paired effects
are averaged.

**Uncertainty.** Whole cases are then resampled in a case-cluster bootstrap:
5,000 resamples with seed 42.

**Reported quantities.**

- The role effect without knowledge.
- The knowledge effect for the single agent.
- The role effect with knowledge.
- The multiplicative interaction,
  `(multi_knowledge × single) / (multi × single_knowledge)`.

**Exclusions and pooling.**

- Missing and incomplete runs are reported, and no runs are selected after
  the fact.
- Token-limited runs and runs that failed at the endpoint are reported
  separately.
- Results are never pooled across:
  - different GPUs, such as a local RTX GPU and an A100;
  - different providers;
  - different source hashes or model versions.

## Gates before declaring the experiment complete

1. A100 correctness and software compatibility are verified on every case.
2. Profiling capability is checked; if needed, a common reduced mode is frozen
   explicitly.
3. The model, prices, token safety cap, cases, knowledge bank and protocol
   hashes are frozen.
4. The four-variant pilot passes and its cost is reviewed.
5. The primary repeated searches are complete, including unsuccessful runs.
6. Role and knowledge-category ablations on two cases are complete, and
   example mechanisms are inspected.
7. The following are produced:
   - confidence intervals and cost curves;
   - library comparisons and measured throughput;
   - justified roofline bounds;
   - the final report and presentation.

Completion is measured against these milestones. Unit tests and source code
alone cannot show that steps 4–7 are complete.
