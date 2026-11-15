# Reduction repair round - 2026-11-15

## Evidence and question

The downloaded A100 source pilot 3582814 completed all 16 searches for
$0.3224324. Five of 32 candidates were correct: four authored tiled matmuls
and one fused bias/ReLU. No reduction succeeded. Ten candidates were numerically
incorrect, seven failed compilation and ten failed the source/launch policy.
Every retained kernel was the first attempt; two attempts provide little
evidence about recovery after failure. Raw data stays on the results branch.

The next development round asks whether API correctness guidance and a longer
repair allowance produce reliable irregular-width reductions, and whether a
larger model is necessary. It does not estimate the isolated effect of guidance
by comparing against the old two-attempt cohort.

## Fixed design

Phase A uses gpt-4.1-mini-2025-04-14. Two cases (softmax_irregular and
layernorm_irregular), four variants and seeds 11/22 give 16 searches. Each gets
eight candidates and 32 calls. Baselines, tolerances, paired timing, fresh-input
fallback and source gate v2 are unchanged. The common feedback remains timing
and compiler resource reports, not per-candidate hardware counters.

Every arm receives the same short API correctness guide: constexpr power-of-two
arange extents, logical versus padded width, masking neutral elements, logical
normalization denominators, concrete scalar values, launch coverage and supported
API shapes. It contains no optimized implementation or hardware tuning choices.
Hardware knowledge cards remain an independent treatment. Guidance is versioned
and saved in each run. Sources:

- https://triton-lang.org/main/python-api/generated/triton.language.arange.html
- https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
- https://triton-lang.org/main/python-api/generated/triton.language.div_rn.html

Failure diagnostics now identify the input seed/edge case and nonfinite-output
count without changing validation. Full current/previous source remains in
context. All earlier diagnostics remain, while duplicate source/config inside
reviews is omitted from the context view; complete raw responses stay logged.
This common context change is part of the new cohort, not a hidden token subsidy.

## Predeclared next step

After a complete Phase A, require at least six of eight searches for **each**
case to finish with a fresh-validated authored incumbent. This is a practical
readiness threshold, not a statistical significance criterion.

If either case falls short, Phase B uses gpt-4.1-2025-04-14 on the identical
case/variant/seed/attempt matrix and identical guidance. If both meet the
threshold, Phase B is skipped. Incomplete Phase A cannot trigger a selective
model comparison. Report all failed attempts and cost-limited searches.

The full GPT-4.1 snapshot is chosen to preserve the same non-reasoning chat
interface and sampling controls as the mini model. It is listed as deprecated
in current documentation; availability must be checked before submission and
no model alias is silently substituted. Official listed prices are $2 input/
$8 output per million tokens (cached-input discounts are conservatively
ignored): https://developers.openai.com/api/docs/models/gpt-4.1

## Budget and reporting

The user approved a **$25 total** inference ceiling on 2026-11-15. Reserved
maximum exposure: prior $0.8739192 + local smoke $0.50 + Phase A $4 + Phase B
$16 = **$21.3739192**, leaving $3.6260808 unallocated. Reservations are not
reported as actual spend. Each phase has a shared provider ceiling and refuses
to overwrite an existing output directory. Re-submission requires accounting
for the previous phase's exposure, not resetting its cap.

The local smoke has four arms, one case and four attempts; it is development
evidence only and never pooled with A100 measurements. The A100 phases record
budget-prefix counts at 1/2/4/8 attempts, first correct/retained attempt, temporal
failure-to-correct transitions, final performance, model usage and failures.
Prefix correctness is not independently fresh-validated at every checkpoint;
only the final selection receives that evaluation. Manual source inspection
distinguishes real repairs from new proposals and superficial rewrites.

The 12-case, 144-search primary study and role/category ablations remain a
separate next stage after the reliability/model decision.
