# Reduction repair round status - 2026-11-15

The completed prior A100 source pilot was rechecked from raw summaries: 16
searches, 32 attempts, five correct candidates, no correct reduction kernels,
and no retained candidate after the first attempt. The audit is archived in
[prior_review.json](repair_round/prior_review.json).

## Changes and validation

- Shared, versioned API correctness guidance for all four arms: constexpr
  power-of-two ranges, logical versus padded dimensions, neutral masking,
  normalization denominators, complete output coverage, and allowed APIs.
- Error feedback names the failed input seed/edge case and nonfinite count.
- Full current and previous source remains available; duplicate historical
  review source is excluded from context but preserved in raw event logs.
- Eight attempts and 32 calls per A100 search; seeds 11/22. The original
  tolerances, baseline kernels, source gate v2 and timing are unchanged.
- Prefix and temporal-recovery analysis, source audits and explicit phase
  decisions are generated automatically. 41 integrity tests passed.

## Submitted work

| Job | Phase | Planned searches | API ceiling | State at checkpoint |
| --- | --- | ---: | ---: | --- |
| 3584053 | Guided GPT-4.1 mini | 16 | $4 | Pending |
| 3584057 | Matched GPT-4.1, conditional | 16 | $16 | Pending dependency |

Both phases use softmax_irregular and layernorm_irregular, all four variants,
two seeds and eight attempts. Both pinned models were present in the live
endpoint catalog. Job 3584057 depends on afterany:3584053 but makes **no model
requests** unless mini completed successfully and either case has fewer than
six of eight fresh-validated, non-baseline final incumbents. If mini meets both
thresholds, strong is skipped; if mini is incomplete, strong is blocked. This
is a practical readiness rule, not a statistical test.

Frozen cluster checkout: `$SCRATCH/kernel-study-repair-20261115`, commit
`0d0fd71`. Slurm logs: `logs/repair-3584053.out/.err` and
`logs/repair-3584057.out/.err`. Outputs: `results/repair-mini-3584053` and
`results/repair-strong-3584057`. The latest scheduler estimate for mini was
16:26 CST on November 15; this is an estimate, not a reservation.

The separate local smoke is still active: four arms, one softmax case, four
attempts, $0.50 ceiling. This checkpoint copies **3/4 completed
searches only**, with explicit partial status. The remaining live outputs are
in the main checkout's `results/local_repair_smoke_v1`. No full local or A100
result is claimed yet. Do not pool local and A100 timings.

One completed local search (multi + knowledge) retained its first softmax
proposal, passing fresh validation at 1.9003x speedup. Its later three attempts
failed. The other two completed searches retained the baseline. This is a
partial development observation, not evidence that extra attempts improved
repair or that knowledge caused the gain.

## Budget and log

User-approved total cap: **$25**. Prior spending plus all reserved ceilings
is **$21.3739192**, leaving $3.6260808 unallocated. The current ledger separates
actual completed usage from ceilings. No extra model runs are authorized by
this checkpoint beyond the declared phases and local smoke.

Step journal: [repair_round_steps.jsonl](../logs/repair_round_steps.jsonl).
Plan: [repair_round_plan.json](../logs/repair_round_plan.json).
Budget: [pilot_budget_checkpoint.json](../logs/pilot_budget_checkpoint.json).
Implementation/design: `main:docs/REPAIR_ROUND.md`.

On return, fetch both phase directories and Slurm logs, inspect status.json,
repair_report.json, source_audit/ and steps.jsonl, and copy the completed local
smoke. Report first correctness/retention at attempts 1/2/4/8, failure-to-correct
transitions, final validated speedups, source mechanisms and actual usage.
Do not resubmit or reset caps without accounting for the existing runs.
