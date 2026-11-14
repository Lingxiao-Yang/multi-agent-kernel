# Real-Model Smoke Test (2026-11-14)

Four real searches, one per variant, ran on an RTX 4060 Laptop GPU with the
pinned `gpt-4.1-mini-2025-04-14` model.

**Setup.**

- Case: `softmax_irregular` (`[513, 781]`, FP32), seed 11.
- One candidate and four model calls per variant.

**Outcome.**

- All four generated candidates passed the automated correctness checks.
- Total usage: **67,355 input + 4,486 output tokens**.
- Estimated uncached inference cost: **$0.0341196**.

| Variant | Final speedup | New candidate retained | Tokens | Estimated cost |
| --- | ---: | --- | ---: | ---: |
| Single | 1.000× | No | 15,797 | $0.0070880 |
| Single + knowledge | 1.797× | Yes | 18,775 | $0.0083104 |
| Multi | 1.000× | No | 16,103 | $0.0079232 |
| Multi + knowledge | 1.699× | Yes | 21,166 | $0.0107980 |

**How to read the table.**

- Final speedups were remeasured against the baseline after validation on
  fresh inputs.
- Every variant had the same candidate and call allowance.
- Token counts include repeated context and any knowledge cards, and were not
  equalized.
- A run with no improvement keeps the original kernel and scores 1.0.

## What this shows

**Shown.** The full pipeline ran end to end, with real model responses and
real GPU measurements:

- the real model endpoint;
- the sequential role and general prompts;
- knowledge selection and the candidate validator;
- timing confirmation and source snapshots;
- the budget ledger, final fallback and analysis.

**Not shown.** With one case, one seed and one attempt, this test does **not**
show:

- a general knowledge benefit;
- an advantage from specialization;
- an interaction effect.

No meaningful case-cluster confidence interval is possible with one case. The
A100 pilot uses two cases and three seeds. The full study uses 12 cases,
three seeds and larger search caps.

## Interface issue fixed before the A100 study

**Problem.** In the `multi` (no knowledge) trace, the model discussed
tensor-core and matmul tile settings while optimizing softmax. The trusted
softmax launcher ignores those fields. The cause was in the prompt context:

- the raw source contains every kernel family;
- the initial prompt did not state clearly which host launcher uses which
  fields.

**Fix.** The shared context for **all four variants** now states:

- which configuration fields are active and which are ignored;
- the mapping from entry points to launchers;
- the tensor shapes.

This describes the API. It is not optimization advice, so hardware guidance
remains the only knowledge that is ablated.

**Status of these results.** These smoke results come from before the fix and
must not be pooled with the A100 campaign run after it.

## API incident

**What happened.** The first search attempt stopped with HTTP 429 (account
credit exhausted) before receiving any model tokens. After credit was added, a
diagnostic completion succeeded and the smoke runs went ahead.

**How it is recorded.**

- The interrupted cohort is marked ineligible for scientific analysis.
- Its summary lists $0.0117716. That figure was a conservative reservation
  made before the request, not an actual charge. A correction file records
  this and leaves the raw log untouched.

**Controller change.** The controller now distinguishes three things:

- interrupted searches;
- the actual cost of completed inference;
- spending reservations.

Raw evidence is in `results/llm_local_smoke/`: complete prompts and responses,
candidate code, measurements and analysis outputs.
