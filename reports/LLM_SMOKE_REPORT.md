# Real-model smoke comparison — 2026-11-14

Four real searches completed on an RTX 4060 Laptop GPU using the pinned
`gpt-4.1-mini-2025-04-14` endpoint. The cohort covers `softmax_irregular`
(`[513,781]`, FP32), seed 11, one candidate and four model calls per variant.
All four generated candidates passed the automated correctness checks.
Total reported usage was **67,355 input tokens + 4,486 output tokens**,
with an estimated uncached inference cost of **$0.0341196**.

| Variant | Final speedup | Retained new candidate | Tokens | Estimated cost |
|---|---:|---|---:|---:|
| Single | 1.000x | No | 15,797 | $0.0070880 |
| Single + knowledge | 1.797x | Yes | 18,775 | $0.0083104 |
| Multi | 1.000x | No | 16,103 | $0.0079232 |
| Multi + knowledge | 1.699x | Yes | 21,166 | $0.0107980 |

The final speedups were remeasured against the baseline after fresh-input
validation. Each arm had the same candidate and call allowances. Tokens
include repeated context and supplied cards; they were not artificially
equalized. No-improvement runs retain the original kernel and score 1.0.

## What this establishes

The real endpoint, sequential role/general prompts, knowledge selection,
candidate validator, timing confirmation, source snapshots, budget ledger,
final fallback and analysis pipeline all ran end to end. These are actual
model responses and GPU measurements, not simulated LLM outcomes.

This one-case, one-seed, one-attempt smoke test does **not** establish a
general knowledge benefit, specialization advantage, or interaction effect.
There is no meaningful case-cluster confidence interval with one case.
The A100 pilot will use two cases and three seeds; the full proposal remains
12 cases, three seeds and larger search caps.

## Interface weakness found before the A100 study

The multi/no-knowledge trace discussed tensor-core and matmul tile settings
while optimizing softmax. Those config fields are ignored by the trusted
softmax launcher. The raw source contains all kernel families, and the
initial prompt did not state the host launch mapping explicitly enough.

The common context has now been improved for **all four variants** with
the active/ignored config fields, entry-point mapping and tensor shapes.
This is API semantics, not an optimization recommendation. The local smoke
results are preserved as pre-clarification development evidence; they must
not be pooled with the post-clarification A100 campaign.

## Earlier API failure

The initial credential was invalid; the replacement could list models but
had exhausted credits. One search stopped at HTTP 429 before receiving any
model tokens. The user then added credit, and a diagnostic completion
succeeded. The interrupted cohort is explicitly ineligible for scientific
analysis. Its old summary's $0.0117716 field was a conservative request
reservation, not a token charge; a correction sidecar preserves this fact
without altering the raw log. The controller now distinguishes interrupted
searches and actual completed inference cost from financial reservations.

Raw evidence: `results/llm_local_smoke/`, including complete prompts,
responses, candidate code, measurements and analysis outputs.
