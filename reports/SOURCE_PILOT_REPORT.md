# Source-generation pilot and experiment revision

The original A100 model pilot (job 3582416) made 48 configuration-only
proposals. Its gains selected supplied implementations. It remains a
parameter-selection control and does not establish that agents can write
optimized kernels.

The revised search presents only the current family's baseline source.
Agents author complete Triton kernels and declarative launch sequences;
function names, signatures and launch boundaries may change. There are no
predefined fusion or tensor-core switches in this search space. The trusted
reference, validator, allocator and timing code remain outside model control.

## Completed local source smoke

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU (development evidence only).
- Model: gpt-4.1-mini-2025-04-14; temperature 0.7.
- Two cases x four variants x seed 11; one candidate and four calls per search.
- Eight searches completed, 32 actual model calls, 108,331 input tokens and
  16,403 output tokens; estimated cost **$0.0695772**.
- Two proposals passed correctness and were retained. Final validation used
  five fresh random inputs plus four edge cases; baseline fallback passed in
  all six unsuccessful searches.

| Case | Single | Single + knowledge | Multi | Multi + knowledge |
| --- | ---: | ---: | ---: | ---: |
| matmul_rectangular | 2.310x | 1.827x | 1.000x | 1.000x |
| softmax_irregular | 1.000x | 1.000x | 1.000x | 1.000x |

These are final paired measurements. The local display GPU is noisy, the
attempt budget is only one, and there are no repeated seeds. Do not infer a
general role or knowledge effect. The analysis JSON's two-case bootstrap
intervals are not useful uncertainty estimates and should not be quoted.

## Source-level mechanisms and failures

| Proposal | What the actual source does | Outcome |
| --- | --- | --- |
| Matmul, single | Replaces scalar K accumulation with tile loads and tl.dot | Correct; retained at 2.310x |
| Matmul, single + knowledge | Adds tiled tl.dot with next-tile loads before current-tile computation | Correct; retained at 1.827x; source ordering alone does not prove hardware overlap |
| Matmul, multi | Attempts explicit shared-memory buffering using unsupported tl.shared_memory | Policy rejection |
| Matmul, multi + knowledge | Attempts a different tiled pipeline; malformed tl.zeros shape | Compilation failure |
| Softmax, multi + knowledge | Authors one fused row-reduction/output kernel | Compilation failure: tl.arange extent 781 is not a power of two |
| Other three softmax proposals | Rearrange temporaries and change grids, block sizes or scalar arguments | Numerical failures: incomplete coverage and/or truncated reductions |

Thus five proposals attempt algorithmic or execution-structure changes,
including both successful matmuls and the failed fused softmax. Three remain
parameter tuning dressed as source edits. We do not count those three as
substantive kernel optimization.

The initial structural screen passed seven proposals, exposing a weakness:
introducing temporaries changes the AST without changing the algorithm.
**Gate version 2** in main normalizes straight-line temporaries, associative
grouping and equivalent call argument spelling. Offline replay rejects all
three cosmetic softmax rewrites and still permits both validated matmuls.
The unsupported API remains rejected. This replay does not change the original
search outcomes, charge new model calls or generate new GPU measurements.
It remains a conservative screen rather than a semantic equivalence proof;
manual inspection of retained source remains required.

Raw data: [local source smoke](../results/local_source_smoke_v1/).
Audit: [candidate source audit](local_source_smoke_v1/source_audit.json),
[gate replay](local_source_smoke_v1/gate_v2_replay.json).

## A100 follow-up

Lonestar6 job **3582814** started at **2026-11-14 21:57:22 CST** on v330-016,
after submission at 21:51:31 (queue wait 351 seconds). It uses one
A100-PCIE-40GB, allocation ASC26098, with a two-hour limit.

All **12/12 baseline-interface checks passed** on A100 before inference.
Those completed validation records are archived in
[source-preflight-3582814](../results/source-preflight-3582814/).

The model campaign is **running**, not yet a completed result: four irregular
family cases x four variants x seed 11 x two attempts = 16 searches and 32
attempts, with a shared $1.25 API ceiling. It is frozen at commit ea03292 and
the original structural gate. Main's later gate-v2 changes were not applied
to the running checkout. Review both candidate attempts to distinguish real
code transformations, cosmetic rewrites and repairs.

The original 24-search A100 control, this local smoke, and the running A100
source pilot are distinct cohorts. Never pool their timings or treatment
effects. New gate-v2 runs must also be identified separately.

## Validation and remaining work

37 CPU integrity tests pass. The local four-family interface preflight and
A100 twelve-case preflight passed. Raw proposals, failures, source hashes,
model usage, latency samples and final-validation records are preserved.

The next decision uses the completed A100 source pilot's authorship, repair
success and cost. It should determine whether to improve shared API guidance
or use a stronger pinned model before freezing the main experiment. The
144-search primary matrix, repeated seeds, targeted role/category ablations,
and final hardware-headroom report remain outstanding. This checkpoint does
not establish 67% completion of the proposed empirical study.
