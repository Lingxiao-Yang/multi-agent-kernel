# Experiment Log

The implementation log is in `docs/WORK_LOG.md` on `main`. This file records
the experiments. Code lives on `main`; results live on this branch. Every
Lonestar6 job is listed in [`logs/lonestar6_jobs.json`](../logs/lonestar6_jobs.json).

## Checkpoint 2 — 2026-11-14, 20:15 CST: first A100 results

**Failed first attempt.**

- Job 3582290 failed before any GPU work. NumPy 2.5.4 rejected the EPYC 7763
  compute node with a spurious `X86_V2` error.
- Fixed on `main` in `0ef781a`: NumPy pinned to `<2.3`, plus an import check
  that fails within seconds.
- The dependent LLM job 3582326 was cancelled. It had spent nothing on
  inference.
- Both cluster checkouts (`$SCRATCH/kernel-study` and
  `$SCRATCH/kernel-study-llm`) now track `main`.

**Engineering pilot (job 3582405).**

- 24/24 evaluations correct on an A100-PCIE-40GB.
- Baseline-versus-itself timing ratios were 0.970–1.047.
- Access to Nsight Compute hardware counters was verified.

**LLM pilot (job 3582416).**

- 24/24 searches completed: 2 cases × 4 variants × 3 seeds, 2 attempts each.
- Cost: $0.448.
- Before submitting, the API key in `~/.openai_env` on Lonestar6 was checked
  with a 12-token test call.

**Results.** Overall geometric means:

| Variant | Geometric mean |
| --- | ---: |
| Multi-agent | 3.33× |
| Multi-agent + knowledge | 2.26× |
| Single | 1.20× |
| Single + knowledge | 1.19× |

- The gap is driven by matmul. The multi-agent variants enabled
  `tensor_core=True` in 5 of 6 searches, for about 5–6.3×. The single-agent
  variants never did (0 of 6).
- Every change was configuration-only; there were no source edits.
- 9 of the 48 attempts were invalid proposals.

Details: [`A100_PILOT_REPORT.md`](A100_PILOT_REPORT.md).

### Gate status (see `docs/PROTOCOL.md` on `main`)

| # | Gate | Status |
| --- | --- | --- |
| 1 | A100 correctness and compatibility on all cases | **Done** (job 3582405) |
| 2 | Profiling capability checked | **Mostly done**: Nsight counters verified; reduced versus full profiling mode still to be frozen |
| 3 | Model, prices, caps, cases, knowledge and protocol frozen | Open: needs the decisions below |
| 4 | Four-variant pilot passes; cost reviewed | Pilot **passed**; cost review **pending** (estimate about $27 for the primary study) |
| 5 | 144 primary searches | Not started |
| 6 | Role and knowledge-category ablations (2 cases) | Not started |
| 7 | Confidence intervals, cost curves, library comparison, roofline, report | Pipeline ready; awaits data |

Estimated overall progress: about 45%.

### Decisions needed before freezing the protocol (gate 3)

1. **Inference budget.** The primary study (144 searches × 20 attempts) is
   estimated at about $27, plus ablations. The approved total so far is $5,
   of which about $0.49 has been spent.
2. **Predefined switches.** Matmul and softmax outcomes are dominated by
   whether an agent flips `tensor_core` or `fused`. Decide whether to:
   - keep the switches and report their effect separately; or
   - also require a code-edit task; or
   - remove the switches from the search space for some cases.
3. **Invalid proposals.** About 19% of attempts are invalid. Decide whether to
   add a schema reminder to the shared context for all variants, or keep it
   unchanged and report the rate as an outcome.
4. **Token safety cap.** Searches used 33–49k tokens for 2 attempts. A
   20-attempt search may approach the 500k cap; review it before freezing.

## How to continue on Lonestar6

**Access.** Log in with password and TACC MFA. The ed25519 host key is pinned
locally in `tmp/known_hosts`. The local SSH helper requires the
`hmac-sha2-512` MAC.

**Environment.**

| Item | Location |
| --- | --- |
| Shared venv | `$SCRATCH/kernel-study-venv` (NumPy pinned), built by `cluster/pilot.sbatch` |
| Engineering checkout | `$SCRATCH/kernel-study` |
| LLM checkout | `$SCRATCH/kernel-study-llm` |
| API key | `~/.openai_env`, mode 600; `source` it before submitting LLM jobs |

**Queue.** `gpu-a100-small` has 24 nodes and was fully allocated with about
50 jobs waiting. The 30-minute engineering job waited about 1.6 h; the LLM job
started immediately after it finished.

**Collecting outputs.**

1. Archive `results/<run>` and `logs/*<jobid>*` over SSH.
2. Copy the new folders into this branch.
3. Run `viz/` to regenerate the plots.
4. Add a row to `logs/lonestar6_jobs.json`.

Do **not** run `scripts/publish_results.py` from a checkout whose local
`reports/` holds older report text. It overwrites this branch's reports.

## Checkpoint 1 — 2026-11-14 (afternoon): local work

**Implementation.**

- All four controller variants and 12 benchmark cases.
- Hardware knowledge cards, Triton baselines and transformations.
- The correctness, timing and profiling harness.
- Cost and budget accounting, the source policy, fallback and the analysis
  pipeline.

**Tests.** 26 integrity tests pass.

| Run | Scale | Outcome |
| --- | --- | --- |
| Engineering pilot (v2) | 24 laptop GPU evaluations | 23 correct; the layer-norm mean was fixed without changing tolerances |
| Engineering rerun (v3) | 24 evaluations, fixed snapshot | All correct ([report](GPU_PILOT_REPORT.md)) |
| Four-variant scripted control | 4 searches, 8 attempts, 32 scripted calls, zero LLM tokens | All completed ([report](CONTROL_MATRIX_REPORT.md)) |
| Real-model smoke test (laptop) | 4 searches, 16 model calls, 71,841 tokens, about $0.0341 | All correct; 2 improvements retained ([report](LLM_SMOKE_REPORT.md)) |

**Prompt fix.** A gap in the shared launch-API description was fixed before
the A100 campaign.
