# Experiment Log (checkpoint: 2026-11-14)

The implementation log is in `docs/WORK_LOG.md` on `main`. This file records
the experiments. Code lives on `main`; results live on this branch.

## Completed locally

**Implementation.**

- All four controller variants and 12 benchmark cases.
- Hardware knowledge cards, Triton baselines and transformations.
- The correctness, timing and profiling harness.
- Cost and budget accounting, the source policy, fallback and the analysis
  pipeline.

**Tests.** 26 integrity tests pass. They cover:

- quota and cost handling;
- rejection of unknown outputs;
- final fallback and interaction arithmetic;
- the meaning of API fields.

**Experiments.**

| Run | Scale | Outcome |
| --- | --- | --- |
| Engineering pilot (v2) | 24 GPU evaluations | 23 correct. The failed layer-norm edge case was kept, and the mean arithmetic was fixed without changing tolerances. |
| Engineering rerun (v3) | 24 GPU evaluations, fixed snapshot | All correct ([report](GPU_PILOT_REPORT.md)). |
| Four-variant scripted control | 4 searches, 8 candidate attempts, 32 scripted calls, zero LLM tokens | All searches completed ([report](CONTROL_MATRIX_REPORT.md)). |
| Real-model smoke test | 4 searches, 4 candidate attempts, 16 model calls, 71,841 tokens, about $0.0341 | All candidates correct; two improvements retained. Development evidence on one laptop case ([report](LLM_SMOKE_REPORT.md)). |
| Diagnostic completion | 39 tokens | Cost about $0.0000216. |

**Prompt fix.** A gap in the shared launch-API description was found and
fixed before the A100 campaign. This keeps hardware guidance as the only
independently ablated knowledge.

## Lonestar6

- **Allocation.** Access to `ASC26098` was verified. The code is checked out
  at `$SCRATCH/kernel-study`, tracking `main`.
- **Engineering job 3582290.**
  - Submitted at 16:16:25 and started at 16:34:38 America/Chicago.
  - Setting up the environment was slow, because CUDA packages had to be
    extracted into the shared scratch environment.
  - At this checkpoint the job had not yet produced a complete A100 cohort.
  - Slurm denied a request to extend its 30-minute limit, so the original
    limit stands.
- **Model job 3582326.**
  - It runs from a separate checkout (`$SCRATCH/kernel-study-llm`), so the
    running engineering job's code is not changed.
  - It waits for job 3582290 to finish, and it runs only if all 24
    engineering evaluations pass.
- **Model pilot campaign.**
  - 2 cases × 4 variants × 3 seeds.
  - 2 candidate attempts and 8 model calls per run.
  - A shared $4.49 cap. Together with the local $0.50 cap and the diagnostic
    call, total spend stays under the $5 inference budget.
  - This is a pilot, not the primary study.

## Backup compute

We assessed Runpod as a fallback A100 provider: public pricing and a snapshot
of the catalog's capacity. No resources were created. See
[`COMPUTE_OPTIONS.md`](COMPUTE_OPTIONS.md).

## Remaining work

1. Validate the A100 pilot.
2. Run repeated real-model A100 runs.
3. Freeze the protocol.
4. Run the primary 144-search study.
5. Run role and knowledge-category ablations.
6. Complete the final cost, uncertainty and mechanism analysis.
7. Write the final report and presentation.

Progress is reported only from evidence. The full study has not been
completed.
