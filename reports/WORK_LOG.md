# Execution log checkpoint — November 14, 2026

The initial implementation/access log is in `docs/WORK_LOG.md`. Subsequent
experimental records live on the **results** branch; implementation commits
live on **code**, as requested by the user.

## Verified local work

- Implemented all four controller variants, 12 benchmark cases, hardware cards,
  Triton baselines and transformations, correctness/timing/profile harness,
  cost/budget accounting, source policy, fallback and analysis pipeline.
- 26 integrity tests pass, including quota/cost handling, unknown output
  rejection, final fallback, interaction arithmetic and API field semantics.
- First engineering pilot: 24 GPU evaluations, 23 correct. Retained the failed
  layer-normalization edge case and fixed the mean arithmetic without changing
  tolerances.
- Fixed-snapshot engineering rerun: all 24 evaluations correct.
- Four-arm deterministic GPU control: 4 searches, 8 candidate attempts,
  32 scripted calls and zero LLM tokens. All searches completed.
- Real-model smoke: 4 searches, 4 candidate attempts, 16 paid model calls,
  71,841 total tokens, estimated $0.0341196. All candidates correct; two
  retained improvements. This is development evidence on one laptop case.
- A 39-token diagnostic completion cost approximately $0.0000216.
- Discovered and fixed missing common launch-API semantics before the A100
  campaign. This keeps hardware guidance as the independently ablated part.

## Lonestar6

- Logged in using the user's account and MFA; verified allocation ASC26098.
- Source checkout: `$SCRATCH/kernel-study`, tracking code.
- Engineering job 3582290 submitted 16:16:25, started 16:34:38 America/Chicago.
  Its initial 18:27 scheduler estimate was conservative. Environment setup
  is taking longer than expected while extracting/installing CUDA packages
  into the shared scratch environment. The job has not yet produced a
  completed A100 benchmark cohort at this checkpoint.
- A request to extend the running job's time limit was denied by Slurm;
  the original 30-minute limit remains. No bypass was attempted.
- Separate checkout `$SCRATCH/kernel-study-llm` avoids changing the running
  engineering job's code. Model job 3582326 is dependency-gated on job
  3582290 and additionally requires 24/24 successful engineering evaluations.
- Model campaign: 2 cases x 4 variants x 3 seeds, 2 candidate attempts and
  8 model calls per run; shared $4.49 cap. It is a pilot, not the full primary
  study. Together with the local $0.50 cap and diagnostic, this stays below
  the approved $5 inference allowance.

## Runpod

- Followed https://agents.runpod.io/ and installed official plugin 2.1.0.
- Verified 17 skill directories, MCP URL, enabled status and successful OAuth.
- New plugin tools require restarting Codex to appear in this session.
- Queried the public GPU catalog: low Secure A100 PCIe 80 GB/SXM 80 GB stock;
  Community stock unconfirmed. See RUNPOD_OPTIONS.md and its JSON snapshot.
- No Runpod GPU or storage resource was created.

## Remaining research work

A100 pilot validation, repeated real-model A100 runs, protocol freeze,
larger primary searches, role/category ablations, final cost/uncertainty and
mechanism analysis, final report and presentation. Completion is reported
by evidence; the full 144-search proposal has not been completed.
