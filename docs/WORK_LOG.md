# Work log

Dates use America/Chicago for project reporting. Machine events use explicit UTC.
No account passwords, MFA tokens or API keys belong in this repository.

## 2026-11-14 — implementation and first validation

- Inspected workspace: only proposal_final.pdf and an older proposal.tex existed.
- Extracted all three PDF pages with pdftotext and visually inspected page 1.
- Found substantive PDF/TEX differences. User confirmed PDF is authoritative:
  20 candidate attempts, 100 calls; token cap is a safety limit, not a matching
  variable; profile baseline and each new incumbent. Bootstrap paired effects
  and roofline/headroom reporting are in the PDF.
- Consulted current TACC Lonestar6, NVIDIA Ampere/best-practices, Triton and
  OpenAI API documentation. URLs are retained in knowledge/cards.json and README.
- Established that local Windows has an RTX 4060 Laptop GPU and WSL Ubuntu.
  Local GPU results must not be presented as A100 results.
- Lonestar6: outbound SSH required execution outside the network sandbox;
  default SSH MAC failed. hmac-sha2-512 reached authentication. Initial supplied
  password did not authenticate. User provided a correction and confirmed MFA
  must be handled manually. No GPU allocations or jobs have yet been verified.
- Created project-local Windows Python environment with NumPy, pytest,
  matplotlib and Paramiko. Linux GPU setup initially failed because uv could
  not rename an extracted wheel on the Windows mount; retry uses /tmp Linux
  storage. Setup logs are in logs/.
- Implemented four kernel families, 12 explicit cases, trusted references,
  edge/random/fresh validation, paired AB/BA CUDA event timing, confirmation,
  compiler-resource feedback, candidate process timeouts and AST integrity gate.
- Implemented general/specialized four-call loops, JSONL/state/source logs,
  greedy retention, all-attempt accounting, model token reporting, scripted
  engineering control, final validation fallback and 12 hardware cards.
- Implemented factorial paired analysis, case-cluster bootstrap, CSV output,
  performance/cost curves, and a balanced 144-search manifest.
- Ran the first test suite: **19 passed** (logs/tests_initial.log). These are
  software integrity tests, not GPU correctness or scientific experiment results.

Pending and subsequent results are recorded below as they occur. A written
implementation is not counted as a completed A100/LLM experiment.

## 2026-11-14 — access, deployment and GPU pilot

- Corrected password plus user-provided MFA successfully authenticated to
  Lonestar6. Allocation ASC26098 and scratch access were verified. Submitted
  A100 pilot job **3582290** to gpu-a100-small, then reduced its limit to 30
  minutes. It remains pending for resources; the scheduler initially estimated
  18:27 America/Chicago. This is an estimate, not a completed allocation.
- Verified replacement OpenAI credential with a read-only models request.
  Chose available pinned gpt-4.1-mini-2025-04-14 for the inexpensive pilot,
  with published input/output rates of $0.40/$1.60 per million tokens.
  The user explicitly approved exporting source/prompts/measurements to
  api.openai.com and a **$5 total pilot cap** after automatic approval review
  requested clarification. No key was persisted in project files.
- Installed local WSL PyTorch 2.11.0+cu128 and Triton 3.6.0. Linux /tmp was not
  persistent across WSL sessions; the final environment is .venv-linux using
  copy-mode installation. Logs retain all setup attempts and failures.
- Completed **24 actual RTX 4060 evaluations** in results/local_pilot_v2:
  23 passed, one fused layernorm_irregular evaluation failed the unchanged
  2e-5 absolute tolerance (maximum observed difference ~3.99e-5).
- Diagnosed cancellation from approximate division in the mean on a constant
  irregular-width row. Replaced mean division with tl.div_rn in both baseline
  and fused normalization kernels. Did not loosen tolerances. Launched
  results/local_pilot_v3 with standard 31-sample timing and source snapshots.
- Pilot v2 was exploratory: resource reporting instrumentation changed during
  execution; v3 records immutable candidate source and code hashes. v2 is kept
  as the failure trace, not the frozen evaluation dataset.
- Observed noisy local baseline-versus-itself timings (one paired ratio ~0.59).
  Laptop timings are development evidence and not A100 performance claims.
- Integrity tests now **21 passed**. An intermediate test invocation failed
  because Windows user/sandbox temp-directory ownership differed; rerunning
  with a fresh project-local pytest temp path passed (logs/tests_v5.log).
- User supplied empty GitHub repository. Initialized local main and pushed
  commit d7fd1e6. Lonestar6 HTTPS read access to the repository was verified;
  its SSH key is not registered with GitHub, so deployment uses HTTPS.
- The real LLM launcher was stopped while waiting for GPU validation, before
  making any inference requests, to reload the corrected source-policy code.
- Added logical traffic/FLOP throughput accounting and an Nsight counter probe.
  Peak fractions stay null until hardware-specific peak values are verified.
