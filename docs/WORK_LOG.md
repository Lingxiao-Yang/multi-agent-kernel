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
