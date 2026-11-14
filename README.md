# Results — budget-matched multi-agent kernel optimization

This branch holds **experiment outputs only**. The implementation, configs and
tests live on [`main`](../../tree/main). Every run directory carries a
`manifest.json` with protocol/source hashes linking it back to that code.

## Layout

| Path | Contents |
| --- | --- |
| `results/local_pilot_v2`, `v3` | Laptop-GPU engineering pilot: predefined transforms, per-case timing and validation |
| `results/gpu_control_matrix` | Four-arm scripted controls (`single`, `single_knowledge`, `multi`, `multi_knowledge`) — engineering checks, not LLM evidence |
| `results/llm_local_smoke`, `llm_local_pilot` | Real-model smoke tests on `softmax_irregular` (prompts/responses JSONL, candidates, summaries) |
| `reports/` | Write-ups (`GPU_PILOT_REPORT.md`, `CONTROL_MATRIX_REPORT.md`, `LLM_SMOKE_REPORT.md`, …), aggregate CSV/JSON and **preliminary plots** |
| `logs/` | Run, setup and access-probe logs |
| `viz/` | Standalone plotting scripts (numpy + matplotlib only) |

## Preliminary plots

- `reports/local_pilot_v2/pilot_speedups.png`, `reports/local_pilot_v3/pilot_speedups.png`
- `reports/gpu_control_matrix/search_curves.png`

## Regenerating plots

```bash
pip install numpy matplotlib
python viz/plot_pilot.py results/local_pilot_v3 reports/local_pilot_v3
python viz/plot_search_curves.py results/gpu_control_matrix reports/gpu_control_matrix/search_curves.png --controls
```

These are preliminary: small case suite, laptop GPU and scripted controls.
Do not read them as the A100 LLM study.
