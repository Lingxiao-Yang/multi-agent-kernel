# Results: Budget-Matched Multi-Agent Kernel Optimization

This branch contains **experiment outputs only**. The implementation,
configurations and tests are on [`main`](../../tree/main).

Every run directory has a `manifest.json` with protocol and source hashes
that link it to the exact code that produced it.

## Reports

| Report | Summary |
| --- | --- |
| [GPU engineering pilot](reports/GPU_PILOT_REPORT.md) | 24/24 correct baseline/transformation evaluations on an RTX 4060 Laptop GPU, including a numerical failure that was found and fixed |
| [Four-variant infrastructure control](reports/CONTROL_MATRIX_REPORT.md) | Scripted end-to-end check of all controller paths on a real GPU |
| [Real-model smoke test](reports/LLM_SMOKE_REPORT.md) | One search per variant with `gpt-4.1-mini` on `softmax_irregular` |
| [Experiment log](reports/WORK_LOG.md) | Checkpoint of completed runs, cluster jobs and remaining work |
| [Compute options](reports/COMPUTE_OPTIONS.md) | Assessment of a backup A100 provider |

## Layout

| Path | Contents |
| --- | --- |
| `results/local_pilot_v2`, `results/local_pilot_v3` | Engineering pilot with predefined transforms: per-case timing and validation |
| `results/gpu_control_matrix` | Four-variant scripted controls (`single`, `single_knowledge`, `multi`, `multi_knowledge`) |
| `results/llm_local_smoke`, `results/llm_local_pilot` | Real-model runs on `softmax_irregular`: prompt/response JSONL, candidates, summaries |
| `reports/` | Write-ups, aggregate CSV/JSON files and preliminary plots |
| `logs/` | Run, environment-setup and API-access logs |
| `viz/` | Standalone plotting scripts (numpy and matplotlib only) |

## Preliminary plots

| Plot | Shows |
| --- | --- |
| `reports/local_pilot_v3/pilot_speedups.png` | Paired speedups of the predefined transformations (reproducible cohort) |
| `reports/local_pilot_v2/pilot_speedups.png` | The same, for the exploratory cohort |
| `reports/gpu_control_matrix/search_curves.png` | Best speedup so far versus attempts, calls and tokens, for each variant |

To regenerate them:

```bash
pip install numpy matplotlib
python viz/plot_pilot.py results/local_pilot_v3 reports/local_pilot_v3
python viz/plot_search_curves.py results/gpu_control_matrix \
  reports/gpu_control_matrix/search_curves.png --controls
```

## Caveats

All results here are preliminary. They come from:

- a small set of cases;
- a laptop GPU;
- scripted controls or single-seed model runs.

They are not the A100 study and must not be pooled with it.
