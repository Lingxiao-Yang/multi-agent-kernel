# Results: Budget-Matched Multi-Agent Kernel Optimization

This branch contains **experiment outputs only**. The implementation,
configurations and tests are on [`main`](../../tree/main).

Every run directory has a `manifest.json` with protocol and source hashes
that link it to the exact code that produced it.

## Reports

| Report | Summary |
| --- | --- |
| [Completed guided repair round](reports/REPAIR_ROUND_REPORT.md) | 32 A100 searches: GPT-4.1 14/16 authored improvements vs mini 5/16; eight-attempt repair analysis; $13.41 cumulative usage |
| **[Source-generation pilot](reports/SOURCE_PILOT_REPORT.md)** | **Agents author Triton source. A100 job 3582814: 16 searches, 5 correct authored kernels (tiled `tl.dot` matmul in all 4 variants, 1.48–3.03×; fused bias + ReLU, 1.86×); no correct softmax or layernorm. Plus an 8-search laptop smoke** |
| [A100 configuration-selection pilot](reports/A100_PILOT_REPORT.md) | Lonestar6 A100-PCIE-40GB: 24/24 engineering evaluations correct; 24-search LLM pilot (2 cases × 4 variants × 3 seeds). Multi-agent variants reached 6.3× on matmul by enabling tensor cores; single agents never did. Configuration-only: no authored source |
| [GPU engineering pilot](reports/GPU_PILOT_REPORT.md) | 24/24 correct baseline/transformation evaluations on an RTX 4060 Laptop GPU, including a numerical failure that was found and fixed |
| [Four-variant infrastructure control](reports/CONTROL_MATRIX_REPORT.md) | Scripted end-to-end check of all controller paths on a real GPU |
| [Real-model smoke test](reports/LLM_SMOKE_REPORT.md) | One search per variant with `gpt-4.1-mini` on `softmax_irregular` |
| [Experiment log](reports/WORK_LOG.md) | Checkpoint of completed runs, cluster jobs and remaining work |

## Layout

| Path | Contents |
| --- | --- |
| `results/repair-mini-3584053`, `results/repair-strong-3584057` | Completed matched guided reduction pilots, 16 searches per model |
| `results/local_repair_smoke_v1` | Completed four-arm guided smoke on RTX 4060 |
| `results/a100_source_pilot_3582814` | A100 source-generation pilot: 16 searches with prompts, authored proposals, candidates and measurements (job 3582814) |
| `results/source-preflight-3582814` | A100 baseline-interface checks run before that campaign |
| `results/local_source_smoke_v1` | Laptop source-generation smoke: 8 searches |
| `results/a100_pilot_3582405` | A100 engineering pilot (Lonestar6 job 3582405) |
| `results/a100_llm_pilot_3582416` | A100 LLM pilot: 24 searches with full prompts, candidates and measurements (job 3582416) |
| `results/local_pilot_v2`, `results/local_pilot_v3` | Engineering pilot with predefined transforms: per-case timing and validation |
| `results/gpu_control_matrix` | Four-variant scripted controls (`single`, `single_knowledge`, `multi`, `multi_knowledge`) |
| `results/llm_local_smoke`, `results/llm_local_pilot` | Real-model runs on `softmax_irregular`: prompt/response JSONL, candidates, summaries |
| `reports/` | Write-ups, aggregate CSV/JSON files and preliminary plots |
| `logs/` | Run, environment-setup and API-access logs; `logs/a100/` holds Slurm output; `logs/lonestar6_jobs.json` records every cluster job |
| `viz/` | Standalone plotting scripts (numpy and matplotlib only) |

## Preliminary plots

| Plot | Shows |
| --- | --- |
| `reports/a100_source_pilot_3582814/speedup_by_variant.png` | A100 source pilot: final speedup by variant and case |
| `reports/a100_source_pilot_3582814/search_curves.png` | A100 source pilot: best speedup so far versus attempts, calls and tokens |
| `reports/a100_llm_pilot_3582416/speedup_by_variant.png` | A100 LLM pilot: final speedup per seed and geometric mean, by variant and case |
| `reports/a100_llm_pilot_3582416/search_curves.png` | A100 LLM pilot: best speedup so far versus attempts, calls and tokens |
| `reports/a100_pilot_3582405/pilot_speedups.png` | A100 engineering pilot: paired speedups of the predefined transformations |
| `reports/local_pilot_v3/pilot_speedups.png` | Paired speedups of the predefined transformations (reproducible cohort) |
| `reports/local_pilot_v2/pilot_speedups.png` | The same, for the exploratory cohort |
| `reports/gpu_control_matrix/search_curves.png` | Best speedup so far versus attempts, calls and tokens, for each variant |

To regenerate them:

```bash
pip install numpy matplotlib
python viz/plot_llm_pilot.py results/a100_llm_pilot_3582416/analysis/runs.csv \
  reports/a100_llm_pilot_3582416/speedup_by_variant.png
python viz/plot_pilot.py results/a100_pilot_3582405 reports/a100_pilot_3582405
python viz/plot_pilot.py results/local_pilot_v3 reports/local_pilot_v3
python viz/plot_search_curves.py results/gpu_control_matrix \
  reports/gpu_control_matrix/search_curves.png --controls
```

## Caveats

All results here are preliminary:

- The A100 source pilot uses one seed and 2 attempts per search; the A100
  configuration pilot uses 2 cases and 2 attempts per search.
- The laptop runs use one GPU that also drives the desktop display.
- The controls are scripted.

None of these is the primary 144-search study. Laptop and A100 results must
never be pooled.
