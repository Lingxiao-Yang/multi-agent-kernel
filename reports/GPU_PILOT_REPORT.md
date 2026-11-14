# GPU engineering pilot — 2026-11-14

The corrected implementation passed **24/24** baseline/transformation
evaluations across all 12 proposed cases on the local **RTX 4060 Laptop GPU**.
Every evaluation used five fresh random seeds plus zero, constant, large and
negative input cases, unchanged numerical tolerances, and mutation/output
overwrite checks. This is development evidence, not the planned A100 study.

## Results and scope

The latest immutable cohort is `results/local_pilot_v3`. Its manifest records
the kernel snapshot, code hashes, environment and timing protocol. Timing
uses 31 samples after warm-up, 10 inner repetitions, alternating candidate
and baseline order, and a second confirmation for apparent gains.

| Family | Measured transformation | Paired speedup range |
|---|---|---:|
| Matrix multiplication | SIMT to tiled tensor-core dot | 8.28–21.23x |
| Row softmax | Separate statistics/application to fused | 1.58–1.93x |
| Layer normalization | Separate statistics/application to fused | 1.37–1.85x |
| Bias + ReLU | Two launches to one fused launch | 1.84–3.39x |

These transformations were predefined engineering controls. They are **not
LLM-generated optimization results** and support no comparison of specialist
roles or supplied knowledge. Matmul gains are relative to the proposal's
simple non-tensor-core baseline, not to cuBLAS/PyTorch. CSV files include the
separate library latency reference.

## Failure discovered and repaired

The exploratory cohort `local_pilot_v2` passed 23/24 evaluations. Fused layer
normalization at shape `[513, 781]` failed on a constant-input edge case, with
a maximum reported absolute discrepancy about `3.99e-5` against an absolute
tolerance `2e-5`. Approximate division in the mean introduced a small residual
that normalization amplified. Both layer-normalization kernels now compute
the mean with Triton's round-to-nearest division, `tl.div_rn`.

The same tolerances and edge tests then passed in v3. The rejected output,
traceback and original pilot results are retained. No failed candidate was
presented as correct, and the tolerance was not relaxed to hide the failure.

## Measurement limitations

- This laptop also drives a Windows desktop; clock changes, display workloads
  and WSL launch overhead limit timing stability. One v2 baseline-versus-itself
  comparison was approximately 0.59x. Most v3 self-comparisons were nearer 1,
  but they still show variation. Use raw samples and repeat on allocated A100.
- CUDA-event timing of repeated eager launches can include GPU idle time
  while waiting for host submissions. It represents this launch path; it is
  not an isolated hardware instruction throughput measurement.
- Compiler registers/shared-memory reports are available. Hardware-counter
  permissions have not yet been verified, and no counter-derived bottleneck
  diagnosis or hardware roofline fraction is claimed.
- The initial v2 pilot ran while reporting instrumentation was still being
  developed. v3 is the reproducible cohort with fixed candidate source.

## Remaining project work

The code, four-arm controller, knowledge cards, measurement harness, analysis
pipeline and deployment scripts are implemented. The real LLM pilot is
separate and its results will be reported only after completion. Lonestar6
job 3582290 has been submitted but has not yet supplied an A100 result.

The full 144-search study, post-pilot A100 protocol freeze, targeted ablations,
hardware-specific headroom analysis and final scientific report remain.
These outstanding experiments prevent treating the software implementation
alone as proof of a particular percentage of total research completion.
