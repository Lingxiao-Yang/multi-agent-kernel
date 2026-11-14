# Four-arm GPU infrastructure control

All four controller variants completed two candidate attempts and eight
scripted role calls on `softmax_irregular`, seed 11. Candidate proposals were
identical deterministic configurations, with **zero LLM tokens**. Actual
GPU kernels were validated, timed, considered for retention, profiled and
checked again on fresh inputs by the full controller pipeline.

Measured final speedups were 1.806x (single), 1.967x (single + knowledge),
1.805x (multi), and 1.819x (multi + knowledge). These differences arise from
measurement/search noise despite identical proposal policy. They are not
role or knowledge effects. The control supports operational correctness of
the four paths and illustrates the need for paired repeated A100 runs.

There is one case and one seed in this infrastructure control; no meaningful
case-cluster confidence interval can be estimated. Raw sources, prompts,
results, latency samples, final checks and budget ledgers are retained in
`results/gpu_control_matrix`. The analysis is clearly labeled engineering
control and is excluded from real-model study summaries by default.
