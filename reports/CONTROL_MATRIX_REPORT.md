# Four-Variant GPU Infrastructure Control

**Purpose.** This control checks that all four controller paths run correctly
end to end on a real GPU. It is not evidence about roles or knowledge.

**Setup.**

- Case: `softmax_irregular`, seed 11.
- Every variant made two candidate attempts and eight role calls, served by a
  scripted provider.
- All variants proposed identical deterministic configurations, so **no LLM
  tokens** were used.
- Real GPU kernels went through the full pipeline: validation, timing,
  retention, profiling and a final check on fresh inputs.

## Results

| Variant | Final speedup |
| --- | ---: |
| `single` | 1.806× |
| `single_knowledge` | 1.967× |
| `multi` | 1.805× |
| `multi_knowledge` | 1.819× |

Every variant proposed exactly the same thing. The spread in final speedup
therefore comes from measurement and search noise, **not** from roles or
knowledge. This shows that the four paths work, and it shows why the A100
study needs paired, repeated runs.

## Scope

With one case and one seed, no meaningful case-cluster confidence interval
can be computed. `results/gpu_control_matrix` holds the complete record:

- source, prompts and results;
- latency samples and final checks;
- budget ledgers.

The analysis output is labeled as an engineering control. By default it is
left out of real-model summaries.

Plot: [`gpu_control_matrix/search_curves.png`](gpu_control_matrix/search_curves.png).
