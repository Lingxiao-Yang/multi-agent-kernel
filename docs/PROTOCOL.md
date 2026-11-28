# Public kernel study protocol

The role comparison asks what changes when four specialist system prompts replace one general system prompt within the same four-stage search. The external qualification compared the local `single` loop with one worker of KernelAgent's `greedy`. The original proposal described 144 searches over twelve tasks. The public campaign uses the budgets and measurements below. [Original proposal](proposal_final.pdf) · [Baseline qualification](AGENT_BASELINE_PLAN.md)

## One search

A case fixes a KernelBench Level 1 problem, public input shapes and dtype, tolerances, verifier and native Triton start. The start has one Python entry point, `kernel_function`. Its wrapper may inspect metadata, allocate output and temporary tensors, launch kernels and return the output. The device code performs the numerical work. Changes to launch parameters, constants and algorithms are all eligible. [Cases and frozen environment](../configs/agent_baseline.json)

1. The two local variants receive the same starting source, task, verifier, hardware specifications and measured starting counters. One capture of the last kernel supplies both sides of each case–seed pair. The search can profile a new incumbent. A run records the source and counter hashes, prompts, responses and measurements.
2. Analysis gives bottleneck hypotheses and supporting measurements. Advice describes at most three proposed edits or trade-offs in short sentences. Implementation returns a complete module in `source` and its `rationale`. Review returns `diagnosis`, `outcome` and `next_step` after the judge runs. Analysis, advice and review use prose. The judge decides retention.
3. Each local search permits four candidate attempts, 16 model calls and up to 500,000 tokens. Each attempt writes its own usage ledger in its search directory. An unchanged proposal, compilation failure or wrong output consumes a candidate slot. An accounting or storage failure interrupts the search and invalidates it for scoring. A transport interruption permits one fresh attempt within the same search's dollar allowance. The first attempt's committed amount, including unresolved reservations, is deducted before retrying.
4. The common judge screens each proposal before import, evaluates the full output on search seed 101 and checks that inputs remain unchanged. It times a correct module against the incumbent. A candidate must exceed a 1.01× latency ratio in each of two paired measurements to replace the incumbent.
5. The final judge replays proposals in submission order on seeds 10001–10005 and on zeros, constants, large and negative inputs. LayerNorm also uses a large-offset input. Pressure tests compare against an FP64 reference with the configured tolerance and eager's measured numerical error. The judge rechecks the selected candidate against the original start. It records the actual source and latency when the search falls back to the start.

## Timing and controls

The A100 protocol measures full forwards, including output allocation. Each side runs once before measurement. The pair then accumulates at least 2,610 ms of GPU warm-up. CUDA events time nine samples of one forward per side, alternating AB/BA order. Each record retains the raw samples, median, quartiles and indices above 1.05 times that side's final warm-up median. All nine samples contribute to the reported median.

Each speed ratio uses both samples from its own paired measurement. The finalist has separate pairs against its incumbent, eager and `torch.compile` in `max-autotune-no-cudagraphs` mode. Compilation time is recorded separately. Calibration and an independent hold each compare the frozen start against itself 20 times at 1.01, with two rounds per trial. Exact FP32 matrix multiplication disables TF32. Launch sweeps keep the frozen algorithm and dtype and repeat finalist measurements three times. [Timing](../kernel_study/judge/timing.py) · [GPU evaluator](../kernel_study/judge/evaluate.py)

## Registration and reporting

The formal comparison uses the original starts for matvec (4), softmax (23), LayerNorm (40) and FP32 matrix multiplication (2). Five fresh seeds per case give 20 pairs and 40 searches for `single` and `multi` without knowledge cards.

Before submission, register the merged mainline commit, these four case IDs, the five seed values and the knowledge-card choice in a fresh directory. Check that the sum of all registered search allowances fits the remaining campaign budget. Each search in the no-card comparison has a $1 allowance, giving a $40 maximum commitment for 40 searches. The matched arms share the snapshot, prices, sampling settings, information and candidate budget. Each case and seed has a recorded run order. The searches and direct finalist measurement run on the same A100 UUID. [Campaign entry point](../kernel_study/campaign/jobs.py)

Count a search as complete when its candidate history and all four rounds are present, each request has one matching reply and ledger settlement, and the shared starting evidence matches. For each task, count searches whose final kernel confirms a gain over the start, and searches that return the start. Report direct finalist latencies in ms, costs and where each endpoint falls relative to the task’s launch-parameter references. Show best correct latency at matched candidate counts and actual dollar spend. For each requested contrast, take the geometric mean of the direct latency ratios within a case, then weight the cases equally. An incomplete comparison reports its observed pairs and leaves the overall ratio empty. If knowledge-card variants are registered, compare roles within each knowledge condition and cards within each role. [Campaign report](../kernel_study/campaign/report.py)

The first 20-pair batch used one SQLite ledger across compute nodes. Accounting errors interrupted eight searches and were recorded as candidate failures in eleven others. Four pairs received all 16 replies on both sides. Retain the raw first-batch data and these four descriptive pairs in the experiment records. Score the full rerun with the original cases and seeds from its fresh registration.

## Analysis in the original proposal

The original twelve-case design specified 20 candidates and 100 calls per search, with three seeds for each of four role and knowledge conditions. Its source-only and configuration-control runs retain their original measurement protocols. The public role comparison reports results separately by model snapshot, provider, source and GPU.

The four-arm analysis uses paired log speedups. It averages seed effects within each case and resamples whole cases 5,000 times with seed 42. It reports the role effect without cards, the card effect for the single agent, the role effect with cards and the interaction `(multi_knowledge × single) / (multi × single_knowledge)`. Computing these quantities requires complete matched observations. [Factorial analysis](../kernel_study/campaign/factorial.py)

The original proposal also calls for role and knowledge-category ablations on two cases, inspection of example mechanisms, uncertainty intervals, cost curves, library comparisons, measured throughput and justified roofline bounds. Those experiments and their report follow the model and protocol freeze. The qualification outcomes and smoke latencies describe their observed tasks and seeds.
