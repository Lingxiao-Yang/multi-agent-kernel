# 代码仓结构重构计划

目标是把 `kernel_study/` 和 `tests/` 按职责分开，让依赖只朝一个方向走，并删掉已经归档的旧一代试验代码。重构只移动、拆分和删除，不改变搜索、评测和记账的行为。以下为已完成的结构调整，改动前的数字取自提交 `87a44ab`。重构后的代码已合入主线。A800 上的完整套件 75 项全部通过，六道题的起点与归档逐项一致。

## 一、现状

| 问题 | 依据 |
| --- | --- |
| 一个平铺的包里有六类职责 | `kernel_study/` 下 24 个模块没有子目录，包含任务定义、可信评测、本地搜索、上游适配、实验调度和报告。 |
| 四组循环导入 | `bench` 与 `public_cases`，`core` 与 `source_space`，`public_cases` 与 `source_space`，`public_referee` 与 `search`。这些循环靠函数内部的延迟导入维持。包内写在函数内部的包内导入共 31 处，其中一部分是为了绕开循环。最后一组表明评测方和被评测的搜索没有分开。 |
| `core.py` 职责过多 | 被 24 个模块中的 17 个导入。里面有 JSON 读写与哈希、事件日志、`Budget`、旧任务列表、旧的参数配置与 launch 说明、环境探测。 |
| 文件名与内容不符 | `reference_search.py` 是台账和计费 provider。`replay.py` 是旧任务重放，但噪声校准函数也在其中并被公共题预检使用。`public_cases.py` 同时有任务加载、起点拼装、native 渲染、压力输入、数值检查和评测主函数。`agent_campaign.py` 同时有上游适配、作业调度、命令行和代码哈希。 |
| 新旧两代代码交织 | 旧任务与公共题通过 `'public_id' in case` 在 `bench.py`、`source_space.py`、`search.py` 共 6 处分支。旧一代还包括 `kernels.py`、`pilot.py`、`repair_analysis.py`、三份旧配置、5 个作业文件和 24 个脚本中的 17 个。 |
| 入口有三种风格 | 一个 console script、7 个带 `__main__` 的包内模块、12 个靠 `sys.path.insert` 导入包的脚本。 |
| 测试按开发批次命名 | `test_integrity.py` 的 12 个测试覆盖 5 个模块，`test_public_campaign.py` 的 16 个测试覆盖 7 个模块，`test_reference_search.py` 把台账单元测试和需要 GPU 的集成测试放在一起。没有 `conftest.py` 和 marker，GPU 与上游依赖靠 11 处 `importorskip` 和 2 个环境变量选择，替换环境探测的样板重复 4 次。 |

## 二、目标结构

依赖顺序为：基础模块、`tasks`、`judge`、`agents`、`campaign`。后面的可以导入前面的，反过来不行。`judge` 不导入 `agents`。

```
kernel_study/
  records.py        JSON、哈希、事件日志、环境探测、代码哈希
  budget.py         Budget 与 Ledger
  providers.py      ChatProvider 与 MeteredChat
  tasks/
    kernelbench.py  实验配置、上游检出、题目输入、张量布局
    starts.py       共同起点的拼装与 native 模块渲染
    kernels.py      起点使用的设备函数
  judge/
    interface.py    声明式 launch 的校验与执行
    policy.py       设备源码白名单
    correctness.py  数值检查、压力输入、FP64 参考
    timing.py       配对计时与噪声校准
    evaluate.py     评测主函数、子进程入口、run_worker
    select.py       终选、候选复核、两方终选的直接配对
  agents/
    local.py        本地四阶段搜索
    edit_gate.py    结构改动检查
    guidance.py     共享的 API 正确性指引
    knowledge.py    知识卡选择
    kernelagent.py  上游 greedy 的适配
  campaign/
    jobs.py         作业表、重跑编号、登记与运行
    preflight.py    零推理预检
    report.py       配对汇总与验收判定
    factorial.py    配对析因与 bootstrap
  cli.py            preflight、run、report 三个子命令
```

分层参照了两个已锁定的上游仓库。KernelBench 在一个包里按职责一文件一事（数据集、评测、计时、打分、静态检查各自独立），测试文件跟随被测模块命名。KernelAgent 按流水线阶段分子包（profiling、benchmarking、prescribing、orchestrator、searching），模型接入单独成包。

### 现有内容的去向

| 现在的位置 | 新位置 |
| --- | --- |
| `core` 的 `read_json`、`write_json`、`digest`、`utcnow`、`event`、`environment` | `records.py` |
| `core.Budget`，`reference_search` 的 `Ledger`、`interruption` | `budget.py` |
| `providers` 的 `Reply`、`ChatProvider`，`reference_search.MeteredChat` | `providers.py` |
| `public_cases` 的 `experiment`、`upstream`、`load_module`、`task_path`、`matvec_path`、`task_inputs`、`layout` | `tasks/kernelbench.py` |
| `public_cases` 的 `device_function`、`start`、`render_native`、`load_native`、`local_forward` | `tasks/starts.py` |
| `public_kernels` 全部 | `tasks/kernels.py` |
| `source_space` 的 `validate_plan`、`contract`、`prepare`，以及 `_body_shapes` 中检查 launch 指向的内核是否存在、参数个数是否匹配的部分 | `judge/interface.py` |
| `source_policy` 全部 | `judge/policy.py` |
| `public_cases` 的 `stress_input`、`fp64_chunks`、`check_forward` | `judge/correctness.py` |
| `bench` 的 `time_pair`、`summarize`、`resource_summary`，`replay` 的 `threshold_from_controls`、`confirmed`、`noise_trials`、`calibrate`、`check_noise`、`first_call` | `judge/timing.py` |
| `public_cases` 的 `bind`、`validate`、`evaluate`，`worker.main`，`search.run_worker` | `judge/evaluate.py` |
| `public_referee` 全部 | `judge/select.py` |
| `search` 的 `search`、`compact_result` 和提示词常量 | `agents/local.py` |
| `source_space` 的搜索约定文本、`check_edit`、`_inline_straight_line_temporaries`，以及 `_body_shapes` 中做结构归一化的部分 | `agents/edit_gate.py` |
| `api_guidance`、`knowledge` | `agents/guidance.py`、`agents/knowledge.py` |
| `agent_campaign` 的 `native_manager`、`native_adapters`、`native_outcome`、`run_native` | `agents/kernelagent.py` |
| `agent_campaign` 的 `jobs`、`attempt_path`、`latest_attempt`、`job_name`、`register`、`run_job` | `campaign/jobs.py` |
| `agent_campaign.code_hashes` | `records.py` |
| `public_preflight` 全部 | `campaign/preflight.py` |
| `reference_report` 全部 | `campaign/report.py` |
| `analysis` 的 `geometric_mean`、`paired_effects` | `campaign/factorial.py` |

`_body_shapes` 是唯一需要拆开的函数。它现在既替评测器检查 launch 是否有效，又替本地搜索判断改动是否只是换名。前一半归 `judge`，后一半归 `agents`，现有的参数个数、缺失内核和三类表面改写的测试覆盖这两半。`reference_report.geometric` 与 `analysis.geometric_mean` 是同一计算，合并后只留一个。

### 删除

旧任务的重放结果已归档，可从登记的提交取回源码。

| 类别 | 内容 |
| --- | --- |
| 包内模块 | `kernels.py`、`pilot.py`、`repair_analysis.py`、`replay.py` 中的重放部分、`metrics.py`、`bench.py` 中旧任务的输入、参考、`prepare`、`validate`、`evaluate`，`core` 中的 `cases`、`baseline_config`、`validate_config`、`launch_contract`，`providers.ScriptedProvider`，`analysis` 中的 `validate_cohort`、`checkpoint_curve`、`analyze`，`search` 和 `source_space` 中的参数选择模式与旧任务分支，`cli.py` 的现有子命令 |
| 配置 | `configs/cases.json`、`configs/primary_manifest.json`、`configs/repair_round.json` |
| 作业文件 | `cluster/` 下现有 5 个文件 |
| 脚本 | `audit_source_runs.py`、`check_round_models.py`、`compare_repair_rounds.py`、`control_matrix.py`、`launch_llm_pilot.py`、`llm_pilot.py`、`local_llm_smoke.py`、`local_repair_smoke.py`、`local_source_smoke.py`、`plot_pilot.py`、`probe_completion.py`、`probe_model.py`、`report_progress.py`、`run_campaign.py`、`run_manifest_entry.py`、`run_repair_round.py`、`source_preflight.py` |

保留的脚本是运维用途的 `ls6_remote.py`、`probe_counters.py`、`audit_artifacts.py`、`package_project.py`、`publish_results.py`、`setup_local_linux.sh`、`inspect_linux.py`。`probe_counters.py` 调用的 `profile_target.py` 目前使用旧的 softmax 任务，改为使用一道公共题的起点。旧阶段的说明文档（`NEXT_EXPERIMENT_PLAN.md`、`REPAIR_ROUND.md`、`SOURCE_EXPERIMENTS.md`、`WORK_LOG.md`）移入 `docs/archive/`。

## 三、测试

```
tests/
  conftest.py
  test_layering.py
  test_budget.py
  test_providers.py
  tasks/     test_kernelbench.py  test_starts.py
  judge/     test_interface.py  test_policy.py  test_correctness.py  test_timing.py  test_select.py
  agents/    test_local.py  test_edit_gate.py  test_guidance.py  test_knowledge.py  test_kernelagent.py
  campaign/  test_jobs.py  test_preflight.py  test_report.py  test_factorial.py
  integration/
```

`conftest.py` 登记两个 marker 和三个 fixture。`gpu` 表示需要 CUDA 和 Triton，没有可用设备时由收集钩子统一跳过。`upstream` 表示需要锁定版本的上游检出。fixture 提供伪造的任务字典、替换后的环境探测，以及从环境变量读取的预检目录（未设置时跳过）。本地开发运行 `pytest -m "not gpu and not upstream"`，目标 GPU 上运行完整套件。`integration/` 只放需要真实设备或预检数据的测试。

`test_layering.py` 解析包内导入，断言没有循环，且每个模块只导入同层或更靠前的层。它防止之后的改动把结构重新缠回去。

| 现有文件 | 去向 |
| --- | --- |
| `test_integrity.py` | 预算 3 项到 `test_budget.py`，源码白名单到 `judge/test_policy.py`，知识卡到 `agents/test_knowledge.py`，析因 2 项到 `campaign/test_factorial.py`。旧任务矩阵、参数配置、脚本化 provider、旧 launch 说明 4 项随旧代码删除。 |
| `test_source_space.py` | launch 校验与参数个数到 `judge/test_interface.py`，三类表面改写与结构改动到 `agents/test_edit_gate.py`，空源码计入失败尝试到 `agents/test_local.py`。cohort 检查随 `validate_cohort` 删除。 |
| `test_search.py`、`test_interrupted_search.py` | 改写后进入 `agents/test_local.py`，见下文。 |
| `test_paired_timing.py` | 库对比使用同轮样本改写到 `judge/test_timing.py`，新分配输出到 `judge/test_interface.py`。历史口径识别随 `metrics` 删除。 |
| `test_baseline_replay.py` | 两轮均须过线的规则到 `judge/test_timing.py`，归档重放随 `replay` 删除。 |
| `test_public_baseline.py` | 按被测函数分到 `tasks/` 两个文件、`judge/test_correctness.py` 和 `campaign/test_preflight.py`。 |
| `test_public_campaign.py`、`test_reference_search.py`、`test_reference_report.py` | 台账到 `test_budget.py`，请求内容到 `test_providers.py`，终选与直接配对到 `judge/test_select.py`，压力输入与输出预填到 `judge/test_correctness.py`，上游适配到 `agents/test_kernelagent.py`，作业表、重跑和冻结到 `campaign/test_jobs.py`，比值与判定到 `campaign/test_report.py`，两个需要真实 GPU 的测试到 `integration/`。 |
| `test_repair_round.py` | 指引 2 项到 `agents/test_guidance.py`，其余 2 项随旧代码删除。 |
| `test_metrics.py`、`test_providers.py` | 前者随 `metrics` 删除，后者保留。 |

有三项测试目前借旧的参数选择模式来验证共享逻辑：错误候选不会被保留、未见输入终验失败后回退、接口失败的搜索不算完成。删除旧模式前先把它们改写到声明式源码模式，用返回固定源码的 provider 替身驱动。改写后的测试先在旧代码上通过，再删除旧模式。

## 四、步骤

每一步是一个或几个只做一件事的提交，移动文件使用 `git mv`。每步结束时完整测试通过，结果记入实验日志。

| 步骤 | 内容 | 完成的标志 |
| --- | --- | --- |
| 0 | 记录基线：CPU 环境与目标 GPU 上的测试通过数，6 道公共题的起点源码哈希和 launch | 数字写入实验日志 |
| 1 | 加 `conftest.py` 和 marker，按上表拆分测试文件，只移动测试函数 | 通过的测试数与基线相同 |
| 2 | 把三项借旧模式覆盖共享逻辑的测试改写到声明式源码模式 | 改写后的测试在旧代码上通过 |
| 3 | 删除旧一代代码、配置、作业文件和脚本 | `'public_id' in case` 的分支全部消失，四组循环剩两组 |
| 4 | 消除剩余循环：`run_worker` 与评测归到一处，`layout` 归任务，`source_space` 拆成接口与改动检查 | 包内没有循环导入，没有函数内部的包内延迟导入 |
| 5 | 建立子包并移动文件，更新代码哈希的文件清单，加入 `test_layering.py` | 分层测试通过，起点哈希与步骤 0 相同 |
| 6 | 统一入口为 `cli.py` 的三个子命令，`profile_target.py` 改用公共题起点 | 包内除评测子进程入口外没有 `__main__`，脚本不再修改 `sys.path` |
| 7 | 更新 README 的目录表与命令，归档旧文档 | README 中的每条命令可以直接运行 |

## 五、验收

- 步骤 1 之后每一步，未删除的测试全部通过，通过数只因上表列明的删除而减少。
- 6 道公共题的起点源码哈希和 launch 与预检记录的 `device_source_hash`、`native_source_hash`、`launch` 一致。起点源码按函数名从文件中截取，移动文件不应改变它。
- `test_layering.py` 通过。
- 在目标 GPU 上运行一次完整套件，包括真实上游管理器的启动和两个入口的独立终验。

## 六、时点与约束

development 阶段会登记代码哈希，登记之后移动或改名任何被哈希的文件都会使冻结和留出阶段的检查失败。重构要么在第一笔付费的 development 作业之前完成，要么等留出阶段结束。

代码哈希目前是一份手写的 16 个模块名清单，本地搜索写入运行记录的哈希则只扫描包的顶层目录。建立子包时两处改为同一个函数，按目录递归收集 `tasks/`、`judge/`、`agents/`、基础模块和 `campaign/jobs.py`、`campaign/preflight.py`，不包含 `campaign/report.py` 和 `campaign/factorial.py`。按目录收集可以避免新增文件漏登记；若保留顶层扫描，文件进入子目录后哈希会变成空集而不报错。

重构期间不混入行为改动。其他进行中的修改先合入，重构提交之后再继续。
