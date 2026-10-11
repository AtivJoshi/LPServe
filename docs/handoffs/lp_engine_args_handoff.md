# LP Scheduler EngineArgs Integration Handoff

## Objective

Let `LLMEngine.from_engine_args(scheduler_type="lp", ...)` build the
existing `LPSchedulerConfig` through the normal Python engine factory:

```text
EngineArgs -> create_engine_configs() -> LPSchedulerConfig
           -> existing LLMEngine factory -> registered LPScheduler
              and VLLMBlockSpaceManager
```

Python engine configuration only. Benchmark configuration, command-line
parsing, dynamic utility policies, and GPU validation are out of scope.

## Changed paths

- `sarathi/engine/arg_utils.py`: LP fields, early validation, LP factory
  branch.
- `tests/test_lp_engine_args.py` (new, 8 CPU tests). `tests/` matches an
  ignore rule; `git status` shows it only with `--ignored`.
- `docs/lp_scheduler_design.md` §13 live-scheduler integration: one
  `EngineArgs` entry-point item.
- `docs/handoffs/lp_engine_args_handoff.md` (this file).
- `validation_output/lp_engine_args/` (evidence).

`sarathi/engine/llm_engine.py`, `sarathi/config.py`, the registries, the
mathematical layer, mapper, executor, scheduler, and `docs/math/` are
unchanged.

## Interface

New optional `EngineArgs` fields, each defaulting to `None`: `lp_b_max`,
`lp_c_max`, `lp_s_max`, `lp_memory_reserve`, `lp_decode_memory_policy_id`,
`lp_decode_utility`, `lp_prefill_token_utility`, `lp_preemption_penalty`,
`lp_numerical_policy`.

With `scheduler_type="lp"`, all nine are required and passed by keyword to
`LPSchedulerConfig` (`b_max`, `c_max`, ..., `numerical_policy`). The
`NumericalPolicy` object is passed unchanged. There are no LP defaults.

Inherited shared engine settings, with their existing defaults and
resolution: `max_num_seqs` (resident limit), the model configuration's
resolved `get_max_model_len()`, and `pipeline_parallel_size` (stage count).
These are not LP research choices. `lp_s_max` stays separate from
`max_num_seqs`; nothing is derived from another field.

`EngineArgs.__post_init__` writes `config.yml` before
`create_engine_configs()` runs. LP validation therefore runs at the start of
`__post_init__`, before that write and before model metadata loading:

| Condition | Error |
|---|---|
| Any LP field `is None` | `ValueError` naming every missing field |
| `lp_numerical_policy` not a `NumericalPolicy` | `TypeError` with the received type |
| `pipeline_parallel_size != 1` | `ValueError` |
| `max_num_batched_tokens` supplied | `ValueError` directing the caller to `lp_b_max` |

Zero values pass, because the missing-value check uses `is None`. Value-domain
validation remains in the mapper and mathematical layer. `NumericalPolicy` is
imported locally inside the LP check; importing `arg_utils` for other
schedulers does not load `lp_relaxation_scheduler` (checked). Other schedulers
keep their existing argument handling.

`config.yml` records the LP fields as plain values. `dataclasses.asdict`
turns the nested frozen `NumericalPolicy` dataclass into an ordinary mapping,
without a Python-object YAML tag. No custom serializer was added.

## Tests

`tests/test_lp_engine_args.py` uses real `EngineArgs`, `CacheConfig`,
`ParallelConfig`, `LPSchedulerConfig`, `MetricsConfig`, and registries. It
mocks only `ModelConfig` (resolved length 32, 2 layers) and, in facade tests,
`BaseLLMEngine` and `PipelineParallelLLMEngine`. Config files go to a
temporary directory.

Fixture (provisional test inputs, not defaults): `max_num_seqs=4`, resolved
`max_model_len=32`, one stage, `lp_b_max=8`, `lp_c_max=4`, `lp_s_max=3`,
reserve 1, `conservative_one_block_v1`, utilities and penalty 1, and
`NumericalPolicy("lp_relaxation_mvp_v1", 1e-7, 1e-6, 1e-9, 1e-9)`.

Covered: value preservation and shared-setting mapping; exact policy-object
identity; each missing field (subtests); zero reserve and penalty; wrong
policy type, two stages, and `max_num_batched_tokens` fail through the real
`LLMEngine.from_engine_args` before `ModelConfig`, engine construction, or
`config.yml`; recorded YAML contents; VLLM and Sarathi paths without LP
inputs; facade selects `BaseLLMEngine` with five positional configurations;
registries resolve `LPScheduler` and `VLLMBlockSpaceManager`.

## Environment and results

Unity `gpu048.unity.rc.umass.edu`, `SLURM_JOB_ID=65529855` (gpu-preempt),
modules `uri/main`, `Python/3.10.8-GCCcore-12.2.0`, `CUDA/12.1.1`,
`env/bin/python` 3.10.8, numpy 2.2.6, scipy 1.15.3, torch 2.3.0+cu121,
transformers 4.57.6, PyYAML 6.0.3. Base `09e0515` plus uncommitted changes.
No model, CUDA, or Ray initialization.

Final run `validation_output/lp_engine_args/20261011T002642Z/`:

| Command (`python -B -m unittest discover -s tests -p ... -v`) | Result |
|---|---|
| `test_lp_engine_args.py` | 8 passed |
| `test_lp_relaxation_scheduler.py` | 5 passed |
| `test_lpserve_state_mapping.py` | 7 passed |
| `test_lpserve_plan_execution.py` | 13 passed |
| `test_lp_scheduler.py` | 16 passed |

Static checks (`6-static.log`): `compile()` syntax check of both Python files
passed; the non-LP import check passed; `git diff --check` was clean; there
were no phase-label matches; and no repository-root `config.yml` was
created. The handoff whitespace check is in `8-handoff-whitespace.log`,
because the handoff was written after the run.

That directory also contains `environment.txt` (host, allocation, modules,
versions, HEAD, working-tree state, SHA-256 of changed and relevant factory
and configuration files), `tested_arg_utils.diff`, and
`tested_test_lp_engine_args.py`. The `.log` files and both `tested_*` files
match ignore rules (`*.log`, `test*`); `git status` shows them only with
`--ignored`.

A failed earlier attempt was retained at
`validation_output/lp_engine_args/20261011T002508Z/` (`ATTEMPT_NOTE.txt`).
The VLLM compatibility case passed `max_num_batched_tokens=16` with model
length 32. The existing `VLLMSchedulerConfig` assertion requires the budget
to be at least the model length. That was a test-fixture error; the value was
changed to 32. Production code did not change between the two runs.

## Limitations

- These are mocked CPU factory checks only. Real engine initialization, Ray
  workers, model loading, and GPU execution through `EngineArgs` are
  unverified and need their own scoped check.
- The repository root must still be on `PYTHONPATH` for the LP import.
- Benchmark configuration, command-line parsing, and dynamic utilities remain
  deferred. D-01–D-04 and other OPEN decisions are unchanged. The
  architecture audit was not repinned.
- After the run, `PROJECT_GUIDE.md` and `docs/project_status.md` were updated
  to record this integration. Those documentation-only edits did not change
  the tested code. The historical CPU-milestone wording in
  `docs/project_status.md` was left as written.

## Correction: positional field order (2026-10-11)

**Defect.** In the reviewed commit `9c894d9`, the nine `lp_*` fields were
declared before `write_metrics`. `EngineArgs` is a dataclass, so this
shifted every later positional argument. A legacy positional call supplying
`write_metrics=False` and an output directory bound those values to
`lp_b_max` and `lp_c_max`. It left `write_metrics=True` and
`output_dir="."`, which would cause an unexpected `config.yml` write.

**Correction.** The LP field block was moved, unchanged, to directly after
`attention_backend`. All 51 fields from `09e0515` now form an identical
prefix (same names, annotations, and defaults), as shown by an AST comparison.
LP keywords, validation, policy identity, and configuration construction are
unchanged. No constructor or compatibility code was added.

**Regression.** `test_legacy_positional_metrics_arguments_keep_their_bindings`
freezes the `09e0515` order through `output_dir`. It builds a non-LP
`EngineArgs` positionally with `write_metrics=False` and a temporary path,
checks every binding, and checks that `open` and `os.makedirs` in
`arg_utils` (both mocked) are not called and no directory is created.

Evidence is in `validation_output/lp_engine_args/20261011T004551Z_positional_order/`, from `gpu048.unity.rc.umass.edu`, job 65529855,
`env/bin/python` 3.10.8, base `9c894d9` plus uncommitted changes.

- `before_correction/2-legacy_positional_expected_failure.log`: before the
  move, the test failed as expected (`True != False : write_metrics`, exit 1).
  No repository-root `config.yml` was created.
- `before_correction/1-legacy_positional_attempt1_test_bug.log` and
  `ATTEMPT1_NOTE.txt`: an earlier failure was caused by the test, not the
  defect. The frozen tuple passed `tokenizer=None`, which `__post_init__`
  replaces with the model name. The fixture now passes the model name.
- `after_correction/1-test_lp_engine_args.log`: 9 passed, exit 0. The
  original 8 tests are unchanged.
- `after_correction/2-static.log`: `git diff --check` was clean, the
  in-memory syntax check passed, and the legacy-prefix AST comparison passed.
  No repository-root `config.yml` was created. Its comment calls the test
  file ignored; it is tracked since `9c894d9`, so `git diff --check`
  covers it.
- `after_correction/environment.txt` and `tested_changes.diff` give the
  host, modules, working-tree state, tested diff, and SHA-256 hashes.

The scheduler suites were not rerun. This correction only reorders dataclass
fields, and the full EngineArgs suite covers LP construction. The earlier
CPU-only limitation is unchanged: real engine initialization through
`EngineArgs` and GPU execution remain unverified.
