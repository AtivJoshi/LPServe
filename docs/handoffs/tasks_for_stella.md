# Fixed comparisons and offline reports

## Scope and current state

Work in LPServe. Use AI to implement this fixed specification, inspect results,
and prepare reports. Complete Tasks 1–3 in order. Do not choose new mathematics,
coefficient policies, or research directions. First read [PROJECT_GUIDE.md](../../PROJECT_GUIDE.md)
and [AGENTS.md](../../AGENTS.md). This task was prepared against `main` at
`35e0eab`; record the revision you actually use.

The current scheduler solves the continuous relaxation of a one-step integer
linear program (ILP). PH1 turns that fractional proposal into whole actions.
A read-only mapper supplies the problem. An executor checks current state and
performs native admission, prefill, decode, and recomputation preemption.
Prefill processes prompt tokens. Decode produces an output-token event.
Preemption frees a whole cache and later repeats prefill. The supported runtime
is synchronous and single-stage.

The [manuscript](../math/main-llm-serving.tex) defines mathematics; the
[design](../lp_scheduler_design.md) defines implementation requirements.
Code shows what exists. Handoffs and saved outputs show what was tested.
[Project status](../project_status.md) records acceptance and chronology.
Downstream manuscript sections still need mathematical review. Do not use them
to replace the current formulation.

## Evidence already available

The corrected CPU framework and bounded GPU paths have retained evidence.
The adapted reference follows pinned vLLM v0.31.0 service preferences inside
LPServe; it is not whole-engine vLLM equivalence.
The [comparison handoff](scheduler_comparison_handoff.md) records uniform,
rank-based, geometric, reciprocal-square, and running-order studies.
Earlier uniform/rank-based runs repeatedly prefilled prompts without output
within their bounds. Later policies drained the recorded cases, but PH1 can
still lose utility through whole-victim and waiting-work choices.

At the corrected constrained 48-request case, reciprocal-square PH1 and the
reference completed in 118 and 117 decisions, with 226 and 102 repeated-prefill
tokens. Earlier GPU attempts failed before scheduling because context 256
needed 16 blocks but the pool had 12. The approved context-192 correction
passed CPU and GPU checks. The ample CPU case took 40 reciprocal-square PH1
decisions and 42 running-order PH1 decisions. These counts measure scheduling
steps, not latency or throughput.

The running-order policy and bounded CPU batch are explicitly accepted:
60 rollouts drained and 2,793 decisions had integer comparisons. It has no
GPU validation. It still has soft waiting priorities and numerical scale
limits. Strict waiting priorities and TP/PP support are unimplemented future
work. See the [acceptance record](scheduler_comparison_handoff.md#acceptance-and-subsequent-analysis-2026-10-26).

## Task 1: Compare five fixed settings with PH1

Start from `reciprocal_square_rank_utility`. For position `r` in running order
followed by arrived waiting order, it uses `w=1/(1+r)^2`. Prefill reward per
token is `w/q`, where `q=min(remaining,c_max,b_max)`; eligible decode reward
and preemption penalty are `w`.

Compare exactly these settings:

1. Existing reciprocal-square policy: `lp_reciprocal_square`.
2. That policy with decode rewards multiplied by four.
3. That policy with preemption penalties multiplied by four.
4. That policy with both terms multiplied by four.
5. Existing `running_order_utility`: `lp_running_order`, unchanged.

Keep prefill rewards unchanged in settings 1–4. Preserve eligibility and zero
coefficients. Do not tune the multiplier after seeing results or compare raw
objective values between policies.

Use CPU execution on `queue48_burst` and `queue48_burst_pool12` only.
Both have 48 burst arrivals, resident limit 8, block size 16,
`b_max=c_max=128`, `s_max=8`, and a 384-decision bound. Pools are 128 and
12 blocks; maximum contexts are 256 and 192 respectively. Keep all token IDs,
prompt lengths, output limits, arrivals, capacities, numerical settings, and
solver settings from the existing case definitions.

For every PH1 decision, retain an ILP comparison for the exact same saved state
and coefficients. A separate ILP rollout soon reaches different states and
cannot replace this comparison. Reuse matching baseline and running-order
records after checking inputs, coefficients, revisions, and completeness.
The [running-order CPU evidence](../../validation_output/scheduler_comparison/20261026T045354Z_running_order/cpu_comparison/)
contains both controls on both cases. Explain any rerun.

Report completion or bounded unfinished outcomes, decisions, preemptions,
repeated-prefill tokens, and service order. Report PH1 differences from integer
optima and explain three representative decisions. Retain complete experiment
and solver records. Stop after the fixed comparison; propose no winning policy.

## Task 2: Reproduce known PH1 failure cases

Use retained traces, starting with these documented examples:

- Fractional preemption: reciprocal-square constrained decisions 20 and 34,
  or running-order constrained decision 2, choose a poor whole victim.
- Fractional admission: `fractional_admission`, or reciprocal-square
  `queue48_burst` decision 0, chooses poorer waiting work.
- Admit then evict: running-order `queue96_burst_pool12` ILP decisions
  16→17 or 55→56. Preserve both decisions.
- Repeated prefill without output: earlier `two_request`, resident limit 1,
  uniform/rank-based runs stop unfinished after eight decisions.

Locate exact artifacts through the comparison handoff. For each example show
starting state, coefficients, LP proposal, PH1 plan, integer optimum where
available, executed actions, and resulting state. Create offline replay
fixtures where practical. Keep sequences when one snapshot cannot show the
behavior. Label unavailable integer results explicitly; any new comparison
must use the original state and coefficients. Do not invent counterexamples,
redesign extraction, or turn bounded noncompletion into a starvation proof.
Stop once these four examples are reproducible and explained.

## Task 3: Report LP dual prices offline

Load saved problems and solve the unchanged current continuous LP. A dual
multiplier measures sensitivity to a constraint bound. Report every constraint's
name, slack, and multiplier, including variable bounds, `useful_execution`,
`actual_memory`, `decode_prerequisite`, and conditional admission-watermark
rows. State when a conditional row is absent.

Reuse the current matrix, bounds, solver method, and options. Document the
solver's minimization sign convention and any conversion to maximization.
Include bound contributions when checking primal/dual objective agreement.
Retain status and residuals; do not invent prices for unsuccessful solves.
Check small examples and representative saved decisions, including positive
watermark states. Report mismatches. Stop at the offline report; do not change
live scheduling or implement a Lagrangian scheduler or subgradient method.

## Relevant Python files

- [`lp_relaxation_scheduler.py`](../../lp_relaxation_scheduler.py) builds, solves, validates, and extracts the current LP; `_build_lp` supplies named rows.
- [`lpserve_state_mapping.py`](../../lpserve_state_mapping.py) captures state and supplies the existing coefficient policies.
- [`lpserve_plan_execution.py`](../../lpserve_plan_execution.py) validates and executes whole plans through native actions.
- [`lp_scheduler.py`](../../sarathi/core/scheduler/lp_scheduler.py) connects mapping, planning, and execution in the live scheduler.
- [`adapted_vllm_scheduler.py`](../../sarathi/core/scheduler/adapted_vllm_scheduler.py) implements the pinned adapted reference.
- [`compare_scheduler_gpu.py`](../../scripts/compare_scheduler_gpu.py) defines cases and shared checks, with a reusable CPU runtime path.
- [`compare_coefficient_policies_cpu.py`](../../scripts/compare_coefficient_policies_cpu.py) runs CPU batches and compares saved decisions.
- [`exact_integer_plan.py`](../../scripts/exact_integer_plan.py) loads saved problems and computes validated integer comparisons.
- [`test_compare_scheduler_gpu.py`](../../tests/test_compare_scheduler_gpu.py) supplies native CPU replay with synthetic samples.

## Common rules and delivery

The multiplier variants and dual report need small offline additions; existing
commands do not expose them. Preserve formulation, PH1, execution rules, and
numerical settings. Critically check the plan and report contradictions before
affected work. Stop for missing evidence, changed contracts, or research choices;
report unexpected outcomes instead of changing inputs.

Deliver a short report per task, replay/report scripts, fixtures, and complete
records of revisions, file hashes or diffs, environments, commands, solver
options/statuses, failures, and unfinished outcomes. CPU replay is not GPU
validation. Claude already on an allocated Unity GPU node must use that node
and its existing repository environment directly; do not use `srun`, request
allocations, submit jobs, SSH elsewhere, or rebuild a failing environment.
Report the exact environment problem. No commits or synchronization are included.
