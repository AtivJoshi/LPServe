# PH1 soft-SLO pilot handoff (2026-11-03)

Bounded pilot for [milestone 1](ph1_soft_slo_milestone.md): measure TTFT/TBT
at the engine output and compare one timed coefficient policy with its untimed
control through the live single-stage PH1 scheduler (TP=PP=1, synchronous).
Tested base `bcd6c93` plus the working-tree changes below. Evidence:
[`validation_output/soft_slo_pilot/20261103T080911Z/`](../../validation_output/soft_slo_pilot/20261103T080911Z/)
(`derived_tables.txt` summarizes it). The `_superseded_preliminary` directory
holds earlier runs of a prior driver revision; it is not the reported evidence.

## Scoped inputs (provisional; D-01–D-07 stay OPEN)

- Targets per request from override or class: tight TTFT/TBT 100/40 ms,
  relaxed 300/120 ms; priority 1 (any other value fails mapping). SLAI
  deadline/strict fields are never set or read.
- `K=min(B,C)`, `R` = captured prompt remainder (includes rebuild),
  batches `1+ceil(R/K)`, forecast 20 ms/batch. Due = arrival+TTFT before the
  first verified token, else last verified token+TBT (kept through rebuild).
  `slack = due - snapshot_time - batches*0.020`;
  `boost = 1 + clamp((T-slack)/T, 0, 2)`.
- Timed: decode `boost`, prefill token `boost/K`, penalty 1. Untimed: 1, `1/K`,
  1. Ineligible terms are 0.
- Ample case: per class one 17-token request at `t0` and one 33-token request
  at `t0+20 ms`, 6 outputs each, relaxed registered first in each group (so
  the untimed tie-break favors it), 32 blocks of 16, context 64, `B=17`, `C=16`,
  `S=2`, 4 residents, 5 s cutoff, 100-decision cap. Pressure case: 31/32-token
  prompts (relaxed/tight), 3 outputs, 5 blocks, `B=32`, `S=2`, 2 residents.
  Settings are frozen in `scripts/soft_slo_pilot.py` (`settings.sha256`
  `8018a002…` in every non-smoke `summary.json`).

## Source findings

- The sample of a step that finishes a prompt is discarded
  (`BaseSequenceManager._process_seq_output`), so the first visible token
  needs a separate decode, consistent with `1+ceil(R/K)`.
- `seq.state.num_output_tokens` is cumulative across recomputation, while
  `output_token_ids` is cleared and `RequestOutput.token_ids` aliases the
  live list. The clock therefore checks a one-token count increment, the
  sampler token, and a copied `RequestOutput` list before advancing.
- Finished requests leave `seq_map`; the clock keeps its own references.

## Implementation

- `lpserve_state_mapping.py`: `RequestTiming` and `SoftSloForecast` frozen
  records; pure `soft_slo_forecast`, `soft_slo_timed_utility`, and
  `soft_slo_untimed_utility`. No existing function changed.
- `scripts/soft_slo_pilot.py`: target resolution, `OutputClock`, a
  reference-only decision recorder installed through the existing
  `_coefficient_policy` hook, fixed-cohort pre-registration with future
  `time.monotonic()` arrivals, cutoff/cap loop, post-trial checks,
  latency metrics, same-state PH1/exact comparisons, the hand-checked flip, a
  CPU engine (native `BaseLLMEngine.add_request/step/_on_step_completed`, CPU
  worker replay), and a GPU engine with a read-only worker identity probe.
- `tests/test_soft_slo_pilot.py`: 18 tests. The file matches the `test*`
  ignore rule and needs `git add -f`.

## CPU evidence

`cpu_gates/gates.txt`: compile, import, `git diff --check`, trailing
whitespace, the new tests (18), `test_lp_relaxation_scheduler` (38),
`test_lp_scheduler` (24), `test_lpserve_plan_execution` (16),
`test_lpserve_state_mapping` (24), `test_lp_engine_args` (11),
`test_exact_integer_plan` (8), `test_compare_scheduler_gpu` (56). All pass;
each module ran in its own process. (Run together in one process,
`test_main_smoke` fails at unmodified HEAD because `sarathi` is already
imported.) The tests cover partial/final prefill, repeated IDs, rebuild,
final output, association failures, cutoff equality and after-cutoff steps,
censoring, open gaps, fail-stop mapping, and the live CPU ample/pressure/
control sessions. Flip (`B=C=16`, `S=1`, now 60 ms): untimed rewards 1/0.75,
PH1 and exact choose relaxed `prefill:16`; timed 1.333/1.5, both choose tight
`prefill:12`. CPU latencies are synthetic and not reported.

## GPU evidence

Node `gpu051`, job 66571617, NVIDIA A16 `GPU-c1783933-ac1a-85f1-4c50-56eb59884cf1`,
driver 595.91.07; Python 3.10.8 (`env/`), torch 2.3.0+cu121, ray 2.58.0,
scipy 1.15.3, transformers 4.57.6. TinyLlama snapshot `fe8a4ea1…`,
`load_format=auto`, FP16 (bf16 cast), FlashAttention, greedy, `ignore_eos`,
no stop strings. Two worker parameters match the safetensors file (shape,
first 16 values exactly, sum within 1e-3 relative). The
profiled 15,613 blocks are ≥ 32 chosen; the cohort needs ≤ 10. One GPU fits,
so no TP plan is needed. Commands and exit codes are in `runs.txt` (all 0).

The ample pair (two warm-ups, then timed, untimed, untimed, timed; every
trial drained) is trace-valid. Every decision has one timing copy, one capture
including exactly the arrived unfinished cohort, coefficients equal to a pure
recomputation, a revalidated plan, and derived emitted order. Sampler count
and order match. 44 of 72 measured decisions are contended, 0 preemptions, and
18 captured states have a different counterfactual PH1 plan. Cleanup is
central and worker empty. On 19 trial-0 snapshots × 2 policies, PH1 equals the
exact optimum (largest gap 6.2e-11; the 6 action mismatches are ties).

| Measured (ms) | tight TTFT median | tight miss | tight mean excess | relaxed TTFT median | relaxed miss | TBT misses | TBT p99 |
|---|---|---|---|---|---|---|---|
| timed ×2 | 123.0, 123.3 | 0.5 | 40.9 | 201.2, 201.0 | 0 | 0/40 | 19–21 |
| untimed ×2 | 160.5, 161.0 | 1.0 | 60.5, 61.0 | 140.9, 141.3 | 0 | 0/40 | 20–21 |

All gaps are observed; there are no unresolved first tokens, open gaps, or
events after the cutoff. With two requests per class this is a pilot
observation, not a performance claim. Diagnostics: measured batch mean is
19.2–19.3 ms (20 ms forecast). Next-token forecast error averages 28–36 ms
(up to 174 ms) because waiting time is not modeled. Timed boosts never reach
the cap.

Observed limitation: in timed runs `tight33` is skipped at decisions 3–7
with one prompt token left while its boost rises from 1.84 to 2.84, because
a one-token chunk earns `boost/K` (at most about 0.18) against decodes worth
about 1.2–1.5. Its TTFT
(181.7 ms) misses the target. This is the milestone's decode-readiness
question; the formulation is unchanged.

Pressure witness (`gpu_pressure`, all four trials, both policies choose the
same actions): at decision 4 with 0 free blocks `relaxed31` is evicted after
two tokens; decisions 5–7 rebuild 33 tokens with no output while
`last_visible` stays at `t0+80–81 ms`; decision 8 emits the next token after a
91.5–93.4 ms gap (under 120 ms). Inherited overgeneration yields 5 tokens for
3 requested, with 33 repeated prefill tokens, so this case is kept out of the
latency comparison. Continuity: the no-preemption control (`gpu_pressure_control`,
8 blocks, 5 outputs) emits the same five greedy tokens for `relaxed31`.

## Open items

General coefficient and preemption policy (D-01–D-04, κ), recomputation
output semantics (D-22), recovery/overhead criteria (Q11), PP/PH2/Lagrangian
work, and the larger campaign remain OPEN or deferred. `docs/project_status.md`
was not updated.
