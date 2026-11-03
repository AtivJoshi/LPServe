# PH1 soft SLO milestone

Agreed on November 3, 2026. This handoff records the agreed plan for studying soft service targets with PH1. The [mathematical manuscript](../math/main-llm-serving.tex) and [scheduler design](../lp_scheduler_design.md) govern the current formulation and implementation.

## Goal and starting point

Build a scheduling framework that explains how different policies trade off service goals and that supports practical approximation algorithms. Use a small number of quantities we can interpret, such as time until an output is due, work remaining, and request priority. Compare the preferences a policy expresses, its actions on the same state, its behavior over a workload, and its end-to-end performance as separate questions. The adapted vLLM study was an exercise in what coefficients can express. The next milestone studies service outcomes with targets supplied independently of any reference scheduler's decisions.

**Keep the current integer linear program (ILP), its linear objective, and its feasibility rules for the first SLO study.** Use Primal Heuristic 1 (PH1), which solves the continuous relaxation and selects whole actions, as the working scheduler. Use exact ILP solves offline on selected states. The first changes are to policy inputs, coefficient updates, and measurement.

## Milestone 1 — Measure soft SLOs and add one timed policy

- Give each request a time-to-first-token (TTFT) target and a time-between-tokens (TBT) target, with class defaults. These service-level objectives (SLOs) are soft targets. Keep the targets separate from priority weights. Start with equal weights and study unequal priorities separately. Leave hard deadline commitments for later.
- Use the current snapshot and add the timing information needed by the policy. Before the first output, the due time is arrival plus the TTFT target. After the first output, it is the time of the last new output plus the TBT target. Use a rough batch-duration estimate and, during prefill, an estimate of remaining chunks plus a decode step. Preserve the streaming clock if cache rebuild is needed after preemption.
- Turn urgency into a bounded boost to the prefill or decode reward while giving ordinary progress positive value. Compute the rewards once per scheduling decision. Hold the explicit preemption penalty fixed across timed and untimed PH1 controls. The coefficient formula, scales, duration estimate, and penalty value will be specified before the pilot.
- Measure from arrival to the first new token visible at the engine output, then between successive new output tokens. Verify those events directly. Prefill completion and a reset of the sequence's output list during recomputation are poor substitutes for the output clock.
- Report miss rate and mean excess in milliseconds, including zeros for on-time events, separately by class for TTFT and TBT. A miss is latency above the target, and excess is the amount above it. Also report median TTFT, 99th-percentile TBT, and the share of streams with a late observed gap. Pooled gap metrics give longer streams more influence, so retain the request-level view.
- Use a fixed arrival cohort and a stated drain limit. Count missing first tokens whose targets have expired as known misses. Keep overdue open gaps, unfinished work, and failures visible. Report repeated work and preemptions in the pressure case.

Start with small cases where we can predict the response to urgency. First use enough memory to isolate timing behavior. Then use a separate pressure case to check that the output clock survives preemption and to observe the work spent rebuilding cache.

## Milestone 2 — Compare policies and explain the differences

Study two workload settings. In the **SLAI-style mixed-target setting**, test whether PH1 can use the extra time available to relaxed streams to improve TTFT while keeping tight-target streams responsive. In the **Sarathi-style common-target setting**, study how decode protection and the token budget affect the tradeoff between TTFT and tail TBT. Compare timed PH1, untimed PH1, and the relevant native LPServe baseline. A decode-heavy PH1 variant can help interpret the common-target results. Choose settings on a pilot and freeze the timed policy before the main comparisons across loads and classes. Label the baseline variants and run configurations. [SLAI, §§6–7](https://users.ece.utexas.edu/~gustavo/papers/BHD25j.pdf) · [Sarathi-Serve, §§4–5](https://arxiv.org/html/2403.02310v3)

Use three views of the comparison.

1. **Same-state decisions.** Use hand-built cases and snapshots from both policies. Compare prompt work, decodes, omissions, admissions, and victims. Check whether each native baseline plan is legal in the ILP before scoring it. Record any conflicting constraint. Score legal plans under the same coefficients and compare them with the exact ILP optimum. Use action matches and mismatches to explain the choices.
2. **Timed replay.** If a common batch-duration model checks out against measured batches, replay the same timed arrivals, targets, and request work with scheduler CPU cost excluded. Each policy constructs its own batches and advances its own clock and state. A replay without a credible time model supports action comparisons only.
3. **Wall-clock runs.** Match model, hardware, workload, and measurement boundary in the same LPServe engine. Measure SLO outcomes, useful throughput, scheduler time, and model execution time. Break down PH1 time where practical. Subtracting solver time from a run cannot recover the replay result because elapsed time changes which requests are available at later decisions.

The milestone should explain where the policy works and where it falls short. Matching benchmark actions or performance is an outcome to measure.

## Experiments that may change the model

Include both of these experiments in the research program. Use small cases to show the effect, then check whether it recurs in representative states or runs. Compare candidate changes against the current exact ILP before mixing in PH1 extraction effects.

| Question | What to compare |
| --- | --- |
| Does the constant reward per prefill token miss the value of becoming ready to decode? | Construct cases where partial prefill earns progress but leaves the request unable to decode, while finishing the prompt makes the next decode possible. If this distinction repeatedly matters with the exact ILP and reasonable coefficients, test a small reward for becoming decode-ready. Measure first-token delivery separately. |
| Does the token budget hide a material difference in batch duration? | Compare batches with equal token counts and different prefill, decode, or context mixes. Use measured durations to see whether the difference changes which plans serve the targets. If it does, test a batch-duration term or limit. |

Use the findings to choose what to change. If PH1 gives up material objective value relative to the exact ILP under the same coefficients, examine PH1 extraction. If the exact optimum follows a poor urgency rule, revisit the coefficients. If the exact model repeatedly misses a distinction such as decode readiness or batch duration, try the smallest change that represents it. If replay and wall-clock results differ, use the timing breakdown and validate the duration model before attributing the gap to scheduler overhead.

## Later decisions

- Revisit PH2 and the Lagrangian formulation after we know which SLO behavior the next method should preserve and which bottleneck it should address. The current ILP and PH1 remain the comparison points. Questions about which constraints belong in a smaller optimization problem and which can be enforced when choosing whole actions remain open. Keep a fixed memory reserve, relocating the admission watermark check, and removing the watermark as separate proposals. The admission watermark and the free-block-before-decode rule are separate checks.
- Set the exact coefficient formula, numeric targets, timing model, workloads, baseline configurations, and acceptance criteria when planning the work. General coefficient defaults and the repository's broader recovery and overhead criteria remain open. [Research context, §§2 and 21](../LP%20Scheduler%20Research%20Context.md)
