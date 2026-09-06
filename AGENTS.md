# Agent Instructions for LPServe

These instructions apply repository-wide. They complement `PROJECT_GUIDE.md`; they do not replace the mathematical, architectural, design, status, environment, or experiment documents it indexes.

## Start with the project guide

Before substantial LP-scheduler research, design, implementation, or review work:

1. Read `PROJECT_GUIDE.md`.
2. Inspect the branch, commit, and working-tree state.
3. When the task involves scheduler, engine, sequence, or block-manager behavior, read `docs/lpserve_scheduler_architecture_summary.md` first for architecture orientation before opening the full audit.
4. Read only the detailed sources relevant to the task, using the guide's claim-specific authority and provenance rules.

Do not reconstruct project context from `README.md`, modern vLLM, another Sarathi/SLAI version, or filenames alone.

## Evidence and decision discipline

- Keep these categories distinct in reasoning, code review, tests, and reports: mathematical requirements; verified existing-framework behavior; normative design requirements; OPEN decisions; BLOCKERs; and observed experimental evidence.
- Never silently resolve an OPEN item in `docs/lp_scheduler_design.md`. An OPEN item is not a default or permission to choose the easiest implementation.
- Implementation convenience does not authorize changing the mathematical formulation. Do not alter mathematical or normative documentation merely to make code easier to write.
- If authoritative sources appear inconsistent, identify the exact conflicting claims, paths or sections, and revision boundaries. Stop the affected decision and report the discrepancy instead of choosing silently.
- When a claim about existing LPServe behavior materially affects implementation, inspect the current source. Recheck affected paths against the commit-pinned architecture audit; code at the checked-out revision is the ultimate evidence for existing behavior, but code presence is not proof of correctness.
- Do not present a design requirement, planned test, inferred behavior, or unexecuted path as verified implementation evidence.
- Respect every applicable BLOCKER in `docs/lp_scheduler_design.md` §16. Do not enable or claim correctness for blocked behavior until the required design change and validation are explicit.

## Modification discipline

- Preserve unrelated user changes. Never overwrite, reformat, or clean them up as collateral work.
- Inspect the working tree before substantial modification and again before reporting completion.
- Modify only files required by the requested task. Avoid broad refactors, renames, dependency churn, or opportunistic cleanup unless explicitly requested.
- Do not commit, push, reset, discard changes, restore files, or rewrite Git history unless explicitly instructed.
- Do not modify the mathematical formulation or normative design unless the task explicitly authorizes that change and the change is an approved project decision.
- When an explicitly approved design decision or implementation contract changes, update the appropriate documentation and traceability as part of an authorized task. Otherwise, preserve the documents and report the needed follow-up.
- Keep historical audits and experiment records provenance-bearing; do not rewrite historical evidence to describe later code.

## Phase boundaries

Follow the ownership model in `docs/lp_scheduler_design.md` §3 and the project sequence in `docs/LP Scheduler Research Context.md` §19.

- **Phase D — mathematical layer:** framework-light problem construction, solver interface, input/result validation, deterministic integer extraction, and plan validation.
- **Phase E — state mapping:** read-only construction of a coherent, immutable Phase D input from current LPServe state.
- **Phase F — execution:** fresh physical and operational prevalidation followed by ordered LPServe-native mutation and `SchedulerOutputs` construction.

Do not blur these layers. Phase E must not mutate scheduler or serving state. Phase F must not run after an unsuccessful or stale Phase D/E result and must comply with the failure and blocker contracts in the design.

The architecture summary is an orientation aid only: `docs/lp_scheduler_design.md` remains authoritative for normative requirements, and `docs/lpserve_scheduler_architecture.md` remains the full descriptive audit evidence.

### Phase D rules

For Phase D work:

- Keep the layer independently CPU-testable without starting an LLM server or requiring a GPU.
- Do not import, store, or accept mutable LPServe `Sequence`, scheduler, block-manager, engine, queue, or callback objects.
- Do not read live global scheduler state, clocks, mutable queues, block managers, or GPU state unless a later approved design explicitly makes such data an input.
- Do not allocate or free blocks, append slots, preempt requests, transition sequence status, mutate queues, or construct live execution side effects.
- Accept request data, capacities, utility values, policy identifiers, numerical policy, ordering keys, and other required state explicitly through immutable, framework-independent inputs. See `docs/lp_scheduler_design.md` §9.
- Return explicit typed/structured success or failure results; do not fabricate an all-zero schedule or silently round malformed solver output.
- Do not prematurely implement Phase E mapping or Phase F execution in a Phase D task.

## OPEN decisions and supplied inputs

Implementation may proceed only when every OPEN decision required by the requested layer has been explicitly resolved by the project or supplied as an explicit input for the scoped test or configuration.

Never invent hidden defaults for solver choice or status handling, numerical tolerances, tie-breaking or iteration order, utility policy or scaling, capacity mapping, memory coefficients or reserve, failure/fallback behavior, stale-state handling, or any other OPEN item in `docs/lp_scheduler_design.md` §17.

A value supplied for a synthetic test, smoke configuration, or experiment is not automatically a project-wide decision. Label it as scoped input, keep it externally visible, and do not encode it as an undocumented production default. If required input is absent, report the blocking OPEN decision and stop the affected implementation path while continuing any independent work that remains valid.

## Testing and verification

- Add focused tests for each new behavior and regression risk.
- Use the authoritative requirements in `docs/lp_scheduler_design.md` §15 and relevant invariants in §§11–14; do not substitute a smaller ad hoc test plan.
- Run proportionate syntax, import, static, unit, and synthetic checks after changes. Advance to state-mapping, integration, GPU smoke, and performance tests only when the applicable earlier correctness gates pass.
- Report the exact commands run and their observed results. Distinguish passed, failed, skipped, and unexecuted checks, including why anything was not run.
- Never claim that a command, test, installation, GPU run, or benchmark succeeded unless its output was observed.
- Inspect actual scheduling decisions, feasibility checks, and state transitions. Throughput or latency alone is never evidence of scheduler correctness.
- Preserve enough commit, configuration, workload, environment, and output information to reproduce environment-dependent results.

## Local and Unity environments

- Use local development for documentation, source inspection, implementation, and CPU/unit tests when appropriate.
- Use the validated Unity setup in `docs/unity_setup.md` for GPU and serving validation unless an explicit task establishes and records another environment.
- On Unity, use the documented allocation workflow and never initialize models, run CUDA validation, or benchmark on a login node.
- Begin GPU work with the smallest useful smoke test and retain its outputs before larger experiments.
- Do not hard-code workstation, Unity path, GPU, CUDA, allocation, or other machine-specific assumptions into the Phase D mathematical layer.

## Completion report

At the end of an implementation task, report concisely:

- files changed;
- behavior implemented;
- exact tests/checks run and observed results, separated into passed, failed, skipped, and unexecuted;
- unresolved issues, blockers, assumptions, or skipped validation;
- whether any OPEN decision or documented contract was affected, and where an approved change was recorded.

Do not claim completion beyond the evidence observed in the current task.
