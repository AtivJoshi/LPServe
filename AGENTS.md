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
- Do not silently turn an OPEN item into a permanent project decision. For the active MVP, an agent may choose a minimal, visible, provisional value when it is necessary to proceed; record it in the appropriate design/configuration surface.
- Implementation convenience does not authorize changing the mathematical formulation. Do not alter mathematical or normative documentation merely to make code easier to write.
- If authoritative sources appear inconsistent, identify the exact conflicting claims, paths or sections, and revision boundaries. Stop the affected decision and report the discrepancy instead of choosing silently.
- When a claim about existing LPServe behavior materially affects implementation, inspect the current source. Recheck affected paths against the commit-pinned architecture audit; code at the checked-out revision is the ultimate evidence for existing behavior, but code presence is not proof of correctness.
- Do not present a design requirement, planned test, inferred behavior, or unexecuted path as verified implementation evidence.
- If an existing framework design, normative requirement, or prior decision appears to add complexity unnecessary for the supported MVP, flag it explicitly in the active user-facing chat before implementing the affected mechanism. Identify the exact source or decision, current evidence, added complexity, smallest alternative, guarantees or future scope that would be lost, and required authoritative-document changes. Do not silently override the contract, weaken correctness or mathematics, resolve an OPEN item implicitly, or bypass a BLOCKER; continue unaffected work when possible and pause only the affected choice pending an approved recorded resolution.
- Follow the MVP compatibility policy in `docs/lp_scheduler_design.md` §16. Existing LPServe/SLAI limitations may be inherited and documented rather than repaired; do not claim that inherited behavior is correct or fixed.

## MVP-first planning and minimal implementation

- Prioritize the shortest path to a basic runnable LP scheduler. Do not broaden a task into repairs for pre-existing serving-framework behavior, exhaustive edge-case design, or premature optimization unless that issue prevents the selected MVP path from running.
- Keep documentation concise and local to the authoritative document. Record only material decisions, invariants, observed limitations, and reproducibility facts; do not create duplicate summaries, speculative notes, or routine status updates.
- Implement the smallest code path required now. Do not add speculative abstractions, configuration knobs, compatibility shims, fallbacks, retries, rollback, or future-proofing without a demonstrated current need.
- Unsupported states and unexpected failures must fail visibly through a clear error or structured failure result. Do not silently patch state, fabricate a schedule, or continue after a meaningful failure.
- Ask for direction only when a choice materially changes the mathematical objective, public semantics, or project scope. Otherwise make the smallest reversible provisional choice and record it visibly.

## Modification discipline

- Preserve unrelated user changes. Never overwrite, reformat, or clean them up as collateral work.
- Inspect the working tree before substantial modification and again before reporting completion.
- Modify only files required by the requested task. Avoid broad refactors, renames, dependency churn, or opportunistic cleanup unless explicitly requested.
- Treat phase names and numbers as project-management labels only. Do not use them in Python filenames, package or module names, identifiers, comments, docstrings, printed output, diagnostics, result stages, categories, or failure messages; use responsibility-based implementation terminology instead.
- Do not commit, push, reset, discard changes, restore files, or rewrite Git history unless explicitly instructed.
- Do not modify the mathematical formulation or normative design unless the task explicitly authorizes that change and the change is an approved project decision.
- When an explicitly approved design decision or implementation contract changes, update the appropriate documentation and traceability as part of an authorized task. Otherwise, preserve the documents and report the needed follow-up.
- Keep historical audits and experiment records provenance-bearing; do not rewrite historical evidence to describe later code.

## Phase boundaries

Follow the ownership model in `docs/lp_scheduler_design.md` §3 and the project sequence in `docs/LP Scheduler Research Context.md` §19.

- **Phase D — mathematical layer:** framework-light problem construction, solver interface, input/result validation, deterministic integer extraction, and plan validation.
- **Phase E — state mapping:** read-only construction of a coherent, immutable Phase D input from current LPServe state.
- **Phase F — execution:** fresh physical and operational prevalidation followed by ordered LPServe-native mutation and `SchedulerOutputs` construction.

Do not blur these layers. Phase E must not mutate scheduler or serving state. Phase F must not run after an unsuccessful or stale Phase D/E result and must comply with the selected MVP compatibility and failure contracts in the design.

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

Implementation may proceed when every OPEN decision required by the requested layer is either explicitly resolved or supplied as an explicit, visible provisional input for the scoped MVP, test, or configuration.

Do not invent hidden defaults. Keep solver choice or status handling, numerical tolerances, tie-breaking or iteration order, utility policy or scaling, capacity mapping, memory coefficients or reserve, failure behavior, stale-state handling, and other values externally visible. A scoped value is not automatically a permanent project-wide decision.

Resolve only what the active MVP needs. If a missing input matters to that path, choose the smallest reversible provisional value or ask for direction when the choice materially changes objective, public semantics, or scope. Do not halt work for unrelated OPEN items.

## Testing and verification

- Add focused tests for each implemented MVP behavior and directly relevant regression risk.
- Use the MVP-oriented requirements in `docs/lp_scheduler_design.md` §15 and relevant invariants in §§11–14. Do not expand a task into exhaustive testing of inherited or deferred edge cases unless they block the supported path.
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
