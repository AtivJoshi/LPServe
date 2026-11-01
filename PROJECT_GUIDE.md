# LPServe Project Guide

**Current-state refresh:** 2026-11-01, source/evidence review at `35e0eab`; no experiments rerun. Earlier provenance below is historical.

**Guide provenance:** Current-target wording updated on 2026-10-05 after read-only review at committed HEAD `14a15356e64e175f5def042d582412b40af4de2d` and user acceptance of the bounded single-request dummy-weight GPU milestone. The CPU milestone was accepted after review at `4f553b4`. Earlier guide updates inspected `78d663a` (2026-10-04) and `67a3362` (2026-09-20). The Phase C architecture audit remains pinned to `c3e0143`; later implementation and documentation commits do not repin that historical audit. Inspect current Git state before relying on this snapshot.

## Project purpose

This repository is the working implementation and experiment base for a research prototype of an LP-relaxation scheduler for continuous-batching LLM inference. The scheduler is implemented natively in the LPServe/SLAI-derived serving framework and has bounded comparisons with reference policies in that framework. The first implementation uses **Primal Heuristic 1**: the myopic ILP's continuous relaxation followed by deterministic, feasibility-preserving integer extraction.

**Research goals clarified 2026-10-16.** The immediate research priority is a
runnable framework for observing coefficient update policies and comparing them
with reference schedulers. Use the canonical [research goals and working
approach](docs/LP%20Scheduler%20Research%20Context.md#2-primary-research-goal)
in section 2 of the research context when choosing each task's scope. The goals
cover policy classification, alternative decision methods, and explicit
revision of the candidate mathematical formulation; they remain revisable.
Finding the best coefficient policy is not the current goal.

Proposed mathematics, observed framework behavior, scheduler requirements,
implementation, and experimental evidence remain different kinds of claims and
must remain visibly separated. Implement the currently selected formulation
faithfully; research revisions require explicit changes to the appropriate
mathematical and implementation contracts. The MVP may intentionally inherit
existing LPServe/SLAI behavior and known defects; that is not evidence that
those behaviors are correct or repaired.

## Repository knowledge map

| Path | Role | What it does not establish |
|---|---|---|
| `docs/math/main-llm-serving.tex` | Mathematical source of truth for the proposed scheduling formulation, especially the myopic ILP and `Primal Heuristic 1: Approximation via LP Relaxation`. | Existing LPServe behavior or an implemented scheduler. Later heuristic, Lagrangian, MPC, and extension sections are not part of the first target unless scope is explicitly revised. |
| `docs/LP Scheduler Research Context.md` | Research context: assumptions, implementation principles, validation requirements, baseline policy, phase boundaries, open research questions, and the intended workflow. | Code-grounded proof of framework behavior or a replacement for the mathematical formulation and normative design. |
| `docs/lpserve_scheduler_architecture_summary.md` | Compact implementation-facing digest of the Phase C architecture audit. Read it before the full audit when quick orientation to scheduler, engine, sequence, and block-management behavior is needed. | Normative LP-scheduler requirements, new evidence beyond the full audit, or authority over the full audit when they conflict. |
| `docs/lpserve_scheduler_architecture.md` | Descriptive, code-grounded Phase C audit of the existing LPServe/SLAI-derived framework at commit `c3e014363dd50e1830d7c85c3d043eab69fdc9e5`. It records request state, queues, allocation, execution/replay, metrics, defects, gaps, and exact evidence labels. | A proposed LP-scheduler specification, proof about later revisions, or runtime validation of every inspected path. |
| `docs/lp_scheduler_design.md` | Normative implementation specification for the proposed scheduler across Phases D, E, and F. It defines responsibilities, interfaces, invariants, validation gates, explicit **OPEN** decisions, and **BLOCKER** boundaries. | Permission to fill OPEN items with convenient defaults or to bypass blockers. It does not turn a design requirement into verified implementation. |
| `docs/project_status.md` | Chronological project state and acceptance record. It records scoped acceptance, later implementation/experiment milestones, and Phase A--C history. | Mathematical, architectural, or design authority. It may lag newer artifacts and must not override them. |
| `docs/handoffs/` | Provenance-bearing implementation and review handoffs retained as supporting evidence. | Normative requirements, mathematical authority, or current status by itself. |
| `docs/experiment_reference.md` | Durable evidence and configuration record for the controlled Phase B baseline runs. | Performance conclusions, scheduler equivalence, or validation of the proposed LP scheduler. |
| `docs/unity_setup.md` | Commit-pinned, validated Unity environment and operational setup for the Phase A smoke baseline. | A benchmark result or a guarantee that the same instructions work unchanged at later commits or in other environments. |
| `sarathi/` | Main Python implementation: scheduler and sequence state under `sarathi/core/`, engine/worker execution, model execution, metrics, and benchmark machinery. | Proposed behavior merely because a design document describes it. |
| `csrc/` | Native CUDA/C++ operators used by the serving implementation. | Scheduler policy specification. |
| `config_path_yml_files/`, `scripts/`, `examples/`, and `data/` | Configuration, experiment/plotting utilities, examples, and traces or derived data. | General conclusions without a recorded, controlled run. |
| `README.md` | Inherited SLAI overview and basic repository orientation. | The authority map or current LP-scheduler project state. |

Generated experiment outputs and console logs are evidence only for the exact revision, environment, configuration, workload, and run that produced them.

## Source authority and conflict rules

Authority is **claim-specific**, not one universal ranking:

- For what the proposed optimization problem means, use `docs/math/main-llm-serving.tex`.
- For research assumptions, validation discipline, phase boundaries, and implementation philosophy, use `docs/LP Scheduler Research Context.md`.
- For what the first LP scheduler must do, use `docs/lp_scheduler_design.md`, including its MVP compatibility policy, and check it against the mathematical source.
- For quick architecture orientation, use `docs/lpserve_scheduler_architecture_summary.md`; it is a digest only. The full audit governs descriptive framework behavior when they conflict, and the design file governs normative LP-scheduler requirements when the summary and design conflict.
- For what the audited existing framework did at `c3e0143`, use `docs/lpserve_scheduler_architecture.md` and its evidence classifications.
- For a claim about existing framework behavior that must be verified, inspect the LPServe source code. Code is the ultimate evidence for the checked-out revision; the architecture document remains the durable description of its pinned audit revision. Code presence does not imply correctness.
- For what was actually run or validated, use the provenance-bearing status, experiment, and environment records plus retained outputs. Planned tests and design requirements are not observed evidence.

Before relying on an architecture claim, compare the current checkout with the audit's pinned revision for the affected code paths. If relevant code has changed, re-audit it and record a new revision or a clearly dated addendum; do not silently rewrite the historical audit.

If authoritative sources appear to conflict, identify the exact claims, paths/sections, and revision boundaries; stop the affected decision; and investigate. Do not silently choose the easiest interpretation, allow a status document to override a specification, treat proposed mathematics as existing framework behavior, or treat existing code as the intended LP policy. Any approved resolution must be recorded in the appropriate mathematical, design, architecture, status, or experiment document.

## Current target and phase model

The planned sequence is:

1. Phase A — establish the LPServe/Unity baseline.
2. Phase B — reproduce existing-policy baselines.
3. Phase C — audit the scheduler architecture without implementing the LP scheduler.
4. Phase D — implement and synthetically validate the pure, framework-light LP solve and extraction layer.
5. Phase E — map a read-only, coherent LPServe state snapshot into Phase D inputs.
6. Phase F — prevalidate and execute integer plans through LPServe-native actions.
7. Phase G — perform integrated correctness validation.
8. Phase H — attribute timing and run controlled performance comparisons.

The corrected myopic ILP, its continuous relaxation, and PH1 whole-action
extraction are implemented. The read-only mapper, native executor, live
single-stage scheduler, admission checks, and configuration migration are
connected. Source was reviewed on 2026-11-01 at `35e0eab` on `main`.
The corrected CPU evidence is in the
[integration handoff](docs/handoffs/lp_scheduler_integration_handoff.md#corrected-connected-cpu-framework-2026-10-13-awaiting-review).
It records 138 focused tests and bounded enumeration at `9a6ca3c` plus
retained changes; later fixes have separate evidence. Contention and
preemption GPU regressions at `82609eb` passed their bounded dummy-weight
cases. These records do not validate every path at current HEAD.

The current path uses actual zero-or-one-block decode demand, the admission
watermark, a useful-execution row, resident capacity, separate service/victim
ranks, and declared execution order. It has no independent planning reserve
or locking extraction. Native recomputation preemption is supported. Ignore
controls, overlapping state-changing calls, and pipeline execution remain
unsupported. `EngineArgs` supports LP construction; bounded comparison GPU
runs exercise real engine initialization. This does not establish general
benchmark/CLI coverage.

The adapted reference is implemented in LPServe and pinned to upstream vLLM
v0.31.0 commit `db9527a`. It is distinct from the inherited `VLLMScheduler`
and is not whole-engine vLLM equivalence. Its
[handoff](docs/handoffs/adapted_vllm_reference_handoff.md#gpu-validation-run-2026-10-18)
records CPU checks and passing mixed/preemption GPU cases at `9b9273a`.
The [comparison handoff](docs/handoffs/scheduler_comparison_handoff.md)
records uniform, rank-based, geometric, reciprocal-square, and running-order
studies, including failed runs and later corrections. The corrected
12-block/context-192 reciprocal-square GPU case drained in 118 decisions;
the reference drained in 117. Decision counts are not latency or throughput.

The user accepted the experimental running-order policy and bounded CPU batch
at `35e0eab`; see the
[acceptance record](docs/handoffs/scheduler_comparison_handoff.md#acceptance-and-subsequent-analysis-2026-10-26).
All 60 rollouts drained and all 2,793 decisions had integer comparisons.
There is no running-order GPU validation. Its waiting priorities remain soft;
the strict waiting-order extension is proposed and unimplemented.
Older review recommendations and “awaiting review” handoffs remain historical
records, not new user acceptance. Full Phase F acceptance is not inferred.

The immediate bounded assignments are in
[the undergraduate task document](docs/handoffs/tasks_for_stella.md).
General recovery/overhead criteria (Q11), coefficient defaults (D-01–D-04),
discarded-work penalties (κ), and repeated-lost-work concerns (F02) remain
open. PH1 can choose poor whole victims or waiting work and can admit then
evict. Running-order coefficients can become numerically too large.
Admission validity does not guarantee completion. Native recomputation can
reset output limits, sampler/output defects remain inherited, and physical
block IDs can differ between managers. General fairness, restart freedom,
generation quality, performance, and reference equivalence are unverified.
The manuscript warning about downstream PH2, dual, MPC, and equivalence
sections still applies. Offline dual reporting does not approve those sections
or implement a new scheduler.

**Future TP/PP extension approved (2026-10-18; not implemented).** Read
[design §19](docs/lp_scheduler_design.md#19-approved-future-tp-and-pp-extension)
and research-context §22.2 before extending parallel execution. They record
fixed TP, overlapping PP, and combined support for LP and the adapted
reference; protected capture, planning without a lock, protected commit;
native in-flight preemption compatibility; completed-victim handling;
dispatch/wait/failure rules; required mathematical alignment; and bounded
CPU/GPU validation. This approval changes no running code, proves no new
parallelism support, and does not repin the historical architecture audit.

The key separation is:

```text
LPServe state -> Phase E snapshot/mapping -> Phase D solve/extraction
              -> validated integer plan -> Phase F native execution
              -> SchedulerOutputs
```

Phase D must remain independent of mutable LPServe objects and GPU execution. Phase E is read-only. Phase F must not run after an unsuccessful prior stage and follows the design's selected MVP compatibility policy.

## MVP triage and minimalism

For the active MVP, distinguish three kinds of work:

- Fix defects introduced by LP-scheduler work and issues that prevent the selected MVP path from running.
- Inherit and document pre-existing LPServe/SLAI behavior, including known defects, rather than repairing it as collateral work.
- Defer broad hardening, exhaustive edge-case coverage, generalization, and performance optimization until the MVP creates a concrete need.

Resolve or supply only the values needed by the active phase. A necessary value may be a visible, provisional configuration or scoped experiment input; do not present it as a permanent research conclusion unless it is explicitly approved as one.

Document only material decisions, invariants, observed issues, and reproducibility facts. Do not create duplicate summaries, speculative notes, exhaustive issue catalogs, or routine status updates. Update the single authoritative document for a meaningful decision.

Implement the smallest code path that satisfies the active goal. Do not add speculative abstractions, configuration knobs, fallback policies, retries, recovery, or edge-case handling without a demonstrated current need. Unsupported states should fail visibly rather than being silently repaired.

If an existing framework design, normative requirement, or prior project
decision appears to impose complexity that is unnecessary for the currently
supported MVP, the planning or implementation agent MUST flag it explicitly in
the active user-facing chat before implementing the affected mechanism. The
report must identify the exact source and decision, explain the added
complexity using current evidence, propose the smallest alternative, state
which guarantees or future scope the alternative would give up, and identify
any authoritative documentation that would need to change. The agent MUST NOT
silently ignore, reinterpret, or override the existing contract. This
complexity challenge does not authorize weakening mathematical or correctness
requirements, filling an OPEN decision implicitly, or bypassing a BLOCKER.
Continue unaffected work when possible, but pause the affected choice until an
approved resolution is recorded.

## Work / research-design sessions

ChatGPT normally supports research discussion, mathematics, task preparation,
and review. Claude normally implements and executes tasks. Claude Desktop
may separately review mathematics. The user normally handles synchronization.
Use one writer per file and reread saved changes before editing. These roles
do not restrict an explicitly assigned documentation-maintenance task.

Every implementation task must critically examine its proposed plan before
implementing it. Report contradictions, counterexamples, and unintended
consequences, including problems in an approved plan. Approval is not proof
of correctness. Continue unaffected work, but stop an affected choice that
would change mathematics, resolve an OPEN decision, or bypass a BLOCKER.

An implementation agent already on an allocated Unity GPU node must use that
node and the existing repository Python environment directly. Do not invoke
`srun`, request another allocation, submit jobs, or SSH elsewhere unless
explicitly requested. Report the exact environment failure instead of
changing nodes or rebuilding the environment.

Begin with this guide, then read only the detailed sources needed for the question:

1. Read the relevant formulation and Primal Heuristic 1 material in `docs/math/main-llm-serving.tex`.
2. Use `docs/LP Scheduler Research Context.md` for the research boundary, validation expectations, and phase discipline.
3. Use `docs/lp_scheduler_design.md` for normative interfaces, invariants, decisions, and blockers.
4. Consult `docs/lpserve_scheduler_architecture_summary.md` first for orientation, then `docs/lpserve_scheduler_architecture.md` for detailed evidence or whenever a framework claim affects implementation; inspect current code whenever a framework claim affects the conclusion.
5. Use status, experiment, and Unity records only for historical or observed claims within their stated provenance.

Keep verified facts, inferences, proposed requirements, unresolved choices, and observed results labeled separately. Research sessions may analyze OPEN decisions, but must not present a selection as approved until it has been explicitly decided and recorded.

## Codex / implementation sessions

Before editing, record the branch, commit, and working-tree state, preserve unrelated user changes, and compare scheduler-relevant code with the audit baseline. For Phase D, read `docs/lp_scheduler_design.md` first for normative requirements, then `docs/lpserve_scheduler_architecture_summary.md` for architecture orientation, then the full audit and current source code only as needed for detailed framework claims; use Sections 7, 9–11, and 14–18 of the design and the corresponding ILP/relaxation/extraction sections of `docs/math/main-llm-serving.tex`.

Implementation tasks should state the objective, files allowed to change, supported behavior and invariants, required checks, out-of-scope work, and an explicit completion boundary. Distinguish executor implementation, live scheduler integration, and GPU validation; name which are included in the task and do not treat one as evidence that the others are complete. Follow the communication, SLAI-reuse, execution-environment, and validation-provenance rules in `AGENTS.md`. Do not infer LPServe behavior from modern vLLM or another Sarathi/SLAI version. Supply only the policy values needed by the active MVP, visibly and provisionally where appropriate; do not bury them as undocumented defaults.

Phase names and numbers are project-management labels, not implementation terminology. They MUST NOT appear in Python filenames, package or module names, identifiers, comments, docstrings, printed output, diagnostics, result stages, categories, or failure messages. Use responsibility-based names such as `lp_relaxation`, `state_mapping`, `precommit_validation`, and `execution` instead.

Validate in phase order with focused evidence for the supported MVP path: syntax/import/static checks; focused CPU tests; synthetic mathematical tests; scheduler-state/integration tests; a tiny GPU smoke test; inspection of scheduling decisions and state transitions; then larger experiments. Do not expand the task into exhaustive inherited edge-case coverage unless it blocks that path. Never report a command, test, or experiment as successful without observing its output, and never substitute aggregate throughput or latency for scheduler-correctness evidence.

For authorized commits, use the single cohesive commit and tested-code
provenance workflow in `AGENTS.md`. Separate implementation and handoff commits
are no longer mandatory.

## Documentation maintenance

- Keep repository-relative paths and commit/environment provenance exact.
- Update `docs/project_status.md` for material implementation, observed evidence, review, and explicit acceptance. Label each separately; recommendations do not establish acceptance. It records chronology, not specifications.
- Record controlled run configuration and evidence in `docs/experiment_reference.md` or a clearly scoped successor; do not overwrite historical provenance.
- Keep `docs/lpserve_scheduler_architecture.md` commit-pinned. Record later behavior in a revisioned audit or dated addendum after reinspection.
- Update `docs/lp_scheduler_design.md` when a design decision is explicitly approved or a contract changes, including rationale, affected interfaces, tests, and experiment metadata where applicable.
- Change the mathematical source only for an intentional mathematical amendment, and trace consequential design changes.
- Document only implemented and observed facts. Preserve OPEN decisions and BLOCKERs until their resolution and validation are explicit.
