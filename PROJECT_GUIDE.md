# LPServe Project Guide

**Guide provenance:** Prepared on 2026-09-05 against committed HEAD `6fbc046eca7c0cb7988f08690757d140a51a03e3`. The committed changes after the Phase C audit baseline `c3e0143` through `6fbc046` were documentation-only and did not repin the audit. The current Phase C-closure / Phase D-preparation documentation also includes uncommitted local changes, so the working tree is not identical to `6fbc046`; inspect current Git state before relying on this snapshot.

## Project purpose

This repository is the working implementation and experiment base for a research prototype of an LP-relaxation scheduler for continuous-batching LLM inference. The scheduler is to be implemented natively in the LPServe/SLAI-derived serving framework and compared with existing policies in that same framework. The first target is **Primal Heuristic 1**: the myopic ILP's continuous relaxation followed by deterministic, feasibility-preserving integer extraction.

Correctness and reproducibility come before performance. Proposed mathematics, observed framework behavior, scheduler requirements, implementation, and experimental evidence are different kinds of claims and must remain visibly separated.

## Repository knowledge map

| Path | Role | What it does not establish |
|---|---|---|
| `docs/math/main-llm-serving.tex` | Mathematical source of truth for the proposed scheduling formulation, especially the myopic ILP and `Primal Heuristic 1: Approximation via LP Relaxation`. | Existing LPServe behavior or an implemented scheduler. Later heuristic, Lagrangian, MPC, and extension sections are not part of the first target unless scope is explicitly revised. |
| `docs/LP Scheduler Research Context.md` | Research context: assumptions, implementation principles, validation requirements, baseline policy, phase boundaries, open research questions, and the intended workflow. | Code-grounded proof of framework behavior or a replacement for the mathematical formulation and normative design. |
| `docs/lpserve_scheduler_architecture_summary.md` | Compact implementation-facing digest of the Phase C architecture audit. Read it before the full audit when quick orientation to scheduler, engine, sequence, and block-management behavior is needed. | Normative LP-scheduler requirements, new evidence beyond the full audit, or authority over the full audit when they conflict. |
| `docs/lpserve_scheduler_architecture.md` | Descriptive, code-grounded Phase C audit of the existing LPServe/SLAI-derived framework at commit `c3e014363dd50e1830d7c85c3d043eab69fdc9e5`. It records request state, queues, allocation, execution/replay, metrics, defects, gaps, and exact evidence labels. | A proposed LP-scheduler specification, proof about later revisions, or runtime validation of every inspected path. |
| `docs/lp_scheduler_design.md` | Normative implementation specification for the proposed scheduler across Phases D, E, and F. It defines responsibilities, interfaces, invariants, validation gates, explicit **OPEN** decisions, and **BLOCKER** boundaries. | Permission to fill OPEN items with convenient defaults or to bypass blockers. It does not turn a design requirement into verified implementation. |
| `docs/project_status.md` | Chronological project state and handoff record. Its current local version records Phase C closure / Phase D preparation while preserving Phase A/B history. | Mathematical, architectural, or design authority. It may lag newer artifacts and must not override them. |
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
- For what the first LP scheduler must do, use `docs/lp_scheduler_design.md`, while preserving every OPEN decision and BLOCKER and checking it against the mathematical source.
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

The Phase C architecture audit and the Phase D–F normative design artifacts exist, and `docs/project_status.md` now records **Phase C closure / Phase D preparation**. Phase D implementation has not started. The next work is to resolve only the OPEN decisions required by the pure Phase D mathematical layer, record the approved resolutions in `docs/lp_scheduler_design.md`, and then implement and validate that layer.

The key separation is:

```text
LPServe state -> Phase E snapshot/mapping -> Phase D solve/extraction
              -> validated integer plan -> Phase F native execution
              -> SchedulerOutputs
```

Phase D must remain independent of mutable LPServe objects and GPU execution. Phase E is read-only. Phase F must not run after an unsuccessful prior stage and remains constrained by the design's unresolved decisions and blockers.

## Work / research-design sessions

Begin with this guide, then read only the detailed sources needed for the question:

1. Read the relevant formulation and Primal Heuristic 1 material in `docs/math/main-llm-serving.tex`.
2. Use `docs/LP Scheduler Research Context.md` for the research boundary, validation expectations, and phase discipline.
3. Use `docs/lp_scheduler_design.md` for normative interfaces, invariants, decisions, and blockers.
4. Consult `docs/lpserve_scheduler_architecture_summary.md` first for orientation, then `docs/lpserve_scheduler_architecture.md` for detailed evidence or whenever a framework claim affects implementation; inspect current code whenever a framework claim affects the conclusion.
5. Use status, experiment, and Unity records only for historical or observed claims within their stated provenance.

Keep verified facts, inferences, proposed requirements, unresolved choices, and observed results labeled separately. Research sessions may analyze OPEN decisions, but must not present a selection as approved until it has been explicitly decided and recorded.

## Codex / implementation sessions

Before editing, record the branch, commit, and working-tree state, preserve unrelated user changes, and compare scheduler-relevant code with the audit baseline. For Phase D, read `docs/lp_scheduler_design.md` first for normative requirements, then `docs/lpserve_scheduler_architecture_summary.md` for architecture orientation, then the full audit and current source code only as needed for detailed framework claims; use Sections 7, 9–11, and 14–18 of the design and the corresponding ILP/relaxation/extraction sections of `docs/math/main-llm-serving.tex`.

Implementation tasks should state the objective, files allowed to change, required behavior and invariants, tests, and out-of-scope work. Do not infer LPServe behavior from modern vLLM or another Sarathi/SLAI version. Do not hard-code utility policy, solver/status policy, tolerances, tie-breaking, memory coefficients or reserve, failure behavior, or other OPEN decisions.

Validate in phase order: syntax/import/static checks; focused CPU tests; synthetic mathematical tests; scheduler-state/integration tests; a tiny GPU smoke test; inspection of scheduling decisions and state transitions; then larger experiments. Never report a command, test, or experiment as successful without observing its output, and never substitute aggregate throughput or latency for scheduler-correctness evidence.

## Documentation maintenance

- Keep repository-relative paths and commit/environment provenance exact.
- Update `docs/project_status.md` only after a phase's acceptance evidence has been observed; it records chronology and handoffs, not specifications.
- Record controlled run configuration and evidence in `docs/experiment_reference.md` or a clearly scoped successor; do not overwrite historical provenance.
- Keep `docs/lpserve_scheduler_architecture.md` commit-pinned. Record later behavior in a revisioned audit or dated addendum after reinspection.
- Update `docs/lp_scheduler_design.md` when a design decision is explicitly approved or a contract changes, including rationale, affected interfaces, tests, and experiment metadata where applicable.
- Change the mathematical source only for an intentional mathematical amendment, and trace consequential design changes.
- Document only implemented and observed facts. Preserve OPEN decisions and BLOCKERs until their resolution and validation are explicit.
