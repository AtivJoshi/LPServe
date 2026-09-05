# LPServe LP-Relaxation Scheduler Design

**Document role:** Normative implementation specification  
**Target repository:** [`AtivJoshi/LPServe`](https://github.com/AtivJoshi/LPServe)  
**Code baseline:** [`c3e014363dd50e1830d7c85c3d043eab69fdc9e5`](https://github.com/AtivJoshi/LPServe/commit/c3e014363dd50e1830d7c85c3d043eab69fdc9e5)  
**Design date:** 2026-09-04  
**Implementation target:** Primal Heuristic 1, LP relaxation followed by deterministic integer extraction

## Table of contents

1. [Purpose, authority, and scope](#1-purpose-authority-and-scope)
2. [Design status and terminology](#2-design-status-and-terminology)
3. [End-to-end responsibility decomposition](#3-end-to-end-responsibility-decomposition)
4. [Scheduling universe and request eligibility](#4-scheduling-universe-and-request-eligibility)
5. [Mathematical objects and domains](#5-mathematical-objects-and-domains)
6. [Capacity model](#6-capacity-model)
7. [Normative ILP and LP relaxation](#7-normative-ilp-and-lp-relaxation)
8. [Planning-memory quantities](#8-planning-memory-quantities)
9. [Pure Phase D interfaces](#9-pure-phase-d-interfaces)
10. [LP solution classification and extraction](#10-lp-solution-classification-and-extraction)
11. [Numerical and feasibility invariants](#11-numerical-and-feasibility-invariants)
12. [Read-only Phase E state mapping](#12-read-only-phase-e-state-mapping)
13. [Validated Phase F action execution](#13-validated-phase-f-action-execution)
14. [Failure contracts](#14-failure-contracts)
15. [Testing and phase acceptance requirements](#15-testing-and-phase-acceptance-requirements)
16. [Deferred blockers and unsupported scope](#16-deferred-blockers-and-unsupported-scope)
17. [Unresolved decision register](#17-unresolved-decision-register)
18. [Traceability, discrepancies, and change control](#18-traceability-discrepancies-and-change-control)

## 1. Purpose, authority, and scope

### 1.1 Purpose

This document specifies what the first LP-relaxation scheduler for LPServe must do, how its mathematical objects map to LPServe state, and which layer owns each correctness obligation. It is the normative design contract for Phases D, E, and F.

The companion document `lpserve_scheduler_architecture.md` answers the descriptive question:

> What does the audited LPServe/SLAI-derived framework currently do?

This document answers the prescriptive question:

> Given that behavior, what must the LP scheduler do, and where must each responsibility live?

Detailed queue walkthroughs, engine diagrams, block-manager internals, and comparisons among existing schedulers remain in the architecture reference. This document cites those findings only where they impose a design constraint.

### 1.2 Source precedence

The sources are authoritative in this order:

1. `docs/math/main-llm-serving.tex` for the mathematical scheduling formulation;
2. `LP Scheduler Research Context.md` for implementation principles, validation requirements, and phase boundaries;
3. `lpserve_scheduler_architecture.md` for verified behavior at the audited LPServe revision;
4. `project_status.md` for chronological project state and phase handoffs;
5. the current LPServe repository when a code fact is absent, ambiguous, or suspected to have changed.

If two sources appear inconsistent, an implementation must stop and record the discrepancy. It must not silently select whichever interpretation is easier to implement.

### 1.3 Normative language

The terms **MUST**, **MUST NOT**, **SHOULD**, and **MAY** express requirements in decreasing order of strength. An item labeled **OPEN** is not a default. Implementation code must not decide an OPEN item implicitly.

The following evidence labels are used:

- **Resolved mathematical requirement:** fixed by `docs/math/main-llm-serving.tex`.
- **Resolved architecture fact:** verified by the Phase C audit at the pinned commit.
- **Normative design requirement:** imposed here to preserve the mathematical and framework contracts.
- **OPEN research decision:** a policy choice that must remain externally visible.
- **BLOCKER:** a known condition that prevents a correctness claim for the affected feature.

### 1.4 In scope

The first implementation includes:

- the myopic ILP from `docs/math/main-llm-serving.tex`;
- its true continuous relaxation, including continuous $x_i$;
- explicit solver-result handling;
- deterministic, feasibility-preserving integer extraction;
- a framework-light Phase D mathematical layer;
- read-only Phase E construction of LP inputs from LPServe;
- validated Phase F translation of integer actions into LPServe-native operations;
- synthetic, state-mapping, integration, and tiny GPU validation.

### 1.5 Out of scope

The following are not part of this design:

- Primal Heuristic 2;
- Lagrangian or hierarchical slow-path/fast-path scheduling;
- MPC or predictive formulations;
- CPU swapping;
- production-grade solver optimization before correctness is established;
- an unproved approximation ratio;
- equivalence claims between LPServe policies and current upstream serving systems;
- pipeline-parallel support before the blockers in Section 16 are resolved.

Only decisions recorded as resolved in Section 17 are selected. Utility
functions, watermarks, tie-breaking, fallback and rollback behavior, pipeline
support, and every remaining OPEN item have no implicit default.

## 2. Design status and terminology

### 2.1 Revision boundary

The Phase C architecture audit and this design's code baseline are pinned to commit `c3e0143`. GitHub `main` was verified to point to that commit when the design was prepared. The later committed changes through `6fbc046` were documentation-only and did not change the audited implementation paths; they do not repin the historical architecture audit. Architecture claims in this document therefore remain claims about `c3e0143`. If implementation begins from a later commit or a working tree with relevant source changes, the affected paths in the architecture reference MUST be rechecked before relying on them.

### 2.2 Terminology bridge

| Mathematical term | LPServe-facing meaning |
|---|---|
| Request (i) | A `Sequence`, identified by `seq_id` |
| Positive prefill $x_i>0$ | `SequenceScheduleMetadata(seq_id, prompt_chunk_len=x_i)` |
| Decode $y_i=1$ | `SequenceScheduleMetadata(seq_id, prompt_chunk_len=0)` |
| Preemption $z_i=1$ | Recomputation preemption: release physical KV blocks and return the sequence to `WAITING` through native replay |
| Do nothing | Omit the request from scheduled, preempted, and ignored output fields and leave its state unchanged |
| Resident | Owned by the scheduler as GPU-resident and possessing the required physical block table |
| Planning memory | Scalar KV-block accounting used by the LP; not proof of allocator feasibility |
| Physical feasibility | Exact legality under current block tables, status, ownership, admission/append gates, and mutation order |

`prompt_chunk_len` MUST never be negative. The audited constructor would treat a negative value as decode metadata rather than rejecting it, so the LP executor must validate this before constructing output.

### 2.3 Separation from the architecture reference

This document relies especially on Sections 6–8, 11–18, 20–23, and 26–28 of `lpserve_scheduler_architecture.md`. Those sections remain the authoritative detailed account of request state, ownership, scheduler output replay, block allocation, recomputation, pipeline behavior, and known defects. Their walkthroughs are not duplicated here.

## 3. End-to-end responsibility decomposition

The scheduler follows this conceptual flow:

```text
LPServe state
    ↓
read-only state snapshot
    ↓
utility construction
    ↓
LP problem construction
    ↓
continuous LP solve
    ↓
integer extraction
    ↓
mathematical plan validation
    ↓
fresh LPServe physical prevalidation
    ↓
LPServe-native action execution
    ↓
SchedulerOutputs
```

The decomposition is conceptual, not a required class hierarchy. Implementations SHOULD introduce only the abstractions needed for purity, testability, and explicit failure handling.

| Responsibility | Owning phase/layer | Mutation allowed? |
|---|---|---:|
| Define variables, constraints, and extraction | Phase D mathematical layer | No |
| Construct utilities | Policy input to Phase D | No LPServe mutation |
| Snapshot requests and resources | Phase E state mapper | No |
| Solve and extract | Phase D | No |
| Validate mathematical plan | Phase D | No |
| Revalidate current physical state | Phase F precommit validator | No |
| Apply queue and central block changes | Phase F executor | Yes, after successful prevalidation |
| Construct `SchedulerOutputs` | Phase F executor | Yes, as part of the same ordered commit |
| Replay engine/worker changes | Existing LPServe engine and workers | Yes |

Phase D MUST NOT import or accept mutable LPServe `Sequence`, scheduler, block-manager, or engine objects. Phase E MUST NOT call allocation, append, free, preemption, status-transition, or queue-mutation methods. Phase F MUST NOT run if any preceding result is unsuccessful.

## 4. Scheduling universe and request eligibility

### 4.1 Mathematical request set

At scheduler decision epoch (t),

$$
U_t
=
\{i : i\text{ is arrived, non-finished, scheduler-owned, and relevant at }t\}.
$$

The Phase E snapshot MUST:

1. begin from the new LP scheduler's authoritative ownership structures;
2. deduplicate by `seq_id`;
3. reject contradictory duplicate objects or ownership classifications;
4. exclude future arrivals;
5. exclude finished requests;
6. preserve every remaining owned request exactly once.

The mapper MUST NOT assume that `waiting ∪ running` is universally complete merely because those collections exist in `BaseScheduler`. The architecture audit showed that SLAI auxiliary queues can own work not represented by that union. The new scheduler must define and test its own authoritative ownership rule rather than inherit SLAI's auxiliary ownership implicitly.

### 4.2 Remaining prompt work

For every $i\in U_t$,

$$
P_i^{\mathrm{rem}}(t)
=
\texttt{get\_prompt\_len()}
-
\texttt{get\_num\_prompt\_tokens\_processed()}.
$$

Phase E MUST reject a snapshot in which this value is negative. A request is prefill-eligible only if $P_i^{\mathrm{rem}}(t)>0$ and its LPServe status, ownership, arrival state, and allocation state permit the corresponding prefill path.

### 4.3 Decode eligibility

A request is decode-eligible only if:

- it is arrived and non-finished;
- prompt processing is complete;
- $P_i^{\mathrm{rem}}(t)=0$;
- it is resident and allocated;
- its status and ownership allow it to be scheduled at this decision boundary;
- no unresolved in-flight condition prevents the action.

Zero `prompt_chunk_len` metadata is an encoding of a previously validated decode action. It is not evidence that decode is legal.

### 4.4 Legally preemptible set

Define

$$
\mathcal Z_t
=
\left\{
i\in U_t:
i\text{ is resident and legally preemptible at the beginning of }t
\right\}.
$$

Membership requires all of the following:

- scheduler ownership as a resident request;
- non-finished state;
- an allocated physical block table;
- a status accepted by the native preemption path;
- safe release of the physical KV state at this decision boundary;
- availability of a validated recomputation-preemption execution path.

The native helper's acceptance of `RUNNING` or `PAUSED` is necessary evidence but is not sufficient proof of legal preemption. Status alone does not prove ownership, allocation, absence of in-flight use, or correct recomputation output semantics.

For any $i\notin\mathcal Z_t$, the model and extracted plan MUST use $z_i=0$. Phase D MUST receive $\mathcal Z_t$ explicitly; it MUST NOT infer legal preemption from a positive relaxed $z_i$, positive recovery, or resident-looking metadata.

Until the recomputation and control-only blockers in Section 16 are repaired and validated, the integrated Phase E mapper MUST expose no executable preemption candidate. Equivalently, its operational $\mathcal Z_t$ is empty even though synthetic Phase D tests may exercise nonempty preemption sets.

### 4.5 Do nothing and non-LP control actions

The canonical do-nothing action is

$$
x_i=y_i=z_i=I_i^P=0.
$$

It leaves queues, status, prompt progress, and block tables unchanged.

Prompt rejection or ignore is not an LP decision variable. Any prompt-length rejection path remains a separate framework control action and must be validated independently, including the control-only-output defect described in Section 16.

## 5. Mathematical objects and domains

### 5.1 Per-request decision variables

For each $i\in U_t$:

| Object | Integer-domain meaning | LP-relaxation domain |
|---|---|---|
| $x_i(t)$ | Number of prefill tokens; $\mathbb Z_{\ge0}$ | $\mathbb R_{\ge0}$ |
| $y_i(t)$ | Whether one decode token is scheduled; $\{0,1\}$ | $[0,1]$ |
| $I_i^P(t)$ | Whether a positive prefill is scheduled; $\{0,1\}$ | $[0,1]$ |
| $z_i(t)$ | Legal preemption; $\{0,1\}$ for $i\in\mathcal Z_t$, otherwise fixed at zero | $[0,1]$ for $i\in\mathcal Z_t$, otherwise fixed at zero |

The continuous relaxation MUST relax $x_i$. Leaving $x_i$ integral would produce a mixed-integer program rather than the specified LP.

### 5.2 Per-request state parameters

| Object | Meaning | Required unit |
|---|---|---|
| $P_i^{\mathrm{rem}}(t)$ | Unprocessed prompt tokens | Tokens |
| $U_i(t)$ | $\min(P_i^{\mathrm{rem}}(t),C_{\max})$ | Tokens |
| $a_i^P(t)$ | Fixed block charge for any positive prefill | Physical KV blocks |
| $c_i^D(t)$ | Planning charge for one decode action | Physical KV blocks |
| $c_i^Z(t)$ | Blocks recovered by legal preemption | Physical KV blocks |
| $\alpha_i(t)$ | Utility of one decode | OPEN scale |
| $\beta_i(t)$ | Utility per prefill token | OPEN scale |
| $\gamma_i(t)$ | Preemption penalty coefficient | OPEN scale |

Utility coefficients MUST be supplied explicitly. The mathematical sources do not establish their values, normalization, or final admissible sign restrictions.

### 5.3 System parameters

| Object | Meaning |
|---|---|
| $B_{\max}$ | Maximum combined number of prefill and decode tokens in the next forward pass |
| $C_{\max}$ | Maximum prefill chunk offered to any one request |
| $S_{\max}$ | Maximum number of scheduled execution actions in the next forward pass |
| $M_t^{\mathrm{free}}$ | Free physical KV blocks at the snapshot boundary |
| $W_t$ | Explicit planning-memory reserve; OPEN policy |
| Resident capacity | Separate LPServe-native cap derived from `max_num_seqs`; not $S_{\max}$ |

## 6. Capacity model

### 6.1 Token-volume budget $B_{\max}$

The target accounting is

$$
\sum_{i\in U_t}(x_i+y_i)\le B_{\max}.
$$

Each prefill token costs one unit and each scheduled decode costs one unit. This matches the accounting form used by LPServe's Sarathi and SLAI policies, but it does not assert equal wall-clock cost for the two token types.

The concrete configuration field and whether the value is static or dynamic are OPEN. LPServe's VLLM-named policy budget is not an acceptable semantic substitute because that policy does not charge decode actions to its admission budget.

### 6.2 Per-request chunk cap $C_{\max}$

For every request,

$$
0\le x_i\le U_i,
\qquad
U_i=\min(P_i^{\mathrm{rem}},C_{\max}).
$$

$C_{\max}$ controls the largest candidate prefill chunk. It is not a memory coefficient. LPServe prompt chunks do not have a universal block-alignment requirement.

The exact binding to a fixed Sarathi chunk, a dynamic chunk, the total token budget, or a new explicit configuration field is OPEN.

### 6.3 Scheduled-action width $S_{\max}$

The action-width constraint is

$$
\sum_{i\in U_t}(I_i^P+y_i)\le S_{\max}.
$$

Preemption is not counted in this expression because `docs/math/main-llm-serving.tex` defines $S_{\max}$ as the number of execution actions sent to the next forward pass. For a valid LPServe output,

$$
\sum_i(\hat I_i^P+\hat y_i)
=
|\texttt{scheduled\_seq\_metadata\_list}|.
$$

The numerical value and configuration source for $S_{\max}$ are OPEN.

### 6.4 Separate resident capacity

LPServe's `max_num_seqs` chiefly constrains resident or active requests, not the width of a single forward pass. The new scheduler MUST enforce resident capacity separately from $S_{\max}$.

Let $R_t$ be the authoritative resident set before the plan, $A_t$ the set of nonresident requests selected for admission, and $P_t$ the set selected for preemption. The precommit validator MUST enforce

$$
R_t'=(R_t\setminus P_t)\cup A_t,
\qquad
|R_t'|\le\texttt{max\_num\_seqs}.
$$

Resident decode and resident partial-prefill actions do not add a new resident. A selected preemption removes one only if the request is actually resident and the preemption is legal. An admission consumes a resident slot even when its selected prompt chunk is small.

This check is deliberately outside the mathematical LP so that the target relaxation retains exactly the three global coupling constraints in `docs/math/main-llm-serving.tex`. Phase E must nevertheless provide enough ownership information for extraction and Phase F to reject a plan that violates resident capacity. If later work adds resident capacity to the LP itself, the structural argument about the number of fractional request blocks must be reconsidered.

### 6.5 Decode caps from existing policies

SLAI's `limit_total_decodes` is a policy-specific additional constraint. It MUST NOT be imported into the target LP without an explicit mathematical revision. Adding it would introduce another global coupling constraint and could alter the almost-integral structural argument.

## 7. Normative ILP and LP relaxation

For readability, the explicit time argument is omitted below.

### 7.1 Integer formulation

The one-step ILP is

$$
\max_{\mathbf x,\mathbf y,\mathbf z,\mathbf I^P}
\quad
\sum_{i\in U_t}
\left(\alpha_i y_i+\beta_i x_i\right)
-
\sum_{i\in\mathcal Z_t}\gamma_i z_i
$$

subject to

$$
\sum_{i\in U_t}(x_i+y_i)\le B_{\max},
\tag{token volume}
$$

$$
\sum_{i\in U_t}(I_i^P+y_i)\le S_{\max},
\tag{action width}
$$

$$
\sum_{i\in U_t}
\left(a_i^P I_i^P+c_i^D y_i\right)
-
\sum_{i\in\mathcal Z_t}c_i^Z z_i
\le M_t^{\mathrm{free}}-W_t,
\tag{planning memory}
$$

$$
I_i^P\le x_i\le U_iI_i^P
\qquad \forall i\in U_t,
\tag{prefill linkage}
$$

$$
y_i\le\mathbf 1\{P_i^{\mathrm{rem}}=0\}
\qquad \forall i\in U_t,
\tag{decode causality}
$$

and

$$
I_i^P+y_i+z_i\le1
\qquad \forall i\in U_t.
\tag{mutual exclusion}
$$

The integer domains are those in Section 5. For $i\notin\mathcal Z_t$, $z_i=0$.

### 7.2 Continuous relaxation

The LP retains every constraint above and replaces the domains with

$$
x_i\in\mathbb R_{\ge0},
\qquad
y_i,I_i^P\in[0,1],
$$

and

$$
z_i\in[0,1]\quad(i\in\mathcal Z_t),
\qquad
z_i=0\quad(i\notin\mathcal Z_t).
$$

The relaxed solution is denoted

$$
(\tilde{\mathbf x},\tilde{\mathbf y},
  \tilde{\mathbf z},\tilde{\mathbf I}^{P}).
$$

It is never directly executable.

### 7.3 Three-global-constraint boundary

The token-volume, action-width, and planning-memory inequalities are the three global coupling constraints. Prefill linkage, causality, mutual exclusion, and preemption support are request-local.

The block-angular structure motivates investigating an almost-integral solution, but neither the Fundamental Theorem of Linear Programming nor the presence of three global constraints alone proves that at most three requests are fractional. Runtime logic MUST handle an arbitrary fractional set.

## 8. Planning-memory quantities

### 8.1 Free memory and reserve

$M_t^{\mathrm{free}}$ is the block manager's current free physical-block count at the read-only snapshot boundary. $W_t$ is an explicit reserve measured in the same unit.

The initial admission watermark and the append path are asymmetric: admission applies a one-percent watermark, whereas append does not preserve it. Therefore $W_t$ MUST NOT silently be set equal to the block manager's watermark. Its value and purpose remain OPEN.

### 8.2 Fixed prefill admission charge

For a positive prefill action,

$$
a_i^P(t)
=
\begin{cases}
0,
& \text{if (i) is resident and already allocated},\\
A_i(t),
& \text{if (i) requires admission or recomputation},
\end{cases}
$$

where

$$
A_i(t)=|\texttt{logical\_token\_blocks}_i|.
$$

The fixed charge is incurred through $a_i^P I_i^P$. An unallocated request must be able to allocate its complete current logical context even if only one prompt token will be processed. Reducing $x_i$ does not reduce $a_i^P$.

A resident partial prefill normally has equal logical and physical block-table lengths and requires no additional admission allocation, so its prefill charge is zero. Phase E MUST verify residency and allocation rather than inferring zero cost from prompt progress alone.

### 8.3 Decode demand

The exact next-step marginal block demand is

$$
d_i^{\mathrm{exact}}(t)
=
\max\left(
0,
|\texttt{logical\_token\_blocks}_i|
-
|\texttt{physical\_block\_table}_i|
\right),
$$

normally zero or one.

The existing native append gate is more conservative: it requires at least one free block even when the exact marginal demand is zero. The planning definition of $c_i^D$ is OPEN between:

- exact marginal demand $d_i^{\mathrm{exact}}$; and
- a conservative one-block charge aligned with the current gate.

Whichever policy is selected MUST be explicit in configuration or policy construction, recorded in experiment metadata, covered by tests, and followed by exact physical prevalidation.

### 8.4 Preemption recovery

For every $i\in\mathcal Z_t$,

$$
c_i^Z(t)
=
|\texttt{physical\_block\_table}_i|.
$$

This is the exact number of unique physical blocks returned by native `free(seq)` for the audited non-sharing block manager. A candidate with $c_i^Z=0$ cannot make progress in safety preemption and MUST be excluded from the repair candidate set.

### 8.5 Planning feasibility is not execution feasibility

Satisfying

$$
\sum_i(a_i^PI_i^P+c_i^Dy_i)
-
\sum_{i\in\mathcal Z_t}c_i^Zz_i
\le M_t^{\mathrm{free}}-W_t
$$

does not prove that LPServe can execute the plan. The scalar inequality omits the admission watermark asymmetry, conservative append gate, ownership and status legality, in-flight safety, duplicate actions, central/worker replay, and partial failure.

Accordingly, Phase D proves planning feasibility only. Phase F owns exact physical and operational prevalidation.

## 9. Pure Phase D interfaces

This section specifies semantic interfaces, not implementation code or mandatory class names.

### 9.1 LP request record

Each request record MUST contain only immutable, framework-independent data:

- stable request ID;
- $P_i^{\mathrm{rem}}$ and $U_i$;
- prefill-, decode-, and preemption-eligibility flags;
- $a_i^P,c_i^D,c_i^Z$;
- $\alpha_i,\beta_i,\gamma_i$;
- an immutable lexicographic `order_key`.

It MUST NOT contain a mutable `Sequence`, scheduler collection, block manager, callback, or engine object.

`order_key` MUST be present, unique within the snapshot, finite/comparable, and
normalized to a documented type, preferably a tuple of integers; malformed,
duplicate, or unstable keys are invalid input. Tests may use `(3,)`. Phase E
must derive a stable live key, expected to resemble
`(monotone_admission_order, seq_id)`; `seq_id` alone is not the contract.

### 9.2 LP problem record

The problem record MUST contain:

- the ordered, unique request records defining $U_t$;
- explicit membership of $\mathcal Z_t$;
- $B_{\max},C_{\max},S_{\max},M_t^{\mathrm{free}},W_t$;
- the selected decode-memory policy identifier;
- the selected numerical-policy identifier or explicit tolerance values;
- a snapshot identifier sufficient for Phase F stale-state detection.

The record MUST be validated and sorted ascending by `order_key` before solver
invocation. This canonical order is used for LP variables and solver-vector
reconstruction.

### 9.3 Solver result

The first correctness-focused Phase D implementation MUST use
`scipy.optimize.linprog` with the HiGHS solver family, isolated behind a
project-owned solver adapter. The adapter MUST explicitly convert this
document's maximization objective to SciPy's minimization convention and MUST
use a deterministic, explicit variable ordering and request-to-variable
association.

The adapter MUST return a project-owned structured result. Raw SciPy
`OptimizeResult` objects and other SciPy-specific state MUST NOT cross the
Phase D interface. Raw diagnostics SHOULD retain the solver identity, SciPy
version, selected method and options, raw status, success flag, message, and
iteration information when available.

A solver result MUST distinguish at least:

- optimal candidate returned;
- infeasible problem;
- unbounded problem;
- numerical failure;
- solver limit or interruption;
- invalid/malformed solver output;
- unexpected solver exception.

The adapter MUST normalize SciPy results as follows:

| SciPy status or outcome | Project-owned category |
|---|---|
| `0` | Optimal candidate |
| `1` | Solver limit or early termination |
| `2` | Infeasible |
| `3` | Unbounded |
| `4` | Numerical or solver difficulty |
| Unknown status or structurally inconsistent result | Invalid/malformed solver output |
| Exception or interruption | Explicit exception or interruption non-success |

Correctness MUST NOT depend on the human-readable solver message. Only a
normalized optimal candidate corresponding to SciPy `status == 0` may proceed
to independent relaxed-solution validation. Every other normalized outcome is
a Phase D non-success and MUST NOT proceed to extraction, including solver
limit or early termination, infeasible, unbounded, numerical or solver
difficulty, invalid/malformed output, exception, and interruption.

A status-0 result is only an optimal candidate. It MUST pass the independent
relaxed-solution validator before it becomes a validated optimal relaxed
solution eligible for extraction. A status-0 result with `success != True` is
invalid/malformed, as is a status-0 result without a usable primal vector.
Human-readable message text is non-contractual and MUST NOT affect outcome
classification.

A vector returned with status `1`, `2`, `3`, `4`, an unknown status, an
exception, or an interruption MUST NOT be exposed across the adapter boundary
as an approved or extractable relaxed solution, even if it appears feasible or
could independently be shown primal-feasible. Rejected raw results MAY retain
diagnostic facts, including whether a vector was present and its shape, but the
vector MUST NOT cross the interface as extractable state. Non-success results
MUST NOT contain fabricated decisions, silently rounded values, or a substitute
all-zero schedule.

The required admission flow is:

```text
raw SciPy result
    ↓
D-11 normalization
    ↓
optimal candidate OR non-success
    ↓ only optimal candidate
independent relaxed-solution validation
    ↓
validated optimal relaxed solution
    ↓
integer extraction
```

For the first correctness-focused Phase D implementation, the adapter MUST
request `method="highs-ds"`, MUST pass `options={"presolve": True}`, and MUST
omit `time_limit` and `maxiter` so that it imposes no finite solver limit. It
MUST NOT explicitly configure crossover or pass undocumented thread-count,
parallelism, or random-seed options through `linprog`. Other SciPy/HiGHS tuning
remains at the documented defaults of the exact pinned SciPy version unless a
later approved decision changes it. D-13 does not select that exact version;
the dependency and environment verification required by D-11 still owns it.

Phase D correctness requires, in order:

1. a D-12-admissible status-0 optimal candidate;
2. successful independent relaxed-solution validation;
3. deterministic extraction under the selected numerical and ordering
   policies; and
4. successful integer-plan validation.

It does not require proof that the solver returned a basic or extreme-point
solution. Finite limits for live or performance use, future method
reconsideration, basis inspection or certification, and advanced solver
controls remain in the OPEN portion of D-13. Numerical feasibility tolerance
and projection are defined by D-15, and the live-scheduler response to a Phase D
non-success remains under D-17.

### 9.4 Integer action plan

A successful extraction returns, for every request ID:

- integer prefill token count $\hat x_i$;
- binary decode decision $\hat y_i$;
- binary preemption decision $\hat z_i$;
- canonical prefill indicator $\hat I_i^P$.

The plan MUST also expose:

- residual token, action-width, and planning-memory capacities;
- the number of indicator-fractional requests observed;
- the preemptions selected dominantly and for safety repair;
- the numerical policy used;
- the source problem/snapshot identifier;
- success or an explicit extraction-failure reason.

Phase D action records and action subsets MUST be emitted ascending by
`order_key`.

### 9.5 Validators

Phase D requires distinct validators for:

1. problem-input validity;
2. relaxed-solution domains and constraints;
3. extracted integer-plan domains and constraints.

Validators MUST return structured success/failure information. Assertions MAY supplement invariant checks in tests, but correctness MUST NOT rely only on assertions that can be disabled.

The relaxed-solution validator MUST preserve the raw primal vector for
diagnostics and expose only its separately normalized, validated vector to
D-14 and extraction.

### 9.6 Purity requirements

With identical ordered LP data, `highs-ds`, explicit presolve, no
project-imposed solver limits, identical numerical and extraction policies, and
the same pinned software environment, Phase D expects stable solver status and
objective behavior. Deterministic problem construction and post-solver
processing MUST produce the same result when given the same relaxed solution.
The solver contract does not promise bitwise or cross-platform determinism, or
identical primal-vector selection among multiple optima across different
solver versions, builds, or platforms.

Phase D MUST NOT read clocks, global scheduler state, mutable queues, block
managers, or randomness unless a future design explicitly introduces and
records such inputs.

## 10. LP solution classification and extraction

### 10.1 Numerical classification

Mathematically, an action indicator is integral when it is exactly zero or one. In software, classification MUST use an explicit tolerance policy.

The Phase D numerical policy MUST set the absolute integrality tolerance to

$$
\varepsilon_{\mathrm{int}}=10^{-6}.
$$

It applies only to the relaxed action indicators $\tilde y_i$,
$\tilde I_i^P$, and $\tilde z_i$. It does not classify or round
$\tilde x_i$, and it MUST NOT use a relative tolerance or a generic closeness
helper with a nonzero implicit relative tolerance.

D-14 classification runs only after the D-15 relaxed-solution validator has
accepted the value and performed the narrow projection D-15 permits. Its input
$v$ must therefore already be finite and in $[0,1]$. For
such an input, use the closed endpoint intervals

$$
\operatorname{classify}(v)=
\begin{cases}
0,
&0\le v\le\varepsilon_{\mathrm{int}},\\[2mm]
1,
&1-\varepsilon_{\mathrm{int}}\le v\le1,\\[2mm]
\mathrm{fractional},
&\varepsilon_{\mathrm{int}}<v<1-\varepsilon_{\mathrm{int}}.
\end{cases}
$$

The endpoint intervals are disjoint because
$\varepsilon_{\mathrm{int}}=10^{-6}<1/2$, so classification has no tie and
does not resolve or depend on D-16.

D-14 is an interpretation of an already validated in-domain value, not a
feasibility tolerance or bound-clamping allowance. In particular, it MUST NOT
classify a raw value such as `-5e-7` as zero. D-15 must first decide whether an
out-of-domain value is rejected or legitimately normalized under Section 11.2.
D-14 performs no such acceptance or normalization.

### 10.2 Integral/fractional partition

A request belongs to $U_{\mathrm{int}}$ only when all three action indicators

$$
\tilde y_i,\quad\tilde I_i^P,\quad\tilde z_i
$$

are classified as exact binary values. If any one remains fractional, the
request belongs to $U_{\mathrm{frac}}$. The value of $\tilde x_i$ does not
affect this partition. For $i\notin\mathcal Z_t$, $\tilde z_i=0$ by
construction.

After classification, zero and one are represented as exact `0.0` and `1.0`
in the classified extraction state; fractional indicators retain their
validated relaxed values. The original tilded solver values MUST remain
available separately for diagnostics.

For $i\in U_{\mathrm{int}}$:

$$
\hat y_i=\tilde y_i,
\qquad
\hat z_i=\tilde z_i,
\qquad
\hat I_i^P=\tilde I_i^P,
\qquad
\hat x_i=\lfloor\tilde x_i\rfloor,
$$

after numerical classification has produced exact binary indicator values.
D-14 does not alter $\tilde x_i$. In particular, classifying a near-one
$\tilde I_i^P$ as exact one does not authorize promoting or clamping
$\tilde x_i$. Any tension created with
$I_i^P\le x_i\le U_iI_i^P$ MUST be rejected by the D-15 handoff check when it
would contradict an indicator-integral lock; D-14 performs no coupled repair.

Extraction MUST retain both the tilded relaxed solution and the hatted plan. It MUST NOT overwrite integral decisions while processing fractional requests.

### 10.3 Residual capacities

After fixing $U_{\mathrm{int}}$, initialize

$$
B_{\mathrm{rem}}
=B_{\max}
-\sum_{i\in U_{\mathrm{int}}}(\hat x_i+\hat y_i),
$$

$$
S_{\mathrm{rem}}
=S_{\max}
-\sum_{i\in U_{\mathrm{int}}}(\hat I_i^P+\hat y_i),
$$

and

$$
M_{\mathrm{rem}}
=(M_t^{\mathrm{free}}-W_t)
-\sum_{i\in U_{\mathrm{int}}}
\left(a_i^P\hat I_i^P+c_i^D\hat y_i-c_i^Z\hat z_i\right).
$$

These residual capacities MUST be recomputed from the original problem data
and the hatted integer state, not from SciPy slack fields or relaxed-solution
residuals. Negative, non-finite, or otherwise invalid final residuals are
extraction or integer-plan validation failures; D-15 tolerance does not apply.

### 10.4 Dominant legal preemptions

For each request in $U_{\mathrm{frac}}\cap\mathcal Z_t$, compare

$$
\tilde y_i,\quad\tilde I_i^P,\quad\tilde z_i.
$$

If preemption is the dominant relaxed action and $c_i^Z>0$, set $\hat z_i=1$ and increase current planning memory by $c_i^Z$.

Compare D-15-normalized values with exact equality only. Within-request ties
use `decode > prefill > preempt`; preemption is dominant only when strictly
greater than both competing actions.

### 10.5 Safety preemption

After dominant preemptions, while $M_{\mathrm{curr}}<0$:

1. consider only unused requests in $U_{\mathrm{frac}}\cap\mathcal Z_t$ with $c_i^Z>0$;
2. select maximum remaining normalized $\tilde z_i$; on equality, the smallest `order_key`;
3. set $\hat z_i=1$;
4. increase $M_{\mathrm{curr}}$ by $c_i^Z$.

If no eligible positive-recovery candidate exists while memory remains negative, extraction MUST fail. It MUST NOT select an ineligible request, create memory credit, or reduce a prefill chunk as a substitute for the fixed admission charge.

Multiple safety preemptions are permitted and must be tested.

### 10.6 Fractional decode and prefill packing

For every remaining unpreempted fractional request in ascending `order_key`,
compare $\tilde y_i$ and $\tilde I_i^P$; on equality, choose decode.

A decode may be selected only if

$$
B_{\mathrm{curr}}\ge1,
\qquad
S_{\mathrm{curr}}\ge1,
\qquad
M_{\mathrm{curr}}\ge c_i^D.
$$

After selection, subtract one token, one action slot, and $c_i^D$ blocks.

A prefill may be selected only if

$$
B_{\mathrm{curr}}\ge1,
\qquad
S_{\mathrm{curr}}\ge1,
\qquad
M_{\mathrm{curr}}\ge a_i^P.
$$

Then set

$$
\hat x_i
=
\min(P_i^{\mathrm{rem}},C_{\max},B_{\mathrm{curr}}).
$$

If $\hat x_i>0$, set $\hat I_i^P=1$ and subtract $\hat x_i$ tokens, one action slot, and the complete $a_i^P$ charge.

If the full $a_i^P$ charge does not fit, the specified initial extraction skips that fractional prefill. It does not shrink the charge or automatically seek additional preemptions. A later policy that preempts specifically to admit such a prefill would be a separate design change.

### 10.7 Canonicalization and final validation

Before success is returned:

$$
\hat I_i^P
=
\mathbf1\{\hat x_i>0\}
\qquad\forall i\in U_t.
$$

Canonicalization MUST NOT conceal a contradiction. If a pre-canonical plan contains $\hat I_i^P=1,\hat x_i=0$ or $\hat I_i^P=0,\hat x_i>0$, extraction must record the inconsistency and fail unless it arose solely from an explicitly permitted representation conversion whose correctness is independently checked.

The complete integer plan MUST pass the validator in Section 11 before Phase E or Phase F can consume it.

### 10.8 Fractional-count observability

The implementation MUST record $ |U_{\mathrm{frac}}| $ for each solve or expose it to the scheduler's instrumentation. It MUST safely process values greater than three, and MUST NOT reject an otherwise valid relaxed solution or extracted plan merely because the count exceeds the structural expectation. An empirical concentration near three is a measurement, not a proof.

The numerical-policy identifier, configured $\varepsilon_{\mathrm{int}}$, and
aggregate counts of zero-, one-, and fractional-classified indicators MUST be
recorded or exposed. Targeted debug or trace output SHOULD make the raw
indicator, its classification, and its distance to the nearest endpoint
inspectable. Routine logs need not contain every per-indicator value. The
tolerance MUST NOT be adjusted to drive $|U_{\mathrm{frac}}|$ toward an
expected theoretical value.

## 11. Numerical and feasibility invariants

### 11.1 Input invariants

Before LP construction:

- request IDs are unique;
- capacities and block counts are finite and expressed in declared units;
- $P_i^{\mathrm{rem}}\ge0$;
- $C_{\max}>0$ and $U_i=\min(P_i^{\mathrm{rem}},C_{\max})$;
- eligibility flags agree with zero upper bounds for unavailable actions;
- $c_i^Z=0$ or absent outside $\mathcal Z_t$;
- every utility coefficient is finite;
- all OPEN policies required by this invocation were supplied explicitly.

The precise admissible lower bounds for $B_{\max},S_{\max},M_t^{\mathrm{free}}$, and $W_t$ must be established by the numerical policy. Counts derived from LPServe cannot be negative.

### 11.2 Relaxed-solution invariants

The independent validator MUST use the project-owned, absolute feasibility
tolerance

$$
\varepsilon_{\mathrm{feas}}=10^{-7}.
$$

It is applied in the LP's declared units, is separate from
$\varepsilon_{\mathrm{int}}=10^{-6}$, and has no relative component. For an
inequality $a^Tv\le b$, acceptance requires
$a^Tv-b\le\varepsilon_{\mathrm{feas}}$; for an equality or fixed-variable
constraint, it requires $|a^Tv-b|\le\varepsilon_{\mathrm{feas}}$.

For every D-12-admissible candidate, the validator MUST:

- verify vector length, deterministic variable ordering, and request
  association;
- reject non-finite variables and preserve the raw primal vector;
- independently recompute all bound, fixed-variable, global, and request-local
  residuals from the project-owned problem record and reject any raw residual
  above $\varepsilon_{\mathrm{feas}}$;
- apply only the permitted projection below, recompute every residual, and
  reject any normalized residual above $\varepsilon_{\mathrm{feas}}$; and
- expose only the normalized validated vector to D-14 and extraction.

Solver-provided residuals are diagnostic only. The independent checks include
all variable domains, $z_i=0$ outside $\mathcal Z_t$, the three global
constraints, prefill linkage, decode causality, and mutual exclusion.

Projection is permitted only for violations of explicit single-variable
bounds within the closed tolerance boundary: $x_i\in
[-\varepsilon_{\mathrm{feas}},0)$ or an indicator in that interval may become
exact `0.0`; an indicator in $(1,1+\varepsilon_{\mathrm{feas}}]$ may become
exact `1.0`; and a variable fixed to zero may become exact `0.0` only when its
distance from zero is at most $\varepsilon_{\mathrm{feas}}$. Valid in-domain
near-endpoint values are not projected by D-15.

D-15 MUST NOT clamp interior fractional values or residuals, alter $x_i$
because $I_i^P$ is near an endpoint, repair coupled or global constraints,
decode causality, or mutual exclusion, or otherwise convert a materially
infeasible vector into another point. Complete post-projection revalidation is
mandatory.

Before an indicator-integral request is locked, prospective D-14
classification MUST remain compatible with unchanged $x_i$: classified
$I_i^P=1$ cannot yield $\lfloor x_i\rfloor=0$, and classified $I_i^P=0$
cannot conceal a positive $x_i$ that violates linkage. The validator MUST
reject such a case rather than promote, demote, or jointly repair the pair.

The SciPy status-0 candidate MUST include a finite reported objective. The
adapter or validator MUST independently recompute it in SciPy's minimization
convention, treat a material mismatch as malformed/inconsistent, and derive
the project maximization value explicitly. The comparison MUST use a documented
scale-aware floating-point allowance distinct from both numerical tolerances.
When projection occurs, the normalized-vector objective MUST also be
recomputed and recorded.

Integrality tolerance and feasibility tolerance are conceptually distinct and MUST NOT be conflated silently.

### 11.3 Integer-plan algebraic invariants

Every successful plan MUST satisfy exactly:

$$
\sum_i(\hat x_i+\hat y_i)\le B_{\max},
$$

$$
\sum_i(\hat I_i^P+\hat y_i)\le S_{\max},
$$

$$
\sum_i(a_i^P\hat I_i^P+c_i^D\hat y_i)
-\sum_{i\in\mathcal Z_t}c_i^Z\hat z_i
\le M_t^{\mathrm{free}}-W_t,
$$

$$
\hat I_i^P=\mathbf1\{\hat x_i>0\},
$$

and

$$
\hat I_i^P+\hat y_i+\hat z_i\le1
\qquad\forall i.
$$

All hatted variables must have their integer domains. Integer plans are not accepted with numerical slack outside their domains.

### 11.4 Request-level semantic invariants

The plan MUST:

- prefill only a prefill-eligible request;
- use $1\le\hat x_i\le\min(P_i^{\mathrm{rem}},C_{\max})$ for a selected prefill;
- decode only a decode-eligible request;
- schedule exactly one token for a decode;
- preempt only a member of $\mathcal Z_t$;
- never schedule and preempt the same request;
- charge the complete $a_i^P$ for every selected unallocated prefill;
- charge zero additional prefill memory only after resident allocation is verified;
- preserve all integral decisions locked before fractional extraction;
- contain exactly one decision record per input request.

### 11.5 Output-boundary invariants

Before `SchedulerOutputs` is returned:

- scheduled IDs are unique;
- preempted IDs are unique;
- ignored IDs are unique;
- the three ID sets are pairwise disjoint;
- every ID is known and currently scheduler-owned;
- every prefill chunk is positive and within its validated bounds;
- every decode chunk is exactly zero;
- no negative chunk is emitted;
- derived token and action counts match the validated plan;
- scheduler-specific counters, if populated, agree with the actual action sets;
- the metadata order follows the validated execution/replay contract.

## 12. Read-only Phase E state mapping

### 12.1 Phase boundary

Phase E converts one coherent LPServe decision-boundary snapshot into the immutable Phase D problem. It MUST be independently testable without invoking an LP solver.

It MUST NOT:

- remove or append a request in any scheduler collection;
- allocate, append, free, or preempt a block table;
- change sequence status or prompt progress;
- construct an output that the engine will replay;
- make a fallback scheduling decision.

### 12.2 Snapshot consistency

The state mapper MUST capture, or validate as a coherent read of:

- scheduler iteration or equivalent snapshot identifier;
- decision time used for arrival filtering and utilities;
- authoritative waiting and resident ownership;
- sequence status and completion state;
- prompt progress;
- logical and physical block-table lengths;
- current free block count;
- pipeline/in-flight information when supported;
- capacity configuration and all selected OPEN policies.

If the framework cannot supply a coherent snapshot under the current execution mode, Phase E must fail explicitly. It must not combine values from incompatible instants.

### 12.3 Complete mathematical-to-LPServe mapping

| Mathematical object | LPServe source or normative construction | Status |
|---|---|---|
| Request ID $i$ | `Sequence.seq_id` | Resolved |
| $U_t$ | Deduplicated arrived, non-finished sequences under the LP scheduler's authoritative ownership | Ownership container design OPEN |
| $\mathcal Z_t$ | Owned resident, allocated, native-status-eligible, safely releasable, validated-recomputation requests | Empty while blockers remain; pipeline predicate OPEN |
| $P_i^{\mathrm{rem}}$ | `get_prompt_len() - get_num_prompt_tokens_processed()` | Resolved |
| Prefill eligibility | Positive remainder plus legal ownership/status/allocation path | Exact status policy must be documented |
| Decode eligibility | Prompt complete, zero remainder, resident/allocated, legal status, not in-flight | Resolved boundary; pipeline predicate OPEN |
| $U_i$ | `min(P_i_remaining, C_max)` | Resolved |
| $B_{\max}$ | Explicit combined prompt/decode token budget | Concrete config binding OPEN |
| $C_{\max}$ | Explicit per-request prefill cap | Concrete config binding OPEN |
| $S_{\max}$ | Explicit scheduled-metadata/action-width cap | Concrete config binding OPEN |
| Resident capacity | `max_num_seqs`, checked against post-plan resident set | Resolved responsibility; resident ledger OPEN |
| $M_t^{\mathrm{free}}$ | `block_manager.get_num_free_gpu_blocks()` | Resolved |
| $W_t$ | Explicit reserve in blocks | OPEN |
| $a_i^P$, resident partial prefill | Zero after resident ownership and block-table allocation are verified | Resolved |
| $a_i^P$, admission/recompute | `len(seq.logical_token_blocks)` | Resolved |
| $c_i^D$ | Exact logical/physical gap or conservative one-block policy | OPEN choice |
| $c_i^Z$ | Current physical block-table length | Resolved for legal candidate |
| $\alpha_i,\beta_i,\gamma_i$ | Explicit utility provider using snapshot data | OPEN policy |

### 12.4 Ownership contract

Every unfinished request MUST have one authoritative scheduling ownership classification. Auxiliary indexes or heaps MAY reference the same object, but they must not create a second owner. The mapper must be able to prove that every included request appears exactly once in the ownership ledger and that every owned, arrived, unfinished request appears in $U_t$.

The new scheduler MUST NOT inherit SLAI's `_active_seq_ids`, `paused_prefills`, and `decode_queue` organization without an explicit design decision and new invariant tests. The architecture audit found that inherited base unfinished-work reporting can omit such auxiliary ownership.

### 12.5 Capacity construction

The mapper MUST keep these meanings separate:

- $B_{\max}$: tokens scheduled now;
- $C_{\max}$: prefill tokens for one request now;
- $S_{\max}$: requests executing now;
- `max_num_seqs`: requests remaining resident after control and admission actions.

One configuration value MAY supply more than one quantity only if that equivalence is an explicit approved policy and the quantities remain separately named and validated.

### 12.6 Stale-plan detection

The Phase D plan MUST carry the Phase E snapshot identifier. Before Phase F commits anything, it must verify that every state-dependent input used for legality and block accounting is still current. At minimum, this includes ownership, status, allocation presence, logical/physical block lengths, prompt progress, completion, free-block state, and any in-flight marker.

The precise mechanism—iteration ID, version counters, lock scope, or another method—is an engineering decision to be made before Phase F. A mismatch is a precommit validation failure, not permission to patch the plan in place.

## 13. Validated Phase F action execution

### 13.1 Preconditions

Phase F may run only when:

- Phase E produced a valid snapshot;
- the LP solver produced an acceptable successful result;
- extraction succeeded;
- mathematical plan validation succeeded;
- the plan's snapshot identity matches the current scheduling boundary;
- all enabled action types have passed their framework-blocker gates.

### 13.2 Complete-plan prevalidation

Before the first mutation, the executor MUST validate the entire plan, not actions one at a time after earlier actions have already committed. It must check:

- every request still exists and has the expected ownership and status;
- prompt/decode/preemption eligibility still holds;
- output ID uniqueness and mutual exclusion;
- prompt chunk bounds;
- post-plan resident capacity;
- preemption recovery and admission/append demands against current block tables;
- exact admission watermark behavior;
- exact append feasibility under the selected commit order;
- availability of all worker-replay semantics required by the output;
- absence of unsupported pipeline or control-only cases.

Successful prevalidation reduces expected failure; it does not make the non-transactional framework atomic.

### 13.3 Commit ordering

The central scheduler and workers must perform logically equivalent operations in the same deterministic order because workers reconstruct block tables by replay rather than receiving central block numbers.

The required high-level order is:

1. apply validated ignored controls, if that separately validated path is enabled;
2. remove validated preemption victims from resident ownership and free their central blocks;
3. apply scheduled actions in the canonical metadata order, allocating or appending as required;
4. finalize scheduler ownership collections;
5. return `SchedulerOutputs` containing the same preempted IDs and scheduled metadata needed for engine/worker replay.

Native replay processes ignored IDs, then preempted IDs, then scheduled metadata. The same ID MUST NOT appear in more than one category.

The exact mutation granularity and post-mutation failure policy remain OPEN. Phase F must not be enabled before that policy is explicit.

### 13.4 Waiting or recomputation prefill admission

For an unallocated prefill with $\hat x_i>0$, Phase F must:

- confirm the request remains waiting, arrived, unfinished, and prompt-incomplete;
- confirm the complete fixed admission charge still passes the native allocation gate;
- reserve a resident slot in the post-plan capacity calculation;
- remove the request from waiting ownership exactly once;
- allocate its complete logical-context block table centrally;
- add it to resident/current ownership;
- emit `SequenceScheduleMetadata(seq_id, prompt_chunk_len=hat_x_i)`.

The worker will allocate the same full logical context while executing only the stated chunk. Admission memory is independent of chunk length.

### 13.5 Resident partial prefill

For a resident prefill, Phase F must:

- confirm allocation remains present and $a_i^P=0$ remains valid;
- confirm positive prompt remainder;
- retain resident ownership;
- avoid a new central admission allocation;
- emit positive-chunk metadata.

The worker's generic append replay should observe no logical/physical gap in the normal case. This expected zero allocation must be verified in integration tests rather than assumed from the scalar LP plan.

### 13.6 Decode

For $\hat y_i=1$, Phase F must:

- confirm prompt completion, resident ownership, allocation, and legal status;
- confirm the selected physical append policy still passes;
- call the central append path in metadata order;
- retain resident ownership;
- emit `SequenceScheduleMetadata(seq_id, prompt_chunk_len=0)`.

The newly sampled token can create the logical/physical gap handled by a later decode action. The current action's $c_i^D$ concerns the gap that exists before this decode.

### 13.7 Recomputation preemption

For $\hat z_i=1$, the intended native action is:

- remove the victim from resident ownership exactly once;
- free its complete central physical block table;
- insert it into waiting ownership through the native preemption path;
- emit its ID in `preempted_seq_ids` and no scheduled metadata for that ID;
- allow engine and workers to replay `reset_for_recompute()` and free their local blocks.

This action MUST remain disabled in integrated execution until all applicable blockers in Section 16 are repaired and tested. Mathematical support for $z_i$ in Phase D does not authorize unsafe runtime preemption.

### 13.8 Do nothing

For a request whose hatted variables are all zero, Phase F makes no change and emits no per-request output entry. A globally empty plan is representable, but its live fallback and liveness semantics are OPEN.

### 13.9 Metadata order and sampler contract

The audited model runner physically constructs prompt inputs before decode inputs. Accordingly, any enabled mixed batch MUST use a canonical prompt-first metadata layout, with deterministic order within the prompt and decode partitions. The exact within-partition ordering is part of the unresolved tie/ordering policy.

This ordering requirement alone does not repair the mixed-sampling-type sampler defect. The executor MUST reject or keep unsupported any case for which sampler output identity and length cannot be validated end to end.

### 13.10 Postconditions

After a successful commit and before returning output, the central scheduler MUST be able to verify:

- every unfinished request still has exactly one authoritative owner;
- every resident request expected to be allocated has a central block table;
- no preempted request remains resident;
- no request was duplicated or lost;
- free-block change matches the ordered central operations;
- `SchedulerOutputs` exactly represents those operations;
- scheduled metadata counts match $\sum_i(\hat I_i^P+\hat y_i)$;
- scheduled token counts match $\sum_i(\hat x_i+\hat y_i)$.

Central/worker block equality cannot be proven from `SchedulerOutputs` alone. Dedicated replay tests are required.

## 14. Failure contracts

### 14.1 Layer-local behavior that is resolved

| Failure location | Required behavior before live-policy selection |
|---|---|
| Invalid Phase D input | Return explicit failure; do not call solver |
| Solver infeasible, unbounded, numerical error, limit, malformed output, or exception | Return explicit non-success; do not extract |
| Invalid relaxed solution | Return explicit non-success; do not extract |
| Extraction cannot restore planning feasibility | Return `ExtractionFailure`; no plan |
| Integer plan violates an invariant | Return explicit validation failure; do not invoke Phase F |
| Phase E inconsistent snapshot | Return explicit mapping failure; no mutation |
| Phase F stale state or failed prevalidation | Return explicit precommit failure; perform zero mutations |

No layer may convert these outcomes into invented relaxed values, an all-zero plan, a baseline-policy action, or a partially accepted plan.

### 14.2 Top-level solver and extraction failure policy

What the live scheduler should do after a Phase D or Phase E failure is OPEN. Possibilities such as returning an empty output, retrying, invoking another policy, or failing the run have materially different correctness and experimental implications. None is selected here.

Consequently, integration into a live scheduler MUST NOT be declared complete until one policy is approved, named, instrumented, and tested. Phase D's explicit failure result is not itself a fallback policy.

### 14.3 Precommit physical failure

If fresh physical prevalidation fails, Phase F must perform no mutation. What the scheduling loop does next remains governed by the unresolved top-level failure policy.

### 14.4 Failure after mutation begins

The audited framework has no transaction, reservation object, undo log, or rollback spanning scheduler collections, the central allocator, engine state, and worker allocators. Allocation can also fail after partially popping blocks from its free list.

Rollback, fail-stop, or another post-mutation failure contract is OPEN. The executor MUST NOT claim atomicity, recovery, or safe continuation until a chosen contract is implemented and tested. Catching an exception and continuing is not an acceptable implicit policy because central and worker state may have diverged.

### 14.5 Observability

Every solver attempt SHOULD retain, without fabricating actions:

- normalized outcome and, for non-success, phase, failure stage, and reason;
- raw solver status and success value;
- diagnostic message as non-contractual text;
- whether a primal vector was present and its shape;
- solver identity, requested method, explicit options, deliberate absence of
  project-level limits, and exact SciPy version;
- bundled HiGHS version when exposed through a stable public mechanism;
- available `nit` and `crossover_nit` information;
- reported objective when finite;
- scheduler iteration/problem/snapshot identifier;
- whether relaxed-solution validation was entered and passed;
- whether extraction was entered;
- observed $|U_{\mathrm{frac}}|$ when classification was reached;
- numerical-policy identifier, configured $\varepsilon_{\mathrm{int}}$, and
  aggregate zero/one/fractional indicator counts;
- configured $\varepsilon_{\mathrm{feas}}$, maximum raw and normalized
  violation by constraint family, and the worst constraint/request ID;
- whether projection occurred and its changed-coordinate count;
- reported, raw-recomputed, and normalized objectives where applicable;
- violated invariant or exhausted candidate set;
- whether mutation had begun;
- relevant request IDs and capacity summaries;
- selected policy identifiers and tolerances.

Targeted debug or trace output SHOULD permit inspection of each raw indicator,
its classified state, and its distance to the nearest endpoint without
requiring routine logs to contain every indicator value.
Full primal vectors and per-coordinate projection details MAY likewise remain
debug artifacts; routine diagnostics MUST retain the validation stage and
rejection reason.

For a limit outcome, diagnostics MUST NOT infer time limit versus iteration
limit from human-readable message text alone.

Experiment summaries must distinguish policy decisions from solver, extraction, validation, and execution failures.

## 15. Testing and phase acceptance requirements

No GPU experiment substitutes for mathematical or state-transition tests. Each phase must satisfy its own gate before the next mutation boundary is enabled.

### 15.1 Phase D problem and solver tests

At minimum:

- empty request set;
- prefill-only, decode-only, preemption-only, and mixed synthetic problems;
- continuous $x_i$ rather than integer $x_i$;
- exact enforcement of $I_i^P\le x_i\le U_iI_i^P$;
- rejection of $I_i^P=1,x_i<1$;
- decode causality;
- mutual exclusion;
- exclusion of $z_i$ outside $\mathcal Z_t$;
- token-, action-, and memory-budget saturation;
- the adapter requests `method="highs-ds"` and explicitly passes
  `presolve=True`;
- the default Phase D configuration passes neither `time_limit` nor `maxiter`;
- a real nondegenerate SciPy/HiGHS LP obtains its known optimal objective and
  vector;
- correct conversion of the maximization objective to SciPy's minimization convention;
- feasible, infeasible, and unbounded cases where constructible;
- controlled or mocked limit and numerical-failure results;
- a deliberately scoped finite-limit adapter test remains a D-12 non-success
  and never reaches extraction;
- unknown and internally inconsistent solver status results;
- missing, wrong-sized, and non-finite relaxed solution vectors;
- D-15 vector-order/request-association checks and raw-vector preservation;
- D-15 simple-bound projection at half and exactly
  $\varepsilon_{\mathrm{feas}}$, plus `nextafter` cases just outside it;
- rejection beyond $\varepsilon_{\mathrm{feas}}$ and preservation of valid
  in-domain near-endpoint values;
- independent raw and post-projection residual checks for every global and
  request-local constraint, including rejection when projection causes a
  constraint violation;
- absence of coupled or multi-variable repair, including D-14 handoff
  compatibility for $(I_i^P,x_i)$;
- objective recomputation, maximization/minimization sign conversion, and
  malformed objective mismatch;
- preservation of raw and normalized vectors and fresh residual-capacity
  computation from the hatted plan;
- solver exception and interruption outcomes;
- status `0` with `success=True` and a structurally valid vector reaches the
  independent relaxed-solution validator;
- a status-0 candidate reaches extraction only after that validator succeeds;
- status `0` with a missing, malformed, wrong-sized, or otherwise unusable
  vector does not reach extraction;
- status `0` with `success != True` is classified as invalid/malformed;
- status `1` is rejected whether its vector is absent or feasible-looking;
- status `2` and status `3` are rejected even if a vector is unexpectedly
  present;
- status `4` is rejected even if a finite or feasible-looking vector is
  present;
- unknown status, exception, and interruption never reach extraction;
- differing human-readable messages do not change classification;
- an independently feasible but lower-objective status-1 fixture is still
  rejected;
- spy/call-order tests prove that no non-success reaches extraction or
  fabricates an all-zero plan;
- deterministic request-to-variable reconstruction;
- repeated solves of a uniquely optimal problem in the same environment return
  the expected objective and vector;
- degenerate multiple-optimum tests require a validated optimal objective and
  feasibility rather than one particular primal vector;
- `crossover_nit`, when present, is zero on the dual-simplex path but is treated
  only as diagnostic evidence;
- no test treats `highs-ds`, status `0`, or zero crossover iterations as proof
  of a basic/extreme-point property or a bound on $|U_{\mathrm{frac}}|$;
- recording of solver identity, exact SciPy version, requested method, explicit
  options, absent project-level limits, status, success, objective, `nit`,
  available `crossover_nit`, validator/extraction outcome, observed
  $|U_{\mathrm{frac}}|$, and environment provenance;
- NaN, infinity, wrong dimensions, duplicate IDs, and invalid coefficient units;
- no fabricated plan on any solver non-success.

### 15.2 Extraction tests

At minimum:

- all-integral solution;
- flooring of integral-request $x_i$;
- preservation of locked integral decisions;
- exact D-14 endpoint classification with
  $\varepsilon_{\mathrm{int}}=10^{-6}$ for `0`,
  $\varepsilon_{\mathrm{int}}/2$, the adjacent representable value immediately
  below $\varepsilon_{\mathrm{int}}$, exactly
  $\varepsilon_{\mathrm{int}}$, the adjacent representable value immediately
  above $\varepsilon_{\mathrm{int}}$, `0.001`, `0.5`, `0.999`, the adjacent
  representable value immediately below $1-\varepsilon_{\mathrm{int}}$,
  exactly $1-\varepsilon_{\mathrm{int}}$, the adjacent representable value
  immediately above $1-\varepsilon_{\mathrm{int}}$, `0.9999994`, and `1`;
- use of an adjacent-representable-value operation such as `nextafter` for
  immediate boundary neighbors rather than decimal-literal approximations;
- exact `0.0` and `1.0` classified representations with fractional values left
  unchanged, while preserving the original relaxed indicators;
- all three binary-classified indicators place a request in
  $U_{\mathrm{int}}$, while any one fractional indicator places it in
  $U_{\mathrm{frac}}$;
- proof that $\tilde x_i$ does not affect the integral/fractional partition;
- deterministic repeated classification and preservation of locked integral
  decisions;
- observability of $|U_{\mathrm{frac}}|$ without using it as an acceptance
  condition;
- proof that D-14 does not silently repair a near-one $\tilde I_i^P$ paired
  with a sub-one $\tilde x_i$;
- separate D-15 tests, not D-14 classifications, for out-of-domain values,
  NaN, infinity, constraint residuals, and any permitted normalization;
- dominant legal preemption;
- exclusion of non-$\mathcal Z_t$ preemption;
- zero-recovery exclusion;
- one safety preemption;
- multiple safety preemptions;
- failure after exhausting positive-recovery candidates;
- decode admission under every residual capacity;
- prefill truncation by residual token capacity;
- full fixed-charge acceptance when $a_i^P$ fits;
- rejection when the full charge does not fit;
- proof by test that reducing the chunk does not reduce $a_i^P$;
- zero charge for resident partial prefill;
- full-context charge for admission/recomputation;
- canonicalization of $\hat I_i^P$;
- D-16 tuple-key validation/rejection, permutation-invariant layout/output,
  exact action and safety-preemption ties, and repeated key-sorted actions;
- more than three fractional requests without rejection merely because the
  observed count exceeds the structural expectation;
- exact final plan validation.

### 15.3 Phase E mapping tests

Use real or faithful LPServe `Sequence` and block-manager state to test:

- exact prompt remainder;
- future-arrival exclusion;
- finished-request exclusion;
- duplicate and conflicting ownership rejection;
- complete inclusion of owned unfinished requests;
- prefill and decode eligibility;
- construction of an empty operational $\mathcal Z_t$ while blockers remain;
- legal preemption predicate after blockers are resolved;
- $M_t^{\mathrm{free}}$ from the allocator;
- zero resident-prefill charge;
- full-context admission and recomputation charge;
- exact logical/physical decode gap;
- conservative decode policy when selected;
- exact physical recovery $c_i^Z$;
- separation of $B_{\max},C_{\max},S_{\max}$, and resident capacity;
- stale snapshot detection;
- proof that mapping performs no mutation.

### 15.4 Phase F execution tests

Before GPU validation, synthetic scheduler-state tests must cover:

- waiting prefill admission;
- resident partial prefill;
- decode with zero and one exact marginal block demand;
- resident-capacity release by preemption followed by admission;
- do nothing;
- queue and ownership postconditions;
- allocation, append, free, and free-block deltas;
- metadata chunk sign and bounds;
- unique/disjoint output IDs;
- central replay order matching worker replay order;
- prompt-first mixed-batch metadata order;
- rejected stale or physically infeasible plan causing zero mutation;
- every enabled post-mutation failure policy;
- preemption/recomputation only after the Section 16 fixes are present;
- control-only outputs only after their engine semantics are fixed;
- central/worker block-table equality in deterministic replay tests.

### 15.5 Integrated validation sequence

After Phases D–F pass their CPU tests:

1. run syntax, import, and static checks;
2. run focused mathematical tests;
3. run state-mapping and executor tests;
4. run one tiny single-GPU smoke test with the smallest supported action set;
5. inspect per-iteration LP inputs, relaxed decisions, extracted decisions, queue transitions, and block deltas;
6. add preemption only after its blockers and tests are cleared;
7. compare against tiny same-framework baselines;
8. begin broader timing or performance experiments only after correctness evidence is retained.

Aggregate throughput or latency cannot establish scheduler correctness.

### 15.6 Acceptance evidence

Each phase handoff must record:

- repository commit;
- environment and dependency changes;
- tests run and their observed output;
- behaviors actually verified;
- unresolved failures and skipped tests;
- newly selected research policies;
- next permitted phase.

The absence of a failure in a GPU run is not evidence that an unexercised invariant holds.

## 16. Deferred blockers and unsupported scope

### 16.1 Recomputation generation-limit semantics — BLOCKER

`reset_for_recompute()` moves existing generated token IDs into the prompt context and clears `output_token_ids`. The current length cap uses the cleared list rather than the cumulative output-token count. A preempted request can therefore generate beyond its requested `max_tokens` across restarts.

The model's causal token context is preserved, but user-visible generation semantics are not. Runtime recomputation preemption MUST remain disabled until the generation limit is based on a correct cumulative notion and targeted tests cover one and multiple restarts.

### 16.2 Recomputation `RequestOutput` semantics — BLOCKER

After recomputation, the audited `RequestOutput` can pair:

- the original prompt string with expanded prompt token IDs; and
- cumulative output text with only post-restart output token IDs.

The intended external contract must be selected, repaired, and tested before preemption results are used in correctness or performance claims.

### 16.3 Control-only scheduler outputs — BLOCKER

A preempt-only or ignore-only output has no scheduled metadata. The single-stage engine drops such an output before replay, while the pipeline path can enqueue control state and then wait for model output from a batch never sent to workers.

The LP can legitimately produce a preempt-only plan. Phase F MUST therefore not emit preempt-only output until engine handling of control-only output is repaired and tested. The executor MUST NOT force an unrelated scheduled action merely to avoid this defect.

### 16.4 Mixed-batch and sampler association — BLOCKER for affected cases

The model runner physically packs prompts before decodes, whereas existing scheduler metadata may be decode-first or interleaved. Sampling-type grouping can also select incorrect tensor rows, and completion uses positional `zip` without checking result IDs or lengths.

The LP executor requires prompt-first metadata, but correctness for mixed sampling types additionally requires identity- and length-validated sampler association. Unsupported cases must be rejected or excluded from claims until the underlying path is fixed and tested.

### 16.5 Non-transactional mutation — BLOCKER for recovery claims

Scheduler queues and central blocks mutate before forward execution, workers replay later, and no cross-layer rollback exists. Prevalidation is mandatory, but no implementation may claim atomic commit or recoverable failure without additional design and evidence.

### 16.6 Pipeline-parallel support — DEFERRED

Pipeline execution permits multiple microbatches in flight. The audited framework anticipates preemption of a request while an older batch is executing, but exposes no safe physical-release predicate, per-batch KV version, or exact completion association sufficient for this design.

This specification therefore does not authorize pipeline-parallel LP execution. In particular, it does not define:

- which in-flight requests belong to $\mathcal Z_t$;
- which actions must be frozen across outstanding batches;
- when freed blocks are safe to reuse;
- how a plan is associated with a later completion callback;
- how failure affects `num_running_batches` and worker queues.

Initial correctness validation is limited to a single pipeline stage. Supporting more stages requires a separate design amendment and new architecture tests; changing only `num_pipeline_stages` is insufficient.

## 17. Decision register

### 17.1 RESOLVED — D-11 LP solver

**Selected rule.** The first correctness-focused Phase D implementation uses
`scipy.optimize.linprog` with the HiGHS solver family behind a project-owned
solver adapter. The adapter does not expose raw SciPy result objects or state
across the Phase D interface.

**Rationale.** Phase D requires only continuous LP solving. `linprog` maps
naturally to the mathematical objective, inequalities, and bounds; returns the
complete relaxed primal vector; and provides HiGHS-backed outcomes that can be
normalized into the project-owned solver-result contract. Independent project
validators remain responsible for dimensions, finiteness, bounds, constraints,
and request association. The interface is CPU-testable, keeps the mathematical
layer framework-light, and has substantially less solver-specific integration
surface than direct `highspy`. The adapter boundary preserves replaceability;
basis inspection and finer HiGHS control are not required by D-11 and remain
relevant to D-13.

**Affected interfaces.** Section 9.3 defines objective-sign conversion,
deterministic variable/request association, project-owned results, diagnostic
fields, status normalization, and the independent validation gate. D-12 selects
optimal-only admission to that gate. The partially resolved D-13 selects the
Phase D reference method, presolve setting, and no-limit policy; its
performance, advanced-control, and basis-certification remainder stays OPEN.

**Required tests.** The D-11-specific adapter tests are listed in Section 15.1.
They include a real optimal solve, objective-sign conversion, status and failure
normalization, malformed-vector rejection, exception/interruption handling,
request-to-variable reconstruction, repeated-solve behavior, and proof that no
non-success reaches extraction.

**Reproducibility metadata and dependency boundary.** Solver identity, SciPy
version, selected method/options, raw status, success flag, message, and
available iteration information SHOULD be retained for diagnostics and recorded
with experiment metadata where relevant. SciPy MUST become an explicit,
reproducible dependency before Phase D execution, and its exact version MUST be
verified against the validated Unity environment. The repository's existing
SciPy import or any transitive installation is not sufficient dependency
management. D-11 does not approve an exact SciPy version pin.

### 17.2 RESOLVED — D-12 Acceptable solver statuses

**Selected rule.** Phase D uses optimal-only admission. Only a normalized
optimal candidate corresponding to SciPy `status == 0` may proceed to the
independent relaxed-solution validator. Every other normalized outcome is a
Phase D non-success and MUST NOT proceed to extraction: solver limit or early
termination, infeasible, unbounded, numerical or solver difficulty,
invalid/malformed or inconsistent result, exception, and interruption. A
non-optimal returned vector is not an extractable relaxed solution even if it
appears feasible or could independently be shown primal-feasible. A status-0
result remains only an optimal candidate until the independent validator
accepts it.

**Rationale.** Primal Heuristic 1 solves the LP relaxation before deterministic
integer extraction. Allowing a merely feasible non-optimal point would define a
different heuristic, weaken the intended objective and LP-relaxation-bound
interpretation, and weaken the premise behind the almost-integral motivation,
which concerns an optimal LP solution and potentially an optimal basic or
extreme-point solution. Rejecting partial and non-optimal outcomes improves
reproducibility and debugging during correctness-focused development. Solver-
limit optimization may be reconsidered later only as an explicit design
change if performance requires it. Status `0` does not guarantee a basic or
extreme-point solution; basis inspection and certification remain in the OPEN
portion of D-13.

**Affected interfaces.** Section 9.3 defines the admission boundary and the
required consistency rules. A status-0 result with `success != True`, or without
a usable primal vector, is invalid/malformed. Human-readable message text does
not classify outcomes. No vector from status `1`, `2`, `3`, `4`, an unknown
status, exception, or interruption may cross the adapter boundary as approved
relaxed-solution state. Rejected results may retain vector-presence and shape
diagnostics. D-15 continues to own feasibility tolerances and clamping; D-17
continues to own the live-scheduler response after non-success.

**Required tests.** Section 15.1 requires focused admission and call-order
tests for valid and malformed status-0 results; status `1` with absent and
feasible-looking vectors; statuses `2`, `3`, and `4` even when vectors are
present; unknown status, exception, and interruption; message independence; an
independently feasible but lower-objective status-1 fixture; and proof that
non-success neither reaches extraction nor fabricates an all-zero plan. The
numerical details of relaxed-solution validation remain under D-15.

**Observability and reproducibility.** Section 14.5 requires the normalized
outcome, raw status and success value, non-contractual diagnostic message,
vector presence and shape, solver method/options, SciPy version, available
iteration/crossover information, finite reported objective, problem/snapshot
identifier, validation and extraction entry, and failure stage/reason. Limit
subtypes MUST NOT be inferred from message text alone. These fields preserve
evidence for rejected results without making their vectors extractable.

### 17.3 PARTIALLY RESOLVED — D-13 Phase D reference solver configuration

**Selected Phase D subset.** The first correctness-focused implementation MUST
call `scipy.optimize.linprog` with `method="highs-ds"`, explicitly pass
`presolve=True`, and omit both `time_limit` and `maxiter`. It MUST NOT explicitly
configure crossover or pass undocumented thread-count, parallelism, or
random-seed options through `linprog`. All other solver tuning uses the
documented defaults of the exact pinned SciPy version unless a later approved
decision changes them. D-13 does not select the SciPy version; D-11's dependency
and environment-verification requirement remains controlling.

**Rationale.** `highs` permits automatic algorithm selection and adds avoidable
variability to the reference correctness implementation. `highs-ds` explicitly
selects HiGHS dual revised simplex and provides a simpler, well-defined solver
path. A simplex path makes a basic or vertex optimum more reasonable to expect,
which is useful when investigating the almost-integral motivation, but the
public SciPy result does not give the project a basis certificate. This choice
therefore proves neither basicness nor a fractional-request bound. Extraction
is defined for an arbitrary fractional set, so basicness is not required for
correctness. Because D-12 rejects every limit-terminated outcome, finite limits
provide no benefit to the initial correctness configuration. Explicit
`presolve=True` avoids silently depending on that default, while deferring
performance tuning until correctness evidence exists.

**Correctness and interface consequences.** Phase D correctness requires a
D-12-admissible status-0 optimal candidate, successful independent
relaxed-solution validation, deterministic extraction, and successful
integer-plan validation. It does not require proof of a basic/extreme-point
solution. An observed $|U_{\mathrm{frac}}|>3$ MUST remain supported and MUST NOT
cause rejection merely because it exceeds the structural expectation.

**Reproducibility contract.** With identical ordered LP data, `highs-ds`,
explicit presolve, no project-imposed solver limits, identical numerical and
extraction policies, and the same pinned software environment, the
implementation expects stable solver status and objective behavior. It does
not require bitwise or cross-platform deterministic solver output, or identical
primal-vector selection among multiple optimal solutions across different
solver versions, builds, or platforms.

**Required tests and observability.** Sections 15.1 and 15.2 require tests of
the exact method/options call, absent default limits, a known nondegenerate
optimum, repeated uniquely optimal solves, degenerate multiple-optimum behavior,
diagnostic-only crossover counts, scoped finite-limit rejection, arbitrary
fractional-set handling, and the absence of false basicness or fractional-bound
claims. Section 14.5 records the requested method, explicit options, absent
project limits, exact SciPy version, public bundled-HiGHS version when
available, status/success/objective, iteration and crossover diagnostics,
$|U_{\mathrm{frac}}|$, validator/extraction outcome, and comparison-environment
provenance.

**OPEN D-13 remainder.** This partial resolution does not select finite time or
iteration limits for live/performance use; possible future use of `highs` or
`highs-ipm`; simplex edge-weight or other solver tuning; future IPM/crossover
configuration; basis inspection or certification; warm starts or basis reuse;
threading, parallelism, or random-seed control if later required; cross-version
behavior for degenerate optima; or any assertion connecting solver output to a
bound on $|U_{\mathrm{frac}}|$. None of this remainder blocks the initial Phase
D correctness implementation.

**D-23 boundary.** Proof of an at-most-three or other fractional-request bound,
request-local extreme-point characterization, approximation guarantees relative
to the LP or ILP, and theoretical inference from observed near-integrality
remain under D-23. Selecting `highs-ds` establishes none of them.

### 17.4 RESOLVED — D-14 Integrality tolerance

**Selected rule.** Phase D uses the absolute integrality tolerance
$\varepsilon_{\mathrm{int}}=10^{-6}$ only for the relaxed action indicators
$\tilde y_i$, $\tilde I_i^P$, and $\tilde z_i$. An already validated in-domain
indicator is zero on $[0,\varepsilon_{\mathrm{int}}]$, one on
$[1-\varepsilon_{\mathrm{int}},1]$, and fractional between those closed
endpoint intervals. The intervals are disjoint, so no tie rule is needed.
A request is indicator-integral only when all three indicators classify as
binary; $\tilde x_i$ does not determine the partition and is not classified or
rounded by D-14.

**Rationale.** Binary actions have a fixed $[0,1]$ scale, so absolute tolerance
is sufficient and relative tolerance would add no useful semantics. `1e-6`
absorbs harmless endpoint-level numerical noise while remaining far below
substantively fractional values such as `0.001`, `0.999`, and `0.5`. A tighter
reference value such as `1e-8` would unnecessarily turn near-endpoint noise into
fractional work, whereas `1e-5` or `1e-4` would classify increasingly meaningful
fractional decisions as integral. The value is not chosen to make
$|U_{\mathrm{frac}}|$ approach an expected theoretical count, and no solver
feasibility tolerance is part of this rule.

**Affected interfaces and D-15 boundary.** Classification runs only after the
D-15 validator has produced a finite value in $[0,1]$. Zero and one become
exact `0.0` and `1.0` in the classified extraction representation; fractional
values retain their validated values, and the raw tilded solution remains
preserved separately. D-14 MUST NOT classify, accept, or clamp an out-of-domain
value and MUST NOT modify $\tilde x_i$ or repair coupled linkage constraints.
D-15 defines feasibility residuals, narrow simple-bound projection, rejection
of non-finite or materially infeasible values, and post-projection validation;
it permits no coupled repair. D-16 is unaffected.

**Required tests.** Section 15.2 specifies inclusive endpoint and adjacent-
representable-value cases for $\varepsilon_{\mathrm{int}}=10^{-6}$; exact
classified representation and raw-value preservation; request partitioning
based only on the three indicators; deterministic repeated classification;
locked-decision preservation; fractional-count observability without
acceptance; absence of coupled $I_i^P/x_i$ repair; and the explicit handoff of
out-of-domain, non-finite, residual, and normalization cases to D-15.

**Observability and reproducibility.** Sections 10.8 and 14.5 require the
numerical-policy identifier, configured $\varepsilon_{\mathrm{int}}$, resulting
$|U_{\mathrm{frac}}|$, and aggregate zero/one/fractional indicator counts.
Targeted debug or trace output should expose the raw indicator, classification,
and distance to the nearest endpoint. Routine logs need not contain every raw
indicator. Recording the tolerance and raw boundary values makes solver-version
differences near the threshold diagnosable without changing the policy after
observing the fractional count.

### 17.5 RESOLVED — D-15 Feasibility tolerance and clamping

**Selected rule and rationale.** The independent relaxed-solution validator
uses the absolute, project-owned tolerance
$\varepsilon_{\mathrm{feas}}=10^{-7}$ in declared LP units. Only tiny
violations of explicit single-variable bounds may be projected to their bound;
all residuals are checked before and after projection. This tolerates endpoint
noise without allowing D-14 or a broad repair to manufacture feasibility.

**Affected interfaces.** Section 11.2 defines the validator, permitted
projection, prohibited repairs, D-14 coupled-variable handoff, and objective
consistency. D-14 and extraction receive only the normalized validated vector,
while the raw vector remains diagnostic. Section 10.3 requires residual
capacities to be recomputed from the hatted plan. D-04, the OPEN D-13
remainder, D-17, physical feasibility and rollback, and D-23 are
unchanged.

**Required tests and observability.** Sections 15.1 and 14.5 cover structural
and non-finite failures, tolerance boundaries, raw and normalized residuals,
no coupled repair, objective consistency, vector preservation, projection
counts, worst violations, and validation failure provenance. Recording the
numerical-policy identifier and both tolerances makes the policy reproducible
without treating solver-provided residuals as authoritative.

### 17.6 RESOLVED — D-16 Tie-breaking and request iteration order

**Selected rule and rationale.** Phase D sorts immutable explicit `order_key`s
ascending; mutable collection order is not a reproducibility contract. The key
controls vector layout and deterministic extraction choices, not LP feasibility
or objective.

**Affected interfaces, tests, and boundaries.** Sections 9–10 define key
validation, canonical output order, and exact-value ties: `decode > prefill >
preempt`, with smallest-key safety-preemption ties. Section 15.2 tests these
rules. Phase E owns live-key derivation; D-21 owns Phase F execution/replay and
`SchedulerOutputs` metadata order; D-17 is unchanged.

### 17.7 OPEN decisions

Every row below is a required explicit decision. No listed candidate is a default.

| ID | Decision | Candidate space stated by sources | Must be resolved by |
|---|---|---|---|
| D-01 | Decode utility $\alpha_i(t)$ | Responsiveness, age, fairness, SLO urgency, or another documented objective | Before end-to-end policy behavior is evaluated |
| D-02 | Prefill utility $\beta_i(t)$ | Progress, age, fairness, SLO urgency, or another documented objective | Before end-to-end policy behavior is evaluated |
| D-03 | Preemption penalty $\gamma_i(t)$ | Explicit restart/recomputation penalty policy | Before runtime preemption is enabled |
| D-04 | Utility domains and scaling | No canonical normalization in sources | Before solver tests using final utilities |
| D-05 | $B_{\max}$ binding | Explicit combined token budget; Sarathi/SLAI-like fields are candidates | Before Phase E completion |
| D-06 | $C_{\max}$ binding | Fixed chunk, dynamic chunk, or explicit new configuration | Before Phase E completion |
| D-07 | $S_{\max}$ binding | Explicit next-forward action limit | Before Phase E completion |
| D-08 | Authoritative ownership/resident ledger | New scheduler-owned representation; must not be inherited accidentally | Before Phase E completion |
| D-09 | Decode planning charge $c_i^D$ | Exact gap or conservative one-block charge | Before final Phase D/Phase E interface freeze |
| D-10 | Memory reserve $W_t$ | Zero, fixed, or state-dependent; not silently the watermark | Before live LP solves |
| D-13 | Remaining solver performance, control, and basis policy | Finite live/performance limits; possible `highs`/`highs-ipm`; edge-weight and other tuning; IPM/crossover; basis certification; warm starts/basis reuse; threading/parallelism/random seed; cross-version degenerate-optimum behavior; any solver-output-to-fractional-bound assertion | Does not block initial Phase D; before the relevant performance, control, or structural claim |
| D-17 | Solver/extraction/mapping failure response | Empty, retry, alternate policy, fail-stop, or another explicit policy | Before live scheduler integration |
| D-18 | Empty-plan liveness | No policy specified | Before live scheduler integration |
| D-19 | Stale-snapshot mechanism | Locking, versions, iteration identity, or another verified mechanism | Before Phase F |
| D-20 | Post-mutation failure contract | Rollback, fail-stop, or another explicit mechanism | Before Phase F is enabled |
| D-21 | Canonical within-prefill and within-decode metadata order | Prompt-first partition required; internal ordering unspecified | Before mixed-batch Phase F tests |
| D-22 | Recomputation output contract | Original prompt/generated suffix versus expanded-context representation | Before preemption is enabled |
| D-23 | Approximation guarantee | No ratio established | Before any theoretical quality claim |
| D-24 | Pipeline-parallel policy | Deferred; safe eligibility and completion association unknown | Before any pipeline support claim |

The decision record for an item MUST include the selected rule, rationale, affected interfaces, tests, and experiment metadata field where relevant.

## 18. Traceability, discrepancies, and change control

### 18.1 Source-to-requirement traceability

| Design topic | Primary source | Architecture dependency |
|---|---|---|
| Variables, objective, and constraints | `docs/math/main-llm-serving.tex`, Myopic ILP | None beyond coefficient mapping |
| Continuous $x_i$ | `docs/math/main-llm-serving.tex`, LP relaxation | `prompt_chunk_len` is the executable integer representation |
| Legal preemption support | `docs/math/main-llm-serving.tex`; Research Context Sections 4–8 | Architecture Sections 7, 15, 17, 20–23 |
| Fixed prefill charge | `docs/math/main-llm-serving.tex`; Research Context Sections 5 and 10 | Architecture Sections 3, 14, 15, 23 |
| Extraction | `docs/math/main-llm-serving.tex`, LP algorithms; Research Context Section 8 | Legal-preemption and physical-validation boundaries |
| Almost-integral caution | `docs/math/main-llm-serving.tex`; Research Context Section 9 | No solver exists in audited code |
| Phase D/E/F separation | Research Context Sections 12, 14, 15, and 19 | Architecture Sections 16, 22, 27 |
| Request mapping | Research Context Sections 4 and 13 | Architecture Sections 6–9 and 23 |
| Native action execution | Research Context Sections 11–15 | Architecture Sections 11–17 |
| Failure and rollback boundary | Research Context Section 14 | Architecture Sections 14, 16, 21, 22, 26 |
| Validation sequence | Research Context Sections 15 and 19 | Architecture Section 24 |
| Historical baseline | `project_status.md` | Phase B observations summarized in architecture metrics discussion |

### 18.2 Recorded source discrepancies

#### Historical project phase

When this design was originally written, `project_status.md` still stated that Phase B was complete and Phase C was next even though the completed `lpserve_scheduler_architecture.md` already existed as the Phase C deliverable. During the current local Phase C-closure work, `project_status.md` was updated to record **Phase C closure / Phase D preparation**. That update is local and uncommitted at the time of this provenance cleanup. The earlier lag is retained here only as historical provenance: status chronology and handoffs do not override the mathematical source, the commit-pinned architecture evidence, or this normative design contract.

#### Exact mathematical integrality versus floating-point classification

The `docs/math/main-llm-serving.tex` pseudocode uses exact membership in
(\{0,1\}), while the research context requires an explicit numerical
tolerance. D-14 resolves the software classification policy at the absolute
tolerance $\varepsilon_{\mathrm{int}}=10^{-6}$ while preserving exact
mathematical domains. D-15 separately fixes relaxed-solution validation at the
absolute tolerance $\varepsilon_{\mathrm{feas}}=10^{-7}$ with only narrow
simple-bound projection. Neither tolerance modifies the LP.

#### Prevalidation versus atomicity

The research context requires complete validation before mutation. The architecture audit establishes that LPServe has no transactional multi-layer commit. These facts are compatible only if prevalidation is understood as a required risk-reduction boundary, not proof of atomicity or rollback.

#### Complete mapping versus unavailable framework predicates

The sources provide exact mappings for prompt remainder, free blocks, prefill admission charge, and preemption recovery. They do not provide a canonical ownership ledger, pipeline-safe release predicate, final capacity bindings, or decode planning policy. Those cells are explicitly OPEN rather than filled with invented mappings.

### 18.3 Baseline provenance

The Phase A/B historical sections of `project_status.md` record a working single-A16 environment and two controlled runs each of LPServe's `sarathi`, `slai_scheduler`, and `vllm` providers. Those runs verified basic framework execution and reproducibility for a tiny workload. They did not validate this LP design, preemption, pipeline execution, semantic generation correctness, or performance.

### 18.4 Rules for implementation changes

An implementation change conforms to this document only if it:

- preserves the mathematical formulation or records an approved mathematical amendment;
- keeps OPEN decisions explicit;
- respects Phase D/E/F mutation boundaries;
- adds or updates the tests implied by the changed contract;
- records observed verification rather than anticipated success;
- updates the traceability and decision register when a decision is resolved;
- re-audits revision-sensitive architecture assumptions after relevant repository changes.

Changes that add a global coupling constraint, alter fixed-charge prefill accounting, broaden $\mathcal Z_t$, permit a new solver status, change extraction ordering, or enable a blocked execution mode require an explicit design revision. They must not enter as incidental implementation details.

### 18.5 Immediate implementation sequence

The next permitted work is:

1. resolve only the decisions required to define Phase D's numerical and solver interface;
2. implement and validate the pure LP and extraction layer;
3. resolve the ownership, capacity-binding, and decode-memory decisions required for Phase E;
4. implement a read-only mapper and prove it does not mutate LPServe;
5. repair or explicitly gate the framework blockers relevant to the Phase F action set;
6. select the failure and commit contracts;
7. implement and validate the native executor;
8. perform integrated single-stage correctness validation;
9. measure performance only after the correctness gates pass.

---

This specification is intentionally incomplete only where the source material leaves a genuine research choice or framework blocker. Such incompleteness is represented by explicit OPEN decisions and BLOCKER gates; it must not be replaced by undocumented defaults.
