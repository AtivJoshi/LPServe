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

1. `main.tex` for the mathematical scheduling formulation;
2. `LP Scheduler Research Context.md` for implementation principles, validation requirements, and phase boundaries;
3. `lpserve_scheduler_architecture.md` for verified behavior at the audited LPServe revision;
4. `project_status.md` for historical Phase A and Phase B state;
5. the current LPServe repository when a code fact is absent, ambiguous, or suspected to have changed.

If two sources appear inconsistent, an implementation must stop and record the discrepancy. It must not silently select whichever interpretation is easier to implement.

### 1.3 Normative language

The terms **MUST**, **MUST NOT**, **SHOULD**, and **MAY** express requirements in decreasing order of strength. An item labeled **OPEN** is not a default. Implementation code must not decide an OPEN item implicitly.

The following evidence labels are used:

- **Resolved mathematical requirement:** fixed by `main.tex`.
- **Resolved architecture fact:** verified by the Phase C audit at the pinned commit.
- **Normative design requirement:** imposed here to preserve the mathematical and framework contracts.
- **OPEN research decision:** a policy choice that must remain externally visible.
- **BLOCKER:** a known condition that prevents a correctness claim for the affected feature.

### 1.4 In scope

The first implementation includes:

- the myopic ILP from `main.tex`;
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

This document does not select utility functions, watermarks, numerical tolerances, a solver, tie-breaking, fallback behavior, rollback behavior, or pipeline support.

## 2. Design status and terminology

### 2.1 Revision boundary

The Phase C architecture audit is pinned to commit `c3e0143`. GitHub `main` was verified to point to the same commit when this design was prepared. Architecture claims in this document therefore apply to that revision. If implementation begins from a later commit, the affected paths in the architecture reference MUST be rechecked before relying on them.

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

Preemption is not counted in this expression because `main.tex` defines $S_{\max}$ as the number of execution actions sent to the next forward pass. For a valid LPServe output,

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

This check is deliberately outside the mathematical LP so that the target relaxation retains exactly the three global coupling constraints in `main.tex`. Phase E must nevertheless provide enough ownership information for extraction and Phase F to reject a plan that violates resident capacity. If later work adds resident capacity to the LP itself, the structural argument about the number of fractional request blocks must be reconsidered.

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
- any deterministic ordering key selected by the unresolved tie-breaking policy.

It MUST NOT contain a mutable `Sequence`, scheduler collection, block manager, callback, or engine object.

### 9.2 LP problem record

The problem record MUST contain:

- the ordered, unique request records defining $U_t$;
- explicit membership of $\mathcal Z_t$;
- $B_{\max},C_{\max},S_{\max},M_t^{\mathrm{free}},W_t$;
- the selected decode-memory policy identifier;
- the selected numerical-policy identifier or explicit tolerance values;
- a snapshot identifier sufficient for Phase F stale-state detection.

The record MUST be validated before solver invocation.

### 9.3 Solver result

A solver result MUST distinguish at least:

- optimal solution returned;
- infeasible problem;
- unbounded problem;
- numerical failure;
- solver limit or interruption;
- invalid/malformed solver output;
- unexpected solver exception.

Only a validated optimal result may carry relaxed variables into extraction. Non-success results MUST NOT contain fabricated decisions, silently rounded values, or a substitute all-zero schedule.

The exact solver and treatment of non-optimal feasible incumbents are OPEN.

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

### 9.5 Validators

Phase D requires distinct validators for:

1. problem-input validity;
2. relaxed-solution domains and constraints;
3. extracted integer-plan domains and constraints.

Validators MUST return structured success/failure information. Assertions MAY supplement invariant checks in tests, but correctness MUST NOT rely only on assertions that can be disabled.

### 9.6 Purity requirements

Given identical validated inputs and the same explicitly selected numerical and ordering policies, Phase D MUST produce the same observable result. It MUST NOT read clocks, global scheduler state, mutable queues, block managers, or randomness unless a future design explicitly introduces and records such inputs.

## 10. LP solution classification and extraction

### 10.1 Numerical classification

Mathematically, an action indicator is integral when it is exactly zero or one. In software, classification MUST use an explicit tolerance policy.

For a configured integrality tolerance $\varepsilon_{\mathrm{int}}>0$, an indicator may be classified as zero or one only according to a documented rule. The value of $\varepsilon_{\mathrm{int}}$, endpoint clamping policy, and treatment of larger bound violations are OPEN.

Tolerance is an implementation interpretation of exact mathematical integrality; it is not permission to accept materially infeasible solver output.

### 10.2 Integral/fractional partition

A request belongs to $U_{\mathrm{int}}$ only when all three action indicators

$$
\tilde y_i,\quad\tilde I_i^P,\quad\tilde z_i
$$

are classified as integral. Otherwise it belongs to $U_{\mathrm{frac}}$. For $i\notin\mathcal Z_t$, $\tilde z_i=0$ by construction.

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

after numerical classification has produced exact binary values. Because $I_i^P\le x_i$, an integral $I_i^P=1$ implies $\lfloor\tilde x_i\rfloor\ge1$, subject to validated solver feasibility.

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

The implementation MUST detect non-finite or invalid residuals before continuing.

### 10.4 Dominant legal preemptions

For each request in $U_{\mathrm{frac}}\cap\mathcal Z_t$, compare

$$
\tilde y_i,\quad\tilde I_i^P,\quad\tilde z_i.
$$

If preemption is the dominant relaxed action and $c_i^Z>0$, set $\hat z_i=1$ and increase current planning memory by $c_i^Z$.

The comparison MUST use an explicitly selected deterministic tie rule. No tie rule is approved by this document. Until one is chosen, tied cases must produce an explicit unresolved-policy error in production-facing integration and may be exercised only by tests that supply a test policy.

### 10.5 Safety preemption

After dominant preemptions, while $M_{\mathrm{curr}}<0$:

1. consider only unused requests in $U_{\mathrm{frac}}\cap\mathcal Z_t$ with $c_i^Z>0$;
2. select a request with maximum remaining $\tilde z_i$, subject to the explicit tie rule;
3. set $\hat z_i=1$;
4. increase $M_{\mathrm{curr}}$ by $c_i^Z$.

If no eligible positive-recovery candidate exists while memory remains negative, extraction MUST fail. It MUST NOT select an ineligible request, create memory credit, or reduce a prefill chunk as a substitute for the fixed admission charge.

Multiple safety preemptions are permitted and must be tested.

### 10.6 Fractional decode and prefill packing

For every remaining unpreempted fractional request, compare $\tilde y_i$ and $\tilde I_i^P$ using the selected deterministic policy.

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

The implementation MUST record $ |U_{\mathrm{frac}}| $ for each solve or expose it to the scheduler's instrumentation. It MUST safely process values greater than three. An empirical concentration near three is a measurement, not a proof.

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

The solver adapter MUST verify, under a separate feasibility tolerance:

- all variables are finite;
- action variables lie within their relaxed domains;
- $x_i\ge0$;
- every global and local constraint is satisfied;
- $z_i=0$ outside $\mathcal Z_t$;
- the returned vector has exactly the expected dimensions and request association;
- solver status permits extraction under the selected solver policy.

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

Every non-success result SHOULD record, without fabricating actions:

- phase and failure category;
- scheduler iteration/snapshot identifier;
- solver status and message where applicable;
- violated invariant or exhausted candidate set;
- whether mutation had begun;
- relevant request IDs and capacity summaries;
- selected policy identifiers and tolerances.

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
- infeasible, unbounded if constructible, numerical-error, limit, malformed-result, and exception statuses;
- NaN, infinity, wrong dimensions, duplicate IDs, and invalid coefficient units;
- no fabricated plan on any solver non-success.

### 15.2 Extraction tests

At minimum:

- all-integral solution;
- flooring of integral-request $x_i$;
- preservation of locked integral decisions;
- integrality tolerance immediately inside and outside each endpoint boundary;
- separate feasibility-tolerance behavior;
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
- deterministic handling under the selected tie and iteration-order policy;
- more than three fractional requests;
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

## 17. Unresolved decision register

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
| D-11 | LP solver | SciPy/HiGHS is only a candidate; alternatives allowed | Before Phase D solver adapter |
| D-12 | Acceptable solver statuses | Optimal only versus explicitly permitted non-optimal incumbent | Before Phase D solver adapter |
| D-13 | Solver limits and extreme-point expectations | Time limits, crossover, basis behavior, none selected | Before performance-relevant integration |
| D-14 | Integrality tolerance | `1e-6` is only an early-test candidate | Before extraction implementation is finalized |
| D-15 | Feasibility tolerance and clamping | Separate from integrality tolerance | Before solver-result validator |
| D-16 | Tie-breaking and request iteration order | Must be deterministic; exact rule unspecified | Before extraction implementation is finalized |
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
| Variables, objective, and constraints | `main.tex`, Myopic ILP | None beyond coefficient mapping |
| Continuous $x_i$ | `main.tex`, LP relaxation | `prompt_chunk_len` is the executable integer representation |
| Legal preemption support | `main.tex`; Research Context Sections 4–8 | Architecture Sections 7, 15, 17, 20–23 |
| Fixed prefill charge | `main.tex`; Research Context Sections 5 and 10 | Architecture Sections 3, 14, 15, 23 |
| Extraction | `main.tex`, LP algorithms; Research Context Section 8 | Legal-preemption and physical-validation boundaries |
| Almost-integral caution | `main.tex`; Research Context Section 9 | No solver exists in audited code |
| Phase D/E/F separation | Research Context Sections 12, 14, 15, and 19 | Architecture Sections 16, 22, 27 |
| Request mapping | Research Context Sections 4 and 13 | Architecture Sections 6–9 and 23 |
| Native action execution | Research Context Sections 11–15 | Architecture Sections 11–17 |
| Failure and rollback boundary | Research Context Section 14 | Architecture Sections 14, 16, 21, 22, 26 |
| Validation sequence | Research Context Sections 15 and 19 | Architecture Section 24 |
| Historical baseline | `project_status.md` | Phase B observations summarized in architecture metrics discussion |

### 18.2 Recorded source discrepancies

#### Historical project phase

`project_status.md` states that Phase B is complete and Phase C is next. The completed `lpserve_scheduler_architecture.md` is the later Phase C deliverable. The status file is therefore historical and must not be used to infer the current phase.

#### Exact mathematical integrality versus floating-point classification

The `main.tex` pseudocode uses exact membership in (\{0,1\}), while the research context requires an explicit numerical tolerance. This design preserves exact mathematical domains and requires a separate, explicit software classification policy. Tolerance does not modify the LP.

#### Prevalidation versus atomicity

The research context requires complete validation before mutation. The architecture audit establishes that LPServe has no transactional multi-layer commit. These facts are compatible only if prevalidation is understood as a required risk-reduction boundary, not proof of atomicity or rollback.

#### Complete mapping versus unavailable framework predicates

The sources provide exact mappings for prompt remainder, free blocks, prefill admission charge, and preemption recovery. They do not provide a canonical ownership ledger, pipeline-safe release predicate, final capacity bindings, or decode planning policy. Those cells are explicitly OPEN rather than filled with invented mappings.

### 18.3 Baseline provenance

The Phase A/B status document records a working single-A16 environment and two controlled runs each of LPServe's `sarathi`, `slai_scheduler`, and `vllm` providers. Those runs verified basic framework execution and reproducibility for a tiny workload. They did not validate this LP design, preemption, pipeline execution, semantic generation correctness, or performance.

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
