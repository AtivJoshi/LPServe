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
17. [Decision register](#17-decision-register)
18. [Traceability and change control](#18-traceability-and-change-control)

## 1. Purpose, authority, and scope

### 1.1 Document contract and source authority

This is the normative Phases D--F implementation contract. Its sources are:

| Source | Authority |
|---|---|
| `docs/math/main-llm-serving.tex` | Mathematical source of truth |
| `LP Scheduler Research Context.md` | Implementation principles, validation, and phase boundaries |
| `lpserve_scheduler_architecture.md` | Descriptive Phase C audit evidence at the audited revision |
| `project_status.md` | Chronological project state and handoffs |
| Current LPServe repository | Code facts absent, ambiguous, or changed since the audit |

Architecture detail and research history remain in their source documents. A
conflict MUST be recorded, not silently resolved. **MUST**, **MUST NOT**,
**SHOULD**, and **MAY** decrease in strength; **OPEN** is never an implicit
default, and **BLOCKER** prohibits the affected correctness claim.

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
functions, watermarks, Phase F within-prefill/within-decode metadata ordering,
fallback and rollback behavior, pipeline support, and every remaining OPEN item
have no implicit default.

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

The conceptual flow is `LPServe state → Phase E read-only snapshot → utility and
LP construction → Phase D solve/extraction/plan validation → Phase F fresh
physical prevalidation → native execution → SchedulerOutputs`. It does not
prescribe a class hierarchy.

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

Phase E starts from the new scheduler's authoritative ownership structures,
deduplicates by `seq_id`, rejects contradictory duplicate objects or ownership,
excludes future and finished requests, and preserves every remaining owned
request exactly once. It MUST NOT assume `waiting ∪ running` is universally
complete or inherit SLAI auxiliary ownership; see the architecture reference.

### 4.2 Remaining prompt work

For every $i\in U_t$,

$$
P_i^{\mathrm{rem}}(t)
=
\texttt{get\_prompt\_len()}
-
\texttt{get\_num\_prompt\_tokens\_processed()}.
$$

Phase E rejects a negative remainder. Prefill requires positive remaining work
and a legal LPServe status, ownership, arrival, allocation, and execution path.

### 4.3 Decode eligibility

Decode requires an arrived, non-finished, prompt-complete request with
$P_i^{\mathrm{rem}}=0$, resident allocation, legal status/ownership, and no
in-flight blocker. Zero `prompt_chunk_len` encodes an already validated
decode; it never proves decode legality.

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

Membership requires resident scheduler ownership, non-finished state, an
allocated block table, accepted native status, safe decision-boundary release,
and a validated recomputation-preemption path. `RUNNING`/`PAUSED` alone is
insufficient. For $i\notin\mathcal Z_t$, $z_i=0$; Phase D receives
$\mathcal Z_t$ explicitly and never infers legality from relaxed values or
metadata. Until the recomputation and control-only blockers clear, integrated
Phase E exposes no executable candidate: operational $\mathcal Z_t$ is empty
although synthetic Phase D tests may be nonempty.

### 4.5 Do nothing and non-LP control actions

The canonical do-nothing action is

$$
x_i=y_i=z_i=I_i^P=0.
$$

It leaves queues, status, prompt progress, and block tables unchanged. Prompt
rejection/ignore is not an LP variable and remains separately validated control
behavior, including the Section 16 control-only-output gate.

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

Each prefill token and scheduled decode costs one unit; this is accounting, not
equal wall-clock cost. Its binding/static policy is OPEN, and the VLLM-named
admission budget is not a substitute because it excludes decode actions.

### 6.2 Per-request chunk cap $C_{\max}$

For every request,

$$
0\le x_i\le U_i,
\qquad
U_i=\min(P_i^{\mathrm{rem}},C_{\max}).
$$

$C_{\max}$ is a chunk cap, not a memory coefficient; its binding is OPEN.
LPServe prompt chunks have no universal block-alignment requirement; alignment
may be a policy choice, but it is not native legality.

### 6.3 Scheduled-action width $S_{\max}$

The action-width constraint is

$$
\sum_{i\in U_t}(I_i^P+y_i)\le S_{\max}.
$$

Preemption is excluded because $S_{\max}$ counts next-forward execution actions.
For valid output,

$$
\sum_i(\hat I_i^P+\hat y_i)
=
|\texttt{scheduled\_seq\_metadata\_list}|.
$$

Its value and binding are OPEN.

### 6.4 Separate resident capacity

LPServe's `max_num_seqs` constrains residents, not forward-pass width, and
MUST be enforced separately from $S_{\max}$.

Let $R_t$ be the authoritative resident set before the plan, $A_t$ the set of nonresident requests selected for admission, and $P_t$ the set selected for preemption. The precommit validator MUST enforce

$$
R_t'=(R_t\setminus P_t)\cup A_t,
\qquad
|R_t'|\le\texttt{max\_num\_seqs}.
$$

Resident decode/partial prefill adds no resident; legal preemption removes one;
admission consumes one regardless of chunk size. Keep this check outside the LP
to retain its three-global-constraint structure. Adding it to the LP requires a
design revision and reconsideration of the structural argument.

### 6.5 Decode caps from existing policies

SLAI's `limit_total_decodes` MUST NOT enter this LP without explicit
mathematical revision because it adds a global constraint.

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

Token volume, action width, and planning memory are the only global coupling
constraints; linkage, causality, mutual exclusion, and preemption support are
request-local. This does not prove an at-most-three fractional bound: runtime
logic MUST support arbitrary fractional sets.

## 8. Planning-memory quantities

### 8.1 Free memory and reserve

$M_t^{\mathrm{free}}$ is the block manager's current free physical-block count at the read-only snapshot boundary. $W_t$ is an explicit reserve measured in the same unit.

The admission/append watermark behavior is asymmetric, so $W_t$ MUST NOT
silently equal the block-manager watermark. Its policy remains OPEN.

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

The charge is $a_i^P I_i^P$: admission/recomputation allocates the full current
logical context even for one prompt token, so reducing $x_i$ never reduces
$a_i^P$. Zero resident-prefill charge requires Phase E verification of residency
and allocation.

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

The native append gate may require one block when this gap is zero. $c_i^D$ is
OPEN between the exact gap and a conservative one-block charge; the selected
policy MUST be explicit, recorded, tested, and physically prevalidated.

### 8.4 Preemption recovery

For every $i\in\mathcal Z_t$,

$$
c_i^Z(t)
=
|\texttt{physical\_block\_table}_i|.
$$

This is the audited non-sharing manager's recovery from `free(seq)`;
$c_i^Z=0$ candidates are excluded from safety repair.

### 8.5 Planning feasibility is not execution feasibility

Satisfying

$$
\sum_i(a_i^PI_i^P+c_i^Dy_i)
-
\sum_{i\in\mathcal Z_t}c_i^Zz_i
\le M_t^{\mathrm{free}}-W_t
$$

does not prove execution feasibility. It omits allocator gates, legality,
in-flight safety, replay, and partial failure. Phase D proves planning
feasibility only; Phase F owns physical and operational prevalidation.

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

The project-owned adapter for `scipy.optimize.linprog`/HiGHS MUST convert
this document's maximization objective to SciPy minimization and use a
deterministic explicit variable/request association. It returns a project-owned
structured result; raw SciPy `OptimizeResult` and other SciPy state MUST NOT
cross the Phase D interface. Diagnostics MAY retain solver/version,
method/options, raw status, success, message, and iteration fields.

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

Only normalized SciPy `status == 0` is an optimal candidate; message text is
non-contractual. It still requires independent relaxed-solution validation:
`success != True` or no usable primal vector is invalid/malformed. Every
other category is non-success and cannot reach extraction. A rejected vector
may retain diagnostics but MUST NOT yield fabricated, silently rounded, or
all-zero actions.

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

The reference adapter MUST request `method="highs-ds"`, pass
`options={"presolve": True}`, and omit project-level `time_limit` and
`maxiter`. It MUST NOT configure crossover or undocumented
thread/parallel/random-seed controls. Other tuning stays at documented defaults
of the pinned SciPy version; its exact selection remains D-11 work.

The only admission path is `status 0 → independent validation → deterministic
extraction → integer-plan validation`. D-13 is PARTIAL: finite limits,
method/basis/performance controls, and any basic/extreme-point or
at-most-three-fractional guarantee remain OPEN (D-15 and D-17 govern numerical
validation and live non-success response).

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

Use the absolute integrality tolerance

$$
\varepsilon_{\mathrm{int}}=10^{-6}
$$

only for D-15-validated, finite $[0,1]$ indicators
$\tilde y_i,\tilde I_i^P,\tilde z_i$. It has no relative component and never
classifies or rounds $\tilde x_i$. Classify only after D-15 has completed any
permitted projection:

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

The closed endpoint intervals are disjoint. D-14 is not feasibility clamping:
it neither accepts raw out-of-domain values nor repairs linkage or any coupled
$(I_i^P,x_i)$ state.

### 10.2 Integral/fractional partition

A request is in $U_{\mathrm{int}}$ only when all of
$\tilde y_i,\tilde I_i^P,\tilde z_i$ classify as binary; otherwise it is in
$U_{\mathrm{frac}}$. $\tilde x_i$ does not affect membership, and
$\tilde z_i=0$ outside $\mathcal Z_t$. Classified endpoints are exact `0.0` or
`1.0`; fractional values and raw solver values remain separately available.

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

after classification. The D-15 handoff rejects any indicator lock incompatible
with unchanged $\tilde x_i$; it does not promote, clamp, or jointly repair the
pair. Keep tilded and hatted state separately and never overwrite locked
integral decisions.

### 10.3 Residual capacities

After locking $U_{\mathrm{int}}$, recompute from original problem data and
hatted decisions, never SciPy slacks or relaxed residuals:

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

Negative, non-finite, or otherwise invalid final residuals fail extraction or
integer-plan validation; D-15 tolerance does not apply.

### 10.4 Dominant legal preemptions

For each $U_{\mathrm{frac}}\cap\mathcal Z_t$ request, compare D-15-normalized
$\tilde y_i,\tilde I_i^P,\tilde z_i$ using exact equality. Select preemption
only when $c_i^Z>0$ and $\tilde z_i$ is strictly larger than both competing
values; ties are `decode > prefill > preempt`. A selected preemption sets
$\hat z_i=1$ and adds $c_i^Z$ to current planning memory.

### 10.5 Safety preemption

While $M_{\mathrm{curr}}<0$, choose an unused
$U_{\mathrm{frac}}\cap\mathcal Z_t$ candidate with $c_i^Z>0$, maximum
remaining normalized $\tilde z_i$, then smallest `order_key`; set
$\hat z_i=1$ and add $c_i^Z$. If no such candidate exists, extraction MUST
fail. It MUST NOT use an ineligible victim, create memory credit, or reduce a
prefill chunk to offset its fixed charge. Multiple safety preemptions are
allowed.

### 10.6 Fractional decode and prefill packing

Process remaining unpreempted fractional requests ascending by `order_key`.
Choose the larger of $\tilde y_i$ and $\tilde I_i^P$; an exact tie chooses
decode. Decode requires $B_{\mathrm{curr}},S_{\mathrm{curr}}\ge1$ and
$M_{\mathrm{curr}}\ge c_i^D$, then consumes one token, action, and $c_i^D$.
Prefill requires the corresponding token/action capacity and
$M_{\mathrm{curr}}\ge a_i^P$, then sets

$$
\hat x_i
=
\min(P_i^{\mathrm{rem}},C_{\max},B_{\mathrm{curr}}).
$$

If positive, set $\hat I_i^P=1$ and consume $\hat x_i$, one action, and the
complete $a_i^P$. If that fixed charge does not fit, skip the prefill: do not
shrink its charge or automatically seek additional preemptions.

### 10.7 Canonicalization and final validation

Before success, set $\hat I_i^P=\mathbf1\{\hat x_i>0\}$ for every request.
Canonicalization MUST NOT conceal an inconsistent pre-canonical pair; it fails
unless the discrepancy arose only from a permitted, independently checked
representation conversion. The complete plan MUST pass Section 11 exactly
before Phase E or F consumes it.

### 10.8 Fractional-count observability

Record or expose $|U_{\mathrm{frac}}|$, numerical policy,
$\varepsilon_{\mathrm{int}}$, and aggregate classification counts. Trace output
SHOULD expose raw indicator classification and endpoint distance; Section 14.5
owns the remaining observability contract. Any fractional count, including one
greater than three, is valid input to extraction and not evidence of basicness,
an extreme point, or an approximation guarantee.

## 11. Numerical and feasibility invariants

### 11.1 Input invariants

Before construction, IDs are unique; capacities, block counts, and utilities are
finite and in declared units; $P_i^{\mathrm{rem}}\ge0$;
$C_{\max}>0$ with $U_i=\min(P_i^{\mathrm{rem}},C_{\max})$; unavailable actions
have zero upper bounds; $c_i^Z$ is zero or absent outside $\mathcal Z_t$; and
every required OPEN policy is explicit. LPServe-derived counts cannot be
negative; the numerical policy defines other admissible lower bounds.

### 11.2 Relaxed-solution invariants

The project-owned feasibility tolerance is absolute,

$$
\varepsilon_{\mathrm{feas}}=10^{-7},
$$

in declared LP units, with no relative component and separate from
$\varepsilon_{\mathrm{int}}$. Accept $a^Tv\le b$ only when
$a^Tv-b\le\varepsilon_{\mathrm{feas}}$, and equality/fixed constraints only
when $|a^Tv-b|\le\varepsilon_{\mathrm{feas}}$.

For every D-12-admissible candidate, validate vector length, deterministic
ordering, request association, and finiteness while preserving the raw vector.
Independently recompute all bound, fixed-variable, global, and request-local
residuals from the project-owned problem; reject raw or normalized residuals
above tolerance. Solver residuals are diagnostic only. Checks include domains,
$z_i=0$ outside $\mathcal Z_t$, the three global constraints, linkage,
causality, and mutual exclusion. Only the normalized validated vector reaches
D-14 and extraction.

Projection is limited to explicit single-variable bounds within tolerance:
$x_i$ or an indicator in $[-\varepsilon_{\mathrm{feas}},0)$ becomes `0.0`;
an indicator in $(1,1+\varepsilon_{\mathrm{feas}}]$ becomes `1.0`; and a
fixed-zero variable within tolerance becomes `0.0`. Valid in-domain
near-endpoint values are not projected. Recompute every residual after
projection; post-projection revalidation is mandatory.

D-15 MUST NOT clamp interior fractional values or residuals, alter $x_i$ for a
near-endpoint $I_i^P$, repair coupled, global, or other multi-variable
constraints, causality, or mutual exclusion, or otherwise convert a materially
infeasible vector into another point.

Before locking an indicator-integral request, prospective D-14 classification
must be compatible with unchanged $x_i$: $I_i^P=1$ cannot yield
$\lfloor x_i\rfloor=0$, and $I_i^P=0$ cannot conceal linkage-violating positive
$x_i$. Reject rather than promote, demote, or jointly repair the pair.

A SciPy status-0 candidate MUST include a finite reported objective. Recompute
it in SciPy minimization convention, reject a material mismatch as malformed,
and derive the project maximization value explicitly. Use a documented
scale-aware allowance distinct from both tolerances; recompute and record the
normalized objective after projection. The two tolerances MUST NOT be conflated.

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

### 11.4 Semantic and output invariants

| Area | Required checks |
|---|---|
| Per-request action | Prefill is eligible with $1\le\hat x_i\le\min(P_i^{\mathrm{rem}},C_{\max})$; decode is eligible and exactly one token; preemption is in $\mathcal Z_t$; no request is both scheduled and preempted. |
| Memory and plan | Charge full unallocated-prefill cost and zero resident-prefill cost only after verified allocation; preserve locked integral decisions; emit one decision record per input request. |
| `SchedulerOutputs` | Scheduled, preempted, and ignored IDs are known, owned, unique, and pairwise disjoint; prefill chunks are positive/bounded, decode chunks are zero, and no chunk is negative. |
| Derived output | Token/action counts and populated counters match the validated plan; metadata order meets the execution/replay contract. |

## 12. Read-only Phase E state mapping

### 12.1 Phase boundary

Phase E builds one coherent, immutable Phase D snapshot and is independently
testable without a solver. It MUST NOT mutate collections, blocks, status, or
prompt progress; construct replayable output; or choose fallback behavior.

### 12.2 Snapshot consistency

The mapper coherently captures snapshot ID/time, authoritative ownership,
status/completion, prompt progress, logical/physical block lengths, free blocks,
supported in-flight state, capacities, and selected OPEN policies. It fails on
incoherent state and never combines incompatible instants.

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

Every unfinished request has one authoritative owner; auxiliary indexes may
reference it but not own it. Phase E proves each included request occurs once
and every owned, arrived, unfinished request is in $U_t$. It MUST NOT inherit
SLAI's `_active_seq_ids`, `paused_prefills`, or `decode_queue` ownership without
an explicit decision and tests.

### 12.5 Capacity construction

Keep $B_{\max}$ (tokens now), $C_{\max}$ (one-request prefill cap),
$S_{\max}$ (execution actions now), and `max_num_seqs` (post-plan residents)
separate. One value may bind several only under explicit approved policy with
separate names and validation.

### 12.6 Stale-plan detection

The plan carries this snapshot ID. Phase F revalidates ownership, status,
allocation, lengths, progress, completion, free blocks, and in-flight markers
before commit. The mechanism is OPEN; a mismatch is precommit failure, never
permission to patch a stale plan.

## 13. Validated Phase F action execution

Phase F runs only after valid Phase E, admissible solve, successful extraction
and plan validation, current snapshot identity, and enabled blocker gates.
Before its first mutation, it validates the whole plan: current
existence/ownership/status/eligibility; ID uniqueness and exclusion; chunk and
resident bounds; recovery, allocation, append, watermark, replay, and
commit-order feasibility; and supported pipeline/control-only state.
Prevalidation reduces risk, not non-atomicity.

| Planned action | Native representation and required prevalidation | Gate |
|---|---|---|
| Unallocated prefill | Waiting/arrived/unfinished/prompt-incomplete; full allocation and resident slot; move ownership once, allocate full logical context, emit `SequenceScheduleMetadata` with positive `prompt_chunk_len=\hat x_i`. | Admission cost is independent of chunk length. |
| Resident prefill | Verify allocation, $a_i^P=0$, positive remainder, and ownership; no new admission; emit positive `prompt_chunk_len` metadata. | Verify zero allocation in integration tests. |
| Decode | Verify prompt completion, resident allocation/status, and append feasibility; append in metadata order; emit `prompt_chunk_len=0`. | $c_i^D$ is the pre-action gap. |
| Preemption | Remove resident owner once, free central blocks, native-return to waiting, emit only `preempted_seq_ids`, replay `reset_for_recompute()`/local frees. | Disabled until applicable Section 16 blockers clear. |
| Do nothing | No mutation or output entry. | Empty-plan fallback/liveness is OPEN. |

Central and workers use one replay-compatible deterministic order: validated
ignored controls, preemption/free, scheduled actions, ownership finalization,
then `SchedulerOutputs`; native replay uses ignored, preempted, then scheduled
IDs, which are disjoint. Enabled mixed batches are prompt-first and sampler-safe;
unsupported identity/length, pipeline, or control-only cases are rejected.
Mutation granularity and post-mutation recovery remain OPEN. Before output,
verify ownership, allocation, no loss/duplication, free-block deltas, and exact
action/token counts. Replay tests, not output alone, establish central/worker
block equality.

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

Every solver attempt SHOULD retain, without fabricating actions: normalized
outcome; phase/stage/reason; raw status, success, non-contractual message, and
vector presence/shape; solver identity/version/method/options and deliberate
absence of limits; available HiGHS, `nit`, and `crossover_nit` data; reported,
raw-recomputed, and normalized objectives; problem/snapshot ID; validator and
extraction entry/outcome; numerical policy, both tolerances, projection count,
worst raw/normalized violation, and classified-indicator/fractional counts;
selected IDs, capacity summary, violated invariant or exhausted candidate set;
and whether mutation began. For a limit, diagnostics MUST NOT infer time versus
iteration cause from message text alone.

Trace output SHOULD expose raw indicator classification and endpoint distance;
full vectors and per-coordinate projection detail MAY remain debug artifacts.
Routine diagnostics MUST retain validation stage and rejection reason.
Experiment summaries distinguish policy decisions from solver, extraction,
validation, and execution failures.

## 15. Testing and phase acceptance requirements

No GPU experiment substitutes for mathematical or state-transition tests. Each phase must satisfy its own gate before the next mutation boundary is enabled.

### 15.1 Phase D problem and solver tests

| Contract | Minimum coverage |
|---|---|
| Model construction | Empty; prefill-, decode-, preemption-only and mixed problems; continuous $x_i$; linkage (including reject $I_i^P=1,x_i<1$), causality, exclusion, token/action/memory saturation, and constructible feasible, infeasible, and unbounded solver cases. |
| Input and layout validation | NaN, infinity, wrong dimensions, duplicate IDs, invalid units, and D-15 vector-order/request association; preserve raw vectors. |
| Adapter and objective | Known nondegenerate SciPy/HiGHS optimum and vector; deterministic reconstruction; maximization-to-minimization conversion; reject missing, non-finite, or materially mismatched reported objectives; recomputed sign conversion. |
| D-13 configuration | `method="highs-ds"`, explicit `presolve=True`, no `time_limit`/`maxiter`, explicit crossover, or undocumented thread/parallel/random-seed controls; when exposed, `crossover_nit == 0` is diagnostic only. Crossover, status 0, and `highs-ds` prove neither basicness/extremity nor a $|U_{\mathrm{frac}}|$ bound. |
| Admission | Status 0 with `success=True` and valid shape reaches validation, then extraction only after validation succeeds; status 0 with bad/missing/wrong-sized/non-finite vector or `success != True` is invalid. Statuses 1--4, unknown/inconsistent status, exception, and interruption are rejected even with feasible-looking vectors; messages do not alter classification. |
| Non-success boundary | Scoped finite-limit and independently feasible lower-objective status-1 fixtures remain D-12 non-successes. Spy/call-order tests prove no non-success reaches extraction or fabricates an all-zero plan. |
| D-15 validation | Projection at half, exactly $\varepsilon_{\mathrm{feas}}$, and `nextafter` beyond it; reject over-tolerance values while retaining valid in-domain endpoints. Recompute raw and normalized global/local residuals, including post-projection failure; no coupled or multi-variable repair. |
| Result integrity and diagnostics | Preserve raw/normalized vectors; recompute hatted-plan residual capacity. Record §14.5 fields, observed $|U_{\mathrm{frac}}|$, and environment provenance. |
| Reproducibility | Repeated unique-optimum solves return the expected result; degenerate-optimum tests require validated objective and feasibility, not one primal vector. |

### 15.2 Extraction tests

| Contract | Minimum coverage |
|---|---|
| Classification and partition | All-integral inputs and flooring; D-14 exact `0.0`/`1.0`, inclusive endpoints, interiors, and adjacent representable values via `nextafter`; retain original indicators and fractional values. All three indicators classify into $U_{\mathrm{int}}$; any fractional indicator into $U_{\mathrm{frac}}$; $\tilde x_i$ is irrelevant. |
| D-14/D-15 handoff | Repeated deterministic classification and locked integral decisions; no near-one $\tilde I_i^P$/sub-one $\tilde x_i$ repair. Out-of-domain values, non-finiteness, residuals, and permitted normalization are D-15 tests, not D-14 classification tests. |
| Fractional-set support | Observe but never use $|U_{\mathrm{frac}}|$ as acceptance; more than three fractional requests remains valid. |
| Preemption | Dominant legal selection; exclude non-$\mathcal Z_t$ and zero-recovery candidates; one and multiple safety preemptions; fail after positive-recovery candidates are exhausted. |
| Packing and charges | Fractional decode/prefill selection and exact ties under each residual capacity; prefill truncation; accept full fixed charge only when it fits; shrinking a chunk cannot shrink $a_i^P$; zero resident-partial-prefill and full admission/recomputation charges. |
| Ordering and final plan | Canonicalize $\hat I_i^P$; validate/reject D-16 tuple keys; permutation-invariant layout/output; exact action and safety-preemption ties; repeated key-sorted actions; exact final integer-plan validation. |

### 15.3 Phase E mapping tests

Use real or faithful LPServe `Sequence` and block-manager state to test:

- request universe: exact prompt remainder; future/finished exclusion; complete
  owned-unfinished inclusion; duplicate/conflicting ownership rejection; and
  prefill/decode eligibility;
- preemption: empty operational $\mathcal Z_t$ while blockers remain, then the
  legal predicate after they clear;
- capacity mapping: allocator free blocks; resident/admission/recomputation
  charges; exact/conservative decode charge; recovery; and separate
  $B_{\max},C_{\max},S_{\max}$, and resident capacity; and
- stale-snapshot detection and proof that mapping performs no mutation.

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

| ID | Status and selected rule | Canonical requirements |
|---|---|---|
| D-11 | **RESOLVED** — `scipy.optimize.linprog`/HiGHS behind a project-owned adapter; SciPy must become an explicit reproducible dependency before Phase D execution, and its exact version is not selected here. | §9.3; tests §15.1; diagnostics §14.5 |
| D-12 | **RESOLVED** — optimal-only: only normalized SciPy `status == 0` may enter validation; only its validated candidate may extract. | §9.3, §11.2, §15.1 |
| D-13 | **PARTIAL** — reference: `highs-ds`, explicit `presolve=True`, no project `time_limit`/`maxiter`, and no explicit crossover or undocumented thread/parallel/random-seed controls. Performance/control/basis remainder is **OPEN**. | §9.3; §10.8; §15.1–15.2 |
| D-14 | **RESOLVED** — absolute $\varepsilon_{\mathrm{int}}=10^{-6}$ for validated indicators only. | §10.1; §15.2 |
| D-15 | **RESOLVED** — absolute $\varepsilon_{\mathrm{feas}}=10^{-7}$, independent validation, narrow projection, and revalidation only. | §11.2; §15.1 |
| D-16 | **RESOLVED** — ascending immutable lexicographic `order_key`; exact extraction ties. | §§9–10; §15.2 |

D-13 provides no basic/extreme-point or fractional-count guarantee; its
performance, control, and basis remainder remains OPEN (§9.3).

### 17.1 OPEN decisions

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

## 18. Traceability and change control

### 18.1 Source-to-requirement traceability

Trace formulation and continuous $x_i$ to `docs/math/main-llm-serving.tex`;
validation, phase boundaries, mapping, execution, and failure to the Research
Context; and framework facts to the architecture audit. `project_status.md`
is chronology only. Implementations MUST retain the cited mathematical and
audited constraints.

### 18.2 Change-control rules

Changes MUST preserve the formulation (or record an approved amendment), keep
OPEN choices explicit, respect D/E/F boundaries, update affected tests and
records, document observed verification, and re-audit revision-sensitive facts.

An explicit design revision is required to add a global coupling constraint,
alter fixed-charge prefill accounting, broaden $\mathcal Z_t$, permit a new
solver admission outcome, change extraction ordering, or enable a blocked
execution mode. These cannot be incidental implementation choices.

---

This specification is intentionally incomplete only where the source material leaves a genuine research choice or framework blocker. Such incompleteness is represented by explicit OPEN decisions and BLOCKER gates; it must not be replaced by undocumented defaults.
