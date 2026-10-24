# LPServe LP-Relaxation Scheduler Design

**Document role:** Normative implementation specification  
**Target repository:** [`AtivJoshi/LPServe`](https://github.com/AtivJoshi/LPServe)  
**Code baseline:** [`c3e014363dd50e1830d7c85c3d043eab69fdc9e5`](https://github.com/AtivJoshi/LPServe/commit/c3e014363dd50e1830d7c85c3d043eab69fdc9e5)  
**Design date:** 2026-09-04  
**Implementation target:** Primal Heuristic 1, LP relaxation followed by deterministic integer extraction

**Corrected-framework amendment:** 2026-10-12; requirements awaiting implementation (§2.1)

**Future parallelism extension approved:** 2026-10-18; design recorded in §19, implementation and validation pending. The existing single-stage contract remains in force until that extension is implemented and checked.

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
10. [Whole-action extraction](#10-whole-action-extraction)
11. [Numerical and feasibility invariants](#11-numerical-and-feasibility-invariants)
12. [Read-only Phase E state mapping](#12-read-only-phase-e-state-mapping)
13. [Validated Phase F action execution](#13-validated-phase-f-action-execution)
14. [Failure contracts](#14-failure-contracts)
15. [Testing and phase acceptance requirements](#15-testing-and-phase-acceptance-requirements)
16. [Known inherited framework limitations in MVP compatibility mode](#16-known-inherited-framework-limitations-in-mvp-compatibility-mode)
17. [Decision register](#17-decision-register)
18. [Traceability and change control](#18-traceability-and-change-control)
19. [Approved future TP and PP extension](#19-approved-future-tp-and-pp-extension)

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
- the deterministic LP-guided whole-action constructor (§10), which preserves the
  amended integer constraints and useful execution;
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
- pipeline-parallel execution before the approved extension in §19 is implemented and its validation gates pass.

Only decisions recorded as resolved in Section 17 are selected. The future
parallelism design is selected in §19; it does not establish implemented or
tested support. Utility functions and every remaining OPEN item have no
implicit default. The admission watermark is read from the shared block manager (§8.1).
Phase F metadata ordering is resolved for the supported
single-stage scope by D-21. The
selected pre-mutation failure response is the scoped fail-stop rule in Section
14.2 and D-17. The selected post-mutation response is the no-rollback,
fail-stop rule in Section 14.4 and D-20. No alternate-policy fallback is
selected.

## 2. Design status and terminology

### 2.1 Revision boundary

The Phase C architecture audit and this design's code baseline are pinned to commit `c3e0143`. GitHub `main` was verified to point to that commit when the design was prepared. The later committed changes through `6fbc046` were documentation-only and did not change the audited implementation paths; they do not repin the historical architecture audit. Architecture claims in this document therefore remain claims about `c3e0143`. If implementation begins from a later commit or a working tree with relevant source changes, the affected paths in the architecture reference MUST be rechecked before relying on them.

**Corrected-framework amendment (2026-10-12).** This revision aligns the design with the
mathematical correction in `docs/math/main-llm-serving.tex` (`subsec:myopic_ilp`,
committed at `b1addc8`; `subsec:lp_relaxation`, committed at `69e2180`) and with
decisions R01–R21 in `docs/lp_scheduler_architecture_discussion_phase_F.4.md`.
Amended passages are marked as amended and implementation pending. They are
requirements, not implemented behavior. At `69e2180` the code still implements
the superseded contracts: the conservative one-block decode charge, the planning
reserve, indicator locking with dominant and safety preemption, `order_key`-only
ordering, head-of-queue idle detection, utility validation that checks finiteness but
does not reject negative values, utility construction that reads live queues
before mapping, and
no pre-registration admission check. Evidence paragraphs that cite those contracts
(for example, scoped inputs with `memory_reserve` or `conservative_one_block_v1`)
remain revision-specific historical evidence; they do not validate the amended
requirements. Manuscript sections after `subsec:lp_relaxation` are unrevised (see
the red revision boundary there) and are not normative for this design.

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

### 2.4 MVP compatibility and minimalism policy

The immediate objective is a basic runnable LP scheduler. For this MVP, the LP
path reuses existing LPServe/SLAI native behavior, including recomputation
preemption, without repairing pre-existing serving-framework defects. Known or
newly discovered inherited framework issues are documented as limitations unless
they prevent the selected MVP path from running. They are not evidence that the
LP scheduler repaired them, and affected runs do not establish corrected
framework semantics.

The MVP implements the smallest path needed for that objective. It does not add
speculative abstractions, fallback policies, retries, rollback, broad
compatibility handling, or exhaustive edge-case coverage. Unsupported states or
unexpected failures fail visibly through a clear error or structured failure
result; they are not silently repaired or converted into fabricated actions.

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

*(amended 2026-10-12; implementation pending)* Partition $U_t$ by allocation state into the resident set
$\mathcal R_t$ (allocated, owned by `running`, including residents omitted from
the next forward pass) and the waiting set $\mathcal Q_t$ (arrived and
unallocated, owned by `waiting`, including recomputation-preempted requests).
Decode and preemption eligibility lie in $\mathcal R_t$; prefill eligibility lies
in $\mathcal R_t\cup\mathcal Q_t$. Requests that have not arrived are outside
$U_t$ and take no action.

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

For the MVP, membership follows the selected scheduler's native preemption
eligibility and ownership path: the request is resident, unfinished, allocated,
and has a native executing status. `RUNNING`/`PAUSED` alone remains insufficient
without the selected scheduler ownership and allocation checks. For
$i\notin\mathcal Z_t$, $z_i=0$; Phase D receives $\mathcal Z_t$ explicitly and
never infers eligibility from relaxed values or metadata. This compatibility
rule inherits the framework's known recomputation, control-only, and in-flight
limitations; it is not a proof of safe release or corrected semantics.

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
| $U_i(t)$ | $\min(P_i^{\mathrm{rem}}(t),C_{\max})$; the manuscript's $q_i$ | Tokens |
| $a_i^P(t)$ | Fixed block charge for any positive prefill | Physical KV blocks |
| $c_i^D(t)$ | Actual additional blocks allocated by one decode, in $\{0,1\}$ (D-09, amended) | Physical KV blocks |
| $c_i^Z(t)$ | Blocks recovered by legal preemption (whole cache) | Physical KV blocks |
| $\pi_i^E(t)$ | Immutable service rank for this decision; smaller first (amended) | Rank |
| $\pi_i^Z(t)$ | Immutable victim rank for $i\in\mathcal Z_t$; smaller first (amended) | Rank |
| $\alpha_i(t)$ | Utility of one decode | OPEN scale |
| $\beta_i(t)$ | Utility per prefill token | OPEN scale |
| $\gamma_i(t)$ | Preemption penalty coefficient | OPEN scale |

Utility coefficients MUST be supplied explicitly. *(amended 2026-10-12; implementation pending)* The mathematical source requires them to be finite and nonnegative on a common utility scale. They may depend on state but are fixed during each decision (R17). Their values, formulas, and normalization remain OPEN (D-01–D-04).

### 5.3 System parameters

| Object | Meaning |
|---|---|
| $B_{\max}$ | Maximum combined number of prefill and decode tokens in the next forward pass |
| $C_{\max}$ | Maximum prefill chunk offered to any one request |
| $S_{\max}$ | Maximum number of scheduled execution actions in the next forward pass |
| $F=M_t^{\mathrm{free}}$ | Free physical KV blocks at the snapshot boundary. The amended `subsec:myopic_ilp` defines the alias $M_{\mathrm{avail}}=F$ with no reserve; later manuscript sections that use $M_{\mathrm{avail}}$ remain unrevised |
| $h$ | Shared block manager's admission watermark in blocks; not a reserve (§8.1; amended) |
| $N_{\max}$ | Resident capacity, `max_num_seqs`; counts all retained residents; not $S_{\max}$ (amended) |

The independent planning reserve $W_t$ is removed (D-10, amended 2026-10-12).

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

$U_i$ is the manuscript's $q_i$. $C_{\max}$ is a chunk cap, not a memory
coefficient; its binding is OPEN.
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
admission consumes one regardless of chunk size.

*(amended 2026-10-12; implementation pending)* Under R08 the resident limit is also an LP constraint:

$$
|\mathcal R_t|-\sum_{i\in\mathcal R_t}z_i+\sum_{i\in\mathcal Q_t}I_i^P\le N_{\max},
\qquad N_{\max}=\texttt{max\_num\_seqs}.
$$

Every retained resident counts, including one omitted from the next forward
pass. The precommit check above remains required against current state. The
former instruction to keep this check outside the LP to preserve a
three-global-constraint structure is superseded (§7.3).

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

subject to the rows below, where

$$
H=F+\sum_{i\in\mathcal R_t}c_i^Zz_i-\sum_{i\in U_t}a_i^PI_i^P
$$

is a linear expression for the free blocks after selected preemptions and
prefills, before decodes; it is not a decision variable:

$$
\sum_{i\in U_t}(x_i+y_i)\le B_{\max},
\tag{token volume}
$$

$$
\sum_{i\in U_t}(I_i^P+y_i)\le S_{\max},
\tag{action width}
$$

$$
H-\sum_{i\in U_t}c_i^D y_i\ge0,
\tag{actual memory}
$$

$$
\sum_{i\in U_t}y_i\le S_{\max}H,
\tag{decode prerequisite}
$$

$$
H\ge h\,I_i^P
\qquad \forall i\in\mathcal Q_t,
\tag{admission watermark}
$$

$$
|\mathcal R_t|-\sum_{i\in\mathcal R_t}z_i+\sum_{i\in\mathcal Q_t}I_i^P\le N_{\max},
\tag{resident capacity}
$$

$$
\sum_{i\in U_t}(I_i^P+y_i)\ge1,
\tag{useful execution}
$$

$$
I_i^P\le x_i\le U_iI_i^P
\qquad \forall i\in U_t,
\tag{prefill linkage}
$$

$$
I_i^P\le\mathbf 1\{i\in\mathcal P_t\},
\qquad
y_i\le\mathbf 1\{i\in\mathcal D_t\}
\qquad \forall i\in U_t,
\tag{eligibility}
$$

where $\mathcal P_t$ and $\mathcal D_t$ are the prefill- and decode-eligible sets
of §§4.2–4.3 (decode eligibility includes $P_i^{\mathrm{rem}}=0$),
and

$$
I_i^P+y_i+z_i\le1
\qquad \forall i\in U_t.
\tag{mutual exclusion}
$$

The integer domains are those in Section 5. For $i\notin\mathcal Z_t$, $z_i=0$.

*(amended 2026-10-12; implementation pending)* This is the committed `subsec:myopic_ilp` formulation. The independent
reserve $W_t$ is removed and no final-state reserve is imposed. When $h=0$ the
watermark rows are implied by actual memory and MAY be omitted. Useful execution
applies only after input and admission validation to a nonempty arrived
universe (D-18, D-26); preemption alone does not satisfy it, and a state without
a legal useful integer plan is infeasible and reported, never given an invented
action. Under the declared execution order (victims, prefills, zero-demand
decodes, one-block decodes; §13), the rows match the shared allocator checks.
Free memory is least, equal to $H$, after the last admission, so one watermark row
per admission covers the whole admission group without charging $h$ per
admission. Because $H$ is an integer and at most $S_{\max}$ decodes are selected,
the decode prerequisite means $H\ge1$ whenever any decode is selected; this
covers the free-block-before-decode check for zero-demand decodes, and actual
memory then covers the one-block decodes. The watermark is an admission check,
not a reserve: decodes may consume the blocks it protects. $S_{\max}$ bounds
execution actions; $N_{\max}$ bounds residents. These arguments assume a coherent
synchronous non-sharing snapshot, correct fixed charges and whole-cache
recovery, independent legal preemptions, and $c_i^D\in\{0,1\}$. They do not
replace Phase F prevalidation or establish whole-engine correctness. Admission
validity does not guarantee eventual completion.

### 7.2 Continuous relaxation

The LP retains every constraint above, including useful execution, resident
capacity, actual memory, the decode prerequisite, and every watermark row, and
replaces the domains with

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

It is never directly executable; it only guides extraction (§10).

### 7.3 Coupling structure and fractional solutions

*(amended 2026-10-12; implementation pending)* Token volume, action width, actual memory, the decode prerequisite,
resident capacity, useful execution, and the per-admission watermark rows couple
requests through shared totals. Eligibility, linkage, mutual exclusion, and
preemption support are request-local. The former three-global-constraint
argument and the expected at-most-three-fractional-request property are
retired: the watermark rows admit optimal vertices with arbitrarily many
fractional requests (analytical construction in `subsec:lp_relaxation`).
Extraction correctness MUST NOT depend on the number of fractional requests, on a
vertex or basic solution, or on solver basis behavior.

An explicit constraint matrix repeats $H$'s coefficients in every watermark row,
so storage can be quadratic in the worst case: up to $|\mathcal Q_t|$ added rows
with $O(|U_t|)$ nonzeros each. No auxiliary variable or multiple-solve
reformulation is approved to change this.

## 8. Planning-memory quantities

### 8.1 Free memory and admission watermark

*(amended 2026-10-12; implementation pending)* $F=M_t^{\mathrm{free}}$ is the block manager's current free
physical-block count at the read-only snapshot boundary. $h$ is the shared
block manager's admission watermark, `watermark_blocks`
$=\lfloor\texttt{watermark}\cdot T\rfloor$ for total pool $T$ (default
watermark 0.01, so $h=0$ when $T<100$). The mapper reads it once; it is not an
LP configuration value.

The native admission check requires the free blocks minus the admitted request's
full demand to be at least $h$. The inherited schedulers' `can_append_slot()`
convention requires at least one free block before each decode, even at zero
gap, and applies no watermark (§8.3). The LP represents these two checks
separately: the admission watermark
($H\ge hI_i^P$) and the decode prerequisite ($\sum_iy_i\le S_{\max}H$). The
independent planning reserve $W_t$ is removed (D-10). No final-state reserve is
imposed, and neither check may be silently replaced by a reserve.

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

*(amended 2026-10-12; implementation pending)* D-09 is decided: $c_i^D$ is this exact demand and MUST be zero or one for
a decode-eligible resident. A gap outside $\{0,1\}$ is an unsupported-state
mapping error, never a clamped demand. The conservative one-block charge
(`conservative_one_block_v1`) is retired for the corrected path. The inherited
schedulers' free-block convention is represented by the decode prerequisite
(§§7.1, 8.1), not by charging every decode a block. Selected decodes with
$c_i^D=0$ execute before those with $c_i^D=1$ (§13).

**Clarification 2026-10-19 (source review at `2c61d3e`; no contract change).**
The LP follows the one-free-block-before-decode convention enforced by the
inherited [vLLM-style](../sarathi/core/scheduler/vllm_scheduler.py),
[Sarathi](../sarathi/core/scheduler/sarathi_scheduler.py),
[SLAI](../sarathi/core/scheduler/slai_scheduler.py), and
[SimpleChunking](../sarathi/core/scheduler/simple_chunking_scheduler.py)
schedulers through `can_append_slot()`. References here to the native append
gate mean this scheduler convention, not a physical allocator requirement:
[`append_slot()`](../sarathi/core/block_space_manager/base_block_space_manager.py)
allocates only for an actual block gap, and worker replay ignores the query's
return value. Zero-demand decodes can execute with no free blocks, as observed
in the [bounded comparison](handoffs/scheduler_comparison_handoff.md#addendum-2026-10-18-memory-pressure-case).
The adapted vLLM reference uses actual additional demand and does not enforce
this convention. Both implementations remain unchanged. Whether to remove the
convention from LP or add it to the adapted reference is deferred; either
requires an explicit later decision and corresponding contract updates.

### 8.4 Preemption recovery

For every $i\in\mathcal Z_t$,

$$
c_i^Z(t)
=
|\texttt{physical\_block\_table}_i|.
$$

This is the audited non-sharing manager's recovery from `free(seq)`.
*(amended 2026-10-12; implementation pending)* Safety repair is retired (§10). A legal victim with $c_i^Z=0$ is not
excluded, because it can release a resident place.

### 8.5 Planning feasibility is not execution feasibility

Satisfying the amended resource rows of §7.1 does not prove execution
feasibility. It omits allocator gates, legality,
in-flight safety, replay, and partial failure. Phase D proves planning
feasibility only; Phase F owns physical and operational prevalidation.

## 9. Pure Phase D interfaces

This section specifies semantic interfaces, not implementation code or mandatory class names.

### 9.1 LP request record

Each request record MUST contain only immutable, framework-independent data:

- stable request ID;
- $P_i^{\mathrm{rem}}$ and $U_i$;
- prefill-, decode-, and preemption-eligibility flags;
- resident or waiting membership ($\mathcal R_t$ or $\mathcal Q_t$) (amended);
- $a_i^P$, exact $c_i^D\in\{0,1\}$, and $c_i^Z$;
- service rank $\pi_i^E$ and, for $i\in\mathcal Z_t$, victim rank $\pi_i^Z$,
  immutable for the decision (amended);
- $\alpha_i,\beta_i,\gamma_i$, finite and nonnegative, constructed as specified in
  §12.3 (amended 2026-10-12);
- an immutable lexicographic `order_key`.

It MUST NOT contain a mutable `Sequence`, scheduler collection, block manager, callback, or engine object.

`order_key` MUST be present, unique within the snapshot, finite/comparable, and
normalized to a tuple of integers; malformed, duplicate, or unstable keys are
invalid input. For the single-scheduler MVP, the current engine's monotone
counter provides each `Sequence` with a unique non-negative integer
`seq_id`. Phase E preserves that raw integer in its immutable snapshot,
normalizes the Phase D `request_id` to its decimal string, and constructs
`order_key=(raw_seq_id,)`. Non-integer, negative, duplicate, or otherwise
unsupported live IDs are mapping failures. This scoped rule MUST be revisited
if request-ID generation changes or a scheduling universe spans independent
ID generators.

*(amended 2026-10-12; implementation pending)* Under D-16, `order_key` remains the stable request identity, the canonical
LP variable and vector order, and the final tie-break. It MUST NOT determine
policy preference: extraction comparisons, victim emission, and execution
order use LP values and ranks first (§§10, 13).

### 9.2 LP problem record

The problem record MUST contain:

- the ordered, unique request records defining $U_t$;
- explicit membership of $\mathcal Z_t$;
- $B_{\max},C_{\max},S_{\max},N_{\max}$, $F=M_t^{\mathrm{free}}$, $h$, and the
  current resident count $|\mathcal R_t|$ (amended 2026-10-12; $W_t$ removed);
- no decode-memory policy identifier: exact demand is mandatory (amended
  2026-10-12);
- the selected numerical-policy identifier or explicit tolerance values;
- a snapshot identifier associating the mapped problem, returned plan, and
  diagnostics; state stability and current-state validation are governed by
  Section 12.6, not by the identifier.

**Snapshot identifier (approved 2026-10-04).** The snapshot ID is
`str(scheduler._iteration_id)`, the decimal string of the inherited scheduler
iteration number already read and validated by the mapper; it is carried
unchanged as the `StateSnapshot` snapshot ID, `LPProblem.problem_id`, the
returned plan's problem ID, and diagnostics. It identifies one scheduling
decision within one scheduler instance only. It is not a content fingerprint,
proof of unchanged state, or globally unique identifier, so plans MUST NOT be
reused across scheduler instances or runs. The mathematical layer's existing
problem/plan identity check is retained. This replaces the earlier SHA-256
content fingerprint and adds no counter, UUID, timestamp, or version system.

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
| `2` | Solver-reported LP infeasibility (amended) |
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

*(amended 2026-10-12; user decision R21; implementation pending)* Status `2` is
reported as **solver-reported LP infeasibility**, a distinct non-success outcome
that retains the raw status, message, and other available diagnostics. It is a
numerical solver report, not an exact mathematical certificate. The statement
that an infeasible relaxation implies an infeasible ILP applies only to actual
mathematical infeasibility; the manuscript's `LPInfeasible` outcome is that
exact abstraction. This outcome is distinct from the constructor's exhaustive
no-useful-work outcome (§10.5) and from other solver or numerical failures. No
certificate checker or additional solve is added. Like every non-success
category, it never reaches extraction.

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
extraction → integer-plan validation`. D-13 is PARTIAL: finite limits and method/basis/performance controls remain
OPEN (D-15 and D-17 govern numerical validation and live non-success response).
No extreme-point or fractional-count guarantee is required or available; the
amended LP admits arbitrarily many fractional requests (§7.3).

### 9.4 Integer action plan

A successful extraction returns, for every request ID in ascending `order_key`:

- integer prefill token count $\hat x_i$;
- binary decode decision $\hat y_i$;
- binary preemption decision $\hat z_i$;
- canonical prefill indicator $\hat I_i^P$.

The plan MUST also expose *(amended 2026-10-12; implementation pending)*:

- the declared victim order (victim rank, then identity) and execution order
  (prefills, zero-demand decodes, then one-block decodes; service rank, then
  identity, within each group) as lists separate from the per-request records;
- final local totals: token use, action count, resident count after the plan,
  $\hat H$, and total decode demand;
- the numerical policy used;
- the source problem/snapshot identifier;
- success or an explicit outcome (§10.5).

The former residual-capacity, dominant/safety-preemption, and
indicator-fractional-count fields are retired with that extraction. A
fractional count MAY still be reported as an observation (§10.8).

### 9.5 Validators

Phase D requires distinct validators for:

1. problem-input validity;
2. relaxed-solution domains and constraints;
3. extracted integer-plan domains and constraints.

Validators MUST return structured success/failure information. Assertions MAY supplement invariant checks in tests, but correctness MUST NOT rely only on assertions that can be disabled.

The relaxed-solution validator MUST preserve the raw primal vector for
diagnostics and expose only its separately normalized, validated vector to
extraction (amended 2026-10-12: the former D-14 classification handoff is
retired).

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

Phase names and numbers are project-management labels only. They MUST NOT
appear in Python filenames, package or module names, identifiers, comments,
docstrings, printed output, diagnostics, result stages, categories, or failure
messages. Python implementation surfaces use responsibility-based terminology.

## 10. Whole-action extraction

*(amended 2026-10-12; implementation pending)* This section replaces the former
indicator classification, integral/fractional partition, residual locking,
dominant preemption, safety repair, positive-support-only packing, and
fractional-count requirements. It specifies the committed `alg:lp_rounding` and
`alg:lp_extraction` of `subsec:lp_relaxation`, approved as P04/R15 with
P05/R16. There is one extraction path; no integral-LP fast path is used. The
current `extract_integer_plan` at `69e2180` implements the superseded method.

### 10.1 Inputs and ordering

Extraction receives only the D-15-validated relaxed vector and the immutable
problem. Execution candidates are all requests in
$\mathcal P_t\cup\mathcal D_t$, including those with zero LP support, ordered by
descending $e_i=\tilde I_i^P+\tilde y_i$, then ascending $\pi_i^E$, then
identity (`order_key`). Victim candidates are all of $\mathcal Z_t$, including
residents whose LP actions are integral and idle and residents with
$\tilde z_i=0$, ordered by descending $\tilde z_i$, then ascending $\pi_i^Z$,
then identity. Validated finite LP values are compared numerically and exactly,
without a tolerance-based comparator; ranks and identity break remaining ties.
Each list is sorted once per decision.

Processing is deterministic for fixed validated LP values. This does not imply
identical solver solutions across solvers, versions, builds, or platforms, or
among multiple LP optima. LP values are primary, so ranks do not guarantee
running-tail eviction or any reference policy's service order.

### 10.2 Trials and legality checks

Start with an empty local plan and no accepted victims; the empty plan is not a
schedule. Scan the execution candidates once. Skip a candidate that is already
an accepted victim, or when fewer than one token or one action slot remains.
Otherwise propose one whole action: a decode (one token) or a prefill chunk

$$
\hat x_i=\min\{U_i,\ B_{\mathrm{rem}},\ \max(1,\lfloor\tilde x_i\rfloor)\},
$$

where $B_{\mathrm{rem}}$ is the token budget remaining in the local plan. If the
trial plan is not legal, add legal victims tentatively in victim order,
excluding the candidate, accepted executions, and accepted victims, until the
trial is legal or the list is exhausted. A victim with $c_i^Z=0$ is not
skipped. A failed trial is discarded completely: accepted actions and victims
are unchanged and the scan continues. No other victim subset is searched.

With exact integer arithmetic and
$\hat H=F+\sum_{i\in\mathcal R_t}c_i^Z\hat z_i-\sum_ia_i^P\hat I_i^P$, a plan is
legal when:

- token use is at most $B_{\max}$ and execution count is at most $S_{\max}$;
- $|\mathcal R_t|-\sum_{i\in\mathcal R_t}\hat z_i+\sum_{i\in\mathcal Q_t}\hat I_i^P\le N_{\max}$;
- $\hat H\ge\sum_ic_i^D\hat y_i$; $\hat H\ge h$ if any admission is selected;
  and $\hat H\ge1$ if any decode is selected;
- every action is eligible, no request both executes and is preempted, and each
  selected chunk is an integer in $[1,U_i]$, with $\hat x_i=0$ otherwise.

These are the integer forms of §7.1 except useful execution, which is required
of the completed plan rather than of each trial.

### 10.3 Acceptance and chunk extension

- Accept the first legal trial and its victims regardless of utility.
  *Accepted limitation (user decision 2026-10-12, recorded under R15):* this can
  accept negative utility even when another legal useful plan has nonnegative
  utility. The two-pass alternative is not adopted.
- After useful execution exists, accept a trial that introduces new victims
  only when its reward minus its new penalties is strictly positive. Accept a
  trial without new victims when its gain is nonnegative, which always holds
  for nonnegative coefficients. Reward is $\alpha_i$ for a decode or
  $\beta_i\hat x_i$ for the trial chunk; new penalties are $\sum\gamma_k$ over
  the trial's new victims.
- Never remove accepted execution or accepted victims.

After the scan, extend selected prefill chunks with the remaining tokens in
ascending service rank, then identity, up to $U_i$, without adding requests or
victims. Extension changes only token use. Return the plan only after complete
independent integer-plan validation (§11.3), including useful execution. If no
trial is accepted, return the no-useful-work outcome (§10.5).

### 10.4 Numerical handling

Resource counts and legality use exact integers. The chunk rule floors the
validated $\tilde x_i$. For an optional addition with new victims, compute
reward $r$ and new penalties $p$ with stable summation and accept only if

$$
r-p>\tau,
\qquad
\tau=\texttt{objective\_abs\_tol}
+\texttt{objective\_rel\_tol}\cdot\max(|r|,|p|),
$$

using the existing objective tolerances ($10^{-9}$ each in the current
numerical policy). This conservative guard may decline tiny positive gains. It
does not apply to the first accepted action or to additions without new
victims. Nonfinite arithmetic is a numerical failure, never a profitable
action. No tolerance value is retuned and no new policy coefficient or
numerical knob is introduced. Indicator classification with
$\varepsilon_{\mathrm{int}}$ has no role in extraction (D-14).

### 10.5 Outcomes

- *Solver-reported LP infeasibility* (§9.3): no extraction.
- *No legal useful integer work:* every eligible first-action trial failed even
  with all other legal victims. Under the assumptions of §10.6 this exhaustively
  establishes that no legal useful integer plan exists, although the LP may be
  feasible.
- *Solver or numerical failure*, including failed relaxed-solution validation
  and nonfinite extraction arithmetic.
- *Implementation or validation failure:* the final plan fails §11. This proves
  no impossibility.

None counts as useful execution; each is a pre-mutation failure under D-17.

### 10.6 Analytical guarantees, assumptions, and limitations

The committed manuscript arguments assume exact arithmetic, validated inputs,
fixed additive charges (full-context admission independent of chunk size),
positive $B_{\max}$, $S_{\max}$, and $N_{\max}$, preemptions that are legal
together in any subset, a coherent synchronous non-sharing snapshot,
$c_i^D\in\{0,1\}$, and no constraints beyond §7.1 (no joint-action, retention,
or victim-count rules). Under these assumptions, every returned plan is legal
and contains useful execution, and the constructor returns useful work whenever
a legal useful integer plan exists. These are mathematical arguments, not
executed evidence; F01 closes only after implementation and the checks of
§15.7. Retention rules, victim limits, group quotas, shared caches, or
action-dependent allocation would invalidate the arguments and require
revision.

Accepted limitations: avoidable negative utility on first acceptance (§10.3);
no utility approximation guarantee (the manuscript gives a family whose ratio
tends to zero); no minimal-victim or victim-count bound, so mass preemption is
possible (discussion P04 review); each candidate is tried once; no restart,
starvation, completion, or policy-recovery guarantee; and no
solver-independent tie outcome.

### 10.7 Complexity

Assume incrementally maintained local totals and constant-time trial checks;
the manuscript's vector notation does not require copying or rescanning all
requests. With $u=|U_t|$, $n$ execution candidates, and $r=|\mathcal Z_t|$,
initialization takes $O(u)$, sorting $O(n\log n+r\log r)$, and the scan
$O(n(r+1))$. Extension, final ordering, and final validation take
$O(u+n\log n+r\log r)$. Construction plus final validation therefore takes
$O(u+nr+n\log n+r\log r)$. This excludes LP construction, the solve, and
relaxed-solution validation; explicit watermark rows can make the constraint
matrix quadratic in size (§7.3). No measured speed is claimed.

### 10.8 Observability

Record the extraction outcome, numerical policy, candidate and victim counts,
accepted actions and victims in declared order, the number of declined optional
trials, final local totals, and, for exhaustive failure, data identifying the
exhausted candidate set. The number of requests with fractional LP indicators
MAY be reported as a descriptive observation; any value is valid and carries no
structural or quality meaning.

## 11. Numerical and feasibility invariants

### 11.1 Input invariants

Before construction, IDs are unique; capacities, block counts, and utilities are
finite and in declared units; $P_i^{\mathrm{rem}}\ge0$;
$C_{\max}>0$ with $U_i=\min(P_i^{\mathrm{rem}},C_{\max})$; unavailable actions
have zero upper bounds; $c_i^Z$ is zero or absent outside $\mathcal Z_t$; and
every required OPEN policy is explicit. LPServe-derived counts cannot be
negative; the numerical policy defines other admissible lower bounds.
*(amended 2026-10-12; implementation pending)* Also: utility coefficients are finite and nonnegative; $c_i^D\in\{0,1\}$
for decode-eligible residents; $h\ge0$ and $N_{\max}>0$ are integers;
$\mathcal R_t$ and $\mathcal Q_t$ partition $U_t$ consistently with ownership
and allocation, with $|\mathcal R_t|\le N_{\max}$; and every required rank is
present and finite. At `69e2180` utilities are checked only for finiteness.

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
$z_i=0$ outside $\mathcal Z_t$, every shared row of §7.1 (token volume, action
width, actual memory, decode prerequisite, each watermark row, resident
capacity, and useful execution), eligibility, linkage, and mutual exclusion
(amended 2026-10-12). $H$ is evaluated once and reused for the watermark rows.
Only the normalized validated vector reaches extraction.

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

*(removed 2026-10-12)* The former prospective-locking compatibility check is
retired with indicator locking. The remaining D-15 rules above are unchanged.

A SciPy status-0 candidate MUST include a finite reported objective. Recompute
it in SciPy minimization convention, reject a material mismatch as malformed,
and derive the project maximization value explicitly. Use a documented
scale-aware allowance distinct from both tolerances; recompute and record the
normalized objective after projection. The two tolerances MUST NOT be conflated.

### 11.3 Integer-plan algebraic invariants

Every successful plan MUST satisfy exactly *(amended 2026-10-12; implementation pending)*, with
$\hat H=F+\sum_{i\in\mathcal R_t}c_i^Z\hat z_i-\sum_ia_i^P\hat I_i^P$:

$$
\sum_i(\hat x_i+\hat y_i)\le B_{\max},
\qquad
\sum_i(\hat I_i^P+\hat y_i)\le S_{\max},
$$

$$
\hat H\ge\sum_ic_i^D\hat y_i,
\qquad
\hat H\ge1\ \text{if}\ \textstyle\sum_i\hat y_i>0,
\qquad
\hat H\ge h\ \text{if}\ \textstyle\sum_{i\in\mathcal Q_t}\hat I_i^P>0,
$$

$$
|\mathcal R_t|-\sum_{i\in\mathcal R_t}\hat z_i+\sum_{i\in\mathcal Q_t}\hat I_i^P\le N_{\max},
\qquad
\sum_i(\hat I_i^P+\hat y_i)\ge1,
$$

$$
\hat I_i^P=\mathbf1\{\hat x_i>0\},
\qquad
\hat I_i^P+\hat y_i+\hat z_i\le1
\qquad\forall i,
$$

and every selected action is eligible. All hatted variables must have their
integer domains. Integer plans are not accepted with numerical slack outside
their domains. The second line is the integer form of actual memory, the decode
prerequisite, and the watermark rows.

### 11.4 Semantic and output invariants

| Area | Required checks |
|---|---|
| Per-request action | Prefill is eligible with $1\le\hat x_i\le\min(P_i^{\mathrm{rem}},C_{\max})$; decode is eligible and exactly one token; preemption is in $\mathcal Z_t$; no request is both scheduled and preempted. |
| Memory and plan | Charge full unallocated-prefill cost and zero resident-prefill cost only after verified allocation; preserve accepted actions, chunks, and victims exactly through validation and execution (amended); emit one decision record per input request. |
| `SchedulerOutputs` | Scheduled, preempted, and ignored IDs are known, owned, unique, and pairwise disjoint; prefill chunks are positive/bounded, decode chunks are zero, and no chunk is negative. |
| Derived output | Token/action counts and populated counters match the validated plan; metadata order meets the execution/replay contract. |

## 12. Read-only Phase E state mapping

### 12.1 Phase boundary

Phase E builds one coherent, immutable Phase D snapshot and is independently
testable without a solver. It MUST NOT mutate collections, blocks, status, or
prompt progress; construct replayable output; or choose fallback behavior.
After validating the authoritative ownership boundary, it returns either a
snapshot containing a nonempty `LPProblem` or a structured mapping failure. It
accepts only a nonempty arrived, unfinished request universe; an empty arrived
universe produced by direct mapping is a mapping failure. Neither result may
retain mutable LPServe objects. Ordinary future-only or completely empty
scheduling is handled by the live scheduler before mapper invocation (D-18).

### 12.2 Snapshot consistency

The mapper coherently captures snapshot ID/time, authoritative ownership,
status/completion, prompt progress, logical/physical block lengths, free blocks,
supported in-flight state, capacities, and selected OPEN policies. It fails on
incoherent state and never combines incompatible instants.

### 12.3 Complete mathematical-to-LPServe mapping

| Mathematical object | LPServe source or normative construction | Status |
|---|---|---|
| Request ID $i$ | Decimal string of the verified non-negative integer `Sequence.seq_id`; raw integer retained in the immutable snapshot | Resolved for the single-scheduler MVP by D-16 |
| `order_key` | `(raw_seq_id,)`, using the current engine's monotone integer counter assignment; stable identity, canonical variable order, and final tie-break only | D-16, amended 2026-10-12 |
| $U_t$ | Deduplicated arrived, non-finished sequences owned by the LP scheduler's inherited `waiting` or `running` collection | Resolved for the single-stage MVP by D-08 |
| $\mathcal R_t$, $\mathcal Q_t$ | Members of $U_t$ owned by `running` and by `waiting` | Amended 2026-10-12; implementation pending |
| $\pi_i^E$, $\pi_i^Z$ | Built from the same captured queue state. Initial vLLM-oriented profile: service rank is current `running` position (including partial prefills), then arrived `waiting` position; victim rank is reverse `running` position | Approved R16; implementation pending |
| $\mathcal Z_t$ | Owned resident, allocated, native-status-eligible requests on the selected native preemption path | MVP compatibility mode inherits known limitations; pipeline predicate OPEN |
| $P_i^{\mathrm{rem}}$ | `get_prompt_len() - get_num_prompt_tokens_processed()` | Resolved |
| Prefill eligibility | Positive remainder plus legal ownership/status/allocation path | Exact status policy must be documented |
| Decode eligibility | Prompt complete, zero remainder, resident/allocated, legal status, not in-flight | Resolved boundary; pipeline predicate OPEN |
| $U_i$ | `min(P_i_remaining, C_max)` | Resolved |
| $B_{\max}$ | Explicit combined prompt/decode token budget | Concrete config binding OPEN |
| $C_{\max}$ | Explicit per-request prefill cap | Concrete config binding OPEN |
| $S_{\max}$ | Explicit scheduled-metadata/action-width cap | Concrete config binding OPEN |
| Resident capacity | $N_{\max}=$ `max_num_seqs` in the LP row and the precommit check; $|\mathcal R_t|=$ `len(running)`; `running` is the authoritative resident ledger | D-08; LP row amended 2026-10-12 (D-27) |
| $F=M_t^{\mathrm{free}}$ | `block_manager.get_num_free_gpu_blocks()` | Resolved |
| $h$ | `block_manager.watermark_blocks` | Amended 2026-10-12; replaces $W_t$, which is removed (D-10) |
| $a_i^P$, resident partial prefill | Zero after resident ownership and block-table allocation are verified | Resolved |
| $a_i^P$, admission/recompute | `len(seq.logical_token_blocks)` | Resolved |
| $c_i^D$ | Exact logical/physical gap in $\{0,1\}$; another gap is a mapping failure | Decided (D-09, 2026-10-12); implementation pending |
| $c_i^Z$ | Current physical block-table length | Resolved for legal candidate |
| $\alpha_i,\beta_i,\gamma_i$ | Finite, nonnegative values constructed from the same captured state by the coefficient data path below, fixed for the decision | Data path amended 2026-10-12; formulas, scales, and $\kappa$ OPEN |

**Utility input interface (approved 2026-10-04).** The mapper receives
utilities as a `Mapping[int, RequestUtility]` keyed by raw integer
`Sequence.seq_id`. The key set must equal the included arrived, unfinished
request set exactly; keys follow the raw-ID integer rules (booleans rejected),
and each value is a `RequestUtility` whose three fields are finite reals.
Malformed input, missing or extra keys, or invalid values are mapping failures.
The mapper copies validated values into new immutable records and retains no
reference to the caller's mapping. Duplicate entries that the caller
overwrote while constructing the dictionary cannot be detected by the mapper.
This choice fixes only the input shape; utility formulas, scaling, and the
D-01–D-04 policy decisions remain OPEN. *(amended 2026-10-12)* The values must
also be nonnegative. This externally supplied mapping remains a validated
payload where applicable, for example for explicit fixtures, but it is not the
sole route for coefficients; state-dependent policies use the data path below.

**Coefficient and rank data path (2026-10-12; P10 step 6; implementation
pending).** Within the existing read-only mapping responsibility, one
scheduling decision proceeds as follows:

1. *Capture once.* Read and validate the live resource, request, and queue
   facts once at the synchronous boundary: ownership and queue positions,
   arrival and completion, prompt progress, logical and physical block counts,
   free and total blocks, $h$, and resident capacity. Copy them into immutable
   captured data. Subsequent rank, coefficient, and LP-problem construction use
   this captured data without another live-state capture. The executor still
   performs the complete current-state validation required by §§12.6–13 before
   mutation; that validation neither rebuilds the mapper nor recomputes ranks or
   coefficients.
2. *Construct ranks.* Build $\pi_i^E$ and $\pi_i^Z$ from the captured queue
   positions (initial vLLM-oriented profile in the table above).
3. *Construct coefficients.* Call a small pure coefficient-construction
   function on the immutable captured data and explicit policy parameters. It
   returns one $(\alpha_i,\beta_i,\gamma_i)$ per request in $U_t$. Validate
   every value as finite and nonnegative; otherwise return a mapping failure.
   The function receives no mutable `Sequence`, scheduler, block manager,
   callback, or engine object, and it performs no solve.
4. *Construct the problem.* Build the immutable `LPProblem` from the captured
   data, ranks, and validated coefficients.

Ranks and coefficients are fixed for the whole decision: solve, extraction,
precommit validation, and execution use the values carried in the immutable
snapshot and plan. The executor MUST NOT recompute ranks or coefficients from
live queues. Explicit uniform utilities remain a usable policy: the function
returns the configured triple for every request. A validated per-request
mapping, as above, also remains an accepted policy input. The data path adds no
second mapper build, second live snapshot, dummy solve, new ownership
structure, or persistent history ledger. It prescribes responsibilities and
sequencing, not classes or new layers.

Final formulas, scales, and the optional $\kappa$ remain OPEN (D-01–D-04). If
the optional discarded-work penalty (R18) is instantiated, its computed-token
and output-progress primitives MUST come from the same capture; allocated
block capacity is not computed work. At `69e2180`, `LPScheduler` builds the
uniform utility mapping by reading the live queues before calling the mapper.
That is harmless for uniform values but would be a second live read for a
state-dependent policy, and it is superseded by this data path.

**Physical allocation representation (approved 2026-10-04).** The immutable
snapshot retains allocation quantities, not block identities: each request
record carries `physical_block_count: int`, equal to
`len(block_manager.get_block_table(seq))` for an allocated resident and `0`
for an unallocated waiting request. The native block manager remains the sole
holder of physical block identities; the mapper copies no block table and
keeps no separate allocation ledger. The logical/physical block gap and
$c_i^Z$ above use this count unchanged. Exact block numbers are no longer
preserved in the snapshot for debugging.

### 12.4 Ownership contract

For the single-stage MVP, the LP scheduler inherits `BaseScheduler` directly,
not `SLAIScheduler`, and uses the inherited collections as its only ownership
structures:

- `waiting` is the authoritative owner of every unallocated unfinished request
  awaiting initial admission or recomputation;
- `running` is the authoritative owner and resident ledger for every allocated
  unfinished request, whether or not that request is selected for the next
  forward pass.

Every unfinished request MUST occur in exactly one of those collections.
Unselected residents remain in `running`. At the supported mapping boundary,
`waiting` requests MUST be unallocated, `running` requests MUST have central
block tables, and resident ownership MUST agree with the block manager. Missing
ownership, cross-collection duplication, contradictory objects sharing an ID,
or an ownership/allocation mismatch is a mapping failure.

The mapper observes this state at the quiescent beginning of a scheduling
decision, before queue or block mutation, with one pipeline stage and no batch
in flight. Auxiliary priority or lookup structures MAY be added later only as
non-owning indexes; they MUST NOT define or extend the request universe. The LP
scheduler does not inherit SLAI's `_active_seq_ids`, `paused_prefills`, or
`decode_queue`. Pipeline ownership remains governed by D-24; the supported
single-stage state-stability contract is defined in Section 12.6 and D-19.

Phase E deduplicates the authoritative collections by `seq_id`, excludes future
and finished requests from $U_t$, and proves that every remaining owned request
appears in $U_t$ exactly once.

### 12.5 Capacity construction

Keep $B_{\max}$ (tokens now), $C_{\max}$ (one-request prefill cap),
$S_{\max}$ (execution actions now), and `max_num_seqs` (post-plan residents,
$N_{\max}$) separate. The admission watermark $h$ is read from the shared block
manager, not configured (amended 2026-10-12). One value may bind several only
under explicit approved policy with separate names and validation.

### 12.6 State stability and precommit validation

**Scope note (2026-10-18):** This remains the existing single-stage contract.
The approved future extension replaces whole-decision state stability with
protected capture, unprotected planning, and protected commit (§19.3).

For the single-stage MVP, state stability comes from the supported execution
contract rather than a new lock or version subsystem. The same engine instance
MUST receive no overlapping state-changing public calls; mapping starts with
one pipeline stage and zero running batches; and mapping, solve, extraction,
fresh prevalidation, and execution occur synchronously within one scheduler
decision call. Pipeline execution and concurrent public engine calls are
outside this contract.

The plan carries the snapshot ID to associate it with the mapped problem and
diagnostics; the ID is not a lock, reservation, or proof of unchanged state.
Immediately before its first mutation, Phase F MUST verify that the plan belongs
to the current mapped problem and validate the complete plan against current
state as specified in Section 13. This includes current request ownership and
eligibility, action and resident bounds, supported in-flight state, and native
memory feasibility for all actions together in execution order. Any identity
mismatch or failed prevalidation is a precommit failure with zero mutation.
Under D-17 the MVP raises its top-level scheduling failure and does not
automatically retry or re-plan.

**Approved simplification (2026-10-04):** The synchronous, non-overlapping
execution contract supplies state stability. Phase F therefore does not perform
a separate field-by-field comparison with the earlier snapshot or rebuild the
mapper to compare its output. This removes a redundant comparison while
preserving complete current-state plan validation. It gives up the separate
comparison's detection of unexpected state changes that violate the execution
contract; integration must verify the supported synchronous execution path.
Pipeline execution and overlapping state-changing calls remain unsupported.
No additional lock, state-version counter, reservation, or layer-local retry
mechanism is required for this scoped MVP.

## 13. Validated Phase F action execution

Phase F runs only after valid Phase E, admissible solve, successful extraction
and plan validation, and a plan snapshot ID matching the current mapped problem.
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
| Preemption | Remove resident owner once, free central blocks, native-return to waiting, emit `preempted_seq_ids`, replay `reset_for_recompute()`/local frees. | MVP compatibility mode: use the native path without repairing its inherited limitations; reject a control-only plan before mutation. Executed since 2026-10-10 for the subset below (native preemption execution). |
| Do nothing | No request, queue, block, or batch-state mutation and no execution entry. | When no arrived unfinished request exists, the live scheduler returns an ordinary empty `SchedulerOutputs` before mapping (D-18, amended 2026-10-12; the current check reads the waiting head). The inherited scheduler iteration counter still advances. With arrived work present, a plan with no prefill or decode action fails `no_progress` before execution mutation. |

Central and workers use one replay-compatible deterministic order: validated
ignored controls, preemption/free, scheduled actions, ownership finalization,
then `SchedulerOutputs`; native replay uses ignored, preempted, then scheduled
IDs, which are disjoint. The MVP does not repair inherited mixed-batch,
sampler-association, pipeline, or control-only behavior. A plan that would
produce controls without scheduled metadata, including a preempt-only plan,
is rejected during precommit validation before mutation; the executor does not
force an unrelated scheduled action to hide that failure. After mutation
begins, D-20 supplies fail-stop behavior without rollback or recovery.

**Amended execution order and queue transitions (2026-10-12; R07, R16, D-21
amended; implementation pending).** Before any mutation, the executor builds the
complete declared order and prevalidates it:

1. preempt and emit the selected victims in ascending victim rank, then
   identity; all victims are freed before any execution, so this order does not
   change resource totals;
2. execute the selected prefills, then the selected decodes with $c_i^D=0$,
   then those with $c_i^D=1$, in ascending service rank, then identity, within
   each group.

Central native operations, emitted `preempted_seq_ids`, and emitted
`scheduled_seq_metadata_list` MUST follow exactly this order. Prevalidation
applies victim legality and recovery, resident capacity, each admission's
watermark check, and the free-block-before-decode check in this order, using a
temporary count of free blocks and residents. The executor keeps its
independent checks; the mathematical validator does not replace them.

Queue transitions: retained residents keep their relative `running` positions
whether served, omitted, or changing phase; physical decode grouping MUST NOT
reorder `running`. Selected admissions are removed from `waiting` and appended
to `running` in declared admission order (service rank, then identity).
Unselected waiting entries keep their order. Native front insertion of each
victim leaves the emitted victim list reversed at the front of `waiting`. For
example, with running A, B, C, waiting D, E, and the vLLM-oriented victim ranks,
victims B and C are emitted as C then B and leave waiting B, C, D, E before
admissions are removed. A victim is not re-admitted in the same decision. All
arrived candidates remain visible; a future waiting head must not hide arrived
work. The shared allocator, its `can_append_slot` check, and the allocation,
append, and preemption operations are unchanged. At `69e2180` the executor
still orders victims and each execution group by `order_key` and does not group
decodes by demand; that order is superseded.

**Approved simplification (2026-10-04):** Use native queue and block operations
and keep final runtime checks close to SLAI. Before output, check
`len(running) <= max_num_seqs`; `running` is the authoritative resident
collection, so no separate active-request ledger is introduced. Verify that
emitted scheduled request IDs and prompt-chunk lengths match the validated
plan in the selected execution order, and that emitted control IDs match its
validated controls. Derive action/token counts from the emitted entries rather
than maintaining a second counting system. This output check is needed because
the executor translates a separately constructed LP plan into native actions.

Do not perform a separate post-execution ownership or allocation audit, either
globally or for affected requests. Focused execution tests verify complete
before-and-after queue membership, allocation, no request loss or duplication,
and expected free-block changes, including unchanged unrelated requests. This
scope gives up some immediate detection of unrelated state corruption. Failure
of a native operation or final runtime check after mutation begins remains
subject to D-20: terminate the run and do not reuse the affected engine state.
When the supported MVP path exercises and claims a cross-layer replay result,
replay tests rather than output alone establish central/worker block equality.
Other action combinations remain focused follow-up coverage under §15.4.

**Clarification (2026-10-06, three-request contention milestone):** The
contention CPU case checks central/worker allocated-request sets, per-request
block counts, and free-block counts after every replay, plus independent pool
integrity in each block manager. It does not establish identical physical
block IDs: the inherited native free iterates `set(block_table)` in
identity-hash order, so after a multi-block table is freed the two allocators
can assign different IDs. Exact-ID equality is claimed only for the earlier
focused cases that observed it, and this behavior is not repaired. See
`docs/handoffs/lp_scheduler_contention_handoff.md`.

**Execution and metadata order (D-21, approved 2026-10-05):** For the supported
single-stage scope, the executor executes and emits all selected prefills
first, then all selected decodes, each group ascending by the existing
`order_key`. Central native operations and emitted
`scheduled_seq_metadata_list` entries use this same order. The order is
built as a new action list; it does not reorder the accepted mathematical plan
in place and changes neither the selected actions nor their chunk sizes. It
matches the model runner's prompt-first input packing and keeps a deterministic
order within each group. It does not repair the inherited mixed-batch and
sampler limitations in §16.4 (D-25). Focused test:
`tests/test_lpserve_plan_execution.py`,
`test_mixed_output_is_prompt_first_in_native_execution_order`.

**Initial executor supported subset (approved 2026-10-05):** *(Describes
implemented behavior at `69e2180`. Where it conflicts with the amended execution
order, D-16, D-18, D-21, or D-26–D-31, the amended requirement governs.)*
`lpserve_plan_execution.execute_plan(scheduler, snapshot, result)` is one
synchronous call that receives the live scheduler, the immutable
`StateSnapshot`, and a `SchedulingSuccess`. It does not invoke the mapper or
solver. It supports only:

- admission of an unallocated waiting request with a positive chunk, using the
  native manager's full initial allocation demand (the full logical context
  for the supported manager, regardless of chunk size), the native admission
  watermark, and the inherited `_allocate` helper; the waiting entry is removed
  once and appended to `running` once, in execution order;
- continuation of a resident partial prefill with zero logical/physical block
  gap and no new allocation;
- one decode for a prompt-complete resident with a gap of zero or one, through
  the inherited `_append_slot` helper; the native append gate requires a free
  block even at gap zero, and only the actual gap is consumed;
- any combination of these in one prompt-first output.

Before mutation it checks record identity against the current
`_iteration_id`, reuses `validate_problem` and `validate_integer_plan`,
applies the progress gate, checks one pipeline stage and zero running batches,
and checks each selected request's current ownership, status, arrival
relative to `snapshot_time`, unfinished state, prompt progress, allocation,
chunk bound against current remainder and $C_{\max}$, resident capacity, and
overlength admission (read-only; the mutating prompt-rejection helper is not
called). Combined native memory feasibility is proved with a temporary running
count of free blocks and residents in execution order; the allocator and block
tables are not copied. A rejection returns the immutable mathematical-layer
`Failure` record (stage `precommit_validation`, or the plan validator's own
failure unchanged) with no output and no mutation. After mutation begins,
exceptions propagate under D-20. The executor does not change
`_iteration_id` or `num_running_batches`, and leaves status transitions and
prompt progress to replay and completion.

*Superseded 2026-10-10 by native preemption execution below:* the initial
subset did not execute runtime preemption and rejected execution plus any
selected preemption as `unsupported_preemption`.

Prompt-ignore controls are not executed by this executor. An all-zero or
preempt-only plan fails as `no_progress`. Selected preemptions are never
removed and no action is forced. Successful output carries an empty ignored
list. Live `LPScheduler` integration, ordinary idle handling, and D-17
exception conversion are not part of the executor.

**Native preemption execution (approved 2026-10-10):** *(Describes implemented
behavior at `69e2180`; its `order_key` victim order and references to dominant,
safety, or integral victims are superseded by the amended execution order and
§10.)* In the same
single-stage synchronous subset (one pipeline stage, zero batches in flight,
no overlapping state-changing calls), the executor runs every selected,
legally eligible resident preemption in a plan that also selects at least
one prefill or decode. It neither caps the number of victims nor requires a
victim to have generated no tokens; eligibility is the mapper's legal
preemption set (§4.4) and is not narrowed. The `unsupported_preemption`
category no longer exists.

- *Victim and control order.* All victims are gathered from the plan,
  whether extraction marked them dominant, safety, or integral. The executor
  calls the inherited `_preempt` once per victim in ascending `order_key`,
  before any scheduled action, and emits `preempted_seq_ids` in that order.
  `_preempt` inserts each victim at index 0 of `waiting`, so several victims
  end at the waiting front in reverse call order. This effect is approved;
  native insertion is not changed and no queue reordering is added.
- *Prevalidation.* Before the first mutation, each victim must be a mapped
  resident in the legal preemption set and must currently be owned by
  `running`, `PAUSED` (native executing), arrived, unfinished, allocated,
  and prompt-consistent. Its recovery is the length of its current physical
  block table, which the supported non-sharing manager frees in full. A
  table that repeats a block, or shares one with another selected victim,
  fails as `preemption_recovery`; recovery is never taken from the planning
  coefficient alone. Each victim credits its recovery to the temporary
  free-block count and removes one resident slot. Every scheduled action is
  then checked in execution order against that account: admissions consume
  their full initial allocation and a resident slot and must leave the
  native watermark; decodes need a free block even at gap zero and consume
  only the actual gap; resident prefills keep their allocation. A later
  infeasible action rejects the whole plan before even the first victim is
  freed. Rejections follow D-17 unchanged.
- *Mutation and replay.* Each victim is removed from `running` once, then
  `_preempt` frees its central blocks and inserts it into `waiting`. The
  executor does not reset the sequence. Prefills and then decodes follow
  (D-21). Central replay of the preempted ID calls `reset_for_recompute()`
  before scheduled actions; worker replay resets its separate copy and frees
  its local blocks before the scheduled allocation/append. A later mapping
  sees the victim as waiting and unallocated, so normal admission allocates
  its full current context and recomputes from prompt token zero.
- *Final checks.* Besides the resident-limit and scheduled-entry checks,
  emitted `preempted_seq_ids` must equal the validated victim order and the
  ignored list must be empty. After the first mutation, D-20 applies.
- *Evidence boundary.* Focused executor tests cover single and multiple
  victims, including a victim with generated tokens, through hand-constructed
  validated plans. The live CPU case
  (`docs/handoffs/lp_scheduler_preemption_handoff.md`) has the real solver
  select the preemption of a partial-prefill resident that has generated no
  tokens, then replays, recomputes, and completes it. Interruption after
  generation is allowed at runtime; §§16.1–16.2 apply to it unchanged.
  *Validated 2026-10-10 for one bounded case only*
  (`docs/handoffs/lp_scheduler_decode_preemption_reference_handoff.md`):
  the real solver preempts a single victim A after its first generated
  token to admit B; B finishes; A's expanded context (original prompt plus
  that token, once) is readmitted, recomputed from token zero, decodes twice,
  and finishes. It passed live on CPU and on one GPU with real TinyLlama
  weights, where A's cumulative history and B's token equaled uninterrupted
  `VLLMScheduler` references. As inherited (§16.1), LP A asked for 2 tokens
  but generated 3 cumulatively (the reference asked for 3), and A's final
  output IDs hold only the 2 post-reset tokens (§16.2). Scoped inputs were
  fixed per-request utilities (A decode 40, prefill-token 1, penalty 1;
  B decode 40, prefill-token 20, penalty 1), supplied by a validation-only
  override of the utility-building method. This case does not establish
  corrected generation caps or output/text semantics, repeated preemption,
  multiple victims, block growth during generation, arbitrary workloads,
  fairness, pipeline support, or performance.
- *Scoped inputs.* The live CPU case uses block size 4, a 4-block pool in
  central and worker managers, `max_model_len=32`, resident limit 1,
  `b_max=4`, `c_max=4`, `s_max=1`, reserve 0, `conservative_one_block_v1`,
  uniform prefill-token utility 1, decode utility 10, preemption penalty 1,
  and `lp_relaxation_mvp_v1`. These are provisional test inputs. D-03 (the
  preemption penalty policy) remains OPEN; runtime preemption runs only with
  an explicitly configured penalty.

Ordinary future-only or completely empty scheduling is decided by the
live `LPScheduler` before the mapper is invoked. It follows the inherited
single-stage scheduler convention: when `running` is empty and `waiting` is
empty or its head has `arrival_time > now`, the scheduler returns an ordinary
empty `SchedulerOutputs`. This check inherits the existing queue-order and
future-request-validity assumptions of the audited schedulers (for example,
the head-of-queue `arrival_time > now` break in `vllm_scheduler.py` and
`sarathi_scheduler.py`); it does not add a separate future-request validation
subsystem. That path invokes neither mapping, `LPProblem` construction, the
solver, extraction, precommit validation, nor execution. It does not change
waiting/running ownership, sequence status or progress, block state, or
`num_running_batches`. As in every inherited `BaseScheduler.schedule()` call,
`_iteration_id` advances before scheduler-specific logic runs, and the empty
`SchedulerOutputs.id` uses that incremented value. The mapper accepts only a
nonempty arrived, unfinished universe: if direct mapping produces an empty
universe, it returns a `MappingFailure`. The mapper may still filter future
requests out of a mixed nonempty snapshot, as before, without validating them
further.

*(amended 2026-10-12; implementation pending)* Under R14, ordinary idle is classified by the absence of any arrived
unfinished request in `running` or `waiting`, not by the waiting head: a
future-arrival head MUST NOT hide later arrived work. The head-based check
described above is the current behavior at `69e2180` and is superseded.

For a nonempty arrived universe, successful integer-plan validation MUST be
followed by a precommit progress gate. If the plan contains no positive prefill
and no decode action, including an all-zero or preempt-only plan, Phase F
returns a structured failure with category `no_progress` and the snapshot ID,
with zero mutation. The top-level scheduler applies D-17: it raises its
dedicated failure exception and produces no `SchedulerOutputs`. It MUST NOT
retry unchanged state, force an unrelated action, invoke another policy, or
change the accepted mathematical layer to conceal the outcome.
*(amended 2026-10-12; implementation pending)* Progress now comes from the useful-execution row and the constructor
(§10); `no_progress` remains a defensive rejection of an erroneous empty or
preempt-only plan.

**Live scheduler integration (approved scoped inputs, 2026-10-05):** *(Describes
implemented behavior at `69e2180`; the configuration migration, idle
classification, and admission validation below amend it.)*
`sarathi/core/scheduler/lp_scheduler.py` defines `LPScheduler(BaseScheduler)`
and `LPSchedulingError`.

- *Configuration and entry point.* `LPSchedulerConfig` in `sarathi/config.py`
  (`SchedulerType.LP = 8`) requires explicit `max_num_seqs`, `max_model_len`,
  `num_pipeline_stages`, `b_max`, `c_max`, `s_max`, `memory_reserve`,
  `decode_memory_policy_id`, `decode_utility`, `prefill_token_utility`,
  `preemption_penalty`, and `numerical_policy`, with no defaults.
  `max_num_batched_tokens` returns `b_max`; `s_max` is separate from
  `max_num_seqs`. A stage count other than one raises `ValueError` at config
  construction. The entry point is direct configuration plus
  `SchedulerRegistry`; the same type is registered with the existing
  `VLLMBlockSpaceManager` (central and worker). Benchmark configuration and
  command-line parsing are unchanged.
- *Python `EngineArgs` entry point (2026-10-11).*
  `LLMEngine.from_engine_args(scheduler_type="lp", ...)` builds
  `LPSchedulerConfig` through `EngineArgs.create_engine_configs()` and the
  existing engine factory and registries. The nine fields `lp_b_max`,
  `lp_c_max`, `lp_s_max`, `lp_memory_reserve`, `lp_decode_memory_policy_id`,
  `lp_decode_utility`, `lp_prefill_token_utility`, `lp_preemption_penalty`,
  and `lp_numerical_policy` (an `lp_relaxation_scheduler.NumericalPolicy`,
  passed unchanged) are required, with no LP defaults. The resident limit,
  model length, and stage count are the inherited shared engine settings
  `max_num_seqs`, the model configuration's resolved maximum length, and
  `pipeline_parallel_size`, with their existing defaults; they are not LP
  research choices. Missing LP fields (checked with `is None`), a
  non-`NumericalPolicy` value, `pipeline_parallel_size != 1`, and any
  supplied `max_num_batched_tokens` (use `lp_b_max`) raise before model
  metadata loading and the `config.yml` write. Value-domain checks remain in
  the mapper and mathematical layer. Validated only by mocked CPU tests; real
  engine initialization through `EngineArgs` is unverified.
- *Scoped values.* The CPU plumbing fixture uses `b_max=8`, `c_max=4`,
  `s_max=3`, resident limit 4, `max_model_len=32`, block size 4, 10 blocks,
  reserve 1, `conservative_one_block_v1`, uniform utilities
  $\alpha=\beta=\gamma=1$, and the accepted numerical policy (§§10.1, 11.2).
  These are provisional plumbing inputs only; D-01–D-07, D-09, and D-10 remain
  OPEN. A positive penalty does not prevent a selected preemption; the
  executor originally rejected it and, since 2026-10-10, executes it under
  native preemption execution above. Resident and native memory gates may
  reject a mathematically valid plan.
- *Utilities.* For a non-idle decision the scheduler builds
  `Mapping[int, RequestUtility]` with the configured uniform triple for every
  owned request with `arrival_time <= now` that is not finished. There is no
  per-request policy storage or callback. The mapper still validates IDs,
  arrival, ownership, and the exact key set. If building the mapping raises
  `AttributeError`/`TypeError`, the scheduler raises `LPSchedulingError` with
  an `lp_relaxation_scheduler.Failure` (stage `utility_construction`).
  *(Amended 2026-10-12:* coefficients and ranks are built from the single
  capture in the mapping responsibility, §12.3.)
- *Decision flow.* `schedule()` first rejects a stage count other than one or
  a nonzero `num_running_batches` by raising `LPSchedulingError` (stage
  `scheduler_entry`) *before* calling `BaseScheduler.schedule()`, so
  `_iteration_id` is unchanged and no empty output is returned. Otherwise the
  inherited method advances `_iteration_id` exactly once and calls
  `_schedule()`, which reads `time.monotonic()` once. It then returns the D-18
  idle output, or calls `map_scheduler_state`, `solve_and_extract`, and
  `execute_plan` in sequence, each only after the previous stage succeeded.
  For supported calls the single increment also applies to ordinary idle and to
  pre-mutation failures. Nothing is retained between decisions.
- *Failure conversion (D-17).* A returned `MappingFailure` or
  `lp_relaxation_scheduler.Failure` from mapping, solving/extraction, or
  executor precommit validation is raised as `LPSchedulingError`, which
  carries that frozen record unchanged (`snapshot_id`, `stage`, `category`,
  `reason`, and `solver_diagnostics` read from it). The scheduler has no broad
  exception handler: exceptions after executor mutation begins propagate
  unchanged (D-20).
- *Responsibilities.* The scheduler owns entry checks, the clock read, idle
  handling, utility construction, stage sequencing, and exception conversion.
  The executor owns precommit validation, prompt-first order (D-21), native
  mutation, and output construction; the scheduler does not reorder output.
  `BaseScheduler` owns `_iteration_id`, `num_running_batches`, and
  completion bookkeeping.

**Configuration migration (2026-10-12; D-30; implementation pending).** Exact
decode demand is mandatory, and $h$ is read from the shared block manager. The
`memory_reserve` and `decode_memory_policy_id` fields of `LPSchedulerConfig` and
the `lp_memory_reserve` and `lp_decode_memory_policy_id` `EngineArgs` fields are
removed from the active interface. Supplying any obsolete argument, including
an explicit zero reserve, MUST raise an explicit error that explains the
migration. Old values MUST NOT be ignored, reinterpreted, or translated into
$h$, and no compatibility mode is added. Explicit token, chunk, and action
capacities and utilities remain required, and unrelated `EngineArgs` and
direct-construction behavior is preserved. Historical scripts and records keep
their version-specific configuration and are not rewritten. The configuration
descriptions and scoped values above describe the interface at `69e2180`.

**Initial request-admission validation (approved R13; D-29; implementation
pending).** For LP scheduling only, a new request is checked after
tokenization and `Sequence` construction (to reuse the allocator's initial-size
calculation) but before registration with the engine sequence manager, workers,
scheduler, or arrival metrics. It is rejected with an explicit request error
when its context length exceeds `max_model_len`, or when its full initial
allocation demand $A_i$ plus $h$ exceeds the total pool $T$. The error carries
the request identity where available, the reason, the required length or
blocks, and the configured limit. Current free-memory contention ($A_i>F$) is
never treated as permanent impossibility. No maximum future output is
reserved, and passing these checks does not guarantee completion. A rejected
request leaves no engine, worker, scheduler, or metric registration; an unused
sequence-ID counter value is not partial registration. An already registered or
recomputation-expanded context that violates these limits uses the existing
structured pre-mutation fail-stop path (D-17). No request cancellation, cleanup
protocol, or ignore-only output is added, and other schedulers' submission
behavior is unchanged. At `69e2180`, `BaseLLMEngine.add_request` registers with
the sequence manager, workers, scheduler, and metrics without this check.

## 14. Failure contracts

### 14.1 Layer-local behavior that is resolved

| Failure location | Required layer-local behavior |
|---|---|
| Invalid Phase D input | Return explicit failure; do not call solver |
| Solver-reported LP infeasibility (SciPy status 2) | Return the distinct non-success outcome with raw status and diagnostics; do not extract; not an exact certificate (§9.3; amended) |
| Solver unbounded, numerical error, limit, malformed output, or exception | Return explicit solver/numerical non-success; do not extract |
| Invalid relaxed solution | Return explicit non-success; do not extract |
| Constructor finds no legal useful integer work | Return the distinct exhaustive no-useful-work outcome; no plan (§10.5; amended) |
| Integer plan violates an invariant | Return explicit validation failure; do not invoke Phase F; this proves no impossibility |
| New LP request is provably inadmissible (R13) | Reject the submission before registration with an explicit request error (§13; amended); the LP path is not entered |
| Phase E inconsistent snapshot | Return explicit mapping failure; no mutation |
| Phase F problem/plan identity mismatch or failed current-state prevalidation | Return explicit precommit failure; perform zero mutations |

No layer may convert these outcomes into invented relaxed values, an all-zero plan, a baseline-policy action, or a partially accepted plan.

### 14.2 Top-level pre-mutation failure policy

For the MVP, every explicit mapping, solver, relaxed-validation, extraction,
integer-plan-validation, or fresh-prevalidation failure before mutation is
fail-stop. The responsible layer first returns its immutable structured failure
without a plan. The live LP scheduler then raises a dedicated scheduler
exception that carries that failure's snapshot/problem ID, stage, category,
reason, and available solver diagnostics. It MUST NOT retain a mutable LPServe
object in the exception.

The failing call produces no `SchedulerOutputs` and ends the current run. It
MUST NOT substitute an empty output or all-zero plan, invoke another scheduling
policy, retry automatically, patch the failed result, or begin execution.
Returning an empty output could leave unfinished requests in the engine loop
indefinitely; retrying unchanged state would generally repeat the same failure;
and an alternate policy would conceal failure of the LP path and confound its
experimental results. A later attempt requires an explicit new invocation
after the caller has intervened or supplied different state or policy inputs.

This fail-stop rule is the complete D-17 response for failures detected before
mutation. A valid plan that selects no work is governed separately by D-18.
Failure after any mutation begins is governed separately by D-20 and MUST NOT
be treated as safely covered by this pre-mutation rule.

For Sections 14.2--14.4, the execution-mutation boundary means the first
change to request ownership, sequence status or progress, block state, or
running-batch state after successful prevalidation. The inherited
`BaseScheduler.schedule()` increment of `_iteration_id` is bookkeeping before
that boundary. An idle or fail-stop call may therefore leave the iteration
counter advanced without implying that execution mutation began or that D-20
applies.

D-18 distinguishes ordinary idle from failure. Future-only or completely empty
scheduling is the normal idle case defined in Section 13: no arrived unfinished
request exists (amended 2026-10-12 from the earlier waiting-head rule). The live scheduler returns an
empty `SchedulerOutputs` for it without invoking mapping or the LP path. Once
the arrived universe is nonempty, a validated plan with no prefill or decode
action is a `no_progress` precommit failure and follows D-17. An empty output
MUST NOT be used to represent or suppress that failure.

*(amended 2026-10-12; implementation pending)* The required outcome distinctions are: ordinary idle (no arrived
unfinished work; no LP path is entered); initial admission impossibility (a
request error before registration); solver-reported LP infeasibility;
exhaustive absence of legal useful integer work; solver or numerical failure;
and implementation or validation failure, including failed precommit
validation. Only ordinary idle returns output. The initial request error
rejects only that submission; every other listed pre-mutation outcome follows
D-17. A failed validation never proves impossibility. A post-mutation exception
follows D-20: it terminates the run, and the affected engine is discarded.

### 14.3 Precommit physical failure

If fresh physical prevalidation fails, Phase F must perform no mutation and
return an explicit precommit failure. The live LP scheduler then applies the
D-17 fail-stop rule in Section 14.2.

### 14.4 Failure after mutation begins

The audited framework has no transaction, reservation object, undo log, or rollback spanning scheduler collections, the central allocator, engine state, and worker allocators. Allocation can also fail after partially popping blocks from its free list.

For the single-stage MVP, D-20 deliberately inherits the existing schedulers'
fail-stop behavior rather than adding a transaction or recovery subsystem.
Phase F performs complete fresh prevalidation before its first mutation. If an
exception occurs after that first mutation, it propagates and terminates the
current run. The scheduler, engine, sequence-manager, and worker state MUST be
treated as potentially divergent and MUST NOT be reused. The implementation
MUST NOT catch the exception and continue, retry the plan, invoke another
policy, return partial `SchedulerOutputs`, or claim rollback, atomicity,
recovery, or safe continuation.

Known deterministic incompatibilities are not deferred to this destructive
failure rule. In particular, any plan that would create a control-only output
with no scheduled metadata, including a preempt-only plan, MUST fail precommit
validation with zero mutation under D-17. The executor MUST NOT manufacture an
unrelated scheduled action to avoid the framework defect. Pipeline execution
remains outside the supported contract under D-24.

### 14.5 Observability

Every solver attempt SHOULD retain, without fabricating actions: normalized
outcome; phase/stage/reason; raw status, success, non-contractual message, and
vector presence/shape; solver identity/version/method/options and deliberate
absence of limits; available HiGHS, `nit`, and `crossover_nit` data; reported,
raw-recomputed, and normalized objectives; problem/snapshot ID; validator and
extraction entry/outcome; numerical policy, tolerances, projection count,
worst raw/normalized violation, and the extraction data of §10.8 (amended);
selected IDs, capacity summary, violated invariant or exhausted candidate set;
and whether mutation began. For a limit, diagnostics MUST NOT infer time versus
iteration cause from message text alone.

Trace output MAY expose LP indicator values; indicator classification is
retired (§10). Full vectors and per-coordinate projection detail MAY remain debug artifacts.
Routine diagnostics MUST retain validation stage and rejection reason.
Experiment summaries distinguish policy decisions from solver, extraction,
validation, and execution failures.

## 15. Testing and phase acceptance requirements

No GPU experiment substitutes for mathematical or state-transition tests. Each phase must satisfy its own gate before the next mutation boundary is enabled.

*(amended 2026-10-12)* §§15.1–15.6 record the earlier milestone gates and their
test traceability at their revisions. The corrected framework additionally
requires §15.7. Earlier checks of superseded behavior (for example,
`order_key`-only extraction order, deterministic ties under the old
extraction, or prompt-first order without demand groups) are historical and
yield to §15.7 where they conflict.

### 15.1 Phase D problem and solver tests

Phase D needs focused synthetic evidence for the supported MVP path, not an
exhaustive matrix of solver or inherited-framework edge cases. The minimum set
is:

1. a hard-coded `main()` smoke case that builds, solves, extracts, and prints
   one plan, proving the module can run independently of LPServe and a GPU;
2. one feasible mixed case containing prefill, decode, preemption, capacity
   limits, and an ordering tie, proving the intended end-to-end mathematical
   path and deterministic extraction;
3. one case with a fractional prefill decision, proving $x_i$ is continuous in
   the relaxation and that extraction returns a feasible integer plan; and
4. one infeasible or unusable solver-result case, proving the layer returns a
   visible failure rather than fabricating a schedule or falling back silently.

The implementation must still honor the Phase D contracts in §§9--11: explicit
SciPy/HiGHS configuration (D-11--D-13), input/result validation, optimal-only
admission (D-12), the selected numerical tolerances (D-14--D-15), and the
canonical order key (D-16). Add a direct focused regression only when a
currently supported behavior breaks. Broader numerical-boundary, degeneracy,
diagnostic, and exhaustive model-constraint matrices are deferred until they
block the MVP or a later phase needs them.

### 15.2 Extraction tests

The mixed and fractional cases in §15.1 are the initial extraction checks. They
must show that extraction uses the documented canonical order, respects the
supplied capacities and legal preemption set, and validates the returned
integer plan. A small `main()` is a smoke check only; it does not replace these
two behavior checks.

Do not add separate tests for every tie, numerical boundary, packing variant,
or unsupported preemption edge case at this stage. Add one focused regression
when a supported extraction behavior fails; otherwise defer that expansion
until it blocks the MVP or is needed by a later phase.

### 15.3 Phase E mapping tests

When Phase E begins, use one real or faithful LPServe `Sequence` and
block-manager case to prove that the supported request data and capacity inputs
produce a coherent Phase D snapshot without mutating LPServe state. The case
must make its request-ID normalization, `order_key`, prefill/decode eligibility,
and block-charge inputs visible.

Defer exhaustive status/ownership variants, duplicate-conflict handling,
alternative decode-charge policies, stale-snapshot cases, and inherited
preemption edge cases until they block the supported MVP path or Phase F needs
them. Phase E remains read-only regardless of the size of its initial test set.

### 15.4 Phase F execution tests

The live scheduler integration check MUST verify the supported D-19 path:
mapping through execution occurs synchronously in one scheduler decision call,
with one pipeline stage and zero running batches before execution mutation.
A separate old-versus-current snapshot comparison is not required.

Before GPU validation, use one focused synthetic scheduler-state case to prove
the smallest supported native execution path: a validated plan admits and
executes work, emits scheduled/control entries matching the validated plan,
respects the resident limit, and reports action/token counts derived from the
emitted entries. In that same case, verify complete before-and-after queue
membership and allocation, no request loss or duplication, expected free-block
changes, and unchanged unrelated requests included in the fixture. These are
test assertions, not a production post-execution state-auditing subsystem.
Use one physically infeasible plan to prove rejection occurs visibly
before scheduler or block-manager mutation, raises the D-17 top-level scheduler
exception, and produces no `SchedulerOutputs`.

Use one direct-mapper future-only case to prove an empty arrived universe
returns a `MappingFailure` without constructing an `LPProblem`, calling
validation, or mutating state. The live future-only empty-output case, which
proves the scheduler returns an empty `SchedulerOutputs` without invoking
mapping, solving, extraction, or execution, belongs to the integration tests of
the live `LPScheduler` and is not part of this mapper test. That integration
test MUST also prove that waiting/running ownership, sequence status and
progress, block state, and `num_running_batches` are unchanged while the
inherited `_iteration_id` advances once and supplies the returned output ID.
Use one arrived unfinished case
with a validated all-zero plan to prove it is rejected as `no_progress` before
mutation. Use one focused control-only case to prove a preempt-only plan reaches
the same progress gate. A larger empty-plan matrix is not required. Use one
focused synthetic post-mutation failure, induced by a test double after the
first native state change and without adding a production fault-injection
mechanism, to prove the exception propagates, no later planned action or
`SchedulerOutputs` is produced, and the mutated fixture is discarded rather
than reused. This test does not require or imply rollback.

Live scheduler integration traceability: `tests/test_lp_scheduler.py` covers
registration and explicit configuration, multi-stage rejection, empty and
future-only idle, the real synchronous pipeline with real native replay
(admission → partial prefill → prompt completion → decode at gap 0 and 1 →
finish/free with matching central/worker block tables), a mixed prompt-first
decision with an unselected resident, real mapping, mathematical, physical
(resident capacity), and `no_progress` failures raised as `LPSchedulingError`,
the unsupported-entry guard, and a post-mutation failure that propagates.

Native preemption traceability (2026-10-10):
`tests/test_lpserve_plan_execution.py` covers recovery before the admission
it enables, multiple victims in ascending call/control order with reversed
waiting-front order, zero-mutation rejection of a later victim, of malformed
recovery, and of an admission that fails after full recovery, control-only
`no_progress` with no `_preempt` call, and an exception after a native
preemption. `tests/test_lp_scheduler.py`
(`LivePreemptionTest.test_solver_selected_preemption_is_recomputed_to_completion`)
covers the solver-selected live case through central/worker replay,
recomputation, completion, and idle.
`LiveDecodePreemptionTest.test_decode_preemption_recomputes_expanded_context_to_completion`
covers the same path for a victim with one generated token, including the
expanded prompt, cumulative token history, and per-epoch output lists. The
real-weight GPU comparison is
`scripts/check_lp_scheduler_decode_preemption_reference_gpu.py`, with CPU
tests in `tests/test_check_lp_scheduler_decode_preemption_reference_gpu.py`.

Add direct focused regressions only when a currently supported action breaks.
Preemption, mixed batches, decode marginal-block variants, worker
block-table equality, and ownership corner cases are not mandatory initial
coverage. Known inherited framework limitations remain documented and deferred
unless they block the supported MVP path; this section does not require their
repair before a basic implementation runs.

### 15.5 Integrated validation sequence

After the applicable focused Phase D--F checks pass:

1. run syntax, import, and static checks;
2. run the focused mathematical, mapping, and executor checks for the supported
   path;
3. run one tiny single-stage GPU smoke test with the smallest supported action
   set; and
4. inspect the resulting LP inputs, extracted decisions, queue transitions, and
   block deltas.

Native preemption exercises, tiny same-framework comparisons, additional
action combinations, and broader timing or performance experiments are useful
follow-up work once the basic path runs; they are not prerequisites for the
initial MVP. Retain correctness evidence before treating any broader experiment
as informative.

Aggregate throughput or latency cannot establish scheduler correctness.

### 15.6 Acceptance evidence

Each phase handoff must record:

- a reference to provenance-bearing run evidence (machine-generated records
  of the tested revision, working-tree state, and file hashes);
- environment and dependency changes;
- tests run and their observed output;
- behaviors actually verified;
- unresolved failures and skipped tests;
- newly selected research policies;
- next permitted phase.

The absence of a failure in a GPU run is not evidence that an unexercised invariant holds.

### 15.7 Corrected-framework CPU milestone checks

*(added 2026-10-12 from P10 step 9; none has been run.)* These checks are
required before the corrected CPU milestone is claimed. They are future
requirements, not evidence.

- *Pure mathematical:* each amended row; $h=0$ and $h>0$; multiple admissions
  sharing one watermark; omitted residents; zero free blocks with a zero-demand
  decode; mixed decode demand; necessary negative-utility first work;
  zero-LP-support execution and integral-idle victims; no legal useful plan;
  the zero- and positive-coefficient mass-preemption examples; and
  strict-improvement values on both sides of $\tau$. Do not require a
  particular optimizer tie solution unless the fixture supplies the relaxed
  vector.
- *Independent enumeration on bounded synthetic states, in test code only:*
  enumerate whole actions, chunks, and victims to decide whether a legal useful
  plan exists; compare with the constructor; independently validate every
  returned plan; include LP-feasible but integer-infeasible cases and the
  many-fractional watermark construction. This is test evidence, not a runtime
  exact-ILP fallback or a proof.
- *Fixture domains:* label each fixture as either an abstract formulation
  fixture or an admission-validated runtime fixture. The manuscript's
  $F=3$, $h=2$ LP-feasible but integer-infeasible example lies outside the
  admission-validated domain: its pool of three blocks fails
  $A_i+h\le T$. It may test the abstract formulation and the constructor, but
  MUST NOT be presented as a validated runtime no-useful-work example. These
  checks do not require constructing a validated runtime state that is
  LP-feasible yet integer-infeasible; whether one exists is not established.
- *Coefficients and ranks:* state-dependent coefficients and ranks change
  between decisions as state changes and stay fixed within a decision
  through execution; explicit uniform inputs are preserved; negative or
  nonfinite coefficients are rejected; and coefficient and rank
  construction mutates no live state and makes no second capture.
- *Mapper and allocation:* full-context rounding at block boundaries; zero
  resident-prefill charge; decode gaps of zero and one, and unsupported gaps;
  exact recovery; watermark and resident limits; malformed allocation state;
  ranks, including omitted residents and arrived work behind a future head;
  and no mutation during snapshot and policy-input construction.
- *Executor and queues:* prefills followed by both decode groups; one free
  block, where A consumes it and B needs none; identical central and metadata
  order; multiple victim reinsertion; retained omitted residents; readmission
  to the running tail; and a future head followed by arrived work.
- *Admission, failure, and integration:* length and pool equality boundaries;
  a request that fits the whole pool but not current free memory; no
  registration on initial rejection; a recomputation-expanded invalid context;
  the defensive `no_progress` guard; induced pre-mutation failures with zero
  execution mutation; post-mutation stop behavior; rejection of obsolete
  configuration arguments; and the existing admission, partial-prefill, decode,
  mixed, recomputation, and `EngineArgs` paths.

Record the tested revision or exact diff and hashes, environment, commands, and
observed results, including failed, skipped, and unexecuted checks. Classify an
inherited failure and explain its effect rather than relabeling it a pass. A
GPU regression requires separate authorization after these CPU checks pass;
recovery and performance experiments remain deferred under Q11 (research
context §21).

## 16. Known inherited framework limitations in MVP compatibility mode

### 16.1 Recomputation generation-limit semantics

`reset_for_recompute()` moves existing generated token IDs into the prompt context and clears `output_token_ids`. The current length cap uses the cleared list rather than the cumulative output-token count. A preempted request can therefore generate beyond its requested `max_tokens` across restarts.

The model's causal token context is preserved, but user-visible generation semantics are not. The MVP inherits this behavior when it uses native recomputation preemption; it does not repair it or claim corrected generation semantics.

### 16.2 Recomputation `RequestOutput` semantics

After recomputation, the audited `RequestOutput` can pair:

- the original prompt string with expanded prompt token IDs; and
- cumulative output text with only post-restart output token IDs.

The MVP inherits this behavior when it uses native recomputation preemption. It does not repair the external contract or use affected results to claim corrected output semantics.

### 16.3 Control-only scheduler outputs

A preempt-only or ignore-only output has no scheduled metadata. The single-stage engine drops such an output before replay, while the pipeline path can enqueue control state and then wait for model output from a batch never sent to workers.

The former formulation could legitimately produce a preempt-only plan; the
amended formulation requires useful execution, so such a plan now indicates an
error (amended 2026-10-12). In MVP compatibility mode,
the executor does not repair this inherited behavior or force an unrelated
scheduled action merely to avoid it. It rejects any control-only plan during
precommit validation with zero mutation. Supporting control-only execution
later requires an explicit engine/replay repair and focused tests.

### 16.4 Mixed-batch and sampler association

The model runner physically packs prompts before decodes, whereas existing scheduler metadata may be decode-first or interleaved. Sampling-type grouping can also select incorrect tensor rows, and completion uses positional `zip` without checking result IDs or lengths.

The LP executor preserves its selected replay order but does not repair the
underlying framework behavior. Affected cases are documented limitations and do
not establish corrected sampler semantics.

### 16.5 Non-transactional mutation

Scheduler queues and central blocks mutate before forward execution, workers
replay later, and no cross-layer rollback exists. Prevalidation remains part
of the selected path, but the MVP adds no rollback or recovery. Under D-20, an
exception after mutation begins terminates the run and the affected engine
state is not reused. The MVP must not claim atomic commit or recoverable
failure.

### 16.6 Pipeline-parallel support — deferred from the initial MVP

**Approved future amendment (2026-10-18):** §19 selects the extension
contract and supersedes the former unresolved design requirements below
for that future scope. In particular, it inherits native in-flight
preemption rather than requiring a new proof of physical cache release.
The historical framework observations remain limitations, and current
single-stage code is not made pipeline-capable by this record.

Pipeline execution permits multiple microbatches in flight. The audited framework anticipates preemption of a request while an older batch is executing, but exposes no safe physical-release predicate, per-batch KV version, or exact completion association sufficient for this design.

Existing schedulers rely on the running-batch counter, lifecycle status, replay
order, and skipping some outputs for requests that have since returned to
`WAITING`; they do not provide scheduler-state locking, block reservations, or
versioned completion association. Their pipeline path also has the documented
control-only-output failure. This is best-effort inherited behavior, not a
verified concurrency contract.

The LP scheduler adds a multi-field snapshot and a solve interval during which
pipeline completion could change ownership, progress, or blocks. The former
formulation could also legitimately produce a preempt-only plan (historical
rationale; the amended formulation requires useful execution, but the inherited
control-only defect of §16.3 remains). The approved replacement contract in §19 requires coherent capture and
commit, explicit native readiness and victim sets, inherited dispatch and
completion association, continued rejection/reporting of unsupported
control-only outputs, and focused replay tests. It does not require a new
cache-version or physical-release subsystem.

The existing single-stage specification and code do not enable pipeline-parallel LP execution. The questions formerly deferred here, now addressed by §19, were:

- which in-flight requests belong to $\mathcal Z_t$;
- which actions must be frozen across outstanding batches;
- when freed blocks are safe to reuse;
- how a plan is associated with a later completion callback;
- how failure affects `num_running_batches` and worker queues.

Earlier correctness evidence is limited to its recorded single-stage scope.
§19 is the approved design amendment; its implementation, mathematical
alignment, and new architecture/GPU checks remain prerequisites to a
parallelism support claim. Changing only `num_pipeline_stages` is insufficient.

### 16.7 Incremental text conversion can include the prompt's final token

**Confirmed inherited defect (2026-10-06).** When the first generated token
is skipped as a special token, `detokenize_incrementally` can treat the
prompt's final token as newly generated text. For the raw prompt
"Explain why plants need sunlight.", both the recorded LP run and the
single-request `VLLMScheduler` reference saved a leading period from the
prompt. CPU replay of `EngineSequenceManager._decode_seq`'s native text
conversion reproduced that period exactly. The generated token IDs did not
contain a period; native text and decoding only the generated IDs differed.

Evidence: [generation diagnostic handoff](../validation_output/lp_inference_generation_debug/20261006T072447Z/HANDOFF.md)
and its retained reference summary and trace. The diagnostic ran at
`c7ebebd` and was reviewed at `0557e30`; the affected tokenizer and
engine-sequence-manager code is unchanged from audit baseline `c3e0143`.
This later observation does not repin or rewrite the historical audit.

Under D-25, the restricted MVP inherits this defect without repairing it.
Saved native text MUST NOT be presented as proof of correct conversion of
the generated token IDs.

### 16.8 EOS-first generation on raw text: cause unverified

In the same evidence, TinyLlama/TinyLlama-1.1B-Chat-v1.0, with real weights,
greedy sampling, four requested output tokens, and no chat template,
generated `[2, 29871, 13, 29966]`. The first generated token, 2, is EOS
(the end-of-sequence signal). `ignore_eos=True` deliberately continued
execution after that signal. The reference reproduced the recorded LP
result for this one prompt; the conditional LP-alone run was not needed.

Why EOS was selected first is unknown. Missing chat formatting remains an
inspection-based hypothesis, not a tested cause. Reference agreement does
not prove correct generation or exclude a defect shared by both paths.
This observation is separate from the confirmed text-conversion defect in
§16.7 and is not classified as a proven LP-scheduler defect.

Deeper EOS diagnosis and text-conversion repair are deferred while the
restricted scheduler MVP proceeds. Revisit these limitations when useful
text answers are required or an unexplained LP/reference token mismatch
appears on a supported workload. This deferral establishes no claim about
generation quality and changes no mathematical, execution, or failure contract.

## 17. Decision register

| ID | Status and selected rule | Canonical requirements |
|---|---|---|
| D-08 | **RESOLVED for the single-stage MVP** — the LP scheduler inherits `BaseScheduler` directly. Inherited `waiting` exclusively owns unallocated unfinished requests; inherited `running` exclusively owns all allocated resident unfinished requests, including residents not selected for the next forward pass. Auxiliary structures are non-owning indexes only. Mapping occurs at the quiescent pre-mutation boundary with no batch in flight. This avoids SLAI-specific ownership and preserves base unfinished-request and completion behavior while making ownership and allocation cross-checkable. | §§4.1–4.4, 6.4, 12.2–12.5; focused test §15.3; D-24 and §19 govern the future extension |
| D-11 | **RESOLVED** — `scipy.optimize.linprog`/HiGHS behind a project-owned adapter; SciPy must become an explicit reproducible dependency before Phase D execution, and its exact version is not selected here. | §9.3; tests §15.1; diagnostics §14.5 |
| D-09 | **RESOLVED 2026-10-12 (R07); implementation pending** — $c_i^D$ is the exact logical/physical gap in $\{0,1\}$; another gap is an unsupported-state mapping error. The free-block-before-decode gate is the decode prerequisite row, and zero-demand decodes execute before one-block decodes. *Formerly OPEN* between the exact gap and the conservative one-block charge (`conservative_one_block_v1`, retired for the corrected path). | §§7.1, 8.3, 12.3, 13; §15.7 |
| D-10 | **RESOLVED 2026-10-12 (R10, R12); implementation pending** — the independent planning reserve $W_t$ is removed. Its intended protection is replaced by the admission-watermark rows $H\ge hI_i^P$ ($i\in\mathcal Q_t$), with $h$ read from the shared block manager and the rows omitted when $h=0$. No final-state reserve is imposed. *Formerly OPEN.* | §§5.3, 7.1, 8.1, 12.3; D-30 |
| D-12 | **RESOLVED** — optimal-only: only normalized SciPy `status == 0` may enter validation; only its validated candidate may extract. *Amended 2026-10-12 (R21):* status 2 is reported as solver-reported LP infeasibility, a distinct non-success outcome that retains diagnostics and is not an exact certificate. | §9.3, §11.2, §14.1, §15.1 |
| D-13 | **PARTIAL** — reference: `highs-ds`, explicit `presolve=True`, no project `time_limit`/`maxiter`, and no explicit crossover or undocumented thread/parallel/random-seed controls. Performance/control/basis remainder is **OPEN**. | §9.3; §10.8; §15.1–15.2 |
| D-14 | **SUPERSEDED for extraction 2026-10-12** — $\varepsilon_{\mathrm{int}}=10^{-6}$ remains the configured value but has no classification, locking, or decision role in the corrected path; any remaining use is diagnostic only. No classification subsystem is added. *Formerly RESOLVED:* absolute $\varepsilon_{\mathrm{int}}$ for validated indicators only. | §§10.4, 10.8 |
| D-15 | **RESOLVED** — absolute $\varepsilon_{\mathrm{feas}}=10^{-7}$, independent validation, narrow projection, and revalidation only. *Amended 2026-10-12:* validation covers every amended row; the prospective-locking compatibility check is removed; tolerance and projection rules are unchanged. | §11.2; §15.1 |
| D-16 | **AMENDED 2026-10-12 (R06, R16); implementation pending** — `order_key` remains the stable identity, canonical variable order, and final tie-break. Policy preference uses immutable per-decision service and victim ranks (smaller first): execution candidates by descending LP execution value, service rank, identity; victims by descending LP preemption value, victim rank, identity; emitted victims by victim rank, identity; execution groups by service rank, identity. Initial vLLM-oriented profile: service = current running order then arrived waiting order; victim = reverse running order. *Formerly:* ascending immutable lexicographic `order_key` with exact extraction ties. The current engine's unique non-negative integer `Sequence.seq_id` is its verified monotone admission identity; Phase E uses `order_key=(raw_seq_id,)` and the accepted decimal-string request ID. Unsupported live ID shapes fail mapping. Revisit this rule if ID generation changes or one scheduling universe spans independent generators. | §§9–10, 12.3; tests §§15.2–15.3 |
| D-17 | **RESOLVED for the MVP** — every mapping, solver, relaxed-validation, extraction, integer-plan-validation, or fresh-prevalidation failure detected before mutation is fail-stop. The layer returns its immutable structured failure without a plan; the live LP scheduler raises a dedicated exception carrying its immutable diagnostics, produces no `SchedulerOutputs`, and ends the run. It performs no automatic retry or re-planning, empty/all-zero substitution, alternate-policy fallback, patching, or execution. D-18 governs a valid empty plan; D-20 governs failures after mutation begins. Implemented by `LPSchedulingError` (§13, live scheduler integration). | §§13, 14.1–14.3; focused tests §15.4 |
| D-18 | **RESOLVED for the single-stage MVP** — ordinary future-only or completely empty scheduling is handled by the live `LPScheduler` before mapper invocation: when `running` is empty and `waiting` is empty or its head has `arrival_time > now`, the scheduler returns an ordinary empty `SchedulerOutputs` without mapping, LP construction, solving, extraction, precommit validation, or execution. This inherits the audited schedulers' waiting-order and future-request-validity assumptions and adds no separate validation subsystem. It also inherits `BaseScheduler.schedule()` bookkeeping: `_iteration_id` advances and supplies the output ID, while waiting/running ownership, sequence status and progress, block state, and `num_running_batches` remain unchanged. The mapper accepts only a nonempty arrived universe; an empty arrived universe reaching direct mapping is a `MappingFailure`. Future requests may still be filtered from a mixed nonempty snapshot. *Amended 2026-10-12 (R14):* idle is classified by the absence of any arrived unfinished request, not by the waiting head; useful execution is an LP row (D-26), and `no_progress` is a defensive guard. If the arrived universe is nonempty and a successfully validated plan contains no positive prefill or decode action, including an all-zero or preempt-only plan, precommit validation returns `no_progress` before execution mutation and the top-level scheduler applies D-17. No retry, forced action, alternate policy, or mathematical-layer change is added. | §§12.1, 13, 14.2, 15.4; D-20 covers failures after execution mutation |
| D-19 | **RESOLVED for the single-stage MVP; simplification approved 2026-10-04** — the same engine receives no overlapping state-changing public calls; mapping begins with one pipeline stage and zero running batches; and mapping through execution is synchronous within one scheduler decision call. This contract supplies state stability. The snapshot ID (the scheduler iteration number, §9.2) associates the problem and plan but is not a lock or proof of unchanged state. Phase F checks problem/plan identity and fully validates the complete plan against current state immediately before mutation, including combined native memory feasibility in execution order. It does not separately compare live fields with the earlier snapshot or rebuild the mapper for comparison. Identity mismatch or failed prevalidation is a zero-mutation precommit failure followed by the D-17 fail-stop response. The separate comparison's detection of unexpected contract-violating state changes is relinquished. No new lock, version, reservation, or layer-local retry subsystem is required. Pipeline mode and concurrent public calls remain unsupported. | §§12.2, 12.6, 13, 14.2–14.3; focused tests §§15.3–15.4; D-24 and §19 govern the future extension |
| D-20 | **RESOLVED for the single-stage MVP** — Phase F fully prevalidates before mutation and then uses native LPServe operations. Any exception after the first mutation propagates and terminates the run; potentially divergent scheduler, engine, sequence-manager, and worker state is not reused. The MVP adds no rollback, retry, alternate-policy fallback, partial output, recovery, or transaction mechanism. Known control-only output shapes, including preempt-only plans, are rejected before mutation under D-17 rather than sent through this destructive path. | §§13, 14.4, 16.3, 16.5; focused tests §15.4; D-24 and §19 govern the future extension |
| D-21 | **RESOLVED for the single-stage MVP (approved 2026-10-05)** — execute and emit all selected prefills first, then all selected decodes, each ascending by existing `order_key`; central native operations and emitted metadata share this order. Built as a separate ordered action list; the accepted plan, its selected actions, and chunk sizes are unchanged. Rationale: matches prompt-first physical input packing and gives a deterministic within-group order. Inherited mixed-batch/sampler limitations remain under §16.4 and D-25. *Extended 2026-10-10:* all selected victims are preempted before any scheduled action, ascending by `order_key`, and `preempted_seq_ids` uses that order; native front insertion leaves them in `waiting` in reverse call order (§13, native preemption execution). *Amended 2026-10-12 (R07, R16); implementation pending:* victims by victim rank, then identity; then prefills, zero-demand decodes, and one-block decodes, each by service rank, then identity; admissions appended in that order; retained residents keep their relative positions (§13, amended execution order). | §13; focused test `tests/test_lpserve_plan_execution.py` (`test_mixed_output_is_prompt_first_in_native_execution_order`) |
| D-26 | **RESOLVED 2026-10-12 (R01, R14); implementation pending** — the LP includes $\sum_i(I_i^P+y_i)\ge1$ for every validated nonempty arrived universe; preemption alone does not count; a state without a legal useful integer plan is reported, not given an invented action. Admission validity does not guarantee completion. | §§7.1, 10, 14.2 |
| D-27 | **RESOLVED 2026-10-12 (R08); implementation pending** — resident capacity is an LP row counting all retained residents, including omitted ones; the precommit resident check remains. | §§6.4, 7.1, 13 |
| D-28 | **RESOLVED 2026-10-12 (R15, R16, P10, user clarification under R15); implementation pending** — one-pass LP-guided whole-action constructor without an integral fast path; first legal action accepted regardless of utility; strict improvement for later additions with new victims, guarded numerically by $\tau$ from the existing objective tolerances; the two-pass alternative is not adopted; limitations of §10.6 accepted. | §10; §15.7 |
| D-29 | **RESOLVED 2026-10-12 (R13); implementation pending** — LP-only rejection of provably inadmissible new requests before engine, worker, scheduler, or metric registration; registered or recomputation-expanded violations keep pre-mutation fail-stop. | §§13, 14.1 |
| D-30 | **RESOLVED 2026-10-12 (P10 step 8); implementation pending** — remove reserve and decode-policy configuration fields and their `lp_*` `EngineArgs` counterparts; reject obsolete supplied arguments, including an explicit zero reserve, with a migration error; no compatibility translation. | §13 |
| D-31 | **RESOLVED 2026-10-12 (user decision, R21); implementation pending** — SciPy/HiGHS status 2 is solver-reported LP infeasibility with raw status and diagnostics retained; it is not an exact certificate and is distinct from exhaustive no-useful-work; no certificate checker or extra solve. | §§9.3, 10.5, 14.1 |
| D-25 | **RESOLVED** — MVP compatibility mode reuses existing LPServe/SLAI behavior, including native recomputation preemption, without repairing inherited framework defects. Material inherited limitations are documented; only issues that block the selected MVP path require action. | §§2.4, 4.4, 13, 16 |

| D-24 | **DESIGN RESOLVED 2026-10-18; implementation and validation pending** — fixed TP, overlapping PP, and combined TP+PP for LP and the adapted-vLLM reference; the inherited execution and preemption mechanisms remain the compatibility basis. No blanket ban on in-flight victims and no stronger LP-only preemption guarantees. Formerly OPEN; source-sensitive mechanics must be checked during implementation. | §19; especially §§19.1–19.4, 19.9 |
| D-32 | **APPROVED for the future extension** — brief shared controller protection for coherent capture and validation/commit; planning on immutable inputs outside that protection. One scheduling decision in progress per engine. No GPU drain, solver-duration lock, or independent competing allocation decision. | §19.3 |
| D-33 | **APPROVED for the future extension** — keep all residents, including outstanding requests, in ownership and resource accounting; execution candidates must be ready at capture; victim eligibility follows the inherited contract and can include in-flight residents. | §§19.4–19.5 |
| D-34 | **APPROVED for the future extension** — tolerate completion and registration during planning, accept older feasible plans, and omit only confirmed completed-and-released victims during fresh commit validation. Preserve the original mathematical proposal separately from actual execution. | §19.6 |
| D-35 | **APPROVED for the future extension** — distinguish dispatch, temporary wait, ordinary idle, and failure. Do not hide solver/numerical errors, fabricate useful work, or add automatic retries or fallback. | §19.7 |
| D-36 | **APPROVED for the future extension** — use per-microbatch token/action budgets, all-resident capacity, and the existing common profiled block pool; no automatic scaling by TP/PP degree and no double charge for outstanding allocations. | §19.5 |
| D-37 | **APPROVED for the future extension** — extend LP and the adapted reference on the same runtime conditions while preserving their distinct policy rules; no upstream whole-engine equivalence claim. | §§19.1, 19.8 |
| D-38 | **APPROVED for the future extension** — bounded CPU and GPU validation of TP-only, genuinely overlapping PP, and combined TP+PP for any combined claim; retain decision-to-completion traces and exact provenance. Performance and recovery evaluation remain separate. | §19.9 |

D-13 provides no basic/extreme-point or fractional-count guarantee, and the
amended LP has none in general (§7.3); its performance, control, and basis
remainder remains OPEN (§9.3).

### 17.1 OPEN decisions

Only OPEN decisions required by the active MVP need an immediate value. A required
value may be supplied as a visible, provisional configuration or scoped
experiment input; it is not thereby a permanent project-wide decision. No listed
candidate is a hidden default.

| ID | Decision | Candidate space stated by sources | Must be resolved by |
|---|---|---|---|
| D-01 | Decode utility $\alpha_i(t)$ | Responsiveness, age, fairness, SLO urgency, or another documented objective; the R17 rank-based family is a candidate only | Before end-to-end policy behavior is evaluated |
| D-02 | Prefill utility $\beta_i(t)$ | Progress, age, fairness, SLO urgency, or another documented objective | Before end-to-end policy behavior is evaluated |
| D-03 | Preemption penalty $\gamma_i(t)$ | Explicit restart/recomputation penalty policy, including the optional R18 discarded-computation term with unselected $\kappa$ | Before runtime preemption is enabled. Runtime preemption was enabled on 2026-10-10 (§13) with $\gamma$ only as an explicit scoped provisional input (uniform 1 in the CPU evidence); the policy remains OPEN and must be resolved before preemption behavior is evaluated as policy |
| D-04 | Utility scaling and normalization | Domains settled 2026-10-12: finite and nonnegative, possibly state-dependent, fixed per decision (manuscript; R17). Scaling and normalization: no canonical choice in sources | Before solver tests using final utilities |
| D-05 | $B_{\max}$ binding | Explicit combined token budget; Sarathi/SLAI-like fields are candidates | Before Phase E completion |
| D-06 | $C_{\max}$ binding | Fixed chunk, dynamic chunk, or explicit new configuration | Before Phase E completion |
| D-07 | $S_{\max}$ binding | Explicit next-forward action limit | Before Phase E completion |
| D-13 | Remaining solver performance, control, and basis policy | Finite live/performance limits; possible `highs`/`highs-ipm`; edge-weight and other tuning; IPM/crossover; basis certification; warm starts/basis reuse; threading/parallelism/random seed; cross-version degenerate-optimum behavior; any solver-output-to-fractional-bound assertion | Does not block initial Phase D; before the relevant performance, control, or structural claim |
| D-22 | Recomputation output contract | Original prompt/generated suffix versus expanded-context representation | Before claiming corrected or public recomputation-output semantics; MVP compatibility mode may use native preemption with the documented inherited limitation |
| D-23 | Approximation guarantee | None for the corrected constructor: the manuscript's utility-limit family has ratio tending to zero; a guarantee would need a different method | Before any theoretical quality claim |
| Q11 | Adapted-policy recovery and overhead evaluation (OPEN, deferred 2026-10-12) | Primary record: `docs/LP Scheduler Research Context.md` §21 | Before any quantified recovery, fidelity, or overhead claim |

D-09 and D-10 moved to the register above as RESOLVED on 2026-10-12.
D-24 moved there as design-resolved on 2026-10-18. Neither that move nor
D-32–D-38 establishes implementation, runtime correctness, or measured overhead.

The decision record for an item MUST include the selected rule, rationale, affected interfaces, tests, and experiment metadata field where relevant.

## 18. Traceability and change control

### 18.1 Source-to-requirement traceability

Trace formulation and continuous $x_i$ to `docs/math/main-llm-serving.tex`;
validation, phase boundaries, mapping, execution, and failure to the Research
Context; and framework facts to the architecture audit. `project_status.md`
is chronology only. Implementations MUST retain the cited mathematical and
audited constraints. Amended requirements (2026-10-12) trace to
`subsec:myopic_ilp` and `subsec:lp_relaxation` at `69e2180` and to discussion
decisions R01–R21; D-09, D-10, D-12, D-14–D-16, D-18, D-21, and D-26–D-31 record
them.

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


## 19. Approved future TP and PP extension

### 19.1 Approval, purpose, and boundary

**Approved with the user on 2026-10-18; documentation only.** This section is
the canonical implementation-facing record of the parallelism discussion.
The local source reviewed was clean `main` at
`2c61d3e68b96fdab2947b6be4d197ec7bb36215a`. No new CPU test, GPU run, or
performance measurement was performed for this record.

Extend both `LPScheduler` and `AdaptedVLLMScheduler` to fixed tensor
parallelism (TP), overlapping pipeline parallelism (PP), and combined TP+PP.
Fixed means that the worker group and model partitioning do not change
during a run. Each scheduler selects the next microbatch; the engine handles
its passage through all pipeline stages. This does not select dynamic
parallelism, replica routing/data parallelism, expert parallelism,
prefill/decode disaggregation, or GPU placement optimization. Select exact
model, degrees, topology, and workload for each implementation/validation task;
do not imply that every model, degree, or multi-node arrangement is covered.

The research purpose remains observing coefficient policies and comparing
decision methods in the same serving framework. Keep the path from captured
inputs to LP relaxation, integer extraction, and native execution visible.
Do not construct a reference schedule first and encode its answer in weights.
No new coefficient formula, solver tuning, performance target, recovery
criterion, or approximation guarantee is selected. Q11 remains OPEN.

This is the approved future replacement for the single-stage assumptions in
D-08, D-17–D-19, D-24, and D-26 where explicitly stated below. Existing
single-stage behavior is not changed by this documentation task. Implementation
and validation of the replacement are required before removing support guards.
The mathematical source remains authoritative; §19.4 identifies the required
alignment before enabling the extension. The historical Phase C audit and
earlier test evidence are not repinned or extended.

### 19.2 Existing mechanisms to reuse and limitations to inherit

Reuse `ParallelConfig`, worker creation and model partitioning,
`PipelineParallelLLMEngine`, `PipelineParallelWorker`, native cache managers,
`SchedulerOutputs`, dispatch queues, completion processing, and native
reset-and-recompute preemption. LP already executes victims through
`BaseScheduler._preempt`; do not build another cache-release path.

At the reviewed revision, the inherited vLLM-style/Sarathi victim loops can
select a `RUNNING` resident; `_preempt` accepts native executing statuses
`RUNNING` and `PAUSED`. SLAI prefers its paused-prefill victims. These policy
preferences differ from physical execution guarantees. The new adapted
reference currently rejects TP>1 and PP>1 and requires paused residents.
Its extension requires explicit readiness/ownership changes, not just removal
of guards. Relevant sources are `sarathi/core/scheduler/{base_scheduler,
vllm_scheduler,sarathi_scheduler,slai_scheduler,adapted_vllm_scheduler}.py`,
`sarathi/engine/{arg_utils,pipeline_parallel_llm_engine}.py`, and
`sarathi/core/sequence_manager/{base_sequence_manager,
worker_sequence_manager}.py`.

Do not restrict victims to non-in-flight requests. Use scheduler ownership,
allocation, and the selected inherited native-status contract to construct
the victim set. Preserve execution/preemption exclusion within the new plan.
Eligibility is supplied to the LP; relaxed values do not establish it.

The extension must not require LP-only block reservations, cache versions,
GPU-completion checks for every victim, transactions, or rollback. Inherit and
document native recomputation/output, sampling, replay, and completion
limitations (§16), addressing a defect only if it prevents the selected path
from running. Existing worker ordering may protect reuse on each GPU even
while an old batch continues elsewhere in the pipeline. Discarding an old
token alone does not prove all old writes and bookkeeping harmless; this is
an inherited execution assumption, not a newly proved guarantee. Reusing the
same mechanisms does not guarantee identical exposure to defects when LP
chooses different action combinations.

### 19.3 Protected capture, unprotected planning, protected commit

Use brief shared controller protection around capture and around fresh
validation/commit. A common controller lock is the selected straightforward
mechanism; choose its exact placement and lock order during implementation.
All relevant writers must honor the same boundary: scheduling ownership and
allocation, engine-side scheduling replay, completion updates, outstanding
batch accounting, and registration. An LP-local lock that completion ignores
is insufficient. Apply the common controller rule to the adapted reference
as well; do not require it to use LP data types or copy the LP policy.

1. Wait for dispatch capacity when the outstanding-batch limit is reached.
   Do not predict an unfinished final-stage result to build a speculative
   ready set. A first-stage wake-up is not final completion.
2. Under capture protection, copy only the controller fields required for
   ownership, eligibility, progress, allocation, free blocks, limits, ranks,
   and coefficients into immutable records. Capture the decision time and
   problem identity. Do not read or copy GPU KV tensors or drain workers.
3. Release protection. Construct coefficients/ranks, solve, and extract from
   the private captured inputs. Do not reread live state for those steps.
   Completions may be applied and new waiting requests registered meanwhile.
4. Under commit protection, validate the original proposal and its permitted
   completion adjustments (§19.6), check fresh combined native feasibility,
   apply native actions, mark selected requests as executing through engine
   replay, and update/publish batch accounting in dispatch order. No completion
   may interleave between the relevant checks and controller mutations.
5. Release protection without waiting for GPU/model results. Use the existing
   dispatch and completion mechanisms; preserve their required ordering.

Keep one planning decision/pending plan per engine. No other scheduling
decision may allocate or dispatch a competing plan during its solve. This
does not limit already-dispatched microbatches to one. Registration may add
waiting requests but must not allocate their cache. Other state-changing
operations that would invalidate these assumptions are outside this initial
contract; do not add speculative cancellation or recovery support.

Protection pauses conflicting CPU record updates, not GPU execution. Fields
need not be copied at the same nanosecond; excluding writers during the copy
provides one coherent controller state. Arbitrarily mixed observations are
not allowed: recording A's four allocated blocks, then reading free blocks
after A completes, and also crediting A's preemption would count its memory
twice. Older coherent inputs are acceptable; inconsistent inputs are not.

Hold no controller protection during coefficient calculation, optimization,
integer extraction, or blocking worker/result calls. Preserve a consistent
lock order with existing sequence-manager locks. This is not cross-layer
atomic execution: post-mutation exceptions still terminate the run and the
affected state is not reused (D-20). Capture and commit may cost time growing
with requests/actions; measure their duration and contention rather than
claiming zero overhead or guaranteed speedup.

### 19.4 Request sets and mathematical alignment

Treat step t as one microbatch-selection opportunity, with earlier batches
possibly outstanding. Keep the existing objective, variables x, y, I^P, z,
and fixed-charge allocation model. No per-stage optimization variables are
required merely because execution is distributed.

| Set | Future mapping contract |
|---|---|
| U_t | Every arrived, unfinished, scheduler-owned request at capture |
| R_t | All allocated residents, including requests with outstanding actions |
| Q_t | Arrived waiting requests with no allocated cache |
| P_t | Requests native-ready for a positive prefill/recompute chunk at capture |
| D_t | Allocated, prompt-complete requests native-ready for one decode at capture |
| Z_t | Owned, allocated, unfinished residents accepted by the inherited preemption contract; may include in-flight residents |

Available execution candidates are P_t union D_t. An outstanding current
execution prevents selecting another execution action for that request. Its
residency and memory do not disappear, and it need not be excluded from Z_t.
Waiting requests may be recomputation returns; abandoned older attempts are
handled through inherited replay/completion semantics. Do not claim that
native status acceptance proves physical GPU cache-release safety.

Implement these meanings in mapping and executor checks, including the current
blanket `PAUSED` resident checks, not only in the scheduler entry guard. Keep
rank and utility policy inputs explicit and fixed for the captured decision.
Do not add newly ready or newly arrived execution candidates during commit.

**Required mathematical alignment before enabling PP:** amend
`docs/math/main-llm-serving.tex` (`subsec:myopic_ilp` and
`subsec:lp_relaxation`) to state this decision boundary, eligibility binding,
runtime waiting boundary, and distinction between original proposal and
effective execution after completed victims are omitted. Review the legality
and useful-progress arguments under these assumptions. In particular,
snapshot ILP infeasibility is not a claim that the whole pipeline cannot make
progress, and dropping a completed victim does not make the modified vector
feasible against the original F and R_t automatically. Keep mathematical
claims conditional on the stated bookkeeping/native model; do not extend
them into a proof of physical pipeline safety. This documentation record
selects the intended alignment but does not itself edit the TeX or extend its
existing proofs. Do not silently override that mathematical source.

### 19.5 Capacity and cache accounting

B_max counts prompt/decode tokens in the new microbatch; S_max counts its
execution actions. N_max (`max_num_seqs`) counts all retained residents across
outstanding batches. C_max remains the per-request prefill cap. Keep these
limits separate. The outstanding-batch cap comes from the native engine;
PP does not automatically divide the configured token budget, split one
scheduler output into PP outputs, or multiply capacity limits by TP/PP degree.

Reuse `BaseLLMEngine._init_cache`: profile local cache-block capacity on every
worker and use their minimum as the common block count. Local bytes per
block reflect that worker's layers/KV heads (`CacheEngine.get_cache_block_size`).
Each PP stage retains its own layers' cache for resident requests, not just
its current microbatch. Aggregate GPU bytes are not a freely pooled allocator.
Do not sum worker block counts as available logical context blocks.

Free blocks at capture already exclude centrally committed allocations for
outstanding work. Do not subtract that work again. Preserve full-context
admission, watermark, actual zero-or-one decode demand, native append gates,
and the selected execution order. No additional reserve is introduced.

### 19.6 Completion during planning and effective execution

Allow completion and registration between capture and commit. With one
allocator decision in progress, registration without allocation, and no
independent dispatch, ordinary completion does not consume new physical cache
blocks: it retains or frees allocations. Appending an output can add a logical
block; physical growth for the next action is charged at scheduling. Previously
ready execution candidates are not advanced by another current action during
this interval. Later-ready requests may be omitted, accepting lower utility
or utilization rather than recomputing merely for freshness.

At commit, first validate the original integer proposal against its original
captured problem. Build a separate effective execution description; do not
mutate the accepted mathematical result. Omit a selected victim only on
positive native completion evidence that the same request finished and its
central allocation was released. A missing ID/table alone is not that evidence.
Reuse completion results and retain only the information needed by the pending
decision; do not add an unbounded lifecycle journal or a cache-version system.

For example, F=2 plus a planned recovery of four blocks provides six blocks.
If that victim finishes, live free memory becomes six and the victim action is
omitted; it must not be credited again to produce ten. Account similarly for
resident places. Completed non-victims can only provide additional room under
the supported update contract. A victim that completes a nonfinal step may
still be preempted through the inherited current-state path.

Freshly validate effective execution against current ownership, readiness,
progress, resident capacity, free blocks, recovery, watermark, append gates,
and native mutation order, all together. Do not validate a victim-dropped
vector against unchanged original resource parameters and pretend it is the
same plan. No execution action or chunk is opportunistically added, enlarged,
or substituted. Do not replan merely because resources or candidates increased.
Any other genuine invalidity remains a visible pre-mutation failure under
§19.7, rather than an automatic retry. Unexpected interference must not be
hidden by repeated solves.

### 19.7 Dispatch, wait, idle, and failure

| Outcome | Rule for the future engine/scheduler integration |
|---|---|
| Dispatch | A successful original proposal yields a freshly validated useful effective execution; commit it |
| Wait | Pipeline capacity is full, or no execution candidate is ready while work is outstanding; commit nothing and process completion before a fresh decision |
| Wait after no useful plan | Solver-reported LP infeasibility or a validated `NoUsefulWork` outcome with outstanding work may describe a temporary state; retain that outcome and diagnostics, execute no partial plan, and reconsider after completion |
| Ordinary idle | No arrived unfinished work and no outstanding batch requiring completion; keep the existing idle meaning |
| Failure | Malformed state, other solver/numerical failure, invalid relaxed/integer result, failed fresh validation outside the permitted completion adjustment, or no useful feasible work when nothing outstanding can change the state |

Waiting after completion is a new normal scheduling opportunity, not retrying
a solver error. Preserve D-12/D-31: solver status 2 is reported infeasibility,
not an exact certificate. Do not classify every unsuccessful solve as wait.
Outstanding work must be processed even if current request ownership became
empty after preemption; do not return final idle and abandon pending results.

Keep the useful-execution row for dispatchable LP plans. A valid all-zero or
preempt-only output is not substituted for useful execution. Wait makes no
queue/block/preemption mutation and need not enter the LP when its reason is
already known. Preserve native iteration/output identity conventions and count
only actual dispatched batches, with one corresponding completion decrement.

Retain rejection/reporting of unsupported control-only outputs (§16.3),
including the adapted reference's typed unexecuted preempt-only proposal.
Do not force an unrelated action or silently repair its policy to get a model
result. Unexpected pre-mutation failure remains fail-stop; after mutation
begins, propagate the failure and discard the run's state. No automatic
retry, fallback, rollback, or general recovery subsystem is selected.

### 19.8 Reference comparison and records

Extend the adapted reference's supported runtime while preserving its FCFS
service order, tail-victim preference, waiting-admission rule, resource model,
and policy-specific proposal/execution layout from research-context §16.1.
Apply native readiness before selecting new execution actions; busy residents
still count and remain possible victims under the agreed contract. Do not
import LP-only chunk/action caps, progress constraints, coefficient logic, or
decode-demand ordering into that policy. Reinspect the pinned upstream sources
where needed to describe differences; label this a parallel LPServe adaptation,
not upstream PP equivalence. Inherited policies can supply additional comparisons.

Run both schedulers under the same common controller coordination, model,
worker topology, workload, and declared comparable resource settings. Identical
coordination does not imply identical cost: LP may spend longer planning and
make different allocation/preemption choices. Do not attribute runtime effects
automatically to coefficients or require identical physical block IDs.

Extend existing traces only as needed: captured request sets/progress/resource
counts and policy inputs, problem/proposal identity, original selected victims
and execution, completion-confirmed omissions, actual emitted output, wait or
failure reason, and dispatch/completion association. Keep planned snapshot
utility separate from effective executed utility; if the latter is reported,
use the stated captured coefficients and actual actions, not an undocumented
refresh or a claim of current optimality. Record TP/PP degrees, topology,
limits, pool/watermark, and the shared coordination rule. Separate capture,
planning, commit, and controller-wait timing where measured. Instrumentation
does not select a Q11 performance/recovery criterion.

### 19.9 Implementation sequence and acceptance boundary

Use responsibility-based implementation names and preserve the D/E/F
separation. The affected source boundaries to inspect are:

| Responsibility | Existing files/interfaces |
|---|---|
| Shared controller coordination and dispatch/completion accounting | `sarathi/engine/base_llm_engine.py`, `sarathi/engine/pipeline_parallel_llm_engine.py`, `sarathi/core/scheduler/base_scheduler.py` |
| Immutable capture, readiness, victim sets, ranks, and coefficients | `lpserve_state_mapping.py`; live coordination in `sarathi/core/scheduler/lp_scheduler.py` |
| Original-plan validation, completed-victim adjustment, fresh native checks, and mutation | `lpserve_plan_execution.py`; retain the framework-light boundary of `lp_relaxation_scheduler.py` |
| Adapted reference proposal, readiness, and native execution | `sarathi/core/scheduler/adapted_vllm_scheduler.py`; preserve research-context §16.1 policy |
| Supported configuration and real engine entry | `sarathi/config.py`, `sarathi/engine/arg_utils.py`, active inference/comparison scripts |
| Focused checks | Existing planner, mapper, executor, live-scheduler, adapted-reference, EngineArgs, and relevant driver tests under `tests/` |

1. Recheck branch, revision, working tree, affected current source, and the
   pinned audit. Read this section, research-context §16.1/§22.2, and the
   mathematical source. Complete §19.4's mathematical alignment and proof
   review before implementing the amended contract. Preserve unrelated work.
2. Implement the shared capture/commit boundary, completion evidence, native
   readiness/victim mapping, effective execution checks, waiting behavior,
   and corresponding adapted-reference extension. Remove support guards only
   when their replacement checks exist. Exact lock placement, compact record
   shapes, and test configuration are routine implementation choices; they
   must preserve this contract and require no speculative abstractions.
3. Run focused CPU checks before GPU execution: existing LP/reference
   regressions; resource/set mapping with busy residents; no duplicate current
   execution; completion during planning; coherent capture (including the
   double-count example); unchanged selected actions; completed-victim omission
   without double recovery; unknown missing-victim rejection; protected commit
   and accurate batch counts; wait/idle/failure distinctions; and solver errors
   that are never hidden by waiting. Use controlled interleavings, not only
   timing-sensitive stress tests. No claim of general inherited repair follows.
4. On an already allocated Unity GPU node, run directly in the existing
   repository Python environment. Do not invoke `srun`, obtain another
   allocation, submit jobs, SSH elsewhere, or rebuild the environment unless
   explicitly requested. Report the exact environment failure if blocked.
5. Start with bounded TP>1/PP=1, TP=1/PP>1 with actual overlap, then TP>1/PP>1
   for a combined claim. Use a compatible model and available hardware; record
   exact degrees rather than invent a universal supported matrix. Use dummy
   weights for a scoped lifecycle milestone where appropriate; they do not
   establish generation quality. Check LP and the adapted reference.
6. Inspect admission, resident partial prefill, decode/block growth, omission
   and later selection, preemption/recomputation (including inherited in-flight
   eligibility), completion, final release, and final idle. Observe overlap in
   outstanding batches/dispatch and completion records; PP configuration alone
   is not overlap evidence. Check per-worker allocation counts/conservation and
   completion association without demanding identical physical block IDs.
7. Retain tested revision or base plus exact changes/hashes, host/allocation,
   interpreter/environment, model assets, commands, outputs, and trace findings.
   State which cases passed, failed, or were unexecuted. Earlier single-worker
   tests do not validate later parallel changes. Bounded success is scoped
   support, not all-workload correctness, fairness, performance, Q11 recovery,
   or upstream equivalence.

The completion boundary separates mathematical alignment, shared-engine and
executor implementation, live integration of both schedulers, and observed
GPU validation. None substitutes for the others. This record authorizes no
implementation run, commit, push, or new experiment by itself. Exact test
degrees/environment, measured coordination overhead, and behavior outside the
selected native paths remain to be established in the future task. Unrelated
OPEN research choices and inherited limitations stay explicit.
