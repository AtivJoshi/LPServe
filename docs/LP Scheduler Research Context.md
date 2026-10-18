# LP Scheduler Research Context

## 1. Purpose

This document defines the research context, algorithmic requirements, implementation principles, validation requirements, and open design questions for implementing and evaluating an LP-based LLM scheduler in the SLAI/Sarathi research framework.

The central implementation principle is:

> Implement the currently selected mathematical formulation faithfully while deriving its software realization from LPServe/SLAI. Research may explicitly revise or replace the formulation; changes to the mathematical and implementation contracts must be recorded in the appropriate sources before implementing the amended contracts.

The document is intended to provide future research and implementation sessions with durable repository-local context for reasoning about the project without requiring knowledge of any other serving-system implementation.

---

# 2. Primary Research Goal

**Goals clarified with the user on 2026-10-16.** This section is the canonical
record of the project's research goals and working approach. Use it when
choosing implementation and experiment scope. These goals are revisable:
record the reason for a change and its consequences rather than treating the
current direction as permanent. This clarification selects no new coefficient
formula, experiment, or mathematical amendment.

The long-term goal is to develop a mathematical objective, or a family of
objectives, that explains the scheduling choices made by existing heuristics
and supports a class of scheduling algorithms. The current myopic ILP is a
candidate formulation. Its explanatory scope, state representation, objective,
and constraints can evolve as analytical and experimental evidence develops.
Distinguish recovering a reference's decisions under stated assumptions,
obtaining similar observed behavior, and obtaining similar performance.

## 2.1 Immediate goal: observe and compare coefficient policies

Build a correct, runnable framework for observing and monitoring how explicit
coefficient update policies behave and comparing them with reference
schedulers. The immediate task is to support that study; finding the best
coefficient policy is not the current goal. A user selects a policy according
to the behavior they want. Eventually, classify policy families by their
behavior and performance across workloads and resource conditions.

Begin with a few simple, interpretable coefficient policies and small states
where expected decisions can be derived. Include a policy aimed at recovering
the adapted upstream vLLM v0.31.0 reference defined in section 16, alongside
simple controls and contrasting preferences. Do not compute the reference's
answer and encode it in coefficients. Predictability in restricted states is a
starting point, not a claim of recovery in general mixed or contended states.
Use both comparisons from corresponding states to explain individual choices
and runs on the same arrival workload to observe how choices change subsequent
states. Label the reference version, adaptation, and serving framework.

Observations must separate captured inputs and coefficient rewards, the LP or
other optimization proposal, integer extraction or repair, and actual
execution. Behavior is a property of the coefficient policy together with the
decision method and runtime. Attribute an anomaly to the relevant stage before
calling it a coefficient-policy effect or an implementation bug.

Observe service, omission, preemption, repeated computation, and output
progress. Record unfinished runs and execution failures explicitly. Frequent
restarts can be a valid behavioral finding even when every decision satisfies
implementation contracts; they should not automatically trigger coefficient
tuning. Distinguish repeated computation from lasting progress, and bounded
noncompletion from a proof of infinite starvation. SLO adherence, TTFT, TBT,
latency, throughput, and further metrics need explicit definitions and
measurement boundaries as they are selected. Detailed evaluation choices
remain in section 21, including OPEN Q11.

## 2.2 Evolving the decision method and mathematical model

The initial method solves the LP relaxation at each non-idle scheduling
decision, including prefill and decode decisions, to solver tolerances and then
constructs a feasible integer plan. It does not solve the ILP optimally. This
is an initial research implementation from which to study behavior. Later,
compare faster or different ILP approximation methods for both behavior and
end-to-end performance, separating scheduling overhead from model execution.
Measure the overhead rather than treating its magnitude as established.

Study which constraints restrict decisions, which are costly to enforce, and
which materially change behavior. These bottlenecks may depend on the
coefficient policy, workload, and approximation method. A rarely binding
constraint is not automatically safe to remove: its removal may expose new
solutions. Keep the current detailed formulation as a reference while
investigating simpler variants. If a variant moves constraints into heuristic
plan repair, define that repair as part of the algorithm, observe its changes
to the proposal, compare its total cost and behavioral effect, and recheck the
relevant correctness arguments. Complete physical validation before execution
remains required.

Eventually, let desired SLO behavior guide coefficient construction and study
how much it can express. Where it is insufficient, investigate additional
state, objective terms, constraints, or alternative formulations. Prefer
interpretable explanations that predict behavior beyond the construction
examples; unrestricted rewards that merely reproduce a reference action do
not establish a useful unifying objective.

Future manuscript proposals, including nonlinear execution-time models, are
exploratory ideas requiring separate derivation and validation. A utility-rate
objective and utility maximization under a time constraint are different
choices whose equivalence must not be assumed. No downstream manuscript
proposal is adopted or made an implementation prerequisite by these goals.

## 2.3 Implementation setting and priorities

The primary implementation repository is:

```
github.com/AtivJoshi/LPServe
```

`LPServe` is itself a fork of:

```
github.com/agrimUT/SLAI
```

The working local repository for implementation, testing, and experimentation is the user's clone of `LPServe`, not an unrelated upstream serving-system repository.

SLAI is a research prototype built on Sarathi-Serve and intentionally retains a reduced feature set to facilitate scheduler research and experimentation. This makes it an appropriate environment for implementing scheduling algorithms directly without reproducing the full complexity of a production-oriented serving engine.

The implementation should prioritize:

- fidelity to the mathematical scheduling algorithm;
- correctness of request state transitions;
- correct KV-cache and resource accounting;
- clean experimental comparisons;
- reproducibility;
- implementation simplicity appropriate for research.

Full feature parity with modern upstream vLLM is not a goal.

---

# 3. Sources of Truth

The following kinds of information must be kept separate.

## 3.1 Mathematical source of truth

The current mathematical source of truth for the proposed scheduler is the repository-local file:

```
docs/math/main-llm-serving.tex
```

especially:

```
Myopic Utility Maximization: The Primal ILP Formulation
\label{subsec:myopic_ilp}

Primal Heuristic 1: LP Relaxation and Integer Extraction
\label{subsec:lp_relaxation}
```

These two subsections were amended together on 2026-10-12 (committed at
`b1addc8` and `69e2180`), following the decisions recorded in
`docs/lp_scheduler_architecture_discussion_phase_F.4.md`. All later manuscript
sections are unrevised: the red revision-boundary paragraph at the end of
`subsec:lp_relaxation` marks them outdated wherever they depend on the old
feasible set, the reserve $W_t$, the old extraction, or the three-resource
decomposition. They include the second primal heuristic, the Lagrangian dual
and its slow and fast paths, the reduction to native vLLM, the MPC and
hierarchical MPC formulations, and later recovery, subsumption, and equivalence
claims. They must not be read as established algorithms, proofs, or
equivalence results for the amended formulation until a separate mathematical
assignment revises them.

The current implementation target is **Primal Heuristic 1 only**.

Later proposals in the LaTeX document, including:

- Primal Heuristic 2;
- Lagrangian relaxation;
- hierarchical slow-path/fast-path scheduling;

are separate research directions and should not be mixed into the first LP scheduler.

## 3.2 Serving-framework source of truth

The working `LPServe` repository is the source of truth for implementation behavior:

```
github.com/AtivJoshi/LPServe
```

It is a fork of:

```
github.com/agrimUT/SLAI
```

Use the current local `LPServe` clone for:

- sequence/request representation;
- scheduler state;
- queue semantics;
- block allocation;
- preemption;
- chunked-prefill execution;
- decode execution;
- scheduler configuration;
- scheduler outputs;
- engine lifecycle;
- metrics and benchmarking.

When implementation behavior matters, inspect the current `LPServe` code directly.

Do not assume that similarly named concepts in modern vLLM, another Sarathi version, or another serving framework behave identically.

## 3.3 This document

This document provides implementation principles, validation requirements, research assumptions, phase boundaries, and open design questions for the project. It is not the complete normative scheduler implementation specification; that role belongs to `docs/lp_scheduler_design.md`.

If this document, `docs/math/main-llm-serving.tex`, `docs/lp_scheduler_design.md`, and the current `LPServe` implementation appear inconsistent, investigate the discrepancy rather than silently choosing one interpretation.

---

# 4. High-Level Scheduling Problem

At every scheduler decision epoch $t$, let $U_t$ denote the set of non-finished, arrived requests relevant to the current scheduling decision.

Conceptually, $U_t$ may contain:

- newly waiting requests;
- partially-prefilled requests;
- decode-ready requests;
- currently GPU-resident requests;
- recomputation-preempted requests that have returned to a schedulable waiting state.

Let

$$
\mathcal{Z}_t
=
\left\{
i \in U_t :
i \text{ is resident and legally preemptible at the beginning of step }t
\right\}
$$

denote the preemptible subset of $U_t$. Waiting, already-preempted, finished, and otherwise non-preemptible requests do not belong to $\mathcal{Z}_t$. Requests whose physical KV state cannot safely be released at the current decision boundary, including relevant pipeline-in-flight requests, must also be excluded.

The precise mappings of $U_t$ and $\mathcal{Z}_t$ to `LPServe`/SLAI `Sequence` objects must be derived from the current implementation and validated before scheduler state is mutated.

For each request $i \in U_t$, the scheduler chooses at most one of the following actions for the next execution step:

1. process a positive number of prefill tokens;
2. generate one decode token;
3. preempt the request, if $i \in \mathcal{Z}_t$;
4. do nothing.

The LP scheduler is intended to choose these execution actions directly. It is not merely a mechanism for assigning queue priorities. The canonical do-nothing action sets all action variables for the request to zero.

*(amended 2026-10-12)* $U_t$ is partitioned by allocation state into the resident
set $\mathcal R_t$ (allocated, including residents omitted from the next forward
pass) and the waiting set $\mathcal Q_t$ (arrived and unallocated, including
recomputation-preempted requests). When $U_t$ is empty, the step is ordinary
idle and no LP is solved. Otherwise, after input and admission validation, the
plan must execute at least one prefill or decode; preemption alone is not
progress. If no legal useful plan exists, that is reported rather than
replaced by an invented action.

---

# 5. Myopic ILP

For every request $i \in U_t$, define

$$
x_i(t)\in\mathbb{Z}_{\ge0},
$$

the number of prefill tokens processed in the step;

$$
y_i(t)\in\{0,1\},
$$

indicating whether one decode token is generated; and

$$
I_i^P(t)\in\{0,1\},
$$

indicating whether the request receives a positive prefill operation.

For every legally preemptible request $i \in\mathcal{Z}_t$, define

$$
z_i(t)\in\{0,1\},
$$

indicating whether the request is preempted. For notational convenience, define

$$
z_i(t)=0
\qquad
\forall i\in U_t\setminus\mathcal{Z}_t.
$$

The request/system parameters are:

- $P_i^{\mathrm{rem}}(t)$: remaining unprocessed prompt tokens;
- $C_{\max}$: maximum candidate prefill chunk size;
- $U_i(t)=\min(P_i^{\mathrm{rem}}(t),C_{\max})$: maximum prefill action available to request $i$ (the manuscript's $q_i$);
- $a_i^P(t)$: fixed planning-memory cost of executing a positive prefill action;
- $c_i^D(t)\in\{0,1\}$: actual additional blocks allocated by one decode action (amended 2026-10-12);
- $c_i^Z(t)$: planning memory recovered by legally preempting request $i$;
- $B_{\max}$: per-step token-volume budget;
- $S_{\max}$: scheduled-action concurrency budget;
- $F=M_t^{\mathrm{free}}$: free KV-cache blocks at the beginning of the step;
- $h$: the shared block manager's admission watermark in blocks;
- $N_{\max}$: the resident limit (`max_num_seqs`), distinct from $S_{\max}$.

The former optional memory reserve $W_t$ is removed (amended 2026-10-12).

Under the audited Sarathi/LPServe allocation behavior,

$$
a_i^P(t)
=
\begin{cases}
0,
& \text{if request $i$ is already resident and allocated},\\
A_i(t),
& \text{if request $i$ requires admission or recomputation},
\end{cases}
$$

where $A_i(t)$ is the number of physical KV-cache blocks required to allocate the request's complete current logical context.

For $i\in\mathcal{Z}_t$,

$$
c_i^Z(t)
$$

is the number of physical blocks that would be released by legally preempting request $i$ at time $t$.

The request-specific utility weights are:

- $\alpha_i(t)$: utility of scheduling one decode token;
- $\beta_i(t)$: utility per scheduled prefill token;
- $\gamma_i(t)$: penalty for legally preempting the request.

The one-step objective is

$$
\max
\left[
\sum_{i\in U_t}
\left(
\alpha_i(t)y_i(t)
+
\beta_i(t)x_i(t)
\right)
-
\sum_{i\in\mathcal{Z}_t}
\gamma_i(t)z_i(t)
\right].
$$

The global token-volume constraint is

$$
\sum_{i\in U_t}
\left(
x_i(t)+y_i(t)
\right)
\le B_{\max}.
$$

The global scheduled-action concurrency constraint is

$$
\sum_{i\in U_t}
\left(
I_i^P(t)+y_i(t)
\right)
\le S_{\max}.
$$

*(amended 2026-10-12)* Let

$$
H=F+\sum_{i\in\mathcal R_t}c_i^Z(t)z_i(t)-\sum_{i\in U_t}a_i^P(t)I_i^P(t),
$$

the free blocks after selected preemptions and prefills, before decodes. $H$ is
a linear expression, not a decision variable. The resource and progress
constraints are

$$
H-\sum_{i\in U_t}c_i^D(t)y_i(t)\ge0,
\qquad
\sum_{i\in U_t}y_i(t)\le S_{\max}H,
\qquad
H\ge h\,I_i^P(t)\quad(i\in\mathcal Q_t),
$$

$$
|\mathcal R_t|-\sum_{i\in\mathcal R_t}z_i(t)+\sum_{i\in\mathcal Q_t}I_i^P(t)\le N_{\max},
\qquad
\sum_{i\in U_t}\left(I_i^P(t)+y_i(t)\right)\ge1.
$$

They express actual memory consumption, the free-block-before-decode
prerequisite, the shared admission watermark (one margin for the whole
admission group, redundant when $h=0$), resident capacity counting every
retained resident, and useful execution. They match the shared allocator checks
when selected actions execute as preemptions, prefills, zero-demand decodes,
and then one-block decodes, under the assumptions stated in the manuscript.

Chunked prefill obeys

$$
I_i^P(t)
\le
x_i(t)
\le
U_i(t)I_i^P(t)
\qquad
\forall i\in U_t.
$$

This pair of inequalities makes $I_i^P(t)$ an exact indicator of a positive prefill action in the integer formulation:

- $I_i^P(t)=0$ forces $x_i(t)=0$;
- $I_i^P(t)=1$ requires $1\le x_i(t)\le U_i(t)$.

Decode causality is represented by

$$
y_i(t)
\le
\mathbf{1}
\left\{
P_i^{\mathrm{rem}}(t)=0
\right\},
$$

which the manuscript states as decode eligibility $y_i\le\mathbf 1\{i\in\mathcal D_t\}$,
together with prefill eligibility $I_i^P\le\mathbf 1\{i\in\mathcal P_t\}$.

Finally,

$$
I_i^P(t)+y_i(t)+z_i(t)\le1
\qquad
\forall i\in U_t
$$

enforces mutual exclusion between prefill, decode, and legal preemption. Because $z_i(t)=0$ for $i\notin\mathcal{Z}_t$, this constraint applies uniformly to all requests.

These constraints define the mathematical core of the first scheduler. *(amended 2026-10-12)* Token volume, action width, actual memory, the decode prerequisite, the watermark rows, resident capacity, and useful execution couple requests. Prefill linkage, eligibility, action mutual exclusion, and preemption eligibility remain request-local restrictions. The former description of exactly three global coupling constraints is superseded.

---

# 6. Utility Weights Are a Research Decision

The LP formulation deliberately leaves

$$  
\alpha_i(t), \qquad  
\beta_i(t), \qquad  
\gamma_i(t)  
$$

general.

They are not incidental implementation constants. They determine the scheduling behavior produced by the optimization problem.

The project should explicitly define and document the utility model being evaluated.

Possible utility designs may encode priorities such as:

- decode responsiveness;
- prefill progress;
- FCFS or age-based preference;
- fairness;
- preemption avoidance;
- SLO-related urgency.

These choices should be treated as research decisions rather than hidden inside scheduler implementation details.

*(amended 2026-10-12)* The manuscript requires the coefficients to be finite and
nonnegative on a common scale; they may depend on state but are fixed during each
decision. The rank-based family of discussion R17 and the optional
discarded-computation penalty of R18 remain candidates. Final formulas, scales,
and any $\kappa$ value remain open. Coefficients MUST NOT encode a schedule
computed first by another scheduler.

Coefficients and policy ranks are built from one read-only capture of the
scheduling state, by a small pure function, before the LP is constructed, and
stay fixed for the whole decision. Explicit uniform utilities remain a usable
policy. The data path is specified in `docs/lp_scheduler_design.md` §12.3.

If simple utility weights are needed during early smoke testing, they should be clearly labeled as temporary test settings rather than the final policy used in research experiments.

---

# 7. LP Relaxation

A true continuous LP relaxation must relax the prefill-token variable $x_i$ as well as the binary action variables. The relaxed domains are

$$
x_i\in\mathbb{R}_{\ge0},
\qquad
y_i,I_i^P\in[0,1],
$$

and

$$
z_i\in[0,1]
\quad
\forall i\in\mathcal{Z}_t,
\qquad
z_i=0
\quad
\forall i\in U_t\setminus\mathcal{Z}_t.
$$

*(amended 2026-10-12)* The relaxation keeps the objective and every amended row
of §5, including useful execution, resident capacity, actual memory, the decode
prerequisite, and each watermark row.

The request-local linkage constraints

$$
I_i^P
\le
x_i
\le
U_i I_i^P
$$

remain part of the relaxation. They prevent a full prefill indicator from being paired with fewer than one prefill token and strengthen the continuous formulation.

Let the relaxed solution be

$$
\tilde{x}_i,
\qquad
\tilde{y}_i,
\qquad
\tilde{z}_i,
\qquad
\tilde{I}_i^P.
$$

A standard LP solver may be used initially. SciPy/HiGHS is a reasonable first candidate if it fits naturally within the `LPServe` environment.

Solver choice must remain isolated from the surrounding scheduler logic so that it can be replaced without redesigning the mathematical layer. Solver status, numerical failure, infeasibility, and exceptions must be represented explicitly rather than converted into fabricated action values.

*(decided 2026-10-12)* A SciPy/HiGHS infeasible status is recorded as
solver-reported LP infeasibility, with raw diagnostics. It is a numerical report,
not an exact certificate; the exact implication from an infeasible relaxation to
an infeasible ILP holds only for actual mathematical infeasibility. It is kept
separate from the extraction's exhaustive finding that no legal useful integer
plan exists. LP feasibility alone does not establish integer feasibility.

The relaxed solution is not an executable scheduler decision. It must first be converted into a feasible integer action plan.

---

# 8. LP Extraction Algorithm

*(amended 2026-10-12)* The integral/fractional partition, indicator locking,
preemption-first rounding, safety preemption, and support-only packing described
here previously are superseded. The authoritative algorithm is
`alg:lp_extraction` in `subsec:lp_relaxation`; its implementation contract is
§10 of `docs/lp_scheduler_design.md`. In summary, one LP-guided pass builds whole
actions:

1. Order every eligible execution candidate, including candidates with zero LP
   support, by descending $\tilde I_i^P+\tilde y_i$, then service rank, then
   identity. Order legal victims by descending $\tilde z_i$, then victim rank,
   then identity. Ranks are immutable per-decision policy inputs, distinct from
   request identity.
2. Try each candidate once with one whole decode or a prefill chunk
   $\min(U_i,\text{remaining tokens},\max(1,\lfloor\tilde x_i\rfloor))$. If the
   trial plan violates an integer resource check, add legal victims in victim
   order until it fits or the list is exhausted. Integral-idle and zero-support
   victims, including zero-block victims that free a resident place, are
   allowed. A failed trial is discarded completely.
3. Accept the first legal trial regardless of utility. Afterward, accept a trial
   with new victims only for strictly positive gain, and a trial without new
   victims for nonnegative gain. Accepted actions and victims are never removed.
4. Extend selected prefill chunks with remaining tokens in service-rank order,
   then validate the complete integer plan.

Under the manuscript's assumptions, every returned plan is legal and useful, and
the pass finds useful work whenever a legal useful integer plan exists; if every
first trial fails, no such plan exists. These are analytical arguments, not
executed evidence. Accepted limitations include avoidable negative utility on
the first acceptance, no utility approximation guarantee, possible mass
preemption with no victim-count bound, and no restart, starvation, completion,
or policy-recovery guarantee. Finite-precision handling (numeric sorting
without tolerance comparators, exact integer resource checks, and a
conservative strict-improvement threshold from the existing objective
tolerances) is specified in the design.

Under the audited allocation behavior, memory is a fixed admission gate for
prefill rather than a per-token fluid capacity: reducing a chunk does not reduce
$a_i^P(t)$, so chunks may be truncated by the token budget but not by dividing
memory by a per-token coefficient.

---

# 9. Fractional Structure

*(amended 2026-10-12)* The former expectation of at most three fractional
requests is retired. The amended relaxation has per-admission watermark rows,
and the manuscript gives an analytical construction in which all of $n$ waiting
requests are fractional at an optimal vertex, for any $n$. Extraction therefore
must not depend on the number of fractional requests, on receiving a vertex, or
on solver basis behavior. Observed fractional counts may be recorded as
descriptive data only.

Do not add new coupling constraints merely for implementation convenience; any
new constraint requires rechecking the legality and progress arguments.

---

# 10. Planning Memory Versus Physical KV Feasibility

A general token-incremental serving model may use the planning-memory expression

$$
c_i^P(t)x_i(t)
+
c_i^D(t)y_i(t)
-
c_i^Z(t)z_i(t),
$$

leading to the constraint

$$
\sum_{i\in U_t}
\left(
c_i^P(t)x_i(t)
+
c_i^D(t)y_i(t)
-
c_i^Z(t)z_i(t)
\right)
\le
F.
$$

This abstraction is more general for serving architectures that allocate KV-cache capacity incrementally as prefill tokens are processed. *(amended 2026-10-12)* It is not supported here: it would need its own allocation model and its own derivation of the admission and decode checks.

The audited Sarathi serving architecture used by `LPServe`, however, allocates the physical blocks required for a request's complete current logical context when the request is admitted. Choosing a smaller positive prefill chunk does not proportionally reduce this allocation. Conversely, a resident partial-prefill request has already incurred the allocation and normally requires no additional blocks for another prefill chunk.

Because of this limitation of the Sarathi serving architecture, the first LPServe-native scheduler uses the fixed admission charge $a_i^P(t)I_i^P(t)$ inside $H$ and the amended rows of §5 *(amended 2026-10-12)*: actual memory $H-\sum_ic_i^Dy_i\ge0$, the decode prerequisite, the admission watermark, and resident capacity.

This is still a planning model. `LPServe` ultimately manages KV cache through physical block tables, and feasibility can depend on:

- current logical and physical block-table lengths;
- current free-block state;
- whether a request is already allocated;
- the block manager's admission watermark;
- append-slot behavior;
- sequence status and collection ownership;
- the order in which preemptions, allocations, and append operations are committed.

The natural planning unit is the number of KV-cache blocks rather than raw bytes or a GPU-memory-utilization percentage.

The state-mapping layer should construct:

- $F=M_t^{\mathrm{free}}$ from the block manager's free-block count, and $h$ from its admission watermark;
- $a_i^P(t)=0$ for an already allocated resident partial prefill;
- $a_i^P(t)=A_i(t)$ for an unallocated request requiring full-context admission or recomputation;
- $c_i^D(t)\in\{0,1\}$ from the exact logical/physical block gap of one decode step; another gap is an unsupported-state error (amended 2026-10-12);
- $c_i^Z(t)$ from the request's currently allocated physical block-table length;
- $\mathcal{Z}_t$ from legal preemption status, ownership, allocation, and in-flight conditions;
- $\mathcal R_t$, $\mathcal Q_t$, $|\mathcal R_t|$, and $N_{\max}$ from `running`, arrived `waiting`, and `max_num_seqs`;
- immutable service and victim ranks from the same captured queue state.

The final LPServe action executor remains responsible for checking that selected actions are physically and operationally legal. It must validate the complete integer plan before mutating scheduler state and must serialize physical mutations in the selected commit order.

Therefore:

> LP planning-memory feasibility and actual block-manager feasibility are distinct requirements.

Do not claim that satisfying the LP resource inequalities proves that every selected action can be executed. Conversely, avoid expanding the LP merely to reproduce every allocator implementation detail. The initial mathematical layer should use the simplest meaningful planning abstraction and rely on exact physical validation at the execution boundary.

*(amended 2026-10-12)* The memory reserve $W_t$ is removed. Its intended role, admission-watermark protection, is represented by the watermark rows. The free-block-before-decode check, to which the watermark does not apply, is represented separately by the decode prerequisite.

---

# 11. Preemption Semantics

The intended initial preemption model is recomputation preemption rather than CPU swapping.

Conceptually:

```
preempt request
→ release its GPU KV state
→ return it to a schedulable waiting state
→ recompute discarded context when it is eventually resumed
```

The exact legal sequence of state changes must be determined from the current `LPServe` implementation.

The executor must preserve `LPServe`/SLAI's sequence-state and block-manager invariants rather than inventing state transitions solely to match the mathematical variable $z_i$.

---

# 12. SLAI-Native Implementation Philosophy

The LP scheduler should be designed natively around the working `LPServe` fork of SLAI.

The current architecture provides concepts such as:

- `BaseScheduler`;
- `waiting` and `running` `Sequence` collections;
- a scheduler-owned block manager;
- allocation, free, append-slot, and preemption operations;
- prompt-progress state in `Sequence`;
- `SequenceScheduleMetadata`;
- `SchedulerOutputs`.

These facilities should be used where appropriate rather than introducing an unrelated scheduling architecture around them.

The intended conceptual flow is:

```
LPServe/SLAI scheduler / Sequence / block state
                  ↓
          LP state construction
                  ↓
           utility construction
                  ↓
            relaxed LP solve
                  ↓
          integer extraction
                  ↓
           LP action decisions
                  ↓
       LPServe/SLAI-native action execution
                  ↓
          SchedulerOutputs
```

This is a conceptual decomposition, not a requirement to create a particular set of classes or modules.

Introduce abstractions only when they:

- simplify the `LPServe`-native implementation;
- improve separation of concerns;
- make the mathematical layer independently testable; or
- materially improve maintainability or experimental clarity.

---

# 13. What the SLAI Architecture Audit Must Determine

Before implementing the scheduler, inspect the actual `LPServe` fork and answer at least the following questions.

*(Historical Phase C question list. Answers are in the architecture audit; later
decisions, including the 2026-10-12 removal of $W_t$ and exact decode demand,
supersede any choice the questions leave open.)*

## Request state

- Which sequences belong in $U_t$?
- Which resident sequences belong in the legally preemptible set $\mathcal{Z}_t$?
- How is remaining prefill work computed?
- How is decode readiness represented?
- Which sequence statuses, collection memberships, allocation states, and pipeline states make preemption legal?
- How are recomputation-preempted sequences represented?
- Which sequence states are eligible for each LP action?

## Compute capacity

- What configuration or runtime quantity corresponds to $B_{\max}$?
- Is the relevant budget fixed or scheduler-dependent?
- How do existing SLAI/Sarathi schedulers account for mixed prefill and decode tokens?

## Concurrency

- What exactly should $S_{\max}$ mean in `LPServe`?
- Does it count resident sequences, scheduled actions, running sequences, or another quantity?
- How should this definition align with `LPServe`'s actual execution constraints?

## Memory

- What free-block information does the block manager expose?
- How does it decide whether a new or recomputation-preempted sequence can be allocated?
- How does it decide whether a resident sequence can append one decode slot?
- How is $a_i^P(t)$ computed for unallocated and already allocated requests?
- How is $c_i^D(t)$ computed or conservatively approximated?
- How is $c_i^Z(t)$ computed from the current physical block table?
- Does the proposed $\mathcal{Z}_t$ include only requests whose KV state can legally be released at the current decision boundary?
- How does the allocation watermark relate to the planning reserve $W_t$?

## Action execution

For each final LP action, determine the exact `LPServe` operations required to execute:

```
prefill x_i tokens
decode one token
preempt
do nothing
```

Determine how each action affects:

- `waiting`;
- `running`;
- sequence status;
- block-manager state;
- `SequenceScheduleMetadata`;
- `SchedulerOutputs`.

The architecture audit should answer these questions before Codex is asked to implement the scheduler.

---

# 14. Correctness Invariants

For every final integer action plan, the implementation must enforce

$$
\sum_i
\left(
\hat{x}_i+\hat{y}_i
\right)
\le
B_{\max},
$$

and

$$
\sum_i
\left(
\hat{I}_i^P+\hat{y}_i
\right)
\le
S_{\max}.
$$

*(amended 2026-10-12)* With $\hat H=F+\sum_{i\in\mathcal R_t}c_i^Z\hat z_i-\sum_ia_i^P\hat I_i^P$,
the integer resource and progress rows must also hold:

$$
\hat H\ge\sum_ic_i^D\hat y_i,
\qquad
\hat H\ge1\ \text{if any decode is selected},
\qquad
\hat H\ge h\ \text{if any admission is selected},
$$

$$
|\mathcal R_t|-\sum_{i\in\mathcal R_t}\hat z_i+\sum_{i\in\mathcal Q_t}\hat I_i^P\le N_{\max},
\qquad
\sum_i(\hat I_i^P+\hat y_i)\ge1.
$$

The prefill indicator must be canonical:

$$
\hat{I}_i^P
=
\mathbf{1}
\left\{
\hat{x}_i>0
\right\}
\qquad
\forall i\in U_t.
$$

Additionally, the final plan must:

- decode only a decode-eligible request;
- prefill only when prompt tokens remain;
- schedule at least one token for every selected prefill action;
- never schedule more prompt tokens than remain;
- never exceed $C_{\max}$;
- never simultaneously prefill, decode, and preempt the same request;
- set $\hat{z}_i=0$ for every $i\notin\mathcal{Z}_t$;
- preempt only requests whose status, ownership, allocation, and in-flight state make preemption legal;
- charge the complete $a_i^P(t)$ for every selected unallocated prefill action;
- charge zero additional prefill memory for an already allocated resident partial prefill when $a_i^P(t)=0$;
- count every retained resident, including omitted residents, against $N_{\max}$;
- execute victims first, then prefills, zero-demand decodes, and one-block decodes, with emitted metadata in the same order;
- maintain LPServe/SLAI sequence-state invariants;
- maintain block-manager invariants;
- never duplicate a request;
- never silently lose a request;
- construct valid `SchedulerOutputs`;
- handle LP solver failure explicitly;
- handle extraction failure explicitly;
- handle physical allocation failure safely.

A mathematically feasible plan must not be allowed to corrupt LPServe state if its physical translation cannot be executed. The executor must revalidate state-dependent eligibility and block feasibility immediately before committing mutations.

---

# 15. Validation Requirements

The mathematical scheduler should be testable without starting an LLM server or requiring a GPU.

*(amended 2026-10-12)* The corrected CPU milestone requires the checks listed in
§15.7 of `docs/lp_scheduler_design.md`, derived from P10 step 9. In summary,
synthetic tests should cover:

- explicit continuous relaxation of $x_i$ and enforcement of
  $I_i^P\le x_i\le U_iI_i^P$;
- canonicalization of $\hat{I}_i^P=\mathbf{1}\{\hat{x}_i>0\}$;
- each amended row: token, action, actual memory, decode prerequisite,
  watermark ($h=0$ and $h>0$), resident capacity, and useful execution;
- multiple admissions sharing one watermark and omitted residents;
- zero free blocks with a zero-demand decode, and mixed decode demand;
- full-context admission cost, zero resident-prefill cost, and chunks that do
  not reduce $a_i^P$;
- necessary negative-utility first work, zero-LP-support execution, and
  integral-idle victims;
- rejected optional trials and strict-improvement values on both sides of the
  numerical threshold;
- the many-fractional watermark construction and LP-feasible but
  integer-infeasible states, labelled as abstract formulation fixtures where
  they lie outside the admission-validated domain (the manuscript's $F=3$,
  $h=2$ example has a three-block pool and fails admission validation). They
  are not validated runtime no-useful-work examples;
- coefficients and ranks that may change between decisions, stay fixed
  within one, and come from the same immutable capture without mutation;
  uniform coefficient inputs are preserved, and negative or nonfinite
  coefficients are rejected;
- no legal useful plan, and agreement with a bounded independent enumeration
  written in test code only;
- the zero- and positive-coefficient mass-preemption examples;
- solver-reported infeasibility, solver numerical failure, and malformed
  output, each kept distinct from exhaustive no-useful-work.

`LPServe`-specific tests should independently cover:

- conversion from actual `Sequence` state to LP inputs;
- waiting-request admission;
- partial-prefill scheduling;
- decode scheduling;
- preemption/recomputation;
- block accounting;
- queue/state updates, including victim reinsertion order, retained omitted residents, readmission to the running tail, and arrived work behind a future waiting head;
- construction of `SchedulerOutputs`;
- pre-registration rejection of provably inadmissible requests and rejection of obsolete configuration arguments.

GPU experiments should not be used to discover basic LP or state-transition bugs that can be exposed by synthetic tests.

---

# 16. Baseline-Comparison Policy

The primary scheduler comparison should use policies running within the **same `LPServe` serving framework** so that scheduler policy is the main changing factor.

Likely core baselines include:

- the new LP scheduler;
- SLAI;
- Sarathi as implemented in SLAI;
- other relevant scheduler policies already available in `LPServe`.

Be precise when naming baselines.

For example, a `VLLMScheduler` implemented inside `LPServe` should be treated as the vLLM-style policy provided by the SLAI/Sarathi research framework rather than automatically equated with the current upstream vLLM serving system.

Likewise, the Sarathi implementation in `LPServe` should be described as the Sarathi policy used in the SLAI framework unless the original external system is reproduced separately.

Experiments with separate serving systems may later be useful for external validity, but they should not be mixed into the main same-framework scheduler comparison without accounting for the serving-stack confound.

*(recorded 2026-10-12; discussion R02, R04, R05, R11)* The primary recovery
target is an **adapted** reference: the service and preemption preferences of
upstream vLLM v0.31.0 (commit `db9527a46873454610df6dbedf79a36d6bf1a7f6`) with
FCFS, chunked prefill, ordinary text generation, tensor and pipeline parallelism
one, synchronous scheduling, and prefix caching, speculative decoding, and the
long-prefill threshold disabled, applied within LPServe's full-context
allocation framework. It does not claim upstream allocation trajectories, exact
chunks or batches, physical execution order, or whole-engine equivalence.
"Youngest" means the tail of the current running order, and a necessary
preemption is one needed to serve the preferred request. This adapted reference
is distinct from LPServe's inherited `vllm_scheduler.py`. Never compute another
scheduler's schedule first and encode it into coefficients. Recovering this
behavior through coefficients is a research question, not a universal default
policy. How recovery and overhead are measured is OPEN under Q11 (§21).

## 16.1 Adapted vLLM reference specification

*(implemented 2026-10-18 at base `40652511bf4c7687eebaa4276f1751dc13820c6f`;
user-approved specification)* The adapted reference is
`AdaptedVLLMScheduler` in
[`adapted_vllm_scheduler.py`](../sarathi/core/scheduler/adapted_vllm_scheduler.py),
scheduler type `adapted_vllm`, configured by `AdaptedVLLMSchedulerConfig`
(`max_num_seqs`, `max_model_len`, one pipeline stage, and
`max_num_batched_tokens` as the combined budget B; all required). It shares
`VLLMBlockSpaceManager` and leaves the inherited `VLLMScheduler`, the LP
contracts, and the allocator unchanged. The LP's `c_max`, `s_max`, post-prefill
decode prerequisite, demand-group order, and useful-execution constraint are
not part of this policy.

Policy, from pinned upstream sources:

- Running pass in current running order, partial prefills included, with one
  budget: decode 1 token, prefill `min(remaining, budget)`
  ([scheduler.py L628–690](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/core/sched/scheduler.py#L628-L690)).
- On allocation failure, evict the running tail until the attempted request
  fits or is itself evicted, then stop the pass (FCFS branch,
  [L760–816](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/core/sched/scheduler.py#L760-L816)). Earlier accepted
  requests are never the tail.
- Skip waiting admission after any preemption
  ([L872–873](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/core/sched/scheduler.py#L872-L873)); otherwise admit in
  waiting order, stopping at the budget, resident limit counting every
  resident ([L885–893](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/core/sched/scheduler.py#L885-L893)), or the
  head's memory failure ([L1223](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/core/sched/scheduler.py#L1223)).
  Admissions join the running tail
  ([L1292](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/core/sched/scheduler.py#L1292)); victims are prepended to
  waiting ([L1580](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/core/sched/scheduler.py#L1580),
  [request_queue.py L92–94](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/core/sched/request_queue.py#L92-L94)).
- The admission gate checks the full current sequence plus watermark
  (`scheduler_reserve_full_isl=True`,
  [config/scheduler.py L191](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/config/scheduler.py#L191);
  [kv_cache_manager.py L505–536](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/core/kv_cache_manager.py#L505-L536)).
- Retained configuration checks: positive limits and
  `max_num_batched_tokens >= max_num_seqs`
  ([config/scheduler.py L329](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/config/scheduler.py#L329)); B may be below
  `max_model_len` because chunked prefill is enabled
  ([L315–327](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/config/scheduler.py#L315-L327)).

Adaptations and boundaries:

- **Resources:** LPServe full-context admission charge with its native
  `int(0.01 * pool)` watermark always applied; resident partial prefill costs
  0; decode costs its logical/physical gap (0 or 1). Upstream's default
  watermark is 0.0 and applies only when `running` is non-empty
  ([kv_cache_manager.py L505–513](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/core/kv_cache_manager.py#L505-L513));
  the shared LPServe rule is used deliberately (R05).
- **Arrival input:** future-dated waiting entries registered by benchmarks are
  skipped without reordering the arrived ones. An arrived request with an
  empty prompt is a visible pre-mutation unsupported state (no empty-prompt
  policy is defined); every selected prefill has positive work within its
  remaining prompt *(corrected 2026-10-18)*.
- **Workers:** one tensor-parallel worker and one pipeline stage; `EngineArgs`
  rejects any other `tensor_parallel_size` or `pipeline_parallel_size` before
  model metadata or configuration writes *(tensor check added 2026-10-18)*.
- **Context limit:** upstream caps tokens at `max_model_len`
  ([L685–690](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/core/sched/scheduler.py#L685-L690)); LPServe has no
  `max_model_len` stop, so a decode at `max_model_len` or an arrived request
  above it is a visible pre-mutation failure (provisional choice).
- **Execution layout:** the complete proposal and layout are validated before
  mutation; victims are executed first in eviction order; selected work is
  stably partitioned into prefills then decodes, preserving selection order,
  for both central calls and metadata. Commit order does not change selection.
- **Control-only:** a preempt-only proposal is raised as
  `AdaptedVLLMControlOnlyProposal` with the proposal before mutation, because
  the engine drops control-only outputs. A permanently inadmissible waiting
  head with nothing resident is also a visible failure.
- **Inherited, not repaired:** discarded final-prefill sample, recomputation
  output-counter reset (overgeneration after restart), stopping semantics,
  mixed-sampling association (greedy only is in scope), and possibly unequal
  central/worker physical block IDs.

Analytical consequence (not executed beyond the focused tests): with this
budget rule and B >= `max_num_seqs`, a reachable state has at most one partial
prefill, at the running tail, and every resident is served each decision unless
the pass stops by self-eviction. Order-scan and decodes-first selection
therefore coincide on reachable states; their difference appears only in
hand-constructed states.

---

# 17. SLAI Environment

The upstream SLAI repository documents a research environment based on:

```
Python 3.10
CUDA 12.1
Sarathi-Serve-derived dependencies
```

and reports testing on NVIDIA RTX ADA 6000 hardware.

The Unity hardware and software environment may differ.

Therefore, establish and validate a clean working `LPServe` environment on Unity before modifying scheduler code.

The environment/setup phase should verify:

- Python environment;
- CUDA compatibility;
- PyTorch and native-extension compatibility;
- required Python dependencies;
- successful package import;
- successful model loading;
- the smallest useful end-to-end `LPServe` run.

Only after unmodified `LPServe` works should scheduler modifications begin.

---

# 18. Implementation Source-of-Truth Policy

The local clone of the user's `LPServe` fork is the implementation source of truth and is the repository Codex can inspect and modify.

Repository:

```
github.com/AtivJoshi/LPServe
```

Upstream fork source:

```
github.com/agrimUT/SLAI
```

Codex prompts must therefore be self-contained and should specify:

- the objective;
- mathematical behavior required;
- relevant `LPServe` files to inspect;
- correctness invariants;
- tests to add or run;
- files that may be changed;
- actions that are explicitly out of scope.

When implementation behavior is uncertain, Codex should inspect the local `LPServe` source rather than being given assumptions about how the framework works.

The goal is to implement the required scheduling behavior in the simplest form that fits `LPServe`'s architecture. Future sessions should not broaden work into inherited framework repairs, speculative abstractions, exhaustive edge-case handling, or documentation bloat unless the issue blocks the active MVP path.

---

# 19. Recommended Implementation Sequence

The project should proceed approximately as follows.

## Phase A — Establish `LPServe` baseline

- clone or update `github.com/AtivJoshi/LPServe`;
- verify that it is based on `github.com/agrimUT/SLAI`;
- establish the Unity environment;
- install unmodified `LPServe`;
- perform the smallest useful GPU smoke validation;
- record the known-good setup.

No LP scheduler modifications should occur before this works.

## Phase B — Reproduce scheduler baselines

Run small controlled experiments with existing `LPServe`/SLAI/Sarathi scheduler policies.

Verify that:

- scheduler selection works;
- the experiment harness works;
- metrics are produced;
- results can be reproduced sufficiently for later comparisons.

## Phase C — SLAI scheduler architecture audit

Inspect:

- `BaseScheduler`;
- `Sequence`;
- scheduler configuration;
- block manager;
- `SequenceScheduleMetadata`;
- `SchedulerOutputs`;
- SLAI scheduler;
- Sarathi scheduler;
- relevant engine and sequence-state transitions.

Produce an explicit mapping from every mathematical LP input/output to `LPServe` state or operations.

Do not implement the LP scheduler during the architecture audit.

## Phase D — Mathematical LP layer

Implement the relaxed LP and extraction algorithm in a framework-light form that can be tested synthetically.

Validate the mathematical layer independently.

## Phase E — `LPServe` state mapping

Construct the required LP state from real `LPServe` sequence and resource state.

Validate the mapping carefully before allowing the LP path to mutate scheduler state.

## Phase F — LP action execution

Translate the extracted integer decisions into real `LPServe` actions:

- prefill;
- decode;
- admission;
- preemption;
- idle decisions.

Preserve block-manager and sequence-state correctness.

## Phase G — Integrated correctness validation

Run:

1. syntax/import/static checks;
2. targeted CPU tests;
3. synthetic scheduler-state tests;
4. a tiny GPU smoke test;
5. inspect actual scheduling decisions and state transitions.

Tiny LP-versus-baseline runs are useful follow-up validation once the basic
path runs; they are not an initial MVP gate. Do not rely only on aggregate
performance metrics.

## Phase H — Performance evaluation

Only after correctness is established:

- measure state-construction time;
- measure LP construction time;
- measure LP solve time;
- measure extraction time;
- measure action-execution overhead;
- measure total scheduler wall time;
- vary active-request count;
- compare LP, SLAI, Sarathi, and other relevant policies;
- design the broader experiment matrix.

---

# 20. Performance and Solver Considerations

A generic LP solver introduces scheduler overhead, so solver cost must eventually be measured rather than assumed negligible.

However, solver optimization should occur **after** the first correct implementation.

Measure at least:

- LP matrix/problem-construction time;
- solve time;
- extraction time;
- total scheduling time;
- scaling with $|U_t|$.

If solver overhead becomes significant, possible later directions include:

- warm starts;
- basis reuse;
- specialized LP structure;
- alternative solvers;
- analytical simplifications;
- reducing solve frequency.

These are optimization questions and should not complicate the first correctness-focused implementation unless necessary.

---

# 21. Open Research and Design Questions

Several choices should remain explicit rather than being buried inside implementation constants.

## Q11 — Adapted-policy recovery and overhead evaluation (OPEN, deferred)

**Status:** OPEN and deliberately deferred by the user on 2026-10-12. This
section is its primary durable record; the design and the discussion record
link here. Deferral is not rejection and establishes no policy fidelity. Q11 is
not a prerequisite for the corrected framework, but it is a prerequisite for any
quantified claim of policy recovery, fidelity, or scheduling overhead.

Target: the adapted vLLM v0.31.0 reference described in §16, within LPServe's
full-context framework. The following remain pending; none is selected:

- definitions of corresponding logical states and of service and preemption
  traces when policies use different chunks and batches;
- the remaining adaptation details and settings of the reference procedure;
- pinned revisions and profiles of Sarathi, SLAI, and any other secondary
  reference;
- coefficient formulas and settings evaluated for recovery (D/P/G scales and
  any discarded-work $\kappa$);
- workloads and numeric capacities;
- recovery criteria and thresholds for major deviations, and attribution of a
  deviation to objective expressiveness, extraction, ordering, or the common
  framework;
- bounded fairness, starvation, restart, and completion observations (delays,
  service gaps, unfinished and error counts, preemption and recomputation);
- separation of scheduler decision time from model execution time and from
  effects of changed batch composition.

No recovery or performance experiment is authorized by this entry.

## LP feasibility versus native execution (source-review limitation)

**Recorded 2026-10-18 at source revision `6d20d484a01063c0ab910ddb0cdf50ced5620022`.**
The LP scheduler's allowed plans can be narrower than the underlying framework's
execution capabilities. The allocator correspondence in section 5 concerns the
selected LP execution contract; it does not establish that every excluded plan
is impossible in the shared runtime.

Concrete example: one block is free, the admission watermark is zero, a
resident decode needs no additional block, and a waiting admission needs one
block. Assume the token and resident limits permit both actions. The LP's
post-prefill decode prerequisite rejects the combination because admission
leaves zero free blocks. Source inspection supports central decode-first
allocation followed by admission. It also supports prompt-first worker replay:
worker admission consumes the block, then the resident append allocates nothing;
the worker calls `can_append_slot()` but does not enforce its return value.
See the [LP executor](../lpserve_plan_execution.py),
[shared allocator](../sarathi/core/block_space_manager/base_block_space_manager.py),
and [worker replay](../sarathi/core/sequence_manager/worker_sequence_manager.py).
This is source-based reasoning, not an executed end-to-end validation result.
The example still permits useful individual actions; it does not establish a
stall, a restart loop, or an extraction bug.

The independently binding per-request chunk cap (`c_max`) and execution-action
cap (`s_max`) can also exclude work the framework permits. These are explicit
LP controls, not universal framework limits; see
[configuration definitions](../sarathi/config.py). Their effect depends on the
chosen values.

Comparisons must distinguish formulation/executor restrictions from coefficient
preferences and extraction choices. Do not automatically impose LP-only
restrictions on the primary adapted reference; label any restricted comparison
separately. Validate the counterexample with focused replay and end-to-end
checks before selecting a correction. This note changes no mathematical,
implementation, or execution contract and authorizes no experiment or repair.

## Utility design

What should

$$  
\alpha_i,\qquad  
\beta_i,\qquad  
\gamma_i  
$$

optimize?

Possible designs imply different behavior for:

- latency;
- throughput;
- fairness;
- SLO prioritization;
- starvation prevention;
- preemption.

## Memory-model construction

The prefill-memory structure for the initial LPServe-native scheduler has been resolved as a fixed admission charge:

$$
a_i^P(t)I_i^P(t),
$$

rather than a token-linear term $c_i^P(t)x_i(t)$.

The state-mapping layer should compute

$$
a_i^P(t)
=
\begin{cases}
0,
& \text{for an already allocated resident partial prefill},\\
A_i(t),
& \text{for a request requiring full-context allocation},
\end{cases}
$$

where $A_i(t)$ is derived from the request's current logical block requirement.

The preemption-recovery coefficient should be

$$
c_i^Z(t)
=
\text{the number of physical blocks currently recoverable from request }i,
$$

and preemption actions should be defined only over the legally preemptible set $\mathcal{Z}_t$.

*(resolved 2026-10-12, discussion R07)* $c_i^D(t)$ is the exact marginal block
demand of the next decode, zero or one. The append gate's free-block requirement
is modeled by the decode prerequisite rather than by charging every decode one
block. The project should still verify, in synthetic integration tests, that the
calculated $a_i^P(t)$, $c_i^D(t)$, and $c_i^Z(t)$ agree with actual block-manager
transitions.

## Memory reserve

*(resolved 2026-10-12, discussion R10 and R12)* The independent reserve $W_t$ is
removed. Its intended role, admission-watermark protection, is represented by
$H\ge hI_i^P$ for admissions. No final-state reserve is imposed. Historical runs
that used a reserve remain evidence only for their configuration.

## Victim selection quality

*(amended 2026-10-12)* Safety repair is superseded. Victims are added in
descending LP preemption value, then victim rank, until a trial fits. This keeps
choices close to the LP solution but does not minimize victims or penalties and
can produce mass preemption. Better victim selection remains a possible later
study if utility quality becomes important; no alternative is approved.

## Fractional structure

*(resolved 2026-10-12)* The at-most-three-fractional-request expectation does not
hold for the amended relaxation (§9). Observed fractional counts may be recorded
descriptively; they carry no correctness meaning.

## Approximation guarantee

The extracted plan has no fixed approximation ratio relative to the relaxed LP
or the original ILP: the manuscript gives a family whose ratio tends to zero. The
LP optimum is an upper bound on the ILP optimum only when the amended LP is solved
exactly. Do not claim an approximation ratio.

## Downstream mathematical revision

The manuscript sections after `subsec:lp_relaxation` require a separate
mathematical assignment before they can be relied on for the amended
formulation (§3.1).

## Solver overhead

Does a generic LP solver remain practical as $|U_t|$ grows under realistic workloads?

This should be answered experimentally after the scheduler is working correctly, under the measurement boundaries of Q11. Explicit watermark rows can make the constraint matrix quadratic in size, and extraction costs $O(u+nr+n\log n+r\log r)$ with incrementally maintained totals; neither statement is a measured speed.

---

# 22. Current Project Scope and Priorities

Use the research goals and working approach in section 2 when selecting the
next task. Its 2026-10-16 clarification guides research direction; dated
milestone records below retain their original evidence boundaries.

## 22.1 MVP compatibility mode

The immediate priority is a basic runnable LP scheduler. The initial LP path
uses existing LPServe/SLAI serving behavior, including native recomputation
preemption, without repairing pre-existing serving-framework defects. Known or
newly discovered inherited issues are documented as limitations unless they
prevent the selected MVP path from running. This is a compatibility choice, not
a claim that those behaviors are correct or repaired.

Planning should resolve only the choices required by the active MVP. Necessary
choices may be visible, provisional configuration or scoped experiment inputs;
defer non-blocking edge cases, broad hardening, generalization, and performance
optimization. Implement the smallest path needed now, and let unsupported
states fail visibly rather than adding speculative recovery or fallback logic.

The implementation is based on the following design principles:

- model each scheduling step as a myopic utility-maximization problem;
- solve a continuous LP relaxation of the amended ILP, including useful execution, resident capacity, actual decode demand, the decode prerequisite, and the admission watermark;
- convert the relaxed solution into legal integer actions with the one-pass LP-guided whole-action constructor, adding victims only as trials require;
- extend selected prefill chunks with residual token capacity only after the complete fixed prefill-admission charge has been checked;
- keep the mathematical planning-memory model conceptually separate from physical KV-block feasibility;
- keep utility design, policy ranks, and numerical tolerances explicit as research choices;
- design the software directly around `LPServe`'s scheduler and block-manager architecture;
- establish focused evidence for the supported MVP path before optimizing solver overhead.

The immediate objective is to establish a clean `LPServe` baseline, understand the SLAI-derived scheduler architecture sufficiently for the selected MVP, and then implement the mathematical scheduler natively inside the `LPServe` fork as soon as possible.

**Current milestone (2026-10-12).** The next milestone is the corrected CPU
framework (discussion Q12/P10). The mathematical correction and this
documentation alignment are recorded. Implementing the planning, mapping,
execution, admission, and configuration changes, and passing design §15.7,
remain pending. Historical CPU and GPU evidence applies only to its original
code and workloads. A GPU regression requires separate authorization after the
CPU checks pass. Recovery and performance experiments remain deferred under
Q11.

---

# 23. Guidance for Future Work

Future research and implementation sessions should normally begin from the current local `LPServe` repository and use:

1. this document for the project research context;
2. `docs/math/main-llm-serving.tex` for the mathematical formulation, especially the myopic ILP and Primal Heuristic 1 / LP relaxation;
3. `docs/lp_scheduler_design.md` for the normative scheduler implementation specification; and
4. the current local `LPServe` source code for existing implementation behavior and details.

When the local repository is available, use these repository-local sources rather than relying on uploaded or stale external copies.

The working repository is:

```
github.com/AtivJoshi/LPServe
```

and it is a fork of:

```
github.com/agrimUT/SLAI
```

When repository behavior matters, inspect the current `LPServe` code rather than assuming how a similarly named concept should work.

The normal workflow is:

```
mathematical requirement
        ↓
inspect current LPServe implementation
        ↓
derive the simplest correct LPServe-native design
        ↓
prepare a precise Codex task
        ↓
Codex modifies the local LPServe fork
        ↓
review the diff and test output
        ↓
document only what has been verified
```

This keeps the implementation grounded in the mathematical formulation while allowing the `LPServe`/SLAI architecture to determine the concrete software design.
