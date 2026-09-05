# LP Scheduler Research Context

## 1. Purpose

This document defines the research context, algorithmic requirements, implementation principles, validation requirements, and open design questions for implementing and evaluating an LP-based LLM scheduler in the SLAI/Sarathi research framework.

The central implementation principle is:

> Preserve the mathematical scheduling formulation while deriving the concrete software design from SLAI itself.

The document is intended to provide future ChatGPT chats with the context needed to reason about the project without requiring knowledge of any other serving-system implementation.

---

# 2. Primary Research Goal

The goal is to build a correct research prototype of a new scheduling algorithm for continuous-batching LLM inference and compare it fairly against existing schedulers, especially SLAI and Sarathi.

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

Three kinds of information must be kept separate.

## 3.1 Mathematical source of truth

The mathematical scheduling problem is defined in the uploaded:

```
main.tex
```

especially:

```
Myopic Utility Maximization: The Primal ILP Formulation

Primal Heuristic 1: Approximation via LP Relaxation
\label{subsec:lp_relaxation}
```

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

This document provides implementation principles, validation requirements, research assumptions, and open design questions for the project.

If this document, `main.tex`, and the current `LPServe` implementation appear inconsistent, investigate the discrepancy rather than silently choosing one interpretation.

---

# 4. High-Level Scheduling Problem

At every scheduler decision epoch \(t\), let \(U_t\) denote the set of non-finished, arrived requests relevant to the current scheduling decision.

Conceptually, \(U_t\) may contain:

- newly waiting requests;
- partially-prefilled requests;
- decode-ready requests;
- currently GPU-resident requests;
- recomputation-preempted requests that have returned to a schedulable waiting state.

Let

\[
\mathcal{Z}_t
=
\left\{
i \in U_t :
i \text{ is resident and legally preemptible at the beginning of step }t
\right\}
\]

denote the preemptible subset of \(U_t\). Waiting, already-preempted, finished, and otherwise non-preemptible requests do not belong to \(\mathcal{Z}_t\). Requests whose physical KV state cannot safely be released at the current decision boundary, including relevant pipeline-in-flight requests, must also be excluded.

The precise mappings of \(U_t\) and \(\mathcal{Z}_t\) to `LPServe`/SLAI `Sequence` objects must be derived from the current implementation and validated before scheduler state is mutated.

For each request \(i \in U_t\), the scheduler chooses at most one of the following actions for the next execution step:

1. process a positive number of prefill tokens;
2. generate one decode token;
3. preempt the request, if \(i \in \mathcal{Z}_t\);
4. do nothing.

The LP scheduler is intended to choose these execution actions directly. It is not merely a mechanism for assigning queue priorities. The canonical do-nothing action sets all action variables for the request to zero.

---

# 5. Myopic ILP

For every request \(i \in U_t\), define

\[
x_i(t)\in\mathbb{Z}_{\ge0},
\]

the number of prefill tokens processed in the step;

\[
y_i(t)\in\{0,1\},
\]

indicating whether one decode token is generated; and

\[
I_i^P(t)\in\{0,1\},
\]

indicating whether the request receives a positive prefill operation.

For every legally preemptible request \(i \in\mathcal{Z}_t\), define

\[
z_i(t)\in\{0,1\},
\]

indicating whether the request is preempted. For notational convenience, define

\[
z_i(t)=0
\qquad
\forall i\in U_t\setminus\mathcal{Z}_t.
\]

The request/system parameters are:

- \(P_i^{\mathrm{rem}}(t)\): remaining unprocessed prompt tokens;
- \(C_{\max}\): maximum candidate prefill chunk size;
- \(U_i(t)=\min(P_i^{\mathrm{rem}}(t),C_{\max})\): maximum prefill action available to request \(i\);
- \(a_i^P(t)\): fixed planning-memory cost of executing a positive prefill action;
- \(c_i^D(t)\): marginal planning-memory cost of one decode action;
- \(c_i^Z(t)\): planning memory recovered by legally preempting request \(i\);
- \(B_{\max}\): per-step token-volume budget;
- \(S_{\max}\): scheduled-action concurrency budget;
- \(M_t^{\mathrm{free}}\): free scheduling memory at the beginning of the step;
- \(W_t\): optional conservative memory reserve.

Under the audited Sarathi/LPServe allocation behavior,

\[
a_i^P(t)
=
\begin{cases}
0,
& \text{if request \(i\) is already resident and allocated},\\
A_i(t),
& \text{if request \(i\) requires admission or recomputation},
\end{cases}
\]

where \(A_i(t)\) is the number of physical KV-cache blocks required to allocate the request's complete current logical context.

For \(i\in\mathcal{Z}_t\),

\[
c_i^Z(t)
\]

is the number of physical blocks that would be released by legally preempting request \(i\) at time \(t\).

The request-specific utility weights are:

- \(\alpha_i(t)\): utility of scheduling one decode token;
- \(\beta_i(t)\): utility per scheduled prefill token;
- \(\gamma_i(t)\): penalty for legally preempting the request.

The one-step objective is

\[
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
\]

The global token-volume constraint is

\[
\sum_{i\in U_t}
\left(
x_i(t)+y_i(t)
\right)
\le B_{\max}.
\]

The global scheduled-action concurrency constraint is

\[
\sum_{i\in U_t}
\left(
I_i^P(t)+y_i(t)
\right)
\le S_{\max}.
\]

The planning-memory constraint is

\[
\sum_{i\in U_t}
\left(
a_i^P(t)I_i^P(t)
+
c_i^D(t)y_i(t)
\right)
-
\sum_{i\in\mathcal{Z}_t}
c_i^Z(t)z_i(t)
\le
M_t^{\mathrm{free}}-W_t.
\]

Chunked prefill obeys

\[
I_i^P(t)
\le
x_i(t)
\le
U_i(t)I_i^P(t)
\qquad
\forall i\in U_t.
\]

This pair of inequalities makes \(I_i^P(t)\) an exact indicator of a positive prefill action in the integer formulation:

- \(I_i^P(t)=0\) forces \(x_i(t)=0\);
- \(I_i^P(t)=1\) requires \(1\le x_i(t)\le U_i(t)\).

Decode causality is represented by

\[
y_i(t)
\le
\mathbf{1}
\left\{
P_i^{\mathrm{rem}}(t)=0
\right\}.
\]

Finally,

\[
I_i^P(t)+y_i(t)+z_i(t)\le1
\qquad
\forall i\in U_t
\]

enforces mutual exclusion between prefill, decode, and legal preemption. Because \(z_i(t)=0\) for \(i\notin\mathcal{Z}_t\), this constraint applies uniformly to all requests.

These constraints define the mathematical core of the first scheduler. The three global coupling constraints are token volume, scheduled action width, and planning memory. Prefill linkage, decode causality, action mutual exclusion, and preemption eligibility remain request-local restrictions.

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

If simple utility weights are needed during early smoke testing, they should be clearly labeled as temporary test settings rather than the final policy used in research experiments.

---

# 7. LP Relaxation

A true continuous LP relaxation must relax the prefill-token variable \(x_i\) as well as the binary action variables. The relaxed domains are

\[
x_i\in\mathbb{R}_{\ge0},
\qquad
y_i,I_i^P\in[0,1],
\]

and

\[
z_i\in[0,1]
\quad
\forall i\in\mathcal{Z}_t,
\qquad
z_i=0
\quad
\forall i\in U_t\setminus\mathcal{Z}_t.
\]

The request-local linkage constraints

\[
I_i^P
\le
x_i
\le
U_i I_i^P
\]

remain part of the relaxation. They prevent a full prefill indicator from being paired with fewer than one prefill token and strengthen the continuous formulation.

Let the relaxed solution be

\[
\tilde{x}_i,
\qquad
\tilde{y}_i,
\qquad
\tilde{z}_i,
\qquad
\tilde{I}_i^P.
\]

A standard LP solver may be used initially. SciPy/HiGHS is a reasonable first candidate if it fits naturally within the `LPServe` environment.

Solver choice must remain isolated from the surrounding scheduler logic so that it can be replaced without redesigning the mathematical layer. Solver status, numerical failure, infeasibility, and exceptions must be represented explicitly rather than converted into fabricated action values.

The relaxed solution is not an executable scheduler decision. It must first be converted into a feasible integer action plan.

---

# 8. LP Extraction Algorithm

The extraction procedure converts the relaxed solution into executable integer actions while maintaining the resource constraints.

## 8.1 Integral/fractional partition

For each request, inspect the relaxed action indicators

\[
\tilde{y}_i,
\qquad
\tilde{I}_i^P,
\qquad
\tilde{z}_i,
\]

where \(\tilde{z}_i=0\) by definition for \(i\notin\mathcal{Z}_t\).

A request is considered indicator-integral when all three values are numerically close to \(0\) or \(1\). Use an explicit numerical tolerance rather than exact floating-point equality. A tolerance such as

```text
1e-6
```

is a reasonable initial test setting, but it must remain explicit and must be validated.

Let

$$
U_{\mathrm{int}}
$$

contain requests with integral relaxed action indicators, and let

$$
U_{\mathrm{frac}}
$$

contain the remaining requests.

For \(i\in U_{\mathrm{int}}\), lock

$$
\hat{y}_i=\tilde{y}_i,
\qquad
\hat{z}_i=\tilde{z}_i,
\qquad
\hat{I}_i^P=\tilde{I}_i^P,
$$

and set

$$
\hat{x}_i=\left\lfloor\tilde{x}_i\right\rfloor.
$$

Because the relaxation contains

$$
I_i^P\le x_i,
$$

an integral value \(\tilde{I}_i^P=1\) implies \(\tilde{x}_i\ge1\), and hence \(\hat{x}_i\ge1\). Flooring therefore cannot produce an integral action with \(\hat{I}_i^P=1\) and \(\hat{x}_i=0\).

The remaining token and sequence capacities are

$$
B_{\mathrm{rem}}
=
B_{\max}
-
\sum_{i\in U_{\mathrm{int}}}
\left(
\hat{x}_i+\hat{y}_i
\right),
$$

and

$$
S_{\mathrm{rem}}
=
S_{\max}
-
\sum_{i\in U_{\mathrm{int}}}
\left(
\hat{I}_i^P+\hat{y}_i
\right).
$$

The remaining planning memory is

$$
M_{\mathrm{rem}}
=
\left(
M_t^{\mathrm{free}}-W_t
\right)
-
\sum_{i\in U_{\mathrm{int}}}
\left(
a_i^P\hat{I}_i^P
+
c_i^D\hat{y}_i
-
c_i^Z\hat{z}_i
\right),
$$

using the convention \(\hat{z}_i=0\) for \(i\notin\mathcal{Z}_t\).

The final plan must be canonicalized and validated so that

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

## 8.2 Relaxed state and integer state

The fractional-extraction routine needs two conceptually different pieces of information:

- the **tilded variables**, which describe the relaxed LP solution;
- the **hatted variables**, which represent the integer action plan being constructed.

The extraction procedure should therefore retain access to both.

The hatted state already contains the decisions fixed for $U_{\mathrm{int}}$ and must not lose or overwrite those decisions.

## 8.3 Preemption-first rounding

Dominant preemption rounding applies only to requests in

\[
U_{\mathrm{frac}}\cap\mathcal{Z}_t.
\]

For each such request, compare

\[
\tilde{y}_i,
\qquad
\tilde{I}_i^P,
\qquad
\tilde{z}_i.
\]

If preemption is the dominant relaxed action and \(c_i^Z(t)>0\), set

\[
\hat{z}_i=1
\]

before performing fractional capacity-consuming actions, and increase the residual planning memory by

\[
c_i^Z(t).
\]

This ordering is important because legal preemption is a capacity-producing action. Memory released through selected preemptions should be accounted for before additional decode or prefill actions are admitted.

Waiting, already-preempted, and otherwise ineligible requests are excluded because they do not belong to \(\mathcal{Z}_t\). The extraction procedure must not infer preemption legality solely from the numerical value of \(\tilde{z}_i\).

Ties must be resolved deterministically. The exact tie-breaking rule must be documented and tested.
## 8.4 Safety preemption

After dominant fractional preemptions are selected, the residual planning memory may still satisfy

\[
M_{\mathrm{curr}}<0.
\]

This can occur because the relaxed LP may rely on the combined memory contribution of several non-dominant fractional preemptions. The repair step may therefore need to select multiple legal preemptions before feasibility is restored.

The candidate set for safety preemption is restricted to unused requests satisfying

\[
i\in U_{\mathrm{frac}}\cap\mathcal{Z}_t
\qquad\text{and}\qquad
c_i^Z(t)>0.
\]

The intended procedure is:

```text
while M_curr < 0:
    select an unused request i from U_frac intersect Z_t
    with c_i^Z(t) > 0 and maximum remaining tilde_z_i

    if no such request exists:
        report extraction failure

    set hat_z_i = 1
    increase M_curr by c_i^Z(t)
```

The loop stops when

$$
M_{\mathrm{curr}}\ge0
$$

or extraction fails.

Restricting the candidate set to \(\mathcal{Z}_t\) prevents the mathematical repair procedure from producing an action that the LPServe executor cannot legally apply. Requiring \(c_i^Z(t)>0\) also ensures that every repair iteration makes strict progress toward restoring memory feasibility.

Choosing the eligible request with the largest remaining \(\tilde{z}_i\) keeps the repair decision close to the relaxed solution. This is an approximation-quality heuristic; it does not imply that the resulting integer solution maximizes the rounded objective.

## 8.5 Decode and prefill extraction

For each remaining unpreempted fractional request, compare

\[
\tilde{y}_i
\]

and

\[
\tilde{I}_i^P.
\]

A fractional decode may be selected only if all relevant residual capacities remain available:

\[
B_{\mathrm{curr}}\ge1,
\]

\[
S_{\mathrm{curr}}\ge1,
\]

and

\[
M_{\mathrm{curr}}\ge c_i^D(t).
\]

After accepting a decode, update

\[
B_{\mathrm{curr}}
\gets
B_{\mathrm{curr}}-1,
\]

\[
S_{\mathrm{curr}}
\gets
S_{\mathrm{curr}}-1,
\]

and

\[
M_{\mathrm{curr}}
\gets
M_{\mathrm{curr}}-c_i^D(t).
\]

A fractional prefill action may be accepted only if

\[
B_{\mathrm{curr}}\ge1,
\qquad
S_{\mathrm{curr}}\ge1,
\qquad
M_{\mathrm{curr}}\ge a_i^P(t).
\]

If these conditions hold, select

\[
\hat{x}_i
=
\min
\left(
P_i^{\mathrm{rem}}(t),
C_{\max},
B_{\mathrm{curr}}
\right).
\]

If \(\hat{x}_i>0\), set

\[
\hat{I}_i^P=1
\]

and update

\[
B_{\mathrm{curr}}
\gets
B_{\mathrm{curr}}-\hat{x}_i,
\]

\[
S_{\mathrm{curr}}
\gets
S_{\mathrm{curr}}-1,
\]

and

\[
M_{\mathrm{curr}}
\gets
M_{\mathrm{curr}}-a_i^P(t).
\]

Under the audited Sarathi allocation behavior, memory is a fixed admission gate for prefill rather than a per-token fluid capacity. Reducing \(\hat{x}_i\) does not reduce \(a_i^P(t)\). Consequently, prefill chunk size may be truncated by the remaining token budget, but not by dividing the remaining memory by a per-token coefficient.

If the complete fixed charge \(a_i^P(t)\) does not fit, the initial safe extraction policy skips that fractional prefill action. A later extraction policy may consider additional legal preemptions, but such behavior must be specified and tested separately.

The extraction routine therefore requires access to \(C_{\max}\), \(\mathcal{Z}_t\), and the per-request memory quantities \(a_i^P(t)\), \(c_i^D(t)\), and \(c_i^Z(t)\).

---

# 9. Almost-Integral Structure

The relaxed LP has three global coupling constraints:

1. token-volume capacity;
2. scheduled-action concurrency;
3. planning-memory capacity.

The remaining constraints are request-local, including:

- prefill linkage \(I_i^P\le x_i\le U_iI_i^P\);
- decode causality;
- action mutual exclusion;
- restriction of preemption support to \(\mathcal{Z}_t\).

Replacing \(c_i^Px_i\) with \(a_i^PI_i^P\) changes a coefficient in the existing global memory constraint; it does not introduce another global coupling constraint. Similarly, restricting \(z_i\) to \(\mathcal{Z}_t\) is a request-local domain restriction.

This block-angular structure suggests that an appropriate optimal basic feasible solution may contain only a small number of request blocks that are not at local extreme points. The current formulation motivates an expected at-most-three-fractional-request property.

However, the Fundamental Theorem of Linear Programming alone does not prove this bound. A formal result also requires:

- an extreme-point characterization of the request-local polytope;
- a block-angular argument relating the three global constraints to the number of nonintegral request blocks;
- a solver that returns an appropriate basic feasible solution.

Numerical tolerances, degeneracy, and solver crossover behavior may affect the observed fractional count. The scheduler must therefore record or otherwise expose the number of fractional requests and must not assume at runtime that \(\lvert U_{\mathrm{frac}}\rvert\le3\).

The extraction implementation must safely process an arbitrary fractional set. Cases that exceed the expected structural bound should be observable and tested.

Do not add new global coupling constraints merely for implementation convenience without reconsidering the structural argument.

---

# 10. Planning Memory Versus Physical KV Feasibility

A general token-incremental serving model may use the planning-memory expression

\[
c_i^P(t)x_i(t)
+
c_i^D(t)y_i(t)
-
c_i^Z(t)z_i(t),
\]

leading to the constraint

\[
\sum_{i\in U_t}
\left(
c_i^P(t)x_i(t)
+
c_i^D(t)y_i(t)
-
c_i^Z(t)z_i(t)
\right)
\le
M_t^{\mathrm{free}}-W_t.
\]

This abstraction is more general for serving architectures that allocate KV-cache capacity incrementally as prefill tokens are processed.

The audited Sarathi serving architecture used by `LPServe`, however, allocates the physical blocks required for a request's complete current logical context when the request is admitted. Choosing a smaller positive prefill chunk does not proportionally reduce this allocation. Conversely, a resident partial-prefill request has already incurred the allocation and normally requires no additional blocks for another prefill chunk.

Because of this limitation of the Sarathi serving architecture, the first LPServe-native scheduler uses the specialized fixed-charge expression

\[
a_i^P(t)I_i^P(t)
+
c_i^D(t)y_i(t)
-
c_i^Z(t)z_i(t),
\]

or, with preemption explicitly restricted to \(\mathcal{Z}_t\),

\[
\sum_{i\in U_t}
\left(
a_i^P(t)I_i^P(t)
+
c_i^D(t)y_i(t)
\right)
-
\sum_{i\in\mathcal{Z}_t}
c_i^Z(t)z_i(t)
\le
M_t^{\mathrm{free}}-W_t.
\]

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

- \(M_t^{\mathrm{free}}\) from the block manager's free-block count;
- \(a_i^P(t)=0\) for an already allocated resident partial prefill;
- \(a_i^P(t)=A_i(t)\) for an unallocated request requiring full-context admission or recomputation;
- \(c_i^D(t)\) from the marginal physical block demand of one decode step, subject to the selected conservative policy;
- \(c_i^Z(t)\) from the request's currently allocated physical block-table length;
- \(\mathcal{Z}_t\) from legal preemption status, ownership, allocation, and in-flight conditions.

The final LPServe action executor remains responsible for checking that selected actions are physically and operationally legal. It must validate the complete integer plan before mutating scheduler state and must serialize physical mutations in the selected commit order.

Therefore:

> LP planning-memory feasibility and actual block-manager feasibility are distinct requirements.

Do not claim that satisfying the scalar LP memory inequality proves that every selected action can be executed. Conversely, avoid expanding the LP merely to reproduce every allocator implementation detail. The initial mathematical layer should use the simplest meaningful planning abstraction and rely on exact physical validation at the execution boundary.

The memory reserve \(W_t\) remains a separate research decision because the current Sarathi allocation watermark does not apply uniformly to every allocation and append path.

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

## Request state

- Which sequences belong in \(U_t\)?
- Which resident sequences belong in the legally preemptible set \(\mathcal{Z}_t\)?
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
- How is \(a_i^P(t)\) computed for unallocated and already allocated requests?
- How is \(c_i^D(t)\) computed or conservatively approximated?
- How is \(c_i^Z(t)\) computed from the current physical block table?
- Does the proposed \(\mathcal{Z}_t\) include only requests whose KV state can legally be released at the current decision boundary?
- How does the allocation watermark relate to the planning reserve \(W_t\)?

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

\[
\sum_i
\left(
\hat{x}_i+\hat{y}_i
\right)
\le
B_{\max},
\]

and

\[
\sum_i
\left(
\hat{I}_i^P+\hat{y}_i
\right)
\le
S_{\max}.
\]

The fixed-charge planning-memory constraint must also hold:

\[
\sum_{i\in U_t}
\left(
a_i^P(t)\hat{I}_i^P
+
c_i^D(t)\hat{y}_i
\right)
-
\sum_{i\in\mathcal{Z}_t}
c_i^Z(t)\hat{z}_i
\le
M_t^{\mathrm{free}}-W_t.
\]

The prefill indicator must be canonical:

\[
\hat{I}_i^P
=
\mathbf{1}
\left\{
\hat{x}_i>0
\right\}
\qquad
\forall i\in U_t.
\]

Additionally, the final plan must:

- decode only a decode-eligible request;
- prefill only when prompt tokens remain;
- schedule at least one token for every selected prefill action;
- never schedule more prompt tokens than remain;
- never exceed \(C_{\max}\);
- never simultaneously prefill, decode, and preempt the same request;
- set \(\hat{z}_i=0\) for every \(i\notin\mathcal{Z}_t\);
- preempt only requests whose status, ownership, allocation, and in-flight state make preemption legal;
- charge the complete \(a_i^P(t)\) for every selected unallocated prefill action;
- charge zero additional prefill memory for an already allocated resident partial prefill when \(a_i^P(t)=0\);
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

At minimum, synthetic tests should cover:

- empty request set;
- all-integral relaxed solution;
- explicit continuous relaxation of \(x_i\);
- enforcement of \(I_i^P\le x_i\le U_iI_i^P\);
- rejection of \(I_i^P=1\) with \(x_i<1\);
- canonicalization of \(\hat{I}_i^P=\mathbf{1}\{\hat{x}_i>0\}\);
- numerical tolerance near \(0\) and \(1\);
- token-budget saturation;
- sequence-budget saturation;
- fixed-charge memory-budget saturation;
- decode causality;
- mutual exclusion;
- prefill chunk cap;
- exclusion of requests outside \(\mathcal{Z}_t\) from preemption;
- dominant fractional preemption over \(\mathcal{Z}_t\);
- deterministic tie handling;
- prefill truncation by residual token capacity;
- full acceptance when the complete fixed prefill charge \(a_i^P\) fits;
- rejection when the complete fixed prefill charge \(a_i^P\) does not fit;
- verification that reducing a chunk does not reduce \(a_i^P\);
- zero additional planning-memory cost for a resident partial prefill with \(a_i^P=0\);
- full-context admission cost for an unallocated waiting or recomputation request;
- more fractional requests than predicted by the expected structural rule;
- solver failure;
- solver infeasibility;
- solver numerical error;
- extraction failure;
- one legal safety preemption;
- multiple legal safety preemptions when one is insufficient;
- exclusion of zero-recovery candidates from safety preemption;
- failure when eligible positive-recovery candidates are exhausted while memory remains negative.

`LPServe`-specific tests should independently cover:

- conversion from actual `Sequence` state to LP inputs;
- waiting-request admission;
- partial-prefill scheduling;
- decode scheduling;
- preemption/recomputation;
- block accounting;
- queue/state updates;
- construction of `SchedulerOutputs`.

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

The goal is to implement the required scheduling behavior in the simplest form that fits `LPServe`'s architecture.

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
5. tiny LP-versus-baseline runs.

Inspect actual scheduling decisions and state transitions rather than relying only on aggregate performance metrics.

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

\[
a_i^P(t)I_i^P(t),
\]

rather than a token-linear term \(c_i^P(t)x_i(t)\).

The state-mapping layer should compute

\[
a_i^P(t)
=
\begin{cases}
0,
& \text{for an already allocated resident partial prefill},\\
A_i(t),
& \text{for a request requiring full-context allocation},
\end{cases}
\]

where \(A_i(t)\) is derived from the request's current logical block requirement.

The preemption-recovery coefficient should be

\[
c_i^Z(t)
=
\text{the number of physical blocks currently recoverable from request }i,
\]

and preemption actions should be defined only over the legally preemptible set \(\mathcal{Z}_t\).

The remaining memory-model research choice concerns \(c_i^D(t)\):

- use the exact marginal block demand of the next decode step; or
- use a conservative one-block charge aligned with the current append-feasibility gate.

Whichever decode policy is selected must be explicit, tested, and followed by exact physical validation. The project should also verify that the calculated \(a_i^P(t)\), \(c_i^D(t)\), and \(c_i^Z(t)\) agree with actual block-manager transitions in synthetic integration tests.

## Memory reserve

Is

$$  
W_t  
$$

necessary?

If so:

- what physical risk is it protecting against?
- should it be fixed or state dependent?
- how should it be calibrated?

## Fractional-repair victim selection

Repeatedly selecting the largest $\tilde{z}_i$ keeps the repair step close to the relaxed LP solution, but it does not guarantee the best rounded objective.

Alternative repair policies may be studied later if approximation quality becomes important.

## Almost-integral property

Under the exact implemented formulation:

- does the expected at-most-three-fractional-request property hold empirically?
- does the chosen solver reliably return suitable extreme-point solutions?
- do any `LPServe`-specific changes alter the theoretical argument?

## Approximation guarantee

The exact theoretical quality of the integer extraction relative to:

- the relaxed LP; and
- the original ILP

should be stated carefully.

Do not claim an approximation ratio merely because the LP solution is almost integral.

## Solver overhead

Does a generic LP solver remain practical as $|U_t|$ grows under realistic workloads?

This should be answered experimentally after the scheduler is working correctly.

---

# 22. Current Project Scope and Priorities

The implementation is based on the following design principles:

- model each scheduling step as a myopic utility-maximization problem;
- solve a continuous LP relaxation;
- convert the relaxed solution into deterministic feasible integer actions;
- process capacity-producing preemptions before fractional admissions;
- use safety preemption when necessary to restore planning-memory feasibility;
- use fluid chunking to exploit residual token capacity after the complete fixed prefill-admission memory charge has been shown feasible;
- keep the mathematical planning-memory model conceptually separate from physical KV-block feasibility;
- keep utility design, decode-memory accounting, memory-reserve policy, and numerical tolerances explicit as research choices;
- design the software directly around `LPServe`'s scheduler and block-manager architecture;
- establish correctness before optimizing solver overhead.

The immediate objective is to establish a clean `LPServe` baseline, understand the SLAI-derived scheduler architecture precisely, and then implement the mathematical scheduler natively inside the `LPServe` fork.

---

# 23. Guidance for Future Chats

Future ChatGPT chats should normally use:

1. this document for the project research context;
2. `main.tex` for the mathematical formulation;
3. the current local `LPServe` fork for implementation details.

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