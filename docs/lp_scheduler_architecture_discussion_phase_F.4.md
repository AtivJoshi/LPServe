# LP scheduler architecture discussion

Started: 2026-10-11.

This file holds the discussion agenda and decision record in the LPServe repository. It was moved here at the user's request on 2026-10-11. It does not amend the mathematical source, normative design, historical audit, or implementation. Recorded decisions answer only the specified parts of the agenda; remaining questions stay open.

## Working boundary

- Discuss one agenda item at a time and record the result before moving on.
- Keep mathematical requirements, source observations, analytical arguments, proposals, executed evidence, and unresolved questions separate.
- Keep the framework flexible, principally through state-dependent alpha, beta, and gamma. Do not compute a reference schedule first and encode its answer in coefficients.
- Compare request service and preemption order inside the LPServe/Sarathi/SLAI framework, using its full-context admission allocation. A whole-engine comparison against upstream vLLM is outside this discussion's target.
- Update only this discussion record during the discussion. Other repository documents, mathematics, and code remain unchanged until the architectural decisions are complete and the follow-on task is authorized. No tests, models, installations, staging, commits, or implementation prompt are authorized by this note.
- Seek a solution within the scheduler's own planning path. No explicit backup policy has been approved. Existing execution and failure contracts remain in force pending explicit amendments.

## Reference and repository provenance

The approved primary upstream reference remains vLLM v0.31.0, resolved commit `db9527a46873454610df6dbedf79a36d6bf1a7f6`. The Q01 discussion retained this reference. No moving production-version target is selected.

Previously approved reference settings: FCFS; chunked prefill enabled; ordinary text generation; tensor and pipeline parallelism both one; synchronous scheduling; prefix caching, speculative decoding, and long-prefill threshold disabled. Explicit token and resident limits remain without numerical values. The new comparison scope is an adaptation to the common framework; adaptation choices remain to be specified.

Read-only repository check on 2026-10-11:

- Repository: `/Users/ativsc/python/LPServe`.
- Branch: `main`.
- HEAD: `7f1457ef7bc63677e724ddba797ecd0a47fc6b41`.
- Existing modifications: `docs/math/AGENTS.md` and `docs/math/EDITING.md`.
- Historical architecture audit remains pinned to `c3e014363dd50e1830d7c85c3d043eab69fdc9e5`.

The repository guide, AGENTS.md, and architecture summary were inspected for this agenda task. No implementation or performance evidence was generated.

## Decisions and directions already recorded

### R01 — Useful execution is required

Status: user-approved requirement; not yet applied to repository mathematics or code.

The ILP must choose useful execution whenever a legal executable plan exists. Prefill or decode counts as useful execution; preemption alone does not.

The intended integer constraint is:

$$
\sum_i (I_i^P + y_i) \ge 1.
$$

Q03–Q05 must settle the meaning of legal execution, when this constraint applies, handling states without legal useful work, and preserving the requirement through relaxation and extraction. This approval does not select an extraction algorithm or establish a runtime guarantee.

### R02 — Compare policies within the common framework

Status: user-directed scope.

Retain the common framework's full-context allocation for this comparison. Reproducing upstream physical allocation, exact chunks, batch packing, or whole-engine behavior is not the goal. Unsupported upstream-specific behavior should be listed and excluded explicitly, rather than silently changing the comparison.

R04 records the primary recovery target. Quantified agreement criteria remain open under Q11; no equivalence theorem or measured agreement is established.

### R03 — Prefer resolving issues within the native planning path

Status: user-directed design preference; no mechanism selected.

Investigate queues, priorities, sorting, coefficients, and extraction before considering an explicit backup policy. Per-request coefficients are already accepted by the mathematical layer; their live construction is an implementation gap, not by itself a missing framework capability.

### R04 — Primary recovery target within the common framework

Status: user-directed scope recorded during Q01 on 2026-10-11; no mechanism selected or implemented.

- The primary target is the approved upstream vLLM v0.31.0 FCFS policy with chunked prefill: roughly decode-prioritized service and youngest-resident preemption when needed, interpreted using the source details below.
- Recover request service and preemption order as closely as the surrounding LPServe framework permits. Differences caused by its fixed resource and execution rules are accepted comparison boundaries, to be identified rather than silently hidden.
- Keep full-context admission allocation. A redesign to upstream incremental allocation is outside this recovery target. Exact prompt chunks, batch membership, physical packing, allocation trajectories, and whole-engine equivalence are not required.
- Prefer state-dependent alpha, beta, and gamma updated at each scheduling decision. General changes to queues, priorities, and sorting may be considered when they help support vLLM, Sarathi, SLAI, and other policies. This does not approve a particular ordering mechanism, coefficient formula, or extraction algorithm.
- Do not compute a reference schedule and then assign coefficients that encode its answer. No explicit backup policy is approved.
- Sarathi and SLAI remain additional policy targets and sources for mechanisms; their exact reference revisions/settings are not selected by this decision.
- The useful-execution requirement R01 remains accepted. Reference fairness, restart, starvation, and completion properties must be established in Q02 before further guarantees are selected in Q08–Q09.

Source clarification, separate from the scope decision:

- The [versioned optimization guide](https://docs.vllm.ai/en/v0.31.0/configuration/optimization/#chunked-prefill) describes decode priority.
- The [commit-pinned scheduler](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/core/sched/scheduler.py) scans running requests, including partial prefills, before waiting requests. FCFS allocation failure triggers eviction from the running tail. Waiting admission is skipped after preemption in that step.
- Thus, youngest means the tail of the current running order, not an unconditional original-arrival-time rule. Necessary preemption refers to allocation for the running request being attempted, not a proof that every alternative without preemption is impossible.

Comparison interpretation to make explicit when defining the adapted reference:

1. Differences between real upstream execution and the adaptation can follow from different resource mechanisms; those are within the accepted boundary.
2. The eventual LP-versus-adapted-reference comparison should use the same common-framework resource rules. Allocation differences between upstream and LPServe cannot by themselves explain disagreement between those two policies inside the common runtime.

This interpretation does not define or implement the adapted reference procedure. Its exact handling of framework restrictions remains to be specified. Quantified fidelity criteria and which guarantees are universal versus policy-specific remain unresolved. No performance or policy-equivalence validation was executed.

### R05 — Meaning of youngest, necessary, and the common comparison

Status: explicitly confirmed by the user during the transition from Q01 to Q02.

- Youngest means the youngest running request: the tail of the current running order, not the latest original arrival.
- Necessary means necessary to serve the preferred request; it does not mean that every alternative execution without eviction is impossible.
- LP and the adapted reference share the common framework's allocation rules. Differences between their decisions still need explanation; upstream-versus-framework allocation differences alone do not explain them.

### R06 — Policy order is separate from request identity

Status: user-approved direction in the latest Q02 annotation discussion; not implemented.

Use existing queue positions or policy priorities to guide selection and extraction; use stable request identity only as the final deterministic tie-break when policy priorities are otherwise equal. Do not impose FCFS on every policy. Victim execution/emission and front reinsertion must preserve the intended subsequent queue order. Keep the existing ownership of waiting and running requests.

This approves revisiting D-16 and the victim-order contract in D-21. It does not select the precise rank fields/comparators, alter prompt-first physical packing, authorize code or normative-document edits during this discussion, or establish that ordering alone guarantees progress or fairness. The later implementation must specify admission and requeue rank updates as well as comparisons within a decision.

### R07 — Exact decode consumption with zero-demand decodes first

Status: user-approved direction on 2026-10-12; not applied to mathematical source, normative design, or code.

- Supply the actual zero-or-one additional decode demand, d_i, instead of charging every decode one block.
- Keep the local framework's check that a block is free before each decode.
- Execute selected zero-demand decodes before selected one-block-demand decodes. Preserve policy order within each demand group and retain prompt-first execution. This is physical execution order of already selected work, not a new service-selection policy or an instruction to reorder the running queue.
- Require at least one free block at the beginning of the decode group whenever any decode is selected, and enough memory for all actual allocations.

The planned explicit ILP row is

$$
\sum_i y_i\le S_{\max} H,
\qquad
H=M_t^{\mathrm{free}}+\sum_i c_i^Z z_i-\sum_i a_i^P I_i^P.
$$

H is an abbreviation for an expression in existing variables, not a new decision variable. It is the number of free blocks after selected victims and prefills. Retain actual-consumption feasibility

$$
H-\sum_i d_i y_i\ge W_t.
$$

For an integer plan H is integral. Since the number of decodes is at most S_max, the additional row is equivalent to H being at least one whenever any decode is selected. Zero-demand decodes then leave H unchanged, and the remaining one-block decodes each have a block available by total-consumption feasibility. This is a guarantee for these integer resource checks under the supported synchronous, non-sharing, zero-or-one-demand assumptions. It is not an extraction or progress guarantee.

The continuous relaxation must contain the same new row, but fractional H and fractional actions do not establish executable whole work. F01 explicitly includes this issue. All extraction and integer validation rules must be updated consistently before these changes can be called complete. Execution, emitted metadata, worker processing, and output association must agree on the new order; no engine-output equivalence is established by the mathematical argument.

This amends the planned D-09 choice and within-decode portion of D-21. The older per-decode prefix-constraint proposal below is superseded for the approved execution order. Admission-watermark handling remains OPEN (P01 below). R10 subsequently removes the independent planning reserve once that watermark is represented explicitly.

Historical rationale confirmed by read-only history review: original design at `6fbc046`, §8.3, offered “a conservative one-block charge aligned with the current gate”; the mapper at `a0e35d7` implemented the provisional `conservative_one_block_v1` policy. The inspected record does not identify the A/B ordering example as the specific original motivation or resolve D-09 permanently.

### R08 — Include resident capacity in the mathematical formulation

Status: explicitly user-approved mathematical amendment on 2026-10-12; not applied to the source or implementation.

With R_t the current resident set, Q_t the unallocated waiting set, and N_resident the resident limit, add

$$
|R_t|-\sum_{i\in R_t}z_i+\sum_{i\in Q_t}I_i^P
\le N_{\mathrm{resident}}.
$$

No new decision variable is required. Omitted residents remain counted; resident prefill and decode do not consume an additional resident place. Supply resident membership/count/limit through immutable planning inputs. Extraction and final integer validation must enforce this constraint too.

This requires an explicit later amendment to design §6.4, which currently keeps this check outside the LP. Adding this row alone changes the original vertex bound from at most three fractional request components to at most four under the same integral-local-polytope assumptions. Progress, decode prerequisites, and any further shared constraints must be included when restating the full structural argument. This observation is not a rounding or liveness guarantee.

### R09 — Continue with LP relaxation and extraction

Status: user-directed priority on 2026-10-12.

Try to satisfy the required validity and useful-execution guarantees through LP relaxation and integer extraction. Solving the corrected ILP to completion is a last-resort architectural option if that direction proves inadequate. This does not approve adding an automatic exact-solver fallback, an alternate scheduling policy, or changing existing failure handling.

### R10 — Remove the separate planning reserve

Status: user-directed simplification on 2026-10-12, conditional on representing the admission watermark explicitly; no authoritative source or code edit yet.

The user clarified that the intended role of the planning reserve was admission-watermark protection. Replace that role with the admission-only constraint; remove the independent W_t term from the planned formulation/configuration rather than retaining an additional memory restriction. Actual-consumption feasibility in R07 becomes H - sum(d_i y_i) >= 0. The separately approved decode prerequisite remains.

This records the intended new design. It does not rewrite the historical mathematical source, which currently defines W_t as a reserve after all selected actions, or historical experiments that supplied nonzero reserve values. Later implementation must update the affected configuration interfaces explicitly rather than silently ignoring an old nonzero reserve argument.

### R11 — Coefficient-based recovery is a research target, not a universal default policy

Status: user clarification on 2026-10-12.

The framework supports multiple policies. The research question is whether appropriate state-dependent alpha, beta, and gamma can recreate the intended service/preemption behavior of the adapted upstream vLLM v0.31.0 policy. It is not a requirement that every coefficient setting mimic that policy. Document major deviations when evaluating a coefficient policy intended for vLLM recovery, including whether they arise from objective expressiveness, extraction, ordering, or the common execution framework. The established exclusions for exact chunks, batches, physical execution order, and whole-engine equivalence remain. Do not compute the reference schedule first and encode it in weights.

F02 continues to track repeated loss of work and indefinite omission; neither this clarification nor the approved resource constraints closes those issues. Quantified fidelity and fairness claims remain to be defined in the relevant later agenda items.

### R12 — Admission-watermark rule approved

Status: P01 explicitly approved by the user on 2026-10-12; not implemented.

Adopt P01 exactly: H >= h I_i^P for each waiting candidate, with the existing definition of H after victims and prefills and before decodes. Preserve victim-first/prefill-before-decode execution. Apply the watermark once to the admission group, not as an additive charge per request; resident partial prefill does not activate it. Omit redundant watermark rows when h=0. R10's removal of the independent planning reserve now has its approved replacement. Mathematical source, configuration, extraction, validation, and normative design must be updated together in the later authorized task.

### R13 — Explicit admission errors approved

Status: P02 explicitly approved by the user on 2026-10-12; not implemented.

Adopt P02's checks for an overlength admission context and full-context demand plus watermark exceeding the total pool. Distinguish these proven admission failures from current memory contention. Check new submissions before registration with engine records, workers, or scheduler and report an explicit request error. For an already registered context, preserve the structured pre-mutation run-stopping failure path; continuing other requests through request-local cancellation is not part of this decision. Admission validity does not guarantee future completion or justify reserving maximum possible future output.

### R14 — Progress applicability and error distinctions approved

Status: P03 explicitly approved by the user on 2026-10-12; not implemented.

Adopt P03's operating rules: ordinary idle only when no unfinished request has arrived; otherwise include the useful-execution row in planning after input/admission validation. Distinguish valid relaxation infeasibility, numerical/solver failure, and feasible-relaxation extraction failure. None of these errors counts as useful execution. Extraction failure does not prove that the request/state is impossible. No reference-policy fallback, automatic exact solve, or retry is approved. Checking the complete arrived set must remain correct after queue-order changes.

### R15 — P04 extraction and strict improvement approved

Status: explicitly approved by the user on 2026-10-12, including the recorded limitations; not implemented or tested.

Adopt P04's single LP-guided whole-action constructor and analytical progress argument for the approved resource model. After useful execution exists, additions requiring new victims must strictly improve utility; additions requiring no new victims may use the nonnegative test. Before first useful execution, negative or zero utility remains acceptable when needed for progress. No external backup, extra solve, or victim-count guarantee is added. The positive-coefficient mass-preemption example and other utility/restart limitations are accepted limitations, not repaired behavior. Numerical comparisons require documented handling in later implementation.

Q05's algorithm choice is settled. F01 remains open for mathematical/normative amendments, implementation, and validation under all approved resource constraints, rather than for choosing an extraction algorithm. F02 remains open. Q06 will specify policy ranks and queue evolution without silently replacing the approved LP-first extraction comparators.

Clarification recorded 2026-10-12 (user decision): R15's phrase "when needed for progress" does not restrict P04. The first legal trial in the scan is accepted regardless of utility, even when another legal useful plan has nonnegative utility. Keep this one-pass rule and record the avoidable negative-utility first action as an accepted utility-quality limitation. Do not adopt the proposed two-pass first-acceptance alternative. Legality and useful-progress guarantees are unchanged. The counterexample is recorded in `docs/math/EDITING.md`.

### R16 — P05 policy ranks and queue evolution approved

Status: explicitly approved by the user on 2026-10-12; not implemented or tested.

Adopt P05: immutable per-step service and victim ranks, ID as final tie-break, LP values primary during P04 candidate selection, victim-rank order for final victim calls/controls, and service-rank order within the approved physical execution groups. Retain waiting/running ownership and resident relative order; append admissions in their declared execution order; use existing front insertion for victims; scan all arrived candidates rather than assuming the waiting head determines readiness. The initial vLLM-oriented ranks use current running then arrived-waiting order for service and reverse running order for victims. These ranks do not guarantee exact service or tail-victim selection when unequal LP values or resource tradeoffs intervene. D-16/D-21 and input/mapping/executor amendments remain deferred to the later authorized implementation task. Q06's design choice is settled; tests and implementation remain pending.

### R17 — P06 coefficient policy discipline approved

Status: explicitly approved by the user on 2026-10-12; no implementation or validation authorized.

Retain per-token prefill rewards and finite state-dependent coefficients fixed during each decision. Study coefficient policies without computing a reference schedule first, and distinguish guarantees for exact ILP, exact relaxation, and the extracted plan. Do not add another objective term, priority solve, or framework constraint for Q07. The rank-based family remains a candidate; approval does not select final D/P/G values or claim general vLLM recovery. Q08/Q09 may affect the final utility formulas and Q11 the acceptance criteria. The bounded analytical guarantees and recorded limitations remain as stated in Q07.

### R18 — P07 approved; hard partial-prefill commitment excluded

Status: explicitly approved by the user on 2026-10-12; not implemented or tested.

Approve P07's optional discarded-computation penalty family, existing queues/extraction, and stated limitations. No numeric kappa or final preemption formula is chosen. The user explicitly does not want the hard service commitment discussed in Q08 section 5; the proposed mandatory partial-prefill action is excluded from this design, not left as an implementation prerequisite. No retention constraint, restart cap, extra solve, or history counter is authorized. This decision does not establish restart freedom or waive reporting persistent omission/recomputation problems. F02 now tracks behavior, evidence, and policy limitations; a hard milestone is not a blocker. Q09 separately considers the meaning and scope of fairness/starvation/completion claims.

### R19 — P08 fairness and completion scope approved

Status: explicitly approved by the user on 2026-10-12; not implemented or experimentally validated.

Keep legal useful progress as the universal planning requirement and evaluate fairness/service goals per intended policy. Add no universal equal-share constraint, hard service commitment, mandatory aging, or persistent client/history accounting. Later bounded comparisons must report delays, service gaps, completion/unfinished/error counts, and preemption/recomputation; investigate significant deviations from the adapted comparison policy rather than hiding them in aggregate throughput. Exact workloads, thresholds, and measurement procedures remain Q11 decisions. Preserve inherited-defect qualifications and the distinction between bounded observations and universal theorems. This approval does not select final coefficient settings or authorize experiments.

### R20 — P09 scope approved; Q11 deferred; Q12 planning requested

Status: approved direction, 2026-10-12; not implemented.

The user approved P09's five-area framework correction and requested a detailed chronological Q12 plan. Q11 remains important and explicitly OPEN: during the later authorized main-document update, record it in section 21 of docs/LP Scheduler Research Context.md and link it from the design. It is not a prerequisite for correcting the approved framework, but it is a prerequisite for quantified policy-recovery/overhead claims.

The user limits the forthcoming TeX changes to subsec:myopic_ilp and subsec:lp_relaxation, with a detailed visibly marked paragraph at the end of the latter explaining the amendments and the outdated downstream sections. Subsequent TeX sections must remain unchanged in that task. This narrows P09's earlier general recommendation to qualify downstream recovery assertions: record their staleness now, revise them in a separate assignment. Current authorization is to develop the plan and record it in this discussion only; mathematical/normative/code edits, tests, and an implementation prompt remain deferred.

## Q02 source review — Reference properties and problem origins

Status: source observations and analytical arguments recorded for discussion. No new implementation mechanism or fairness requirement is selected. No tests, models, or performance experiments were run.

### Evidence boundary

- Primary reference: upstream vLLM v0.31.0 at `db9527a46873454610df6dbedf79a36d6bf1a7f6`, under the approved configuration.
- Secondary implementation observations: the local Sarathi and SLAI files at LPServe HEAD `7f1457ef7bc63677e724ddba797ecd0a47fc6b41`. These observations do not select final Sarathi/SLAI benchmark configurations or claim equivalence to every upstream version of those systems.
- The inspected scheduler, policy, shared base block manager, and recomputation paths have no committed differences from the historical audit revision `c3e014363dd50e1830d7c85c3d043eab69fdc9e5`. This check does not repin that audit or extend its executed evidence.
- The existing mathematical source and implementation have not yet received R01's progress constraint.

### What the reviewed policies do

| Policy | Source observation | Limit of the claim |
|---|---|---|
| Upstream vLLM FCFS | Running requests are considered before waiting requests. Allocation failure triggers running-tail preemption; preempted requests return to the waiting front. Waiting admission is skipped after preemption in that step. | A concrete service/recourse procedure; this source review supplies no unconditional fairness or eventual-completion theorem for all workloads. |
| Local Sarathi | Decodes are scheduled before resident partial prefills, followed by waiting admission. Running order is sorted by original arrival age. Decode allocation failure evicts from the remaining running tail, or the current decoder if no such victim remains. Waiting admission may use FCFS or shortest-prompt sorting. | Continuing resident work is explicit. The local shortest-prompt option differs from FCFS. The paper's stall-free batching claim is not a claim that memory preemption never occurs in this implementation. |
| Local SLAI | Due decodes are selected by their last schedulable times, then resident partial prefills, then waiting admissions, then additional decodes. It evicts paused prefills when decode allocation fails, or the decoder if none remain. Waiting order can be FCFS, shortest prompt, or user tier followed by prompt length. | Deadline preference and resident continuation are explicit. These are not equal-service guarantees or general proofs against starvation. |

Local evidence:

- [Sarathi scheduling](../sarathi/core/scheduler/sarathi_scheduler.py), especially lines 120–184 and 194–239.
- [SLAI scheduling](../sarathi/core/scheduler/slai_scheduler.py), especially lines 125–126, 150–203, and 205–297.
- [Local FCFS priority](../sarathi/core/policy.py) uses original arrival age. This differs from interpreting youngest solely by the upstream current running-list position; Q06 must preserve that distinction.
- [Shared native preemption](../sarathi/core/scheduler/base_scheduler.py), lines 115–121.
- [Upstream commit-pinned scheduler](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/core/sched/scheduler.py).

### Progress, restart, and fairness interpretation

Source observation: these policies directly try concrete work in their preferred order. They do not reject a useful action because a numerical eviction penalty exceeds its reward. Sarathi and SLAI explicitly continue allocated unfinished prefills before admitting new prefills. Their native admission gate also prevents simply admitting another request when its full allocation does not fit.

Analytical interpretation: this continuation behavior avoids the earlier toy policy that deliberately alternates two unfinished prefills and discards the previous request every turn. That toy example is not evidence that the approved upstream FCFS policy, local Sarathi FCFS, or local SLAI actually follows that alternating behavior.

Source observation: memory-triggered recomputation remains present. The reviewed code provides no fixed bound on the number of restarts per request. This is not a proof of perpetual restarting under the approved configuration. No source or executed evidence reviewed here establishes that restart or starvation problems are rare.

FCFS helps protect earlier requests from overtaking by newly arriving requests. A conditional eventual-service argument still needs finite work ahead, admissible resource demands, continuing execution, and suitable stopping behavior. Equal token shares, bounded latency, and arbitrary-workload completion do not follow from the FCFS label alone. A complete proof for the adapted reference remains open.

Analytical shortest-prompt example: keep a long request waiting while enough new short requests arrive before each decision to consume all service capacity. If arrived requests are repeatedly sorted by prompt length, the long request can remain waiting indefinitely. Both local Sarathi and SLAI expose such sorting options without an aging term in that sorting key. This concerns an allowed policy variant and an ongoing-arrival workload, not an executed example or a claim that FCFS behaves the same way.

Head-of-queue allocation failure also stops waiting admission in the reviewed procedures. A later fitting request can therefore be delayed. This can leave capacity unused; it does not by itself prove that the entire step is idle, since resident requests may still execute. We do not equate the references with an exhaustive search over every legal plan.

### Problems to carry forward

| Issue | Origin | Relevant next questions |
|---|---|---|
| Idle may beat useful work when work requires costly preemption | Current integer objective/feasible set; exact integer optimization does not remove it | Q04; R01 is already approved |
| Fractional solution can yield no whole execution during extraction, despite useful integer work being possible | Relaxation/extraction | Q05 |
| Numerical reward policy can prefer restart over continuing an unfinished resident; no built-in starvation or lasting-progress requirement | Policy choice allowed by the current formulation, including choices made by state-dependent coefficients; not inherently caused by fractionality | Q07–Q09 |
| Fixed one-block planning charge for every decode can exclude native-feasible batches | LP-specific planning accounting, even with exact integer optimization | Q03 |
| Resident capacity and admission gates are checked only after mathematical selection | Mathematical-plan versus native-execution gap; fractional extraction is not the sole cause | Q03 |
| Fractional packing and victim emission use request identity, which does not represent all policy/requeue orders | Ordering/extraction/execution contract | Q06 |
| Live utilities are uniform | Policy construction not yet implemented; per-request coefficients already exist as mathematical inputs | Q07; do not call this a missing framework capability |
| Full-context admission, conservative native decode gate, and recomputation/output/replay defects | Shared framework behavior or inherited defects, with separately documented boundaries | Q03 and Q09; do not make repairing every inherited defect a prerequisite |

Two analytical examples clarify the first and fourth rows:

1. Suppose the only useful plan earns reward 5 and requires an eviction with penalty 8. Idle earns 0 and the useful plan earns -3, so the current integer objective prefers idle. R01 changes the admissible choices when legal work exists; its extraction consequences remain unresolved.
2. Suppose one block is free and two resident decodes each need zero additional physical blocks. With enough token/action capacity, the native append gate succeeds for both and neither consumes that free block. The mapper nevertheless assigns planning charge 1 to each, so the LP/ILP cannot select the pair without additional recovery. This is an extra planning restriction inside the common framework, not the excluded upstream incremental-allocation mismatch.

Sources for these classifications:

- [Current ILP](math/main-llm-serving.tex), the myopic formulation near lines 228–304 and extraction/utility analysis near lines 603–618.
- [LP construction and extraction](../lp_relaxation_scheduler.py), `_build_lp` and `extract_integer_plan`.
- [Mapped decode charge](../lpserve_state_mapping.py), lines 483–499.
- [Native block gates and allocation](../sarathi/core/block_space_manager/base_block_space_manager.py), lines 76–110; [full-context block charge](../sarathi/core/block_space_manager/vllm_block_space_manager.py), lines 12–13.
- [Execution checks](../lpserve_plan_execution.py), lines 340–354 and 376–388.
- [Current uniform utilities](../sarathi/core/scheduler/lp_scheduler.py), `_build_utilities`.
- [Inherited recomputation reset](../sarathi/core/datatypes/sequence.py), lines 172–177, clears output token IDs; stopping at lines 189–191 uses the current output length. This limits arbitrary-preemption completion claims and is shared by the local native paths.

### Focused primary literature context

These are examples of distinct mechanisms, not new recovery targets or approved extensions:

- [Sarathi-Serve, OSDI 2024](https://www.usenix.org/system/files/osdi24-agrawal.pdf), Section 4.2/Algorithm 3: ongoing decodes, then unfinished resident prefill, then new admissions. Its stall-free claim addresses stalls caused by batching prefill with decode; it is not a universal memory-recovery or fairness theorem.
- [SLAI paper](https://users.ece.utexas.edu/~gustavo/papers/BHD25j.pdf), Sections 4 and 6: the throughput-optimality theorem concerns RAD, a different scheduler. SLAI uses deadline-aware decode selection and continuing active prefills, with FCFS/SPF alternatives. Do not transfer RAD's theorem to the local SLAI implementation or interpret a latency evaluation as an all-workload fairness guarantee.
- [FastServe, arXiv v3](https://arxiv.org/html/2305.05920v3), Sections 4.1–4.2: periodic promotion addresses long waiting; choosing an initial service quantum using estimated prefill time avoids premature first-iteration interruption. Host-memory cache preservation reduces discarded work. These mechanisms are distinct from merely setting a large eviction penalty, and cache offloading is outside the current common allocator.
- [VTC, OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/sheng): an explicit input/output-token service measure and per-client accounting support a proved service-difference bound in its model. This concerns fairness between clients with pending work, not a direct guarantee of completion for every request under our recomputation mechanism.

### Consequence for the next discussion

We should not treat every reference policy as universally idle-free, equally fair, starvation-free, and restart-free. Equally, lack of a universal reference theorem does not excuse additional LP failures that can be demonstrated under corresponding common-framework states.

Carry the existing approved useful-execution rule forward. Resolve actual common-framework legality and planning restrictions in Q03, then its progress and extraction consequences in Q04–Q05. Do not yet select retention constraints, fairness counters, a backup, or a queue redesign.

## Q02 annotation follow-up — Required issues and concrete examples

Status: answers to the latest annotations, while remaining on Q02. The recommendations below are proposals, not approved amendments to the mathematics, normative design, or execution contracts. Source inspection only; the examples were not executed.

### Explicit required future issues

- **F01 — Useful integer execution must survive relaxation/extraction (OPEN).** A relaxed solution satisfying the intended progress row can still yield no whole prefill or decode. Resolve Q05 with an actual progress argument under the chosen native-legality assumptions; neither rejection as `no_progress` nor changing sort order alone establishes this. R01 is approved; subsequent R15 approves P04 with strict optional improvement. F01 now tracks remaining authoritative amendments, implementation, and validation of that choice.
- **F02 — Repeated lost work and indefinite omission (OPEN).** An integer optimum can still favor repeatedly restarting requests or indefinitely overlooking one. Resolve the policy assumptions and required guarantees in Q07–Q09, distinguishing useful work in one step from lasting progress and eventual service. Coefficient updates may help, but their guarantees through the actual scheduler remain unproved. No retention or fairness mechanism is approved.

These issues were already on the agenda. The explicit entries make them required follow-ups rather than incidental observations.

### Mathematical feasibility versus native executability

Two different failures are possible, even with an exact integer solver:

1. **Too restrictive.** One block is free. A and B are resident, ready to decode, and each next token fits in an already allocated block. Token and action limits permit both. The shared native allocator can execute both: its free-block check succeeds twice and neither action consumes the block. LPServe maps each decode to a planning charge of one, so its mathematical problem requires two blocks for the pair. It can omit work or choose unnecessary recovery relative to the shared allocator.
2. **Too permissive.** The resident limit is two. A and B are allocated residents and C is waiting. C's full allocation fits with the admission watermark, and token/action limits permit its prefill. A mathematical plan that only prefills C, without preempting A or B, can satisfy the current ILP's constraints. A and B remain residents even though omitted from execution, so admission would create three residents. The executor rejects the complete plan before mutation. `S_max` limits executed actions, not all allocated residents.

The second case is an intentional implementation/design boundary, not an unreported defect in the validator: design §6.4 explicitly keeps resident capacity outside the LP to retain the three-global-constraint structure. Closing the selection gap would require a design revision. Neither an exact solve nor the progress row alone makes the missing capacity rule appear.

Sources: [mapped decode charge](../lpserve_state_mapping.py), lines 483–499; [myopic ILP](math/main-llm-serving.tex), lines 270–304; [resident-capacity design](lp_scheduler_design.md), §6.4; [pre-mutation capacity and admission validation](../lpserve_plan_execution.py), lines 340–354.

### Why the decode overcharge is LP-specific, and what fixing it entails

Local Sarathi and SLAI call the shared allocator while constructing their schedules. They check whether a free block exists before a decode and append a new block only if the logical/physical block gap is one. They do not subtract an extra hypothetical block for each zero-gap decode. In the first example, both pass the memory check for A and B if their policy/token limits allow them. This is not a claim that their policies always select both, or that they never experience real memory contention.

Proposed direction: separate **blocks actually consumed** from **conditions required before an action**. Use the measured zero-or-one gap for consumption, and represent the native free-block prerequisite consistently with the chosen execution order. The existing mathematical definition of `c_i^D` already permits zero; changing the mapper's one-block charge is therefore not, by itself, a new variable or objective. D-09 nevertheless remains an OPEN implementation choice. The prerequisite needs additional treatment in the feasible-plan rules.

Simply changing the charge to zero is insufficient:

- With zero free blocks, a zero-gap decode consumes zero blocks but the native append gate still refuses it.
- With one free block, suppose A needs one new block and B needs none. Executing A then B fails at B's free-block check; executing B then A succeeds. A total-consumption inequality alone cannot distinguish these orders.

One coherent formulation direction is to require that at least one block remains **before every selected decode**, accounting for earlier selected actions in the executor's declared order, while subtracting only actual allocation. This can be expressed as conditional constraints using the existing action indicators; no new scheduling policy or allocator change is inherently needed. The exact formulation, its effect on the three-global-constraint argument, and extraction support are not decided here. We must not substitute an executor rejection check for a guarantee that mathematical selection finds executable work. A different physical action order would also be an explicit D-21 amendment, not an implicit change.

Sources: [shared append gate and allocation](../sarathi/core/block_space_manager/base_block_space_manager.py), lines 97–110; [Sarathi decode loop](../sarathi/core/scheduler/sarathi_scheduler.py), lines 137–157; [SLAI decode loop](../sarathi/core/scheduler/slai_scheduler.py), lines 160–182; [executor append simulation](../lpserve_plan_execution.py), lines 376–389; [decode-charge choice](lp_scheduler_design.md), §8.3/D-09.

### Ordering recommendation, without changing queue ownership

Direction subsequently approved in R06; implementation details remain open. Original recommendation: keep `waiting` as unallocated unfinished requests and `running` as all allocated residents. Distinguish a stable request identity from the policy order used for choosing work, extracting fractional choices, choosing victims, and reinserting them. Prefer existing queue positions or existing policy fields where sufficient; retain identity as a final deterministic tie-break. Do not make FCFS the universal framework order: other policies may order by deadlines or prompt length. No new queue container or secondary ownership ledger is demonstrated necessary.

For example, a preempted request with ID 20 can be at the waiting front, ahead of a request with ID 10. Sorting by identity considers ID 10 first, despite that queue order. Coefficients can still be constructed from these state/queue priorities; ordering should not independently precompute the reference schedule.

Victim emission also affects future service. If the desired victim sequence is C then B, native front insertion produces waiting order B then C. Current identity-ascending emission of B then C instead produces C then B. This difference matters across scheduling steps, even when the victim set is unchanged. It must be addressed explicitly if that requeue behavior is part of the target.

This recommendation revisits D-16 and the victim-order portion of D-21. It does not silently change either contract, does not require changing prompt-first physical packing, and does not solve the fractional-progress or memory-accounting issues by itself. The policy-rank representation, extraction comparator, victim comparator, and relation to execution order remain OPEN.

Sources: [current mapped identity order](../lpserve_state_mapping.py), lines 400–402; [executor ordering](../lpserve_plan_execution.py), lines 255–273; [native front insertion](../sarathi/core/scheduler/base_scheduler.py), lines 115–121; design D-16/D-21.

## Q02 clarification — Block capacity, planning legality, and proposed correction

Status: source inspection at unchanged HEAD `7f1457ef7bc63677e724ddba797ecd0a47fc6b41`. R06 records the ordering approval. All accounting/constraint changes below remain proposals. No tests or models were run. The examples concern the local LPServe block manager used by its LP, Sarathi, and SLAI schedulers, not upstream vLLM's different allocation procedure.

### Blocks and the actual LP execution path

A block provides KV-cache capacity for `block_size` token positions of one request. KV cache means the attention data saved for earlier token positions. `EngineArgs.block_size` defaults to 16; it is configurable and no new experiment setting is selected here. The physical bytes also depend on the model and data type. Unused positions in one request's allocated block are not generally available to a different request under this non-sharing allocator.

A decode does not normally require a fresh block every step. For this implementation, the exact allocation demand at a scheduling boundary is the number of logical token blocks minus the number of allocated physical blocks, already checked by the mapper to be zero or one for decode-ready residents. Use those existing counts rather than adding another token to `get_len()` by assumption: sampled tokens already update the logical blocks before the next scheduling decision. The following examples describe capacity required for the upcoming decode, avoiding that off-by-one ambiguity.

With block size 16, A can have one allocated block but need backing for 17 token positions at its next decode; it needs another block. B can have one allocated block and need backing for 16 positions; it needs none. Both are decode-ready, with prefill completed.

The current LP executor already calls the inherited `_append_slot`, which calls the shared block manager's `append_slot`. It only physically allocates a new block if the logical count exceeds the physical count. LP's extra charge occurs solely in planning: the mapper observes the gap but passes `decode_charge=1` for every decoder. It does not instruct the executor to wastefully allocate a block for a zero-gap request.

The mapper handoff documents this as the provisional `conservative_one_block_v1` policy, not a permanent resolution of D-09. Its analytical effect is conservative: budgeting a block for every selected decoder guarantees the per-decode free-block check, but excludes some executable combinations. This is not a physical requirement that each token needs its own block, and the inspected record does not justify it as a final recovery policy.

### Three distinct free-memory rules

1. **Local decode admission check:** `can_append_slot()` asks whether at least one block is free before a decode. Both A and B face that same check. The allocation operation consumes one block for A and none for B. One block need not remain after the final decoder.
2. **Local waiting-admission watermark:** admitting an unallocated context must leave `int(0.01 * total_blocks)` free blocks under the shared manager's default. This rule applies to admission; it is not a permanent decode reserve.
3. **LP planning reserve:** configured `W_t` must remain after the complete mathematical plan. It is an additional policy restriction, independently supplied. A positive reserve can exclude otherwise executable work; it must not silently stand in for either allocator check.

With one block free and no additional planning reserve, A then B fails the shared decode check at B; B then A passes both checks and ends with zero free blocks. This order dependence belongs to the shared check used by local Sarathi/SLAI as well as LP. The current LP overcharge rejects the pair in either order. Changing only the charge to its exact value would remove that overcharge but would not express the order-dependent check.

The conservative check is a framework behavior we are preserving, not a fundamental storage requirement for a zero-gap decode. Changing it would require an explicitly scoped shared-framework change and consistent behavior for all compared schedulers; no such change is approved or recommended as a prerequisite here.

### Resident limit and known mathematical/execution gaps

`S_max` counts requests selected for execution in this forward pass. `max_num_seqs` bounds allocated residents, including those omitted from this pass. Their numerical values can be equal without their meanings becoming equal. The generic base configuration's “batch size” description is misleading for this distinction; current LP configuration, design §6.4, and admission checks establish the actual behavior.

For a valid, stable supported state, the current resource/eligibility gaps identified by this focused review are:

| Case | Mathematical plan can pass | Execution rejection |
|---|---|---|
| Resident count | Admit C while A and B remain resident, with limit two; all token/memory/action constraints fit | Three residents exceed two |
| Admission watermark | Total pool 100 blocks, two free, waiting context needs two, LP reserve zero; all other limits fit | Admission would leave zero, below the one-block watermark |
| Maximum admitted context length | Context length 33 with configured maximum 32; full allocation fits, and a small prefill chunk satisfies the ILP | Executor rejects admission of the overlength context; the mapper currently makes such waiting requests prefill-eligible |
| Useful execution | Idle or preempt-only satisfies current integer constraints | Executor rejects `no_progress`; R01 is approved but not yet implemented |

These are not approximation errors. Overlength requests need an explicit rejection/lifecycle decision, rather than silently leaving them waiting forever or narrowing mathematical eligibility to conceal unsupported behavior. Corrupt ownership/block tables, unsupported concurrent execution, and post-mutation exceptions are separate invalid-state or runtime failures, not ordinary missing resource constraints in a valid-state ILP. This review does not prove absence of all serving-engine defects.

### Historical candidate for matching the shared allocation checks

Subsequent decision: R07 replaces this candidate's per-decode prefix constraints with zero-demand-first decode execution and one aggregate decode prerequisite. The admission-watermark part remains an unapproved candidate. R08 approves the resident constraint.

The ownership division remains: the mapper reads allocation facts once into immutable inputs; optimization chooses actions; the existing executor and block manager allocate/free actual blocks. The solver needs neither physical block IDs nor live allocator calls. Calling a current-state yes/no allocation check separately for every candidate is insufficient for a simultaneous plan: two candidates might each fit one free block individually but not both together.

Proposed immutable quantities: existing free-block count, exact zero-or-one decode demands, existing full-context admission demands and preemption recoveries, admission watermark, current resident count/limit, and a fixed within-pass execution order. No schedule is first computed by another policy. These are resource facts, not new utility knobs.

For clarity, this candidate keeps the existing victim-first, then prefill, then decode execution grouping. For a candidate integer plan define:

- F = initially free blocks + blocks released by selected victims.
- a_i = full-context admission blocks for a waiting request, and zero for resident prefill.
- d_i = exact additional decode blocks, zero or one.
- h = admission watermark; W = separately chosen LP planning reserve.
- R = current resident count; N = allowed resident count.

Retain actual-consumption feasibility:

$$
F-\sum_i a_i I_i^P-\sum_i d_i y_i\ge W.
$$

For each waiting candidate k, in the declared prefill execution order, require:

$$
F-\sum_{j\text{ before }k\text{ in prefill order}} a_j I_j^P
\ge (a_k+h) I_k^P.
$$

For each decode candidate k, in the declared decode execution order, require:

$$
F-\sum_i a_i I_i^P
-\sum_{j\text{ before }k\text{ in decode order}} d_j y_j
\ge y_k.
$$

Finally require:

$$
R-\sum_i z_i+\sum_{i\text{ waiting}}I_i^P\le N.
$$

Why these inequalities match the checks for whole actions: a selected admission must leave h blocks immediately after its allocation; a selected decoder must see at least one free block immediately before its allocation. Unselected candidates impose only nonnegative intermediate free memory, already implied by final nonnegative memory because no blocks are freed during execution after the initial victim stage. Resident count decreases with victims and increases monotonically with waiting admissions, so its final bound covers each intermediate admission too. Existing action eligibility and mutual exclusion remain required. This proves only these resource checks, assuming correct immutable counts, healthy non-sharing allocator state, the declared execution order, and the supported synchronous boundary.

This candidate uses existing action indicators and no new integer variables or large numerical penalty constants. It adds coupled constraints. Therefore the old argument relying on only three global constraints must be revisited, and extraction must preserve these new inequalities as well as useful execution. It is not a completed extractor design or a promise that the current extractor will work unchanged. Final whole-plan validation stays in place as a safety check, not a substitute for selection correctness.

### Follow-ups explicitly retained

- **F03 — Consistent allocation feasibility (RESOURCE RULES DECIDED; extraction/verification OPEN):** R07 selects exact decode consumption, zero-demand-first execution, and the aggregate decode prerequisite. R12 approves the P01 admission-watermark representation; R10 removes the independent planning reserve. Restate the structural argument and revise extraction for the complete chosen feasible set. Do not merge exact decode charges alone. Later checks must cover zero/one-block demand, zero/one/multiple free blocks, the mixed-decode case, admission-watermark boundaries, and selected preemptions. Keep allocation ownership with the shared manager and retain the single mapper build/no mutation before validation contracts.
- **F04 — Resident capacity and request rejection (RULES DECIDED; implementation/verification OPEN):** R08 approves the resident-capacity constraint and R13 approves P02 admission-error handling; implementation, extraction, and verification remain pending. Later checks must include omitted residents and replacing a resident through selected preemption. The intended guarantee is that valid-state integer plans satisfy the supported executor's predictable legality rules; it is not immunity to arbitrary engine failures.
- R06 is approved. Its exact representation and interaction with the resource-check execution order remain to be specified. Reordering candidates without updating the resource-check order would invalidate the argument above.
- F01 and F02 remain required. Fixing allocation feasibility does not establish extraction progress, fairness, or lasting progress after recomputation.

Sources: [block capacity](../sarathi/core/datatypes/block.py), lines 15–38; [default block size](../sarathi/engine/arg_utils.py), line 59; [logical token updates](../sarathi/core/datatypes/sequence.py), lines 77–119; [shared allocation checks and operations](../sarathi/core/block_space_manager/base_block_space_manager.py), lines 54–110; [mapped charges](../lpserve_state_mapping.py), lines 443–499; [executor legality](../lpserve_plan_execution.py), lines 206–219 and 327–389; [executor allocation calls](../lpserve_plan_execution.py), `_execute`; [LP configuration meanings](../sarathi/config.py), lines 430–443; [mapper handoff](handoffs/lpserve_state_mapping_handoff.md), lines 81–112; [current mathematical charges](math/main-llm-serving.tex), lines 251–265. Scheduler/block/sequence paths inspected here have no committed changes from the historical audit baseline; that audit remains pinned.

## Q02 checkpoint — Remaining issues after R07–R09

Recorded 2026-10-12. Only the discussion record has been edited; current code and mathematical source still have the previously identified gaps. These are source observations and analytical distinctions, not newly executed evidence.

### Required extraction follow-up

F01 remains OPEN and explicitly covers R01, R07, and R08 together: preserve useful execution, resident capacity, actual memory consumption, and the decode-start prerequisite through fractional relaxation and whole-action extraction. Fractional preemption credit may disappear when making whole decisions; H can be fractional in the LP. Locking existing integral actions and only repairing a negative scalar memory budget is not yet proved sufficient. Remaining admission rules must also be included once selected. The accepted integer constraints do not close this issue. R09 keeps LP relaxation/extraction as the primary approach.

### Exact integer solutions: unresolved rejection and behavior cases

| Issue | What remains | Where to settle it |
|---|---|---|
| Admission watermark | A full-context admission can fit the current ILP yet leave fewer free blocks than the shared admission check permits. Example: 100 total blocks, watermark one, two free, admission demand two, planning reserve zero. Its exact representation is not yet approved. | Q03 |
| Overlength waiting context | The mapper can make it prefill-eligible, while the executor refuses admission above max_model_len. Choose explicit rejection/lifecycle handling, not silent indefinite omission. | Q03 |
| No legal useful plan | R01 cannot require an impossible action. Distinguish no arrived work, temporary lack of legal work, and a request that can never be admitted under configured resources. A nonempty arrived universe currently reaches no_progress/fail-stop or solver infeasibility rather than a fully specified impossibility policy. | Q03–Q04 |
| Planning reserve | A positive W_t deliberately excludes some executable plans and can force extra recovery. Choose it explicitly; do not confuse it with the admission watermark or decode prerequisite. | Q03 |
| Legal but undesired policy decisions | Even an exact integer optimum can select service/victim order unlike the adapted vLLM policy, repeatedly discard work, or overlook a request. Resource feasibility and one-step progress do not establish recovery, fairness, or eventual completion. | Q06–Q09; F02 |
| Shared recomputation/output limitations | An executable plan can inherit generation-count reset, output representation, replay, or block-ID issues. These are not eliminated by exact optimization. Determine which affect the eventual claim without making all inherited repairs a prerequisite. | Q09–Q11 |

Invalid ownership, corrupted block state, unsupported concurrent calls, solver/numerical failures, and exceptions after mutation remain separate operational failure categories. No new backup or retry handling is approved. Changes to ordering must also preserve the arrived-request visibility assumed by the current head-based ordinary-idle check; that is a Q06 implementation obligation, not a new ILP constraint.

### Recommended discussion order

Q02's issue classification is sufficient to proceed. The recommendation is to take Q03 next, starting with the admission watermark, then overlength/never-admissible requests and the independent planning reserve. Q04 must settle progress applicability before Q05 designs extraction for the complete feasible set. It is appropriate to defer extraction design until then, but not to mark it resolved or begin implementation without it. No additional fairness mechanism or shared-framework repair is selected by this recommendation.

## Remaining admission/progress proposals — 2026-10-12

Status: P01–P03 were subsequently explicitly approved on 2026-10-12 and are recorded as R12–R14. The detailed proposal text is retained below as the approved specification for later incorporation. No mathematical source, normative design, or implementation edits have been authorized during this discussion; no tests or models were run.

### P01 — Admission watermark without a separate planning reserve

Let h be the shared block manager's admission watermark in blocks. Use R07's H: initially free blocks plus selected recovery minus all selected prefill allocation. For each unallocated waiting candidate i, add

$$
H\ge h I_i^P.
$$

Together with H - sum(d_i y_i) >= 0, this means H is nonnegative when there is no admission, and at least h when there is an admission. Do not sum watermark charges across requests: several selected admissions still require only the same h blocks remaining after the last admission. Resident partial prefill does not activate the admission condition. If h=0 the added rows are redundant and can be omitted.

Why the rule is exact for the shared admission check in the supported order: all selected victims are freed first; all selected prefills then run before any decode; only waiting admissions consume blocks during that prefill group. Free memory therefore decreases monotonically, with its smallest prefill-stage value H immediately after the last admission. H >= h suffices for every earlier admission too. No action-order-specific admission prefix constraints are needed under these assumptions.

Example: initially two blocks free, h=1, admission demand two -> H=0, admission prohibited. Initially three free with the same demand -> H=1, admission allowed. A subsequent one-block decode may consume that remaining block, because h is an admission check, not an end-of-step reserve.

No new decision variable is required. There is a row per waiting candidate when h>0. Its effects on LP fractionality and extraction must be analyzed with R01/R07/R08; do not keep the old three-constraint structural claim unchanged. Selected chunks, victims, shared allocator operations, and synchronous one-snapshot execution are preserved.

### P02 — Explicit errors for provably inadmissible requests

Recommend explicit errors for a known violation of the configured admission limits:

- current admission/recomputation context length exceeds max_model_len;
- full-context block demand A_i plus watermark h exceeds the total cache pool T, so admission cannot pass even if all blocks are free.

These checks prove inability to admit that context under the fixed configuration. They do not prove that every request passing them will eventually finish, and they do not justify reserving its full requested future output at arrival. Never label current contention (A_i greater than currently free blocks) as permanent impossibility. Invalid zero/negative scheduler budgets or nonsensical capacities are configuration errors, not impossible requests.

For a newly submitted request, the proposed location is after tokenization/size determination but before it is registered with the engine sequence manager, workers, or scheduler. Return/raise an explicit request error with request identity where available, reason, required length/blocks, and configured limit. This avoids leaving a rejected request partly registered and lets the caller decide whether to continue submitting other requests. It requires an explicit implementation change: current BaseLLMEngine.add_request registers with the sequence manager, then workers, then scheduler; adding only a rejecting scheduler.add_seq hook would be too late.

For a context already inside the engine (including recomputation), recheck admission legality before planning an admission. Use the existing structured pre-mutation failure path for now, explicitly noting that it stops the run. Request-local cancellation of an already registered request while the rest continue needs a separate coordinated lifecycle change; it is not authorized by merely asking for an explicit error. The existing prompt-ignore helper mutates the queue/status and is not a read-only mapper validation operation. Preserve the prohibition on silently narrowing mathematical eligibility to hide unsupported work.

### P03 — Progress applies to each nonempty arrived valid planning state

Recommend this simple operating rule:

1. Validate configuration and classify explicitly invalid/inadmissible contexts. No returned error counts as useful model execution.
2. If no unfinished request has arrived, return ordinary idle without optimization. Use the arrived request set; an ordering change must not hide an arrived request behind a future-arrival queue head.
3. Otherwise, solve the amended LP with the approved integer progress row sum(I_i^P + y_i) >= 1 included in its relaxation, and require the resulting integer plan to contain useful execution. Preemption may support that work but does not count by itself.
4. If valid relaxation infeasibility is established, no useful integer plan exists under those constraints. Return an explicit cannot-progress error with diagnostics; do not spin on idle in the supported synchronous state with no batch in flight. Numerical/solver failures are separate errors and are not infeasibility certificates.
5. If the LP is feasible but extraction fails or returns no useful work, report an extraction/planning failure. This does not prove that the state or any request is impossible. F01 must resolve how the intended extraction finds useful work whenever an integer plan exists; P03 does not solve that problem or authorize a backup.

This avoids a preliminary call to another scheduling policy or a mandatory exact-ILP feasibility solve. Infeasible integer instances do not require an invented feasible action: requiring work in the optimization allows infeasibility to be reported explicitly. Normal resource contention is handled inside the optimization through retention, selection, and legal preemption. A claim of no legal whole plan from a feasible relaxation would need an additional proof/certificate; extraction failure alone is insufficient.

### Pending decisions and evidence boundary

P01–P03 are approved as R12–R14. P01 selects the admission constraint; P02 distinguishes initial request errors from already-registered runtime failure; P03 specifies idle versus required work versus inability to progress. F01/F02 remain OPEN. The next design task can address extraction for the combined feasible set; no error-triggered backup is selected.

Sources inspected: shared admission condition in `sarathi/core/block_space_manager/base_block_space_manager.py`, lines 54–82; context-length refusal in `lpserve_plan_execution.py`, lines 327–354; mutating inherited rejection in `sarathi/core/scheduler/base_scheduler.py`, lines 123–135; registration order in `sarathi/engine/base_llm_engine.py`, lines 338–398; recomputation context expansion in `sarathi/core/datatypes/sequence.py`, lines 172–177. Examples and correctness arguments above are analytical, not executed tests.

## Q03 review — Coverage and remaining dependencies

Reviewed on 2026-10-12 at unchanged HEAD `7f1457ef7bc63677e724ddba797ecd0a47fc6b41`. The user's request to start Q03 is satisfied by this coverage review because its resource decisions were already discussed. No duplicate decision is needed.

| Q03 topic | Agreed treatment | Remaining dependency |
|---|---|---|
| Full-context prefill allocation | R02: waiting admission/recomputation allocates the complete current context; continuing an allocated partial prefill needs zero additional blocks. Chunk length does not reduce the admission charge. | Later mapper/executor validation must retain this distinction; no upstream incremental-allocation redesign. |
| Block rounding | Use whole blocks from the shared manager's context demand, equivalent to rounding context length up by block size in the supported non-sharing manager. No artificial block alignment for prefill chunks. | Actual block size and logical/physical counts must be used; do not introduce a one-token off-by-one at decode. |
| Decode allocation and check | R07: exact zero-or-one demand, zero-demand decodes first, and the explicit decode-start requirement. | F01 extraction; consistent execution/metadata/worker order; future supported-demand changes require revisiting the proof. |
| Resident and execution limits | R08 counts all retained residents separately from S_max's executing requests. | F01 must preserve both limits, including omitted residents and selected victims. Numerical capacity values remain experiment inputs, not chosen here. |
| Admission watermark | R12/P01 explicitly applies it after admissions and before decodes. | F01 and the LP structural argument must cover the new rows. Changing the action grouping would require rechecking this equivalence. |
| Separate planning reserve | R10/R12 remove it from the selected design. | Later configuration migration must be explicit; preserve historical runs that used it. |
| Impossible admission and no useful work | R13/R14 distinguish request errors, ordinary idle, proved infeasibility, and extraction failure. | Correct request registration boundary, arrival visibility, and numerical failure classification. No request-local cancellation after registration is selected. |

### Local Sarathi/SLAI memory contention

Already inspected, with a focused source recheck for this review: both use the same full-context admission allocation and shared free-block check, so both can encounter contention caused by those rules. Local Sarathi's decode loop frees victims from the remaining running tail, or preempts the current decoder if none remain. Local SLAI's decode loop frees paused prefills, or preempts the decoder if none remain. Both stop waiting admission at a head request that fails the shared allocation check. These mechanisms do not establish that contention is rare, that every request finishes, or that either scheduler performs an exhaustive search for useful work.

Sources: `sarathi/core/scheduler/sarathi_scheduler.py`, lines 137–157 and 205–224; `sarathi/core/scheduler/slai_scheduler.py`, lines 159–181 and 235–247. The earlier audit comparison for these unchanged paths remains valid; the historical audit is not repinned. Frequency of these cases remains an empirical question under Q11 if a later comparison needs that claim; no experiments were run and none are required to choose the resource rules.

### What later decisions can affect

- Extraction (Q05/F01) must enforce the full approved feasible set and useful execution. Accepted integer constraints do not prove that current rounding succeeds. The old three-global-constraint fractional bound must not be reused unchanged.
- Policy order (Q06) must be distinguished from physical decode execution order. Any proposal that interleaves allocations differently must revisit the admission/decode proofs; it must not silently weaken R07/R12.
- Retention or service guarantees from Q08/Q09 could add constraints and alter which progress plans exist. If approved later, their effect on extraction and progress applicability must be checked again.
- Overhead and recovery evaluation (Q11) must identify the final constraints, coefficient policy, and physical execution rules. Comparable resource inputs are required when attributing deviations to scheduling choices.

Q03's resource-policy decisions are covered. Implementation and validation are not complete. Q04's operating-rule decisions are covered by R01/R14; constructive progress through extraction remains in Q05. The next substantive unresolved design item is Q05. This review does not select an extraction algorithm, resolve F02 fairness/restarts, or authorize implementation.

## Q05 proposal — One LP-guided constructor of whole actions

Status: proposed and subsequently user-approved on 2026-10-12 with the strict-improvement refinement (R15); not implemented. Source review at unchanged HEAD `7f1457ef7bc63677e724ddba797ecd0a47fc6b41` and analytical arguments only. No tests, solver experiments, models, or performance runs were executed. R01 and R06–R14 are the constraints on this proposal.

### The amended LP need not have only a few fractional requests

The original three-request vertex bound does not carry over unchanged. More importantly, R12 adds a watermark row H >= h I_i^P for each waiting request; these rows can produce arbitrarily many fractional requests at an LP vertex.

Concrete analytical construction: h=1, initially two free blocks, nine waiting requests each requiring one admission block and having q_i=1, and token/action budgets of two. Provide at least two spare resident places. Each waiting request has unit prefill-token utility. A background resident can account for the other 98 blocks in a 100-block pool so the default watermark is one; give its execution zero utility and preemption a sufficiently large penalty, and place its LP action variables at zero. This is compatible with the accepted fixed-allocation model and nonnegative coefficients.

Set x_i=I_i^P=1/5 for every waiting request. Total execution is 9/5, satisfying the progress, token, and action constraints. H=2-9/5=1/5, so all nine watermark rows bind. The resident-capacity constraint also fits. With all other action variables fixed at zero and x_i=I_i^P, the nine equalities I_i^P=2-sum_j I_j^P determine a unique point; it is a vertex. It is optimal for the stated waiting rewards without preemption: summing the watermark rows gives total admission <=9/5. A large background-preemption penalty makes preemption unprofitable in the relaxation. All nine requests are fractional although a legal integer plan (one admission) exists.

The same construction with n waiting requests uses I_i^P=x_i=2/(n+1), so there is no constant bound independent of waiting count for the approved relaxation. This is not an executed workload or a claim that typical LP results always look like this. When h=0, the watermark rows vanish; the original proof can instead preserve the five shared linear totals for token use, action count (both upper and progress lower bounds), actual memory, resident count, and decode prerequisite, yielding a bound of at most five non-local-vertex request components under the same vertex assumptions. Neither bound is an extraction guarantee, and the constructor below does not require a vertex solution.

Do not implement exhaustive search over a supposedly always-small fractional set. A possible two-mode LP reformulation separating no admissions from admission-permitted states could restore a constant shared-constraint count, but requires changing the approved relaxation and additional solve work. It is not recommended as a prerequisite for basic progress.

### Proposed extraction rule

A valid LP solution supplies preferences. Whole-action feasibility is checked using the approved immutable resource facts. There is one constructor, no external reference policy, and no execution-time backup.

An optional trivial fast path may use an already integral feasible LP plan directly (flooring prefill token counts if indicators are exactly integral preserves the accepted constraints). It must pass full integer validation. This is not a guarantee of a small fractional set or a substitute for the general construction below; numerical classification must not be assumed to confer feasibility.

General construction:

1. Start with an empty *partial* plan and no committed victims. Sort execution candidates by descending LP execution indicator e_i = tilde(I_i^P) + tilde(y_i), then the approved policy order, then stable ID. Include every state-eligible execution candidate, even with e_i=0. This comparator is approved in R15; the policy-rank definitions are addressed in Q06.
2. For each candidate not already selected as a victim, propose one whole execution action. A decode uses one token. A prefill chunk is min(q_i, remaining_token_budget, max(1, floor(tilde(x_i)))); skip when fewer than one token/action slot remains. Under fixed-context allocation, using this positive chunk rather than one token does not change the candidate's memory or resident demand.
3. Consider that action together with the current partial plan. If resource checks fail, tentatively add legal resident victims that are neither selected for execution nor already victims, excluding the candidate itself. Examine victims in descending tilde(z_i), breaking ties by the approved victim policy order and ID. Integral-idle residents and zero-LP-preemption-support residents remain eligible victims. Add victims until the plan fits or the list is exhausted. This changes only a local candidate plan, never live queues or blocks.
4. If the candidate cannot fit even after all such victims, discard the tentative action and tentative new victims; continue the scan. If it fits and no execution is yet selected, accept the action and required victims even if its utility is negative: R01 requires progress. Once useful work already exists, accept an addition requiring new victims only if its incremental objective is strictly positive; if no new victims are required, nonnegative improvement suffices (R15). Incremental objective means new execution reward minus penalties of new victims. If that check fails, discard this candidate and its tentative new victims. Do not search other victim subsets for a better objective.
5. Never remove accepted execution actions during the rest of the pass. After scanning candidates, remaining token budget may extend already selected prefill chunks up to their q_i, in policy order. This cannot change admission cost, resident count, or other resource checks, and nonnegative beta means it cannot reduce the objective. No new requests or victims are introduced by that extension.
6. Return the complete plan only after full integer validation, including useful execution and the approved physical order. Retain complete executor prevalidation before any live mutation. Emit selected victims in the declared victim order; execute prefills first, then zero-demand decodes, then one-block-demand decodes with consistent metadata order.

The marginal-utility check does not claim optimal victim selection or an approximation ratio. It simply avoids knowingly lowering the accepted integer plan's objective to add extra work after the mandatory progress requirement is met. First-action selection can still lose utility relative to the integer optimum. Rankings and utility arithmetic need deterministic numerical treatment, but resource counts are integers.

### Small resource check for each tentative whole plan

Track initially free blocks plus selected recovery as F; selected prefill allocation as P; total selected decode demand as D; selected waiting-admission count as A; selected decode count as K; selected victim count as V; and initial resident count as R. Define H=F-P. A candidate plan fits exactly the approved resource rules when:

- selected tokens <= B_max and execution actions <= S_max;
- R - V + A <= N_resident;
- H >= D;
- if A>0, H>=h;
- if K>0, H>=1.

Additionally each action must be state-eligible, each positive prefill chunk must be <=q_i, and no request can be both executed and preempted. These are the integer forms of R07/R08/R12, with W removed. There is no per-block allocation simulation or new persistent ownership ledger. Candidate construction uses immutable inputs and a few local totals; the shared block manager still owns real allocation.

As long as a victim is eligible and does not conflict with selected execution, adding it increases free memory and releases a resident place. It cannot worsen any of these feasibility checks. Its utility cost affects desirability, not legality. Legal zero-memory-recovery victims must not be excluded if they release a resident place.

### Analytical validity and progress argument

Scope: valid synchronous non-sharing state, independent legal preemptions, fixed admission costs independent of positive chunk size, d_i in {0,1}, positive token/action/chunk budgets, and exactly the currently approved constraints. No hidden retention, victim-count, fairness, or joint-action requirements. LP input/solve/numerical failures remain separate errors under existing contracts.

Validity follows because only feasible whole additions are accepted, accepted execution is never removed, and all later modifications are checked. Final chunk extension reduces only remaining token capacity and stays within bounds. Integer resource conditions above imply the approved allocation checks in the approved order; final validation still catches implementation/invariant faults.

Progress proof: suppose a legal useful integer plan exists. Retain one of its execution actions (one token suffices if it is a prefill), retain its victims, and remove its other execution actions. Token and action use decrease; dropped admissions reduce memory and resident demand; decode demand decreases; dropping the final admission or decoder removes rather than strengthens its conditional prerequisite. The retained plan remains legal. Therefore every nonempty legal plan has a legal one-execution subplan.

Before the constructor accepts its first execution, failed trials leave the partial plan empty. For the execution action from that legal subplan, all of its witness victims are available except the request being executed. The constructor's fixed victim order eventually includes those victims, possibly with others. Additional legal victims only improve resource feasibility, so the action fits by that point. Every eligible action is considered, including zero-LP-support actions. The first feasible action is accepted regardless of negative utility. Thus, whenever a legal useful integer plan exists, construction accepts useful work; later additions cannot remove it.

Conversely, if every single-action candidate fails even after all other eligible victims, no useful integer plan exists under these assumptions. This is a constructive feasibility argument for this particular model, not an inference from a fractional solver solution or a general ILP algorithm. Invalid inputs, numerical errors, or violated assumptions must not be relabeled as proven infeasibility.

### Why integral-idle requests cannot remain locked

Even a small fractional set can need resources from an LP-integral idle resident. Analytical example: watermark h=2 (e.g. pool 200), free blocks three, two waiting admissions each costing two blocks with q_i=1, and an allocated background resident holding the remaining blocks. Both waiting requests can have I_i^P=x_i=1/2, giving H=1 and satisfying both H>=2 I_i^P and progress. High preemption penalty can keep the background resident at integral idle. Neither whole waiting admission fits the initial three blocks because it must leave two free; either can fit after preempting the background resident. Locking its LP-idle decision would prevent useful extraction. This is not an executed example.

### P04 review — Mass preemption is possible with positive coefficients

Analytical review on 2026-10-12; no solver run or executed test. The examples below assess the original P04 rule. The user subsequently approved P04 with the narrow refinement and these limitations (R15); no implementation change has been made.

The user's example is valid under the approved model: 111 blocks, free=0, h=1; A holds 10, B holds one, and 100 other residents hold one each. B has zero-demand decode reward 100; W needs 109 admission blocks; other rewards and all penalties are zero; token/action limits are two. The LP can optimally select B and z_A=0.05, giving H=0.5 and satisfying sum(y)<=2H. P04 rounds recovery by preempting A, then considers zero-support W first among remaining execution candidates. To admit W it needs 110 free blocks, so it adds all 100 remaining eligible victims. Its nonnegative marginal-utility rule accepts the zero-gain addition. Total victims: 101. This preserves legality and useful execution but adds large avoidable disruption.

This is not limited to zero coefficients. Keep those resources, set the prefill chunk cap to one (so W has x_W=I_W), give A preemption penalty 1 and each one-block resident penalty 0.1, give W prefill reward 10.5, and give every other resident zero-demand decode reward 0.001. B keeps reward 100. Thus all available execution rewards and preemption penalties are strictly positive. All victims cost 0.1 per freed block. The same fractional point y_B=1, z_A=0.05, all other variables zero is optimal, with utility 99.95. To verify optimality, write w=I_W=x_W and k for total other decodes. The decode-start row requires total recovered blocks >=109w+(y_B+k)/2. Therefore utility is at most 99.95 y_B - 0.4w - 0.049k <=99.95. Other constraints can only tighten that upper bound; the stated fractional point attains it. An LP solver is not required to return this optimum, but P04 must account for this permitted optimum.

With W first among zero-support execution candidates, P04 first preempts A for B, obtaining utility 99. It then accepts W and 100 additional victims because the marginal utility is 10.5-100*0.1=0.5>0. Total utility becomes 99.5. Preempting just one small resident and decoding B instead is legal and has utility 99.9, already better than the mass-preemption plan. Thus the issue includes both rounding a fractional victim into an unnecessarily expensive whole victim and evaluating later additions against that already degraded partial plan. This is a concrete drawback of extraction, not merely a zero-utility tie or a claim that the exact integer objective always prefers these 101 victims.

Narrow refinement, subsequently APPROVED in R15: after useful execution has been secured, an addition requiring any new victims must have strictly positive marginal utility; an addition requiring no new victims may retain the nonnegative test. Numerical implementation must distinguish demonstrable positive improvement from roundoff using a documented scale-aware comparison. Before the first useful action, negative or zero utility must remain acceptable when needed for the approved progress rule. This refinement does not change the validity/progress proof because it only rejects optional additions after useful execution is secured. It eliminates the user's zero-gain example but does not eliminate the strictly positive example above.

P04 has no victim-count bound, minimum-victim guarantee, or guarantee against avoidable recomputation. Positive gamma and the marginal check price preemptions but do not provide those guarantees. Requiring LP support for optional victim-creating additions would block the particular zero-support examples but not mass preemption for supported candidates; globally forbidding additional victims after first progress would restrict batch selection and policy flexibility. Neither restriction is adopted here. Removing redundant victims after constructing the full plan would not fix this example: all victims are needed once W is included. A stronger guarantee requires a separate explicit choice about reconsidering accepted work/victims, limiting preemption, or prioritizing retention; keep this OPEN with the existing P04 utility-quality and F02 restart limitations rather than silently adding another algorithm.

### Changes, costs, and explicit limits

- Replace irreversible locking of all integral indicators and the current dominant-preemption/scalar-memory-repair procedure with the unified construction. Do not freeze LP-idle/preempt choices or restrict execution to positive fractional support.
- Keep the ILP, its approved amended relaxation, one mapper build, one LP solve, synchronous state boundary, whole-plan prevalidation, selected final actions/chunks/victims, and existing failure contracts. No second scheduling policy or automatic exact solver is introduced.
- LP execution/preemption indicators guide both orders; alpha/beta/gamma also govern the marginal-utility check. The algorithm does not compute a vLLM schedule. It supplies no theorem of vLLM recovery, fairness, starvation freedom, bounded restarts, optimal utility, or whole-engine output correctness.
- Sort once. With local resource totals, a straightforward implementation takes O(n*r + n log n) worst-case work, where n is execution candidates and r is eligible resident victims (both bounded by the planning-universe size). Trials can rescan victims, so this can be quadratic; no runtime speed claim is made. Do not add subset search to save a small amount of utility.
- The almost-integral structure can help preferences in favorable states because fully selected LP actions come first, but correctness does not depend on few fractional requests. Proposed fast-path/integral handling must obey finite-precision validation; no solver basis certification is needed for the general proof.
- The guarantee is *useful work whenever a legal useful plan exists*, not a legal plan in an infeasible state. If LP feasibility and successful construction diverge, distinguish mathematically proved integer infeasibility under these assumptions from an implementation/extraction failure.
- Future retention, limits on preemptions, group quotas, minimum batch size, shared cache, or action-dependent allocation could invalidate the deletion/monotonicity argument. Revisit the proof if any is approved. F02 remains OPEN: within-step progress does not ensure lasting progress or eventual completion.

P04 and the strict-improvement refinement are approved in R15 as changes to the extraction design and mathematical algorithm; implementation and authoritative-document amendments remain deferred. Source review identifies the current behavior in `lp_relaxation_scheduler.py`, `extract_integer_plan`, and the original vertex/extraction proofs in `docs/math/main-llm-serving.tex`. Later authorized validation must include the fractional-watermark example, the integral-idle-victim example, one/all zero-demand decodes, mixed decode demand, admission watermark and resident-count boundaries, negative-utility necessary progress, no legal useful plan, and exact action/metadata replay. None of these checks has been executed for this proposal.

## Q06 proposal — Policy ranks and queue evolution

Status: P05 proposed and subsequently user-approved on 2026-10-12 (R16); not implemented. R06 already separates identity from policy order, R07 fixes physical decode grouping, and R15 approves P04's LP-first comparisons. Q06 fills the remaining ordering details. No new ILP variable, objective row, solver call, ownership queue, or state-version mechanism is proposed.

### Two temporary ranks, with explicit roles

Supply a service rank and a victim rank as immutable per-request inputs from the same decision state as the coefficients. Smaller rank means earlier. The existing lists supply positions; other policies can use available state such as deadlines or remaining work. These ranks are input values, not decision variables or a second computed schedule. Recompute once per scheduling decision and keep fixed through solve, extraction, and execution. Stable request identity is the final tie-break. Do not sort ownership lists every step to express changing policy priorities.

Use exactly these comparisons:

- P04 execution candidates: descending LP execution value, then ascending service rank, then ID.
- P04 victim candidates: descending LP preemption value, then ascending victim rank, then ID.
- Already selected victims, for calls and emitted controls: ascending victim rank, then ID. This is an explicit output-order rule; it need not equal the order in which extraction discovered those victims across candidate trials. All selected victims are freed before any execution, so this permutation preserves the approved resource totals and feasibility proof.
- Already selected execution: prefills, zero-demand decodes, one-block-demand decodes; service rank then ID within each group. Metadata must use exactly the same order as central operations. Remaining prefill-token extension in P04 also uses service rank then ID.

Two ranks are needed because service preference and victim preference can differ. For the vLLM-oriented profile, derive service rank from current running position followed by current arrived-waiting position, and victim rank from reverse current running position. Running includes resident partial prefills: do not silently make all decodes precede those partial prefills. A different policy can rank service by deadlines and victims by recomputation cost; no claim of complete recovery of any other scheduler follows from these examples. Directly sorting alpha, beta, and gamma into one list would mix per-token execution reward and eviction penalty, whose meanings differ.

LP values remain primary under R15. For example, running A,B,C with z_A=0.7 and z_C=0.2 considers A as a victim before C even though C has earlier victim rank. A profile using reverse running order therefore does not guarantee tail eviction; coefficient-policy fidelity and major deviations remain Q07/Q11 work. Policy ranks do not resolve multiple distinct optimal LP solutions. Determinism of extraction for fixed validated LP values is different from solver-independent identical schedules. Do not add tiny objective perturbations or extra lexicographic solves implicitly.

### Queue transitions and examples

Keep waiting as the owner of unallocated unfinished requests and running as the owner of all allocated residents, including omitted ones.

- New submissions append to waiting in submission order. Consider all arrived eligible entries; a future or currently unfit head must not hide a later arrived feasible candidate. This preserves R14 and P04's complete candidate scan. It deliberately does not add a universal head-of-line stopping rule.
- Serving or omitting a retained resident leaves its relative running-list position unchanged. A partial prefill becoming decode also keeps its position. Remove only completed requests and selected victims. Physical decode-demand grouping must not reorder the running list.
- Remove selected admissions from waiting and append them to running in the selected-prefill execution order (service rank then ID restricted to admissions). Leave unselected waiting entries in their existing relative order. This reuses the current executor's append-on-admission mechanism. A re-admitted request joins the running tail; do not restore its old running position from original arrival time.
- Execute/emit selected victims in the declared victim rank order, using existing front insertion for each. For running A,B,C and waiting D,E, the vLLM-oriented rank emits C then B; front insertion yields waiting B,C,D,E before removing any selected old waiting admissions. Selected new admissions and victims remain disjoint; a victim is not re-admitted in this decision. Under other victim ranks, reversal of the emitted victim list is the explicit resulting waiting prefix; no independent reinsertion policy knob is proposed yet.

Ranks are recomputed from the resulting state next step. Example: serving C while retaining omitted A and B leaves running A,B,C; the executor must not rebuild it as C,A,B. Physical zero-demand-before-one-demand execution likewise does not change this order. A policy wanting a different preference next step changes ranks rather than using accidental physical reorderings.

### Source observations, dependencies, and verification boundary

Local source reviewed at unchanged HEAD 7f1457ef7bc63677e724ddba797ecd0a47fc6b41: `BaseScheduler.add_seq` appends; `_preempt` inserts at waiting front; `lpserve_plan_execution._execute` removes selected victims/admissions, appends admissions, and leaves retained residents in place. `lpserve_state_mapping` currently sets order_key=(raw_id,), and local `FCFS.get_priority` uses original arrival age, not current list position. `LPScheduler._schedule` currently uses the waiting head in its early idle check; that must be revised consistently with R14 before queue-rank semantics can be relied on. Base scheduler and policy paths have no diff from the historical audit commit; the audit is not repinned.

Pinned upstream observations: [vLLM scheduler](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/core/sched/scheduler.py) scans running in list order (including partial prefills), takes the running tail under FCFS preemption, appends admissions, and prepends preempted requests. This motivates the proposed profile, not a proof that P04 recreates its selection. Upstream stops/defers admissions under procedural conditions that P04 does not universally impose; those deviations remain part of Q07/Q11. Sarathi/SLAI exact ranking profiles are not selected here.

Later authorized work must amend D-16/D-21, the immutable input schema, mathematical extraction description, mapping, executor ordering, and focused validation together. The ILP feasible set and utility expression are unchanged by P05. Preserve one mapper build, synchronous state, full pre-mutation validation, final selected actions/chunks/victims, and existing failure handling. Check omitted residents, phase transitions, multiple admissions, multiple victims/front-insertion reversal, re-admission, future arrivals ahead of arrived requests, unequal LP values versus ranks, and physical memory-demand grouping. No tests were executed. Rank support does not resolve mass preemption, repeated restarts, fairness, or starvation. Future requests for strict ordering, different reinsertion semantics, or fairness accounting need explicit decisions under Q07–Q09.

## Q07 analysis and proposal — Coefficient meaning and bounded guarantees

Status: analytical discussion and P06 recommendations subsequently approved on 2026-10-12 (R17); final coefficient formulas/scales remain unselected. R15/R16 are approved designs, not implemented behavior. No solver run, tests, models, or policy comparison were executed. Repository HEAD remains 7f1457ef7bc63677e724ddba797ecd0a47fc6b41. Only this discussion record is edited.

### Scope and items not reopened

Do not reopen resource legality/progress (R01/R07–R15), ownership/rank rules (R16), or the accepted P04 mass-preemption limitation. Restarts and recomputation-aware protection belong to Q08; starvation/fairness/completion to Q09; quantitative fidelity and traces to Q11. This question concerns what coefficient policies mean and what can be claimed for each optimization/extraction stage. A useful-work guarantee does not require a particular coefficient policy under the approved constructor's assumptions.

### Concrete, limited candidate coefficient family

P06 proposes retaining the current reward semantics, finite nonnegative snapshot-dependent coefficients fixed for the complete scheduling decision, and explicit distinction between a preference recipe and a recovery theorem. Do not add objective terms, solves, or ordering constraints for Q07 now.

One candidate family for later study is alpha_i=D/(1+r_i), beta_i=P/(1+r_i), and gamma_i=G/(1+s_i), where r_i is zero-based service position in the R16 profile, s_i is zero-based running position for eligible resident victims, and D,P,G are strictly positive finite utility scales. Use only the action-eligible coefficients; waiting requests cannot be victims. These are formulas generating the existing three coefficient inputs, not new ILP knobs or a computed reference schedule. D/P controls decode reward relative to prefill-token reward; G sets retention cost relative to execution. D=P is a possible starting member that adds no extra phase preference beyond running/waiting ranks; it is not an approved numeric setting. D>P adds a decode bias, including over resident partial prefill, so it must not silently be described as the exact upstream running-list rule. No particular scale values or final formula are selected. Other finite decreasing rank functions remain possible.

Example state updates: a waiting request's r changes when earlier entries leave or victims are reinserted; an admission joins the running tail and gets new r/s next step; completing prefill changes the available action from beta-weighted tokens to an alpha-weighted decode while retaining resident position. These react to state without precomputing another scheduler's decisions. This recipe intentionally does not yet price lost computation; Q08 may alter gamma. Waiting-time aging, deadline targets, and new history counters are not introduced here.

### Compare whole-action rewards, not coefficient magnitudes

The existing objective pays alpha_i for one decode, beta_i*x_i for a prefill chunk, and subtracts the sum of gamma for selected victims. Therefore alpha=8 versus beta=1 does not imply decode dominance over a 16-token chunk worth 16. Actual coexistence or competition still depends on the common resource constraints. An optional addition is judged with its actual chunk and newly added victims under R15, not its hypothetical full prompt reward.

Dividing beta by q_i would change the policy: beta_i=p_i/q_i makes a full q_i-token chunk worth p_i, but a smaller chunk x_i earns p_i*x_i/q_i. It is not a fixed reward for giving a request a turn, and q_i can change between steps. P06 recommends keeping the per-token meaning rather than adding this normalization as an alleged correctness fix. If later recovery work demonstrates a need to reward starting service independently of chunk size, an additional coefficient on the existing I_i^P could express that, but this objective extension is NOT proposed or approved now.

### What can be proved at a fixed state

1. Exact optimal-solution response: keep the feasible set and all other coefficients fixed. Increasing alpha_i cannot decrease y_i across exact optima before/after the change; increasing beta_i cannot decrease x_i; increasing gamma_i cannot increase z_i. This holds for the integer and relaxed problems separately when both are solved optimally. Proof: write the two optimality inequalities and add them; the coefficient increase times the corresponding action difference has the required sign. This says nothing comparable about other requests, about the prefill indicator when x increases, about changing state/other coefficients, or about the extracted output. Numerical/approximate solver output also needs its own qualifications.

2. Finite strict preference is possible for the integer optimum under explicit bounds. For an eligible preferred request a, let L bound all other execution reward, for example L=sum_{j!=a} alpha_j + B_max*max_{j!=a} beta_j with ineligible terms omitted and empty maxima zero; let Gamma=sum gamma over legal victims. If some useful legal integer plan serves a and its minimum positive service reward (alpha_a for decode, beta_a for prefill) exceeds L+Gamma, every integer optimum serves a. Any plan serving a is worth at least that minimum minus Gamma, whereas a plan omitting a is worth at most L. This sufficient bound is deliberately conservative, not necessary, and does not determine victims or the rest of the batch. Applying dominating bounds through a whole priority hierarchy can create very large numerical ranges. Do not use arbitrary infinity or 10^9 as an unproved priority mechanism. The one-whole-action comparison does not automatically establish the same property for fractional actions or P04; those require separate analysis.

3. A restricted end-to-end order result is available. Assume the entire candidate execution set consists of n ready resident decodes; every subset of at most K=min(B_max,S_max,n) is executable without preemption; memory and the decode-start row impose no additional restriction on the relaxed decode choices; resident capacity is already valid. Give all legal victims strictly positive gamma and strictly decreasing positive alpha in current running order. Dropping preemptions then preserves feasibility and improves utility. The remaining problem is 0<=y_i<=1, sum y_i<=K (and progress), whose unique optimal execution vector selects the first K running requests. Both the exact ILP and LP select that set. P04/R16 then accept those K unit actions without victims and cannot add another action beyond the budgets, so the extracted selected set is the same. Their physical zero/one-demand ordering may differ, as already approved. This is a one-decision result; repeating it gives corresponding stepwise selections only while the stated assumptions and corresponding ordered states continue to hold. It does not cover waiting admissions, mixed prefill, memory pressure, or completion semantics. It is analytical evidence for a limited recovery claim, not an approved validation workload.

For general mixed states, an integer-optimum ordering argument must not be transferred to the LP or extraction. Fractional actions can obtain partial memory/reward advantages; P04 chooses by LP execution/preemption values before ranks and rounds victims as whole requests. The accepted mass-preemption example already shows a gap with positive coefficients. Updating coefficients each step does not remove that within-step gap or prove eventual completion.

### Manuscript claim boundary

At the checked revision, `docs/math/main-llm-serving.tex` near line 1197 claims perfect native-vLLM subsumption with alpha and gamma tending to infinity, beta=1, and preemption forced by zero free memory. That is not a proven result for the amended ILP, relaxation, or P04. The strongest limited reading is an abstract high-decode-reward, low-preemption preference under favorable resource assumptions; uniform beta does not supply queue order, and two unspecified infinite limits do not specify which preference dominates. The original formulation permits idle at zero reserve; the approved progress rule is conditional on useful integer feasibility, not an inference from free memory alone.

The separate passage near lines 1145–1152 refers to the generalized greedy Lagrangian fast path, not LP-relaxation extraction. Its original-arrival formulas and infinite-scale argument do not prove recovery for the R16 current-queue profile. Preserve the source text during this discussion; a later authorized mathematical edit needs a claim-specific theorem or a clearly qualified empirical preference statement. Do not transfer its assertion to P04 or infer what an earlier chat assumed.

### Dependencies and recommendation

P06 recommends retaining the objective's existing units, studying finite rank/state-based coefficients without precomputing schedules, and stating guarantees separately for ILP optimum, LP optimum, and extracted plan. The displayed family is a concrete candidate for discussion, not a final policy or an experiment authorization. Q07 does not justify another framework extension yet.

Pending dependencies: Q08 may change retention/recomputation penalties or constraints and require rechecking P04's monotonic-feasibility proof; Q09 may introduce service-history inputs or requirements and alter coefficient scales; Q11 must define acceptable service/preemption deviations and distinguish adapted-reference policy behavior from LP optimality/extraction effects. Final D-01–D-03 utility policies and numeric scales remain OPEN until that work supports an explicit choice. General exact recovery and solver-independent tie behavior remain unproved; no ordered multi-solve objective or guaranteed universal scheduler emulation is adopted.

## Q08 analysis and proposal — Discourage discarded work without claiming restart freedom

Status: P07 subsequently approved on 2026-10-12 (R18), with the hard partial-prefill commitment explicitly excluded. This is source inspection and analytical reasoning at unchanged local HEAD 7f1457ef7bc63677e724ddba797ecd0a47fc6b41. No solver experiments, scheduler tests, models, or performance runs were executed. R17 records approval of P06. F02 remains open: useful work in one decision is different from preserving work across decisions.

### Existing decisions help but do not settle Q08

R16 preserves resident order and returns victims to waiting predictably. Its vLLM-oriented profile prefers current running order to waiting order. R15 does not preempt an accepted execution action within the same decision, and demands strict positive utility for optional additions requiring new victims. These rules discourage some gratuitous disruption but supply neither a cross-step retention guarantee nor a bound on repeated restarts. The accepted mass-preemption limitation is not reopened.

A request may receive prefill tokens, be preempted, and later recompute those same tokens. Every step can satisfy the useful-execution rule without producing lasting progress toward completion. Merely retaining a resident also does not guarantee it receives service. A counterexample that alternates two unfinished prefills demonstrates a possible bad coefficient policy, not observed behavior of the approved vLLM profile.

### What the reviewed schedulers actually do

The pinned [upstream vLLM scheduler](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/core/sched/scheduler.py) considers running work first, uses FCFS tail victims when allocation fails, and skips waiting admission after preemption in that step. This avoids deliberately replacing resident work just to start a waiting request in that procedure. It is not an unconditional theorem of finite restart count or completion.

Local `sarathi_scheduler.py` considers decodes, then resident partial prefills, then waiting admissions; decode allocation failure can preempt a remaining running request or the current decoder. Local `slai_scheduler.py` serves due decodes, then resident partial prefills, then waiting admissions, with further decode work afterward; decode pressure can evict a paused prefill or the decoder. Their continuation preference is useful precedent, but unfinished prefills are not universally protected. These files and `sequence.py` have no diff from the historical audit commit c3e014363dd50e1830d7c85c3d043eab69fdc9e5. No fixed per-request restart bound or frequency evidence was established from these inspected paths; this is not proof that perpetual restart actually occurs in their approved configurations.

[FastServe v3, Sections 4.1–4.2](https://arxiv.org/html/2305.05920v3) illustrates distinct mechanisms: choose an initial scheduling quantum long enough for the predicted initialization, promote waiting jobs, and preserve cache through host-memory offloading. Its offloading mechanism avoids discarding the cache on each such pause; LPServe's recomputation path does not provide that mechanism. Do not transfer its guarantees or performance to a coefficient-only adaptation, and do not add offloading to this scope.

### P07: make discarded computation an optional explicit gamma input

For a policy intended to discourage recomputation, consider gamma_i(t)=gamma_i^rank(t)+kappa*L_i(t), with finite kappa>=0 in utility per cached-token estimate. L_i estimates the number of already computed context tokens whose cache would be discarded by evicting this resident now. It is not allocated capacity, full unprocessed prompt length, total lifetime work, or a wall-clock cost prediction.

For partial prefill, L_i is the existing processed-prompt count. Two 1,000-token prompts can have the same full allocation while one has processed 900 tokens and the other only 100; the extra penalties are then 900*kappa and 100*kappa. For ordinary synchronous decode, a simple logical estimate is current prompt length plus max(current output length-1,0): the most recently sampled output normally has not yet been consumed as an input. `BaseSequenceManager._process_seq_output` advances prompt progress and returns for prefill; the later decode path appends an output token. `Sequence.reset_for_recompute` merges current output into the prompt and resets prompt progress. This estimate must be checked against the existing first-decode/recompute boundaries in later implementation; it is not proof of actual worker-cache contents or a repair for inherited block-ID divergence.

Use existing sequence progress/output fields at the same synchronous snapshot boundary. Prompt progress is already represented in RequestStateSnapshot; output length or the derived estimate may need a primitive input for read-only utility construction. No new persistent restart ledger, second mapper build, model initialization, or timing profiler is needed for this candidate. Resource legality and memory recovery still use actual allocation/recovery facts, never L_i. The pure optimizer consumes the resulting gamma as before. With only a coefficient change, R15's feasibility/progress proof and R16's ordering rules are unchanged.

This is an optional coefficient-policy family, not a mandatory addition for every policy and not a new ILP constraint. Kappa=0 retains the rank-only family. Positive kappa can change victim preference away from the youngest resident, including when the younger request holds more computed work, so such a policy must be identified separately from the initial vLLM-recovery candidate. Do not claim it improves fidelity without evidence. No numeric kappa or final D-03 formula is selected here.

### Limits and consequences of that proposal

- Larger penalties discourage an eviction in an exact optimum at a fixed state, but do not impose a hard exclusion. Under P04 the first useful action can still require victims despite negative utility. Its accepted whole-victim rounding limitation also remains.
- L_i measures the current cache, not previous losses. If a request is repeatedly evicted after the same small amount of prefill, this term can repeat the same value; it does not accumulate immunity to eviction. Restart counters or accumulated-loss penalties require explicit history semantics and cannot be added casually or claimed to guarantee completion.
- High kappa can keep expensive residents while delaying waiting work or lower-cost requests. Token count is only a simple recomputation proxy: it is not equal GPU time across contexts. This tradeoff must be considered with fairness in Q09 rather than hiding it as a correctness fix.
- State-dependent alpha/beta/gamma can sometimes enforce retention under specified bounds, but Q07's integer bounds do not automatically transfer through the relaxation and P04. No universal restart-free guarantee is claimed for this finite penalty recipe.

### Historical stronger option — Explicitly excluded by R18

If a hard partial-prefill milestone is required, a simple explicit rule is available in the current fixed-allocation model: while any eligible resident partial prefill exists, select the earliest such resident a in running order, require I_a^P=1 at every decision until its current prefill completes, and hence z_a=0 by mutual exclusion. Give it at least one token. Existing variables suffice; this is an additional service constraint, not a change to mathematical eligibility and not merely a large gamma.

Why it is feasible in the stated scope: that resident's prefill costs zero new blocks, uses one action and at least one token, activates neither waiting-admission watermark nor decode-start check, and does not increase resident count. With a valid current state and positive budgets/chunk cap, serving it alone is legal. At least one token of its finite remaining prompt is completed per successful step; existing queue rules keep it the earliest resident partial prefill until it finishes, so it finishes its current prefill within at most that initial remaining-token count of successful decisions, unless externally cancelled or an execution error terminates the run.

Consequences: it reserves an action/token for that prefill even when the intended policy would serve urgent decodes instead, and prevents using it as a victim to make a preferred decode possible. With one action slot it can delay all decodes until that prefill completes. It is therefore a material policy restriction and a possible vLLM/Sarathi/SLAI fidelity deviation, not a neutral repair. P04 would need to honor the mandatory action before its ordinary scan, and its proof must be restated for that constrained problem. The current approved extractor must not silently assume this action will survive because its LP execution value equals one.

The rule guarantees only completion of the current prefill phase. It does not guarantee later decode completion, prevent eviction immediately after prefill, protect every other resident, or stop repeated full-prefill/restart cycles. Protecting until decode or request completion is harder because decode may need growing memory and the free-block gate. Thus no blanket protection, cooldown, restart cap, retained-cache mechanism, or extra mandatory action is adopted here. R18 subsequently excludes this hard commitment from the current design. Do not treat it as a pending implementation prerequisite or reintroduce it without a new user decision.

### Proposal and dependencies

P07 recommends keeping the approved constructor/queues, considering the finite discarded-work term as an explicitly labeled policy option, and adding no universal retention constraint or new history counter now. Accepting P07 would approve this approach and its limits, not assert that repeated restarting is solved or that a final coefficient policy is selected.

After R18, keep F02 OPEN for observed behavior and claim limitations, not as a requirement to adopt a hard milestone. Q09 determines how retention trades off with fairness and whether service-history inputs are necessary; a hard service/retention amendment must recheck P04's proof and the definition of useful feasibility. Q11 should distinguish evictions, estimated discarded computed tokens, actually repeated token computation, and advancement beyond prior attained progress. Eviction count alone cannot establish progress loss, and prompt-token throughput can count the same work repeatedly. Exact trace fields/history accounting remain a later validation-design decision, not permission to add an implementation ledger now. Inherited stopping/output reset defects remain relevant to any eventual-completion theorem and do not need universal repair for bounded restart-policy comparisons.

## Q09 analysis and proposal — Fairness, starvation, and completion claims

Status: P08 subsequently approved on 2026-10-12 (R19); implementation and experimental validation remain pending. R18 approves P07 and excludes the hard partial-prefill commitment. This section is primary-source review and analytical reasoning, not executed validation. Local HEAD remains 7f1457ef7bc63677e724ddba797ecd0a47fc6b41. The earlier no-fallback, useful-work, and resource/extraction decisions remain in force.

### Define different properties separately

- Useful execution: at least one legal positive prefill/decode action in a decision whenever such a plan exists. This is R01/R15; it says nothing about which request receives work.
- Starvation: an unfinished request is indefinitely denied service despite continuing opportunities for legal service. For a finite observation, report its actual wait/service gap or unfinished status, not proof of infinite starvation. A request repeatedly executing and losing the same work exhibits restart-induced lack of completion rather than literal absence of service; record it separately.
- Completion: a request eventually reaches its intended stopping condition after finite useful output. First service and frequent service are insufficient if recomputation repeatedly destroys progress. Correct final output/stopping semantics are separate from merely removing the request from scheduler queues.
- Fairness: a declared comparison of service across requests, clients, or priority classes. FCFS admission order, equal token shares, and equal latency are different properties. Two clients submitting 100 and one requests respectively need not receive equal aggregate service under request-FCFS. Client-level fairness requires client identity and accounting, neither of which is selected as a new LPServe requirement here.
- Latency and stability: first-token delay, output-token gaps, and completion delay are observable outcomes. Stability concerns backlog not growing without bound under the assumed arrival/load model; it is not a deterministic latency bound or an equal-service guarantee.

### Focused primary evidence on other policies

| Policy/source | What is supported | What not to infer |
|---|---|---|
| [Upstream vLLM v0.31.0 pinned scheduler](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/core/sched/scheduler.py) and [versioned tuning guide](https://docs.vllm.ai/en/v0.31.0/configuration/optimization/) | Running-list/FCFS and memory-recourse procedure; documented tradeoffs between first-token and inter-token latency. | No universal fairness, starvation, or arbitrary-workload completion theorem was established in the reviewed versioned sources. Results about older vLLM baselines in papers are not validation of v0.31.0. |
| [Sarathi-Serve, OSDI 2024](https://www.usenix.org/system/files/osdi24-agrawal.pdf), Sections 4.2 and 5 | Stall-free batching addresses decode stalls caused by co-running prefills. Evaluation reports median TTFT, P99 TBT, and serving capacity across the studied models/workloads. | It is not a theorem of equal service or absence of starvation under arbitrary memory pressure. The inherited local implementation and its options must be identified separately. |
| [SLAI paper, 2025](https://users.ece.utexas.edu/~gustavo/papers/BHD25j.pdf), Sections 4, 6, 7 | RAD has a throughput-optimality result under the paper's model and load conditions. SLAI is the practical latency-oriented scheduler; its reported experiment reduces median TTFT by 53% and increases capacity by 26% under the specified TTFT/TBT requirements on openchat_shareGPT4, Mistral-7B, RTX ADA 6000. | RAD's theorem does not establish SLAI's equal-service fairness or transfer to our inherited implementation. Reported latency statistics are not universal per-request completion guarantees. |
| [FastServe v3](https://arxiv.org/html/2305.05920v3), Sections 4.1 and 6.2 | Periodic priority promotion addresses starvation of long jobs. Evaluation includes 95th-percentile latency and throughput under tail-latency requirements. | This is a concrete mitigation plus empirical evidence; this review does not establish a universal starvation theorem for LPServe, especially without its cache-preservation mechanisms. |
| [VTC, OSDI 2024](https://www.usenix.org/system/files/osdi24-sheng.pdf), Section 4.1 | Explicit weighted input/output-token service accounting per client; Theorem 4.4 bounds service differences between continuously backlogged clients. Additional dispatch-delay guarantees have stated capacity/load assumptions; empirical results illustrate the fairness behavior. | Client fairness is not identical request latency or a theorem for arbitrary cache-discard/recomputation. Transferring the result would require its accounting and algorithm assumptions, not merely increasing request age weights. |

Theoretical versus empirical boundaries: continuously backlogged means a client keeps queued work throughout the interval. VTC's finite bound depends on bounded request/batch quantities and its specified service accounting, including when prefill service is charged. RAD's queue-stability theorem is for its analytical resource/arrival assumptions and admissible load. Neither proof automatically describes the approved LP relaxation/extraction pipeline. No broad literature survey or new scheduler recovery target is introduced by these examples.

Local source rechecked: Sarathi and SLAI can sort arrived waiting requests by prompt length. With an endless supply of preferred short requests, a long request can be repeatedly postponed under such a policy; this is an analytical possibility, not an observed rate or a claim about their FCFS configurations. These local sources and sequence reset/stop handling have no diff from the historical audit revision. No evidence reviewed establishes that starvation/restart pathologies are rare.

### Assumptions and a limited completion argument

A finite drain case means arrivals stop and the remaining workload has finite input/output requirements. For a simple analytical completion result, also require correct stopping semantics, successful finite-duration steps, no repeated recomputation (or only finitely many resets), and legal useful work available until all remaining requests finish. After the last arrival/reset, total remaining token work is a finite nonnegative integer and every successful useful step reduces it by at least one. Thus the system drains in finitely many steps. This argument requires no mandatory service commitment, but it does not apply to unlimited restart cycles or states with no executable work.

For ongoing arrivals, specify the arrival process, load relative to capacity, maximum request demands, and resource feasibility. Bounded average latency/backlog requires stronger load conditions than eventual service of an individual FCFS request. Overload alone does not prove starvation: an FCFS system can eventually serve each finite-index request while waiting times grow. Likewise finite first-service delay does not imply prompt/output completion when work is lost.

P08 proposes no unconditional fairness/completion theorem for every coefficient setting. For future bounded common-framework comparisons, failure to finish an admissible finite workload, persistent omission, or repeated restarts should be investigated and explained rather than hidden by aggregate throughput. A finite timeout is unfinished-at-the-limit evidence, not proof of infinite starvation. No specific workload, timeout, capacity, or experiment is authorized by this proposal.

### Coefficients and possible mechanisms

The existing state-dependent inputs can favor older or underserved requests. Aging may be a policy option, but do not silently add it to the vLLM-oriented profile: it changes service order and may increase preemption. A time-since-last-service reward needs clear reset semantics and possibly new persistent state; a client-fair policy needs client/service accounting. These additions are not recommended without a chosen policy target or a demonstrated discrepancy.

Larger age or service-deficit rewards alone do not prove a property through P04: resource fit, fractional LP values, whole-victim extraction, and restart costs still matter. Unbounded age growth also requires controlled numeric scaling. The R01 guarantee remains the universal planning property; fairness goals may differ between coefficient policies. Declining Q08's hard commitment does not silently weaken the existing useful-work requirement.

### Completion evidence and inherited defects

`Sequence.reset_for_recompute` moves generated tokens into the prompt and clears output_token_ids; `check_stop` compares max_tokens to the current output length. Repeated preemption after generation can therefore defeat the intended cumulative output limit. Apparent delayed completion or extra output must be separated from scheduler starvation. `RequestOutput` inconsistencies after reset and known text-conversion/mixed-output issues limit claims of correct delivered output. Existing central/worker block-ID divergence can undermine execution validity, not establish a fairness failure. Current no-control-only output and fail-stop contracts remain unchanged.

Do not repair every inherited defect as a prerequisite for a bounded policy result. Instead, any later completion/output claim must either establish that the exercised path is unaffected, account for the limitation explicitly in the claim, or obtain separately authorized repairs. Hiding failures, treating an error as successful completion, or pretending a timeout demonstrates starvation is not acceptable.

### P08 recommendation and Q11 dependencies

Retain the approved legal-progress guarantee and policy flexibility. Do not add universal equal-share constraints, hard per-request service commitments, aging, or client counters now. Define and report behavior per intended policy. For later evaluation, record first service and first output separately; longest observed gap without execution; completion/unfinished/error counts; completion-time distributions including explicit unfinished counts; evictions and repeated computation; and breakdowns by request size/arrival cohort (and priority class only if used). Repeated computation is not new output service. Wall-clock latency and scheduling-step gaps answer different questions and should not be conflated.

For the vLLM-oriented profile, compare service/preemption traces and persistent delays against the agreed adaptation inside the same framework. Document major deviations and investigate LP-specific noncompletion or omission in bounded cases; numerical acceptance thresholds remain Q11 work. Equal service between clients is outside the current request-order target unless explicitly selected later. This does not preclude a future VTC-like policy, but its accounting/proof would be a distinct research task.

P08 is a proposal about scope and evidence, not permission to implement/run a benchmark and not a waiver of the accepted F02 issue tracking. F02 remains open for observed omissions, recomputation, and claim limits; the rejected hard-prefill milestone is no longer a blocker. Final coefficient settings remain open, Q10 will consolidate necessary changes, Q11 will select metrics/workloads/thresholds, and Q12 will authorize any eventual implementation milestone.

## Q10 consolidation — Smallest coherent change package

Status: P09 consolidates approved R01–R19 and was subsequently approved by the user (R20) on 2026-10-12. No additional scheduling mechanism or implementation authorization is introduced. The only current edit is this discussion record. Local branch main remains at 7f1457ef7bc63677e724ddba797ecd0a47fc6b41 with the existing math-editing-instruction changes preserved. Q11 is explicitly deferred and OPEN; Q12/P10 supplies the chronological plan and narrows the forthcoming TeX editing scope. Final utility settings remain pending.

### 1. Amend the existing ILP and its relaxation together

Keep the existing objective sum(alpha_i*y_i+beta_i*x_i-gamma_i*z_i), variables x/y/I/z, state eligibility, token/action upper bounds, chunk bounds, and action mutual exclusion. Define R as current residents, Q as unallocated waiting candidates, N as resident capacity, F as initially free blocks, h as the shared admission watermark, and d_i in {0,1} as actual decode allocation demand. H below is a linear expression, not a new variable:

$$
H=F+\sum_{i\in R}c_i^Zz_i-\sum_i a_i^P I_i^P.
$$

The consolidated amended resource/progress rows are:

$$
\begin{aligned}
\sum_i(I_i^P+y_i)&\ge1,\\
|R|-\sum_{i\in R}z_i+\sum_{i\in Q}I_i^P&\le N,\\
H-\sum_i d_i y_i&\ge0,\\
\sum_i y_i&\le S_{\max}H,\\
H&\ge h I_i^P\quad(i\in Q).
\end{aligned}
$$

Apply useful execution after input/admission validation when arrived unfinished work exists; a state with no legal useful integer plan reports failure rather than an invented action. Ordinary idle is only the no-arrived-unfinished-work case. Remove the independent planning reserve W, rather than keeping it in addition to the admission watermark. Omit redundant watermark rows when h=0. Preserve full-context admission and zero additional resident-prefill charge. S_max limits selected execution actions; N limits retained residents including omitted ones.

No new integer or continuous decision variables are necessary. Add three shared rows (progress, resident capacity, decode prerequisite) and up to |Q| watermark rows; amend the existing memory row. An explicit matrix representation may repeat H's coefficients across watermark rows, giving quadratic nonzero storage in the worst case even though the number of added rows is linear. No additional solve or assumed solver-runtime bound is introduced. Do not silently add an auxiliary variable or switch to multiple admission-mode solves to optimize this structure.

### 2. Replace extraction with the approved P04 constructor

Use one LP solution to rank whole execution candidates and legal victims with R16 tie-breaks. Keep zero-LP-support candidates and eligible integral-idle victims available. Build actions together with required victims using exact integer resource totals; discard unsuccessful trial additions locally. Accept the first legal useful action even at negative utility; after that, require strict improvement for additions with new victims and nonnegative improvement without them. Never remove accepted execution. Preserve P04's chunk rule and optional extension of already selected prefills within remaining tokens. Validate the final whole plan.

Replace the old integral-idle locking, dominant-fractional-preemption selection, and scalar-deficit-only repair. Do not add a reference-policy backup, exact ILP fallback, or subset search. The analytical useful-plan guarantee depends on the approved additive, fixed-context, independent-victim model and exhaustive first-action candidate scan. The original constant small fractional-set proof is no longer the basis of correctness. Worst-case straightforward extraction work is O(n*r+n log n), for n execution candidates and r resident victims. Utility optimality, minimal victims, mass-preemption avoidance, restart freedom, and starvation freedom are not supplied; accepted counterexamples remain in the record.

### 3. Supply correct immutable resource and policy inputs

The mapper/policy-input path must expose current resident/waiting membership, resident count/limit, actual block-rounded full-context admission and recovery, actual zero-or-one decode demand, admission watermark, and per-step service/victim ranks distinct from identity. Utilities are finite state-dependent alpha/beta/gamma on the common scale and are fixed for the decision. Reuse current sequence and block-manager facts; do not infer lost work from allocated capacity.

State-dependent utility construction is an implementation gap in existing supported mathematics, not a new mathematical capability. Provide the data path for the R17 candidate family and R18 optional discarded-work penalty without making either a universal hardcoded reference scheduler. Only if that optional policy is instantiated, expose the needed primitive computed-token estimate/output-progress facts from the same synchronous state; no persistent fairness/restart ledger is required. Final coefficient formulas and scales remain explicit policy inputs to select later. Preserve one mapper build and read-only construction; do not introduce a second mapping pass or mutable-framework objects into the pure mathematical layer.

### 4. Align selected-plan execution and queue evolution

Validate and emit selected victims in victim-rank/ID order; use existing front insertion, with its explicitly defined reversed waiting prefix. Execute selected prefills first, then zero-demand decodes, then one-block-demand decodes; service rank/ID orders each group. Central operations and emitted metadata must agree. Preserve retained residents' relative running positions, including omitted requests and phase transitions; append selected admissions in their declared execution order. Keep every arrived candidate visible rather than relying on a future waiting head to decide idle.

These changes affect mapper inputs, mathematical extraction ordering, executor validation/mutation ordering, and live scheduler entry logic. They preserve final selected actions, chunks, and victims. The shared full-context allocation implementation and its free-block-before-decode check do not change. The scheduler handles resource totals and uses shared allocation operations, not a second block allocator.

### 5. Complete admission checks and failure distinctions

Implement R13 validation for context length and full admission demand plus watermark exceeding total pool. New invalid submissions receive explicit errors before engine/worker/scheduler registration; current contention is not permanent impossibility. Already registered/recomputation-invalid states retain the approved pre-mutation fail-stop behavior, with no per-request cancellation subsystem added.

Distinguish ordinary idle, proven admission impossibility, validated LP infeasibility, proven absence of useful integer work under the constructor's exhaustive argument, solver/numerical failure, and implementation/validation failure. A fractional feasible LP alone does not prove integer usefulness. A failed arbitrary extraction/validation check does not prove impossibility. Precise result-category names are implementation details, but they must preserve these meanings and the zero-mutation failure boundary. All-zero/preempt-only execution is still forbidden. Any post-mutation exception terminates the run and requires discarding the affected engine.

### Preserved architecture and scope

Keep the four existing responsibilities: pure problem/solve/extraction/validation; read-only state mapping; full pre-mutation validation then execution; live coordination. Keep waiting/running ownership, omitted residents, synchronous one-stage/no-in-flight operation, one mapping, one LP solve, complete validation, and shared allocation/recomputation operations. No locks, reservations, snapshot-version system, new ownership queues, fallback/retry/rollback, incremental allocation, cache offloading, new per-request service reward term, lexicographic multi-solve objective, mandatory retention/aging, or client-history counters are included.

The mathematical constraints describe common execution legality. Coefficient functions and policy ranks describe policy preferences. Extraction is shared across policies and does not precompute an adapted vLLM schedule. These are reusable components, not proof of universal scheduler recovery. The approved initial comparison target remains adapted vLLM v0.31.0 within LPServe; no upstream allocation or whole-engine equivalence is implied.

### Authoritative updates required in a later authorized task

- `docs/math/main-llm-serving.tex`: amend the myopic ILP, relaxation, and extraction together; replace the obsolete unconditional fractional-count/rounding argument with scoped arguments; qualify unsupported native-vLLM recovery assertions and distinguish the separate Lagrangian method. Preserve unrelated concurrent mathematical work and do not rewrite historical evidence.
- `docs/lp_scheduler_design.md`: revise resource/input schemas and affected D-09/D-10/D-16/D-18/D-21 contracts with R01–R19 traceability; keep unresolved utility decisions D-01–D-03 explicit until selected. Admission/error integration and actual-demand configuration must match. Existing execution-safety/failure contracts remain.
- `docs/LP Scheduler Research Context.md`: update affected research assumptions/algorithm description without implying new empirical evidence. The historical architecture audit remains pinned. Chronology/handoffs record subsequent actual implementation and validation at their tested revisions, not anticipated success.
- Source surfaces include the four LP responsibility files plus the minimal input/configuration and submission-validation integration required by R13. Preserve EngineArgs/direct configuration integration. Removing W and replacing the conservative decode policy must have explicit configuration migration/error behavior; old values must not silently change meaning.

### Remaining choices and verification boundary

The required mechanism set can be settled now. Remaining choices do not justify adding another scheduling mechanism: final utility formulas/scales (including whether kappa is used), documented finite-precision ordering/strict-improvement comparisons, and the exact configuration-interface migration need to be fixed before relevant implementation/evaluation. Q11 must choose the common-framework reference adaptation details, workloads, capacities, acceptable service/preemption deviations, and separate decision-time versus model-time measurements; this consolidation approves none of those experiments. Q12 must select an implementation milestone and checks, not assume Unity changes are visible on the Mac.

Required later verification must cover each amended row and its interaction with extraction/physical ordering, optional strict improvement and the known positive-coefficient limitation, queue reinsertion/admission/omission/arrival visibility, admission errors before registration, no-legal-work classification, pre-mutation zero-change behavior, and existing supported admission/partial-prefill/decode/mixed/recomputation/EngineArgs paths. No tests have been run here. F01 remains implementation/validation work, F02 remains policy behavior/evidence work; the excluded hard-prefill commitment is not a blocker. Final scope does not mean implemented, proven policy-equivalent, or performance-validated.

## Q12 plan — Correct the framework before selecting recovery experiments

Status: P10, proposed chronological work plan recorded on 2026-10-12. The user approved P09 and explicitly deferred Q11. This entry records the plan only; it does not authorize or report edits to mathematical/normative sources, implementation, tests, models, or a Claude implementation prompt. R01–R20 remain the accepted architectural decisions. The implementation details selected below are explicit recommendations for the follow-on task, not claims about current code.

### Milestone and completion boundary

Recommended next milestone: amend the two core mathematical subsections and their normative implementation contracts, then implement and CPU-validate the corrected LP planning, mapping, execution, and request-admission path as one coherent change. Include memory contention, mixed zero/one-block decode demand, whole-request preemption, and queue evolution. This is not limited to memory-abundant cases. It does not establish recovery of the adapted vLLM policy, fairness, starvation freedom, corrected inherited output semantics, scheduler overhead, or model performance.

Keep the four existing responsibilities and one synchronous snapshot/LP solve. Keep full-context allocation, selected-plan preservation, complete validation before mutation, fail-stop errors, and the existing shared allocation functions. No reference-policy backup, exact-ILP runtime fallback, retry, rollback, second mapper build, locks, state versions, new ownership queues, hard service commitments, or incremental allocator is added.

A later small GPU regression is a separate authorization and evidence boundary after CPU integration passes. Run any authorized Unity work directly in the already allocated node using the existing repository Python environment; no srun, new allocation, job submission, SSH, dependency installation, or environment rebuild. Record and stop on an environment problem. Historical CPU/GPU results remain evidence only for their original code and workloads.

### 0. Establish the editing and implementation baseline

Before each authorized batch, record the actual branch, full revision, working-tree changes, and affected file contents on that machine. Do not assume the Unity checkout matches the Mac. The Mac inspection for this plan still finds main at 7f1457ef7bc63677e724ddba797ecd0a47fc6b41, existing changes to docs/math/AGENTS.md and docs/math/EDITING.md, and this untracked discussion file.

Reread the saved target passages before editing. Preserve a pre-edit manuscript snapshot outside the tracked manuscript, following the math editing rules. The present math editing records concern another, PH2-focused assignment; append or reconcile this separately authorized assignment in EDITING.md without overwriting that work or rewriting AGENTS.md. One writer per file. Keep bibliography content unchanged and check its checksum before/after TeX batches. No staging, commits, or pushes are part of this plan without separate authorization.

Gate: the saved passages and the actual implementation baseline are identified; concurrent work and historical evidence remain intact.

### 1. Amend the core ILP in subsec:myopic_ilp

Keep the existing objective sum(alpha_i y_i + beta_i x_i - gamma_i z_i), existing x/y/I/z variables, finite nonnegative coefficients fixed per decision, action eligibility, chunk linkage/bounds, token/action budgets, and execution/preemption mutual exclusion. Keep existing notation where possible: c_i^D is the actual decode demand, not an additional decision variable. In the supported LPServe path it is zero or one.

Define current resident set R, unallocated arrived waiting set Q, resident limit N, free blocks F, and the shared admission watermark h, all from the same state. Let H = F + sum_R c_i^Z z_i - sum_i a_i^P I_i^P denote free memory after selected preemptions and all selected prefills, before decodes. H is only a linear expression. Keep block rounding in the immutable input calculation: full-context admission costs a_i^P, zero additional resident-prefill cost, and whole physical-cache recovery c_i^Z. Do not replace these with token-proportional approximations.

Replace the reserve-adjusted memory rule and add the approved rows:

- useful execution: sum_i(I_i^P + y_i) >= 1;
- resident capacity: |R| - sum_R z_i + sum_Q I_i^P <= N;
- actual memory consumption: H - sum_i c_i^D y_i >= 0;
- decode prerequisite: sum_i y_i <= S_max H;
- admission watermark: H >= h I_i^P for each i in Q; omit redundant rows if h=0.

Remove W as an independent planning parameter, including its definitions and explanations in these two subsections. Do not replace it with a final-state watermark. Retain existing equation labels where their roles still match; give new rows their own labels. Do not globally rename symbols or change downstream passages. Retain M_avail as a local notation alias for F, explicitly without a reserve subtraction; this avoids an unnecessary symbol rename while removing the independent reserve.

State when progress applies: valid arrived unfinished work enters the useful-plan optimization; no arrived unfinished work is normal idle outside the solve. If no useful integer plan exists, the requirement is infeasible and must be reported, not satisfied by preemption alone or an invented action. Passing initial admission checks is not a promise of eventual completion.

Explain why these resource rows match the supported order: victims first, prefills second, zero-demand decodes third, one-block-demand decodes last. After the last admission, H is the least free memory within the prefill group, so one watermark margin suffices. H>=1 when any decode is selected permits all zero-demand decodes; H>=sum c_i^D y_i then permits the one-block decodes. Omitted residents continue to count against N. S_max counts execution actions, not residents.

The incremental-prefill paragraph lies inside this subsection. Keep it explicitly separate as an unsupported alternative needing its own allocation/prerequisite derivation; remove any assertion that the new checks or extraction automatically transfer to it. Do not implement or claim exact incremental allocation.

Gate: every new row has a defined input and a simple execution meaning; no new decision variables or priority solves are required. Mathematical legality and supported execution legality agree under the stated independent-cache, synchronous assumptions.

### 2. Rewrite the relaxation and extraction in subsec:lp_relaxation

Relax exactly the amended ILP, retaining all new rows and the objective. Continue requiring successful solver status, finite output, and independent relaxed-solution validation. Preserve numerical/solver failures as failures; no progress guarantee overrides a failed solve.

Replace the old integral-action locking, dominant fractional preemption, scalar memory repair, and positive-support-only packing with approved P04/R15 and R16:

1. Start one empty local plan. Sort all eligible execution candidates by descending LP execution value, then service rank, then ID. Include zero LP support.
2. Propose a whole decode or the approved positive prefill chunk min(q_i, remaining tokens, max(1, floor(LP x_i))). Check all integer resource totals.
3. If necessary, tentatively add independent legal victims in descending LP preemption value, then victim rank, then ID. Exclude the candidate and already selected executions. Allow integral-idle and zero-support victims, including zero-block victims that release a needed resident place.
4. Discard an unsuccessful trial completely and continue. Accept the first feasible execution even if its utility is negative. For later additions requiring new victims, require strictly positive incremental reward minus new penalties; without new victims, nonnegative improvement suffices. Do not search other victim subsets.
5. Never remove accepted execution. Extend selected prefill chunks within remaining token capacity in service-rank order, without new requests or victims. Validate the whole final plan.

Use one general construction path initially; omit the optional integral-LP fast path. This reduces branching and avoids a second set of extraction behavior to explain and validate. It does not change the accepted P04 guarantee or introduce a backup policy.

Replace the obsolete at-most-three-fractional-requests argument as an argument for the amended model. The per-waiting-request watermark rows admit arbitrarily many fractional requests even at an optimal vertex; retain the recorded analytical counterexample. A special h=0 bound may be mentioned only with its assumptions, not used as an algorithm requirement. Replace the old feasibility proof with two short arguments: each accepted whole addition remains legal; any legal useful plan contains a legal single-execution subplan, whose required victims will be reached by the exhaustive first-action scan. State all monotonicity assumptions and distinguish this analytical argument from executed validation.

State the limits: no utility approximation guarantee, minimal-victim guarantee, mass-preemption bound, restart freedom, starvation freedom, universal scheduler recovery, or solver-independent tie outcome. Preserve the strict-improvement counterexample and the positive-coefficient limitation in the discussion/design, with a concise mathematical qualification rather than copying the full discussion into the paper. Straightforward construction costs O(n*r+n log n); no small fractional-set or measured runtime claim is needed.

Gate: pseudocode, proof, integer resource checks, and failure outcomes describe the same algorithm. F01 is not marked implemented or validated merely because the proof and pseudocode are written.

### 3. Add the required end-of-subsection warning; freeze downstream TeX

Place one detailed, visibly colored editorial paragraph at the very end of subsec:lp_relaxation, immediately before subsec:primal_greedy. The current document already loads xcolor; use its existing infrastructure, for example textcolor{red}, without changing the preamble. The paragraph should read substantially as follows, adjusted only to the final notation and actual changes made:

> Revision boundary. The preceding two subsections now require useful execution in every valid arrived-work state that admits a legal useful plan; count all retained residents, including omitted residents; use actual zero-or-one-block decode consumption; remove the independent planning reserve; and separately enforce the admission watermark and the free-block prerequisite for decoding. Full-context prefill allocation, state-dependent finite coefficients, and the existing action variables are retained. The LP relaxation now includes these constraints, and integer extraction constructs whole execution actions together with any required legal preemptions, considers zero-support candidates, preserves accepted execution, and requires strict positive utility improvement for optional additions that introduce new victims. Separate service and victim ranks guide the declared comparisons, and execution uses prefill, zero-demand-decode, then one-block-demand-decode order. The former three-request fractionality argument and locking/repair proof no longer justify this formulation; they have been replaced by the stated scoped legality and useful-progress arguments. All subsequent sections remain unrevised and are outdated wherever they depend on the old feasible set, reserve, extraction, or three-resource decomposition. In particular, the second primal heuristic, Lagrangian dual and its slow/fast paths, native-vLLM reduction, MPC/hierarchical formulations, and later recovery/extension claims require consistency review and revision. They must not be read as established algorithms, proofs, or equivalence results for the amended formulation. This revision establishes neither general vLLM recovery nor fairness, restart freedom, completion, or performance guarantees.

The paragraph records mathematical changes, not a claim that code has been implemented. Check the saved diff to ensure all TeX after the next subsection boundary is unchanged. Do not fix downstream equations, claims, or labels in this milestone. Track their required follow-ups in EDITING.md and the research context. In particular, do not silently carry over the downstream infinity-coefficient vLLM claim.

After the two-subsection batch, check labels/references and compile using the existing permitted compiler. Record actual results in EDITING.md. Compilation is a syntax/build check, not proof of correctness; no PDF rendering or inspection is included.

### 4. Bring the main design and research documents into agreement

Before coding against the new requirements, amend docs/lp_scheduler_design.md with the mathematical rows, immutable fields, P04 construction, P05 ranks/queue rules, exact execution groups, initial-admission location, and error meanings. Trace changed contracts explicitly: D-09 actual demand, D-10 removal of W, D-16 ordering, D-18 progress/idle, D-21 execution/victim order, and the extraction/validation sections. Update D-14's former locking-related role without silently retuning numerical tolerances; preserve applicable D-15 relaxed-validation rules. Reconcile D-01–D-04 resource/policy input descriptions, but do not mark unselected utility functions or capacity values resolved. Preserve D-17/D-19/D-20 safety contracts and D-25 inherited-defect scope. Record approved amendments as requirements awaiting implementation, not as current source behavior.

Update the algorithm/resource descriptions in docs/LP Scheduler Research Context.md and retire contradictory descriptions of locking, reserve, decode overcharge, and the general three-fractional-request expectation. Make section 21 the primary durable home for Q11: OPEN — adapted-policy recovery and overhead evaluation. Record pending corresponding-state/trace definitions, remaining adaptation choices and settings, pinned secondary-policy versions, workloads/capacities, major-deviation thresholds, fairness/restart observations, and separation of scheduler computation from model execution and batch-composition effects. Cross-link this item from the normative design and this discussion. Deferral is not rejection and does not establish policy fidelity.

Do not repin or rewrite the historical architecture audit or its descriptive evidence. Update PROJECT_GUIDE only as needed to point to the current task and preserve historical milestone boundaries. Update project_status and an existing suitable handoff only after actual implementation/validation and acceptance evidence exists; do not write anticipated passes or prematurely declare acceptance.

Gate: mathematical requirements, implementation contracts, and research claim boundaries no longer contradict each other for this milestone. Q11 remains visibly OPEN in a main document.

### 5. Implement the immutable mathematical inputs and pure planning layer

In lp_relaxation_scheduler.py, extend existing immutable inputs rather than accepting live Sequence/block-manager/queue objects. Carry resident/waiting membership, resident limit/count, h, exact charges/recovery, and separate ranks, with explicit validation. Keep eligibility distinct from instantaneous feasibility so eligible work is not hidden to make validation pass.

Update LP matrix construction, labels/residual diagnostics, relaxed validation, and exact integer validation for every amended row. Remove W and the old charge-policy assumption throughout this path. Implement the one P04 constructor, with temporary local trial totals and no live mutation. Retain one structured decision per input request and preserve the finally accepted actions/chunks/victims through all later layers.

Numerical choice for this plan: resource counts and integer feasibility use exact integer arithmetic. Sort validated finite LP values numerically, followed by the explicit ranks and ID; do not use a tolerance-based comparator that can contradict itself. Use floor on the validated LP token value for the approved chunk rule, with its minimum-one safeguard. Do not restore indicator locking through approximate-integrality classification.

For an optional addition with new victims, compute new execution reward r and new penalties p with stable summation. Reuse the existing objective absolute/relative tolerances (currently 1e-9 each) to define tau = abs_tol + rel_tol * max(|r|, |p|); accept only if r-p > tau. This is a conservative numerical implementation of strict improvement: tiny positive gains may be declined, but the first useful action is not subject to this filter. No-victim additions remain allowed at zero reward. Nonfinite aggregate arithmetic is a numerical failure, not a profitable action. Document this deliberate difference from exact-arithmetic Delta>0 and test values on both sides of the threshold. No new policy coefficient or numerical knob is introduced.

Use distinct structured outcomes for established LP infeasibility, exhaustive absence of any legal useful integer plan under P04's assumptions, and solver/numerical/implementation failures. An arbitrary validation failure is never called a proof of impossibility. Keep no_progress as a defensive rejection of an erroneously empty/preempt-only output, not as the mechanism providing progress.

Gate: focused pure CPU checks cover all rows, complete candidate scanning, strict optional improvement, and error distinctions before integrating mutable framework state.

### 6. Map resources and construct policy inputs from one read-only state

In lpserve_state_mapping.py and the live policy-input path, read shared block-manager/sequence facts once at the synchronous boundary: current ownership, actual full-context block demand, physical recovery, zero-or-one decode gap, total/free blocks, watermark, and resident capacity. Reuse the shared allocator's allocation-size computation where available. A gap outside the supported range is an explicit invalid/unsupported-state error, not a silently clamped demand.

Build immutable service/victim ranks from current queues for the approved vLLM-oriented profile: running order followed by arrived waiting order for service, reverse running order for victims. ID only breaks remaining ties. Ranks describe preferences, not a separately computed reference schedule. Preserve all arrived eligible candidates, including those behind an unfit or future queue head.

Provide a small pure coefficient-construction entry point using this same immutable state. Produce finite alpha/beta/gamma before problem construction, fixed for the step. Preserve existing explicit uniform utilities as a usable policy; exercise state-dependent policies using explicit CPU fixtures. Final production/recovery formulas and D/P/G/kappa settings remain later choices. If a discarded-work estimate is supplied, derive it from the reviewed sequence-progress primitives and verify final-prefill/first-decode/recomputation boundaries; do not equate allocated blocks with completed work or add a persistent history ledger.

The coordinator may arrange capture, coefficient calculation, and mathematical input creation within the existing mapping responsibility; it must not collect a second live snapshot. No mutable callback or framework object enters the pure mathematical layer.

Gate: mapped costs match shared allocation behavior on focused states; state-dependent coefficients/ranks can change between decisions without changing ownership or mutating the snapshot.

### 7. Align executor checks, action order, and queue transitions

In lpserve_plan_execution.py, construct the complete declared action order before any mutation. Validate all selected victims/actions, exact physical block needs, context limits, resident capacity, watermark, and the free-before-decode check in that order. Do not rely only on the mathematical validator or drop the executor's independent checks.

Execute and emit selected victims by victim rank/ID, then prefills, zero-demand decodes, and one-block-demand decodes, using service rank/ID within each execution group. Keep central calls and emitted metadata/control order identical. Reuse existing allocation, append, and preemption operations; do not change the shared allocator or its can_append_slot check.

Keep retained residents in their original relative running order whether served, omitted, or changing phase. Append new/re-admitted residents in selected admission order. Keep unselected waiting entries in order. Existing front insertion reverses the emitted victim list into the waiting prefix; document and check that outcome. Physical decode grouping must not reorder resident ownership.

Gate: a mathematically accepted plan executes through the existing allocation checks for supported coherent states, including the one-free-block mixed-demand example. Unexpected rejection still stops before mutation; it is investigated as a discrepancy rather than repaired by a backup. Exceptions after mutation stop the run and require discarding the engine.

### 8. Finish request-admission, configuration, and live failure integration

Before a new LP request is registered with the engine sequence manager, workers, scheduler, or arrival metrics, check its known context length and full initial allocation demand against max_model_len and total pool minus the admission watermark. Place this after tokenization/Sequence construction as needed to reuse the allocator's size calculation. Do not compare against currently free memory for permanent rejection, and do not reserve all future output. Reject with request identity when available, reason, required length/blocks, and configured limit. Scope the new submission behavior to LP scheduling; other policies' lifecycle semantics are not changed as collateral work.

An already registered or recomputation-expanded invalid context uses the existing structured pre-mutation fail-stop path. Do not add request cancellation, worker cleanup protocols, ignore-only output, or a recovery subsystem.

Configuration choice for this plan: exact decode demand is mandatory for the corrected path, not a selectable conservative alternative. Remove active memory_reserve/decode_memory_policy_id configuration fields and their lp_* EngineArgs counterparts; update active constructors/examples/tests together. Explicitly reject obsolete supplied arguments, including a zero old reserve, with a migration explanation; never silently reinterpret them. No compatibility mode or translation of an old reserve into h is added. Read h from the shared block manager. Preserve unrelated EngineArgs/direct-construction behavior and retain explicit token/chunk/action capacities and utilities. Historical scripts/records remain version-specific evidence and are not rewritten to pretend to use the new interface.

In sarathi/core/scheduler/lp_scheduler.py, keep one time read, one mapping, one solve/extraction, and one validation/execution path. Classify ordinary idle using absence of any arrived unfinished work, not just waiting[0]. Enforce these distinctions: invalid initial request (no registration); no arrived work (normal idle); established absence of legal useful work (cannot-progress failure); solver/numerical failure; malformed or invalid plan (implementation/validation failure); post-mutation exception (terminate/discard engine). A rejection never triggers fallback, partial execution, retry, or silent empty output.

Gate: EngineArgs/direct configuration, new submission checks, normal idle, required progress, and all failure boundaries agree with the revised design. Failed initial admission leaves no engine/worker/scheduler registration; an unused sequence-ID counter value is not partial registration.

### 9. Validate the complete correction before claiming the milestone

Add and run focused checks in dependency order, in the authorized implementation environment. These are future checks, not evidence from this discussion:

- Pure mathematical checks: each new row; h=0 and h>0; multiple admissions sharing one watermark; omitted residents; zero free blocks with zero-demand decode; mixed decode demand; necessary negative-utility first work; zero-LP-support execution and integral-idle victims; no legal useful plan; zero- and positive-coefficient mass-preemption examples; strict numerical comparison boundaries. Do not require one particular optimizer tie solution unless the fixture explicitly supplies it.
- A small independent enumeration check on bounded synthetic states: enumerate whole actions/chunks/victims only in test code to determine whether any legal useful plan exists. Compare that answer with the constructor and independently validate every returned plan. Include LP-feasible/integer-infeasible cases and the many-fractional-watermark construction. This is test evidence alongside the proof, not an exact-ILP runtime fallback or a general correctness proof by testing.
- Mapper/allocation checks: full-context rounding at block boundaries, resident prefill zero charge, decode gap zero/one, exact recovery, watermark/resident limits, and malformed allocation state. Snapshot/policy construction must not mutate live state.
- Executor/queue checks: prefills followed by both decode groups; A consumes the last free block while B needs none; same central/metadata ordering; multiple victim reinsertion; retained omitted residents; readmission to the tail; future head followed by arrived work. Use supported replay paths, with inherited physical block-ID and output limitations stated separately.
- Admission/failure/integration checks: length/pool equality boundaries; request can fit in the whole pool but not current free memory; no registration on initial rejection; recomputation-expanded invalid context; no-progress defensive guard; induced pre-mutation failures with zero execution mutation; post-mutation stop behavior; existing admission/partial-prefill/decode/mixed/recomputation and EngineArgs paths.

Checks should target the changed guarantees and identified regressions, not every inherited defect. Use existing test helpers and runtime mechanisms; do not create a benchmark suite or load models for CPU evidence. Record tested revision or exact working-tree diff/hashes, environment/interpreter, commands, observed results, and failed/skipped/unexecuted checks. A test that fails due to an inherited issue must be classified and its effect on this milestone explained, not relabeled a pass.

Gate: the updated math/design and implemented four-layer behavior agree; applicable focused CPU checks pass; all remaining limitations are named. F01 can then be closed for its stated supported model and tested implementation. F02 and Q11 stay open for policy behavior and evidence; full-engine, output, fairness, and performance claims remain excluded.

### 10. Close documentation and define the next handoff boundary

Review the full saved diff against this scope. Confirm that subsequent TeX sections are unchanged except that the new warning precedes them, historical artifacts are preserved, and no old reserve/charge/locking claim remains normative for the new path. Keep future-policy and inherited-defect qualifications explicit. Record actual results in the appropriate existing handoff and update project_status only with observed evidence and acceptance state. Keep this discussion as decision history rather than copying it wholesale into each main document.

Report mathematical amendments, implemented behavior, checks, limitations, and unexecuted GPU/policy work separately. The milestone may be complete as CPU-validated framework work while Q11 remains OPEN. GPU correctness regression must precede later performance claims but is not silently authorized or counted as complete here. No Claude implementation prompt is produced by this planning task.

### Explicit additional choices and remaining decisions

The following small choices are newly made explicit in P10 for the follow-on plan: (1) initial CPU-validated correction milestone, with GPU regression separate; (2) one P04 path without the optional integral fast path; (3) retain existing mathematical symbols and one LP solve; (4) reuse current objective tolerances as a conservative strict-improvement guard and use ordinary deterministic numeric/rank/ID ordering; (5) remove obsolete reserve/conservative-charge interfaces with explicit rejection and no compatibility mode; (6) scope new submission rejection behavior to LP mode; (7) make research-context section 21 the primary home of deferred Q11. These are visible implementation/planning recommendations; none is hidden as an existing approved numerical default or empirical result.

The following decisions can wait until the framework correction is complete:

- Final state-dependent utility formulas and D/P/G values, optional kappa and its tradeoff between keeping computed work and matching youngest-victim preference. No hard commitment, universal aging, or fairness ledger is pending as a requirement.
- Q11: exact adapted-reference procedure/configuration details, corresponding-state and service/preemption-trace definitions, acceptable major deviations, workloads and numeric capacities, bounded fairness/restart observations, and overhead measurement boundaries. Keep upstream vLLM pinned to v0.31.0; choose Sarathi/SLAI implementations and profiles before claims about their recovery.
- Whether the corrected path's observed solver/extraction cost justifies later optimizations; do not add subset search, alternate formulations, fast paths, or extra solves in anticipation.
- Scope/authorization of the subsequent bounded GPU regression and then recovery/performance experiments. Existing inherited defects require repair only if they block that chosen claim or path.
- A separate mathematical assignment to update PH2, the dual/slow/fast paths, MPC/hierarchical formulations, and recovery claims. Until then the requested end-of-subsection warning remains in force.

Do not postpone correctness details that the implementation needs: exact integer rows, complete first-action search, numerical comparison behavior, config migration, queue order, admission location, and error semantics are specified above. If implementation reveals a new state outside the proof assumptions, report that concrete mismatch and amend the affected contract explicitly; do not silently narrow eligibility or label every rejection an inherited defect.


## Ordered agenda

Q01 has scope decision R04 and clarification R11; its remaining precision and secondary-reference choices stay open. Q02 has the source review and required future issues. Q03 resource decisions and Q04 operating rules are covered by the decisions above; implementation and verification remain pending. Q05 extraction and Q06 ordering designs are approved in R15/R16; their implementation and validation remain pending. Q07 policy discipline is approved in R17; final coefficient choices remain open. Q08 recommendations are approved in R18 with the hard partial-prefill commitment excluded. Q09 scope is approved in R19. Q10/P09 is approved in R20. Q11 is explicitly deferred and OPEN. Q12/P10 records the proposed chronological milestone/documentation plan; implementation authorization and final utility/evaluation choices remain pending.

### Q01 — What exactly are we comparing and guaranteeing?

- Primary reference and common-framework scope: recorded in R04; vLLM remains pinned to v0.31.0.
- Remaining: which exact Sarathi and SLAI revisions/settings are additional references?
- Remaining: which upstream-only features or framework restrictions need explicit entries in the adaptation boundary? Inspect only features relevant to the approved reference.
- Remaining, to resolve after Q02 and under Q09/Q11: quantified service/preemption agreement, the meaning of no additional issues, and universal versus policy-specific guarantees.

### Q02 — What do the references actually provide, and where do our issues originate?

- Under the agreed scope, what do the references do about idle steps, queue ordering, preemption, repeated restarts, fairness, starvation, and completion?
- Which properties are proved, merely observed, or not established? What evidence exists about how often problems occur?
- For each mismatch, is its cause the ILP, relaxation, extraction, policy inputs not yet implemented, or a shared serving-framework limitation?

### Q03 — What is a legal executable plan in our common framework?

Status: resource-policy discussion covered; see Q03 review. Implementation, extraction implications, and frequency evidence remain with their assigned follow-ups.

- How must full-context allocation, block rounding, decode allocation gates, admission watermark, planning reserve, resident limits, and action limits be represented or checked?
- Can an ILP-feasible plan fail native validation, and what is the smallest way to close any relevant gap?
- Do Sarathi and SLAI experience the allocation contention described earlier? How do they handle it, and what frequency evidence exists?

### Q04 — How does the approved useful-execution rule apply?

Status: operating rules approved in R14/P03; extraction design approved in R15/P04. F01 retains implementation and validation follow-up.

- When is the constraint active, and how do we distinguish normal idle, no legal work, and failure to find legal work?
- How do we establish whether useful work is possible, including necessary preemptions?
- What changes does this require in the ILP and its relaxation, while leaving coefficient choice flexible?

### Q05 — How can extraction preserve useful execution without a backup policy?

Status: P04 plus strict optional improvement approved in R15 with recorded limitations. Implementation, authoritative amendments, and validation remain pending.

- Can a simple change to rounding, action selection, or joint selection of work and victims provide the guarantee?
- Must extraction reconsider locked choices, requests with zero fractional support, or preemptions selected without accompanying work?
- If a simple change is insufficient, what are the alternatives and their computation costs? What must be proved for the chosen approach?

### Q06 — What queue, priority, and sorting rules do we need?

Status: P05 approved in R16; authoritative amendments, implementation, and validation remain pending.

- Can current queue positions supply the needed ordering, or is a separate policy rank necessary?
- How should ordering guide fractional extraction, ties, omitted residents, victim selection, and reinsertion after preemption?
- How can these rules support several policies without making FCFS universal or confusing service order with physical batch packing?

### Q07 — What can coefficient updates represent and guarantee?

Status: P06 recommendations approved in R17; final utility formulas/scales and wider recovery claims remain open with Q08/Q09/Q11 dependencies.

- Which functions of request state, queue position, work already completed, and resources are sufficient for the desired preferences?
- How do per-token prefill rewards compare with giving a request a turn, and how should reward scales be chosen?
- Under which assumptions can we prove service-order recovery? Does each claim concern an integer optimum, an LP optimum, or the actual extracted plan?
- Which stronger guarantees need something beyond coefficients, and how should the manuscript's existing recovery claim be assessed?

### Q08 — How do we avoid repeated loss of progress?

Status: P07 approved in R18; hard partial-prefill commitment excluded. Final penalty scales and behavioral validation remain pending.

- How do reference policies and selected literature approaches choose victims, preserve work, or limit restarts?
- Can recomputation-aware gamma updates and queue rules suffice under useful assumptions? When would retention or another explicit requirement be needed?
- How should execution preference and memory recovery interact without hardcoding a reference scheduler?

### Q09 — What fairness, starvation, and completion guarantees do we require?

Status: P08 approved in R19; policy settings, measurement thresholds, and experimental evidence remain pending.

- How do we define each property, and what arrival, workload, resource, and stopping assumptions does it need?
- Can aging or other coefficient updates provide it through the actual extraction path, or are service requirements necessary?
- How do we preserve policy flexibility when different references provide different guarantees?
- Which inherited recomputation or output limitations affect the chosen completion claim, and which can remain outside this milestone?

### Q10 — What is the smallest coherent set of framework changes?

Status: P09 is approved in R20; no new scheduling mechanism or implementation authorization is introduced. Q12/P10 provides the dependency-ordered follow-on plan.

- After Q01–Q09, which changes are actually necessary: ordering inputs, ordered objectives, service or retention requirements, resource constraints, or extraction changes?
- Which existing mechanisms and execution contracts can be preserved, and which need explicit amendment?
- What do the selected changes cost in integer variables, solves, computation, and mathematical arguments?
- Are the changes reusable policy mechanisms, or do they effectively implement a reference procedure?

### Q11 — How will we establish recovery and measure overhead?

Status: explicitly deferred by the user on 2026-10-12; OPEN and important. P10 step 4 will put the durable item in research-context section 21 during the authorized main-document update. No recovery thresholds or experiments are approved.

- How do we compare corresponding logical states and service/preemption traces when policies use different chunks and batches?
- Which analytical guarantees and later bounded checks distinguish unnecessary idle, lost progress, unfairness, and inherited limitations?
- What constitutes sufficient agreement for each reference policy?
- How do we measure scheduling computation within the common runtime separately from execution and changed batch composition?

### Q12 — What is the next authorized milestone and documentation plan?

Status: P10 above records the detailed plan. Only this discussion record is updated now; implementation and main-document edits remain unstarted.

- Which decisions are complete, and which still need resolution before implementation?
- Which approved changes belong in the mathematical source and normative design? Which evidence belongs in later validation records?
- What exact implementation scope, checks, acceptance criteria, and completion boundary should the later task have?

## Annotation coverage

| User annotation | Agenda coverage |
|---|---|
| 1: add the hard ILP rule | R01; Q03–Q05 |
| 2: how other schedulers avoid repeated restarts | Q02; Q08 |
| 3: compare the vLLM heuristic within the common framework; list exclusions | R02; Q01; Q03 |
| 4: separate unimplemented coefficient construction from missing framework capabilities | R03; Q02; Q07; Q10 |
| 5: simple extraction fixes and alternatives | Q05 |
| 6: resolve issues using native queue/priority/sorting behavior, without explicit backup | R03; Q05–Q08; Q10 |
| 7: reference/literature restart and fairness guarantees; coefficients versus more machinery | Q02; Q07–Q09 |
| 8: shared Sarathi/SLAI allocation issue, handling, and frequency | Q02; Q03 |

## Recording future decisions

For each agenda item, record the answer, assumptions, source evidence or analytical argument, selected change and rationale, affected authoritative documents/contracts, and remaining questions. Clearly label rejected options and unexecuted validation. Do not treat an unanswered agenda item as resolved by another decision.

Current item: Q12. R20 records approval of P09, explicit Q11 deferral, and the two-subsection TeX boundary. P10 records the chronological plan, proposed CPU-validated correction milestone, required downstream-warning paragraph, explicit implementation choices, future checks, and remaining decisions. Q11 and F02 remain OPEN; F01 still requires implementation/validation. No mathematical source, normative document, implementation, or historical evidence was changed, and no tests/models were run. Only this discussion record was updated.
