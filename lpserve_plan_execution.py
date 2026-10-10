"""Synchronous native execution of a validated LP-relaxation integer plan.

``execute_plan(scheduler, snapshot, result)`` receives the live scheduler, the
immutable ``StateSnapshot`` the plan was computed from, and the successful
``SchedulingSuccess`` returned by the mathematical layer. It validates the
complete chosen plan against current scheduler state before any mutation, then
executes the supported actions through inherited LPServe operations and
returns matching native ``SchedulerOutputs`` (docs/lp_scheduler_design.md
sections 12.6, 13, and 14).

Supported actions (the executor subset recorded in the design):

- admission of an unallocated waiting request with a positive prompt chunk;
- continuation of a resident partial prefill without new allocation;
- one decode for a prompt-complete resident;
- native recomputation preemption of every selected legal resident victim,
  when the plan also selects at least one prefill or decode;
- any combination of these in one output.

Prompt-ignore controls are not executed. A plan with no prefill or decode
action (all-zero or preempt-only) returns category ``no_progress`` before
mutation and leaves the mathematical plan unchanged.

Order: all selected victims are preempted first, ascending by ``order_key``,
and ``preempted_seq_ids`` uses that order. The inherited ``_preempt`` inserts
each victim at the front of ``waiting``, so several victims end there in
reverse call order. Then all selected prefills, then all selected decodes,
each group ascending by ``order_key`` (D-21). This order changes neither
selected actions nor chunk sizes. Memory recovered from victims is credited
before the scheduled actions are checked.

Failure behavior:

- Before mutation, a rejection returns an immutable ``Failure`` record without
  output and without changing queues, blocks, sequence state, or batch state.
  The integer-plan validator's own ``Failure`` is returned unchanged.
- After the first mutation, any exception propagates immediately (D-20). The
  executor does not catch it, continue later actions, return partial output,
  retry, roll back, or recover; the caller must discard the affected state.

The executor neither invokes the mapper or solver nor increments
``_iteration_id`` or ``num_running_batches``; inherited live scheduling owns
that bookkeeping. Sequence status transitions, prompt progress, and the
recomputation reset of a preempted sequence are left to existing engine
replay and step completion.
"""

from __future__ import annotations

import math

from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import SequenceScheduleMetadata
from sarathi.core.datatypes.sequence_status import SequenceStatus

import lp_relaxation_scheduler as lrs
import lpserve_state_mapping as lsm

STAGE_PRECOMMIT_VALIDATION = "precommit_validation"

CATEGORY_UPSTREAM_FAILURE = "upstream_failure"
CATEGORY_MALFORMED_INPUT = "malformed_input"
CATEGORY_SNAPSHOT_MISMATCH = "snapshot_mismatch"
CATEGORY_UNSUPPORTED_STATE = "unsupported_state"
CATEGORY_NO_PROGRESS = "no_progress"
CATEGORY_OWNERSHIP_MISMATCH = "ownership_mismatch"
CATEGORY_INELIGIBLE_ACTION = "ineligible_action"
CATEGORY_CHUNK_BOUND = "chunk_bound"
CATEGORY_OVERLENGTH_ADMISSION = "overlength_admission"
CATEGORY_RESIDENT_CAPACITY = "resident_capacity"
CATEGORY_ADMISSION_GATE = "admission_gate"
CATEGORY_APPEND_GATE = "append_gate"
CATEGORY_PREEMPTION_RECOVERY = "preemption_recovery"

ACTION_ADMISSION = "admission"
ACTION_RESIDENT_PREFILL = "resident_prefill"
ACTION_DECODE = "decode"


class _PrecommitError(Exception):
    def __init__(self, category: str, reason: str) -> None:
        super().__init__(reason)
        self.category = category
        self.reason = reason


def _require(condition: bool, category: str, reason: str) -> None:
    if not condition:
        raise _PrecommitError(category, reason)


def execute_plan(scheduler, snapshot, result):
    """Validate and execute ``result.plan``; return ``SchedulerOutputs`` or ``Failure``.

    ``scheduler`` is read through ``waiting``, ``running``, ``_iteration_id``,
    ``num_running_batches``, ``scheduler_config`` (``num_pipeline_stages``,
    ``max_num_seqs``, ``max_model_len``), and ``block_manager``. Only after
    complete prevalidation does it mutate, through native list operations and
    the inherited ``_preempt``/``_allocate``/``_append_slot`` helpers.
    """
    snapshot_id = getattr(snapshot, "snapshot_id", None)
    if not isinstance(snapshot_id, str):
        snapshot_id = None
    try:
        validated = _validate_before_mutation(scheduler, snapshot, result)
    except _PrecommitError as err:
        return lrs.Failure(
            snapshot_id, STAGE_PRECOMMIT_VALIDATION, err.category, err.reason,
        )
    if isinstance(validated, lrs.Failure):
        return validated
    victims, actions = validated
    # Execution mutation begins here. Exceptions propagate unhandled.
    return _execute(scheduler, snapshot, victims, actions)


# ---------------------------------------------------------------------------
# Precommit validation (no mutation)
# ---------------------------------------------------------------------------


def _validate_before_mutation(scheduler, snapshot, result):
    """Return the ordered victim and action lists, a plan-validation
    ``Failure``, or raise ``_PrecommitError``. Reads state only."""
    _require(
        isinstance(snapshot, lsm.StateSnapshot), CATEGORY_MALFORMED_INPUT,
        f"snapshot must be a StateSnapshot, got {type(snapshot).__name__}",
    )
    if isinstance(result, (lrs.Failure, lsm.MappingFailure)):
        raise _PrecommitError(
            CATEGORY_UPSTREAM_FAILURE,
            f"upstream result is a failure ({result.stage}: "
            f"{result.category}: {result.reason}); execution is not permitted",
        )
    _require(
        isinstance(result, lrs.SchedulingSuccess), CATEGORY_MALFORMED_INPUT,
        f"result must be a SchedulingSuccess, got {type(result).__name__}",
    )

    problem = snapshot.lp_problem
    plan = result.plan
    _require(
        isinstance(problem, lrs.LPProblem), CATEGORY_MALFORMED_INPUT,
        "snapshot.lp_problem must be an LPProblem, got "
        f"{type(problem).__name__}",
    )
    _require(
        isinstance(plan, lrs.IntegerPlan), CATEGORY_MALFORMED_INPUT,
        f"result.plan must be an IntegerPlan, got {type(plan).__name__}",
    )

    # Identity: snapshot, problem, result, plan, and the current scheduler
    # decision. This associates records; it does not prove unchanged state.
    iteration_id = scheduler._iteration_id
    expected_id = str(iteration_id)
    identities = (
        ("snapshot.snapshot_id", snapshot.snapshot_id),
        ("snapshot.lp_problem.problem_id", problem.problem_id),
        ("result.problem_id", result.problem_id),
        ("result.plan.problem_id", plan.problem_id),
    )
    for name, value in identities:
        _require(
            value == expected_id, CATEGORY_SNAPSHOT_MISMATCH,
            f"{name}={value!r} does not match the current scheduler "
            f"iteration {expected_id!r}",
        )
    _require(
        snapshot.scheduler_iteration_id == iteration_id,
        CATEGORY_SNAPSHOT_MISMATCH,
        f"snapshot.scheduler_iteration_id={snapshot.scheduler_iteration_id!r}"
        f" does not match the current scheduler iteration {iteration_id!r}",
    )

    # Supported stage and in-flight state at the decision boundary.
    config = scheduler.scheduler_config
    _require(
        config.num_pipeline_stages == 1, CATEGORY_UNSUPPORTED_STATE,
        f"unsupported pipeline stage count {config.num_pipeline_stages!r}; "
        "only 1 is supported",
    )
    _require(
        scheduler.num_running_batches == 0, CATEGORY_UNSUPPORTED_STATE,
        "unsupported in-flight state: num_running_batches="
        f"{scheduler.num_running_batches!r}; only 0 is supported",
    )
    max_num_seqs = config.max_num_seqs
    _require(
        len(scheduler.running) <= max_num_seqs, CATEGORY_UNSUPPORTED_STATE,
        f"resident count {len(scheduler.running)} already exceeds "
        f"max_num_seqs {max_num_seqs}",
    )

    # Mathematical legitimacy, reusing the accepted validators unchanged.
    canonical = lrs.validate_problem(problem)
    if isinstance(canonical, lrs.Failure):
        return canonical
    _require(
        canonical == problem, CATEGORY_MALFORMED_INPUT,
        "snapshot.lp_problem is not in canonical order_key order",
    )
    validated = lrs.validate_integer_plan(problem, plan)
    if isinstance(validated, lrs.Failure):
        return validated

    # Progress gate: no selected preemption is removed and no action forced.
    preempt_ids = [d.request_id for d in plan.decisions if d.preempt]
    execution_count = sum(
        1 for d in plan.decisions if d.prefill_tokens > 0 or d.decode
    )
    _require(
        execution_count > 0, CATEGORY_NO_PROGRESS,
        "the validated plan selects no prefill or decode action "
        f"(selected preemptions: {preempt_ids})",
    )

    # Snapshot request records must associate one-to-one with the problem.
    snap_by_id = {}
    for req in snapshot.requests:
        _require(
            isinstance(req, lsm.RequestStateSnapshot)
            and req.request_id == str(req.raw_seq_id)
            and req.request_id not in snap_by_id,
            CATEGORY_SNAPSHOT_MISMATCH,
            "snapshot requests must be unique RequestStateSnapshot records "
            "whose request_id is the decimal raw_seq_id",
        )
        snap_by_id[req.request_id] = req
    _require(
        len(snap_by_id) == len(problem.requests)
        and all(
            r.request_id in snap_by_id
            and snap_by_id[r.request_id].order_key == r.order_key
            for r in problem.requests
        ),
        CATEGORY_SNAPSHOT_MISMATCH,
        "snapshot request records do not match the mapped problem requests",
    )

    # Current ownership of the inherited collections, unique by seq_id.
    owners = {}
    for label, collection in (
        (lsm.OWNERSHIP_WAITING, scheduler.waiting),
        (lsm.OWNERSHIP_RUNNING, scheduler.running),
    ):
        for seq in collection:
            _require(
                seq.seq_id not in owners, CATEGORY_OWNERSHIP_MISMATCH,
                f"seq_id {seq.seq_id!r} is owned more than once (in "
                f"{owners.get(seq.seq_id, (None,))[0]!r} and {label!r})",
            )
            owners[seq.seq_id] = (label, seq)

    # Replay-compatible order: prefills, then decodes, each ascending by
    # order_key. New lists; the accepted plan is not reordered in place.
    selected = sorted(
        (d for d in plan.decisions if d.prefill_tokens > 0 or d.decode),
        key=lambda d: (0 if d.prefill_tokens > 0 else 1, d.order_key),
    )

    block_manager = scheduler.block_manager
    free_blocks = block_manager.get_num_free_gpu_blocks()
    watermark_blocks = block_manager.watermark_blocks
    resident_count = len(scheduler.running)

    # Every selected victim, whether extraction marked it dominant or safety,
    # ascending by order_key. Each is credited with its actual physical table,
    # which the native non-sharing manager frees in full, and releases one
    # resident slot before any scheduled action is checked.
    victims = []
    recovered = set()
    for decision in sorted(
        (d for d in plan.decisions if d.preempt), key=lambda d: d.order_key,
    ):
        snap = snap_by_id[decision.request_id]
        rid = snap.raw_seq_id
        _require(
            snap.ownership == lsm.OWNERSHIP_RUNNING
            and snap.preemption_eligible
            and decision.request_id in problem.legal_preemption_ids,
            CATEGORY_INELIGIBLE_ACTION,
            f"seq_id {rid}: preemption selected for a request that is not a "
            "mapped legal resident",
        )
        seq = _check_current_request(
            owners, snap, snapshot.snapshot_time, block_manager,
        )
        _require(
            seq.is_executing(), CATEGORY_INELIGIBLE_ACTION,
            f"seq_id {rid}: preemption requires a native executing status",
        )
        table = block_manager.get_block_table(seq)
        _require(
            len(set(table)) == len(table) and not recovered.intersection(table),
            CATEGORY_PREEMPTION_RECOVERY,
            f"seq_id {rid}: physical block table {table} repeats a block or "
            "shares one with another selected victim",
        )
        recovered.update(table)
        free_blocks += len(table)
        resident_count -= 1
        victims.append(seq)

    actions = []
    for decision in selected:
        snap = snap_by_id[decision.request_id]
        seq = _check_current_request(
            owners, snap, snapshot.snapshot_time, block_manager,
        )
        rid = snap.raw_seq_id
        remainder = seq.get_prompt_len() - seq.get_num_prompt_tokens_processed()
        chunk = decision.prefill_tokens

        if chunk > 0:
            _require(
                remainder > 0 and not seq.prompt_processing_finished,
                CATEGORY_INELIGIBLE_ACTION,
                f"seq_id {rid}: prefill selected but the current prompt "
                f"remainder is {remainder}",
            )
            _require(
                chunk <= min(remainder, problem.c_max), CATEGORY_CHUNK_BOUND,
                f"seq_id {rid}: prefill chunk {chunk} exceeds min(current "
                f"remainder {remainder}, c_max {problem.c_max})",
            )
            if snap.ownership == lsm.OWNERSHIP_WAITING:
                _require(
                    not block_manager.is_allocated(seq),
                    CATEGORY_INELIGIBLE_ACTION,
                    f"seq_id {rid}: waiting admission is already allocated",
                )
                # Checked read-only here; the mutating prompt-rejection
                # helper is never called.
                _require(
                    seq.get_len() <= config.max_model_len,
                    CATEGORY_OVERLENGTH_ADMISSION,
                    f"seq_id {rid}: length {seq.get_len()} exceeds "
                    f"max_model_len {config.max_model_len}",
                )
                resident_count += 1
                _require(
                    resident_count <= max_num_seqs, CATEGORY_RESIDENT_CAPACITY,
                    f"seq_id {rid}: admission would raise the resident count "
                    f"to {resident_count}, above max_num_seqs {max_num_seqs}",
                )
                demand = block_manager.get_num_initial_blocks(seq)
                _require(
                    free_blocks - demand >= watermark_blocks,
                    CATEGORY_ADMISSION_GATE,
                    f"seq_id {rid}: admission needs {demand} blocks with "
                    f"{free_blocks} free in execution order; the native "
                    f"watermark requires {watermark_blocks} to remain",
                )
                free_blocks -= demand
                kind = ACTION_ADMISSION
            else:
                _require(
                    _block_gap(block_manager, seq) == 0,
                    CATEGORY_INELIGIBLE_ACTION,
                    f"seq_id {rid}: resident prefill logical/physical block "
                    f"gap {_block_gap(block_manager, seq)} must be 0",
                )
                kind = ACTION_RESIDENT_PREFILL
        else:
            _require(
                snap.ownership == lsm.OWNERSHIP_RUNNING,
                CATEGORY_INELIGIBLE_ACTION,
                f"seq_id {rid}: decode selected for a non-resident request",
            )
            _require(
                remainder == 0 and seq.prompt_processing_finished,
                CATEGORY_INELIGIBLE_ACTION,
                f"seq_id {rid}: decode selected but the current prompt "
                f"remainder is {remainder}",
            )
            gap = _block_gap(block_manager, seq)
            _require(
                gap in (0, 1), CATEGORY_INELIGIBLE_ACTION,
                f"seq_id {rid}: decode logical/physical block gap {gap} is "
                "outside {0, 1}",
            )
            # The native append gate requires a free block even for gap 0;
            # only the actual gap is consumed.
            _require(
                free_blocks > 0, CATEGORY_APPEND_GATE,
                f"seq_id {rid}: decode append needs a free block, but none "
                "remain in execution order",
            )
            free_blocks -= gap
            kind = ACTION_DECODE
        actions.append((kind, seq, chunk))
    return victims, actions


def _check_current_request(owners, snap, decision_time, block_manager):
    """Return the live sequence for one selected request after common checks."""
    rid = snap.raw_seq_id
    owner = owners.get(rid)
    _require(
        owner is not None and owner[0] == snap.ownership,
        CATEGORY_OWNERSHIP_MISMATCH,
        f"seq_id {rid}: expected current owner {snap.ownership!r}, found "
        f"{owner[0] if owner else None!r}",
    )
    seq = owner[1]
    arrival = seq.arrival_time
    _require(
        isinstance(arrival, (int, float)) and not isinstance(arrival, bool)
        and math.isfinite(arrival) and arrival <= decision_time,
        CATEGORY_INELIGIBLE_ACTION,
        f"seq_id {rid}: arrival_time {arrival!r} is not at or before the "
        f"decision time {decision_time!r}",
    )
    _require(
        not seq.is_finished(), CATEGORY_INELIGIBLE_ACTION,
        f"seq_id {rid}: request is finished",
    )
    expected_status = (
        SequenceStatus.WAITING if snap.ownership == lsm.OWNERSHIP_WAITING
        else SequenceStatus.PAUSED
    )
    _require(
        seq.get_status() == expected_status, CATEGORY_INELIGIBLE_ACTION,
        f"seq_id {rid}: status {seq.get_status()!r}, expected "
        f"{expected_status!r}",
    )
    processed = seq.get_num_prompt_tokens_processed()
    remainder = seq.get_prompt_len() - processed
    _require(
        processed >= 0 and remainder >= 0
        and seq.prompt_processing_finished == (remainder == 0),
        CATEGORY_INELIGIBLE_ACTION,
        f"seq_id {rid}: inconsistent prompt progress (processed={processed}, "
        f"remainder={remainder}, prompt_processing_finished="
        f"{seq.prompt_processing_finished!r})",
    )
    if snap.ownership == lsm.OWNERSHIP_RUNNING:
        _require(
            block_manager.is_allocated(seq), CATEGORY_INELIGIBLE_ACTION,
            f"seq_id {rid}: resident request is not allocated",
        )
    return seq


def _block_gap(block_manager, seq):
    return len(seq.logical_token_blocks) - len(block_manager.get_block_table(seq))


# ---------------------------------------------------------------------------
# Execution (mutation)
# ---------------------------------------------------------------------------


def _execute(scheduler, snapshot, victims, actions):
    waiting = scheduler.waiting
    running = scheduler.running
    # Native preemption frees central blocks and inserts at the waiting
    # front. The sequence keeps its status here; replay of the emitted
    # preempted ID resets it for recomputation and frees worker blocks.
    preempted = []
    for seq in victims:
        running.pop(_index_by_identity(running, seq))
        scheduler._preempt(seq)
        preempted.append(seq.seq_id)
    metadata = []
    for kind, seq, chunk in actions:
        if kind == ACTION_ADMISSION:
            waiting.pop(_index_by_identity(waiting, seq))
            scheduler._allocate(seq)
            running.append(seq)
        elif kind == ACTION_DECODE:
            scheduler._append_slot(seq)
        metadata.append(
            SequenceScheduleMetadata.from_sequence(seq, prompt_chunk_len=chunk)
        )

    outputs = SchedulerOutputs(
        id=scheduler._iteration_id,
        ignored_seq_ids=[],
        preempted_seq_ids=preempted,
        scheduled_seq_metadata_list=metadata,
    )

    # Final runtime checks. A failure here is after mutation and propagates.
    max_num_seqs = scheduler.scheduler_config.max_num_seqs
    if len(running) > max_num_seqs:
        raise RuntimeError(
            f"resident count {len(running)} exceeds max_num_seqs "
            f"{max_num_seqs} after plan execution"
        )
    expected = [(seq.seq_id, chunk) for _, seq, chunk in actions]
    emitted = [
        (m.seq_id, m.prompt_chunk_len)
        for m in outputs.scheduled_seq_metadata_list
    ]
    if emitted != expected:
        raise RuntimeError(
            f"emitted scheduled entries {emitted} differ from the validated "
            f"ordered actions {expected}"
        )
    expected_victims = [seq.seq_id for seq in victims]
    if outputs.ignored_seq_ids or outputs.preempted_seq_ids != expected_victims:
        raise RuntimeError(
            f"emitted controls (ignored {outputs.ignored_seq_ids}, preempted "
            f"{outputs.preempted_seq_ids}) differ from the validated victims "
            f"{expected_victims}"
        )
    problem = snapshot.lp_problem
    if (
        outputs.num_batched_prompt_tokens + outputs.num_batched_output_tokens
        > problem.b_max
        or len(outputs.scheduled_seq_metadata_list) > problem.s_max
    ):
        raise RuntimeError(
            "emitted token or action counts exceed the validated limits"
        )
    return outputs


def _index_by_identity(collection, seq):
    for index, item in enumerate(collection):
        if item is seq:
            return index
    raise RuntimeError(f"seq_id {seq.seq_id} is no longer in the collection")
