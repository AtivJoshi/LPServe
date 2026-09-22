"""Read-only LPServe state mapper for the LP-relaxation scheduling layer.

Observes a scheduler holder's inherited ``waiting``/``running`` collections
and its block manager, and returns a frozen, framework-independent snapshot
suitable for ``lp_relaxation_scheduler.LPProblem`` construction (see
docs/lp_scheduler_design.md, especially sections 4, 9, and 12). It performs
no LPServe mutation: it calls only read-only scheduler, sequence, and block
manager accessors, never allocation, append, free, preemption, status
transition, or queue-mutation methods.

Supported state shape (docs/lp_scheduler_design.md section 12.4 and this
module's handoff record):

- exactly one pipeline stage and zero running batches (a quiescent,
  pre-mutation boundary with no batch in flight);
- ``waiting`` exclusively owns unallocated unfinished requests, and
  ``running`` exclusively owns allocated unfinished (resident) requests;
- every raw ``Sequence.seq_id`` is an exact non-negative, non-boolean
  integer, unique across both collections;
- an included waiting request has status ``WAITING``, is unallocated, and
  has positive, incomplete prompt remainder;
- an included resident request has status ``PAUSED`` and is allocated; a
  resident with a positive prompt remainder must have its full logical
  context already allocated (a zero logical/physical block gap), and a
  resident with zero remainder (decode) may have a logical/physical block
  gap of zero or one.

Every unresolved policy value (capacities, memory reserve, the decode-memory
charge policy identifier, per-request utilities, and the numerical policy)
is a required explicit argument. Unsupported, contradictory, malformed, or
incoherent state is returned as a ``MappingFailure``; this module never
fabricates, repairs, or silently narrows LP input.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from sarathi.core.datatypes.sequence_status import SequenceStatus

import lp_relaxation_scheduler as lrs

STAGE_STATE_MAPPING = "state_mapping"
CATEGORY_MAPPING_FAILURE = "mapping_failure"

OWNERSHIP_WAITING = "waiting"
OWNERSHIP_RUNNING = "running"

# The only decode-memory charge policy identifier this module accepts. It is
# a scoped fixture/mapper input, not a permanent resolution of the decode
# planning-charge OPEN decision.
DECODE_MEMORY_POLICY_CONSERVATIVE_ONE_BLOCK = "conservative_one_block_v1"


# ---------------------------------------------------------------------------
# Immutable records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RequestUtility:
    """One explicit utility triple for a single included request."""

    decode_utility: float
    prefill_token_utility: float
    preemption_penalty: float


@dataclass(frozen=True)
class RequestStateSnapshot:
    """Frozen, primitive-only observation of one included request."""

    raw_seq_id: int
    request_id: str
    order_key: tuple
    ownership: str
    status: str
    arrival_time: float
    prompt_len: int
    prompt_tokens_processed: int
    prompt_tokens_remaining: int
    prompt_processing_finished: bool
    logical_block_count: int
    physical_block_numbers: Optional[tuple]
    prefill_eligible: bool
    decode_eligible: bool
    preemption_eligible: bool
    prefill_fixed_charge: int
    decode_charge: int
    preemption_recovery: int
    utility: RequestUtility


@dataclass(frozen=True)
class StateSnapshot:
    """One coherent, immutable observation ready for the mathematical layer."""

    snapshot_id: str
    snapshot_time: float
    scheduler_iteration_id: int
    num_pipeline_stages: int
    num_running_batches: int
    resident_count: int
    resident_limit: int
    free_physical_blocks: int
    memory_reserve: int
    decode_memory_policy_id: str
    numerical_policy: lrs.NumericalPolicy
    requests: tuple
    lp_problem: lrs.LPProblem


@dataclass(frozen=True)
class MappingFailure:
    """A precise, structured mapping rejection. Never carries a fabricated plan."""

    snapshot_id: Optional[str]
    stage: str
    category: str
    reason: str


class _MappingError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise _MappingError(reason)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


# ---------------------------------------------------------------------------
# Snapshot identity
# ---------------------------------------------------------------------------


def _compute_snapshot_id(
    iteration_id, snapshot_time, num_pipeline_stages, num_running_batches,
    resident_count, resident_limit, b_max, c_max, s_max, m_free,
    memory_reserve, decode_memory_policy_id, numerical_policy, request_states,
):
    """Deterministic SHA-256 content identifier over primitive snapshot data.

    ``request_states`` must already be in ascending canonical (raw_seq_id)
    order. Uses only primitive values; never Python's randomized ``hash()``,
    object identity, or a wall clock read here.
    """
    request_rows = [
        [
            rs.ownership, rs.raw_seq_id, rs.status, rs.arrival_time,
            rs.prompt_len, rs.prompt_tokens_processed,
            rs.prompt_tokens_remaining, rs.prompt_processing_finished,
            rs.logical_block_count,
            None if rs.physical_block_numbers is None
            else list(rs.physical_block_numbers),
            rs.prefill_eligible, rs.decode_eligible, rs.preemption_eligible,
            rs.prefill_fixed_charge, rs.decode_charge, rs.preemption_recovery,
            rs.utility.decode_utility, rs.utility.prefill_token_utility,
            rs.utility.preemption_penalty,
        ]
        for rs in request_states
    ]
    payload = [
        "lpserve_state_mapping_snapshot_v1",
        iteration_id, snapshot_time, num_pipeline_stages, num_running_batches,
        resident_count, resident_limit, b_max, c_max, s_max, m_free,
        memory_reserve, decode_memory_policy_id,
        numerical_policy.policy_id, numerical_policy.feasibility_tol,
        numerical_policy.integrality_tol, numerical_policy.objective_abs_tol,
        numerical_policy.objective_rel_tol,
        request_rows,
    ]
    blob = json.dumps(payload, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------


def map_scheduler_state(
    scheduler,
    *,
    snapshot_time: float,
    b_max: int,
    c_max: int,
    s_max: int,
    memory_reserve: int,
    decode_memory_policy_id: str,
    utilities: Iterable[Tuple[int, RequestUtility]],
    numerical_policy: lrs.NumericalPolicy,
):
    """Read ``scheduler`` state and return a ``StateSnapshot`` or ``MappingFailure``.

    ``scheduler`` is read through ``waiting``, ``running``, ``_iteration_id``,
    ``num_running_batches``, ``scheduler_config.num_pipeline_stages``,
    ``scheduler_config.max_num_seqs``, and ``block_manager``; only read-only
    block-manager accessors (``block_size``, ``is_allocated``,
    ``get_block_table``, ``get_num_free_gpu_blocks``) are called. No
    mutating method is called and no LPServe object is retained in the
    result.

    ``utilities`` is an iterable of ``(raw_seq_id, RequestUtility)`` pairs,
    one per request expected to be included after arrival/completion
    filtering; a missing, duplicate, or extra row is a mapping failure.
    """
    try:
        return _map_scheduler_state(
            scheduler, snapshot_time, b_max, c_max, s_max, memory_reserve,
            decode_memory_policy_id, utilities, numerical_policy,
        )
    except _MappingError as err:
        return MappingFailure(
            None, STAGE_STATE_MAPPING, CATEGORY_MAPPING_FAILURE, err.reason,
        )


_MISSING = object()


def _read(obj, name):
    value = getattr(obj, name, _MISSING)
    _require(value is not _MISSING, f"required field {name!r} is missing")
    return value


def _map_scheduler_state(
    scheduler, snapshot_time, b_max, c_max, s_max, memory_reserve,
    decode_memory_policy_id, utilities, numerical_policy,
):
    _require(
        _is_real(snapshot_time), "snapshot_time must be a finite real number",
    )
    for name, value in (
        ("b_max", b_max), ("s_max", s_max), ("memory_reserve", memory_reserve),
    ):
        _require(
            _is_int(value) and value >= 0,
            f"{name} must be a non-negative integer",
        )
    _require(_is_int(c_max) and c_max > 0, "c_max must be a positive integer")
    _require(
        isinstance(decode_memory_policy_id, str) and decode_memory_policy_id,
        "decode_memory_policy_id must be a non-empty string",
    )
    _require(
        decode_memory_policy_id == DECODE_MEMORY_POLICY_CONSERVATIVE_ONE_BLOCK,
        "unsupported decode_memory_policy_id "
        f"{decode_memory_policy_id!r}; only "
        f"{DECODE_MEMORY_POLICY_CONSERVATIVE_ONE_BLOCK!r} is accepted",
    )
    _require(
        isinstance(numerical_policy, lrs.NumericalPolicy),
        "numerical_policy must be a NumericalPolicy",
    )

    scheduler_config = _read(scheduler, "scheduler_config")
    num_pipeline_stages = _read(scheduler_config, "num_pipeline_stages")
    max_num_seqs = _read(scheduler_config, "max_num_seqs")
    num_running_batches = _read(scheduler, "num_running_batches")
    iteration_id = _read(scheduler, "_iteration_id")
    waiting = _read(scheduler, "waiting")
    running = _read(scheduler, "running")
    block_manager = _read(scheduler, "block_manager")
    block_manager_block_size = _read(block_manager, "block_size")

    _require(
        num_pipeline_stages == 1,
        f"unsupported pipeline stage count {num_pipeline_stages!r}; only 1 "
        "is supported",
    )
    _require(
        num_running_batches == 0,
        f"unsupported in-flight state: num_running_batches="
        f"{num_running_batches!r}; only 0 is supported",
    )
    _require(
        _is_int(max_num_seqs) and max_num_seqs > 0,
        "scheduler_config.max_num_seqs must be a positive integer",
    )
    _require(_is_int(iteration_id), "scheduler _iteration_id must be an integer")

    resident_count = len(running)
    _require(
        resident_count <= max_num_seqs,
        f"resident count {resident_count} exceeds max_num_seqs {max_num_seqs}",
    )

    # Ownership: unique raw_seq_id across the complete waiting/running
    # universe, checked before any arrival/completion filtering so a
    # duplicate is caught even if one copy would otherwise be excluded.
    owners = {}
    for label, collection in (
        (OWNERSHIP_WAITING, waiting), (OWNERSHIP_RUNNING, running),
    ):
        for seq in collection:
            raw_id = getattr(seq, "seq_id", None)
            _require(
                _is_int(raw_id) and raw_id >= 0,
                f"{label} request has an unsupported seq_id shape: {raw_id!r}",
            )
            if raw_id in owners:
                raise _MappingError(
                    f"raw_seq_id {raw_id} is owned by more than one entry "
                    f"(already seen owned by {owners[raw_id][0]!r}, again "
                    f"in {label!r})"
                )
            owners[raw_id] = (label, seq)

    included = []
    for raw_id in sorted(owners):
        label, seq = owners[raw_id]
        arrival_time = seq.arrival_time
        if not (_is_real(arrival_time) and arrival_time <= snapshot_time):
            continue
        if seq.is_finished():
            continue
        included.append((raw_id, label, seq))

    util_map = {}
    for row in utilities:
        _require(
            isinstance(row, tuple) and len(row) == 2,
            "each utility row must be a (raw_seq_id, RequestUtility) pair, "
            f"got {row!r}",
        )
        raw_id, utility = row
        _require(
            _is_int(raw_id),
            f"utility row raw_seq_id must be an integer, got {raw_id!r}",
        )
        _require(
            isinstance(utility, RequestUtility),
            f"utility row for raw_seq_id {raw_id} must be a RequestUtility",
        )
        for name in (
            "decode_utility", "prefill_token_utility", "preemption_penalty",
        ):
            value = getattr(utility, name)
            _require(
                _is_real(value),
                f"utility {name} for raw_seq_id {raw_id} must be a finite "
                f"number, got {value!r}",
            )
        _require(
            raw_id not in util_map,
            f"duplicate utility row for raw_seq_id {raw_id}",
        )
        util_map[raw_id] = utility

    included_ids = {raw_id for raw_id, _, _ in included}
    missing = included_ids - util_map.keys()
    extra = util_map.keys() - included_ids
    _require(
        not missing, f"missing utility rows for raw_seq_id(s) {sorted(missing)}",
    )
    _require(
        not extra, f"extra utility rows for raw_seq_id(s) {sorted(extra)}",
    )

    request_states = []
    request_inputs = []
    legal_ids = set()
    for raw_id, label, seq in included:
        request_id = str(raw_id)
        order_key = (raw_id,)
        status = seq.get_status()
        prompt_len = seq.get_prompt_len()
        processed = seq.get_num_prompt_tokens_processed()
        _require(
            _is_int(prompt_len) and prompt_len >= 0,
            f"raw_seq_id {raw_id}: malformed prompt length {prompt_len!r}",
        )
        _require(
            _is_int(processed) and processed >= 0,
            f"raw_seq_id {raw_id}: malformed processed-prompt count "
            f"{processed!r}",
        )
        remainder = prompt_len - processed
        _require(
            0 <= remainder <= prompt_len,
            f"raw_seq_id {raw_id}: negative or excessive prompt progress "
            f"yields remainder {remainder} (prompt_len={prompt_len}, "
            f"processed={processed})",
        )
        finished_flag = seq.prompt_processing_finished
        _require(
            finished_flag == (remainder == 0),
            f"raw_seq_id {raw_id}: prompt_processing_finished="
            f"{finished_flag!r} disagrees with prompt remainder {remainder}",
        )
        seq_block_size = seq.block_size
        _require(
            seq_block_size == block_manager_block_size,
            f"raw_seq_id {raw_id}: sequence block_size {seq_block_size!r} "
            f"disagrees with block-manager block_size "
            f"{block_manager_block_size!r}",
        )
        logical_block_count = len(seq.logical_token_blocks)
        is_allocated = block_manager.is_allocated(seq)

        if label == OWNERSHIP_WAITING:
            _require(
                status == SequenceStatus.WAITING,
                f"raw_seq_id {raw_id}: waiting-owned request has status "
                f"{status!r}, expected WAITING",
            )
            _require(
                not is_allocated,
                f"raw_seq_id {raw_id}: waiting-owned request is allocated",
            )
            _require(
                remainder > 0,
                f"raw_seq_id {raw_id}: waiting-owned request has "
                f"non-positive prompt remainder {remainder}",
            )
            _require(
                not finished_flag,
                f"raw_seq_id {raw_id}: waiting-owned request has "
                "prompt_processing_finished True",
            )
            physical_block_numbers = None
            prefill_eligible, decode_eligible, preemption_eligible = (
                True, False, False,
            )
            prefill_fixed_charge = logical_block_count
            decode_charge = 0
            preemption_recovery = 0
        else:
            _require(
                status == SequenceStatus.PAUSED,
                f"raw_seq_id {raw_id}: resident request has unsupported "
                f"status {status!r}, expected PAUSED",
            )
            _require(
                is_allocated,
                f"raw_seq_id {raw_id}: resident request is not allocated",
            )
            physical_block_numbers = tuple(block_manager.get_block_table(seq))
            gap = logical_block_count - len(physical_block_numbers)
            if remainder > 0:
                _require(
                    gap == 0,
                    f"raw_seq_id {raw_id}: resident partial-prefill "
                    f"logical/physical block gap {gap} must be 0",
                )
                prefill_eligible, decode_eligible, preemption_eligible = (
                    True, False, True,
                )
                prefill_fixed_charge = 0
                decode_charge = 0
                preemption_recovery = len(physical_block_numbers)
            else:
                _require(
                    gap in (0, 1),
                    f"raw_seq_id {raw_id}: resident decode logical/physical "
                    f"block gap {gap} is outside {{0, 1}}",
                )
                prefill_eligible, decode_eligible, preemption_eligible = (
                    False, True, True,
                )
                prefill_fixed_charge = 0
                decode_charge = 1
                preemption_recovery = len(physical_block_numbers)

        if preemption_eligible:
            legal_ids.add(request_id)

        prefill_upper_bound = min(remainder, c_max) if prefill_eligible else 0
        utility = util_map[raw_id]

        request_states.append(RequestStateSnapshot(
            raw_seq_id=raw_id,
            request_id=request_id,
            order_key=order_key,
            ownership=label,
            status=status.name,
            arrival_time=float(seq.arrival_time),
            prompt_len=prompt_len,
            prompt_tokens_processed=processed,
            prompt_tokens_remaining=remainder,
            prompt_processing_finished=finished_flag,
            logical_block_count=logical_block_count,
            physical_block_numbers=physical_block_numbers,
            prefill_eligible=prefill_eligible,
            decode_eligible=decode_eligible,
            preemption_eligible=preemption_eligible,
            prefill_fixed_charge=prefill_fixed_charge,
            decode_charge=decode_charge,
            preemption_recovery=preemption_recovery,
            utility=utility,
        ))
        request_inputs.append(lrs.RequestInput(
            request_id=request_id,
            order_key=order_key,
            prompt_tokens_remaining=remainder,
            prefill_upper_bound=prefill_upper_bound,
            prefill_eligible=prefill_eligible,
            decode_eligible=decode_eligible,
            preemption_eligible=preemption_eligible,
            prefill_fixed_charge=prefill_fixed_charge,
            decode_charge=decode_charge,
            preemption_recovery=preemption_recovery,
            decode_utility=utility.decode_utility,
            prefill_token_utility=utility.prefill_token_utility,
            preemption_penalty=utility.preemption_penalty,
        ))

    m_free = block_manager.get_num_free_gpu_blocks()
    _require(
        _is_int(m_free) and m_free >= 0,
        f"block manager reported a malformed free-block count {m_free!r}",
    )

    snapshot_id = _compute_snapshot_id(
        iteration_id, snapshot_time, num_pipeline_stages, num_running_batches,
        resident_count, max_num_seqs, b_max, c_max, s_max, m_free,
        memory_reserve, decode_memory_policy_id, numerical_policy,
        request_states,
    )

    if not request_inputs:
        return MappingFailure(
            snapshot_id, STAGE_STATE_MAPPING, CATEGORY_MAPPING_FAILURE,
            "the arrived, unfinished request universe is empty; the "
            "accepted LPProblem type cannot represent an empty request set",
        )

    problem = lrs.LPProblem(
        requests=tuple(request_inputs),
        legal_preemption_ids=frozenset(legal_ids),
        b_max=b_max, c_max=c_max, s_max=s_max, m_free=m_free,
        w=memory_reserve,
        problem_id=snapshot_id,
        decode_memory_policy_id=decode_memory_policy_id,
        numerical_policy=numerical_policy,
    )
    canonical = lrs.validate_problem(problem)
    if isinstance(canonical, lrs.Failure):
        return MappingFailure(
            snapshot_id, STAGE_STATE_MAPPING, CATEGORY_MAPPING_FAILURE,
            "the mapped problem was rejected by validate_problem: "
            f"{canonical.category}: {canonical.reason}",
        )

    return StateSnapshot(
        snapshot_id=snapshot_id,
        snapshot_time=float(snapshot_time),
        scheduler_iteration_id=iteration_id,
        num_pipeline_stages=num_pipeline_stages,
        num_running_batches=num_running_batches,
        resident_count=resident_count,
        resident_limit=max_num_seqs,
        free_physical_blocks=m_free,
        memory_reserve=memory_reserve,
        decode_memory_policy_id=decode_memory_policy_id,
        numerical_policy=numerical_policy,
        requests=tuple(request_states),
        lp_problem=canonical,
    )
