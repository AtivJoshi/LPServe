"""Read-only LPServe state mapper for the LP-relaxation scheduling layer.

One scheduling decision maps live state in four steps
(docs/lp_scheduler_design.md section 12.3):

1. ``capture_scheduler_state`` reads the scheduler holder's inherited
   ``waiting``/``running`` collections, its configuration, and its block
   manager exactly once and copies every needed fact into an immutable
   ``CapturedState``;
2. a rank profile (``vllm_oriented_ranks``) builds service and victim ranks
   from the captured queue positions;
3. ``construct_coefficients`` builds one utility triple per request from the
   captured data and an explicit coefficient policy;
4. ``build_state_snapshot`` builds the immutable ``LPProblem``.

``map_scheduler_state`` runs these steps in order. Only the capture reads
live state, and it calls only read-only scheduler, sequence, and block
manager accessors, never allocation, append, free, preemption, status
transition, or queue-mutation methods.

Supported state shape (docs/lp_scheduler_design.md sections 12.4 and 13):

- exactly one pipeline stage and zero running batches (a quiescent,
  pre-mutation boundary with no batch in flight);
- ``waiting`` exclusively owns unallocated unfinished requests, and
  ``running`` exclusively owns allocated unfinished (resident) requests;
  every ``running`` entry has arrived and is unfinished, so ``len(running)``
  is the authoritative resident count;
- every raw ``Sequence.seq_id`` is an exact non-negative, non-boolean
  integer, unique across both collections, and every owned request has a
  finite real ``arrival_time``;
- the central block manager's ``block_tables`` key set exactly equals the
  raw IDs owned by ``running``; tables hold distinct blocks, no block is
  shared, and free plus allocated blocks equal the total pool;
- an included waiting request has status ``WAITING``, is unallocated, has
  positive prompt remainder, and is admissible: context length at most
  ``max_model_len`` and full initial allocation plus the shared watermark
  at most the total pool;
- an included resident has status ``PAUSED`` and is allocated; a partial
  prefill has a zero logical/physical block gap and allocates nothing, and a
  decode has an exact gap (decode demand) of zero or one.

Unsupported, contradictory, malformed, or incoherent state -- including
malformed input that would otherwise raise an ordinary
``AttributeError``/``TypeError``/``ValueError``/``KeyError``/
``AssertionError``/``OverflowError`` -- is returned as a ``MappingFailure``;
this module never fabricates, repairs, or silently narrows LP input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional

from sarathi.core.datatypes.sequence_status import SequenceStatus

import lp_relaxation_scheduler as lrs

STAGE_STATE_MAPPING = "state_mapping"
CATEGORY_MAPPING_FAILURE = "mapping_failure"

OWNERSHIP_WAITING = "waiting"
OWNERSHIP_RUNNING = "running"


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
class CapturedRequest:
    """Frozen, primitive-only facts about one included request.

    ``queue_position`` is the index in the owning collection. For a waiting
    request, ``prefill_fixed_charge`` is the native full initial allocation;
    ``decode_demand`` is the exact logical/physical block gap of a decode;
    ``preemption_recovery`` is the current physical block-table length.
    """

    raw_seq_id: int
    request_id: str
    order_key: tuple
    ownership: str
    queue_position: int
    status: str
    arrival_time: float
    prompt_len: int
    prompt_tokens_processed: int
    prompt_tokens_remaining: int
    prompt_processing_finished: bool
    output_len: int
    context_len: int
    logical_block_count: int
    physical_block_count: int
    prefill_eligible: bool
    decode_eligible: bool
    preemption_eligible: bool
    prefill_fixed_charge: int
    decode_demand: int
    preemption_recovery: int


@dataclass(frozen=True)
class CapturedState:
    """The one immutable capture of a scheduling decision.

    ``requests`` is ascending by raw ID. ``running_order`` lists every
    resident in current ``running`` order; ``arrived_waiting_order`` lists
    the included waiting requests in current ``waiting`` order.
    """

    snapshot_id: str
    snapshot_time: float
    scheduler_iteration_id: int
    num_pipeline_stages: int
    num_running_batches: int
    resident_count: int
    resident_limit: int
    max_model_len: int
    block_size: int
    total_blocks: int
    free_physical_blocks: int
    watermark_blocks: int
    requests: tuple
    running_order: tuple
    arrived_waiting_order: tuple


@dataclass(frozen=True)
class RequestStateSnapshot(CapturedRequest):
    """Captured facts plus the ranks and utility fixed for this decision."""

    service_rank: Optional[int]
    victim_rank: Optional[int]
    utility: RequestUtility


@dataclass(frozen=True)
class StateSnapshot(CapturedState):
    """One coherent, immutable observation ready for the mathematical layer.

    ``requests`` holds ``RequestStateSnapshot`` records.
    """

    numerical_policy: lrs.NumericalPolicy
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
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


_ORDINARY_ERRORS = (
    AttributeError, TypeError, ValueError, KeyError, AssertionError,
    OverflowError,
)


def _guard(snapshot_id, fn, *args):
    """Run one mapping step, converting rejections into ``MappingFailure``."""
    try:
        return fn(*args)
    except _MappingError as err:
        return MappingFailure(
            snapshot_id, STAGE_STATE_MAPPING, CATEGORY_MAPPING_FAILURE,
            err.reason,
        )
    except _ORDINARY_ERRORS as err:
        # A malformed scheduler field, collection shape, policy input, or
        # read-only state access raised an ordinary data/access error rather
        # than an explicit _require check.
        return MappingFailure(
            snapshot_id, STAGE_STATE_MAPPING, CATEGORY_MAPPING_FAILURE,
            f"malformed input or scheduler state raised "
            f"{type(err).__name__}: {err}",
        )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def map_scheduler_state(
    scheduler,
    *,
    snapshot_time: float,
    b_max: int,
    c_max: int,
    s_max: int,
    coefficient_policy,
    rank_profile,
    numerical_policy: lrs.NumericalPolicy,
) -> "StateSnapshot | MappingFailure":
    """Capture ``scheduler`` once and return a ``StateSnapshot`` or
    ``MappingFailure``.

    ``coefficient_policy`` and ``rank_profile`` are explicit policy inputs;
    see ``construct_coefficients`` and ``vllm_oriented_ranks``. Neither
    receives a live object.
    """
    captured = capture_scheduler_state(scheduler, snapshot_time=snapshot_time)
    if isinstance(captured, MappingFailure):
        return captured
    ranks = _guard(captured.snapshot_id, _ranks, captured, rank_profile)
    if isinstance(ranks, MappingFailure):
        return ranks
    coefficients = construct_coefficients(captured, coefficient_policy)
    if isinstance(coefficients, MappingFailure):
        return coefficients
    return build_state_snapshot(
        captured, ranks=ranks, coefficients=coefficients, b_max=b_max,
        c_max=c_max, s_max=s_max, numerical_policy=numerical_policy,
    )


def capture_scheduler_state(scheduler, *, snapshot_time):
    """The single live read of one decision: ``CapturedState`` or
    ``MappingFailure``. ``scheduler`` is read through ``waiting``,
    ``running``, ``_iteration_id``, ``num_running_batches``,
    ``scheduler_config`` (``num_pipeline_stages``, ``max_num_seqs``,
    ``max_model_len``), and read-only ``block_manager`` accessors."""
    return _guard(None, _capture, scheduler, snapshot_time)


def vllm_oriented_ranks(captured):
    """Initial vLLM-oriented rank profile.

    Service rank: current ``running`` position (including partial
    prefills), then arrived ``waiting`` position. Victim rank: reverse
    ``running`` position; waiting requests have none. Returns one
    ``(service_rank, victim_rank)`` pair per request of
    ``captured.requests``. This profile is not imposed on other policies.
    """
    service, victim = {}, {}
    residents = len(captured.running_order)
    for position, raw_id in enumerate(captured.running_order):
        service[raw_id] = position
        victim[raw_id] = residents - 1 - position
    for position, raw_id in enumerate(captured.arrived_waiting_order):
        service[raw_id] = residents + position
    return tuple(
        (service[r.raw_seq_id], victim.get(r.raw_seq_id))
        for r in captured.requests
    )


def rank_based_utility(captured, request, *, decode_scale, prefill_scale,
                       penalty_scale):
    """Rank-based coefficient policy for ``construct_coefficients``.

    Bind the three scales (for example with ``functools.partial``) to obtain
    a ``policy(captured, request)``. With ``r`` the zero-based position of
    ``request`` in ``running_order`` followed by ``arrived_waiting_order``
    and ``s`` its zero-based ``running_order`` position::

        decode_utility        = decode_scale / (1 + r)
        prefill_token_utility = prefill_scale / (1 + r)
        preemption_penalty    = penalty_scale / (1 + s) for an eligible
                                resident victim, otherwise 0

    The prefill utility stays per token. Each scale must be a finite,
    strictly positive, non-boolean real; anything else is a mapping failure.
    Reads only the immutable captured data. This is one candidate family
    (docs/lp_scheduler_design.md D-01 to D-04 remain OPEN), not a selected
    policy.
    """
    for name, value in (
        ("decode_scale", decode_scale), ("prefill_scale", prefill_scale),
        ("penalty_scale", penalty_scale),
    ):
        _require(
            _is_real(value) and value > 0,
            f"rank-based utility {name} must be a finite positive real "
            f"number, got {value!r}",
        )
    raw_id = request.raw_seq_id
    service_position = (
        captured.running_order + captured.arrived_waiting_order
    ).index(raw_id)
    penalty = 0.0
    if request.preemption_eligible:
        penalty = penalty_scale / (1 + captured.running_order.index(raw_id))
    return RequestUtility(
        decode_utility=decode_scale / (1 + service_position),
        prefill_token_utility=prefill_scale / (1 + service_position),
        preemption_penalty=penalty,
    )


def coordinated_rank_utility(captured, request, *, rank_base, c_max, b_max):
    """Coordinated rank coefficient policy for ``construct_coefficients``.

    Bind ``rank_base`` and the decision's configured ``c_max`` and ``b_max``
    (for example with ``functools.partial``) to obtain a
    ``policy(captured, request)``. With ``r`` the zero-based position of
    ``request`` in ``running_order`` followed by ``arrived_waiting_order``::

        w = rank_base ** r
        q = min(prompt_tokens_remaining, c_max, b_max)   (prefill-eligible)

        decode_utility        = w for a decode-eligible request, otherwise 0
        prefill_token_utility = w / q for a prefill-eligible request,
                                otherwise 0
        preemption_penalty    = w for an eligible resident victim,
                                otherwise 0

    A complete available chunk of ``q`` prefill tokens and one decode
    therefore earn the same request-specific value ``w``. The weight uses the
    service position for every term; the rank profile's separate victim rank
    is not read. ``rank_base`` must be a finite real in (0, 1]; ``c_max`` and
    ``b_max`` must be positive integers. Reads only the immutable captured
    data. This is one provisional candidate (D-01 to D-04 remain OPEN), not a
    selected policy.
    """
    _require(
        _is_real(rank_base) and 0 < rank_base <= 1,
        f"coordinated utility rank_base must be a finite real in (0, 1], "
        f"got {rank_base!r}",
    )
    for name, value in (("c_max", c_max), ("b_max", b_max)):
        _require(
            _is_int(value) and value > 0,
            f"coordinated utility {name} must be a positive integer, got "
            f"{value!r}",
        )
    raw_id = request.raw_seq_id
    weight = rank_base ** (
        captured.running_order + captured.arrived_waiting_order
    ).index(raw_id)
    prefill = 0.0
    if request.prefill_eligible:
        chunk = min(request.prompt_tokens_remaining, c_max, b_max)
        _require(
            chunk >= 1,
            f"raw_seq_id {raw_id}: coordinated utility needs a positive "
            f"available chunk, got {chunk}",
        )
        prefill = weight / chunk
    return RequestUtility(
        decode_utility=weight if request.decode_eligible else 0.0,
        prefill_token_utility=prefill,
        preemption_penalty=weight if request.preemption_eligible else 0.0,
    )


def reciprocal_square_rank_utility(captured, request, *, c_max, b_max):
    """Reciprocal-square coordinated coefficient policy for
    ``construct_coefficients``.

    Bind the decision's configured ``c_max`` and ``b_max`` (for example with
    ``functools.partial``) to obtain a ``policy(captured, request)``. With
    ``r`` the zero-based position of ``request`` in ``running_order``
    followed by ``arrived_waiting_order``::

        w = 1 / (1 + r) ** 2
        q = min(prompt_tokens_remaining, c_max, b_max)   (prefill-eligible)

        decode_utility        = w for a decode-eligible request, otherwise 0
        prefill_token_utility = w / q for a prefill-eligible request,
                                otherwise 0
        preemption_penalty    = w for an eligible resident victim,
                                otherwise 0

    The terms are those of ``coordinated_rank_utility``; only the weight
    decays more slowly with rank. The exponent is fixed at two. The weight
    uses the service position for every term; the rank profile's separate
    victim rank is not read. ``c_max`` and ``b_max`` must be positive
    integers. Reads only the immutable captured data. This is one
    provisional candidate (D-01 to D-04 remain OPEN), not a selected policy.
    """
    for name, value in (("c_max", c_max), ("b_max", b_max)):
        _require(
            _is_int(value) and value > 0,
            f"reciprocal-square utility {name} must be a positive integer, "
            f"got {value!r}",
        )
    raw_id = request.raw_seq_id
    position = (
        captured.running_order + captured.arrived_waiting_order
    ).index(raw_id)
    weight = 1.0 / (1 + position) ** 2
    prefill = 0.0
    if request.prefill_eligible:
        chunk = min(request.prompt_tokens_remaining, c_max, b_max)
        _require(
            chunk >= 1,
            f"raw_seq_id {raw_id}: reciprocal-square utility needs a positive "
            f"available chunk, got {chunk}",
        )
        prefill = weight / chunk
    return RequestUtility(
        decode_utility=weight if request.decode_eligible else 0.0,
        prefill_token_utility=prefill,
        preemption_penalty=weight if request.preemption_eligible else 0.0,
    )


def construct_coefficients(captured, policy):
    """Pure coefficient construction for one decision.

    ``policy`` is one of:

    - a ``RequestUtility``: the explicit uniform triple for every request;
    - a ``Mapping[int, RequestUtility]`` keyed by raw ``seq_id`` whose key
      set equals the captured request set exactly;
    - a callable ``policy(captured, request) -> RequestUtility`` evaluated on
      immutable captured data only (a state-dependent policy), such as
      ``rank_based_utility`` with its scales bound.

    Returns a tuple of validated ``RequestUtility`` copies aligned with
    ``captured.requests``, or ``MappingFailure``. Every value must be finite
    and non-negative. No live object is passed and no solve is performed.
    """
    return _guard(
        getattr(captured, "snapshot_id", None), _coefficients, captured, policy,
    )


def build_state_snapshot(
    captured, *, ranks, coefficients, b_max, c_max, s_max, numerical_policy,
):
    """Build the immutable ``StateSnapshot`` and canonical ``LPProblem`` from
    captured data, ranks, and validated coefficients."""
    return _guard(
        captured.snapshot_id, _build, captured, ranks, coefficients, b_max,
        c_max, s_max, numerical_policy,
    )


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


_MISSING = object()


def _read(obj, name):
    value = getattr(obj, name, _MISSING)
    _require(value is not _MISSING, f"required field {name!r} is missing")
    return value


def _capture(scheduler, snapshot_time):
    _require(
        _is_real(snapshot_time), "snapshot_time must be a finite real number",
    )
    scheduler_config = _read(scheduler, "scheduler_config")
    num_pipeline_stages = _read(scheduler_config, "num_pipeline_stages")
    max_num_seqs = _read(scheduler_config, "max_num_seqs")
    max_model_len = _read(scheduler_config, "max_model_len")
    num_running_batches = _read(scheduler, "num_running_batches")
    iteration_id = _read(scheduler, "_iteration_id")
    waiting = _read(scheduler, "waiting")
    running = _read(scheduler, "running")
    block_manager = _read(scheduler, "block_manager")
    block_size = _read(block_manager, "block_size")
    total_blocks = _read(block_manager, "num_total_gpu_blocks")
    watermark_blocks = _read(block_manager, "watermark_blocks")

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
    for name, value, minimum in (
        ("scheduler_config.max_num_seqs", max_num_seqs, 1),
        ("scheduler_config.max_model_len", max_model_len, 1),
        ("block_manager.block_size", block_size, 1),
        ("block_manager.num_total_gpu_blocks", total_blocks, 1),
        ("block_manager.watermark_blocks", watermark_blocks, 0),
    ):
        _require(
            _is_int(value) and value >= minimum,
            f"{name} must be an integer of at least {minimum}, got {value!r}",
        )
    _require(_is_int(iteration_id), "scheduler _iteration_id must be an integer")

    resident_count = len(running)
    _require(
        resident_count <= max_num_seqs,
        f"resident count {resident_count} exceeds max_num_seqs {max_num_seqs}",
    )

    # Ownership: unique raw_seq_id across the complete waiting/running
    # universe, checked before any arrival/completion filtering.
    owners = {}
    for label, collection in (
        (OWNERSHIP_WAITING, waiting), (OWNERSHIP_RUNNING, running),
    ):
        for position, seq in enumerate(collection):
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
            owners[raw_id] = (label, seq, position)

    # Allocation ownership must agree in both directions: every
    # running-owned raw_seq_id has a central block table, and no table
    # belongs to a waiting-owned or unowned (orphan) raw_seq_id.
    running_ids = {r for r, (lb, _, _) in owners.items() if lb == OWNERSHIP_RUNNING}
    waiting_ids = {r for r, (lb, _, _) in owners.items() if lb == OWNERSHIP_WAITING}
    block_tables = _read(block_manager, "block_tables")
    block_table_keys = set(block_tables.keys())
    for key in block_table_keys:
        _require(
            _is_int(key) and key >= 0,
            f"central block manager has a malformed block-table key: {key!r}",
        )
    missing_tables = running_ids - block_table_keys
    unexpected_tables = block_table_keys - running_ids
    if missing_tables or unexpected_tables:
        waiting_owned_tables = unexpected_tables & waiting_ids
        orphan_tables = unexpected_tables - waiting_ids
        raise _MappingError(
            "central block-table ownership disagrees with running: missing "
            f"resident tables for raw_seq_id(s) {sorted(missing_tables)}; "
            "waiting-owned tables for raw_seq_id(s) "
            f"{sorted(waiting_owned_tables)}; orphan tables for "
            f"raw_seq_id(s) {sorted(orphan_tables)}"
        )

    # Non-sharing pool accounting: distinct blocks per table, no block in
    # two tables, and free + allocated == total.
    free_blocks = block_manager.get_num_free_gpu_blocks()
    _require(
        _is_int(free_blocks) and free_blocks >= 0,
        f"block manager reported a malformed free-block count {free_blocks!r}",
    )
    allocated = []
    for raw_id in sorted(running_ids):
        table = block_manager.get_block_table(owners[raw_id][1])
        _require(
            len(set(table)) == len(table),
            f"raw_seq_id {raw_id}: physical block table {table} repeats a "
            "block",
        )
        allocated.extend(table)
    _require(
        len(set(allocated)) == len(allocated),
        "a physical block is shared by two resident block tables; shared "
        "blocks are unsupported",
    )
    _require(
        free_blocks + len(allocated) == total_blocks,
        f"free blocks {free_blocks} plus allocated blocks {len(allocated)} "
        f"differ from the total pool {total_blocks}",
    )

    captured, running_order, waiting_order = [], [], []
    for raw_id in sorted(owners):
        label, seq, position = owners[raw_id]
        arrival_time = getattr(seq, "arrival_time", None)
        _require(
            _is_real(arrival_time),
            f"raw_seq_id {raw_id}: malformed arrival_time {arrival_time!r}; "
            "must be a finite real number",
        )
        arrived = arrival_time <= snapshot_time
        finished = seq.is_finished()
        if label == OWNERSHIP_RUNNING:
            _require(
                arrived and not finished,
                f"raw_seq_id {raw_id}: running holds a "
                f"{'finished' if finished else 'future'} request; every "
                "resident must be arrived and unfinished",
            )
        elif not arrived or finished:
            continue
        captured.append(_capture_request(
            raw_id, label, position, seq, block_manager, block_size,
            max_model_len, total_blocks, watermark_blocks,
        ))

    for seq in running:
        running_order.append(seq.seq_id)
    included_waiting = {r.raw_seq_id for r in captured
                        if r.ownership == OWNERSHIP_WAITING}
    for seq in waiting:
        if seq.seq_id in included_waiting:
            waiting_order.append(seq.seq_id)

    snapshot_id = str(iteration_id)
    _require(
        captured,
        "the arrived, unfinished request universe is empty; the "
        "accepted LPProblem type cannot represent an empty request set",
    )
    return CapturedState(
        snapshot_id=snapshot_id,
        snapshot_time=float(snapshot_time),
        scheduler_iteration_id=iteration_id,
        num_pipeline_stages=num_pipeline_stages,
        num_running_batches=num_running_batches,
        resident_count=resident_count,
        resident_limit=max_num_seqs,
        max_model_len=max_model_len,
        block_size=block_size,
        total_blocks=total_blocks,
        free_physical_blocks=free_blocks,
        watermark_blocks=watermark_blocks,
        requests=tuple(captured),
        running_order=tuple(running_order),
        arrived_waiting_order=tuple(waiting_order),
    )


def _capture_request(raw_id, label, position, seq, block_manager, block_size,
                     max_model_len, total_blocks, watermark_blocks):
    status = seq.get_status()
    prompt_len = seq.get_prompt_len()
    processed = seq.get_num_prompt_tokens_processed()
    output_len = seq.get_output_len()
    context_len = seq.get_len()
    for name, value in (
        ("prompt length", prompt_len), ("processed-prompt count", processed),
        ("output length", output_len), ("context length", context_len),
    ):
        _require(
            _is_int(value) and value >= 0,
            f"raw_seq_id {raw_id}: malformed {name} {value!r}",
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
    _require(
        seq.block_size == block_size,
        f"raw_seq_id {raw_id}: sequence block_size {seq.block_size!r} "
        f"disagrees with block-manager block_size {block_size!r}",
    )
    logical_block_count = len(seq.logical_token_blocks)
    is_allocated = block_manager.is_allocated(seq)
    decode_demand = 0
    preemption_recovery = 0
    prefill_fixed_charge = 0

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
        physical_block_count = 0
        prefill_fixed_charge = block_manager.get_num_initial_blocks(seq)
        _require(
            _is_int(prefill_fixed_charge) and prefill_fixed_charge >= 0,
            f"raw_seq_id {raw_id}: malformed initial allocation "
            f"{prefill_fixed_charge!r}",
        )
        # A registered or recomputation-expanded context that can never be
        # admitted is a pre-mutation failure, not a schedulable request.
        _require(
            context_len <= max_model_len,
            f"raw_seq_id {raw_id}: registered context length {context_len} "
            f"exceeds max_model_len {max_model_len}",
        )
        _require(
            prefill_fixed_charge + watermark_blocks <= total_blocks,
            f"raw_seq_id {raw_id}: full initial allocation "
            f"{prefill_fixed_charge} blocks plus watermark {watermark_blocks} "
            f"exceeds the total pool of {total_blocks} blocks",
        )
        prefill_eligible, decode_eligible, preemption_eligible = (
            True, False, False,
        )
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
        physical_block_count = len(block_manager.get_block_table(seq))
        preemption_recovery = physical_block_count
        gap = logical_block_count - physical_block_count
        if remainder > 0:
            _require(
                gap == 0,
                f"raw_seq_id {raw_id}: resident partial-prefill "
                f"logical/physical block gap {gap} must be 0",
            )
            prefill_eligible, decode_eligible, preemption_eligible = (
                True, False, True,
            )
        else:
            _require(
                gap in (0, 1),
                f"raw_seq_id {raw_id}: resident decode logical/physical "
                f"block gap {gap} is outside {{0, 1}}",
            )
            decode_demand = gap
            prefill_eligible, decode_eligible, preemption_eligible = (
                False, True, True,
            )

    return CapturedRequest(
        raw_seq_id=raw_id,
        request_id=str(raw_id),
        order_key=(raw_id,),
        ownership=label,
        queue_position=position,
        status=status.name,
        arrival_time=float(seq.arrival_time),
        prompt_len=prompt_len,
        prompt_tokens_processed=processed,
        prompt_tokens_remaining=remainder,
        prompt_processing_finished=finished_flag,
        output_len=output_len,
        context_len=context_len,
        logical_block_count=logical_block_count,
        physical_block_count=physical_block_count,
        prefill_eligible=prefill_eligible,
        decode_eligible=decode_eligible,
        preemption_eligible=preemption_eligible,
        prefill_fixed_charge=prefill_fixed_charge,
        decode_demand=decode_demand,
        preemption_recovery=preemption_recovery,
    )


# ---------------------------------------------------------------------------
# Ranks, coefficients, and problem construction
# ---------------------------------------------------------------------------


def _ranks(captured, rank_profile):
    _require(callable(rank_profile), "rank_profile must be callable")
    ranks = rank_profile(captured)
    _require(
        isinstance(ranks, tuple) and len(ranks) == len(captured.requests),
        "rank_profile must return one (service_rank, victim_rank) pair per "
        "captured request",
    )
    return ranks


def _coefficients(captured, policy):
    _require(
        isinstance(captured, CapturedState),
        "coefficients require a CapturedState",
    )
    requests = captured.requests
    if isinstance(policy, RequestUtility):
        triples = [policy] * len(requests)
    elif isinstance(policy, Mapping):
        for raw_id in policy:
            _require(
                _is_int(raw_id),
                f"coefficient key must be an integer raw_seq_id, got "
                f"{raw_id!r}",
            )
        included = {r.raw_seq_id for r in requests}
        missing = included - policy.keys()
        extra = policy.keys() - included
        _require(
            not missing,
            f"missing coefficients for raw_seq_id(s) {sorted(missing)}",
        )
        _require(
            not extra, f"extra coefficients for raw_seq_id(s) {sorted(extra)}",
        )
        triples = [policy[r.raw_seq_id] for r in requests]
    elif callable(policy):
        triples = [policy(captured, r) for r in requests]
    else:
        raise _MappingError(
            "coefficient_policy must be a RequestUtility, a mapping from "
            "raw_seq_id to RequestUtility, or a callable of captured data, "
            f"got {type(policy).__name__}"
        )
    copies = []
    for request, utility in zip(requests, triples):
        raw_id = request.raw_seq_id
        _require(
            isinstance(utility, RequestUtility),
            f"coefficient for raw_seq_id {raw_id} must be a RequestUtility",
        )
        values = (
            utility.decode_utility, utility.prefill_token_utility,
            utility.preemption_penalty,
        )
        for name, value in zip(
            ("decode_utility", "prefill_token_utility", "preemption_penalty"),
            values,
        ):
            _require(
                _is_real(value) and value >= 0,
                f"coefficient {name} for raw_seq_id {raw_id} must be a "
                f"finite non-negative number, got {value!r}",
            )
        copies.append(RequestUtility(*values))
    return tuple(copies)


def _build(captured, ranks, coefficients, b_max, c_max, s_max,
           numerical_policy):
    for name, value in (("b_max", b_max), ("c_max", c_max), ("s_max", s_max)):
        _require(
            _is_int(value) and value > 0, f"{name} must be a positive integer",
        )
    _require(
        isinstance(numerical_policy, lrs.NumericalPolicy),
        "numerical_policy must be a NumericalPolicy",
    )
    requests = captured.requests
    _require(
        isinstance(ranks, tuple) and len(ranks) == len(requests)
        and isinstance(coefficients, tuple)
        and len(coefficients) == len(requests),
        "ranks and coefficients must align with the captured requests",
    )
    states, inputs = [], []
    for captured_request, (service, victim), utility in zip(
        requests, ranks, coefficients,
    ):
        state = RequestStateSnapshot(
            **{name: getattr(captured_request, name)
               for name in CapturedRequest.__dataclass_fields__},
            service_rank=service, victim_rank=victim, utility=utility,
        )
        states.append(state)
        inputs.append(lrs.RequestInput(
            request_id=state.request_id,
            order_key=state.order_key,
            resident=state.ownership == OWNERSHIP_RUNNING,
            prompt_tokens_remaining=state.prompt_tokens_remaining,
            prefill_upper_bound=(
                min(state.prompt_tokens_remaining, c_max)
                if state.prefill_eligible else 0
            ),
            prefill_eligible=state.prefill_eligible,
            decode_eligible=state.decode_eligible,
            preemption_eligible=state.preemption_eligible,
            prefill_fixed_charge=state.prefill_fixed_charge,
            decode_demand=state.decode_demand,
            preemption_recovery=state.preemption_recovery,
            decode_utility=utility.decode_utility,
            prefill_token_utility=utility.prefill_token_utility,
            preemption_penalty=utility.preemption_penalty,
            service_rank=service,
            victim_rank=victim,
        ))

    problem = lrs.LPProblem(
        requests=tuple(inputs),
        legal_preemption_ids=frozenset(
            s.request_id for s in states if s.preemption_eligible),
        b_max=b_max, c_max=c_max, s_max=s_max,
        resident_limit=captured.resident_limit,
        resident_count=captured.resident_count,
        m_free=captured.free_physical_blocks,
        admission_watermark=captured.watermark_blocks,
        problem_id=captured.snapshot_id,
        numerical_policy=numerical_policy,
    )
    canonical = lrs.validate_problem(problem)
    _require(
        not isinstance(canonical, lrs.Failure),
        "the mapped problem was rejected by validate_problem: "
        f"{getattr(canonical, 'category', '')}: "
        f"{getattr(canonical, 'reason', '')}",
    )
    return StateSnapshot(
        **{name: getattr(captured, name)
           for name in CapturedState.__dataclass_fields__
           if name != "requests"},
        requests=tuple(states),
        numerical_policy=numerical_policy,
        lp_problem=canonical,
    )
