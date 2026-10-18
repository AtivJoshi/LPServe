"""Adapted upstream vLLM v0.31.0 reference scheduler.

``AdaptedVLLMScheduler`` reproduces the service and preemption preferences of
the upstream vLLM v0.31.0 scheduler (commit
``db9527a46873454610df6dbedf79a36d6bf1a7f6``, ``vllm/v1/core/sched/
scheduler.py`` ``Scheduler.schedule``) inside LPServe's full-context
allocation framework. It is distinct from the inherited ``VLLMScheduler``
and claims no upstream allocation trajectory, batch, physical layout, or
whole-engine equivalence. The specification and source provenance are in
``docs/LP Scheduler Research Context.md`` section 16.

Supported profile: FCFS, chunked prefill, ordinary text generation,
synchronous scheduling, one pipeline stage, no prefix caching, no
speculative decoding, and no long-prefill threshold.

Policy selection, computed without mutation:

1. Running pass, in current ``running`` order (resident partial prefills
   included). One combined token budget B: a decode takes one token, a
   prefill takes ``min(remaining prompt, remaining budget)``. The pass ends
   when the budget or the running list is exhausted.
2. When the attempted request's block demand exceeds the free blocks, the
   current running tail is evicted until it fits or the attempted request
   itself is evicted; then the running pass stops. Earlier accepted requests
   are never the tail, so accepted work is preserved.
3. After any eviction, no waiting request is admitted.
4. Otherwise arrived waiting requests are considered in current waiting order
   (future-dated entries are skipped without reordering arrived ones). The
   pass stops at the first resident-limit or admission-memory failure; a
   later smaller request is not searched for.

Resources come from the shared block manager. Admission allocates the full
current logical context and requires ``free - charge >= watermark_blocks``.
A resident partial prefill needs no blocks. A decode needs its
logical-versus-physical block gap (0 or 1); a zero-demand decode runs with
zero free blocks. All residents count against ``max_num_seqs``.

Execution layout: the complete proposal and its layout are validated before
mutation. Victims are preempted first in eviction order through the native
front insertion, so several victims end in reverse eviction order at the
waiting front. Selected work is then stably partitioned into prefills
followed by decodes, keeping selection order within each group; central
native calls and emitted metadata use that order, matching prompt-first input
packing. The commit order does not change which actions were selected.

Failure behavior:

- An unsupported entry state raises ``AdaptedVLLMSchedulingError`` before
  inherited bookkeeping.
- Unsupported or malformed state, a permanently inadmissible waiting head,
  and a failed layout check raise ``AdaptedVLLMSchedulingError`` before any
  native mutation.
- A proposal that only preempts raises ``AdaptedVLLMControlOnlyProposal``,
  carrying the faithful proposal, before mutation: the engine drops
  control-only outputs, so they cannot be executed.
- An exception after mutation begins propagates unchanged; the caller must
  discard the affected state. There is no rollback, retry, or recovery.
"""

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import SequenceScheduleMetadata
from sarathi.core.datatypes.sequence_status import SequenceStatus
from sarathi.core.scheduler.base_scheduler import BaseScheduler

ACTION_RESIDENT_PREFILL = "resident_prefill"
ACTION_DECODE = "decode"
ACTION_ADMISSION = "admission"

CATEGORY_UNSUPPORTED_STATE = "unsupported_state"
CATEGORY_MALFORMED_STATE = "malformed_state"
CATEGORY_INADMISSIBLE_HEAD = "inadmissible_head"
CATEGORY_CONTROL_ONLY = "control_only_proposal"
CATEGORY_LAYOUT = "layout_validation"

RUNNING_EXHAUSTED = "running_exhausted"
RUNNING_BUDGET_EXHAUSTED = "token_budget_exhausted"
RUNNING_SELF_EVICTED = "attempted_request_evicted"

WAITING_SKIPPED_AFTER_PREEMPTION = "skipped_after_preemption"
WAITING_EXHAUSTED = "arrived_waiting_exhausted"
WAITING_BUDGET_EXHAUSTED = "token_budget_exhausted"
WAITING_RESIDENT_LIMIT = "resident_limit"
WAITING_ADMISSION_MEMORY = "admission_memory"


@dataclass(frozen=True)
class ReferenceAction:
    """One selected execution action. ``num_tokens`` is the prompt chunk of a
    prefill or 1 for a decode; ``block_demand`` the blocks it allocates."""

    seq_id: int
    kind: str
    num_tokens: int
    block_demand: int

    @property
    def is_prefill(self) -> bool:
        return self.kind != ACTION_DECODE


@dataclass(frozen=True)
class ReferenceProposal:
    """The complete policy decision, computed before any mutation.

    ``actions`` is in service-selection order and ``victims`` in eviction
    order. ``free_blocks``, ``watermark_blocks``, and the queue orders are the
    values observed before selection."""

    decision_time: float
    running_order: Tuple[int, ...]
    arrived_waiting_order: Tuple[int, ...]
    free_blocks: int
    watermark_blocks: int
    token_budget: int
    victims: Tuple[int, ...]
    actions: Tuple[ReferenceAction, ...]
    running_stop_reason: str
    waiting_stop_reason: str

    @property
    def execution_order(self) -> Tuple[ReferenceAction, ...]:
        """Prefills, then decodes, each in selection order."""
        return tuple(a for a in self.actions if a.is_prefill) + tuple(
            a for a in self.actions if not a.is_prefill
        )


class AdaptedVLLMSchedulingError(RuntimeError):
    """A failure raised before native execution mutation."""

    def __init__(
        self, category: str, reason: str,
        proposal: Optional[ReferenceProposal] = None,
    ) -> None:
        self.category = category
        self.reason = reason
        self.proposal = proposal
        super().__init__(
            f"adapted vLLM reference failed before execution mutation "
            f"(category {category!r}): {reason}"
        )


class AdaptedVLLMControlOnlyProposal(AdaptedVLLMSchedulingError):
    """The policy selected preemption only. ``proposal`` is the faithful
    decision; no queue, block, or sequence state was changed."""

    def __init__(self, proposal: ReferenceProposal) -> None:
        super().__init__(
            CATEGORY_CONTROL_ONLY,
            f"proposal preempts {list(proposal.victims)} and selects no "
            "prefill or decode; the engine drops control-only outputs, so "
            "it is reported instead of executed",
            proposal,
        )


def _require(condition: bool, category: str, reason: str) -> None:
    if not condition:
        raise AdaptedVLLMSchedulingError(category, reason)


class AdaptedVLLMScheduler(BaseScheduler):
    """Constructed from an ``AdaptedVLLMSchedulerConfig`` through
    ``SchedulerRegistry``."""

    def schedule(self) -> SchedulerOutputs:
        # BaseScheduler.schedule() returns an ordinary empty output when its
        # running-batch limit is reached; reject that state visibly instead.
        stages = self.scheduler_config.num_pipeline_stages
        _require(
            stages == 1, CATEGORY_UNSUPPORTED_STATE,
            f"unsupported pipeline stage count {stages!r}; only 1 is supported",
        )
        _require(
            self.num_running_batches == 0, CATEGORY_UNSUPPORTED_STATE,
            "unsupported in-flight state: num_running_batches="
            f"{self.num_running_batches!r}; only 0 is supported",
        )
        return super().schedule()

    def _schedule(self) -> SchedulerOutputs:
        proposal = self._propose(time.monotonic())
        if not proposal.actions:
            if proposal.victims:
                raise AdaptedVLLMControlOnlyProposal(proposal)
            if proposal.arrived_waiting_order:
                # Nothing is resident, so the arrived head failed against the
                # whole free pool and can never be admitted.
                raise AdaptedVLLMSchedulingError(
                    CATEGORY_INADMISSIBLE_HEAD,
                    f"arrived waiting head {proposal.arrived_waiting_order[0]} "
                    f"cannot be admitted with nothing resident "
                    f"({proposal.waiting_stop_reason}); it never can be",
                    proposal,
                )
            # Ordinary idle: no resident and no arrived waiting request.
            return SchedulerOutputs(
                self._iteration_id,
                ignored_seq_ids=[],
                preempted_seq_ids=[],
                scheduled_seq_metadata_list=[],
            )
        return self._commit(proposal)

    # ------------------------------------------------------------------
    # Policy selection (no mutation)
    # ------------------------------------------------------------------

    def _propose(self, now: float) -> ReferenceProposal:
        manager = self.block_manager
        config = self.scheduler_config
        arrived = self._validate_state(now)

        running = list(self.running)
        running_order = tuple(s.seq_id for s in running)
        free = manager.get_num_free_gpu_blocks()
        free_before = free
        watermark = manager.watermark_blocks
        budget = config.max_num_batched_tokens
        victims = []
        actions = []

        index = 0
        running_stop = RUNNING_EXHAUSTED
        while index < len(running) and budget > 0:
            seq = running[index]
            if seq.prompt_processing_finished:
                kind, tokens = ACTION_DECODE, 1
                demand = self._block_gap(seq)
            else:
                kind = ACTION_RESIDENT_PREFILL
                tokens = min(self._remaining_prompt(seq), budget)
                demand = 0
            evicted_self = False
            while demand > free:
                victim = running.pop()
                victims.append(victim.seq_id)
                free += len(manager.block_tables[victim.seq_id])
                if victim is seq:
                    evicted_self = True
                    break
            if evicted_self:
                running_stop = RUNNING_SELF_EVICTED
                break
            free -= demand
            budget -= tokens
            actions.append(ReferenceAction(seq.seq_id, kind, tokens, demand))
            index += 1
        else:
            if index < len(running):
                running_stop = RUNNING_BUDGET_EXHAUSTED

        if victims:
            waiting_stop = WAITING_SKIPPED_AFTER_PREEMPTION
        else:
            waiting_stop = WAITING_EXHAUSTED
            residents = len(running)
            for seq in arrived:
                if budget <= 0:
                    waiting_stop = WAITING_BUDGET_EXHAUSTED
                    break
                if residents >= config.max_num_seqs:
                    waiting_stop = WAITING_RESIDENT_LIMIT
                    break
                charge = manager.get_num_initial_blocks(seq)
                if free - charge < watermark:
                    waiting_stop = WAITING_ADMISSION_MEMORY
                    break
                tokens = min(self._remaining_prompt(seq), budget)
                free -= charge
                budget -= tokens
                residents += 1
                actions.append(
                    ReferenceAction(seq.seq_id, ACTION_ADMISSION, tokens, charge)
                )

        return ReferenceProposal(
            decision_time=now,
            running_order=running_order,
            arrived_waiting_order=tuple(s.seq_id for s in arrived),
            free_blocks=free_before,
            watermark_blocks=watermark,
            token_budget=config.max_num_batched_tokens,
            victims=tuple(victims),
            actions=tuple(actions),
            running_stop_reason=running_stop,
            waiting_stop_reason=waiting_stop,
        )

    def _validate_state(self, now: float):
        """Reject unsupported or malformed state; return the arrived waiting
        requests in waiting order."""
        manager = self.block_manager
        max_model_len = self.scheduler_config.max_model_len
        seen = set()
        for seq in self.running + self.waiting:
            _require(
                seq.seq_id not in seen, CATEGORY_MALFORMED_STATE,
                f"seq_id {seq.seq_id} appears more than once in running/waiting",
            )
            seen.add(seq.seq_id)

        for seq in self.running:
            sid = seq.seq_id
            _require(
                seq.get_status() == SequenceStatus.PAUSED,
                CATEGORY_UNSUPPORTED_STATE,
                f"resident {sid} has status {seq.get_status().name}; only "
                "PAUSED residents are supported between synchronous steps",
            )
            _require(
                manager.is_allocated(seq), CATEGORY_MALFORMED_STATE,
                f"resident {sid} has no block table",
            )
            gap = self._block_gap(seq)
            processed = seq.get_num_prompt_tokens_processed()
            if seq.prompt_processing_finished:
                _require(
                    processed == seq.get_prompt_len(), CATEGORY_MALFORMED_STATE,
                    f"decode resident {sid} has processed {processed} of "
                    f"{seq.get_prompt_len()} prompt tokens",
                )
                _require(
                    gap in (0, 1), CATEGORY_UNSUPPORTED_STATE,
                    f"decode resident {sid} has block gap {gap}; only 0 or 1 "
                    "is supported",
                )
                _require(
                    seq.get_len() < max_model_len, CATEGORY_UNSUPPORTED_STATE,
                    f"decode resident {sid} has context {seq.get_len()} at "
                    f"max_model_len {max_model_len}; upstream would not "
                    "schedule it and LPServe has no max_model_len stop",
                )
            else:
                _require(
                    0 <= processed < seq.get_prompt_len(),
                    CATEGORY_MALFORMED_STATE,
                    f"partial-prefill resident {sid} has processed "
                    f"{processed} of {seq.get_prompt_len()} prompt tokens",
                )
                _require(
                    gap == 0, CATEGORY_UNSUPPORTED_STATE,
                    f"partial-prefill resident {sid} has block gap {gap}; "
                    "its full context must already be allocated",
                )

        arrived = []
        for seq in self.waiting:
            sid = seq.seq_id
            arrival = seq.arrival_time
            _require(
                isinstance(arrival, (int, float)) and not isinstance(arrival, bool)
                and math.isfinite(arrival),
                CATEGORY_MALFORMED_STATE,
                f"waiting {sid} has malformed arrival time {arrival!r}",
            )
            _require(
                seq.get_status() == SequenceStatus.WAITING
                and not manager.is_allocated(seq)
                and seq.get_num_prompt_tokens_processed() == 0
                and not seq.prompt_processing_finished,
                CATEGORY_MALFORMED_STATE,
                f"waiting {sid} is not an unallocated WAITING request with no "
                "prompt progress",
            )
            if arrival > now:
                continue
            # An empty prompt would become a zero-token admission, which the
            # metadata encodes as a decode; no empty-prompt policy exists.
            _require(
                seq.get_prompt_len() > 0, CATEGORY_UNSUPPORTED_STATE,
                f"arrived waiting {sid} has an empty prompt",
            )
            _require(
                seq.get_len() <= max_model_len, CATEGORY_UNSUPPORTED_STATE,
                f"arrived waiting {sid} has context {seq.get_len()} above "
                f"max_model_len {max_model_len}",
            )
            arrived.append(seq)

        # Physical coherence: every block is either free or in exactly one
        # resident's table, and only residents own tables.
        running_ids = {s.seq_id for s in self.running}
        _require(
            set(manager.block_tables) == running_ids, CATEGORY_MALFORMED_STATE,
            f"block tables {sorted(manager.block_tables)} do not match "
            f"residents {sorted(running_ids)}",
        )
        numbers = [b.block_number for t in manager.block_tables.values() for b in t]
        numbers += [b.block_number for b in manager.gpu_allocator.free_blocks]
        _require(
            sorted(numbers) == list(range(manager.num_total_gpu_blocks)),
            CATEGORY_MALFORMED_STATE,
            "allocated and free blocks are not a partition of the block pool",
        )
        return arrived

    def _block_gap(self, seq) -> int:
        return len(seq.logical_token_blocks) - len(
            self.block_manager.block_tables[seq.seq_id]
        )

    @staticmethod
    def _remaining_prompt(seq) -> int:
        return seq.get_prompt_len() - seq.get_num_prompt_tokens_processed()

    # ------------------------------------------------------------------
    # Layout validation and native execution
    # ------------------------------------------------------------------

    def _commit(self, proposal: ReferenceProposal) -> SchedulerOutputs:
        manager = self.block_manager
        by_id = {s.seq_id: s for s in self.running + self.waiting}
        ordered = proposal.execution_order

        # Layout validation before mutation: victims are residents disjoint
        # from the selected work, and every allocation and the worker's
        # admission gate hold at their position in commit order.
        running_ids = [s.seq_id for s in self.running]
        selected = [a.seq_id for a in proposal.actions]
        _require(
            len(set(selected)) == len(selected)
            and not set(selected) & set(proposal.victims)
            and set(proposal.victims) <= set(running_ids),
            CATEGORY_LAYOUT,
            f"victims {list(proposal.victims)} and actions {selected} are not "
            "disjoint residents and unique actions",
        )
        for action in proposal.actions:
            # Prefill metadata must be positive (zero encodes a decode) and
            # within the current remaining prompt.
            if action.is_prefill:
                remaining = self._remaining_prompt(by_id[action.seq_id])
                valid = 0 < action.num_tokens <= remaining
            else:
                valid = action.num_tokens == 1
            _require(
                valid, CATEGORY_LAYOUT,
                f"{action.kind} {action.seq_id} has {action.num_tokens} "
                "tokens; a prefill needs 1 to its remaining prompt and a "
                "decode exactly 1",
            )
        tokens = sum(a.num_tokens for a in proposal.actions)
        _require(
            tokens <= proposal.token_budget, CATEGORY_LAYOUT,
            f"selected tokens {tokens} exceed budget {proposal.token_budget}",
        )
        admissions = sum(a.kind == ACTION_ADMISSION for a in proposal.actions)
        residents = len(running_ids) - len(proposal.victims) + admissions
        _require(
            residents <= self.scheduler_config.max_num_seqs, CATEGORY_LAYOUT,
            f"resident count {residents} would exceed max_num_seqs",
        )
        free = manager.get_num_free_gpu_blocks() + sum(
            len(manager.block_tables[v]) for v in proposal.victims
        )
        for action in ordered:
            if action.kind == ACTION_ADMISSION:
                _require(
                    free - action.block_demand >= manager.watermark_blocks,
                    CATEGORY_LAYOUT,
                    f"admission {action.seq_id} fails the admission gate in "
                    "commit order",
                )
            _require(
                action.block_demand <= free, CATEGORY_LAYOUT,
                f"action {action.seq_id} demands {action.block_demand} blocks "
                f"with {free} free in commit order",
            )
            free -= action.block_demand

        # Mutation. Exceptions from here on propagate unchanged.
        preempted = []
        for seq_id in proposal.victims:
            seq = by_id[seq_id]
            self.running.remove(seq)
            self._preempt(seq)
            preempted.append(seq_id)
        metadata = []
        for action in ordered:
            seq = by_id[action.seq_id]
            if action.kind == ACTION_ADMISSION:
                self.waiting.remove(seq)
                self._allocate(seq)
                self.running.append(seq)
            elif action.kind == ACTION_DECODE:
                self._append_slot(seq)
            metadata.append(SequenceScheduleMetadata(
                seq.seq_id,
                prompt_chunk_len=action.num_tokens if action.is_prefill else 0,
            ))

        if manager.get_num_free_gpu_blocks() != free:
            raise RuntimeError(
                f"free blocks {manager.get_num_free_gpu_blocks()} after "
                f"execution differ from the validated {free}"
            )
        return SchedulerOutputs(
            id=self._iteration_id,
            ignored_seq_ids=[],
            preempted_seq_ids=preempted,
            scheduled_seq_metadata_list=metadata,
        )
