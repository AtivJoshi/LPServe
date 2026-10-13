"""Live LP-relaxation scheduler.

``LPScheduler`` connects current scheduler state to the accepted LP layers in
one synchronous decision (docs/lp_scheduler_design.md sections 12.6, 13, and
14):

    current state -> map_scheduler_state -> solve_and_extract
                  -> execute_plan -> SchedulerOutputs

It is a thin coordinator. Ownership stays in the inherited ``waiting`` and
``running`` collections. The mapper captures live state once and builds ranks,
coefficients, and the problem from that capture; output order, precommit
validation, and native mutation belong to the executor.

Ordinary idle means no arrived unfinished request in ``running`` or
``waiting``; a future waiting head does not hide arrived work behind it.

``check_new_request`` is the LP-only initial admission check that the engine
calls before registering a new request; it raises ``LPRequestError``.

Failure behavior:

- An unsupported entry state (more than one pipeline stage, or a nonzero
  running-batch count) raises ``LPSchedulingError`` before inherited
  bookkeeping, so ``_iteration_id`` is unchanged.
- A failure returned by mapping (including rank and coefficient
  construction), solving/extraction, or executor precommit validation raises ``LPSchedulingError`` carrying that
  immutable failure record. No later stage runs and no ``SchedulerOutputs``
  is produced. Inherited bookkeeping has already advanced ``_iteration_id``
  once; no execution mutation has occurred.
- An exception raised after executor mutation begins propagates unchanged. It
  is not caught, retried, translated, or recovered; the caller must end the
  run and discard the affected state.

The caller must honor the synchronous, non-overlapping engine contract; this
class adds no locking or concurrent-call support.
"""

import math
import time

from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.scheduler.base_scheduler import BaseScheduler

STAGE_SCHEDULER_ENTRY = "scheduler_entry"
CATEGORY_UNSUPPORTED_STATE = "unsupported_state"

# The LP layer modules live at the repository root and are imported inside the
# methods that use them, so registering this scheduler does not make every
# scheduler-registry import depend on the repository root being importable.


class LPSchedulingError(RuntimeError):
    """A pre-mutation LP scheduling failure.

    ``failure`` is the immutable record returned by the failing stage: a
    ``lpserve_state_mapping.MappingFailure`` or a
    ``lp_relaxation_scheduler.Failure``. No live scheduler, sequence, or block
    manager object is retained.
    """

    def __init__(self, failure) -> None:
        self.failure = failure
        super().__init__(
            f"LP scheduling failed before execution mutation "
            f"(snapshot {self.snapshot_id!r}, stage {failure.stage!r}, "
            f"category {failure.category!r}): {failure.reason}"
        )

    @property
    def snapshot_id(self):
        # MappingFailure names it snapshot_id; Failure names it problem_id.
        return getattr(
            self.failure, "snapshot_id", getattr(self.failure, "problem_id", None)
        )

    @property
    def stage(self) -> str:
        return self.failure.stage

    @property
    def category(self) -> str:
        return self.failure.category

    @property
    def reason(self) -> str:
        return self.failure.reason

    @property
    def solver_diagnostics(self):
        solver = getattr(self.failure, "solver", None)
        return solver.diagnostics if solver is not None else None


class LPRequestError(ValueError):
    """A new request the LP scheduler can never admit.

    Raised before the request is registered anywhere. ``seq_id`` is the
    request identity, ``required`` the needed length or blocks, and ``limit``
    the configured limit.
    """

    def __init__(self, seq_id, reason, required, limit) -> None:
        self.seq_id = seq_id
        self.reason = reason
        self.required = required
        self.limit = limit
        super().__init__(
            f"LP scheduler rejected request seq_id={seq_id!r}: {reason} "
            f"(required {required}, limit {limit})"
        )


class LPScheduler(BaseScheduler):
    """Constructed from an ``LPSchedulerConfig`` through ``SchedulerRegistry``."""

    def schedule(self) -> SchedulerOutputs:
        # BaseScheduler.schedule() returns an ordinary empty output when its
        # running-batch limit is reached; reject that state visibly instead.
        import lp_relaxation_scheduler as lrs

        num_pipeline_stages = self.scheduler_config.num_pipeline_stages
        if num_pipeline_stages != 1:
            raise LPSchedulingError(lrs.Failure(
                None, STAGE_SCHEDULER_ENTRY, lrs.OUTCOME_INVALID,
                CATEGORY_UNSUPPORTED_STATE,
                f"unsupported pipeline stage count {num_pipeline_stages!r}; "
                "only 1 is supported",
            ))
        if self.num_running_batches != 0:
            raise LPSchedulingError(lrs.Failure(
                None, STAGE_SCHEDULER_ENTRY, lrs.OUTCOME_INVALID,
                CATEGORY_UNSUPPORTED_STATE,
                "unsupported in-flight state: num_running_batches="
                f"{self.num_running_batches!r}; only 0 is supported",
            ))
        return super().schedule()

    def check_new_request(self, seq) -> None:
        """Reject a new request that can never be admitted.

        Called after ``Sequence`` construction and before any registration.
        Equality at either limit is accepted. Current free-memory contention
        is not checked, and no future output is reserved.
        """
        context_len = seq.get_len()
        max_model_len = self.scheduler_config.max_model_len
        if context_len > max_model_len:
            raise LPRequestError(
                seq.seq_id, "context length exceeds max_model_len",
                f"{context_len} tokens", f"{max_model_len} tokens",
            )
        manager = self.block_manager
        demand = manager.get_num_initial_blocks(seq)
        total = manager.num_total_gpu_blocks
        watermark = manager.watermark_blocks
        if demand + watermark > total:
            raise LPRequestError(
                seq.seq_id,
                "full initial allocation plus the shared watermark "
                f"({watermark} blocks) exceeds the total block pool",
                f"{demand + watermark} blocks", f"{total} blocks",
            )

    def _schedule(self) -> SchedulerOutputs:
        import lp_relaxation_scheduler as lrs
        import lpserve_plan_execution as lpe
        import lpserve_state_mapping as lsm

        now = time.monotonic()

        # Ordinary idle (D-18 as amended): no arrived unfinished request.
        if self._is_idle(now):
            return SchedulerOutputs(
                self._iteration_id,
                ignored_seq_ids=[],
                preempted_seq_ids=[],
                scheduled_seq_metadata_list=[],
            )

        config = self.scheduler_config
        snapshot = lsm.map_scheduler_state(
            self,
            snapshot_time=now,
            b_max=config.b_max,
            c_max=config.c_max,
            s_max=config.s_max,
            coefficient_policy=self._coefficient_policy(),
            rank_profile=lsm.vllm_oriented_ranks,
            numerical_policy=config.numerical_policy,
        )
        if isinstance(snapshot, lsm.MappingFailure):
            raise LPSchedulingError(snapshot)

        result = lrs.solve_and_extract(snapshot.lp_problem)
        if isinstance(result, lrs.Failure):
            raise LPSchedulingError(result)

        # Exceptions after the executor's first mutation propagate unchanged.
        outputs = lpe.execute_plan(self, snapshot, result)
        if isinstance(outputs, lrs.Failure):
            raise LPSchedulingError(outputs)
        return outputs

    def _is_idle(self, now) -> bool:
        """True only when no arrived unfinished request exists. A request with
        a malformed arrival time is not idle; the mapper rejects it."""
        if self.running:
            return False
        for seq in self.waiting:
            if seq.is_finished():
                continue
            arrival = seq.arrival_time
            try:
                future = (
                    isinstance(arrival, (int, float))
                    and not isinstance(arrival, bool)
                    and math.isfinite(arrival) and arrival > now
                )
            except OverflowError:  # an int too large for a float
                future = False
            if not future:
                return False
        return True

    def _coefficient_policy(self):
        """The explicit coefficient policy passed to the mapper: the
        configured uniform triple. A validation harness may override this
        method to supply another explicit policy accepted by
        ``lpserve_state_mapping.construct_coefficients``."""
        import lpserve_state_mapping as lsm

        config = self.scheduler_config
        return lsm.RequestUtility(
            config.decode_utility,
            config.prefill_token_utility,
            config.preemption_penalty,
        )
