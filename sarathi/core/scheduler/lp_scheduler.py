"""Live LP-relaxation scheduler.

``LPScheduler`` connects current scheduler state to the accepted LP layers in
one synchronous decision (docs/lp_scheduler_design.md sections 12.6, 13, and
14):

    current state -> map_scheduler_state -> solve_and_extract
                  -> execute_plan -> SchedulerOutputs

It is a thin coordinator. Ownership stays in the inherited ``waiting`` and
``running`` collections; prompt-first output order, precommit validation, and
native mutation belong to the executor.

Failure behavior:

- An unsupported entry state (more than one pipeline stage, or a nonzero
  running-batch count) raises ``LPSchedulingError`` before inherited
  bookkeeping, so ``_iteration_id`` is unchanged.
- A failure returned by utility construction, mapping, solving/extraction, or
  executor precommit validation raises ``LPSchedulingError`` carrying that
  immutable failure record. No later stage runs and no ``SchedulerOutputs``
  is produced. Inherited bookkeeping has already advanced ``_iteration_id``
  once; no execution mutation has occurred.
- An exception raised after executor mutation begins propagates unchanged. It
  is not caught, retried, translated, or recovered; the caller must end the
  run and discard the affected state.

The caller must honor the synchronous, non-overlapping engine contract; this
class adds no locking or concurrent-call support.
"""

import time

from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.scheduler.base_scheduler import BaseScheduler

STAGE_SCHEDULER_ENTRY = "scheduler_entry"
STAGE_UTILITY_CONSTRUCTION = "utility_construction"
CATEGORY_UNSUPPORTED_STATE = "unsupported_state"
CATEGORY_MALFORMED_INPUT = "malformed_input"

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


class LPScheduler(BaseScheduler):
    """Constructed from an ``LPSchedulerConfig`` through ``SchedulerRegistry``."""

    def schedule(self) -> SchedulerOutputs:
        # BaseScheduler.schedule() returns an ordinary empty output when its
        # running-batch limit is reached; reject that state visibly instead.
        import lp_relaxation_scheduler as lrs

        num_pipeline_stages = self.scheduler_config.num_pipeline_stages
        if num_pipeline_stages != 1:
            raise LPSchedulingError(lrs.Failure(
                None, STAGE_SCHEDULER_ENTRY, CATEGORY_UNSUPPORTED_STATE,
                f"unsupported pipeline stage count {num_pipeline_stages!r}; "
                "only 1 is supported",
            ))
        if self.num_running_batches != 0:
            raise LPSchedulingError(lrs.Failure(
                None, STAGE_SCHEDULER_ENTRY, CATEGORY_UNSUPPORTED_STATE,
                "unsupported in-flight state: num_running_batches="
                f"{self.num_running_batches!r}; only 0 is supported",
            ))
        return super().schedule()

    def _schedule(self) -> SchedulerOutputs:
        import lp_relaxation_scheduler as lrs
        import lpserve_plan_execution as lpe
        import lpserve_state_mapping as lsm

        now = time.monotonic()

        # Ordinary idle (D-18): inherited head-of-queue arrival convention.
        if not self.running and (
            not self.waiting or self.waiting[0].arrival_time > now
        ):
            return SchedulerOutputs(
                self._iteration_id,
                ignored_seq_ids=[],
                preempted_seq_ids=[],
                scheduled_seq_metadata_list=[],
            )

        config = self.scheduler_config
        utilities = self._build_utilities(now)

        snapshot = lsm.map_scheduler_state(
            self,
            snapshot_time=now,
            b_max=config.b_max,
            c_max=config.c_max,
            s_max=config.s_max,
            memory_reserve=config.memory_reserve,
            decode_memory_policy_id=config.decode_memory_policy_id,
            utilities=utilities,
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

    def _build_utilities(self, now):
        """Return the uniform configured utility for every arrived, unfinished
        owned request, keyed by raw ``seq_id``. The mapper validates IDs,
        arrival times, ownership, and the exact key set."""
        import lp_relaxation_scheduler as lrs
        import lpserve_state_mapping as lsm

        config = self.scheduler_config
        utility = lsm.RequestUtility(
            config.decode_utility,
            config.prefill_token_utility,
            config.preemption_penalty,
        )
        try:
            return {
                seq.seq_id: utility
                for seq in self.waiting + self.running
                if seq.arrival_time <= now and not seq.is_finished()
            }
        except (AttributeError, TypeError) as err:
            raise LPSchedulingError(lrs.Failure(
                str(self._iteration_id), STAGE_UTILITY_CONSTRUCTION,
                CATEGORY_MALFORMED_INPUT,
                f"utility construction raised {type(err).__name__}: {err}",
            )) from None
