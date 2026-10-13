"""Finding 1 on the starting code: the HEAD preemption helper's
check_decision rejects correct corrected-order records (B before A)."""
import sys
from unittest import mock
old_dir, repo = sys.argv[1], sys.argv[2]
sys.path[:0] = [old_dir, repo, repo + "/tests"]
import check_lp_scheduler_preemption_gpu as old
assert old.__file__.startswith(old_dir), old.__file__
import test_lpserve_plan_execution as fx
import lpserve_plan_execution as lpe
from check_lp_scheduler_gpu import to_jsonable, outputs_record
class t:  # local stand-ins for the serialization helpers
    to_jsonable = staticmethod(to_jsonable)
holder = fx._SchedulerHolder(num_gpu_blocks=3)
fx._add_resident(holder, 0, 4, processed=4, num_output_tokens=1)
fx._add_resident(holder, 1, 4, processed=4)
snapshot, result = fx._map_and_solve(
    holder, fx._utilities(r0=(1.0, 0.0, 1.0), r1=(1.0, 0.0, 1.0)))
it, free = holder._iteration_id, holder.block_manager.get_num_free_gpu_blocks()
outputs = lpe.execute_plan(holder, snapshot, result)
snap, plan, emitted = to_jsonable(snapshot), to_jsonable(result.plan), outputs_record(outputs)
events = [dict(call="map_scheduler_state", result_type="StateSnapshot", result=snap,
               num_running_batches=0, iteration_id=it),
          dict(call="solve_and_extract", result_type="SchedulingSuccess",
               result=t.to_jsonable(result)),
          dict(call="execute_plan", result_type="SchedulerOutputs", result=emitted),
          dict(call="schedule", outputs=emitted),
          dict(call="_run_workers", method="execute_model",
               sampler_outputs=[[s, 7] for s, _ in emitted["scheduled"]])]
before = dict(iteration_id=it - 1, free_blocks=free, waiting=[], running=[0, 1])
for k, v in dict(MAX_NUM_SEQS=4, B_MAX=8, C_MAX=4, S_MAX=3, DECODE_UTILITY=1.0,
                 PREFILL_TOKEN_UTILITY=0.0, PREEMPTION_PENALTY=1.0).items():
    setattr(old, k, v)
print("finding1: emitted (correct corrected order):", emitted["scheduled"])
try:
    old.check_decision("old", dict(preempted=[], scheduled=emitted["scheduled"]),
                       before, events, t.to_jsonable(fx.NUMERICAL_POLICY))
    print("finding1: HEAD helper accepted the correct records")
except Exception as e:
    print(f"finding1: HEAD helper REJECTED correct records: {type(e).__name__}: {e}")
