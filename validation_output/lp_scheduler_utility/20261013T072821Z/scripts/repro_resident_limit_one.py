"""Critical-review reproduction (evidence only, not a repository test): the
same two requests with resident limit 1 instead of 2, everything else equal,
bounded to 8 decisions. Records whether any output token is ever produced
under uniform 1/1/1 and rank-based D=P=G=1."""
import functools
import sys

sys.path.insert(0, "tests")
import lpserve_state_mapping as lsm  # noqa: E402
import test_lp_scheduler as t  # noqa: E402

t.setUpModule()
config = dict(t.LiveRankBasedUtilityTest.CONFIG, max_num_seqs=1)
for rank_based in (False, True):
    harness = t._Harness(num_gpu_blocks=8, **config)
    if rank_based:
        policy = functools.partial(lsm.rank_based_utility, decode_scale=1.0,
                                   prefill_scale=1.0, penalty_scale=1.0)
        harness.scheduler._coefficient_policy = lambda: policy
    seqs = {"A": harness.add(0, 5, 1.0, 2), "B": harness.add(1, 8, 2.0, 2)}
    names = {0: "A", 1: "B"}
    print(f"== {'rank-based D=P=G=1' if rank_based else 'uniform 1/1/1'}, "
          f"resident limit 1")
    for step in range(8):
        sched = harness.scheduler
        running = [names[s.seq_id] for s in sched.running]
        waiting = [names[s.seq_id] for s in sched.waiting]
        outputs = harness.step(10.0 + step)
        print(f"  step {step}: running={running} waiting={waiting} "
              f"preempted={[names[k] for k in outputs.preempted_seq_ids]} "
              f"scheduled={[(names[m.seq_id], m.prompt_chunk_len) for m in outputs.scheduled_seq_metadata_list]} "
              f"outputs A={seqs['A'].get_output_len()} B={seqs['B'].get_output_len()}")
    print(f"  after 8 decisions: total output tokens = "
          f"{sum(s.get_output_len() for s in seqs.values())}")
