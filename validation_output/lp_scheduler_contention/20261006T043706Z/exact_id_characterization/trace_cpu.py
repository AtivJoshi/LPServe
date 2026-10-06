import sys; sys.path.insert(0, "tests")
import test_lp_scheduler as t
t.setUpModule()
h = t._Harness(max_num_seqs=3, b_max=8, c_max=2, s_max=2)
seqs = {i: h.add(i, 4, arrival_time=1.0 + i, max_tokens=4) for i in range(3)}
bm, wbm = h.scheduler.block_manager, h.worker_seqs.block_manager
step = 0
while h.scheduler.has_unfinished_seqs():
    out = h.schedule(10.0 + step); step += 1
    print(step, t._emitted(out), "sched central", h.central_tables())
    h.replay(out)
    print("   after central", h.central_tables(), "worker", h.worker_tables())
    print("   free central", [b.block_number for b in bm.gpu_allocator.free_blocks],
          "worker", [b.block_number for b in wbm.gpu_allocator.free_blocks])
