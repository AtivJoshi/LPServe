import sys; sys.path.insert(0, "tests")
import test_lp_scheduler as t
t.setUpModule()
h = t._Harness(max_num_seqs=3, b_max=8, c_max=2, s_max=2)
for i in range(3): h.add(i, 4, arrival_time=1.0 + i, max_tokens=4)
step, first = 0, None
while h.scheduler.has_unfinished_seqs():
    out = h.schedule(10.0 + step); step += 1; h.replay(out)
    c, w = h.central_tables(), h.worker_tables()
    if first is None and c != w: first = (step, c, w)
    assert {k: len(v) for k, v in c.items()} == {k: len(v) for k, v in w.items()}
print("diverged" if first else "equal", first)
