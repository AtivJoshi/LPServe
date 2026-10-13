"""Reproduce review findings 2 and 3 on the starting code (CPU only)."""
import sys, tempfile
sys.path.insert(0, "tests")
import lp_relaxation_scheduler as lrs
from sarathi.engine.arg_utils import EngineArgs

out = tempfile.mkdtemp()
pol = lrs.NumericalPolicy()
base = dict(model="unused", scheduler_type="lp", max_num_seqs=4, max_model_len=32,
            output_dir=out, lp_b_max=8, lp_c_max=4, lp_s_max=3, lp_decode_utility=1,
            lp_prefill_token_utility=1, lp_preemption_penalty=1, lp_numerical_policy=pol)
for name in ("lp_memory_reserve", "lp_decode_memory_policy_id"):
    try:
        EngineArgs(**base, **{name: None})
        print(f"finding2: explicit {name}=None ACCEPTED (silently treated as omission)")
    except Exception as e:
        print(f"finding2: explicit {name}=None rejected: {type(e).__name__}: {e}")

import test_lp_scheduler as t
t.setUpModule()
h = t._Harness()
h.add(0, 4, arrival_time=10 ** 400, max_tokens=2)
try:
    h.schedule(5.0)
    print("finding3: no exception")
except Exception as e:
    print(f"finding3: schedule raised {type(e).__name__}: {e}")
