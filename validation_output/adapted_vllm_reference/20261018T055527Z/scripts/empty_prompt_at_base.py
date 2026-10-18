"""Run the new empty-prompt regression against the scheduler source at the
base revision (read with `git show`, not checked out). Expected: the test
FAILS there, showing it detects the defect. Exit status 0 means the test
failed as expected."""
import importlib.util, subprocess, sys, tempfile, unittest
from pathlib import Path
from unittest import mock

BASE = sys.argv[1]
root = Path.cwd()
sys.path[:0] = [str(root), str(root / "tests")]
src = subprocess.run(["git", "show", f"{BASE}:sarathi/core/scheduler/adapted_vllm_scheduler.py"],
                     check=True, capture_output=True, text=True).stdout
path = Path(tempfile.mkdtemp()) / "adapted_vllm_scheduler_base.py"
path.write_text(src)
spec = importlib.util.spec_from_file_location("adapted_vllm_scheduler_base", path)
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)

import test_adapted_vllm_scheduler as t
from sarathi.config import SchedulerType
from sarathi.core.scheduler.scheduler_registry import SchedulerRegistry

t.setUpModule()
with mock.patch.dict(SchedulerRegistry._registry,
                     {SchedulerType.ADAPTED_VLLM: base.AdaptedVLLMScheduler}), \
        mock.patch.object(t, "avs", base):
    h = t._Harness(num_gpu_blocks=4, max_num_seqs=2, max_num_batched_tokens=8)
    h.add(0, 0, 1.0, 1)
    out = h.schedule(10.0)
    print("base proposal actions:", h.proposals[0].actions)
    print("base emitted:", [(m.seq_id, m.prompt_chunk_len) for m in out.scheduled_seq_metadata_list],
          "native calls:", h.calls, "running:", h.ids(h.scheduler.running),
          "num_running_batches:", h.scheduler.num_running_batches)
    suite = unittest.TestSuite([t.PreMutationFailureTest(
        "test_empty_prompt_rejected_before_native_mutation")])
    result = unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(suite)
print("regression failed at base as expected:", not result.wasSuccessful())
sys.exit(0 if not result.wasSuccessful() else 1)
