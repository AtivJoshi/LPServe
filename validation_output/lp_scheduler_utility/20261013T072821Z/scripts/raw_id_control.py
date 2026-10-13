"""Negative control (evidence only): replace the policy's compact queue
position with the raw-ID position and confirm that the reversed-ID
first-choice check then fails. Shows the check discriminates captured
queue state from raw request IDs. Nothing in the repository is changed."""
import sys
from unittest import mock

sys.path.insert(0, "tests")
import lpserve_state_mapping as lsm  # noqa: E402
import test_lp_scheduler as t  # noqa: E402


def raw_id_utility(captured, request, *, decode_scale, prefill_scale,
                   penalty_scale):
    position = sorted(r.raw_seq_id for r in captured.requests).index(
        request.raw_seq_id)
    return lsm.RequestUtility(decode_scale / (1 + position),
                              prefill_scale / (1 + position), 0.0)


t.setUpModule()
case = t.LiveRankBasedUtilityTest("test_reversed_ids_keep_first_choice")
for label, policy in (("compact queue position (shipped)", None),
                      ("raw-ID position (control)", raw_id_utility)):
    patch = (mock.patch.object(lsm, "rank_based_utility", policy)
             if policy else mock.patch.object(lsm, "rank_based_utility",
                                              lsm.rank_based_utility))
    with patch:
        harness, _ = case._harness(True, (1, 0))
        outputs = harness.step(10.0)
        emitted = [(m.seq_id, m.prompt_chunk_len)
                   for m in outputs.scheduled_seq_metadata_list]
    names = {1: "A", 0: "B"}
    print(f"{label}: first decision emitted "
          f"{[(names[k], c) for k, c in emitted]}; "
          f"first-choice check {'passes' if emitted == [(1, 5)] else 'FAILS'}")
