"""Evidence extraction only: run the committed test's own replay
(LiveRankBasedUtilityTest._run_case) for both policies and both ID
assignments, and write its primitive trace. Run from the repository root."""
import json
import sys

sys.path.insert(0, "tests")
import test_lp_scheduler as t  # noqa: E402

out = sys.argv[1]
t.setUpModule()
case = t.LiveRankBasedUtilityTest(
    "test_uniform_and_rank_based_first_choice_and_completion")
traces = {}
for ids in ((0, 1), (1, 0)):
    for rank_based in (False, True):
        trace = case._run_case(rank_based, ids=ids)
        traces[f"{trace['policy']}_ids_A{ids[0]}_B{ids[1]}"] = trace
with open(f"{out}/trace.json", "w") as f:
    json.dump(traces, f, indent=1)

lines = ["Compact comparison (scheduling steps are decision indices, not "
         "wall-clock latency)", ""]
fmt = "{:<26} {:>5} {:>15} {:>15} {:>15} {:>5} {:>10} {:>6}"
lines.append(fmt.format("run", "steps", "first prefill", "first decode",
                        "completion", "pre", "unfinished", "errors"))
for key, trace in traces.items():
    s = trace["summary"]
    ab = lambda d: f"A{d['A']} B{d['B']}"
    lines.append(fmt.format(key, s["steps"], ab(s["first_prefill"]),
                            ab(s["first_decode"]), ab(s["completion"]),
                            s["preemptions"], s["unfinished"], s["errors"]))
lines += ["", "Per-decision choices (running|waiting before -> emitted; "
          "alpha/beta/gamma; relaxed objective):"]
for key, trace in traces.items():
    lines.append(f"[{key}] watermark_blocks={trace['watermark_blocks']}")
    for st in trace["steps"]:
        coeffs = " ".join(
            f"{n}=({r['alpha']:.3g},{r['beta']:.3g},{r['gamma']:.3g})"
            for n, r in sorted(st["requests"].items()))
        lines.append(
            f"  step {st['step']}: {','.join(st['running_before']) or '-'}|"
            f"{','.join(st['waiting_before']) or '-'} -> "
            f"{st['emitted'][0][0]}:{st['emitted'][0][1]}  {coeffs}  "
            f"relaxed={st['relaxed_objective']:.6g} "
            f"frac={st['fractional_request_count']} "
            f"free c/w={st['central_free_after']}/{st['worker_free_after']}"
            + (f" finished={st['finished']}" if st["finished"] else ""))
with open(f"{out}/comparison.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
