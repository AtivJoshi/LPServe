"""Evidence extraction only: per-request and per-decision summary of the six
GPU comparison runs from their retained summary.json and decision_trace.json
(decision indices, not wall-clock latency). Reads nothing else.

Run from the repository root: python -B <this script> <evidence dir>
"""
import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
RUNS = [f"{p}_resident{n}" for n in (2, 1)
        for p in ("adapted_vllm", "lp_uniform", "lp_rank_based")]
NAMES = {"0": "A", "1": "B"}

lines = ["GPU comparison (decision indices; dummy weights; not latency)", ""]
fmt = "{:<26} {:>6} {:>18} {:>4} {:>8} {:>8} {:>8} {:>7} {:>9} {:>9} {:>9}"
lines.append(fmt.format("run", "valid", "outcome", "dec", "1st pre", "1st dec",
                        "done", "events", "preempt", "recomp", "omitted"))
pair = lambda acc, k: "/".join("-" if acc[n][k] is None else str(acc[n][k])  # noqa: E731
                               for n in "AB")
rows = {}
for run in RUNS:
    s = json.loads((OUT / run / "summary.json").read_text())
    t = json.loads((OUT / run / "decision_trace.json").read_text())
    exit_status = (OUT / run / "exit_status.txt").read_text().split()[0]
    acc = (t.get("final") or {}).get("accounting")
    rows[run] = (s, t)
    if acc is None:
        lines.append(f"{run:<26} {str(s['validation_passed']):>6} "
                     f"{s['workload_outcome']:>18}  {exit_status}")
        continue
    recomputed = "/".join(str(max(0, acc[n]["prefill_tokens_executed"]
                                  - len(t["prompts"][n]))) for n in "AB")
    lines.append(fmt.format(
        run, str(s["validation_passed"]), s["workload_outcome"],
        len(t["decisions"]), pair(acc, "first_prefill"),
        pair(acc, "first_decode"), pair(acc, "completion"),
        pair(acc, "generated_token_events"), pair(acc, "preemptions"),
        recomputed, "/".join(str(len(acc[n]["omitted_decisions"])) for n in "AB")))
lines += ["", "Columns are A/B. recomp = prefill tokens executed beyond the "
          "original prompt (repeated computation); omitted = decisions in which "
          "an unfinished arrived request was neither served nor preempted.", ""]

lines.append("Capacity and arrivals")
for run, (s, t) in rows.items():
    cap = s.get("cache_capacity") or {}
    arr = t.get("arrival_times") or {}
    first = (t.get("decisions") or [{}])[0].get("decision_time")
    lines.append(f"  {run}: profiled {cap.get('profiled_available_blocks')} "
                 f"initialized {cap.get('chosen_initialized_blocks')}; arrivals "
                 f"A={arr.get('A')} B={arr.get('B')} first decision {first}")
lines.append("")


def state(b):
    run = ",".join(NAMES[str(s)] for s in b["running"]) or "-"
    wait = ",".join(NAMES[str(s)] for s in b["waiting"]) or "-"
    seqs = " ".join(
        f"{NAMES[k]}({v['prompt_tokens_processed']}/{v['prompt_len']},"
        f"o{len(v['output_token_ids'])})" for k, v in sorted(b["seqs"].items()))
    return f"R[{run}] W[{wait}] free={b['free_blocks']} {seqs}"


lines.append("Per decision: before-state -> policy record -> emitted (free "
             "central/worker after)")
for run, (s, t) in rows.items():
    lines.append(f"[{run}]")
    for d in t.get("decisions", []):
        head = f"  {d['index']}: {state(d['before'])}"
        if "schedules" not in d or not d["schedules"]:
            lines.append(head + " -> FAILURE (see summary.json)")
            continue
        out = d["schedules"][0]["outputs"]
        emitted = [(NAMES[str(a)], c) for a, c in out["scheduled"]]
        pre = [NAMES[str(v)] for v in out["preempted_seq_ids"]]
        if d["proposals"]:
            p = d["proposals"][0]
            sel = [(NAMES[str(a["seq_id"])], a["kind"][:3], a["num_tokens"])
                   for a in p["actions"]]
            policy = (f"select {sel} victims {pre} stop=({p['running_stop_reason']},"
                      f"{p['waiting_stop_reason']})")
        else:
            snap, res = d["lp_events"][0]["result"], d["lp_events"][1]["result"]
            coeff = " ".join(
                f"{NAMES[str(r['raw_seq_id'])]}=({r['utility']['decode_utility']:.3g},"
                f"{r['utility']['prefill_token_utility']:.3g},"
                f"{r['utility']['preemption_penalty']:.3g})"
                for r in snap["requests"])
            rel = " ".join(
                f"{NAMES[x['request_id']]}:x={x['x']:.3g},y={x['y']:.3g},"
                f"I={x['prefill_indicator']:.3g},z={x['z']:.3g}"
                for x in res["relaxed"]["decisions"])
            plan = res["plan"]
            ints = " ".join(f"{NAMES[x['request_id']]}:({x['prefill_tokens']},"
                            f"{x['decode']},{x['preempt']})" for x in plan["decisions"])
            diag = res["solver"]["diagnostics"]
            policy = (f"coeff(a,b,g) {coeff}; status {diag['raw_status']} "
                      f"it {diag['iterations']}; relaxed {rel} "
                      f"obj {res['relaxed']['normalized_objective']:.4g}; "
                      f"plan {ints} obj {plan['objective']:.4g} "
                      f"frac {plan['extraction']['fractional_request_count']}")
        a = d["after"]
        lines.append(head)
        lines.append(f"     {policy}")
        lines.append(f"     -> preempt {pre} emit {emitted} (free "
                     f"{a['free_blocks']}/{d['worker_after']['free_blocks']}) "
                     f"samples {d['sampler_outputs']}")
(OUT / "comparison.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
