"""Summarize the memory-pressure runs in this directory (evidence only).

    python3 <out>/scripts/summarize.py <out> > <out>/comparison.txt
"""
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
for policy in ("adapted_vllm", "lp_uniform", "lp_rank_based"):
    run = out / f"memory_pressure_{policy}"
    s = json.loads((run / "summary.json").read_text())
    t = json.loads((run / "decision_trace.json").read_text())
    print(f"=== {policy}: validation_passed={s['validation_passed']} "
          f"outcome={s['workload_outcome']} decisions={s['decisions']} "
          f"unfinished={s['unfinished']} exit={(run / 'exit_status.txt').read_text().strip()}")
    print("idx free running waiting | preempted scheduled | victim reasons "
          "(preemption-free alternative) | relaxed->integer objective, extraction changed | "
          "central==worker physical IDs")
    for d in t["decisions"]:
        e = d["schedules"][0]["outputs"]
        v = d["victim_need"]
        lp = d.get("lp_comparison")
        lps = ("-" if lp is None else
               f"{lp['relaxed_objective']:.3f}->{lp['integer_objective']:.3f} "
               f"{lp['extraction_changed_relaxed']}")
        print(f"{d['index']:>2} {d['before']['free_blocks']} {d['before']['running']} "
              f"{d['before']['waiting']} | {e['preempted_seq_ids']} {e['scheduled']} | "
              f"{v['reasons']} ({v['preemption_free_alternative']}) | {lps} | "
              f"{d['physical_block_ids_equal']}")
        print(f"   samples {d['sampler_outputs']}")
    final = t["final"]
    print("final free central/worker", final["central"]["free_blocks"],
          final["worker"]["free_blocks"], "residual allocations",
          final["residual_allocations"], "idle", t["idle"] is not None)
    keys = ("first_prefill", "first_decode", "completion", "generated_token_events",
            "native_output_count", "overgeneration", "preemptions",
            "preemptions_after_output", "discarded_prompt_tokens",
            "generated_tokens_moved_to_context", "prefill_actions",
            "prefill_tokens_executed", "prefill_tokens_first_time",
            "prefill_tokens_replayed", "prefill_tokens_expanded_context")
    for name, a in final["accounting"].items():
        print(f"  {name}: " + " ".join(f"{k}={a[k]}" for k in keys)
              + f" omitted={a['omitted_decisions']}")
    print()
