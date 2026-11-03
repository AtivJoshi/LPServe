"""Derive the reported tables from the retained run artifacts (read-only)."""
import hashlib, json, statistics, sys
from pathlib import Path
E = Path(sys.argv[1])
ms = lambda v: "-" if v is None else f"{v*1000:.1f}"
frozen = {l.split()[1]: l.split()[0] for l in (E / "provenance/sha256.txt").read_text().splitlines()}
for run in sorted(p.parent.name for p in E.glob("*/summary.json")):
    s = json.load(open(E / run / "summary.json"))
    fh = s["provenance"]["file_sha256"]
    same = all(fh[f] == frozen[f] for f in ("lpserve_state_mapping.py", "scripts/soft_slo_pilot.py", "tests/test_soft_slo_pilot.py"))
    print(f"{run}: validation_passed={s['validation_passed']} code_matches_frozen={same} head={s['provenance']['git_head'][:7]} settings_sha={s['settings']['sha256'][:12]} gates={s.get('gates')}")
s = json.load(open(E / "gpu_ample/summary.json"))
wi = s["worker_identity"]
print("\nGPU identity:", wi["device_name"], wi["device"], s["environment"]["gpu"], "load_format", wi["load_format"], wi["dtype"], "params", wi["num_parameters"])
print("weights match file:", {k: v["file_head"][:2] == v["worker_head"][:2] for k, v in s["weight_check"].items()}, "cache", s["cache_capacity"]["profiled_available_blocks"], "->", s["cache_capacity"]["chosen_initialized_blocks"], "fit", s["cohort_fit"], "init s", round(s["engine_init_seconds"], 1))
print("\nGPU ample metrics (ms)")
print(f"{'trial':16} {'class':8} TTFT obs  med   miss  mexcess | TBT gaps miss mexc  p99  late-share noGap overdueOpen | afterCutoff")
for lab, m in s["metrics"].items():
    for cls, v in m["by_class"].items():
        t, b = v["ttft"], v["tbt"]
        print(f"{lab:16} {cls:8} {t['observed']}/{t['cohort']} {ms(t['median']):>7} {t['miss_rate']!s:>5} {ms(t['mean_excess_observed']):>6} | {b['observed_gaps']:>3} {b['miss_rate']:.2f} {ms(b['mean_excess']):>4} {ms(b['p99']):>5} {b['share_streams_with_late_gap']:.2f} {b['streams_without_observed_gap']} {len(b['overdue_open_gaps'])} | {len(m['events_after_cutoff'])}")
print("per-request TTFT ms:")
for lab, m in s["metrics"].items():
    print(" ", lab, {r["name"]: ms(r.get("ttft")) for r in m["per_request"]})
print("\nDiagnostics:")
for lab, a in s["analyses"].items():
    fe = a["forecast_errors"]; errs = [r["error_seconds"] for r in fe["next_token"]]; be = fe["batch_error_seconds"]
    print(f" {lab:16} decisions={len(a['decisions'])} contended={sum(d['contention']['contended'] for d in a['decisions'])} diffs={sum(d['action_difference'] for d in a['decisions'])} boost={a['boost_regions']} next-token err ms mean {statistics.fmean(errs)*1000:.1f} [{min(errs)*1000:.1f}, {max(errs)*1000:.1f}] batch ms mean {fe['mean_batch_seconds']*1000:.2f} err max {max(be)*1000:.1f}")
rows = [(r["decision"], p, r[p]["comparison"]) for r in s["same_state_exact"] for p in ("timed", "untimed")]
print(f"\nSame-state PH1 vs exact ({len(rows)} solves on trial0_timed snapshots): statuses {sorted({(c['lp_status'], c['integer_status']) for _,_,c in rows})}; action mismatches {sum(not c['same_actions'] for _,_,c in rows)}; max |extraction_loss| {max(abs(c['extraction_loss']) for _,_,c in rows):.2e}; max relaxation_gap {max(c['relaxation_gap'] for _,_,c in rows):.2e}")
print("flip example:", {p: (v["full_prefill_reward"], v["ph1_actions"], v["exact_actions"]) for p, v in s["flip_example"].items()})
print("\nPressure witness (GPU):")
for case, lab in (("gpu_pressure", "trial0_timed"), ("gpu_pressure", "trial1_untimed"), ("gpu_pressure", "trial2_untimed"), ("gpu_pressure", "trial3_timed"), ("gpu_pressure_control", "single_timed")):
    t = json.load(open(E / case / "trials" / f"{lab}.json"))
    a = json.load(open(E / case / "summary.json"))["analyses"][lab]
    names = {c["seq_id"]: c["name"] for c in t["cohort"]}
    toks, times = {}, {}
    for e in t["events"]:
        toks.setdefault(names[e["seq_id"]], []).append(e["token_id"]); times.setdefault(names[e["seq_id"]], []).append(e["time"])
    pre = [(st["index"], names[p]) for st in t["steps"] for p in st["preempted"]]
    gaps = {n: [round((b - x) * 1000, 1) for x, b in zip(v, v[1:])] for n, v in times.items()}
    print(f" {case}/{lab}: preempted {pre} tokens {toks} gaps_ms {gaps} work " + str({k: (w['repeated_prefill_tokens'], w['extra_outputs']) for k, w in a['work'].items()}))
    for st in t["steps"]:
        if pre and st["index"] > pre[0][0] and not st["events"]:
            vid = [k for k, v in names.items() if v == pre[0][1]][0]
            print(f"   rebuild d{st['index']} scheduled {st['scheduled']} last_visible-t0 {st['timing_inputs'][0][str(vid)]['last_visible'] - t['t0']:.4f}")
