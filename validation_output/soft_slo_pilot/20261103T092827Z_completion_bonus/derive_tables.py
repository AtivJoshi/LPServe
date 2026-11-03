"""Derive the completion-bonus tables from the retained artifacts (read-only)."""
import json, sys
from pathlib import Path
E = Path(sys.argv[1])
ms = lambda v: "-" if v is None else f"{v*1000:.1f}"
frozen = {l.split()[1]: l.split()[0] for l in (E / "provenance/sha256.txt").read_text().splitlines()}
FILES = ("lp_relaxation_scheduler.py", "lpserve_state_mapping.py", "scripts/soft_slo_pilot.py", "tests/test_soft_slo_pilot.py")
for run in sorted(p.parent.name for p in E.glob("*/summary.json")):
    s = json.load(open(E / run / "summary.json"))
    fh = s["provenance"]["file_sha256"]
    print(f"{run}: passed={s['validation_passed']} code_matches_frozen={all(fh[f] == frozen[f] for f in FILES)} head={s['provenance']['git_head'][:7]} settings_sha={s['settings']['sha256'][:12]} gates={s.get('gates')} failure={s['failure'] and s['failure']['message']}")
g = json.load(open(E / "gpu_ample_bonus/summary.json"))
wi = g["worker_identity"]
print(f"\nGPU: {wi['device_name']} {g['environment']['gpu']} host={g['environment']['hostname']} job={g['environment']['slurm']['SLURM_JOB_ID']} load_format={wi['load_format']} {wi['dtype']} params={wi['num_parameters']} weights_match={[v['file_head'] == v['worker_head'] for v in g['weight_check'].values()]} cache={g['cache_capacity']['profiled_available_blocks']}->{g['cache_capacity']['chosen_initialized_blocks']} hf_commit={g['config']['hf_config_commit_hash']}")
for run in ("gpu_ample_bonus", "gpu_incumbents_bonus"):
    s = json.load(open(E / run / "summary.json"))
    print(f"\n=== {run} (order {s['registration_order']}, comparison {s['argv'][s['argv'].index('--comparison') + 1]})")
    for lab, m in s["metrics"].items():
        t = json.load(open(E / run / "trials" / f"{lab}.json"))
        a = s["analyses"][lab]
        names = {c["seq_id"]: c["name"] for c in t["cohort"]}
        t0 = t["t0"]
        pre = sum(len(st["preempted"]) for st in t["steps"])
        assoc = all(len(st["sampler"]) == len(st["scheduled"]) and [x[0] for x in st["sampler"]] == [x[0] for x in st["scheduled"]] for st in t["steps"])
        fin = all(r["finished_by_cutoff"] for r in m["per_request"])
        cont = sum(d["contention"]["contended"] for d in a["decisions"])
        earned = sum(bool(d["bonus_earned"]) for d in a["decisions"])
        diffs = sum(d["action_difference"] for d in a["decisions"])
        ov = s.get("incumbent_overlap", {}).get(lab)
        print(f" {lab:20} decisions={len(t['steps'])} finished_in_cutoff={fin} preempt={pre} sampler_assoc={assoc} contended={cont} bonus_earned_decisions={earned} cf_diffs={diffs}" + (f" overlap={ov['overlaps']} inc_tokens_during_main_first={ {k: v['tokens_during_main_first_tokens'] for k, v in ov['streams'].items()} }" if ov else ""))
        main = [r for r in m["per_request"] if r["name"] not in ("relaxed8a", "relaxed8b")]
        tm = sum(bool(r["ttft_miss"]) for r in main if r["slo_class"] == "tight"); rm = sum(bool(r["ttft_miss"]) for r in main if r["slo_class"] == "relaxed")
        tg = sum(sum(r["gap_misses"]) for r in m["per_request"] if r["slo_class"] == "tight"); rg = sum(sum(r["gap_misses"]) for r in m["per_request"] if r["slo_class"] == "relaxed")
        ng = {c: sum(len(r["gaps"]) for r in m["per_request"] if r["slo_class"] == c) for c in ("tight", "relaxed")}
        print("   TTFT ms", {r["name"]: ms(r["ttft"]) for r in m["per_request"]})
        print(f"   main-cohort TTFT misses tight {tm}/2 relaxed {rm}/2 | TBT misses tight {tg}/{ng['tight']} relaxed {rg}/{ng['relaxed']}")
        for r in m["per_request"]:
            if r["name"] in ("relaxed8a", "relaxed8b"):
                gaps = r["gaps"]
                print(f"   incumbent {r['name']}: gaps {len(gaps)} max {ms(max(gaps))} misses {sum(r['gap_misses'])} gaps_ms {[round(x*1000,1) for x in gaps]}")
        # tight33 timeline.
        sid = next(c["seq_id"] for c in t["cohort"] if c["name"] == "tight33")
        processed = 0
        prompt = next(c["prompt_len"] for c in t["cohort"] if c["name"] == "tight33")
        done_step = first_decode = None
        for st, d in zip(t["steps"], a["decisions"]):
            for x, chunk in st["scheduled"]:
                if x == sid and chunk > 0:
                    processed += chunk
                    if processed == prompt and done_step is None:
                        done_step = (st, d)
            if first_decode is None and any(e["seq_id"] == sid for e in st["events"]):
                first_decode = st
        st, d = done_step
        moved = {names[int(k)]: v for k, v in d["counterfactual_actions"].items() if d["actions"].get(k) != v}
        took = {names[int(k)]: v for k, v in d["actions"].items() if d["counterfactual_actions"].get(k) != v}
        actual = {names[int(k)]: v for k, v in d["actions"].items()}
        other = {names[int(k)]: v for k, v in d["counterfactual_actions"].items()}
        print(f"   tight33: prefill finished at d{st['index']} (+{ms(st['returned']-t0)} ms, bonus_earned={sid in d['bonus_earned']}), first decode d{first_decode['index']} (+{ms(first_decode['returned']-t0)} ms); at d{st['index']} actual", actual, "vs", d["counterfactual_policy"], other)
