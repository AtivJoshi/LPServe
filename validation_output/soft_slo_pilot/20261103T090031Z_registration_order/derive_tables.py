"""Derive the registration-order tables from the retained artifacts (read-only)."""
import json, sys
from pathlib import Path
E = Path(sys.argv[1])
ms = lambda v: "-" if v is None else f"{v*1000:.1f}"
frozen = {l.split()[1]: l.split()[0] for l in (E / "provenance/sha256.txt").read_text().splitlines()}
for run in sorted(p.parent.name for p in E.glob("*/summary.json")):
    s = json.load(open(E / run / "summary.json"))
    fh = s["provenance"]["file_sha256"]
    same = all(fh[f] == frozen[f] for f in ("lpserve_state_mapping.py", "scripts/soft_slo_pilot.py", "tests/test_soft_slo_pilot.py"))
    print(f"{run}: passed={s['validation_passed']} order={s.get('registration_order')} code_matches_frozen={same} head={s['provenance']['git_head'][:7]} settings_sha={s['settings']['sha256'][:12]} gates={s.get('gates')} failure={s['failure'] and s['failure']['message']}")
g = json.load(open(E / "gpu_ample_relaxed_first/summary.json"))
wi = g["worker_identity"]
print(f"\nGPU: {wi['device_name']} {g['environment']['gpu']} host={g['environment']['hostname']} job={g['environment']['slurm']['SLURM_JOB_ID']} load_format={wi['load_format']} {wi['dtype']} params={wi['num_parameters']} weights_match={[v['file_head'] == v['worker_head'] for v in g['weight_check'].values()]} cache={g['cache_capacity']['profiled_available_blocks']}->{g['cache_capacity']['chosen_initialized_blocks']} versions={g['environment']['versions']}")
for order in ("relaxed_first", "tight_first"):
    run = f"gpu_ample_{order}"
    s = json.load(open(E / run / "summary.json"))
    print(f"\n=== {run}  prompts identical to relaxed_first: {s['prompts'] == g['prompts']}")
    for lab, m in s["metrics"].items():
        t = json.load(open(E / run / "trials" / f"{lab}.json"))
        a = s["analyses"][lab]
        names = {c["seq_id"]: c["name"] for c in t["cohort"]}
        cls = {c["seq_id"]: c["slo_class"] for c in t["cohort"]}
        # First full (16-token) prompt chunk for each arrival group.
        firsts = []
        for group in (("relaxed17", "tight17"), ("relaxed33", "tight33")):
            for st in t["steps"]:
                hit = [names[sid] for sid, chunk in st["scheduled"] if chunk == 16 and names[sid] in group]
                if hit:
                    firsts.append(f"{'/'.join(hit)}@d{st['index']}"); break
        fin = all(r["finished_by_cutoff"] for r in m["per_request"])
        pre = sum(len(st["preempted"]) for st in t["steps"])
        cont = sum(d["contention"]["contended"] for d in a["decisions"])
        diff = sum(d["action_difference"] for d in a["decisions"])
        bc = m["by_class"]
        cell = lambda c: (f"{c}: TTFT med {ms(bc[c]['ttft']['median'])} miss {sum(bool(r['ttft_miss']) for r in m['per_request'] if r['slo_class'] == c)}/2 "
                          f"TBT miss {sum(sum(r['gap_misses']) for r in m['per_request'] if r['slo_class'] == c)}/{bc[c]['tbt']['observed_gaps']}")
        print(f" {lab:16} order={[names[c['seq_id']] for c in t['cohort']]} first16={firsts} finished_in_cutoff={fin} decisions={len(t['steps'])} contended={cont} preempt={pre} diffs={diff}")
        print(f"   {cell('tight')} | {cell('relaxed')}")
        print("   per-request TTFT ms:", {r["name"]: ms(r["ttft"]) for r in m["per_request"]})
