"""Read-only extraction of the preemption trace's boundary, per-step mapped
facts, relaxed/integer objectives, and central/worker block operations.
Loads only the run's JSON files; asserts nothing beyond what it prints."""
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
s = json.loads((out / "summary.json").read_text())
t = json.loads((out / "decision_trace.json").read_text())
b = s["boundary"]
print("boundary relaxed objective", b["relaxed"]["normalized_objective"],
      "worst normalized violation", b["relaxed"]["worst_normalized_violation"])
for d in b["relaxed"]["decisions"]:
    print("  relaxed", d)
p = b["plan"]
print("  plan", [(d["request_id"], d["prefill_tokens"], d["decode"], d["preempt"]) for d in p["decisions"]],
      "objective", p["objective"], "victim_order", p["victim_order"], "execution_order", p["execution_order"],
      "free_blocks_before_decodes", p["free_blocks_before_decodes"], "resident_count_after", p["resident_count_after"])
print("  central ops", b["central_block_ops"]); print("  worker ops", b["worker_block_ops"])
print("fields: (raw, ownership, processed, physical, charge, recovery, decode_demand, prefill_elig, decode_elig, preempt_elig)")
for st in t["steps"]:
    ev = st["events"]; snap = ev[0]["result"]; sol = ev[1]["result"]
    reqs = [(r["raw_seq_id"], r["ownership"], r["prompt_tokens_processed"], r["physical_block_count"],
             r["prefill_fixed_charge"], r["preemption_recovery"], r["decode_demand"], r["prefill_eligible"],
             r["decode_eligible"], r["preemption_eligible"]) for r in snap["requests"]]
    print(st["index"], st["name"], "m_free", snap["lp_problem"]["m_free"], "h", snap["lp_problem"]["admission_watermark"],
          "legal_victims", snap["lp_problem"]["legal_preemption_ids"], "relaxed", sol["relaxed"]["normalized_objective"],
          "integer", sol["plan"]["objective"], "fractional", sol["plan"]["extraction"]["fractional_request_count"])
    print("   requests", reqs)
    print("   central tables before", st["before"]["block_tables"], "after", st["after"]["block_tables"],
          "| worker after", st["worker_after"]["block_tables"])
    print("   central ops", st["central_block_ops"], "| worker ops", st["worker_after"]["block_ops"])
print("finish order", s["finish_order"], {k: (v["token_ids"], v["finish_reason"]) for k, v in s["final_outputs"].items()})
i = t["idle"]; print("idle calls", [e["call"] for e in i["events"]], "iteration", i["before"]["iteration_id"], "->", i["after"]["iteration_id"])
