"""CPU-only stand-in for the GPU driver in the runner propagation check:
writes a placeholder trace and exits with FAKE_EXIT_<policy> (default 0)."""
import argparse, os, sys
from pathlib import Path
p = argparse.ArgumentParser()
for a in ("--case", "--policy", "--model-path", "--output-dir"):
    p.add_argument(a)
args = p.parse_args()
Path(args.output_dir, "decision_trace.json").write_text("{}")
code = int(os.environ.get(f"FAKE_EXIT_{args.policy}", "0"))
print(f"fake driver {args.policy} exit {code}")
sys.exit(code)
