"""CPU-only stand-in for the offline comparison: exits FAKE_OFFLINE_EXIT."""
import os, sys
code = int(os.environ.get("FAKE_OFFLINE_EXIT", "0"))
print(f"RESULT: fake offline exit {code}")
sys.exit(code)
