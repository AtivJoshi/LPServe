# Interrupted CPU-gate run (not evidence of a complete check)

This directory was created on `gpu052.unity.rc.umass.edu` (SLURM job
`65493676`) at 2026-10-10T06:27:05Z by the CPU-gate command for the
dummy-weight GPU preemption check. The command was interrupted at the user's
request while the second suite was starting; the allocation was later
preempted by Unity and that session ended.

What it holds:

- `cpu_checks/environment.txt`: host, job, modules, GPU, HEAD
  `04beb8789f17e11d55aec72d9f168353d486b2ba`, `git status`, versions, and
  file hashes. These hashes equal the files used by the later complete run.
- `cpu_checks/1-test_lp_relaxation_scheduler.log`: complete; 5 tests OK,
  exit status 0.
- `cpu_checks/2-test_lpserve_state_mapping.log`: incomplete; only the
  command line was written before the interruption. No result exists.

Never started here: the executor, live-scheduler, and driver suites, the
static checks, and any GPU attempt. No engine, worker, or cache state was
created. The complete CPU gates and the GPU attempt were run fresh on a new
allocation in a separate timestamped directory.
