# Trace attempt 1: evidence-script comparison bug

The first run of `trace_live_preemption.py` (copy kept here as
`trace_live_preemption.attempt1.py`) exited with status 1 and
`"passed": false` because the `boundary_extraction` check was false.

Cause: the script stored `test_lp_scheduler._emitted(outputs)` (a list of
tuples) and compared it with `[[1, 4]]` (a list of lists) before JSON
serialization, so the comparison was false even though the recorded
boundary outputs were `scheduled [(1, 4)]` and `preempted_seq_ids [0]`
(see `trace_console.log` and `decision_trace.json` here). All other checks
were true. This is a bug in the evidence script, not a scheduler result.

Fix for attempt 2 (the script one directory up): the recorded `scheduled`
entries are converted to lists. No test, production, or input change was
made between the attempts.
