"""CPU checks for the soft-SLO coefficient policies, output clock, latency
metrics, and the pilot driver's live single-stage engine path.

No model, Ray worker, or GPU is used. The live tests run the driver's
``run_session`` on its CPU engine: the registered LP scheduler, the native
engine ``add_request``/``step``/completion path, and a worker sequence
manager replaying copies, with synthetic samples.
"""

import functools
import math
import sys
import unittest
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import soft_slo_pilot as pilot  # noqa: E402
from check_lp_scheduler_gpu import CheckFailure  # noqa: E402

import lp_relaxation_scheduler as lrs  # noqa: E402
import lpserve_state_mapping as lsm  # noqa: E402

K = 16


def _resident(raw_id, *, remaining, output_len, arrival=0.0):
    """A captured resident: partial prefill if ``remaining`` else decode."""
    decode = remaining == 0
    return lsm.CapturedRequest(
        raw_seq_id=raw_id, request_id=str(raw_id), order_key=(raw_id,),
        ownership=lsm.OWNERSHIP_RUNNING, queue_position=raw_id,
        status="PAUSED", arrival_time=arrival, prompt_len=32,
        prompt_tokens_processed=32 - remaining,
        prompt_tokens_remaining=remaining,
        prompt_processing_finished=decode, output_len=output_len,
        context_len=32 + output_len, logical_block_count=2,
        physical_block_count=2, prefill_eligible=not decode,
        decode_eligible=decode, preemption_eligible=True,
        prefill_fixed_charge=0, decode_demand=0, preemption_recovery=2)


def _captured(requests, now):
    return lsm.CapturedState(
        snapshot_id="t", snapshot_time=now, scheduler_iteration_id=0,
        num_pipeline_stages=1, num_running_batches=0,
        resident_count=sum(r.ownership == lsm.OWNERSHIP_RUNNING
                           for r in requests),
        resident_limit=4, max_model_len=64, block_size=16, total_blocks=32,
        free_physical_blocks=20, watermark_blocks=0, requests=tuple(requests),
        running_order=tuple(r.raw_seq_id for r in requests
                            if r.ownership == lsm.OWNERSHIP_RUNNING),
        arrived_waiting_order=tuple(r.raw_seq_id for r in requests
                                    if r.ownership == lsm.OWNERSHIP_WAITING))


def _timing(raw_id, cls, last_visible=None, arrival=0.0, priority=1):
    return lsm.RequestTiming(raw_id, cls, **pilot.SLO_CLASSES[cls],
                             priority=priority, arrival_time=arrival,
                             last_visible=last_visible)


def _timed(timing):
    return functools.partial(lsm.soft_slo_timed_utility, timing=timing,
                             batch_forecast=0.020, b_max=17, c_max=16)


class PolicyTest(unittest.TestCase):

    def test_boost_regions_terms_and_zero_ineligible(self):
        # Decode-ready tight request, last token at 0.10, now 0.10:
        # slack = 0.14 - 0.10 - 0.02 = 0.02; boost = 1 + 0.02/0.04 = 1.5.
        decode = _resident(0, remaining=0, output_len=3)
        # Partial prefill of 17 tokens with relaxed TTFT, now 0.10:
        # batches 1 + 2 = 3, slack = 0.30 - 0.10 - 0.06 = 0.14,
        # boost = 1 + (0.30 - 0.14) / 0.30.
        prefill = _resident(1, remaining=17, output_len=0)
        timing = {0: _timing(0, "tight", 0.10), 1: _timing(1, "relaxed")}
        captured = _captured([decode, prefill], 0.10)
        out = lsm.construct_coefficients(captured, _timed(timing))
        self.assertAlmostEqual(out[0].decode_utility, 1.5)
        self.assertEqual(out[0].prefill_token_utility, 0.0)
        self.assertEqual(out[0].preemption_penalty, 1.0)
        boost = 1 + (0.30 - 0.14) / 0.30
        self.assertAlmostEqual(out[1].prefill_token_utility, boost / K)
        self.assertEqual(out[1].decode_utility, 0.0)
        self.assertEqual(out[1].preemption_penalty, 1.0)
        f = lsm.soft_slo_forecast(captured, prefill, timing=timing,
                                  batch_forecast=0.020, b_max=17, c_max=16)
        self.assertEqual((f.target_kind, f.estimated_batches, f.nominal_chunk),
                         ("ttft", 3, 16))
        # Cap (far overdue) and floor (slack equal to the target).
        late = _captured([decode], 0.50)
        timing_far = {0: _timing(0, "tight", 0.0)}
        f_late = lsm.soft_slo_forecast(late, decode, timing=timing_far,
                                       batch_forecast=0.020, b_max=17,
                                       c_max=16)
        self.assertEqual((f_late.boost, f_late.boost_region), (3.0, "cap"))
        relaxed = {0: _timing(0, "relaxed", 0.0)}
        f_floor = lsm.soft_slo_forecast(_captured([decode], 0.0), decode,
                                        timing=relaxed, batch_forecast=0.0,
                                        b_max=17, c_max=16)
        self.assertEqual((f_floor.boost, f_floor.boost_region), (1.0, "floor"))

    def test_untimed_terms(self):
        captured = _captured([_resident(0, remaining=0, output_len=1),
                              _resident(1, remaining=5, output_len=0)], 1.0)
        out = lsm.construct_coefficients(captured, functools.partial(
            lsm.soft_slo_untimed_utility, b_max=17, c_max=16))
        self.assertEqual(tuple(out[0].__dict__.values()), (1.0, 0.0, 1.0))
        self.assertEqual(tuple(out[1].__dict__.values()), (0.0, 1 / 16, 1.0))

    def test_rebuild_keeps_the_tbt_clock_and_counts_rebuild_work(self):
        # After recomputation: output list empty, 33 tokens to rebuild,
        # but a visible token was seen at 0.20 -> TBT clock, 1 + 3 batches.
        rebuild = lsm.CapturedRequest(**dict(
            _resident(0, remaining=33, output_len=0).__dict__,
            ownership=lsm.OWNERSHIP_WAITING, status="WAITING",
            prompt_len=33, prompt_tokens_processed=0,
            preemption_eligible=False, physical_block_count=0,
            prefill_fixed_charge=3, preemption_recovery=0))
        timing = {0: _timing(0, "relaxed", 0.20)}
        f = lsm.soft_slo_forecast(_captured([rebuild], 0.25), rebuild,
                                  timing=timing, batch_forecast=0.020,
                                  b_max=32, c_max=16)
        self.assertEqual((f.target_kind, f.estimated_batches), ("tbt", 4))
        self.assertAlmostEqual(f.due, 0.32)
        self.assertAlmostEqual(f.slack, 0.32 - 0.25 - 0.08)

    def test_invalid_timing_inputs_fail_mapping_visibly(self):
        decode = _resident(0, remaining=0, output_len=2)
        captured = _captured([decode], 1.0)
        cases = {
            "no timing input": {},
            "only priority 1": {0: _timing(0, "tight", 0.5, priority=2)},
            "differs from the captured arrival": {
                0: _timing(0, "tight", 0.5, arrival=0.1)},
            "no visible-output time": {0: _timing(0, "tight", None)},
            "outside": {0: _timing(0, "tight", 2.0)},
        }
        for text, timing in cases.items():
            out = lsm.construct_coefficients(captured, _timed(timing))
            self.assertIsInstance(out, lsm.MappingFailure, text)
            self.assertIn(text, out.reason)

    def test_flip_example_ph1_and_exact(self):
        flip = pilot.flip_example()
        untimed, timed = flip["untimed"], flip["timed"]
        self.assertEqual(untimed["full_prefill_reward"], {"0": 1.0, "1": 0.75})
        self.assertAlmostEqual(timed["full_prefill_reward"]["0"], 4 / 3)
        self.assertAlmostEqual(timed["full_prefill_reward"]["1"], 1.5)
        self.assertEqual(untimed["ph1_actions"], {0: "prefill:16"})
        self.assertEqual(untimed["exact_actions"], {0: "prefill:16"})
        self.assertEqual(timed["ph1_actions"], {1: "prefill:12"})
        self.assertEqual(timed["exact_actions"], {1: "prefill:12"})

    def test_targets_from_class_defaults_and_overrides(self):
        r = dict(name="a", slo_class="tight")
        self.assertEqual(pilot.resolve_targets(r), (0.100, 0.040))
        self.assertEqual(pilot.resolve_targets(dict(r, tbt_target=0.05)),
                         (0.100, 0.05))
        for bad in (dict(r, slo_class="other"), dict(r, ttft_target=0)):
            with self.assertRaises(CheckFailure):
                pilot.resolve_targets(bad)


class _FakeSeq:
    """Native-shaped sequence for clock tests: cumulative count, current
    output list (cleared by recomputation), and finish status."""

    def __init__(self, seq_id, arrival=0.0):
        self.seq_id, self.arrival_time = seq_id, arrival
        self.state = SimpleNamespace(num_output_tokens=0)
        self.output_token_ids, self.finished = [], False

    def emit(self, token, finish=False):
        self.output_token_ids.append(token)
        self.state.num_output_tokens += 1
        self.finished = finish

    def recompute(self):
        self.output_token_ids = []

    def is_finished(self):
        return self.finished


def _clock(*seqs, cutoff=None):
    clock = pilot.OutputClock(cutoff)
    for s in seqs:
        clock.register(s, f"r{s.seq_id}", "tight", 0.1, 0.04, 1)
    return clock


def _out(seq):
    return (seq.seq_id, list(seq.output_token_ids), seq.finished)


class OutputClockTest(unittest.TestCase):

    def test_prefill_repeated_ids_rebuild_and_final_output(self):
        a = _FakeSeq(0)
        clock = _clock(a, cutoff=10.0)
        # Partial and final prefill: sampler runs, no visible token.
        for t, chunk in ((1.0, 16), (2.0, 1)):
            self.assertEqual(clock.observe(t, 0, [(0, chunk)], [], [[0, 7]],
                                           [_out(a)]), [])
        self.assertIsNone(clock.requests[0]["last_visible"])
        # Two decodes with the same token ID are two events.
        for t in (3.0, 4.0):
            a.emit(5)
            (event,) = clock.observe(t, 1, [(0, 0)], [], [[0, 5]], [_out(a)])
            self.assertEqual((event["token_id"], event["time"]), (5, t))
        self.assertEqual(clock.requests[0]["count"], 2)
        # Recomputation clears the native list but not the clock.
        a.recompute()
        self.assertEqual(clock.observe(5.0, 2, [(0, 16)], [], [[0, 9]],
                                       [_out(a)]), [])
        self.assertEqual(clock.requests[0]["last_visible"], 4.0)
        self.assertEqual(clock.timing_inputs()[0].last_visible, 4.0)
        a.emit(6, finish=True)
        (event,) = clock.observe(6.0, 3, [(0, 0)], [], [[0, 6]], [_out(a)])
        self.assertTrue(event["finished"])
        self.assertEqual(clock.requests[0]["tokens"], [5, 5, 6])
        self.assertEqual(clock.requests[0]["finish_time"], 6.0)
        self.assertIsInstance(clock.timing_inputs(), MappingProxyType)

    def test_association_failures_are_visible(self):
        cases = [
            ("count and order", [[1, 6], [0, 5]], False),
            ("count and order", [[0, 5]], False),
            ("differs from sampler", [[0, 6], [1, 6]], False),
            ("returned outputs", [[0, 5], [1, 6]], True),
        ]
        for text, sampler, swap_outputs in cases:
            a, b = _FakeSeq(0), _FakeSeq(1)
            clock = _clock(a, b)
            a.emit(5)
            b.emit(6)
            outputs = [_out(a), _out(b)]
            with self.assertRaises(CheckFailure) as caught:
                clock.observe(1.0, 0, [(0, 0), (1, 0)], [], sampler,
                              outputs[::-1] if swap_outputs else outputs)
            self.assertIn(text, str(caught.exception))
        clock = _clock(_FakeSeq(0))
        with self.assertRaisesRegex(CheckFailure, "produced no visible token"):
            clock.observe(1.0, 0, [(0, 0)], [], [[0, 5]], [(0, [], False)])
        jump = _FakeSeq(0)
        clock = _clock(jump)
        jump.emit(1)
        jump.emit(2)
        with self.assertRaisesRegex(CheckFailure, "moved by 2"):
            clock.observe(1.0, 0, [(0, 0)], [], [[0, 2]], [_out(jump)])

    def test_step_returning_after_cutoff_is_flagged(self):
        a = _FakeSeq(0)
        clock = _clock(a, cutoff=1.0)
        a.emit(5)
        (on_cutoff,) = clock.observe(1.0, 0, [(0, 0)], [], [[0, 5]], [_out(a)])
        a.emit(6)
        (late,) = clock.observe(1.5, 1, [(0, 0)], [], [[0, 6]], [_out(a)])
        self.assertFalse(on_cutoff["after_cutoff"])
        self.assertTrue(late["after_cutoff"])


def _row(name, cls, events, finish=None, arrival=0.0):
    targets = pilot.SLO_CLASSES[cls]
    return dict(name=name, slo_class=cls, arrival_time=arrival,
                event_times=events, finish_time=finish, **targets)


class MetricsTest(unittest.TestCase):

    def test_ttft_cutoff_censoring_and_strict_miss(self):
        rows = [
            _row("on_target", "tight", [0.100, 0.140], finish=0.140),
            _row("late", "tight", [0.150], finish=None),
            _row("missing_past_due", "tight", []),
            _row("unresolved", "relaxed", [], arrival=0.9),
            _row("at_cutoff", "relaxed", [1.0], finish=1.0, arrival=0.8),
            _row("after_cutoff", "relaxed", [1.2], finish=1.2),
        ]
        m = pilot.latency_metrics(rows, cutoff=1.0)
        tight, relaxed = m["by_class"]["tight"]["ttft"], m["by_class"][
            "relaxed"]["ttft"]
        self.assertEqual((tight["observed"], tight["known_missing_misses"]),
                         (2, 1))
        self.assertEqual(tight["miss_rate"], 2 / 3)  # 0.100 is not a miss
        self.assertAlmostEqual(tight["mean_excess_observed"], 0.025)
        self.assertAlmostEqual(
            tight["censored_excess_lower_bounds"][0]["lower_bound"], 0.9)
        # Relaxed: at-cutoff observed (0.2 <= 0.3); after-cutoff known miss
        # (due 0.3 <= 1.0); arrival 0.9 due 1.2 unresolved -> bounds.
        self.assertEqual(relaxed["observed"], 1)
        self.assertIsNone(relaxed["miss_rate"])
        self.assertEqual(relaxed["miss_rate_bounds"], [1 / 3, 2 / 3])
        self.assertEqual(m["events_after_cutoff"],
                         [dict(name="after_cutoff", time=1.2)])

    def test_tbt_gaps_open_gaps_and_p99(self):
        rows = [
            _row("a", "tight", [0.1, 0.13, 0.20, 0.24], finish=0.24),
            _row("b", "tight", [0.1], finish=None),
            _row("c", "tight", [0.1, 0.12], finish=None),
        ]
        m = pilot.latency_metrics(rows, cutoff=0.15)
        tbt = m["by_class"]["tight"]["tbt"]
        # Within cutoff: a 0.1, 0.13 (gap 0.03); c gap 0.02; b none.
        self.assertEqual(tbt["observed_gaps"], 2)
        self.assertEqual((tbt["miss_rate"], tbt["mean_excess"]), (0.0, 0.0))
        self.assertEqual((tbt["streams_with_observed_gap"],
                          tbt["streams_without_observed_gap"]), (2, 1))
        overdue = {g["name"] for g in tbt["overdue_open_gaps"]}
        self.assertEqual(overdue, {"b"})  # 0.05 > 0.04; a 0.02, c 0.03
        m = pilot.latency_metrics(rows, cutoff=1.0)
        tbt = m["by_class"]["tight"]["tbt"]
        self.assertEqual(tbt["observed_gaps"], 4)
        self.assertEqual(tbt["miss_rate"], 1 / 4)  # 0.07; 0.04 is on target
        self.assertAlmostEqual(tbt["mean_excess"], 0.03 / 4)
        self.assertAlmostEqual(tbt["p99"], 0.07)
        self.assertEqual(tbt["share_streams_with_late_gap"], 1 / 2)
        self.assertEqual(pilot.nearest_rank([3, 1, 2], 0.5), 2)


class LiveCpuTest(unittest.TestCase):
    """The driver's sessions on the CPU engine (registration lead shortened
    to keep the tests fast)."""

    def setUp(self):
        patcher = mock.patch.object(pilot, "REGISTRATION_LEAD_SECONDS", 0.05)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _session(self, case, labels, sampler=pilot.distinct_sampler,
                 engine=None):
        engine = engine or pilot.build_cpu_engine(case, sampler)
        prompts = pilot.prompt_ids(pilot.CASES[case])
        return engine, pilot.run_session(
            engine, case, prompts, labels,
            functools.partial(pilot._worker_state, engine))

    def test_ample_pair_is_checked_and_drains_without_preemption(self):
        labels = [("t_timed", "timed", True), ("t_untimed", "untimed", True)]
        engine, (trials, analyses, metrics) = self._session("ample", labels)
        self.assertEqual([t["stop_reason"] for t in trials],
                         ["drained", "drained"])
        self.assertTrue(all(t["failure"] is None for t in trials))
        gates = pilot.summarize_pair(analyses)
        self.assertEqual(gates["preemptions"], 0)
        self.assertGreaterEqual(gates["action_differences"], 1)
        self.assertGreaterEqual(gates["contended_decisions"], 1)
        for trial in trials:
            events = trial["clock"].events
            self.assertEqual(len(events), 24)
            for r in trial["clock"].requests.values():
                self.assertEqual(r["count"], 6)
                self.assertIsNotNone(r["finish_time"])
            # Every decision mapped once, at or after every included arrival.
            for step in trial["steps"]:
                self.assertEqual(len(step["record"]["hooks"]), 1)
            record = pilot.trial_record(trial)
            self.assertEqual(len(record["steps"]), len(trial["steps"]))
        # The first decision of both policies sees the same two arrivals and
        # chooses differently (timed: the tight request gets the full chunk).
        first = {k: a["decisions"][0] for k, a in analyses.items()}
        tight17 = trials[0]["cohort"][1]["seq_id"]
        self.assertEqual(first["t_timed"]["actions"][str(tight17)],
                         "prefill:16")
        self.assertTrue(first["t_timed"]["action_difference"])
        for m in metrics.values():
            for cls in ("tight", "relaxed"):
                self.assertEqual(m["by_class"][cls]["ttft"]["observed"], 2)
                self.assertEqual(m["by_class"][cls]["tbt"]["observed_gaps"],
                                 10)
        pilot.check_quiescent_engine(engine, pilot.CASES["ample"], "end")

    def test_repeated_token_ids_are_separate_events(self):
        _, (trials, _, _) = self._session(
            "ample", [("rep", "untimed", True)], sampler=lambda sid, step: 42)
        self.assertEqual(trials[0]["stop_reason"], "drained")
        for r in trials[0]["clock"].requests.values():
            self.assertEqual(r["tokens"], [42] * 6)

    def test_pressure_witness_spans_a_rebuild(self):
        _, (trials, analyses, _) = self._session(
            "pressure", [("p", "timed", True)])
        trial, analysis = trials[0], analyses["p"]
        self.assertEqual(trial["stop_reason"], "drained")
        victims = [(s["index"], sid) for s in trial["steps"]
                   for sid in s["preempted"]]
        self.assertEqual(len(victims), 1)
        index, victim = victims[0]
        clock = trial["clock"].requests[victim]
        before = [e for e in trial["clock"].events
                  if e["seq_id"] == victim and e["decision"] < index]
        after = [e for e in trial["clock"].events
                 if e["seq_id"] == victim and e["decision"] > index]
        self.assertGreater(len(before), 0)  # post-output eviction
        rebuild = [s for s in trial["steps"] if s["index"] > index
                   and any(sid == victim and chunk > 0
                           for sid, chunk in s["scheduled"])]
        self.assertTrue(rebuild)
        self.assertTrue(all(not any(e["seq_id"] == victim for e in s["events"])
                            for s in rebuild))  # rebuild without output
        # The clock (and the timing input during rebuild) keeps the last
        # pre-eviction token; the next token's gap spans the rebuild.
        for s in rebuild:
            self.assertEqual(s["record"]["hooks"][0][victim].last_visible,
                             before[-1]["time"])
        self.assertGreater(after[0]["decision"], rebuild[-1]["index"])
        gap_start, gap_end = before[-1]["time"], after[0]["time"]
        self.assertIn(gap_end - gap_start, [
            b - a for a, b in zip(clock["event_times"],
                                  clock["event_times"][1:])])
        work = analysis["work"][clock["name"]]
        self.assertEqual(work["preemptions"], 1)
        self.assertGreater(work["repeated_prefill_tokens"], 0)
        self.assertGreater(work["extra_outputs"], 0)  # inherited overgeneration

    def test_pressure_control_fits_without_preemption(self):
        _, (trials, analyses, _) = self._session(
            "pressure_control", [("c", "timed", True)])
        self.assertEqual(trials[0]["stop_reason"], "drained")
        self.assertEqual(pilot.summarize_pair(analyses)["preemptions"], 0)
        for r in trials[0]["clock"].requests.values():
            self.assertEqual(r["count"], 5)

    def test_misassociated_samples_fail_the_trial_and_stop(self):
        engine = pilot.build_cpu_engine("ample")
        native = engine._run_workers

        def swapped(method, *args, **kwargs):
            result = native(method, *args, **kwargs)
            if method == "execute_model":
                engine.validation_sampler_outputs.reverse()
            return result
        engine._run_workers = swapped
        _, (trials, analyses, _) = self._session(
            "ample", [("a", "timed", True), ("b", "untimed", True)],
            engine=engine)
        self.assertEqual(len(trials), 1)
        self.assertIn("do not match scheduled", trials[0]["failure"]["message"])
        self.assertEqual(analyses, {})

    def test_mapping_failure_is_fail_stop_with_partial_trace(self):
        with mock.patch.object(pilot, "PRIORITY", 2):
            _, (trials, _, _) = self._session("ample", [("a", "timed", True)])
        failure = trials[0]["failure"]
        self.assertEqual((failure["type"], failure["stage"]),
                         ("LPSchedulingError", "state_mapping"))
        self.assertIn("only priority 1", failure["reason"])
        self.assertEqual(len(trials[0]["steps"]), 0)
        failed = trials[0]["failed_step"]["record"]
        self.assertEqual([len(failed[k]) for k in ("hooks", "snapshots",
                                                   "results", "executions",
                                                   "outputs")],
                         [1, 1, 0, 0, 0])
        self.assertIsInstance(failed["snapshots"][0], lsm.MappingFailure)
        self.assertEqual(trials[0]["final_central"]["running"], [])
        self.assertEqual(pilot.trial_record(trials[0])["failed_step"]["index"],
                         0)

    def test_cutoff_stops_with_unfinished_work_visible(self):
        case = dict(pilot.CASES["ample"], cutoff_seconds=0.0)
        with mock.patch.dict(pilot.CASES, {"ample": case}):
            engine, (trials, analyses, metrics) = self._session(
                "ample", [("a", "timed", True), ("b", "timed", True)])
        self.assertEqual(len(trials), 1)
        self.assertEqual(trials[0]["stop_reason"], "cutoff")
        self.assertEqual(trials[0]["steps"], [])
        ttft = metrics["a"]["by_class"]["tight"]["ttft"]
        self.assertEqual((ttft["unresolved"], ttft["observed"]), (2, 0))
        self.assertEqual(ttft["miss_rate_bounds"], [0.0, 1.0])
        self.assertEqual(len(trials[0]["final_central"]["waiting"]), 4)


if __name__ == "__main__":
    unittest.main()
