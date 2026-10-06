"""Program-behavior tests for scripts/run_lp_inference.py.

A small fake engine stands in for BaseLLMEngine. These tests check input
handling, request/output association, failure propagation, and artifact
writing only; they do not validate scheduler correctness.
"""

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(REPO_ROOT))

import run_lp_inference as rli  # noqa: E402

from sarathi.core.datatypes.request_output import RequestOutput  # noqa: E402


class FakeTokenizer:
    """One token per whitespace-separated word, plus a leading BOS (1)."""

    def encode(self, text):
        return [1] + [100 + len(w) for w in text.split()]


class FakeEngine:
    """Minimal engine surface used by run_requests.

    ``finish_plan[k]`` lists the input indexes whose request finishes at step
    ``k + 1``; other steps return one unfinished output. ``fail_at`` raises at
    that step number. Sequence IDs start at ``first_seq_id`` so they differ
    from input positions.
    """

    def __init__(self, finish_plan, fail_at=None, first_seq_id=7,
                 tokenizer=None):
        self.tokenizer = tokenizer or FakeTokenizer()
        self.seq_manager = SimpleNamespace(seq_map={})
        self.finish_plan = finish_plan
        self.fail_at = fail_at
        self.next_seq_id = first_seq_id
        self.order = []
        self.unfinished = set()
        self.step_calls = 0
        self.model_config = SimpleNamespace(dtype="torch.float16",
                                            max_model_len=32, hf_config=None)
        self.scheduler = SimpleNamespace(
            block_manager=SimpleNamespace(num_total_gpu_blocks=100))

    def add_request(self, prompt, sampling_params, prompt_token_ids=None):
        assert prompt_token_ids is None
        seq_id = self.next_seq_id
        self.next_seq_id += 3
        self.seq_manager.seq_map[seq_id] = SimpleNamespace(
            seq_id=seq_id, prompt=prompt,
            prompt_token_ids=self.tokenizer.encode(prompt))
        self.order.append(seq_id)
        self.unfinished.add(seq_id)

    def has_unfinished_requests(self):
        return bool(self.unfinished)

    def _output(self, seq_id, finished):
        seq = self.seq_manager.seq_map[seq_id]
        return RequestOutput(seq_id, seq.prompt, seq.prompt_token_ids,
                             f" answer{seq_id}", [seq_id, 2, 3, 4], finished,
                             "length" if finished else None)

    def step(self):
        self.step_calls += 1
        if self.step_calls == self.fail_at:
            raise RuntimeError("simulated execution failure")
        plan = (self.finish_plan[self.step_calls - 1]
                if self.step_calls <= len(self.finish_plan) else [])
        outputs = []
        for index in plan:
            seq_id = self.order[index]
            outputs.append(self._output(seq_id, True))
            self.unfinished.discard(seq_id)
        if not outputs and self.unfinished:
            outputs.append(self._output(min(self.unfinished), False))
        return outputs


PROMPTS = ["Explain why plants need sunlight.",
           "  Describe how rain forms.\n",
           "Why sleep?"]


def encoded(prompts):
    return rli.encode_prompts(FakeTokenizer(), prompts)


class ReadPromptsTest(unittest.TestCase):

    def write(self, text):
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                        encoding="utf-8")
        f.write(text)
        f.close()
        self.addCleanup(os.unlink, f.name)
        return f.name

    def test_accepts_one_to_three_strings_unchanged(self):
        for prompts in (PROMPTS[:1], PROMPTS, ["héllo ✓", " x "]):
            path = self.write(json.dumps(prompts))
            self.assertEqual(rli.read_prompts(path), prompts)

    def test_rejects_malformed_input(self):
        cases = {
            "not json": "[\"a\",",
            "object": json.dumps({"a": "b"}),
            "string": json.dumps("a"),
            "empty list": "[]",
            "four prompts": json.dumps(["a", "b", "c", "d"]),
            "number": json.dumps(["a", 3]),
            "null": json.dumps([None]),
            "nested": json.dumps([["a"]]),
            "empty": json.dumps(["a", ""]),
            "whitespace": json.dumps([" \n\t "]),
        }
        for name, text in cases.items():
            with self.subTest(name):
                with self.assertRaises(rli.InputError):
                    rli.read_prompts(self.write(text))
        with self.assertRaises(rli.InputError):
            rli.read_prompts("/nonexistent/prompts.json")


class EncodePromptsTest(unittest.TestCase):

    def test_length_bounds(self):
        limit = rli.MAX_MODEL_LEN - rli.MAX_TOKENS  # prompt tokens allowed
        fits = " ".join(["w"] * (limit - 1))  # plus BOS = limit tokens
        self.assertEqual(len(encoded([fits])[0]), limit)
        with self.assertRaisesRegex(rli.InputError, "exceeds model length"):
            encoded([fits + " w"])

    def test_rejects_zero_tokens(self):
        tokenizer = SimpleNamespace(encode=lambda text: [])
        with self.assertRaisesRegex(rli.InputError, "no tokens"):
            rli.encode_prompts(tokenizer, ["a"])


class RunRequestsTest(unittest.TestCase):

    def test_results_in_input_order_when_finished_out_of_order(self):
        engine = FakeEngine(finish_plan=[[], [2], [0], [], [1]])
        ids = encoded(PROMPTS)
        progress = {}
        results = rli.run_requests(engine, PROMPTS, ids, object(), progress)
        self.assertEqual([r["index"] for r in results], [0, 1, 2])
        self.assertEqual([r["seq_id"] for r in results], [7, 10, 13])
        self.assertEqual(progress["seq_ids"], {7: 0, 10: 1, 13: 2})
        for index, r in enumerate(results):
            self.assertEqual(r["prompt"], PROMPTS[index])
            self.assertEqual(r["prompt_token_ids"], ids[index])
            self.assertEqual(r["generated_text"], f" answer{r['seq_id']}")
            self.assertEqual(r["generated_token_ids"], [r["seq_id"], 2, 3, 4])
            self.assertEqual(r["finish_reason"], "length")
        self.assertEqual(engine.step_calls, 5)

    def test_all_requests_added_before_first_step(self):
        engine = FakeEngine(finish_plan=[[0, 1, 2]])
        original_step = engine.step

        def step():
            self.assertEqual(len(engine.order), 3)
            return original_step()

        engine.step = step
        rli.run_requests(engine, PROMPTS, encoded(PROMPTS), object())

    def test_failure_propagates_without_retry(self):
        engine = FakeEngine(finish_plan=[[1], [], [0, 2]], fail_at=2)
        progress = {}
        with self.assertRaisesRegex(RuntimeError, "simulated"):
            rli.run_requests(engine, PROMPTS, encoded(PROMPTS), object(),
                             progress)
        self.assertEqual(engine.step_calls, 2)
        self.assertEqual(list(progress["finished"]), [10])

    def test_rejects_duplicate_finished_output(self):
        engine = FakeEngine(finish_plan=[[0], [0, 1]])
        with self.assertRaisesRegex(rli.RunError, "finished twice"):
            rli.run_requests(engine, PROMPTS[:2], encoded(PROMPTS[:2]),
                             object())
        self.assertEqual(engine.step_calls, 2)

    def test_rejects_unknown_finished_output(self):
        engine = FakeEngine(finish_plan=[[0]])
        original_step = engine.step

        def step():
            outputs = original_step()
            outputs[0].seq_id = 999
            return outputs

        engine.step = step
        with self.assertRaisesRegex(rli.RunError, "unknown sequence 999"):
            rli.run_requests(engine, PROMPTS[:1], encoded(PROMPTS[:1]),
                             object())

    def test_completion_guard_stops_stalled_run(self):
        engine = FakeEngine(finish_plan=[])
        ids = encoded(PROMPTS[:1])
        with self.assertRaisesRegex(rli.RunError, "still unfinished"):
            rli.run_requests(engine, PROMPTS[:1], ids, object())
        self.assertEqual(engine.step_calls, len(ids[0]) + rli.MAX_TOKENS)

    def test_rejects_engine_encoding_mismatch(self):
        engine = FakeEngine(finish_plan=[[0]])
        ids = [[1, 2, 3]]
        with self.assertRaisesRegex(rli.RunError, "encoded differently"):
            rli.run_requests(engine, PROMPTS[:1], ids, object())
        self.assertEqual(engine.step_calls, 0)

    def test_rejects_prompt_and_token_list_count_mismatch(self):
        cases = {"two prompts, one token list": (PROMPTS[:2], [[1]], 2, 1),
                 "one prompt, two token lists": (PROMPTS[:1], [[1], [2]], 1,
                                                 2)}
        for name, (prompts, ids, n_prompts, n_ids) in cases.items():
            with self.subTest(name):
                engine = FakeEngine(finish_plan=[[0, 1]])
                with self.assertRaisesRegex(
                        rli.RunError, f"got {n_prompts} prompts but {n_ids} "
                        "token lists"):
                    rli.run_requests(engine, prompts, ids, object())
                self.assertEqual(engine.order, [])
                self.assertEqual(engine.seq_manager.seq_map, {})
                self.assertEqual(engine.step_calls, 0)


class CommandTest(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.model = self.tmp / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text("{}")
        self.prompts = self.tmp / "prompts.json"
        self.prompts.write_text(json.dumps(PROMPTS))
        self.out = self.tmp / "run"
        env = {"PYTHONPATH": str(REPO_ROOT)}
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)

    def argv(self, out=None, prompts=None):
        return ["--model-path", str(self.model),
                "--prompts-file", str(prompts or self.prompts),
                "--output-dir", str(out or self.out)]

    def run_main(self, engine, argv=None):
        with mock.patch.object(rli, "create_engine",
                               return_value=engine) as create, \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            status = rli.main(argv or self.argv())
        return status, create

    def test_complete_command_path(self):
        engine = FakeEngine(finish_plan=[[1], [2, 0]])
        status, create = self.run_main(engine)
        self.assertEqual(status, 0)
        create.assert_called_once()
        self.assertEqual(create.call_args.args[0], str(self.model))
        results = json.loads((self.out / "results.json").read_text())
        self.assertEqual([(r["index"], r["seq_id"], r["prompt"])
                          for r in results],
                         [(0, 7, PROMPTS[0]), (1, 10, PROMPTS[1]),
                          (2, 13, PROMPTS[2])])
        summary = json.loads((self.out / "summary.json").read_text())
        self.assertTrue(summary["success"])
        self.assertIsNone(summary["failure"])
        self.assertEqual(summary["seq_ids"], {"0": 7, "1": 10, "2": 13})
        self.assertEqual(summary["settings"], json.loads(json.dumps(
            rli.SETTINGS)))
        self.assertEqual(summary["prompts_file_sha256"],
                         rli.sha256_file(self.prompts))
        self.assertEqual(summary["command"]["main_argv"], self.argv())
        self.assertIn("git_head", summary["provenance"])
        self.assertFalse(summary["ray"]["started_by_this_run"])

        # A second run into the same directory is refused before any work.
        with self.assertRaises(SystemExit) as exit_info, \
                redirect_stderr(io.StringIO()):
            self.run_main(FakeEngine(finish_plan=[[0, 1, 2]]))
        self.assertEqual(exit_info.exception.code, 2)

    def test_failure_writes_summary_and_no_results(self):
        engine = FakeEngine(finish_plan=[[1], [], [0, 2]], fail_at=2)
        status, _ = self.run_main(engine)
        self.assertEqual(status, 1)
        self.assertEqual(engine.step_calls, 2)
        self.assertFalse((self.out / "results.json").exists())
        summary = json.loads((self.out / "summary.json").read_text())
        self.assertFalse(summary["success"])
        self.assertEqual(summary["failure"]["type"], "RuntimeError")
        self.assertIn("simulated", summary["failure"]["message"])
        self.assertEqual(summary["finished_before_failure"],
                         [dict(index=1, seq_id=10)])

    def test_invalid_input_rejected_before_engine(self):
        bad = self.tmp / "bad.json"
        bad.write_text(json.dumps(["a", "b", "c", "d"]))
        with self.assertRaises(SystemExit) as exit_info, \
                redirect_stderr(io.StringIO()):
            self.run_main(FakeEngine(finish_plan=[]), self.argv(prompts=bad))
        self.assertEqual(exit_info.exception.code, 2)
        self.assertFalse(self.out.exists())

    def test_overlong_prompt_fails_before_any_request(self):
        self.prompts.write_text(json.dumps(["w " * 40]))
        engine = FakeEngine(finish_plan=[[0]])
        status, _ = self.run_main(engine)
        self.assertEqual(status, 1)
        self.assertEqual(engine.order, [])
        summary = json.loads((self.out / "summary.json").read_text())
        self.assertEqual(summary["failure"]["type"], "InputError")

    def test_existing_summary_rejected(self):
        self.out.mkdir()
        (self.out / "summary.json").write_text("{}")
        with self.assertRaises(SystemExit) as exit_info, \
                redirect_stderr(io.StringIO()):
            self.run_main(FakeEngine(finish_plan=[]))
        self.assertEqual(exit_info.exception.code, 2)
        self.assertEqual((self.out / "summary.json").read_text(), "{}")


class ImportSafetyTest(unittest.TestCase):

    def test_import_starts_nothing(self):
        code = ("import sys, os; sys.path.insert(0, sys.argv[1]); "
                "before = set(os.listdir('.')); import run_lp_inference; "
                "assert 'ray' not in sys.modules, 'ray imported'; "
                "assert 'sarathi' not in sys.modules, 'sarathi imported'; "
                "assert 'torch' not in sys.modules, 'torch imported'; "
                "assert set(os.listdir('.')) == before, 'files created'; "
                "print('ok')")
        with tempfile.TemporaryDirectory() as cwd:
            result = subprocess.run(
                [sys.executable, "-B", "-c", code, str(SCRIPTS),
                 "--output-dir", "x"],
                cwd=cwd, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "ok")


if __name__ == "__main__":
    unittest.main()
