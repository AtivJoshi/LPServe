"""CPU checks for reference validation, token-history assembly, and token
comparison in the real-weight decode-preemption comparison driver. No model,
engine, Ray, or GPU is used."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import check_lp_scheduler_decode_preemption_reference_gpu as driver  # noqa: E402

PROVENANCE = {key: f"{key}-value" for key in driver.PROVENANCE_KEYS}
ENVIRONMENT = {key: f"{key}-value" for key in driver.ENVIRONMENT_KEYS}
SHARED = dict(asset=dict(snapshot_revision="fixture"), load_format="auto",
              dtype="float16", max_model_len=64, block_size=16,
              initialized_num_gpu_blocks=4,
              reference_max_tokens=dict(A=3, B=1), lp_max_tokens=dict(A=2, B=1))
EXPECTED = dict(provenance=PROVENANCE, environment=ENVIRONMENT, shared=SHARED)
SHARED_AFTER_INIT = dict(vocab_size=32000, eos_token_id=2)
TOKENS = dict(A=[403, 29871, 13], B=[29871])


def _reference(label):
    tokens = TOKENS[label]
    return dict(
        scheduler="vllm", request=label, passed=True, failure=None,
        scheduler_identity=dict(driver.REFERENCE_SCHEDULER_IDENTITY),
        prompt_token_ids=list(driver.PROMPT_TOKEN_IDS[label]),
        sampling=driver.sampling_record("vllm", label),
        real_weights=dict(requested_load_format="auto",
                          model_config_load_format="auto",
                          fingerprint_matches_file=True),
        shared=copy.deepcopy(SHARED),
        cache_capacity=dict(chosen_initialized_blocks=4,
                            central_manager_total_blocks=4,
                            worker=dict(cache_tensor_block_dims=[4])),
        provenance=dict(PROVENANCE, git_status_short="differs harmlessly"),
        environment=dict(ENVIRONMENT, cwd="differs harmlessly"),
        shared_after_init=dict(SHARED_AFTER_INIT),
        weight_fingerprint_sha256="fingerprint",
        generated_token_ids=list(tokens),
        final_output=dict(seq_id=0, finished=True, finish_reason="length",
                          token_ids=list(tokens),
                          prompt_token_ids=list(driver.PROMPT_TOKEN_IDS[label])),
    )


def _set(path, value):
    def perturb(reference):
        target = reference
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    return perturb


def _lp_steps(a=(403, 29871, 13), b=(29871,), prefill_samples=None):
    """Completed LP steps in the expected schedule. By default every prefill
    entry samples the token that the following decode samples, as native
    greedy completion can, so a duplicated prefill sample would show."""
    p = prefill_samples or dict(a1=a[0], b=b[0], a2=a[1])
    rows = [
        ([[0, 16]], 11), ([[0, 1]], p["a1"]), ([[0, 0]], a[0]),
        ([[1, 16]], 12), ([[1, 16]], 13), ([[1, 16]], p["b"]),
        ([[1, 0]], b[0]), ([[0, 16]], 14), ([[0, 2]], p["a2"]),
        ([[0, 0]], a[1]), ([[0, 0]], a[2]),
    ]
    return [dict(index=i, scheduled=s, sampler_outputs=[[s[0][0], t]])
            for i, (s, t) in enumerate(rows)]


class DriverImportTest(unittest.TestCase):
    def test_import_loads_no_engine_worker_ray_or_torch(self):
        code = (
            "import sys; sys.path.insert(0, 'scripts'); "
            "import check_lp_scheduler_decode_preemption_reference_gpu; "
            "print(sorted(m for m in ('torch', 'ray', 'safetensors', "
            "'sarathi.engine.base_llm_engine', 'sarathi.worker.base_worker') "
            "if m in sys.modules))"
        )
        result = subprocess.run(
            [sys.executable, "-B", "-c", code], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        )
        self.assertEqual(result.stdout.strip(), "[]")

    def test_reused_helper_constants_match_this_case(self):
        found = driver.check_helper_constants()
        self.assertEqual(
            found["check_lp_scheduler_preemption_reference_gpu.PROMPT_TOKEN_IDS"],
            driver.PROMPT_TOKEN_IDS,
        )
        self.assertEqual(
            found["check_lp_scheduler_preemption_gpu.INITIALIZED_NUM_GPU_BLOCKS"],
            4,
        )


class ReferenceValidationTest(unittest.TestCase):
    def test_caps_differ_for_a_only(self):
        self.assertEqual(driver.REFERENCE_MAX_TOKENS, dict(A=3, B=1))
        self.assertEqual(driver.LP_MAX_TOKENS, dict(A=2, B=1))
        self.assertEqual(driver.sampling_record("vllm", "A")["max_tokens"], 3)
        self.assertEqual(driver.sampling_record("lp", "A")["max_tokens"], 2)
        self.assertEqual(
            {k: v for k, v in driver.sampling_record("vllm", "A").items()
             if k != "max_tokens"},
            {k: v for k, v in driver.sampling_record("lp", "A").items()
             if k != "max_tokens"},
        )

    def test_compatible_passing_references_are_accepted(self):
        for label in ("A", "B"):
            with self.subTest(label=label):
                record = driver.validate_reference(_reference(label), label,
                                                   EXPECTED)
                self.assertEqual(record["generated_token_ids"], TOKENS[label])
                self.assertEqual(record["max_tokens"],
                                 driver.REFERENCE_MAX_TOKENS[label])

    def test_incompatible_references_are_rejected(self):
        two = TOKENS["A"][:2]
        cases = {
            "wrong_label": ("A", _set(["request"], "B")),
            "wrong_prompt": ("B", _set(["prompt_token_ids"],
                                       list(range(2000, 2047)))),
            "failed_reference": ("A", _set(["passed"], False)),
            "recorded_failure": ("A", _set(["failure"], dict(type="X"))),
            "dummy_weights": ("A", _set(
                ["real_weights", "model_config_load_format"], "dummy")),
            "unverified_weights": ("A", _set(
                ["real_weights", "fingerprint_matches_file"], False)),
            "different_asset": ("A", _set(["shared", "asset",
                                           "snapshot_revision"], "other")),
            "different_capacity": ("A", _set(
                ["cache_capacity", "chosen_initialized_blocks"], 5)),
            "different_scheduler": ("A", _set(
                ["scheduler_identity", "max_num_batched_tokens"], 32)),
            "different_code": ("A", _set(["provenance", "file_sha256"], "x")),
            "different_environment": ("B", _set(["environment", "hostname"],
                                                "x")),
            # A's reference must use its own 3-token cap, not LP A's 2.
            "a_reference_with_lp_cap": ("A", _set(
                ["sampling", "max_tokens"], 2)),
            "a_reference_two_tokens": ("A", lambda r: (
                r.update(generated_token_ids=list(two)),
                r["final_output"].update(token_ids=list(two)))),
            "b_reference_three_tokens": ("B", lambda r: (
                r.update(generated_token_ids=[1, 2, 3]),
                r["final_output"].update(token_ids=[1, 2, 3]))),
            "non_greedy": ("A", _set(["sampling", "temperature"], 1.0)),
            "non_integer_token": ("B", _set(["generated_token_ids"], ["7"])),
            "boolean_token": ("B", _set(["generated_token_ids"], [True])),
            "out_of_vocabulary": ("B", _set(["generated_token_ids"], [32000])),
            "final_differs_from_history": ("A", _set(
                ["final_output", "token_ids"], [403, 29871, 14])),
            "unfinished_output": ("A", _set(["final_output", "finished"],
                                            False)),
            "wrong_finish_reason": ("B", _set(["final_output", "finish_reason"],
                                              "stop")),
        }
        for name, (label, perturb) in cases.items():
            with self.subTest(name):
                reference = _reference(label)
                perturb(reference)
                with self.assertRaises(driver.CheckFailure):
                    driver.validate_reference(reference, label, EXPECTED)

    def test_load_references_requires_both_compatible_summaries(self):
        with tempfile.TemporaryDirectory() as root:
            for label in ("A", "B"):
                (Path(root) / label).mkdir()
                (Path(root) / label / "summary.json").write_text(
                    json.dumps(_reference(label)))
            references = driver.load_references(root, EXPECTED)
            self.assertEqual(
                {k: v["generated_token_ids"] for k, v in references.items()},
                TOKENS,
            )
            mismatch = _reference("B")
            mismatch["weight_fingerprint_sha256"] = "other"
            (Path(root) / "B" / "summary.json").write_text(json.dumps(mismatch))
            with self.assertRaisesRegex(driver.CheckFailure, "differ"):
                driver.load_references(root, EXPECTED)


class HistoryAssemblyTest(unittest.TestCase):
    LABELS = {0: "A", 1: "B"}

    def test_decode_events_only_in_completion_order(self):
        assembled = driver.assemble_history(_lp_steps(), self.LABELS)
        self.assertEqual(assembled["history"], TOKENS)
        self.assertEqual(
            assembled["decode_events"],
            [[2, "A", 403], [6, "B", 29871], [9, "A", 29871], [10, "A", 13]],
        )
        # Every prefill entry's sample is retained as discarded, once.
        self.assertEqual(
            [row[:3] for row in assembled["discarded_prefill_samples"]],
            [[0, "A", 16], [1, "A", 1], [3, "B", 16], [4, "B", 16],
             [5, "B", 16], [7, "A", 16], [8, "A", 2]],
        )

    def test_distinct_prefill_samples_never_enter_history(self):
        steps = _lp_steps(prefill_samples=dict(a1=900, b=901, a2=902))
        assembled = driver.assemble_history(steps, self.LABELS)
        self.assertEqual(assembled["history"], TOKENS)
        discarded = [row[3] for row in assembled["discarded_prefill_samples"]]
        self.assertTrue({900, 901, 902} <= set(discarded))

    def test_malformed_sampler_outputs_are_rejected(self):
        cases = {
            "missing": lambda s: s[2].update(sampler_outputs=[]),
            "wrong_seq": lambda s: s[2].update(sampler_outputs=[[1, 403]]),
            "extra": lambda s: s[2].update(
                sampler_outputs=[[0, 403], [0, 404]]),
            "non_integer": lambda s: s[2].update(sampler_outputs=[[0, "403"]]),
            "boolean": lambda s: s[2].update(sampler_outputs=[[0, True]]),
            "unknown_seq": lambda s: s[2].update(
                scheduled=[[5, 0]], sampler_outputs=[[5, 403]]),
        }
        for name, perturb in cases.items():
            with self.subTest(name):
                steps = _lp_steps()
                perturb(steps)
                with self.assertRaises(driver.CheckFailure):
                    driver.assemble_history(steps, self.LABELS)


class TokenComparisonTest(unittest.TestCase):
    REFERENCES = {label: dict(generated_token_ids=tokens)
                  for label, tokens in TOKENS.items()}

    def test_matching_histories(self):
        comparisons, equal = driver.compare_histories(self.REFERENCES,
                                                      dict(TOKENS))
        self.assertTrue(equal)
        self.assertEqual(set(comparisons), {"A", "B", "A_first_token"})
        self.assertTrue(all(c["equal"] for c in comparisons.values()))

    def test_mismatches_are_rejected(self):
        cases = {
            "a_last_token": (dict(A=[403, 29871, 14], B=[29871]),
                             {"A"}, 2),
            "a_first_token": (dict(A=[404, 29871, 13], B=[29871]),
                              {"A", "A_first_token"}, 0),
            "b_token": (dict(A=list(TOKENS["A"]), B=[99]), {"B"}, None),
            # A's native final output IDs alone (post-reset tokens only).
            "a_final_output_only": (dict(A=[29871, 13], B=[29871]),
                                    {"A", "A_first_token"}, 0),
            "a_missing": (dict(B=[29871]), {"A", "A_first_token"}, 0),
        }
        for name, (history, unequal, a_index) in cases.items():
            with self.subTest(name):
                comparisons, equal = driver.compare_histories(
                    self.REFERENCES, history)
                self.assertFalse(equal)
                self.assertEqual(
                    {k for k, c in comparisons.items() if not c["equal"]},
                    unequal,
                )
                if a_index is not None:
                    self.assertEqual(
                        comparisons["A"]["first_difference_index"], a_index,
                    )


if __name__ == "__main__":
    unittest.main()
