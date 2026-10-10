"""CPU checks for reference-summary validation and token comparison in the
real-weight preemption comparison driver. No model, engine, Ray, or GPU is
used."""

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

import check_lp_scheduler_preemption_reference_gpu as driver  # noqa: E402

PROVENANCE = {key: f"{key}-value" for key in driver.PROVENANCE_KEYS}
ENVIRONMENT = {key: f"{key}-value" for key in driver.ENVIRONMENT_KEYS}
SHARED = dict(asset=dict(snapshot_revision="fixture"), load_format="auto",
              dtype="float16", max_model_len=64, block_size=16,
              initialized_num_gpu_blocks=4)
EXPECTED = dict(provenance=PROVENANCE, environment=ENVIRONMENT, shared=SHARED)
SHARED_AFTER_INIT = dict(vocab_size=32000, eos_token_id=2)
TOKENS = dict(A=[11087], B=[13271])


def _reference(label):
    tokens = TOKENS[label]
    return dict(
        scheduler="vllm", request=label, passed=True, failure=None,
        scheduler_identity=dict(driver.REFERENCE_SCHEDULER_IDENTITY),
        prompt_token_ids=list(driver.PROMPT_TOKEN_IDS[label]),
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


class DriverImportTest(unittest.TestCase):
    def test_import_loads_no_engine_worker_ray_or_torch(self):
        code = (
            "import sys; sys.path.insert(0, 'scripts'); "
            "import check_lp_scheduler_preemption_reference_gpu; "
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
            found["check_lp_scheduler_preemption_gpu.DECODE_UTILITY"], 40.0,
        )
        self.assertEqual(
            found["check_lp_scheduler_reference_gpu.MODEL_REPO"],
            driver.MODEL_REPO,
        )


class ReferenceValidationTest(unittest.TestCase):
    def test_compatible_passing_references_are_accepted(self):
        for label in ("A", "B"):
            with self.subTest(label=label):
                record = driver.validate_reference(_reference(label), label,
                                                   EXPECTED)
                self.assertEqual(record["generated_token_ids"], TOKENS[label])
                self.assertEqual(record["label"], label)

    def test_incompatible_references_are_rejected(self):
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
            "different_configuration": ("B", _set(["shared", "max_model_len"],
                                                  32)),
            "different_capacity": ("A", _set(
                ["cache_capacity", "chosen_initialized_blocks"], 5)),
            "different_scheduler": ("A", _set(
                ["scheduler_identity", "max_num_batched_tokens"], 32)),
            "different_code": ("A", _set(["provenance", "file_sha256"], "x")),
            "different_environment": ("B", _set(["environment", "hostname"],
                                                "x")),
            "too_many_tokens": ("A", _set(["generated_token_ids"], [1, 2])),
            "non_integer_token": ("A", _set(["generated_token_ids"], ["7"])),
            "boolean_token": ("A", _set(["generated_token_ids"], [True])),
            "out_of_vocabulary": ("B", _set(["generated_token_ids"], [32000])),
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


class TokenComparisonTest(unittest.TestCase):
    def test_matching_and_mismatching_tokens(self):
        references = {label: dict(generated_token_ids=tokens)
                      for label, tokens in TOKENS.items()}
        comparisons, equal = driver.compare_lp_tokens(references, dict(TOKENS))
        self.assertTrue(equal)
        self.assertTrue(all(c["equal"] for c in comparisons.values()))

        comparisons, equal = driver.compare_lp_tokens(
            references, dict(A=[11087], B=[99]))
        self.assertFalse(equal)
        self.assertTrue(comparisons["A"]["equal"])
        self.assertFalse(comparisons["B"]["equal"])
        self.assertEqual(comparisons["B"]["first_difference_index"], 0)
        self.assertEqual(
            (comparisons["B"]["reference_token_at_difference"],
             comparisons["B"]["lp_token_at_difference"]), (13271, 99),
        )


if __name__ == "__main__":
    unittest.main()
