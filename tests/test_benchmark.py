"""Offline scripted fixtures only; these tests measure no real model quality."""
from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import benchmark
import typesafe_llm as decoder


class ScriptedEvaluator:
    """Offline fixture: return listed labels or raise the listed exception."""

    def __init__(self, steps):
        self.steps = iter(steps)
        self.payloads = []

    def evaluate(self, payload):
        self.payloads.append(copy.deepcopy(payload))
        step = next(self.steps)
        if isinstance(step, BaseException):
            raise step
        labels = payload["questions"][decoder.QUESTION_ID]["criteria"]
        return decoder.APIResult({
            "model": "offline-scripted-fixture-not-a-model-measurement",
            "usage": {"input_tokens": 5, "output_tokens": 1},
            "answers": {decoder.QUESTION_ID: {
                "type": "choice", "choice": step, "confidence": 1,
                "probabilities": {label: float(label == step) for label in labels},
            }},
        }, "offline-fixture-request", 0.0)


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.next_run = 0

    def campaign(self, evaluator, cases, *options):
        self.next_run += 1
        suite_path = self.root / f"suite-{self.next_run}.json"
        suite_path.write_text(json.dumps({"cases": cases}), encoding="utf-8")
        args = benchmark.parser().parse_args([str(suite_path), *options])
        instructions = benchmark.validate_options(args)
        validated = benchmark.load_suite(suite_path)
        plan = benchmark.make_plan(validated, args, instructions)
        path = self.root / f"campaign-{self.next_run}"
        result = benchmark.run_campaign(evaluator, cases=validated, plan=plan,
                                        campaign_dir=path, stderr=io.StringIO())
        return result, path

    @staticmethod
    def cases(count=1, expected=None, prefix=""):
        return [{"id": f"fixture-{index}", "prompt": "Offline fixture task.",
                 "prefix": prefix, "expected": ["a"] if expected is None else expected}
                for index in range(count)]

    def test_shared_cap_includes_partial_sample_across_repeats(self):
        evaluator = ScriptedEvaluator(["a", decoder.STOP, "a", "a"])
        result, path = self.campaign(evaluator, self.cases(2),
                                    "--repeats", "2", "--max-total-calls", "3")
        self.assertEqual(len(evaluator.payloads), 3)
        self.assertEqual(result["status"], "budget_exhausted")
        totals = result["aggregate"]
        self.assertEqual((totals["scheduled"], totals["completed"], totals["incomplete"], totals["not_run"]),
                         (4, 1, 1, 2))
        self.assertEqual(totals["api_calls_started"], 3)
        self.assertEqual(totals["completed_accuracy"], 0.25)
        persisted = json.loads((path / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["samples"][0]["output"], "a")
        self.assertEqual(persisted["samples"][1]["output"], "a")
        self.assertFalse(persisted["samples"][1]["completed_exact_match"])

    def test_failed_attempt_is_charged_and_no_later_case_runs(self):
        evaluator = ScriptedEvaluator(["a", decoder.STOP, "a", decoder.APIError("offline failure"), decoder.STOP])
        result, path = self.campaign(evaluator, self.cases(3), "--max-total-calls", "9")
        self.assertEqual(len(evaluator.payloads), 4)
        self.assertEqual(result["status"], "error")
        totals = result["aggregate"]
        self.assertEqual(totals["api_calls_started"], 4)
        self.assertEqual(totals["successful_responses"], 3)
        self.assertEqual(totals["not_run"], 1)
        self.assertEqual(totals["reported_token_totals"], {"input_tokens": 15, "output_tokens": 3})
        persisted = json.loads((path / "results.json").read_text(encoding="utf-8"))
        self.assertTrue(persisted["samples"][0]["completed_exact_match"])
        self.assertEqual(persisted["samples"][1]["output"], "a")
        self.assertTrue(persisted["samples"][1]["raw_exact_match"])
        self.assertFalse(persisted["samples"][1]["completed_exact_match"])

    def test_budget_fence_refuses_dispatch_after_limit_or_failure(self):
        evaluator = ScriptedEvaluator([decoder.STOP])
        budget = benchmark.SharedBudget(evaluator, 1)
        payload = decoder.make_payload("fixture", "a", decoder.make_vocabulary("a"), "offline")
        budget.evaluate(payload)
        with self.assertRaises(decoder.APIError):
            budget.evaluate(payload)
        self.assertEqual(len(evaluator.payloads), 1)
        failing = ScriptedEvaluator([decoder.APIError("offline failure"), decoder.STOP])
        budget = benchmark.SharedBudget(failing, 3)
        with self.assertRaises(decoder.APIError):
            budget.evaluate(payload)
        with self.assertRaises(decoder.APIError):
            budget.evaluate(payload)
        self.assertEqual((budget.calls, len(failing.payloads)), (1, 1))

    def test_interruption_preserves_completed_and_partial_outputs(self):
        evaluator = ScriptedEvaluator(["a", decoder.STOP, "a", KeyboardInterrupt(), decoder.STOP])
        result, path = self.campaign(evaluator, self.cases(3))
        self.assertEqual(result["status"], "interrupted")
        self.assertEqual(len(evaluator.payloads), 4)
        persisted = json.loads((path / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["aggregate"]["not_run"], 1)
        self.assertTrue(persisted["samples"][0]["completed_exact_match"])
        self.assertEqual(persisted["samples"][1]["output"], "a")
        self.assertFalse(persisted["samples"][1]["completed_exact_match"])

    def test_expected_answers_cannot_change_requests_or_generation_config(self):
        first = ScriptedEvaluator(["a", decoder.STOP])
        second = ScriptedEvaluator(["a", decoder.STOP])
        accepted, first_path = self.campaign(first, self.cases(expected=["a"]))
        rejected, second_path = self.campaign(second, self.cases(expected=["scoring-only secret"]))
        self.assertEqual(first.payloads, second.payloads)
        first_config = json.loads((first_path / "samples/000001/config.json").read_text(encoding="utf-8"))
        second_config = json.loads((second_path / "samples/000001/config.json").read_text(encoding="utf-8"))
        self.assertEqual(first_config, second_config)
        self.assertNotIn("expected", second_config)
        manifest = (second_path / "manifest.json").read_text(encoding="utf-8")
        self.assertNotIn("scoring-only secret", manifest)
        self.assertEqual(accepted["aggregate"]["completed_exact_matches"], 1)
        self.assertEqual(rejected["aggregate"]["completed_exact_matches"], 0)

    def test_character_truncation_never_counts_as_completed_match(self):
        evaluator = ScriptedEvaluator(["a", decoder.STOP])
        result, _ = self.campaign(evaluator, self.cases(expected=["Pa"], prefix="P"),
                                  "--max-new-chars", "1")
        sample = result["samples"][0]
        self.assertEqual(sample["output"], "Pa")
        self.assertTrue(sample["raw_exact_match"])
        self.assertFalse(sample["completed_exact_match"])
        self.assertEqual(result["aggregate"]["completed_accuracy"], 0)
        self.assertEqual(result["aggregate"]["completed"], 0)
        self.assertEqual(len(evaluator.payloads), 1)

    def test_completed_match_includes_prefix_and_does_not_normalize(self):
        evaluator = ScriptedEvaluator(["a", decoder.STOP])
        result, _ = self.campaign(evaluator, self.cases(expected=["pa", "a", "Pa "], prefix="P"))
        sample = result["samples"][0]
        self.assertEqual(sample["output"], "Pa")
        self.assertTrue(sample["completed"])
        self.assertFalse(sample["raw_exact_match"])
        self.assertFalse(sample["completed_exact_match"])


if __name__ == "__main__":
    unittest.main()
