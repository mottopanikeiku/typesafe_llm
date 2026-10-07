"""Offline policy comparisons and an independent tiny exhaustive reference."""
from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import offline_demo as demo


def completed_scores() -> dict[str, float]:
    """Enumerate all positive paths in the finite table, with no beam pruning."""
    scores = {}

    def visit(prefix: str, log_probability: float, actions: int) -> None:
        probabilities = demo.SCORES[prefix]
        total = math.fsum(probabilities.values())
        for label, token in demo.VOCABULARY.items():
            probability = probabilities[label]
            if probability == 0:
                continue
            log_score = log_probability + math.log(probability) - math.log(total)
            if token is None:
                scores[prefix] = log_score / (actions + 1)
            else:
                visit(prefix + token, log_score, actions + 1)

    visit("", 0.0, 0)
    return scores


class OfflineDemoTests(unittest.TestCase):
    def test_real_generator_compares_policies_without_an_http_client(self):
        with patch.object(demo.decoder, "TypeSafeHTTP", side_effect=AssertionError("No network client")):
            rows = demo.compare()
        self.assertEqual([row["output"] for row in rows], ["a", "aa", "a", "b"])
        self.assertEqual([row["evaluated_prefixes"] for row in rows], [2, 3, 2, 4])
        self.assertTrue(all(row["completed"] for row in rows))
        self.assertEqual(rows[2]["search_stop_reason"], "max_calls")
        self.assertEqual(rows[3]["search_stop_reason"], "frontier_exhausted")
        for row in rows:
            self.assertLessEqual(row["evaluated_prefixes"], row["evaluation_budget"])

    def test_full_beam_matches_exhaustive_reference_and_narrow_beam_does_not(self):
        scores = completed_scores()
        self.assertEqual(set(scores), {"a", "aa", "b"})
        self.assertAlmostEqual(scores["a"], math.log(0.6 * 0.6) / 2)
        self.assertAlmostEqual(scores["aa"], math.log(0.6 * 0.4) / 3)
        self.assertAlmostEqual(scores["b"], math.log(0.4) / 2)
        self.assertGreater(scores["aa"], scores["a"])
        self.assertEqual(max(scores, key=scores.get), "b")
        rows = demo.compare()
        for row in rows:
            self.assertAlmostEqual(row["mean_log_score"], scores[row["output"]])
        self.assertEqual(rows[-1]["output"], max(scores, key=scores.get))
        self.assertLess(rows[1]["mean_log_score"], rows[-1]["mean_log_score"])


class ExhaustiveFixtureTests(unittest.TestCase):
    def test_full_search_matches_every_finite_reference_without_http(self):
        with patch.object(demo.decoder, "TypeSafeHTTP", side_effect=AssertionError("No network client")):
            result = demo.exhaustive_comparison()
        self.assertEqual(len(result["fixtures"]), 6)
        self.assertEqual(
            [len(fixture["completed_paths"]) for fixture in result["fixtures"]],
            [3, 6, 6, 1, 2, 4],
        )
        for fixture in result["fixtures"]:
            with self.subTest(fixture=fixture["name"]):
                self.assertTrue(fixture["comparisons"][-1]["matches_exhaustive_score"])
                for row in fixture["comparisons"]:
                    self.assertLessEqual(row["evaluated_prefixes"], row["evaluation_budget"])
                    if row["completed"]:
                        paths = [path for path in fixture["completed_paths"]
                                 if path["labels"] == row["selected_labels"]]
                        self.assertEqual(len(paths), 1)
                        self.assertEqual(row["output"], paths[0]["output"])
                        self.assertAlmostEqual(row["mean_log_score"], paths[0]["mean_log_score"])

    def test_diversity_can_help_and_hurt_at_the_same_width_and_call_count(self):
        result = demo.exhaustive_comparison()
        helpful, harmful = result["fixtures"][1:3]
        standard, diverse = helpful["comparisons"][1:3]
        self.assertEqual([standard["output"], diverse["output"]], ["ba", "ac"])
        self.assertFalse(standard["matches_exhaustive_score"])
        self.assertTrue(diverse["matches_exhaustive_score"])
        standard, diverse = harmful["comparisons"][1:3]
        self.assertEqual([standard["output"], diverse["output"]], ["aa", "ba"])
        self.assertTrue(standard["matches_exhaustive_score"])
        self.assertFalse(diverse["matches_exhaustive_score"])
        for fixture in (helpful, harmful):
            self.assertEqual([row["evaluated_prefixes"] for row in fixture["comparisons"][1:3]], [5, 5])

    def test_alternate_tokenizations_are_not_collapsed_by_text(self):
        fixture = demo.FIXTURES[4]
        paths = demo.exhaustive_paths(fixture)
        self.assertEqual([path["output"] for path in paths], ["ab", "ab"])
        self.assertEqual([path["actions_including_stop"] for path in paths], [3, 2])
        self.assertAlmostEqual(paths[0]["mean_log_score"], math.log(.45) / 3)
        self.assertAlmostEqual(paths[1]["mean_log_score"], math.log(.55) / 2)
        self.assertLess(paths[0]["log_probability"], paths[1]["log_probability"])
        self.assertGreater(paths[0]["mean_log_score"], paths[1]["mean_log_score"])
        row = demo.run_policy(fixture, "unpruned", "beam", 3, 3)
        self.assertEqual(row["selected_labels"], ["a", "b", "STOP"])
        self.assertEqual(row["evaluated_prefixes"], 3)
        self.assertEqual(row["cache_hits"], 1)

    def test_equal_depth_collision_retains_higher_likelihood_path(self):
        fixture = demo.FIXTURES[5]
        paths = demo.exhaustive_paths(fixture)
        self.assertEqual(len([path for path in paths if path["output"] == "abc"]), 2)
        row = demo.run_policy(fixture, "unpruned", "beam", 4, 4)
        self.assertEqual(row["selected_labels"], ["TOKEN_0003", "c", "STOP"])
        self.assertAlmostEqual(row["mean_log_score"], math.log(.5 * .9) / 3)
        self.assertEqual(row["evaluated_prefixes"], 4)

    def test_character_cap_does_not_renormalize_excluded_edges(self):
        fixture = {
            "vocabulary": {"A": "a", "LONG": "ab", "STOP": None},
            "scores": {
                "": {"A": .2, "LONG": .6, "STOP": .2},
                "a": {"A": .8, "LONG": 0, "STOP": .2},
            },
            "max_new_chars": 1,
        }
        paths = demo.exhaustive_paths(fixture)
        self.assertEqual([path["output"] for path in paths], ["a", ""])
        self.assertAlmostEqual(paths[0]["mean_log_score"], math.log(.2 * .2) / 2)
        self.assertAlmostEqual(paths[1]["mean_log_score"], math.log(.2))
        row = demo.run_policy(fixture, "unpruned", "beam", 2, 2)
        self.assertAlmostEqual(row["mean_log_score"], math.log(.2))

    def test_cap_does_not_count_an_unfinished_path_as_optimal(self):
        fixture = demo.exhaustive_comparison()["fixtures"][3]
        capped = fixture["comparisons"][3]
        self.assertEqual(capped["output"], "ab")
        self.assertFalse(capped["completed"])
        self.assertFalse(capped["matches_exhaustive_score"])

    def test_checked_in_synthetic_results_are_reproducible(self):
        import json
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "results" / "synthetic-search.json"
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), demo.exhaustive_comparison())


if __name__ == "__main__":
    unittest.main()
