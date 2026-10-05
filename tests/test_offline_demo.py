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


if __name__ == "__main__":
    unittest.main()
