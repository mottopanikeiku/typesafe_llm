"""Finite score-table regressions, not measurements of language-model quality."""
from __future__ import annotations

import math
import unittest

from search import BeamSearch, PrefixEvaluation


class BeamSearchTests(unittest.TestCase):
    def observe(self, search, probabilities, call):
        evaluation = PrefixEvaluation(
            call=call,
            api_choice=max(probabilities, key=probabilities.get),
            probabilities=probabilities,
        )
        search.observe(evaluation)
        return evaluation

    def finish(self, search, table):
        visited = []
        for call in range(1, 100):
            prefix = search.pending_prefix
            if prefix is None:
                return search.result(), visited
            self.assertIn(prefix, table, f"Unexpected frontier prefix: {prefix!r}")
            visited.append(prefix)
            self.observe(search, table[prefix], call)
        self.fail("Finite score table did not exhaust its frontier")

    def test_wider_beam_recovers_lower_first_step_completion(self):
        vocabulary = {"A": "a", "B": "b", "STOP": None}
        table = {
            "": {"A": 0.6, "B": 0.4, "STOP": 0.0},
            "a": {"A": 0.9, "B": 0.0, "STOP": 0.1},
            "b": {"A": 0.0, "B": 0.0, "STOP": 1.0},
        }
        narrow, narrow_visits = self.finish(BeamSearch("", vocabulary, 1, 1), table)
        wide, wide_visits = self.finish(BeamSearch("", vocabulary, 2, 1), table)

        self.assertTrue(narrow.completed)
        self.assertEqual(narrow.text, "a")
        self.assertNotIn("b", narrow_visits)
        self.assertTrue(wide.completed)
        self.assertEqual(wide.text, "b")
        self.assertIn("b", wide_visits)
        self.assertEqual([step.label for step in wide.steps], ["B", "STOP"])
        self.assertAlmostEqual(wide.log_probability, math.log(0.4))
        self.assertAlmostEqual(wide.score, math.log(0.4) / 2)
        self.assertGreater(wide.score, narrow.score)

    def test_token_diversity_recovers_completion_when_one_ending_crowds_the_beam(self):
        vocabulary = {"A": "a", "B": "b", "C": "c", "STOP": None}
        table = {
            "": {"A": .6, "B": .4, "C": 0, "STOP": 0},
            "a": {"A": .5, "B": .05, "C": .45, "STOP": 0},
            "b": {"A": .8, "B": .01, "C": .19, "STOP": 0},
            "aa": {"A": .99, "B": 0, "C": 0, "STOP": .01},
            "ba": {"A": .99, "B": 0, "C": 0, "STOP": .01},
            "ac": {"A": 0, "B": 0, "C": 0, "STOP": 1},
        }
        standard, _ = self.finish(BeamSearch("", vocabulary, 2, 2, diversity="none"), table)
        diverse, _ = self.finish(BeamSearch("", vocabulary, 2, 2, diversity="token"), table)
        self.assertEqual(standard.text, "ba")
        self.assertTrue(diverse.completed)
        self.assertEqual(diverse.text, "ac")
        self.assertAlmostEqual(diverse.score, math.log(.6 * .45) / 3)

    def test_immediate_stop_competes_with_later_completions(self):
        vocabulary = {"A": "a", "STOP": None}
        result, visited = self.finish(BeamSearch("", vocabulary, 1, 1), {
            "": {"A": 0.6, "STOP": 0.4},
            "a": {"A": 0.9, "STOP": 0.1},
        })

        self.assertEqual(visited, ["", "a"])
        self.assertTrue(result.completed)
        self.assertEqual(result.text, "")
        self.assertEqual([step.label for step in result.steps], ["STOP"])
        self.assertEqual(result.steps[0].prefix_before, "")
        self.assertEqual(result.steps[0].prefix_after, "")
        self.assertAlmostEqual(result.score, math.log(0.4))

    def test_zero_character_allowance_still_requires_a_stop_evaluation(self):
        search = BeamSearch("fixed", {"A": "a", "STOP": None}, 1, 0)
        before = search.result()
        self.assertEqual(search.pending_prefix, "fixed")
        self.assertFalse(before.completed)
        self.assertEqual(before.text, "fixed")
        self.assertEqual(before.steps, ())
        self.assertIsNone(before.score)

        self.observe(search, {"A": 0.8, "STOP": 0.2}, 1)
        result = search.result()
        self.assertIsNone(search.pending_prefix)
        self.assertTrue(result.completed)
        self.assertEqual(result.text, "fixed")
        self.assertEqual([step.label for step in result.steps], ["STOP"])
        self.assertAlmostEqual(result.log_probability, math.log(0.2))

    def test_mid_layer_result_preserves_successfully_discovered_child(self):
        search = BeamSearch("", {"A": "a", "B": "b", "STOP": None}, 2, 2)
        self.observe(search, {"A": 0.6, "B": 0.4, "STOP": 0.0}, 1)
        self.assertEqual(search.depth, 1)
        self.assertEqual(search.pending_prefix, "a")
        child_evaluation = self.observe(search, {"A": 0.0, "B": 1.0, "STOP": 0.0}, 2)

        self.assertEqual(search.pending_prefix, "b")
        self.assertEqual(search.depth, 1)
        result = search.result()
        self.assertFalse(result.completed)
        self.assertEqual(result.text, "ab")
        self.assertEqual([step.label for step in result.steps], ["A", "B"])
        self.assertEqual(result.steps[-1].evaluation, child_evaluation)
        self.assertAlmostEqual(result.score, math.log(0.6) / 2)

    def test_mid_layer_result_preserves_completion_before_other_parent_runs(self):
        search = BeamSearch("", {"A": "a", "B": "b", "STOP": None}, 2, 2)
        self.observe(search, {"A": 0.6, "B": 0.4, "STOP": 0.0}, 1)
        completion_evaluation = self.observe(
            search, {"A": 0.0, "B": 0.8, "STOP": 0.2}, 2,
        )

        self.assertEqual(search.pending_prefix, "b")
        result = search.result()
        self.assertTrue(result.completed)
        self.assertEqual(result.text, "a")
        self.assertEqual([step.label for step in result.steps], ["A", "STOP"])
        self.assertEqual(result.steps[-1].evaluation, completion_evaluation)
        self.assertAlmostEqual(result.score, math.log(0.6 * 0.2) / 2)

    def test_initial_prefix_and_fragment_cap_are_not_repaired_or_truncated(self):
        vocabulary = {"SHORT": "x", "LONG": "yz", "STOP": None}
        search = BeamSearch("seed:", vocabulary, 2, 2)
        result, visited = self.finish(search, {
            "seed:": {"SHORT": 0.25, "LONG": 0.75, "STOP": 0.0},
            "seed:x": {"SHORT": 0.0, "LONG": 0.9, "STOP": 0.1},
            "seed:yz": {"SHORT": 0.6, "LONG": 0.0, "STOP": 0.4},
        })

        self.assertCountEqual(visited, ["seed:", "seed:x", "seed:yz"])
        self.assertTrue(result.completed)
        self.assertEqual(result.text, "seed:yz")
        self.assertEqual(
            [(step.prefix_before, step.prefix_after, step.label) for step in result.steps],
            [("seed:", "seed:yz", "LONG"), ("seed:yz", "seed:yz", "STOP")],
        )
        self.assertAlmostEqual(result.score, math.log(0.75 * 0.4) / 2)
        snapshot = search.snapshot()
        self.assertEqual(snapshot["active"], [])
        self.assertEqual(snapshot["completed"]["prefix"], "seed:yz")
        self.assertEqual(snapshot["completed"]["new_tokens"], 2)

    def test_duplicate_text_keeps_best_tokenization_and_stable_tie_provenance(self):
        vocabulary = {"A": "a", "AB": "ab", "BC": "bc", "C": "c", "STOP": None}
        for second_probability, expected_labels, expected_call in [
            (0.9, ["AB", "C", "STOP"], 3),
            (0.8, ["A", "BC", "STOP"], 2),
        ]:
            with self.subTest(second_probability=second_probability):
                search = BeamSearch("", vocabulary, 3, 3)
                self.observe(search, {"A": 0.5, "AB": 0.5, "BC": 0.0, "C": 0.0, "STOP": 0.0}, 1)
                self.assertEqual(search.pending_prefix, "a")
                self.observe(search, {"A": 0.0, "AB": 0.0, "BC": 0.8, "C": 0.0, "STOP": 0.2}, 2)
                self.assertEqual(search.pending_prefix, "ab")
                self.observe(search, {
                    "A": 0.0, "AB": 0.0, "BC": 0.0,
                    "C": second_probability, "STOP": 1.0 - second_probability,
                }, 3)

                self.assertEqual(search.depth, 2)
                self.assertEqual([node["prefix"] for node in search.snapshot()["active"]], ["abc"])
                self.assertEqual(search.pending_prefix, "abc")
                self.observe(search, {"A": 0.0, "AB": 0.0, "BC": 0.0, "C": 0.0, "STOP": 1.0}, 4)
                self.assertIsNone(search.pending_prefix)
                result = search.result()
                self.assertTrue(result.completed)
                self.assertEqual(result.text, "abc")
                self.assertEqual([step.label for step in result.steps], expected_labels)
                self.assertEqual(result.steps[1].evaluation.call, expected_call)
                self.assertAlmostEqual(result.log_probability, math.log(0.5 * second_probability))
                self.assertAlmostEqual(result.score, math.log(0.5 * second_probability) / 3)

    def test_rounded_distributions_are_normalized_without_mutating_raw_scores(self):
        vocabulary = {"A": "a", "B": "b", "STOP": None}
        root_scores = {"A": 0.667, "B": 0.0, "STOP": 0.332}
        end_scores = {"A": 0.333, "B": 0.0, "STOP": 0.666}
        original_root, original_end = dict(root_scores), dict(end_scores)
        result, visited = self.finish(BeamSearch("", vocabulary, 2, 1), {
            "": root_scores,
            "a": end_scores,
        })

        self.assertEqual(visited, ["", "a"])
        self.assertTrue(result.completed)
        self.assertEqual(result.text, "a")
        expected_log = math.log(0.667 / 0.999) + math.log(0.666 / 0.999)
        self.assertAlmostEqual(result.log_probability, expected_log)
        self.assertAlmostEqual(result.score, expected_log / 2)
        self.assertEqual(root_scores, original_root)
        self.assertEqual(end_scores, original_end)

    def test_zero_probability_stop_does_not_complete_a_capped_path(self):
        vocabulary = {"A": "a", "B": "b", "STOP": None}
        search = BeamSearch("", vocabulary, 3, 1)
        result, visited = self.finish(search, {
            "": {"A": 1.0, "B": 0.0, "STOP": 0.0},
            "a": {"A": 1.0, "B": 0.0, "STOP": 0.0},
        })

        self.assertEqual(visited, ["", "a"])
        self.assertFalse(result.completed)
        self.assertEqual(result.text, "a")
        self.assertEqual([step.label for step in result.steps], ["A"])
        self.assertEqual(result.log_probability, 0.0)
        self.assertEqual(result.score, 0.0)
        self.assertIsNone(search.snapshot()["completed"])

    def test_equal_scores_follow_vocabulary_order_not_label_or_text_order(self):
        vocabulary = {"B": "b", "A": "a", "STOP": None}
        table = {
            "": {"B": 0.5, "A": 0.5, "STOP": 0.0},
            "b": {"B": 0.0, "A": 0.0, "STOP": 1.0},
            "a": {"B": 0.0, "A": 0.0, "STOP": 1.0},
        }
        for width in (1, 2):
            with self.subTest(width=width):
                result, visited = self.finish(BeamSearch("", vocabulary, width, 1), table)
                self.assertEqual(visited, ["", "b"] if width == 1 else ["", "b", "a"])
                self.assertTrue(result.completed)
                self.assertEqual(result.text, "b")
                self.assertEqual([step.label for step in result.steps], ["B", "STOP"])


if __name__ == "__main__":
    unittest.main()
