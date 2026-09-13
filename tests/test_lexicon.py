"""Offline lexical fixtures; none of these outputs measure Jev quality."""
from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import benchmark
from lexicon import WordLexicon
import typesafe_llm as decoder


class LexicalFixture:
    def __init__(self, base, *, character=None, fail_at=None, malformed=False):
        self.base = base
        self.character = character
        self.fail_at = fail_at
        self.malformed = malformed
        self.payloads = []

    def evaluate(self, payload):
        self.payloads.append(copy.deepcopy(payload))
        if len(self.payloads) == self.fail_at:
            raise decoder.APIError("offline final-selection failure")
        answers = {}
        for name, question in payload["questions"].items():
            options = question["criteria"]
            if name != decoder.QUESTION_ID:
                chosen = next(iter(options))
            elif payload["state"]["answer_prefix"] == "berlin":
                chosen = decoder.STOP
            elif self.character is not None:
                chosen = self.character
            else:
                chosen = next(label for label in options if label not in self.base)
            probabilities = {label: float(label == chosen) for label in options}
            if self.malformed and name != decoder.QUESTION_ID:
                probabilities.pop(chosen)
            answers[name] = {"type": "choice", "choice": chosen, "probabilities": probabilities}
        return decoder.APIResult({"model": "offline-lexical-fixture", "answers": answers,
                                  "usage": {"input_tokens": 7, "output_tokens": 2}})


class LexicalTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.base = decoder.make_vocabulary("ab ")
        path = self.root / "words.json"
        path.write_text(json.dumps({"words": ["berlin", "bermuda", "boat"]}), encoding="utf-8")
        self.lexicon = WordLexicon.load(path)
        self.config = {"model": "offline", "max_calls": 8, "max_new_chars": 20,
                       "seed": 7, "temperature": 0, "top_k": 0, "top_p": 1,
                       "shuffle_options": False, "verbose": False}

    def generate(self, client, *, lexicon=None, prefix="b", **limits):
        stdout = io.StringIO()
        summary = decoder.generate(client, prompt="Offline fixture instruction.", prefix=prefix,
                                   vocabulary=self.base, config={**self.config, **limits},
                                   run_dir=self.root / "run", stdout=stdout, stderr=io.StringIO(),
                                   context={"fixture": "original task data"},
                                   lexicon=self.lexicon if lexicon is None else lexicon)
        return summary, stdout.getvalue()

    def test_selected_word_conditions_next_request_and_both_stages_count(self):
        client = LexicalFixture(self.base)
        summary, output = self.generate(client)
        self.assertEqual(output, "berlin")
        self.assertEqual(summary["stop_reason"], "stop")
        self.assertEqual([p["state"]["answer_prefix"] for p in client.payloads], ["b", "b", "berlin"])
        self.assertEqual(summary["api_calls_started"], 3)
        self.assertEqual(summary["reported_token_totals"], {"input_tokens": 21, "output_tokens": 6})
        events = [json.loads(line) for line in (self.root / "run/trace.jsonl").read_text().splitlines()]
        selections = [event for event in events if event["event"] == "decision"]
        self.assertEqual(selections[0]["emitted_text"], "erlin")
        self.assertEqual(selections[0]["vocabulary"][selections[0]["selected_label"]], "erlin")

    def test_last_attempt_is_not_spent_on_unusable_proposals(self):
        client = LexicalFixture(self.base)
        summary, output = self.generate(client, max_calls=1)
        self.assertEqual(output, "b")
        self.assertEqual(summary["stop_reason"], "max_calls")
        self.assertEqual(client.payloads, [])

    def test_selection_failure_never_emits_a_proposed_word(self):
        client = LexicalFixture(self.base, fail_at=2)
        summary, output = self.generate(client)
        self.assertEqual(output, "b")
        self.assertEqual((self.root / "run/answer.txt").read_text(), "b")
        self.assertEqual(summary["stop_reason"], "error")
        self.assertEqual(summary["api_calls_started"], 2)
        self.assertEqual(summary["successful_responses"], 1)
        self.assertEqual(len(client.payloads), 2)

    def test_malformed_proposal_stops_before_final_selection(self):
        client = LexicalFixture(self.base, malformed=True)
        summary, output = self.generate(client)
        self.assertEqual(output, "b")
        self.assertEqual(summary["stop_reason"], "error")
        self.assertEqual(len(client.payloads), 1)

    def test_original_character_remains_selectable_even_when_word_is_proposed(self):
        client = LexicalFixture(self.base, character="a")
        summary, output = self.generate(client, max_calls=2)
        self.assertEqual(output, "ba")
        self.assertEqual(summary["new_characters"], 1)
        self.assertEqual(summary["api_calls_started"], 2)

    def test_single_completion_needs_no_ranking_request(self):
        path = self.root / "single.json"
        path.write_text(json.dumps({"words": ["berlin"]}), encoding="utf-8")
        client = LexicalFixture(self.base)
        summary, output = self.generate(client, lexicon=WordLexicon.load(path), max_calls=1)
        self.assertEqual(output, "berlin")
        self.assertEqual(summary["api_calls_started"], 1)
        self.assertEqual(summary["stop_reason"], "max_calls")

    def test_word_crossing_character_cap_is_not_partially_emitted(self):
        summary, output = self.generate(LexicalFixture(self.base), max_new_chars=3)
        self.assertEqual(output, "b")
        self.assertEqual(summary["stop_reason"], "max_new_chars")
        self.assertEqual(summary["new_tokens"], 0)

    def test_unknown_word_prefix_keeps_character_fallback(self):
        client = LexicalFixture(self.base, character="a")
        summary, output = self.generate(client, prefix="zzq", max_calls=1)
        self.assertEqual(output, "zzqa")
        self.assertEqual(summary["api_calls_started"], 1)

    def test_campaign_cap_covers_proposals_and_final_selection(self):
        suite_path = self.root / "suite.json"
        suite_path.write_text(json.dumps({"characters": "ab ", "cases": [
            {"id": "first", "prompt": "Offline fixture instruction.", "prefix": "b", "expected": ["berlin"]},
            {"id": "second", "prompt": "Another offline fixture.", "prefix": "b", "expected": ["berlin"]},
        ]}), encoding="utf-8")
        args = benchmark.parser().parse_args([str(suite_path), "--lexicon", self.lexicon.source,
                                             "--max-total-calls", "3"])
        instructions = benchmark.validate_options(args)
        cases = benchmark.load_suite(suite_path)
        plan = benchmark.make_plan(cases, args, instructions)
        client = LexicalFixture(self.base)
        results = benchmark.run_campaign(client, cases=cases, plan=plan,
                                         campaign_dir=self.root / "campaign", stderr=io.StringIO(),
                                         lexicon=args.word_lexicon)
        self.assertEqual(len(client.payloads), 3)
        self.assertEqual(results["aggregate"]["completed_exact_matches"], 1)
        self.assertEqual(results["aggregate"]["not_run"], 1)
        self.assertEqual(results["status"], "budget_exhausted")


if __name__ == "__main__":
    unittest.main()
