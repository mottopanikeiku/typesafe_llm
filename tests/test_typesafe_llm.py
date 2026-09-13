"""Offline tests only. Fake model outputs are NOT measurements of Jev."""
from __future__ import annotations

import copy
import io
import json
import math
import random
from http.client import IncompleteRead
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import typesafe_llm as m


class FakeEvaluator:
    """Deterministically spell a supplied fixture, not an actual language model."""

    def __init__(self, target="cold", fail_at=None, interrupt_at=None, missing_usage=False):
        self.target = target
        self.fail_at = fail_at
        self.interrupt_at = interrupt_at
        self.missing_usage = missing_usage
        self.payloads = []

    def evaluate(self, payload):
        self.payloads.append(copy.deepcopy(payload))
        if len(self.payloads) == self.fail_at:
            raise m.APIError("Synthetic API failure")
        if len(self.payloads) == self.interrupt_at:
            raise KeyboardInterrupt()
        prefix = payload["state"]["answer_prefix"]
        char = self.target[len(prefix)] if len(prefix) < len(self.target) else None
        label = next(label for label, c in m.make_vocabulary(m.ALPHABETS["lower"]).items() if c == char)
        probabilities = {key: float(key == label) for key in payload["questions"][m.QUESTION_ID]["criteria"]}
        body = {
            "model": "fake-model-for-offline-tests",
            "answers": {m.QUESTION_ID: {
                "type": "choice", "choice": label, "probabilities": probabilities, "confidence": 1,
            }},
        }
        if not self.missing_usage:
            body["usage"] = {"input_tokens": 100, "output_tokens": 10}
        return m.APIResult(body, f"fake-request-{len(self.payloads)}", 1.0)


class TokenEvaluator:
    """Scripted text fragments, not live Jev quality evidence."""

    def __init__(self, vocabulary, pieces):
        self.vocabulary = vocabulary
        self.pieces = iter(pieces)
        self.prefixes = []

    def evaluate(self, payload):
        self.prefixes.append(payload["state"]["answer_prefix"])
        piece = next(self.pieces)
        label = next(key for key, value in self.vocabulary.items() if value == piece)
        return m.APIResult({"model": "offline-token-fixture", "answers": {
            m.QUESTION_ID: {"type": "choice", "choice": label,
                            "probabilities": {key: float(key == label) for key in self.vocabulary}},
        }})


class DecoderTests(unittest.TestCase):
    def setUp(self):
        self.vocab = m.make_vocabulary(m.ALPHABETS["lower"])
        self.config = {
            "model": "jev-1.13.0", "max_calls": 20, "max_new_chars": 20,
            "seed": 7, "temperature": 0, "top_k": 0, "top_p": 1,
            "shuffle_options": False, "verbose": False,
        }
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "run"
        self.stdout, self.stderr = io.StringIO(), io.StringIO()

    def run_fake(self, fake=None, prefix="", **overrides):
        return m.generate(fake or FakeEvaluator(), prompt="test prompt", prefix=prefix,
                          vocabulary=self.vocab, config={**self.config, **overrides}, run_dir=self.path,
                          stdout=self.stdout, stderr=self.stderr)

    def test_lower_alphabet(self):
        self.assertEqual(len(self.vocab), 29)
        self.assertEqual(self.vocab["SPACE"], " ")
        self.assertIsNone(self.vocab["STOP"])

    def test_ascii_alphabet(self):
        vocab = m.make_vocabulary(m.ALPHABETS["ascii"])
        self.assertEqual(len(vocab), 97)
        self.assertEqual(vocab["NEWLINE"], "\n")
        self.assertEqual(vocab["U005C"], "\\")

    def test_deduplicate_characters(self):
        self.assertEqual(len(m.make_vocabulary("aaab")), 3)

    def test_non_ascii_characters(self):
        vocab = m.make_vocabulary("éλ😀")
        self.assertEqual(vocab["U1F600"], "😀")

    def test_empty_characters_rejected(self):
        with self.assertRaises(ValueError):
            m.make_vocabulary("")

    def test_unsafe_control_characters_rejected(self):
        with self.assertRaises(ValueError):
            m.make_vocabulary("a\x1b")

    def test_duplicate_fragments_do_not_get_extra_probability_slots(self):
        vocabulary = m.make_vocabulary("ab", ["ab", "ab", "a", "STOP"])
        self.assertEqual(list(vocabulary.values()), ["a", "b", "ab", "STOP", None])

    def test_invalid_fragments_rejected_before_generation(self):
        for tokens in ["word", [""], [None], ["a\x1b[31m"], ["\ud800"]]:
            with self.subTest(tokens=repr(tokens)), self.assertRaises(ValueError):
                m.make_vocabulary("ab", tokens)

    def test_choice_limit_counts_stop_and_deduplicates_before_checking(self):
        characters = "".join(chr(0x4E00 + index) for index in range(254))
        vocabulary = m.make_vocabulary(characters, [characters[0]])
        self.assertEqual(len(vocabulary), 255)
        with self.assertRaises(ValueError):
            m.make_vocabulary(characters, ["extra fragment"])

    def test_fragments_condition_each_subsequent_step(self):
        self.vocab = m.make_vocabulary("abc ", ["ab", " c"])
        fake = TokenEvaluator(self.vocab, ["ab", " c", None])
        summary = self.run_fake(fake)
        self.assertEqual(self.stdout.getvalue(), "ab c")
        self.assertEqual(fake.prefixes, ["", "ab", "ab c"])
        self.assertEqual(summary["new_tokens"], 2)
        self.assertEqual(summary["new_characters"], 4)
        self.assertEqual(summary["api_calls_started"], 3)
        self.assertEqual(summary["stop_reason"], "stop")

    def test_character_limit_never_splits_or_resamples_a_fragment(self):
        self.vocab = m.make_vocabulary("abc ", ["ab", " c"])
        summary = self.run_fake(TokenEvaluator(self.vocab, ["ab", " c"]), max_new_chars=3)
        self.assertEqual((self.path / "answer.txt").read_text(), "ab")
        self.assertEqual(summary["new_tokens"], 1)
        self.assertEqual(summary["stop_reason"], "max_new_chars")
        events = [json.loads(line) for line in (self.path / "trace.jsonl").read_text().splitlines()]
        last = [event for event in events if event["event"] == "decision"][-1]
        self.assertEqual(last["selected_text"], " c")
        self.assertEqual(last["emitted_text"], "")
        self.assertEqual(last["prefix_after"], "ab")

    def test_order_shuffle_preserves_options(self):
        ordered = m.make_payload("q", "", self.vocab, "m")
        shuffled = m.make_payload("q", "", self.vocab, "m", order_rng=random.Random(7))
        a, b = [v["questions"][m.QUESTION_ID]["criteria"] for v in (ordered, shuffled)]
        self.assertEqual(a, b)
        self.assertNotEqual(list(a), list(b))

    def test_greedy(self):
        chosen, policy = m.choose_label({"a": .7, "b": .3}, "b", 0, 0, 1, random.Random(1))
        self.assertEqual(chosen, "a")
        self.assertEqual(policy, {"a": 1})

    def test_greedy_tie_uses_server_choice(self):
        chosen, _ = m.choose_label({"a": .5, "b": .5}, "b", 0, 0, 1, random.Random(1))
        self.assertEqual(chosen, "b")

    def test_top_k_one(self):
        chosen, policy = m.choose_label({"a": .7, "b": .3}, "a", 1, 1, 1, random.Random(1))
        self.assertEqual(chosen, "a")
        self.assertEqual(policy, {"a": 1})

    def test_nucleus_cutoff(self):
        _, policy = m.choose_label({"a": .7, "b": .2, "c": .1}, "a", 1, 0, .6, random.Random(1))
        self.assertEqual(policy, {"a": 1})

    def test_temperature_one_normalizes_rounding(self):
        _, policy = m.choose_label({"a": .7, "b": .29}, "a", 1, 0, 1, random.Random(1))
        self.assertAlmostEqual(policy["a"], .7 / .99)
        self.assertAlmostEqual(sum(policy.values()), 1)

    def test_temperature_sqrt(self):
        _, policy = m.choose_label({"a": .8, "b": .2}, "a", 2, 0, 1, random.Random(1))
        self.assertAlmostEqual(policy["a"], 2 / 3)

    def test_extremely_small_temperature_stable(self):
        _, policy = m.choose_label({"a": .8, "b": .2}, "a", 1e-300, 0, 1, random.Random(1))
        self.assertEqual(policy, {"a": 1})

    def test_zero_probability_not_sampled(self):
        _, policy = m.choose_label({"a": 1, "b": 0}, "a", 1, 0, 1, random.Random(1))
        self.assertNotIn("b", policy)

    def test_sampling_seed_reproducible(self):
        def sequence():
            rng = random.Random(42)
            return [m.choose_label({"a": .5, "b": .5}, "a", 1, 0, 1, rng)[0] for _ in range(20)]
        self.assertEqual(sequence(), sequence())

    def test_invalid_decoding_settings(self):
        for temperature, top_k, top_p in [(float("nan"), 0, 1), (-1, 0, 1), (1, -1, 1), (1, 0, 0)]:
            with self.subTest(values=(temperature, top_k, top_p)), self.assertRaises(ValueError):
                m.choose_label({"a": 1}, "a", temperature, top_k, top_p, random.Random(1))

    def test_invalid_response(self):
        with self.assertRaises(m.APIError):
            m.read_choice({}, self.vocab)

    def test_missing_probability_label_rejected(self):
        result = FakeEvaluator().evaluate(m.make_payload("q", "", self.vocab, "m"))
        del result.body["answers"][m.QUESTION_ID]["probabilities"]["a"]
        with self.assertRaises(m.APIError):
            m.read_choice(result.body, self.vocab)

    def test_invalid_probability_values_rejected(self):
        for value in [True, -0.1, 1.1, float("nan"), float("inf"), "0.5"]:
            with self.subTest(value=value):
                result = FakeEvaluator().evaluate(m.make_payload("q", "", self.vocab, "m"))
                result.body["answers"][m.QUESTION_ID]["probabilities"]["a"] = value
                with self.assertRaises(m.APIError):
                    m.read_choice(result.body, self.vocab)

    def test_all_zero_probabilities_rejected(self):
        result = FakeEvaluator().evaluate(m.make_payload("q", "", self.vocab, "m"))
        result.body["answers"][m.QUESTION_ID]["probabilities"] = dict.fromkeys(self.vocab, 0)
        with self.assertRaises(m.APIError):
            m.read_choice(result.body, self.vocab)

    def test_complete_generation_and_stop(self):
        fake = FakeEvaluator()
        summary = self.run_fake(fake)
        self.assertEqual(self.stdout.getvalue(), "cold")
        self.assertEqual((self.path / "answer.txt").read_text(), "cold")
        self.assertEqual(summary["stop_reason"], "stop")
        self.assertEqual(summary["api_calls_started"], 5)
        self.assertEqual(summary["reported_token_totals"]["input_tokens"], 500)
        prefixes = [p["state"]["answer_prefix"] for p in fake.payloads]
        self.assertEqual(prefixes, ["", "c", "co", "col", "cold"])

    def test_request_cap(self):
        summary = self.run_fake(max_calls=2)
        self.assertEqual(summary["api_calls_started"], 2)
        self.assertEqual(summary["stop_reason"], "max_calls")
        self.assertEqual(self.stdout.getvalue(), "co")

    def test_character_cap(self):
        summary = self.run_fake(max_new_chars=2)
        self.assertEqual(summary["api_calls_started"], 2)
        self.assertEqual(summary["stop_reason"], "max_new_chars")

    def test_continue_prefix(self):
        summary = self.run_fake(prefix="co")
        self.assertEqual(self.stdout.getvalue(), "cold")
        self.assertEqual(summary["new_characters"], 2)
        self.assertEqual(summary["api_calls_started"], 3)

    def test_immediate_stop_is_preserved(self):
        summary = self.run_fake(FakeEvaluator(target=""))
        self.assertEqual(summary["stop_reason"], "stop")
        self.assertEqual(summary["new_characters"], 0)
        self.assertEqual(summary["api_calls_started"], 1)

    def test_api_error_does_not_retry_and_preserves_prefix(self):
        fake = FakeEvaluator(fail_at=2)
        summary = self.run_fake(fake)
        self.assertEqual(len(fake.payloads), 2)
        self.assertEqual(summary["stop_reason"], "error")
        self.assertEqual(summary["successful_responses"], 1)
        self.assertEqual((self.path / "answer.txt").read_text(), "c")

    def test_interrupt_preserves_prefix(self):
        summary = self.run_fake(FakeEvaluator(interrupt_at=2))
        self.assertEqual(summary["stop_reason"], "interrupted")
        self.assertEqual((self.path / "answer.txt").read_text(), "c")

    def test_unknown_usage_not_invented(self):
        summary = self.run_fake(FakeEvaluator(missing_usage=True))
        self.assertEqual(summary["responses_missing_usage"]["input_tokens"], 5)
        self.assertEqual(summary["reported_token_totals"]["input_tokens"], 0)

    def test_trace_contains_payload_raw_response_and_policy(self):
        self.run_fake()
        events = [json.loads(line) for line in (self.path / "trace.jsonl").read_text().splitlines()]
        self.assertEqual(events[0]["event"], "request")
        self.assertIn("payload", events[0])
        self.assertEqual(events[1]["event"], "response")
        self.assertEqual(events[2]["event"], "decision")
        self.assertIn("raw_probabilities", events[2])
        self.assertIn("decoding_probabilities", events[2])
        self.assertEqual(events[-1]["event"], "summary")

    def test_existing_directory_not_overwritten(self):
        self.path.mkdir()
        with self.assertRaises(FileExistsError):
            self.run_fake()

    def test_dry_run_needs_no_key_or_network(self):
        with patch.dict("os.environ", {}, clear=True), patch("sys.stdout", new_callable=io.StringIO):
            with patch.object(m.TypeSafeHTTP, "evaluate", side_effect=AssertionError("Unexpected network")):
                self.assertEqual(m.main(["test", "--dry-run"]), 0)


class HTTPTests(unittest.TestCase):
    def test_unsafe_url_rejected(self):
        for url in ["http://api.typesafe.ai", "https://x:y@example.com", "https://example.com?key=a"]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                m.TypeSafeHTTP("test-key", url, 3)

    def test_blank_key_rejected(self):
        with self.assertRaises(ValueError):
            m.TypeSafeHTTP("", m.DEFAULT_BASE_URL, 3)

    def test_http_payload_and_headers(self):
        class Response(io.BytesIO):
            headers = {"x-typesafe-request-id": "test-request"}
        response = Response(b'{"model":"fake", "answers":{}, "usage":{}}')
        client = m.TypeSafeHTTP("secret-key", m.DEFAULT_BASE_URL, 3)
        with patch.object(client._opener, "open", return_value=response) as opener:
            result = client.evaluate({"test": "payload"})
        request = opener.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.typesafe.ai/v1/systemone")
        self.assertEqual(request.get_header("Authorization"), "Bearer secret-key")
        self.assertEqual(json.loads(request.data), {"test": "payload"})
        self.assertEqual(result.request_id, "test-request")
        self.assertEqual(opener.call_count, 1)

    def test_rate_limit_no_retry_and_key_redacted(self):
        client = m.TypeSafeHTTP("secret-key", m.DEFAULT_BASE_URL, 3)
        error = HTTPError("https://api.typesafe.ai/v1/systemone", 429, "limited", {}, io.BytesIO(b"secret-key"))
        with patch.object(client._opener, "open", side_effect=error) as opener:
            with self.assertRaises(m.APIError) as caught:
                client.evaluate({})
        self.assertNotIn("secret-key", str(caught.exception))
        self.assertIn("[REDACTED]", str(caught.exception))
        self.assertEqual(opener.call_count, 1)

    def test_connection_error_no_retry(self):
        client = m.TypeSafeHTTP("secret-key", m.DEFAULT_BASE_URL, 3)
        with patch.object(client._opener, "open", side_effect=URLError("unreachable")) as opener:
            with self.assertRaises(m.APIError):
                client.evaluate({})
        self.assertEqual(opener.call_count, 1)

    def test_truncated_response_is_reported_without_retry(self):
        class Response(io.BytesIO):
            headers = {}

            def read(self, size=-1):
                raise IncompleteRead(b'{"answers":', 100)

        client = m.TypeSafeHTTP("secret-key", m.DEFAULT_BASE_URL, 3)
        with patch.object(client._opener, "open", return_value=Response()) as opener:
            with self.assertRaises(m.APIError):
                client.list_models()
        self.assertEqual(opener.call_count, 1)

    def test_unreadable_http_error_body_keeps_status_and_no_retry(self):
        class BrokenBody(io.BytesIO):
            def read(self, size=-1):
                raise IncompleteRead(b"", 100)

        client = m.TypeSafeHTTP("secret-key", m.DEFAULT_BASE_URL, 3)
        error = HTTPError("https://api.typesafe.ai/v1/systemone", 503, "down", {}, BrokenBody())
        with patch.object(client._opener, "open", side_effect=error) as opener:
            with self.assertRaises(m.APIError) as caught:
                client.evaluate({})
        self.assertIn("503", str(caught.exception))
        self.assertEqual(opener.call_count, 1)

    def test_redirect_is_blocked(self):
        self.assertIsNone(m.NoRedirect().redirect_request(None, None, 302, "redirect", {}, "https://other.example"))

    def test_bad_json_rejected(self):
        class Response(io.BytesIO):
            headers = {}
        client = m.TypeSafeHTTP("secret-key", m.DEFAULT_BASE_URL, 3)
        with patch.object(client._opener, "open", return_value=Response(b"<html>error</html>")):
            with self.assertRaises(m.APIError):
                client.evaluate({})


if __name__ == "__main__":
    unittest.main()
