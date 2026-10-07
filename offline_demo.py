#!/usr/bin/env python3
"""Compare real decoder policies with exhaustive invented tables, never an API.

Run python offline_demo.py for the original example, or add --exhaustive for
all fixtures. Scores are synthetic, not Jev outputs or a quality benchmark.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import tempfile
from pathlib import Path

import typesafe_llm as decoder


VOCABULARY = decoder.make_vocabulary("ab")
SCORES = {
    "": {"a": 0.6, "b": 0.4, "STOP": 0.0},
    "a": {"a": 0.4, "b": 0.0, "STOP": 0.6},
    "b": {"a": 0.0, "b": 0.0, "STOP": 1.0},
    "aa": {"a": 0.0, "b": 0.0, "STOP": 1.0},
}
POLICIES = (
    ("greedy", "greedy", 1, 4),
    ("beam width 1", "beam", 1, 4),
    ("beam width 2, capped", "beam", 2, 2),
    ("beam width 2", "beam", 2, 4),
)


def diversity_fixture(helpful: bool) -> dict:
    vocabulary = decoder.make_vocabulary("abc")
    scores = {
        "": {"a": .6, "b": .4, "c": 0, "STOP": 0},
        "a": {"a": .5, "b": .05, "c": .45, "STOP": 0},
        "b": {"a": .8, "b": .01, "c": .19, "STOP": 0},
    }
    # Ordinary width two keeps ba and aa; diversity keeps ba and ac.
    for prefix in ("aa", "ab", "ac", "ba", "bb", "bc"):
        stop = 1.0 if prefix == ("ac" if helpful else "aa") else .01
        scores[prefix] = {"a": 1 - stop, "b": 0, "c": 0, "STOP": stop}
    return {
        "name": "diversity helps" if helpful else "diversity hurts",
        "vocabulary": vocabulary, "scores": scores, "max_new_chars": 2,
    }


FIXTURES = (
    {"name": "original", "vocabulary": VOCABULARY, "scores": SCORES, "max_new_chars": 3},
    diversity_fixture(True),
    diversity_fixture(False),
    {
        "name": "characters only", "vocabulary": decoder.make_vocabulary("ab"),
        "scores": {
            "": {"a": 1.0, "b": 0.0, "STOP": 0.0},
            "a": {"a": 0.0, "b": 1.0, "STOP": 0.0},
            "ab": {"a": 0.0, "b": 0.0, "STOP": 1.0},
        }, "max_new_chars": 2,
    },
    {
        "name": "overlapping fragment lengths",
        "vocabulary": decoder.make_vocabulary("ab", ["ab"]),
        "scores": {
            "": {"a": .45, "b": 0, "TOKEN_0002": .55, "STOP": 0},
            "a": {"a": 0, "b": 1, "TOKEN_0002": 0, "STOP": 0},
            "ab": {"a": 0, "b": 0, "TOKEN_0002": 0, "STOP": 1},
        }, "max_new_chars": 2,
    },
    {
        "name": "same-length fragment collision",
        "vocabulary": decoder.make_vocabulary("abc", ["ab", "bc"]),
        "scores": {
            "": {"a": .5, "b": 0, "c": 0, "TOKEN_0003": .5, "TOKEN_0004": 0, "STOP": 0},
            "a": {"a": 0, "b": 0, "c": 0, "TOKEN_0003": 0, "TOKEN_0004": .8, "STOP": .2},
            "ab": {"a": 0, "b": 0, "c": .9, "TOKEN_0003": 0, "TOKEN_0004": 0, "STOP": .1},
            "abc": {"a": 0, "b": 0, "c": 0, "TOKEN_0003": 0, "TOKEN_0004": 0, "STOP": 1},
        }, "max_new_chars": 3,
    },
)


class ScoreTableEvaluator:
    """Use the production evaluator interface without constructing HTTP clients."""

    def __init__(self, scores: dict | None = None):
        self.scores = SCORES if scores is None else scores

    def evaluate(self, payload: dict) -> decoder.APIResult:
        probabilities = self.scores[payload["state"]["answer_prefix"]]
        return decoder.APIResult({
            "model": "offline-score-table-not-jev",
            "answers": {decoder.QUESTION_ID: {
                "type": "choice", "choice": max(probabilities, key=probabilities.get),
                "probabilities": probabilities,
            }},
        })


def run_policy(fixture: dict, name: str, search: str, width: int,
               budget: int, diversity: str = "none") -> dict:
    """Run the real generator; retain deterministic results, not local timings."""
    with tempfile.TemporaryDirectory(prefix="typesafe-offline-") as temporary:
        output = io.StringIO()
        run_dir = Path(temporary) / "run"
        summary = decoder.generate(
            ScoreTableEvaluator(fixture["scores"]), prompt="Invented finite score table",
            prefix="", vocabulary=fixture["vocabulary"],
            config={
                "model": "offline-score-table-not-jev", "search": search,
                "beam_width": width, "beam_diversity": diversity,
                "max_calls": budget, "max_new_chars": fixture["max_new_chars"],
                "temperature": 0, "top_k": 0, "top_p": 1, "seed": 0,
                "shuffle_options": False, "verbose": False,
            }, run_dir=run_dir, stdout=output, stderr=io.StringIO(),
        )
        if summary["error"] is not None:
            raise RuntimeError(summary["error"])
        decisions = [event for line in (run_dir / "trace.jsonl").read_text().splitlines()
                     if (event := json.loads(line))["event"] == "decision"]
        return {
            "policy": name, "evaluation_budget": budget,
            "evaluated_prefixes": summary["evaluated_prefixes"],
            "cache_hits": summary["cache_hits"], "output": output.getvalue(),
            "selected_labels": [event["selected_label"] for event in decisions],
            "completed": summary["stop_reason"] == "stop",
            "log_probability": summary["sequence_log_probability"],
            "mean_log_score": summary["sequence_score"],
            "search_stop_reason": summary["search_stop_reason"],
        }


def compare() -> list[dict]:
    """Preserve the original four-policy teaching example."""
    return [run_policy(FIXTURES[0], *policy) for policy in POLICIES]


def exhaustive_paths(fixture: dict) -> list[dict]:
    """Enumerate every positive STOP path independently of BeamSearch.

    Never merge paths with the same text: their action counts can differ.
    Normalize the entire vocabulary before excluding over-limit fragments,
    just as the decoder's score definition requires.
    """
    completions = []

    def visit(prefix: str, labels: tuple[str, ...], log_probability: float) -> None:
        probabilities = fixture["scores"][prefix]
        total = math.fsum(probabilities.values())
        for label, token in fixture["vocabulary"].items():
            probability = probabilities[label]
            if probability == 0:
                continue
            path = labels + (label,)
            score = log_probability + math.log(probability) - math.log(total)
            if token is None:
                completions.append({
                    "output": prefix, "labels": list(path), "actions_including_stop": len(path),
                    "log_probability": score, "mean_log_score": score / len(path),
                })
            elif len(prefix) + len(token) <= fixture["max_new_chars"]:
                visit(prefix + token, path, score)

    visit("", (), 0.0)
    return completions


def exhaustive_comparison() -> dict:
    results = []
    for fixture in FIXTURES:
        paths = exhaustive_paths(fixture)
        best = max(paths, key=lambda path: path["mean_log_score"])
        # These small fixtures have at most len(scores) live states in a layer.
        # This width removes beam pruning, rather than claiming a general bound.
        full = len(fixture["scores"])
        policies = (
            ("beam width 1", 1, full, "none"),
            ("beam width 2", 2, full, "none"),
            ("beam width 2, token diversity", 2, full, "token"),
            ("beam width 2, cap 2", 2, 2, "none"),
            ("unpruned beam", full, full, "none"),
        )
        rows = []
        for name, width, budget, diversity in policies:
            row = run_policy(fixture, name, "beam", width, budget, diversity)
            row["matches_exhaustive_score"] = row["completed"] and math.isclose(
                row["mean_log_score"], best["mean_log_score"], rel_tol=0, abs_tol=1e-12,
            )
            rows.append(row)
        results.append({**fixture, "completed_paths": paths, "exhaustive_best": best,
                        "comparisons": rows})
    return {
        "source": "Invented score tables; not Jev measurements; zero API calls.",
        "score_definition": "Mean normalized Choice log probability per action, including STOP.",
        "fixtures": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exhaustive", action="store_true", help="compare all finite fixtures")
    parser.add_argument("--output", type=Path, help="write deterministic JSON instead of stdout")
    args = parser.parse_args()
    result = exhaustive_comparison() if args.exhaustive else {
        "source": "Invented score table; not a Jev measurement; no API calls.",
        "comparisons": compare(),
    }
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
