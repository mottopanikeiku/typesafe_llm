#!/usr/bin/env python3
"""Compare real decoder policies on invented scores, without a key or network.

These scores are a teaching example, not Jev outputs or a quality benchmark.
Run: python offline_demo.py
"""
from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import typesafe_llm as decoder


VOCABULARY = decoder.make_vocabulary("ab")
# Every reachable prefix has a full fixed-vocabulary distribution. The locally
# weaker b branch completes with certainty; a can complete or continue to aa.
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


class ScoreTableEvaluator:
    """Return invented responses through the decoder's existing evaluator API."""

    def evaluate(self, payload: dict) -> decoder.APIResult:
        probabilities = SCORES[payload["state"]["answer_prefix"]]
        return decoder.APIResult({
            "model": "offline-score-table-not-jev",
            "answers": {decoder.QUESTION_ID: {
                "type": "choice",
                "choice": max(probabilities, key=probabilities.get),
                "probabilities": probabilities,
            }},
        })


def compare() -> list[dict]:
    """Use the production generator, cleaning its temporary traces on return."""
    rows = []
    with tempfile.TemporaryDirectory(prefix="typesafe-offline-") as temporary:
        for index, (name, search, width, budget) in enumerate(POLICIES):
            output = io.StringIO()
            summary = decoder.generate(
                ScoreTableEvaluator(), prompt="Offline score-table illustration",
                prefix="", vocabulary=VOCABULARY,
                config={
                    "model": "offline-score-table-not-jev", "search": search,
                    "beam_width": width, "beam_diversity": "none",
                    "max_calls": budget, "max_new_chars": 3,
                    "temperature": 0, "top_k": 0, "top_p": 1,
                    "seed": 0, "shuffle_options": False, "verbose": False,
                },
                run_dir=Path(temporary) / str(index),
                stdout=output, stderr=io.StringIO(),
            )
            if summary["error"] is not None:
                raise RuntimeError(summary["error"])
            rows.append({
                "policy": name,
                "evaluation_budget": budget,
                "evaluated_prefixes": summary["evaluated_prefixes"],
                "output": output.getvalue(),
                "completed": summary["stop_reason"] == "stop",
                "mean_log_score": summary["sequence_score"],
                "search_stop_reason": summary["search_stop_reason"],
            })
    return rows


if __name__ == "__main__":
    print(json.dumps({
        "source": "Invented score table; not a Jev measurement; no API calls.",
        "comparisons": compare(),
    }, indent=2, allow_nan=False))
