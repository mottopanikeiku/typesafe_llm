#!/usr/bin/env python3
"""Inspect a character-decoder trace without making any API calls."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="A run directory or its trace.jsonl file.")
    parser.add_argument("--top", type=int, default=5, help="Number of top raw alternatives per decision.")
    parser.add_argument("--limit", type=int, default=40, help="Max displayed decisions; 0 shows all.")
    args = parser.parse_args(argv)
    if args.top < 1 or args.limit < 0:
        parser.error("--top must be positive; --limit must be nonnegative.")
    path = args.path / "trace.jsonl" if args.path.is_dir() else args.path
    print("Call  Picked        API           Raw P    Local P  Entropy   Top raw alternatives")
    print("-" * 108)
    count, shown, summary = 0, 0, None
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except ValueError as exc:
                    raise ValueError(f"Invalid JSON at line {line_number}: {exc}") from None
                if event.get("event") == "summary":
                    summary = event
                if event.get("event") != "decision":
                    continue
                count += 1
                if args.limit and shown >= args.limit:
                    continue
                shown += 1
                probabilities = event["raw_probabilities"]
                total = math.fsum(probabilities.values())
                normalized = [p / total for p in probabilities.values() if p > 0]
                entropy = -math.fsum(p * math.log2(p) for p in normalized)
                selected = event["selected_label"]
                top = sorted(probabilities, key=probabilities.get, reverse=True)[:args.top]
                vocabulary = event.get("vocabulary", {})
                alternatives = ", ".join(f"{vocabulary.get(label, label)!r}:{probabilities[label]:.3f}" for label in top)
                picked_text = vocabulary.get(selected, selected)
                api_text = vocabulary.get(event["api_choice"], event["api_choice"])
                print(f"{event['call']:>4}  {picked_text!r:<12}  {api_text!r:<12}  "
                      f"{probabilities[selected]:>6.3f}   "
                      f"{event['decoding_probabilities'][selected]:>6.3f}   "
                      f"{entropy:>6.2f}   {alternatives}")
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"\nShowing {shown} of {count} decisions. Entropy is in bits over normalized Choice scores,")
    print("not Jev's native token distribution. Local P includes your decoding transformations.")
    if summary:
        print(f"Stop reason: {summary['stop_reason']}; attempted API calls: {summary['api_calls_started']}; "
              f"new characters: {summary['new_characters']}.")
        if summary.get("error"):
            print(f"Run error: {summary['error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
