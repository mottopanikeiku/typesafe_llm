#!/usr/bin/env python3
"""Bounded free-running exact-match evaluation of the TypeSafe character decoder.

Suite JSON: {"characters": "optional shared alphabet", "cases": [{"id": "unique",
"prompt": "task", "expected": ["accepted full answer"], "prefix": "optional",
"context": <optional JSON>, "max_new_chars": 40}]}.
Expected answers are scoring data only, never decoder configuration or requests.
Artifacts: manifest.json contains the request plan (no expected answers);
results.json contains aggregate metrics and every scheduled sample, including
not-run samples, expected answers, exact output, decoder summary and run path.
Both files are atomically replaced after each sample. A running sample after a
hard process kill can be inspected through its decoder run path; no auto-resume.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, TextIO

import typesafe_llm as decoder


SCHEMA_VERSION = 3
USAGE_KEYS = ("input_tokens", "output_tokens")


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number: {value}")


def load_suite(path: Path) -> list[dict[str, Any]]:
    suite = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object,
                       parse_constant=reject_constant)
    # Also reject escaped lone surrogates and numeric overflow before any requests.
    json.dumps(suite, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if not isinstance(suite, dict) or set(suite) - {"characters", "cases"}:
        raise ValueError("Suite must be an object with cases and optional characters only.")
    cases = suite.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Suite cases must be a nonempty list.")
    shared_characters = suite.get("characters", decoder.ALPHABETS["ascii"])
    if not isinstance(shared_characters, str):
        raise ValueError("Suite characters must be a string.")
    decoder.make_vocabulary(shared_characters)
    seen: set[str] = set()
    validated = []
    allowed = {"id", "prompt", "expected", "prefix", "context", "max_new_chars"}
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or set(case) - allowed:
            raise ValueError(f"Case {index + 1} must be an object with supported case fields only.")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen:
            raise ValueError(f"Case {index + 1} needs an explicit, nonempty, unique string id.")
        seen.add(case_id)
        if not isinstance(case.get("prompt"), str) or not case["prompt"].strip():
            raise ValueError(f"Case {case_id!r} needs a nonempty prompt.")
        expected = case.get("expected")
        if not isinstance(expected, list) or not expected or not all(isinstance(s, str) for s in expected):
            raise ValueError(f"Case {case_id!r} expected must be a nonempty list of strings.")
        prefix = case.get("prefix", "")
        if not isinstance(prefix, str):
            raise ValueError(f"Case {case_id!r} prefix must be a string.")
        if "max_new_chars" in case and (type(case["max_new_chars"]) is not int or case["max_new_chars"] < 1):
            raise ValueError(f"Case {case_id!r} max_new_chars must be a positive integer.")
        validated.append({**case, "prefix": prefix, "characters": shared_characters})
    return validated


class SharedBudget:
    """Charge each attempted underlying call before dispatch, including failures."""

    def __init__(self, client: decoder.Evaluator, limit: int) -> None:
        if type(limit) is not int or limit < 1:
            raise ValueError("The shared call limit must be a positive integer.")
        self.client = client
        self.limit = limit
        self.calls = 0
        self.responses = 0
        self.failed = False
        self.reported_usage = dict.fromkeys(USAGE_KEYS, 0)
        self.missing_usage = dict.fromkeys(USAGE_KEYS, 0)
        self.models: set[str] = set()
        self.request_seconds = 0.0

    @property
    def remaining(self) -> int:
        return self.limit - self.calls

    def evaluate(self, payload: dict[str, Any]) -> decoder.APIResult:
        if self.failed:
            raise decoder.APIError("Campaign halted after a previous failed attempt; not dispatched.")
        if self.remaining <= 0:
            raise decoder.APIError("Shared campaign call budget exhausted; not dispatched.")
        self.calls += 1
        started = time.perf_counter()
        try:
            result = self.client.evaluate(payload)
        except BaseException:
            self.failed = True
            raise
        finally:
            self.request_seconds += time.perf_counter() - started
        self.responses += 1
        model = result.body.get("model")
        if isinstance(model, str):
            self.models.add(model)
        usage = result.body.get("usage")
        for key in USAGE_KEYS:
            value = usage.get(key) if isinstance(usage, dict) else None
            if type(value) is int and value >= 0:
                self.reported_usage[key] += value
            else:
                self.missing_usage[key] += 1
        return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("suite", nargs="?", type=Path, default=Path("examples/benchmark.json"))
    p.add_argument("--model", default=os.environ.get("TYPESAFE_DEFAULT_MODEL", "").strip() or decoder.DEFAULT_MODEL)
    p.add_argument("--base-url", default=os.environ.get("TYPESAFE_BASE_URL", "").strip() or decoder.DEFAULT_BASE_URL)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--max-total-calls", type=int, default=100, help="Hard shared HTTP-attempt cap, including failed calls.")
    p.add_argument("--max-calls", type=int, default=256, help="Per-sample explored-prefix attempt cap, clipped to shared remainder.")
    p.add_argument("--max-new-chars", type=int, default=40, help="Per-sample character cap unless overridden in the suite.")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--seed", type=int, default=0, help="Sample seed is this value plus its zero-based schedule index.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--shuffle-options", action="store_true")
    p.add_argument("--instructions-file", type=Path)
    p.add_argument("--tokens-json", type=Path, help="Fixed JSON list of text fragments shared by all cases; never derive from targets.")
    p.add_argument("--search", choices=("beam", "greedy"), default="beam")
    p.add_argument("--beam-width", type=int, default=32)
    p.add_argument("--beam-diversity", choices=("token", "none"), default="token")
    p.add_argument("--out-dir", type=Path, default=Path("runs"))
    p.add_argument("--dry-run", action="store_true")
    return p


def validate_options(args: argparse.Namespace) -> str:
    if any(getattr(args, key) < 1 for key in ("max_total_calls", "max_calls", "max_new_chars", "repeats")):
        raise ValueError("Call limits, character limit and repeats must be positive integers.")
    if (not math.isfinite(args.temperature) or args.temperature < 0 or args.top_k < 0
            or not math.isfinite(args.top_p) or not 0 < args.top_p <= 1):
        raise ValueError("Use finite temperature >= 0, top-k >= 0 and 0 < top-p <= 1.")
    decoder.validate_search_options(args.search, args.beam_width, args.beam_diversity, args.temperature, args.top_k, args.top_p)
    if not args.model.strip():
        raise ValueError("Model cannot be empty.")
    # Constructor validation is local only; this placeholder is not a credential
    # and is never used for evaluation. Dry-run must validate transport options too.
    decoder.TypeSafeHTTP("offline-validation", args.base_url, args.timeout)
    if args.out_dir.exists() and not args.out_dir.is_dir():
        raise ValueError("Output directory is not a directory.")
    instructions = decoder.INSTRUCTIONS
    if args.instructions_file is not None:
        instructions = args.instructions_file.read_text(encoding="utf-8")
    if not instructions.strip():
        raise ValueError("Instructions cannot be empty.")
    instructions.encode("utf-8")
    args.model.encode("utf-8")
    args.tokens = []
    if args.tokens_json is not None:
        args.tokens = json.loads(args.tokens_json.read_text(encoding="utf-8"),
                                 object_pairs_hook=unique_object, parse_constant=reject_constant)
        if not isinstance(args.tokens, list) or not all(isinstance(token, str) for token in args.tokens):
            raise ValueError("--tokens-json must contain a list of strings.")
        json.dumps(args.tokens, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return instructions


def make_plan(cases: list[dict[str, Any]], args: argparse.Namespace, instructions: str) -> dict[str, Any]:
    samples = []
    characters = cases[0]["characters"]
    if any(case["characters"] != characters for case in cases):
        raise ValueError("All cases must use the same suite-level character vocabulary.")
    decoder.make_vocabulary(characters, args.tokens)
    for repeat in range(args.repeats):
        for case in cases:
            index = len(samples)
            samples.append({
                "index": index, "case_id": case["id"], "repeat": repeat + 1,
                "seed": args.seed + index, "prompt": case["prompt"], "prefix": case["prefix"],
                "context": case.get("context"),
                "max_calls": args.max_calls, "max_new_chars": case.get("max_new_chars", args.max_new_chars),
                "run_path": f"samples/{index + 1:06d}",
            })
    return {
        "schema_version": SCHEMA_VERSION, "decoder_version": decoder.VERSION,
        "suite_path": str(args.suite.resolve()), "model": args.model,
        "base_url": args.base_url, "timeout": args.timeout,
        "temperature": args.temperature, "top_k": args.top_k, "top_p": args.top_p,
        "shuffle_options": args.shuffle_options, "instructions": instructions,
        "search": args.search, "beam_width": args.beam_width, "beam_diversity": args.beam_diversity,
        "characters": characters, "tokens": args.tokens,
        "seed_schedule": "seed + zero-based sample index; repeat-major, suite order; local only",
        "max_total_calls": args.max_total_calls,
        "hard_max_attempts": min(args.max_total_calls, len(samples) * args.max_calls),
        "scheduled": len(samples), "samples": samples,
        "score_definition": "Strict full-string equality including prefix; completed only if stop_reason == stop. No normalization.",
        "measurement_note": "Choice-driven generation, not native LM token probabilities. No automatic retries or campaign resume.",
    }


def aggregate(records: list[dict[str, Any]], budget: SharedBudget, elapsed: float) -> dict[str, Any]:
    scheduled = len(records)
    started = sum(row["status"] != "not_run" for row in records)
    completed = sum(row["completed"] for row in records)
    matches = sum(row["completed_exact_match"] for row in records)
    raw_matches = sum(row["raw_exact_match"] for row in records)
    return {
        "scheduled": scheduled, "started": started, "completed": completed,
        "incomplete": started - completed, "not_run": scheduled - started,
        "completed_exact_matches": matches, "raw_exact_matches": raw_matches,
        "completed_accuracy": matches / scheduled,
        "accuracy_among_completed": matches / completed if completed else None,
        "completion_coverage": completed / scheduled, "started_coverage": started / scheduled,
        "api_calls_started": budget.calls, "successful_responses": budget.responses,
        "remaining_calls": budget.remaining, "elapsed_seconds": round(elapsed, 6),
        "request_elapsed_seconds": round(budget.request_seconds, 6),
        "new_characters": sum(row["new_characters"] for row in records),
        "total_output_characters": sum(len(row["output"]) for row in records if row["output"] is not None),
        "reported_token_totals": budget.reported_usage.copy(),
        "responses_missing_usage": budget.missing_usage.copy(),
        "model_versions_seen": sorted(budget.models),
    }


def run_campaign(client: decoder.Evaluator, *, cases: list[dict[str, Any]],
                 plan: dict[str, Any], campaign_dir: Path, stderr: TextIO = sys.stderr) -> dict[str, Any]:
    """Run an already validated plan in a new directory. No request uses expected."""
    vocabulary = decoder.make_vocabulary(plan["characters"], plan["tokens"])
    campaign_dir.mkdir(parents=True, exist_ok=False)
    budget = SharedBudget(client, plan["max_total_calls"])
    expected_by_id = {case["id"]: case["expected"] for case in cases}
    records = [{
        "index": sample["index"], "case_id": sample["case_id"], "repeat": sample["repeat"],
        "seed": sample["seed"], "run_path": sample["run_path"],
        "expected": expected_by_id[sample["case_id"]], "status": "not_run", "output": None,
        "completed": False, "raw_exact_match": False, "completed_exact_match": False,
        "new_characters": 0, "api_calls_started": 0, "elapsed_seconds": 0.0, "summary": None,
    } for sample in plan["samples"]]
    started = time.perf_counter()
    results: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "campaign_path": str(campaign_dir.resolve()),
        "status": "running", "error": None, "samples": records,
        "metric_definition": "completed_accuracy = completed_exact_matches / scheduled; accuracy_among_completed excludes unfinished samples.",
    }

    def save() -> None:
        results["aggregate"] = aggregate(records, budget, time.perf_counter() - started)
        atomic_json(campaign_dir / "results.json", results)
        atomic_json(campaign_dir / "manifest.json", {
            **plan, "campaign_path": str(campaign_dir.resolve()), "status": results["status"],
            "api_calls_started": budget.calls,
        })

    save()
    try:
        for sample, record in zip(plan["samples"], records):
            if budget.remaining == 0:
                results["status"] = "budget_exhausted"
                break
            record["status"] = "running"
            save()
            config = {key: plan[key] for key in (
                "model", "base_url", "timeout", "temperature", "top_k", "top_p", "shuffle_options",
                "search", "beam_width", "beam_diversity",
            )}
            config.update(seed=sample["seed"], max_calls=min(sample["max_calls"], budget.remaining),
                          max_new_chars=sample["max_new_chars"], verbose=False)
            run_dir = campaign_dir / sample["run_path"]
            output = io.StringIO()
            calls_before = budget.calls
            sample_started = time.perf_counter()
            try:
                summary = decoder.generate(
                    budget, prompt=sample["prompt"], prefix=sample["prefix"],
                    context=sample["context"], vocabulary=vocabulary,
                    instructions=plan["instructions"], config=config, run_dir=run_dir,
                    stdout=output, stderr=stderr,
                )
            except KeyboardInterrupt:
                summary = {"stop_reason": "interrupted", "error": None}
            except Exception as exc:
                summary = {"stop_reason": "error", "error": f"{type(exc).__name__}: {exc}"}
            text = output.getvalue()
            answer_path = run_dir / "answer.txt"
            if answer_path.exists():
                with answer_path.open(encoding="utf-8", newline="") as stream:
                    text = stream.read()
            completed = summary["stop_reason"] == "stop"
            raw_match = text in record["expected"]
            record.update(
                status=summary["stop_reason"], output=text, completed=completed,
                raw_exact_match=raw_match, completed_exact_match=completed and raw_match,
                new_characters=max(0, len(text) - len(sample["prefix"])), summary=summary,
                api_calls_started=budget.calls - calls_before,
                elapsed_seconds=round(time.perf_counter() - sample_started, 6),
            )
            if summary["stop_reason"] in ("error", "interrupted") or budget.failed:
                results["status"] = summary["stop_reason"] if summary["stop_reason"] in ("error", "interrupted") else "error"
                results["error"] = summary.get("error")
                save()
                break
            save()
        else:
            results["status"] = "finished"
    except KeyboardInterrupt:
        results["status"] = "interrupted"
    except Exception as exc:
        results["status"] = "error"
        results["error"] = f"{type(exc).__name__}: {exc}"
    save()
    return results


def main(argv: list[str] | None = None) -> int:
    p = parser()
    args = p.parse_args(argv)
    try:
        instructions = validate_options(args)
        cases = load_suite(args.suite)
        plan = make_plan(cases, args, instructions)
        if args.dry_run:
            print(json.dumps({"dry_run": True, **plan}, ensure_ascii=False, indent=2, allow_nan=False))
            return 0
        key = os.environ.get("TYPESAFE_API_KEY", "").strip()
        if not key:
            p.error("Set TYPESAFE_API_KEY before live evaluation. --dry-run needs no key or network.")
        client = decoder.TypeSafeHTTP(key, args.base_url, args.timeout)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        # Reserve a unique parent atomically; run_campaign owns its fresh child.
        parent = Path(tempfile.mkdtemp(prefix="benchmark_", dir=args.out_dir))
        campaign_dir = parent / "campaign"
        print(f"[benchmark] {campaign_dir}\n[limit] <= {plan['hard_max_attempts']} HTTP attempts; no retries.", file=sys.stderr)
        results = run_campaign(client, cases=cases, plan=plan, campaign_dir=campaign_dir)
        print(json.dumps({"status": results["status"], "aggregate": results["aggregate"],
                          "results_path": str(campaign_dir / "results.json")}, indent=2))
        if results["error"]:
            print(results["error"], file=sys.stderr)
        return 130 if results["status"] == "interrupted" else 1 if results["status"] == "error" else 0
    except KeyboardInterrupt:
        print("Interrupted; inspect the campaign's atomic results and per-sample checkpoints.", file=sys.stderr)
        return 130
    except (OSError, ValueError, decoder.APIError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
