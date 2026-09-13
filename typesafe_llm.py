#!/usr/bin/env python3
"""Experimental autoregressive decoder for TypeSafe Jev. Python 3.10+, no dependencies.

Contextual beam search over a fixed vocabulary of text tokens and STOP.
Scores are TypeSafe Choice outputs, NOT native next-token probabilities.

    export TYPESAFE_API_KEY='...'
    python typesafe_llm.py 'What is the opposite of hot? One lowercase word.'

Protocol references are in SOURCES.md. All API calls require your own key.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import secrets
import string
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from http.client import HTTPException
from pathlib import Path
from typing import Any, Protocol, TextIO
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from search import BeamSearch, PrefixEvaluation

VERSION = "5.0.0"
DEFAULT_BASE_URL = "https://api.typesafe.ai"
DEFAULT_MODEL = "jev-latest"
QUESTION_ID = "next_token"
STOP = "STOP"
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_CHOICES = 255
ALPHABETS = {
    "lower": string.ascii_lowercase + " .",
    "ascii": "".join(chr(i) for i in range(32, 127)) + "\n",
}
INSTRUCTIONS = (
    "Choose the candidate text that is a prefix of the correct, concise answer to user_prompt. "
    "Use context as task data when present. Each candidate extends answer_prefix by exactly "
    "one token; candidates can end inside a word and need not be complete answers. "
    "Prefer the continuation that can lead to the correct answer, not an unrelated word. "
    "Choose STOP only if answer_prefix is already a complete correct answer."
)

class APIError(RuntimeError):
    """An HTTP, connection, or protocol error; never silently retried."""


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward the API key to a redirect destination.
        return None


@dataclass
class APIResult:
    body: dict[str, Any]
    request_id: str | None = None
    elapsed_ms: float = 0.0


class Evaluator(Protocol):
    def evaluate(self, payload: dict[str, Any]) -> APIResult: ...


class TypeSafeHTTP:
    """Minimal HTTPS client. No retries, no redirects, no secret logging."""

    def __init__(self, api_key: str, base_url: str, timeout: float) -> None:
        parsed = urlsplit(base_url)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username
                or parsed.password or parsed.query or parsed.fragment
                or any(c.isspace() for c in base_url)):
            raise ValueError("The API base URL must be HTTPS, without credentials/query/fragment.")
        if not api_key.strip() or any(c.isspace() for c in api_key.strip()):
            raise ValueError("The API key must be nonempty and contain no whitespace.")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Timeout must be positive and finite.")
        self._key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._opener = build_opener(NoRedirect())

    def _request(self, path: str, payload: dict[str, Any] | None) -> APIResult:
        data = None if payload is None else json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        request = Request(
            self.base_url + path, data=data,
            method="GET" if payload is None else "POST",
            headers={
                "Authorization": f"Bearer {self._key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": f"typesafe-autoregressive-decoder/{VERSION}",
            },
        )
        started = time.perf_counter()
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
                request_id = response.headers.get("x-typesafe-request-id")
        except HTTPError as exc:
            with exc:
                try:
                    excerpt = exc.read(1500).decode("utf-8", errors="replace")
                except (HTTPException, OSError):
                    excerpt = "(response body could not be read)"
            excerpt = excerpt.replace(self._key, "[REDACTED]")
            hints = {
                401: "Check TYPESAFE_API_KEY.",
                403: "Check your API/account permissions.",
                404: "Check the model name and API base URL; try --list-models.",
                422: "Check the request schema; try a smaller character set.",
                429: "Rate limited. Wait before starting another run.",
            }
            hint = hints.get(exc.code, "The request was not retried.")
            raise APIError(f"HTTP {exc.code}. {hint} Response: {excerpt}") from None
        except (HTTPException, URLError, OSError, ValueError) as exc:
            message = str(exc).replace(self._key, "[REDACTED]")
            raise APIError(f"Request failed (not retried): {message}") from None
        if len(raw) > MAX_RESPONSE_BYTES:
            raise APIError("API response exceeded the 8 MiB safety limit.")
        try:
            body = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            raise APIError("The API returned a non-JSON response.") from None
        if not isinstance(body, dict):
            raise APIError("The API response must be a JSON object.")
        return APIResult(body, request_id, (time.perf_counter() - started) * 1000)

    def evaluate(self, payload: dict[str, Any]) -> APIResult:
        return self._request("/v1/systemone", payload)

    def list_models(self) -> APIResult:
        return self._request("/v1/models", None)


def make_vocabulary(
    characters: str, tokens: list[str] | None = None,
) -> dict[str, str | None]:
    """Keep character fallback and optional exact text fragments; None means STOP."""
    special = {" ": "SPACE", "\n": "NEWLINE", "\t": "TAB", ".": "PERIOD"}
    vocabulary: dict[str, str | None] = {}
    for char in dict.fromkeys(characters):
        if not char.isprintable() and char not in ("\n", "\t"):
            raise ValueError("Character sets may not contain terminal control characters.")
        label = char if char in string.ascii_letters + string.digits else special.get(
            char, f"U{ord(char):04X}"
        )
        vocabulary[label] = char
    if not vocabulary:
        raise ValueError("The character set cannot be empty.")
    if tokens is not None:
        if not isinstance(tokens, list):
            raise ValueError("Tokens must be a JSON array of nonempty strings.")
        seen = set(vocabulary.values())
        for token in tokens:
            if not isinstance(token, str) or not token:
                raise ValueError("Tokens must be nonempty strings.")
            if any(not c.isprintable() and c not in ("\n", "\t") for c in token):
                raise ValueError("Tokens may not contain terminal control characters.")
            if token not in seen:
                vocabulary[f"TOKEN_{len(vocabulary):04d}"] = token
                seen.add(token)
    vocabulary[STOP] = None
    if len(vocabulary) > MAX_CHOICES:
        raise ValueError(f"TypeSafe Choice supports at most {MAX_CHOICES} options, including STOP.")
    return vocabulary


def make_payload(
    prompt: str, prefix: str, vocabulary: dict[str, str | None], model: str,
    context: Any = None, instructions: str = INSTRUCTIONS,
    order_rng: random.Random | None = None,
) -> dict[str, Any]:
    items = vocabulary.items()
    if order_rng is not None:
        items = list(items)
        order_rng.shuffle(items)
    criteria: dict[str, str | None] = {}
    for label, token in items:
        if token is None:
            criteria[label] = "The complete answer is " + json.dumps(prefix, ensure_ascii=False)
        else:
            criteria[label] = "The answer begins with " + json.dumps(prefix + token, ensure_ascii=False)
    state: dict[str, Any] = {"user_prompt": prompt, "answer_prefix": prefix}
    if context is not None:
        state["context"] = context
    return {
        "model": model, "state": state,
        "questions": {QUESTION_ID: {
            "type": "choice", "instructions": instructions, "criteria": criteria,
        }},
    }


def read_choice(
    body: dict[str, Any], vocabulary: dict[str, str | None],
) -> tuple[str, dict[str, float]]:
    try:
        answer = body["answers"][QUESTION_ID]
        selected, probabilities = answer["choice"], answer["probabilities"]
    except (KeyError, TypeError):
        raise APIError(f"Response is missing the {QUESTION_ID} Choice answer.") from None
    if answer.get("type") != "choice" or not isinstance(selected, str) or selected not in vocabulary:
        raise APIError("Response contains an invalid Choice type or selected label.")
    if not isinstance(probabilities, dict) or set(probabilities) != set(vocabulary):
        raise APIError("Returned probability labels do not match the requested vocabulary.")
    checked: dict[str, float] = {}
    for label in vocabulary:
        value = probabilities[label]
        if (isinstance(value, bool) or not isinstance(value, (float, int))
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise APIError(f"Invalid probability for {label!r}: {value!r}")
        checked[label] = float(value)
    if math.fsum(checked.values()) <= 0:
        raise APIError("All returned probabilities are zero; cannot decode.")
    return selected, checked


def choose_label(
    probabilities: dict[str, float], api_choice: str, temperature: float,
    top_k: int, top_p: float, rng: random.Random,
) -> tuple[str, dict[str, float]]:
    """Return the chosen label and the *local* decoding distribution.

    Input must be validated by read_choice. Raw probabilities are preserved
    separately in the trace. Temperature/top-k/top-p are not API parameters.
    """
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("Temperature must be finite and nonnegative.")
    if top_k < 0 or not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_k must be nonnegative and top_p must be in (0, 1].")
    # API selection breaks ties caused by rounded probabilities, but never
    # overrides a strictly larger reported probability.
    ranked = sorted(probabilities, key=lambda k: (probabilities[k], k == api_choice), reverse=True)
    if temperature == 0:
        return ranked[0], {ranked[0]: 1.0}
    ranked = [k for k in ranked if probabilities[k] > 0]
    if top_k:
        ranked = ranked[:top_k]
    maximum = math.log(probabilities[ranked[0]])
    weights = [math.exp((math.log(probabilities[k]) - maximum) / temperature) for k in ranked]
    total = math.fsum(weights)
    scaled = [w / total for w in weights]
    if top_p < 1:
        cumulative, keep = 0.0, 0
        for p in scaled:
            cumulative += p
            keep += 1
            if cumulative >= top_p:
                break
        ranked, scaled = ranked[:keep], scaled[:keep]
    total = math.fsum(scaled)
    distribution = {k: p / total for k, p in zip(ranked, scaled) if p > 0}
    selected = rng.choices(list(distribution), weights=list(distribution.values()), k=1)[0]
    return selected, distribution


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def checkpoint_text(path: Path, text: str) -> None:
    """Replace a checkpoint atomically, rather than leaving a partially written answer."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        stream.write(text)
    temporary.replace(path)


def validate_search_options(
    search: str, beam_width: int, beam_diversity: str, temperature: float, top_k: int, top_p: float,
) -> None:
    if search not in ("greedy", "beam"):
        raise ValueError("Search must be greedy or beam.")
    if beam_diversity not in ("none", "token"):
        raise ValueError("Beam diversity must be none or token.")
    if type(beam_width) is not int or beam_width < 1:
        raise ValueError("Beam width must be a positive integer.")
    if search == "beam" and (temperature != 0 or top_k != 0 or top_p != 1):
        raise ValueError("Sampling controls require --search greedy; beam search uses all reported scores.")


def generate(
    client: Evaluator, *, prompt: str, prefix: str, vocabulary: dict[str, str | None],
    config: dict[str, Any], run_dir: Path, stdout: TextIO, stderr: TextIO,
    context: Any = None, instructions: str = INSTRUCTIONS,
) -> dict[str, Any]:
    """Generate into a fresh directory; speculative beam text is never streamed."""
    validate_search_options(config["search"], config["beam_width"], config["beam_diversity"],
                            config["temperature"], config["top_k"], config["top_p"])
    vocabulary = dict(vocabulary)
    if config["shuffle_options"]:
        items = list(vocabulary.items())
        random.Random(f"{config['seed']}:options").shuffle(items)
        vocabulary = dict(items)
    beam = (BeamSearch(prefix, vocabulary, config["beam_width"], config["max_new_chars"], config["beam_diversity"])
            if config["search"] == "beam" else None)
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "config.json", {
        "decoder_version": VERSION, "trace_version": 5, "python_version": sys.version,
        "prompt": prompt, "initial_prefix": prefix, "context": context,
        "instructions": instructions, "vocabulary": vocabulary, **config,
    })
    answer_path = run_dir / "answer.txt"
    checkpoint_text(answer_path, prefix)
    text, calls, successes, new_tokens = prefix, 0, 0, 0
    path_log_probability, path_tokens, cache_hits = 0.0, 0, 0
    token_totals = {"input_tokens": 0, "output_tokens": 0}
    missing_usage = {"input_tokens": 0, "output_tokens": 0}
    models: set[str] = set()
    cache: dict[str, PrefixEvaluation] = {}
    rng = random.Random(config["seed"])
    started = time.perf_counter()
    reason, error = "max_calls", None
    with (run_dir / "trace.jsonl").open("x", encoding="utf-8", newline="\n") as trace:
        def log(event: str, **fields: Any) -> None:
            trace.write(json.dumps({
                "event": event, "time": datetime.now(timezone.utc).isoformat(), **fields,
            }, ensure_ascii=False, allow_nan=False) + "\n")
            trace.flush()

        def evaluate(current_prefix: str) -> PrefixEvaluation | None:
            nonlocal calls, successes, cache_hits
            if current_prefix in cache:
                evaluation = cache[current_prefix]
                cache_hits += 1
                log("cache_hit", prefix=current_prefix, source_call=evaluation.call)
                return evaluation
            if calls >= config["max_calls"]:
                return None
            payload = make_payload(prompt, current_prefix, vocabulary, config["model"], context, instructions)
            # One owner counts before dispatch. Failed attempts are never refunded.
            calls += 1
            log("request", call=calls, payload=payload)
            result = client.evaluate(payload)
            successes += 1
            log("response", call=calls, request_id=result.request_id,
                elapsed_ms=result.elapsed_ms, body=result.body)
            model = result.body.get("model")
            if isinstance(model, str):
                models.add(model)
            usage = result.body.get("usage")
            for key in token_totals:
                value = usage.get(key) if isinstance(usage, dict) else None
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    token_totals[key] += value
                else:
                    missing_usage[key] += 1
            api_choice, probabilities = read_choice(result.body, vocabulary)
            evaluation = PrefixEvaluation(calls, api_choice, probabilities)
            cache[current_prefix] = evaluation
            return evaluation

        def commit(evaluation: PrefixEvaluation, selected: str, policy: dict[str, float], over_limit: bool = False) -> None:
            nonlocal text, new_tokens, path_log_probability, path_tokens
            probabilities = evaluation.probabilities
            token = vocabulary[selected]
            emitted = "" if over_limit else token
            next_text = text if emitted is None else text + emitted
            total = math.fsum(probabilities.values())
            if not over_limit:
                path_log_probability += math.log(probabilities[selected]) - math.log(total)
                path_tokens += 1
            log("decision", call=evaluation.call, prefix_before=text, prefix_after=next_text,
                api_choice=evaluation.api_choice, selected_label=selected,
                selected_text=token, emitted_text=emitted, vocabulary=vocabulary,
                raw_probabilities=probabilities, raw_probability_sum=total,
                decoding_probabilities=policy, selection_method=config["search"],
                path_log_probability=path_log_probability)
            if config["verbose"]:
                top = sorted(probabilities, key=probabilities.get, reverse=True)[:5]
                details = ", ".join(f"{vocabulary[k]!r}={probabilities[k]:.3f}" for k in top)
                stderr.write(f"\n[{evaluation.call}] selected={token!r}; API={vocabulary[evaluation.api_choice]!r}; {details}\n")
                stderr.flush()
            if emitted:
                text = next_text
                new_tokens += 1
                checkpoint_text(answer_path, text)
                stdout.write(emitted)
                stdout.flush()

        def checkpoint_beam() -> None:
            if beam is None:
                return
            snapshot = beam.snapshot()
            log("frontier", call=calls, **snapshot)
            candidate = beam.result()
            checkpoint_text(run_dir / "candidate.txt", candidate.text)
            if config["verbose"]:
                leaders = ", ".join(repr(row["prefix"]) for row in snapshot["active"][:3])
                stderr.write(f"\n[beam depth={beam.depth} calls={calls}/{config['max_calls']}] "
                             f"active: {leaders}; best complete: "
                             f"{candidate.text if candidate.completed else None!r}\n")
                stderr.flush()

        try:
            stdout.write(prefix)
            stdout.flush()
            if beam is not None:
                while beam.pending_prefix is not None:
                    evaluation = evaluate(beam.pending_prefix)
                    if evaluation is None:
                        reason = "max_calls"
                        break
                    depth = beam.depth
                    beam.observe(evaluation)
                    if beam.depth != depth:
                        checkpoint_beam()
                else:
                    reason = "frontier_exhausted"
            else:
                while calls < config["max_calls"]:
                    if len(text) - len(prefix) >= config["max_new_chars"]:
                        reason = "max_new_chars"
                        break
                    evaluation = evaluate(text)
                    if evaluation is None:
                        reason = "max_calls"
                        break
                    selected, policy = choose_label(
                        evaluation.probabilities, evaluation.api_choice,
                        config["temperature"], config["top_k"], config["top_p"], rng,
                    )
                    token = vocabulary[selected]
                    over_limit = token is not None and len(text) - len(prefix) + len(token) > config["max_new_chars"]
                    commit(evaluation, selected, policy, over_limit)
                    if over_limit:
                        reason = "max_new_chars"
                        break
                    if token is None:
                        reason = "stop"
                        break
        except KeyboardInterrupt:
            reason = "interrupted"
        except Exception as exc:
            reason, error = "error", f"{type(exc).__name__}: {exc}"

        search_stop_reason = reason
        if beam is not None:
            checkpoint_beam()
            result = beam.result()
            # Only the selected path becomes output; exploration never edits stdout.
            for step in result.steps:
                commit(step.evaluation, step.label, {step.label: 1.0})
            if reason not in ("error", "interrupted"):
                reason = "stop" if result.completed else "max_new_chars" if reason == "frontier_exhausted" else reason
        summary = {
            "stop_reason": reason, "search_stop_reason": search_stop_reason, "error": error,
            "api_calls_started": calls, "successful_responses": successes,
            "new_characters": len(text) - len(prefix), "total_characters": len(text),
            "new_tokens": new_tokens, "evaluated_prefixes": len(cache), "cache_hits": cache_hits,
            "sequence_log_probability": path_log_probability,
            "sequence_score": path_log_probability / path_tokens if path_tokens else None,
            "score_definition": "Mean normalized Choice log probability per selected token, including STOP.",
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "reported_token_totals": token_totals, "responses_missing_usage": missing_usage,
            "model_versions_seen": sorted(models),
        }
        log("summary", **summary)
        checkpoint_text(answer_path, text)
        write_json(run_dir / "summary.json", summary)
    return summary


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("prompt", nargs="?", help="Question or instruction to answer.")
    p.add_argument("--prompt-file", type=Path, help="Read the prompt from UTF-8 text instead.")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--prefix", default="", help="Answer text already written.")
    group.add_argument("--prefix-file", type=Path, help="Continue an answer.txt from a previous run.")
    p.add_argument("--context-json", type=Path, help="Extra task data, e.g. the original payoff state.")
    p.add_argument("--instructions-file", type=Path, help="Override the fixed next-token selection instructions.")
    p.add_argument("--alphabet", choices=ALPHABETS, default="lower", help="lower: a-z, space, period; ascii: printable ASCII + newline.")
    p.add_argument("--characters-json", type=Path, help="Custom JSON string or list of single characters; overrides --alphabet.")
    p.add_argument("--tokens-json", type=Path, help="JSON array of text fragments to add to the character vocabulary.")
    p.add_argument("--search", choices=("beam", "greedy"), default="beam", help="beam: compare character paths; greedy: commit each local choice.")
    p.add_argument("--beam-width", type=int, default=32, help="Maximum live hypotheses per token depth.")
    p.add_argument("--beam-diversity", choices=("token", "none"), default="token", help="Reserve beam coverage for distinct ending tokens; none uses probability alone.")
    p.add_argument("--model", default=os.environ.get("TYPESAFE_DEFAULT_MODEL", "").strip() or DEFAULT_MODEL)
    p.add_argument("--base-url", default=os.environ.get("TYPESAFE_BASE_URL", "").strip() or DEFAULT_BASE_URL)
    p.add_argument("--timeout", type=float, default=30.0, help="Timeout in seconds for network operations.")
    p.add_argument("--max-calls", type=int, default=256, help="Hard limit on all explored-prefix HTTP attempts (no retries).")
    p.add_argument("--max-new-chars", type=int, default=80, help="Additional characters, excluding any supplied prefix.")
    p.add_argument("--temperature", type=float, default=0.0, help="With --search greedy: 0 = argmax, >0 = local sampling.")
    p.add_argument("--top-k", type=int, default=0, help="Sampling: keep this many options; 0 keeps all.")
    p.add_argument("--top-p", type=float, default=1.0, help="Sampling: nucleus probability cutoff after temperature/top-k.")
    p.add_argument("--seed", type=int, help="Local sampling seed; logged. Does not seed the server.")
    p.add_argument("--shuffle-options", action="store_true", help="Shuffle criteria once per run, then keep that order; logged.")
    p.add_argument("--out-dir", type=Path, default=Path("runs"), help="Parent directory for a unique run folder.")
    p.add_argument("--verbose", "-v", action="store_true", help="Show search frontiers and selected-token probabilities on stderr.")
    p.add_argument("--dry-run", action="store_true", help="Print the first request body without a key or a network call.")
    p.add_argument("--list-models", action="store_true", help="Make one authenticated GET request and print available models.")
    return p


def main(argv: list[str] | None = None) -> int:
    p = parser()
    args = p.parse_args(argv)
    if (not math.isfinite(args.temperature) or args.temperature < 0 or args.top_k < 0
            or not math.isfinite(args.top_p) or not 0 < args.top_p <= 1
            or not math.isfinite(args.timeout) or args.timeout <= 0
            or args.max_calls < 1 or args.max_new_chars < 1):
        p.error("Use finite temperature >= 0, top-k >= 0, 0 < top-p <= 1, and positive timeout/limits.")
    if args.list_models and args.dry_run:
        p.error("--list-models and --dry-run are mutually exclusive.")
    try:
        validate_search_options(args.search, args.beam_width, args.beam_diversity, args.temperature, args.top_k, args.top_p)
        if args.list_models:
            client = TypeSafeHTTP(os.environ.get("TYPESAFE_API_KEY", ""), args.base_url, args.timeout)
            print(json.dumps(client.list_models().body, indent=2, ensure_ascii=False))
            return 0
        if (args.prompt is None) == (args.prompt_file is None):
            p.error("Supply exactly one positional prompt or --prompt-file.")
        prompt = args.prompt if args.prompt_file is None else args.prompt_file.read_text(encoding="utf-8")
        if not prompt.strip():
            p.error("The prompt cannot be empty.")
        prefix = args.prefix if args.prefix_file is None else args.prefix_file.read_text(encoding="utf-8")
        context = None if args.context_json is None else json.loads(args.context_json.read_text(encoding="utf-8"))
        instructions = INSTRUCTIONS
        if args.instructions_file is not None:
            instructions = args.instructions_file.read_text(encoding="utf-8")
        if not instructions.strip():
            p.error("The next-token instructions cannot be empty.")
        characters = ALPHABETS[args.alphabet]
        if args.characters_json is not None:
            custom = json.loads(args.characters_json.read_text(encoding="utf-8"))
            if isinstance(custom, list) and all(isinstance(c, str) and len(c) == 1 for c in custom):
                characters = "".join(custom)
            elif isinstance(custom, str):
                characters = custom
            else:
                p.error("--characters-json must contain a string or a list of single-character strings.")
        tokens = None if args.tokens_json is None else json.loads(args.tokens_json.read_text(encoding="utf-8"))
        if args.tokens_json is not None and not isinstance(tokens, list):
            p.error("--tokens-json must contain a JSON array of nonempty strings.")
        vocabulary = make_vocabulary(characters, tokens)
        seed = args.seed if args.seed is not None else secrets.randbits(64)
        if args.dry_run:
            order_rng = random.Random(f"{seed}:options") if args.shuffle_options else None
            payload = make_payload(prompt, prefix, vocabulary, args.model, context, instructions, order_rng)
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0
        key = os.environ.get("TYPESAFE_API_KEY", "").strip()
        if not key:
            p.error("Set TYPESAFE_API_KEY before making API calls. --dry-run needs no key.")
        client = TypeSafeHTTP(key, args.base_url, args.timeout)
        run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + secrets.token_hex(3)
        run_dir = args.out_dir / run_name
        config = {name: getattr(args, name) for name in (
            "model", "base_url", "timeout", "max_calls", "max_new_chars", "temperature",
            "top_k", "top_p", "shuffle_options", "verbose", "search", "beam_width", "beam_diversity",
        )}
        config["seed"] = seed
        print(f"[run] {run_dir}\n[limit] <= {args.max_calls} HTTP attempts; no retries. "
              "Traces contain your prompt/context and generated text.", file=sys.stderr)
        if args.temperature == 0 and (args.top_k or args.top_p < 1):
            print("[note] top-k/top-p have no effect in greedy mode (temperature=0).", file=sys.stderr)
        summary = generate(client, prompt=prompt, prefix=prefix, vocabulary=vocabulary,
                           config=config, run_dir=run_dir, stdout=sys.stdout, stderr=sys.stderr,
                           context=context, instructions=instructions)
        print(f"\n[{summary['stop_reason']}] {summary['new_characters']} new characters "
              f"in {summary['new_tokens']} tokens; "
              f"{summary['api_calls_started']} attempted calls; {summary['elapsed_seconds']}s.\n"
              f"[search ended: {summary['search_stop_reason']}] "
              f"{summary['evaluated_prefixes']} scored prefixes; {summary['cache_hits']} cache hits; "
              f"mean path log score={summary['sequence_score']}.\n"
              f"[saved] {run_dir / 'answer.txt'}", file=sys.stderr)
        if summary["error"]:
            print(summary["error"], file=sys.stderr)
            return 1
        return 130 if summary["stop_reason"] == "interrupted" else 0
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except (APIError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
