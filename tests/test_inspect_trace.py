"""Offline checks for the trace viewer; fixtures are invented score tables."""
from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import inspect_trace
import offline_demo as demo
import typesafe_llm as decoder


def make_run(directory: Path, search: str) -> Path:
    fixture = demo.FIXTURES[0]
    run_dir = directory / search
    decoder.generate(
        demo.ScoreTableEvaluator(fixture["scores"]), prompt="Invented finite score table",
        prefix="", vocabulary=fixture["vocabulary"],
        config={
            "model": "offline-score-table-not-jev", "search": search,
            "beam_width": 2, "beam_diversity": "none", "max_calls": 4,
            "max_new_chars": fixture["max_new_chars"], "temperature": 0, "top_k": 0,
            "top_p": 1, "seed": 0, "shuffle_options": False, "verbose": False,
        }, run_dir=run_dir, stdout=io.StringIO(), stderr=io.StringIO(),
    )
    return run_dir


def inspect(*argv: str) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = inspect_trace.main(list(argv))
    return code, stdout.getvalue(), stderr.getvalue()


class InspectTraceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_beam_run_lists_only_the_selected_path_and_frontiers(self):
        run_dir = make_run(self.root, "beam")
        code, output, _ = inspect(str(run_dir), "--frontiers")
        self.assertEqual(code, 0)
        # Selected path b, STOP; the explored a/aa branch is not a decision.
        self.assertIn("Showing 2 of 2 decisions.", output)
        self.assertIn("Search frontiers: showing", output)
        self.assertIn("best complete: 'b'", output)
        self.assertIn("Stop reason: stop; attempted API calls: 4; new characters: 1.", output)

    def test_stop_is_named_and_one_hot_entropy_is_not_negative(self):
        run_dir = make_run(self.root, "beam")
        _, output, _ = inspect(str(run_dir))
        stop_row = output.splitlines()[3].split()
        # Call 3 scored prefix "b" (after "" and "a"): STOP has probability 1.
        self.assertEqual(stop_row[:6], ["3", "STOP", "STOP", "1.000", "1.000", "0.00"])
        self.assertNotIn("None", output.split("Showing")[0])
        self.assertEqual(inspect_trace.action_text({"TOKEN_0002": "STOP"}, "TOKEN_0002"), "'STOP'")

    def test_trace_file_path_and_limit(self):
        run_dir = make_run(self.root, "greedy")
        code, output, _ = inspect(str(run_dir / "trace.jsonl"), "--limit", "1")
        self.assertEqual(code, 0)
        self.assertIn("Showing 1 of 2 decisions.", output)
        self.assertNotIn("Search frontiers", output)

    def test_invalid_json_reports_line_number(self):
        trace = self.root / "trace.jsonl"
        trace.write_text('{"event": "summary"}\n{not json\n', encoding="utf-8")
        code, _, error = inspect(str(trace))
        self.assertEqual(code, 1)
        self.assertIn("Invalid JSON at line 2", error)

    def test_invalid_display_limits_are_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            inspect_trace.main([str(self.root), "--top", "0"])


if __name__ == "__main__":
    unittest.main()
