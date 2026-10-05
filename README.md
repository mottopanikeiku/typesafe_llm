# Text generation from structured choices

This is a small Python experiment that generates text by asking TypeSafe's Jev API to score a fixed alphabet plus STOP.

**Question:** can a model designed for structured decisions produce useful text when its choices are treated as character continuations, and does beam search help enough to justify exploring more prefixes?

[typesafe_llm.py](typesafe_llm.py) sends the prompt and current prefix with the same output vocabulary on every step; candidate descriptions show the extended text. [search.py](search.py) keeps competing paths, accumulates normalized log scores, and selects a STOP-terminated path by mean log score, including STOP. [benchmark.py](benchmark.py) checks exact answers without sending expected answers to the model.

**Public result:** the decoder and offline checks work as a search experiment, not evidence of a useful language model. Live results are private; no live accuracy or cost comparison is published here.

## What can be checked locally

[offline_demo.py](offline_demo.py) runs the actual generator against an **invented score table**, with no key or network. Its finite branches allow an [exhaustive reference test](tests/test_offline_demo.py) to check the selected score independently.

| Policy | Prefix evaluations | Selected text | Mean log score |
|---|---:|---|---:|
| Greedy | 2 | `a` | −0.511 |
| Beam, width 1 | 3 | `aa` | −0.476 |
| Beam, width 2, capped at 2 evaluations | 2 | `a` | −0.511 |
| Beam, width 2 | 4 | `b` | −0.458 |

These numbers come from the [demo's score table and policies](offline_demo.py), not Jev. The wider beam recovers the best-scoring completion in this example, but evaluates more prefixes; at the smaller budget it does not improve on greedy. Mean-log ranking can also prefer a longer path. None of this establishes which policy produces better real answers.

## Reproduce without API calls

From a checkout, using Python 3.10 or newer (as used by the [CI workflow](.github/workflows/checks.yml)):

```bash
nice -n 19 python offline_demo.py
nice -n 19 python -m unittest discover -s tests -v
nice -n 19 python typesafe_llm.py 'Complete a short sentence.' --dry-run
```

A normal CPU laptop is enough. Only the Python standard library is needed; no GPU, model download, API key, or paid compute is required. The dry run prints a request, not a generated answer.

Actual generation requires your own TypeSafe key and account-specific API charges. It was **not run for this change**. The [public preview terms](https://typesafe.ai/terms) restrict benchmark publication, distillation, and competing products. Keep `runs/` private and clarify the applicable terms before running or sharing experiments. Detailed CLI usage, trace formats, source checks, and change history are preserved in [docs/USAGE.md](docs/USAGE.md).

## Limitations

- Choice scores describe our supplied candidates, not native token probabilities, hidden reasoning, or a separately trained language model.
- The [search diagnostic suite](examples/search_holdout.json) has four short cases. Existing diagnostic suites have already been used for tuning or evaluation; they are not fresh holdouts or evidence of general language quality.
- Beam search scores each uncached explored prefix, whereas greedy follows one path. More exploration costs more API requests; call caps are not dollar caps.
- Beam pruning and length normalization are heuristics, not optimality or correctness guarantees. A positive STOP score does not establish a correct answer.
- Exact matching rejects formatting differences and unfinished outputs. The wrapper does not repair spelling, copying, arithmetic, or stopping failures.

## Built on

TypeSafe's [System One](https://docs.typesafe.ai/concepts/system-one.md) and [Choice API](https://docs.typesafe.ai/primitives/choice.md) provide the external scorer. Beam search, token diversity, and path ranking are local decoder policies, not documented Jev text-generation features. Official protocol, SDK, pricing, and terms references are collected in [docs/SOURCES.md](docs/SOURCES.md).
