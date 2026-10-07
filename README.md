# Text generation from structured choices

I built a small Python decoder that generates text by asking TypeSafe's Jev API to score a fixed alphabet, optional text fragments, and STOP.

**Question:** does exploring more prefixes—or preserving different token endings—recover a better-scoring completion?

**Result:** in [six invented finite examples](results/synthetic-search.json), I enumerated every positive STOP path and compared it with the real decoder. Token diversity helped in one example and hurt in another at the same beam width and evaluation count. Different tokenizations of the same text changed its mean-log score. These are **synthetic search results, not live Jev measurements or evidence of language quality**.

[typesafe_llm.py](typesafe_llm.py) builds requests and caches scored prefixes; [search.py](search.py) implements beam pruning and ranking; [offline_demo.py](offline_demo.py) supplies invented distributions and an independent exhaustive reference.

## What the finite comparisons show

All numbers below come from [the committed synthetic results](results/synthetic-search.json). Each table is fully specified there, including vocabulary, character cap, complete paths, selected labels, and scores.

| Example | Ordinary beam, width 2 | Token-diverse beam, width 2 | Exhaustive best | Evaluations per policy |
|---|---|---|---|---:|
| Diversity helps | `ba`, −1.914868 | `ac`, −0.436444 | `ac`, −0.436444 | 5 |
| Diversity hurts | `aa`, −0.401324 | `ba`, −1.914868 | `aa`, −0.401324 | 5 |

Scores are the **mean normalized Choice log probability per selected action, including STOP**. Diversity reserves beam slots for distinct last-action labels before filling remaining slots by probability. It changes which paths survive; it does not change their scores. Neither policy dominates these deliberately constructed examples.

The suite contains **22 completed paths and 30 policy comparisons** across six fixtures. A beam wide enough to avoid pruning matches the exhaustive best score in all six; the two-evaluation cap does not match it in any. These counts check implementation behavior on chosen examples, not a success rate on a representative dataset. The original greedy-versus-beam example is still available without flags.

### The same text can have different scores

In the overlapping-fragment fixture, both paths produce `ab`:

| Selected actions | Path probability | Actions including STOP | Mean log score |
|---|---:|---:|---:|
| `ab`, STOP | 0.55 | 2 | −0.298919 |
| `a`, `b`, STOP | 0.45 | 3 | −0.266169 |

Length normalization prefers the **lower-probability, longer action sequence**. The wide beam selects that sequence and reuses the cached `ab` evaluation. A separate fixture checks two equal-length fragment paths converging on `abc`: the higher-likelihood path survives the collision. Character-only and fragment vocabularies use different invented distributions; their scores are **not comparable model likelihoods**.

## Reproduce without API calls

On a normal CPU laptop with Python 3.10 or newer, no dependencies or credentials:

```bash
nice -n 19 python offline_demo.py --exhaustive --output results/synthetic-search.json
nice -n 19 python -m unittest discover -s tests -v
nice -n 19 python typesafe_llm.py 'Complete a short sentence.' --dry-run
```

Paid compute and API cost: **$0**. The dry run prints a request, not a generated answer. Tests check complete token paths against independently enumerated paths, call limits, cache reuse, score normalization at character caps, and the committed JSON.

## Limits and live use

- The fixtures are small and hand-built to expose search tradeoffs. They do not prove general beam optimality or useful text generation.
- Choice scores describe supplied candidates, not native token probabilities or hidden reasoning. Adding fragments changes both the choices and the scoring denominator.
- STOP termination is not answer correctness. Capped unfinished paths are never counted as exhaustive matches.
- The [four-case search diagnostic](examples/search_holdout.json) has already been used; it is not a fresh holdout.
- Live results remain private. Actual generation needs an authorized TypeSafe key and account-specific charges; no live calls were made for this work.

## Built on

TypeSafe's [System One](https://docs.typesafe.ai/concepts/system-one.md) and [Choice API](https://docs.typesafe.ai/primitives/choice.md) supply the external scorer. Beam search and token diversity are local policies, not documented Jev generation features. The [preview terms](https://typesafe.ai/terms) restrict benchmark publication and distillation; clarify applicable terms before any live comparison. CLI details and earlier history are in [docs/USAGE.md](docs/USAGE.md); official references are in [docs/SOURCES.md](docs/SOURCES.md).

Written with AI coding assistance.
