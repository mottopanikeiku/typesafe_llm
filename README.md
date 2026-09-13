# TypeSafe autoregressive decoder

An experimental text generator built from Jev's System One Choice evaluations.
Every request offers **the exact same fixed vocabulary**, including STOP.
The default decoder searches competing character paths rather than committing
to the first locally preferred letter. Candidate descriptions show `prefix + token`;
the label-to-output mapping never changes. No dictionary proposes continuations,
and no spelling is repaired after selection.

**Python 3.10+; no third-party packages or SDK installation required.**
You need your own TypeSafe API key. The default model is `jev-latest`, as
recommended by TypeSafe's current API documentation. Use `--list-models` and
`--model` to pin an available version for repeatable comparisons.

This constructs an external autoregressive decoding process, not an independently
trained language model. It does not expose weights, hidden activations, native
token probabilities, or the RLCD reward function. Jev is trained for structured
decisions, not character generation: spelling, copying, arithmetic, and stopping
can fail even on short tasks. Do not use generated text without checking it.

## How generation works

Every evaluation resends the original prompt, optional task `context`, and the
**full candidate `answer_prefix`**. There is no server-side conversation memory,
automatic context truncation, or summarization.

1. Build the character vocabulary and STOP once. Optional `--tokens-json`
   fragments are also fixed before generation starts.
2. Ask one Choice question with that entire vocabulary. A token always emits the
   same exact text, but its description shows the resulting candidate prefix.
   STOP describes the current prefix as a complete answer. Descriptions are
   contextual; output options are fixed.
3. Normalize the complete reported distribution and extend competing paths with
   positively scored tokens. Keep STOP candidates separately. A character cap
   restricts local expansion, never the options sent to Jev.
4. At each token depth, retain at most `--beam-width` live paths. Default
   `--beam-diversity token` first reserves the best path for distinct ending
   tokens, as width permits, then fills spare slots by cumulative probability.
   `--beam-diversity none` uses probability alone.
5. Select the best STOP-terminated path by mean normalized log probability,
   **including STOP**. If none exists, preserve the best available partial path.
   Only this final path is emitted; speculative branches are not streamed.

For a path of $N$ selected actions, including STOP when present, the score is
$\frac{1}{N}\sum_t \log(q_t / \sum_v q_{t,v})$. The sum in each denominator
includes the **whole fixed vocabulary**, including options that exceed a local
character cap. Diversity changes pruning, not these model scores. Same-text paths
merge only within a token-depth layer; alternate fragment tokenizations can share
a cached prefix evaluation without sharing their accumulated path score.

This is bounded heuristic search, not an optimality or correctness guarantee.
Length normalization can favor longer sequences; diversity can discard a strong
path to preserve coverage. A positively scored STOP is not proof of a correct
answer. Completed paths take precedence over partial paths, even at a search cap.

There are no word shortlists, answer-specific vocabularies, repetition bans,
additional language models, or hidden lookups. The same alphabet and optional
token file give different prompts the same output options. `--search greedy`
retains the lower-cost, irreversible local-selection path and supports sampling.

These are scores over **our supplied candidate set**, not Jev's native
next-token distribution. Search can recover paths that greedy decoding loses;
it cannot turn an unreliable scorer into a reliably fluent language model.

TypeSafe's [public preview terms](https://typesafe.ai/terms) restrict use to
evaluation and prohibit benchmark publication, distillation, and developing
similar or competing products/services. Keep results private and clarify your
account's applicable terms before publication, training, or commercialization.
See [SOURCES.md](SOURCES.md) for the official model, API, pricing, and terms sources.

## Start

Open a terminal in `typesafe_llm` and set the key:

```bash
export TYPESAFE_API_KEY='your-api-key'
python typesafe_llm.py 'What is the opposite of hot? Answer with one lowercase word.' --max-calls 256 --max-new-chars 10
```

On Windows PowerShell, set the variable with:

```powershell
$env:TYPESAFE_API_KEY = 'your-api-key'
python typesafe_llm.py 'What is the opposite of hot? Answer with one lowercase word.' --max-calls 256 --max-new-chars 10
```

This project does not automatically load `.env` files. For your own trusted
`.env` file on Linux/macOS, load it into the environment first:

```bash
chmod 600 .env
set -a
. ./.env
set +a
```

Do not paste your key into source code or commit it. `.env`, `.env.*`, and
`runs/` are ignored by Git. Never force-add them.

In beam mode, stdout receives the chosen answer after search finishes. Greedy
mode streams each committed token. Progress and the run path go to stderr.
`candidate.txt` holds provisional beam results; `answer.txt` holds the selected
output. The example prompt is not a guarantee that the model answers correctly.

## Useful runs

### Inspect one step in the playground, without an API call

```bash
python typesafe_llm.py 'What is the opposite of hot? One lowercase word.' --dry-run > request.json
```

The result contains the full HTTP body: `model`, `state`, and `questions`. In the
playground, copy its `state` and `questions` values into the corresponding fields.
No key is needed for `--dry-run`. It prints the actual first request without
generating an answer. `examples/first_request.json` contains a complete example.
For another prefix, use `--prefix ... --dry-run`: both state and candidate
descriptions must update. Keep the same alphabet and token mapping.

### See each token's alternatives

```bash
python typesafe_llm.py 'What is the opposite of hot? One lowercase word.' --verbose --max-calls 256 --max-new-chars 10
```

### Continue a phrase rather than starting from nothing

```bash
python typesafe_llm.py 'Complete this statement correctly.' --prefix 'the opposite of hot is ' --max-new-chars 12 --max-calls 256
```

### Low-cost greedy decoding or local sampling

Use `--search greedy --temperature 0` to commit the highest-scored local choice.
For sampling:

```bash
python typesafe_llm.py 'Write one short sentence about rain. Use lowercase.' --search greedy --temperature 0.8 --top-k 5 --top-p 0.95 --seed 42 --max-calls 81 --max-new-chars 80
```

In greedy mode at temperature zero, the decoder selects the highest reported probability;
Jev's selected label breaks exact ties. At positive temperatures, it computes
weights proportional to `p ** (1 / temperature)`, restricts to top-k if requested,
applies the top-p cutoff, renormalizes, and samples locally. Zero-probability
options remain impossible. Raw values are saved separately from this policy.
Sampling controls are rejected in beam mode instead of being silently ignored.

The seed controls local sampling and option shuffling, not server-side behavior.
Identical seeds do not guarantee identical text when API responses differ.

### Allow capitals, numbers, punctuation, and newlines

```bash
python typesafe_llm.py 'What is 3 times 4? Answer using digits only.' --alphabet ascii --max-calls 256 --max-new-chars 6
```

The default `lower` alphabet has 29 options: a-z, space, period, and STOP.
`ascii` has 97: all printable ASCII characters, newline, and STOP.
TypeSafe documents a maximum of **255 Choice options**, including STOP; the
decoder rejects larger vocabularies locally. More options do not necessarily
improve generation quality. The prefix appears in `state` and each contextual
candidate description, so larger vocabularies and longer prefixes increase
request size substantially.

### Use a smaller custom alphabet

```bash
python typesafe_llm.py 'Compute three times four. Return only the number.' --characters-json examples/math_characters.json --max-calls 256 --max-new-chars 6
```

A character file is a JSON string, or a JSON array of one-character strings.
Duplicates are removed. STOP is always added. Unicode characters are supported
by the wrapper, but live model support is unverified. Terminal escape/control
characters other than tab/newline are rejected.

### Add multi-character fragments

```bash
python typesafe_llm.py 'Write one short sentence about rain.' --tokens-json examples/subword_tokens.json --max-calls 256 --max-new-chars 80
```

The token file is a JSON array of nonempty strings. Fragments are added to the
character alphabet, not substituted for it, so single-character fallback remains.
Duplicates are removed before enforcing the 255-option limit. Whitespace is
literal; an ordinary fragment `"STOP"` is distinct from the STOP action.

The example contains fixed common fragments, not answers to the benchmark.
Do not derive a vocabulary from expected answers: that leaks scoring data into
generation. Fragments may reduce calls per emitted character, but can also make
answers worse. They are optional and are not Jev's internal tokenizer.
All supplied fragments remain available on every step, not only when they match
a word prefix. Omit `--tokens-json` for strictly one-character emissions.


### Revisit the payoff experiment

```bash
python typesafe_llm.py 'Compare the expected points for red and blue. Give a short calculation, then name the higher one. Use lowercase words and spell out numbers.' --context-json examples/payoff_context.json --max-new-chars 100 --max-calls 512
```

The original bag and payoff state is passed as `context`, alongside the prompt
and growing answer. No correct calculation is inserted by the wrapper. Any
resulting explanation is model-generated text, not a readout of hidden reasoning.
Do not treat it as ground truth without checking it.

### Test option-order sensitivity

```bash
python typesafe_llm.py 'What is the opposite of hot? One lowercase word.' --shuffle-options --seed 42 --max-calls 256 --max-new-chars 10
```

The options are shuffled **once per run**, then that order is reused. Exact order
is recorded in every request. A separate random-number generator keeps option
shuffling from consuming the sampling generator's draws.

### Continue a saved partial answer

Replace `RUN_FOLDER` with the run path printed by the program:

```bash
python typesafe_llm.py 'Your original prompt' --prefix-file RUN_FOLDER/answer.txt --max-calls 256
```

Use the original prompt, context, alphabet, and relevant decoding settings. This
starts a **new** run with a new budget and an immutable initial prefix. It does
not restore a saved search frontier, cache, or random-number-generator state.
The old run is not overwritten.

### Check available model names

```bash
python typesafe_llm.py --list-models
```

This makes one authenticated GET request, prints the JSON response, and exits.
It is separate from a generation run. The wrapper never silently falls back to
a different model when the requested model fails.

## Bounded evaluation campaigns

Inspect a complete plan without a key or network calls:

```bash
python benchmark.py --dry-run --repeats 2 --max-total-calls 7
```

Compare contextual greedy decoding with token-diverse search on the same suite:

```bash
python benchmark.py examples/search_holdout.json --search greedy --max-total-calls 64 --max-calls 16 --max-new-chars 10 --seed 42
python benchmark.py examples/search_holdout.json --search beam --beam-width 32 --beam-diversity token --max-total-calls 1024 --max-calls 256 --max-new-chars 10 --seed 42
```

The search policy was fixed before the first evaluation of `search_holdout.json`.
After that evaluation it, like `holdout.json` and `validation.json`, is no longer
untouched data for future tuning. These small suites are diagnostics, not a
general language-model benchmark.

Both commands require the environment key. Each has its **own** campaign budget.
`--max-total-calls` caps attempts across every case and repetition, including
failed attempts. Per-sample limits are clipped to the remaining shared budget.
Errors and interruption stop the campaign; no retries or automatic resumption.

The campaign default shared cap is only **100 attempts**, even though a single
beam sample permits up to 256. Inspect `--dry-run` and set a deliberate campaign
budget; a small shared cap can leave later cases unrun.

Other controls include `--repeats`, `--model`, `--beam-width`, `--beam-diversity`,
greedy sampling controls, `--shuffle-options`, `--instructions-file`, and
`--tokens-json`. Seeds control local sampling and option order, not the server.
Do not tune against a holdout and then describe it as unseen evaluation.

Suite JSON contains `cases`, with explicit unique `id`, `prompt`, and an
`expected` array of accepted full answers. Optional case fields: `prefix`,
`context`, and `max_new_chars`. A top-level `characters` selects one shared
alphabet; otherwise printable ASCII plus newline is used. Per-case character
overrides are rejected. Every case and repeat uses the same label-to-output
mapping; descriptions reflect its own candidate prefix. All cases and the shared
vocabulary are validated before the first call.

Accepted answers must cover valid alternatives: the young-dog case accepts both
`pup` and `puppy`. That list was corrected after its first evaluation; original
local results remain unchanged. Strict matching is not a semantic grader.

Expected answers remain exclusively on the scoring side: they are never sent
in API requests or generation configuration. Strict equality includes any
initial prefix, spaces, capitalization, and newlines. A correct-looking answer
cut off without STOP is **not** a completed match. A successful beam result must
contain a positively scored STOP, not necessarily Jev's local argmax choice.

Each campaign saves atomic `manifest.json` and `results.json` under `runs/`,
plus a normal decoder run per sample. Results include every scheduled case,
even those not run. `completed_accuracy` is completed exact matches divided by
**all scheduled samples**; `accuracy_among_completed` and coverage are separate.
Raw matches, outputs, stop reasons, attempted calls, elapsed time, reported usage,
and resolved model versions remain available for inspection.

Live calibration and holdout evaluation have been run locally. Their private
artifacts, including failures, are deliberately not committed. The included
suites are small diagnostics, not evidence of general language-model quality.

## Saved files

Each run creates a unique subfolder under `runs/` (change the parent using
`--out-dir`). Existing runs are never overwritten.

| File | Contents |
|---|---|
| `answer.txt` | Initial prefix plus selected output; atomically checkpointed after each committed token. No speculative beam text. |
| `candidate.txt` | Beam only: provisional best result at layer boundaries and finalization. It may change before final selection. |
| `trace.jsonl` | Exact requests/responses, request IDs, latency, cache hits, search frontiers, and final-path decisions. |
| `config.json` | Decoder/trace versions, prompt, prefix, context, fixed vocabulary, instructions, search policy, and seed. |
| `summary.json` | Output/search stop reasons, attempts, responses, sequence score, cache counts, emitted text counts, usage, and model versions. |

The trace distinguishes the API-selected label from the locally selected label,
and raw Choice probabilities from the normalized/filtered decoding policy.
Rounding can make raw values sum to something other than exactly one. That sum
is retained; no corrected value silently replaces the original data.

Trace schema 5 records every attempted request and received response. `frontier`
events describe explored hypotheses; `decision` events contain only the final
selected path in beam mode. Their `call` references identify the actual scored
prefix, including reused evaluations from alternate tokenizations. They need not
be monotonically increasing. Frontier token counts include STOP when present;
summary `new_tokens` counts emitted text tokens only.

Decisions include the fixed vocabulary, `selected_text`, actual `emitted_text`,
selection method, raw scores, and cumulative path log probability. Beam selection
is deterministic, so its local selection distribution is one-hot, not an
additional model probability. A fragment crossing the character cap is never
split: greedy decoding stops without emitting it; beam search does not expand
that edge, while retaining the full distribution for scoring. Historical traces
remain readable.

Usage totals sum only reported values. `responses_missing_usage` identifies how
many otherwise received responses omitted each token count. Failed/timed-out
requests may not report usage. These totals are **not a billing reconciliation**.

**These are plaintext research logs containing your prompt, context, and generated
text.** API keys and Authorization headers are not written to them. Keep traces
private and check the applicable TypeSafe terms before sharing performance data.

## Inspect a trace without more API calls

```bash
python inspect_trace.py RUN_FOLDER
python inspect_trace.py RUN_FOLDER/trace.jsonl --frontiers --top 8 --limit 0
```

The table shows each picked label, the API's original selection, raw probability,
local selection probability, entropy, and the top alternatives. Entropy is over
the normalized Choice distribution. It is **not native next-token entropy**.
`--frontiers` shows provisional hypotheses separately from the selected path.
The footer distinguishes the search termination reason from the selected path's
STOP and reports cache reuse and sequence score.

## Budgets, stopping, and errors

Greedy mode uses one call per selected token or STOP. Beam mode uses a call per
**uncached explored prefix**, not per emitted character. A width-$B$ character
search to $L$ new characters can evaluate up to $1 + BL$ prefixes before the
request cap intervenes. Defaults are width 32, token diversity, 256 attempts,
and 80 new characters; that budget does not guarantee reaching 80 characters.

The fixed options remain offered at every boundary, including the character cap.
Prefix evaluations are cached only within the same run, prompt, context, model,
instructions, vocabulary, and option order. Cached calls cost no new request.
HTTP requests are sequential; separate prefixes never share a speculative state.

Each request includes contextual candidate prefixes. There is no tokenizer-based
context-window guard; start with short outputs and inspect request sizes.

`--max-calls` is a hard cap on attempted generation HTTP calls, including failed
calls. There are **no automatic retries or redirects**. A timeout, HTTP error,
malformed probability distribution, or network error ends the run and preserves
the best observed search result or committed greedy output. `--timeout` limits network operations, not the wall-clock time
for the whole run.

Call caps are not dollar caps. Published example prices differ by date/model,
and the current account-specific input-token price has not been verified.
Reported usage is recorded without inventing a USD charge or assuming failed
requests are free. Confirm pricing and billing controls in your TypeSafe account
before scaling; available credits are not a reason to launch an unbounded run.

`--max-new-chars` limits emitted Unicode characters, excluding an existing prefix.
A cap does not by itself mean Jev finished its answer. In beam mode,
`search_stop_reason` records why exploration ended; `stop_reason` is `stop` when
the selected path contains STOP and no error/interruption occurred. A completed
path can therefore coexist with `search_stop_reason=max_calls`; search was not
exhaustive. Without a completion, the best observed partial result is retained.

The wrapper never forces minimum length, bans repetition, fabricates STOP, or
repairs output. Ctrl+C preserves observed work and exits with code 130. API
errors use code 1; selected STOP and intentional caps use code 0. During beam
search, `candidate.txt` provides the last checkpoint if the process is forcibly
killed; there is no automatic search-state resumption.

The default endpoint is the official API root. `--base-url` or
`TYPESAFE_BASE_URL` can select another HTTPS root, for example an authorized
proxy. Your key goes to that selected host, so only configure a trusted endpoint.
Redirects are blocked instead of forwarding credentials elsewhere.

## Hacking the question itself

`INSTRUCTIONS` holds the one next-token question. `--instructions-file` overrides
it for the entire run; the same instructions are used for every prefix.
Instructions, exact requests, and local decisions are retained in the run trace.

## Tests and verification

```bash
python -m unittest discover -s tests -v
```

The offline suite covers fixed output membership and order, shared benchmark
vocabularies, diverse and ordinary beam coverage, duplicate tokenizations,
cache provenance, mid-frontier failure preservation, sampling, malformed scores,
token boundaries, STOP, interruptions, hard budgets, truncated HTTP bodies,
credential handling, expected-answer isolation, and completion-aware scoring.
GitHub Actions runs these checks and both CLI dry runs on Python 3.10, 3.12,
and 3.14, without an API key or paid calls.

Scripted evaluators and mocked HTTP responses are labeled offline fixtures;
their outputs are not measurements of Jev. Live experiments use the real API
and save separate local traces. Request/response contracts and model limitations
are documented in [SOURCES.md](SOURCES.md).

## Changes in 5.0

- Contextual candidate descriptions with an unchanged output vocabulary.
- Bounded token-diverse beam search; probability-only beam and greedy controls.
- Separate live and completed paths, length-normalized sequence scoring, and
  parent-linked decision provenance without per-branch probability copies.
- Run-local prefix caching and preservation of observed work at limits/errors.
- Provisional beam checkpoints, final-path-only output, frontier inspection, and
  distinct search/output termination reasons.
- Trace schema 5, campaign schema 3, and a separately frozen diagnostic suite.

## Changes in 4.0

- Fixed option labels, descriptions, and membership across every step and prompt.
- One shared benchmark vocabulary; per-case alphabet overrides are rejected.
- Single-character default; explicitly configured fragments remain fixed options.
- Removed dictionary data, proposal stages, and the `--decoder`/`--lexicon` flags.
- Option shuffling happens once per run rather than changing each request.
- Trace schema 4 and campaign schema 2 identify the fixed-vocabulary cutover.

## Changes in 3.0

Version 3 introduced dictionary-assisted word shortlists. That changed the
candidate set as the prefix grew, so it was not a fixed-vocabulary experiment.
This approach was removed in version 4; its assisted results are not evidence
of improved character-by-character prediction.

## Changes in 2.0

- Current documented model alias with resolved versions retained in traces.
- Full-prefix candidate scoring instead of bare next-character descriptions.
- Optional text fragments, character fallback, and the documented Choice limit.
- Token-aware checkpoints and traces without splitting capped fragments.
- Bounded benchmark campaigns, strict scoring, and separate holdout cases.
- Clean errors for truncated HTTP responses, with no automatic retries.
