# TypeSafe autoregressive decoder

An experimental text generator built from Jev's System One Choice evaluations.
The default **word-guided decoder** asks Jev to rank dictionary continuations,
then choose a shortlisted word, an original character/fragment, or STOP.
Selected text is appended to `answer_prefix`, and the process repeats.
No other model proposes the answer, and no spelling is repaired after selection.

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

Every call resends the original prompt, optional task `context`, and the **full
current `answer_prefix`**. Context is not forgotten between letters. There is no
server-side conversation memory, automatic context truncation, or summarization.

At the start of an answer or a word boundary, selection uses the original
vocabulary first. Dictionary ranking begins only after an actual word prefix
exists; this avoids seeding answers from an unrelated global-frequency shortlist.

The default `--decoder lexical` uses two stages when lexical ranking is needed:

1. Find up to 1,024 frequency-ranked dictionary continuations of the current
   word. Ask independent Choice questions in groups of up to 128 over the same
   actual state. The supplied 20,000-word English lexicon is independent of Jev
   outputs and benchmark answers.
2. Keep up to two candidates from each group. Ask a final Choice over those
   candidates, **all original character/fragment options**, and STOP.

For example, after `b`, a shortlisted word may offer the suffix `erlin`, producing
the candidate prefix `berlin`. It is still Jev—not a lookup table of answers—that
must select that candidate. Original options such as `a` remain selectable.
Unknown words retain character fallback, and selected words are never split,
rewritten, or automatically followed by spaces.

Proposal probabilities are only used to rank candidates within their own groups;
they are not averaged across groups or substituted for final Choice probabilities.
Conditioning every selection on previous emitted text makes the process
autoregressive. These are probabilities over **our candidate set**, not Jev's
native vocabulary or a guarantee of correctness.

Use `--decoder character` for the direct-selection baseline, without lexical
ranking. Optional `--tokens-json` fragments work in either mode.

Word guidance is an accuracy/latency tradeoff: it sends larger requests and often
uses two calls per selection, so it can be substantially slower. It can still
produce wrong facts, miss punctuation/capitalization, or choose bad fragments.
The English lexicon is a proposal source, not a universal vocabulary or a trained
language model. Shortlist truncation and corpus frequency introduce their own bias.

TypeSafe's [public preview terms](https://typesafe.ai/terms) restrict use to
evaluation and prohibit benchmark publication, distillation, and developing
similar or competing products/services. Keep results private and clarify your
account's applicable terms before publication, training, or commercialization.
See [SOURCES.md](SOURCES.md) for the official model, API, pricing, and terms sources.

## Start

Open a terminal in `typesafe_llm` and set the key:

```bash
export TYPESAFE_API_KEY='your-api-key'
python typesafe_llm.py 'What is the opposite of hot? Answer with one lowercase word.' --max-calls 12 --max-new-chars 10
```

On Windows PowerShell, set the variable with:

```powershell
$env:TYPESAFE_API_KEY = 'your-api-key'
python typesafe_llm.py 'What is the opposite of hot? Answer with one lowercase word.' --max-calls 12 --max-new-chars 10
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

Generated text streams to stdout. Progress, diagnostics, and the run's file
path go to stderr. Each run also saves the complete answer automatically. The
example question is a test prompt, not a guarantee that the model answers it.

## Useful runs

### Inspect one step in the playground, without an API call

```bash
python typesafe_llm.py 'What is the opposite of hot? One lowercase word.' --dry-run > request.json
```

The result contains the full HTTP body: `model`, `state`, and `questions`. In the
playground, copy its `state` and `questions` values into the corresponding fields.
No key is needed for `--dry-run`. It prints the actual first request: normally
original-vocabulary selection for an empty prefix, or word ranking for an existing
word prefix. It never fabricates rankings or a final answer. `examples/first_request.json`
is a smaller direct-selection example, corresponding to `--decoder character`.

### See each token's alternatives

```bash
python typesafe_llm.py 'What is the opposite of hot? One lowercase word.' --verbose --max-calls 12
```

### Continue a phrase rather than starting from nothing

```bash
python typesafe_llm.py 'Complete this statement correctly.' --prefix 'the opposite of hot is ' --max-new-chars 12 --max-calls 13
```

### Local sampling instead of greedy selection

```bash
python typesafe_llm.py 'Write one short sentence about rain. Use lowercase.' --temperature 0.8 --top-k 5 --top-p 0.95 --seed 42 --max-calls 81 --max-new-chars 80
```

At temperature zero, the decoder selects the highest reported probability;
Jev's selected label breaks exact ties. At positive temperatures, it computes
weights proportional to `p ** (1 / temperature)`, restricts to top-k if requested,
applies the top-p cutoff, renormalizes, and samples locally. Zero-probability
options remain impossible. Raw values are saved separately from this policy.

The seed controls local sampling and option shuffling, not server-side behavior.
Identical seeds do not guarantee identical text when API responses differ.

### Allow capitals, numbers, punctuation, and newlines

```bash
python typesafe_llm.py 'What is 3 times 4? Answer using digits only.' --alphabet ascii --max-calls 8 --max-new-chars 6
```

The default `lower` alphabet has 29 options: a-z, space, period, and STOP.
`ascii` has 97: all printable ASCII characters, newline, and STOP.
TypeSafe documents a maximum of **255 Choice options**, including STOP; the
decoder rejects larger vocabularies locally. More options do not necessarily
improve generation quality. Each candidate repeats the current answer prefix,
so request size grows with both answer length and vocabulary size.

### Use a smaller custom alphabet

```bash
python typesafe_llm.py 'Compute three times four. Return only the number.' --characters-json examples/math_characters.json --max-calls 8
```

A character file is a JSON string, or a JSON array of one-character strings.
Duplicates are removed. STOP is always added. Unicode characters are supported
by the wrapper, but live model support is unverified. Terminal escape/control
characters other than tab/newline are rejected.

### Compare word guidance with direct selection

```bash
python typesafe_llm.py 'What is the capital of Germany? Reply with one lowercase word.' --decoder lexical --max-calls 12 --max-new-chars 15 --verbose
python typesafe_llm.py 'What is the capital of Germany? Reply with one lowercase word.' --decoder character --max-calls 12 --max-new-chars 15 --verbose
```

`--lexicon PATH` selects a custom JSON object with a nonempty `words` array.
Words must contain lowercase ASCII letters only, ordered from most to least
frequent. Matching continuations preserve an already capitalized/uppercase word
prefix. The source file and SHA-256 are recorded in each run and benchmark plan.
No file is downloaded or model trained at startup.

The bundled `data/english_words.json` is adapted from Hermit Dave's
[FrequencyWords / OpenSubtitles2018](https://github.com/hermitdave/FrequencyWords),
under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
It retains the first 20,000 lowercase ASCII alphabetic entries in source frequency
order, omitting counts and other entries. Attribution and transformations are also
recorded in the data file. Do not build a lexicon from benchmark answers.

### Add multi-character fragments

```bash
python typesafe_llm.py 'Write one short sentence about rain.' --tokens-json examples/subword_tokens.json --max-calls 40 --max-new-chars 80
```

The token file is a JSON array of nonempty strings. Fragments are added to the
character alphabet, not substituted for it, so single-character fallback remains.
Duplicates are removed before enforcing the 255-option limit. Whitespace is
literal; an ordinary fragment `"STOP"` is distinct from the STOP action.

The example contains fixed common fragments, not answers to the benchmark.
Do not derive a vocabulary from expected answers: that leaks scoring data into
generation. Fragments may reduce calls per emitted character, but can also make
answers worse. They are optional and are not Jev's internal tokenizer.

### Revisit the payoff experiment

```bash
python typesafe_llm.py 'Compare the expected points for red and blue. Give a short calculation, then name the higher one. Use lowercase words and spell out numbers.' --context-json examples/payoff_context.json --max-new-chars 100 --max-calls 101
```

The original bag and payoff state is passed as `context`, alongside the prompt
and growing answer. No correct calculation is inserted by the wrapper. Any
resulting explanation is model-generated text, not a readout of hidden reasoning.
Do not treat it as ground truth without checking it.

### Test option-order sensitivity

```bash
python typesafe_llm.py 'What is the opposite of hot? One lowercase word.' --shuffle-options --seed 42 --max-calls 12
```

Each request's exact option order is recorded. Option shuffling has a separate
random-number generator, so it does not consume the sampling generator's draws.

### Continue a saved partial answer

Replace `RUN_FOLDER` with the run path printed by the program:

```bash
python typesafe_llm.py 'Your original prompt' --prefix-file RUN_FOLDER/answer.txt --max-calls 40
```

Use the original prompt, context, alphabet, and relevant decoding settings. This
starts a **new** run with a new request budget; it is not an exact continuation
of a saved random-number-generator state. The old run is not overwritten.

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

Run a development comparison, then evaluate the separate lexical validation suite:

```bash
python benchmark.py examples/benchmark.json --max-total-calls 96 --max-calls 16 --max-new-chars 15 --seed 42
python benchmark.py examples/lexical_holdout.json --max-total-calls 128 --max-calls 16 --max-new-chars 15 --seed 42
```

`examples/holdout.json` retains the earlier diagnostic set. Its cases were later
used during diagnosis, so it is no longer an unseen holdout for version 3.
The lexical suite was initially held out, then reused to verify the word-boundary
correction. Treat it as validation, not a fresh test set for subsequent tuning.

Both commands require the environment key. Each has its **own** campaign budget.
`--max-total-calls` caps attempts across every case and repetition, including
failed attempts. Per-sample limits are clipped to the remaining shared budget.
Errors and interruption stop the campaign; no retries or automatic resumption.

Use `--decoder character` to compare the earlier policy, or `--lexicon` to select
another independent vocabulary. Other controls include `--repeats`, `--model`,
`--temperature`, `--top-k`, `--top-p`, `--shuffle-options`, `--instructions-file`,
and `--tokens-json`. Seeds control local sampling and option order, not the server.
Do not tune against a holdout and then describe it as unseen evaluation.

Suite JSON contains `cases`, with explicit unique `id`, `prompt`, and an
`expected` array of accepted full answers. Optional fields: `prefix`, `context`,
`characters`, and `max_new_chars`. A top-level `characters` selects a shared
alphabet; otherwise printable ASCII plus newline is used. All cases and
vocabularies are validated before the first call.

Expected answers remain exclusively on the scoring side: they are never sent
in API requests or generation configuration. Strict equality includes any
initial prefix, spaces, capitalization, and newlines. A correct-looking answer
cut off by a cap is **not** a completed match; only model-selected STOP counts.

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
| `answer.txt` | Initial prefix plus emitted text; atomically checkpointed after each token. |
| `trace.jsonl` | Exact request/response bodies, request IDs, latency, and local decisions. |
| `config.json` | Decoder/trace versions, mode, lexicon source/hash, prompt, prefix, context, base vocabulary, instructions, policy, and seed. |
| `summary.json` | Stop reason, attempts, responses, emitted tokens/characters, reported usage, and observed model versions. |

The trace distinguishes the API-selected label from the locally selected label,
and raw Choice probabilities from the normalized/filtered decoding policy.
Rounding can make raw values sum to something other than exactly one. That sum
is retained; no corrected value silently replaces the original data.

Trace schema 3 identifies `proposals` and `selection` request/response stages.
Final decisions include the exact step vocabulary, `selected_text`, and actual
`emitted_text`. Proposal winners do not themselves emit text. If a selected
fragment would exceed the character cap, the decoder emits nothing and stops;
it never splits or resamples the fragment. The inspector displays actual candidate
text and can still inspect older character traces.

Usage totals sum only reported values. `responses_missing_usage` identifies how
many otherwise received responses omitted each token count. Failed/timed-out
requests may not report usage. These totals are **not a billing reconciliation**.

**These are plaintext research logs containing your prompt, context, and generated
text.** API keys and Authorization headers are not written to them. Keep traces
private and check the applicable TypeSafe terms before sharing performance data.

## Inspect a trace without more API calls

```bash
python inspect_trace.py RUN_FOLDER
python inspect_trace.py RUN_FOLDER/trace.jsonl --top 8 --limit 0
```

The table shows each picked label, the API's original selection, raw probability,
local selection probability, entropy, and the top alternatives. Entropy is over
the normalized Choice distribution. It is **not native next-token entropy**.

## Budgets, stopping, and errors

Direct selection uses one call per selected token or STOP. Word-guided selection
uses a ranking call plus a final-selection call when there are word groups to
rank; zero/singleton proposal groups do not need a ranking call. Both stages count
toward the same budget. If fewer than two attempts remain for a two-stage step,
generation stops without spending the last attempt on unusable proposals.

Each request includes the growing prefix. Word-group descriptions repeat it many
times, so long outputs can approach the provider's request-size limit sooner.
No tokenizer-based context-window guard is available; API errors preserve prior
output. Start with short outputs. `--timeout 90` can allow more time for large
ranking requests without enabling retries.

`--max-calls` is a hard cap on attempted generation HTTP calls, including failed
calls. There are **no automatic retries or redirects**. A timeout, HTTP error,
malformed probability distribution, or network error ends the run and preserves
previous output. `--timeout` limits network operations, not the wall-clock time
for the whole run.

Call caps are not dollar caps. Published example prices differ by date/model,
and the current account-specific input-token price has not been verified.
Reported usage is recorded without inventing a USD charge or assuming failed
requests are free. Confirm pricing and billing controls in your TypeSafe account
before scaling; available credits are not a reason to launch an unbounded run.

`--max-new-chars` limits emitted Unicode characters, excluding an existing prefix.
Hitting a cap truncates the experiment; it does not mean Jev selected STOP or
finished its answer. The wrapper intentionally does not suppress an early STOP,
force minimum length, ban repetition, or insert characters automatically: such
changes would alter the experiment. Ctrl+C preserves completed output and exits
with code 130. API errors use code 1; model STOP and intentional caps use code 0.

The default endpoint is the official API root. `--base-url` or
`TYPESAFE_BASE_URL` can select another HTTPS root, for example an authorized
proxy. Your key goes to that selected host, so only configure a trusted endpoint.
Redirects are blocked instead of forwarding credentials elsewhere.

## Hacking the question itself

`INSTRUCTIONS` holds the direct-selection and proposal-ranking question;
`LEXICAL_INSTRUCTIONS` holds the final word-guided selection question.
`--instructions-file` overrides **final selection only**, not word-group ranking.
Instructions, exact requests, and local decisions are retained in the run trace.

## Tests and verification

```bash
python -m unittest discover -s tests -v
```

The offline suite covers sampling, malformed distributions, token boundaries,
STOP, interruptions, hard per-run and campaign budgets, truncated HTTP bodies,
credential handling, expected-answer isolation, and completion-aware scoring.
GitHub Actions runs these checks and both CLI dry runs on Python 3.10, 3.12,
and 3.14, without an API key or paid calls.

Scripted evaluators and mocked HTTP responses are labeled offline fixtures;
their outputs are not measurements of Jev. Live experiments use the real API
and save separate local traces. Request/response contracts and model limitations
are documented in [SOURCES.md](SOURCES.md).

## Changes in 3.0

- Default word-guided decoding with a fixed, attributed English frequency lexicon.
- Model-ranked continuations without answer-specific rules or removal of characters.
- Shared attempt/usage accounting across ranking and final selection.
- Lexicon fingerprints, staged traces, and readable word alternatives.
- Direct-selection comparison mode and fresh lexical holdout cases.

## Changes in 2.0

- Current documented model alias with resolved versions retained in traces.
- Full-prefix candidate scoring instead of bare next-character descriptions.
- Optional text fragments, character fallback, and the documented Choice limit.
- Token-aware checkpoints and traces without splitting capped fragments.
- Bounded benchmark campaigns, strict scoring, and separate holdout cases.
- Clean errors for truncated HTTP responses, with no automatic retries.
