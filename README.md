# TypeSafe -> a tiny text generator

An experimental, character-by-character decoder for Jev. It asks a System One
Choice question over an alphabet plus STOP, appends one selected character to
`answer_prefix`, and repeats. No other model proposes the answer.

**Python 3.10+; no third-party packages or SDK installation required.**
You need your own TypeSafe API key. The default model is `jev-latest`, as
recommended by TypeSafe's current API documentation. Use `--list-models` and
`--model` to pin an available version for repeatable comparisons.

This is a runnable wrapper, not a claim that Jev is a good text generator. It
does not expose weights, hidden activations, native token probabilities, or the
RLCD reward function. Generation quality needs to be measured on your account.

## Start

Extract the archive, open a terminal in `typesafe_llm`, and set the key:

```bash
export TYPESAFE_API_KEY='your-api-key'
python typesafe_llm.py 'What is the opposite of hot? Answer with one lowercase word.' --max-calls 12 --max-new-chars 10
```

On Windows PowerShell, set the variable with:

```powershell
$env:TYPESAFE_API_KEY = 'your-api-key'
python typesafe_llm.py 'What is the opposite of hot? Answer with one lowercase word.' --max-calls 12 --max-new-chars 10
```

This project does not automatically load `.env` files. Do not paste your key into
source code or commit it to a repository.

Generated characters stream to stdout. Progress, diagnostics, and the run's file
path go to stderr. Each run also saves the complete answer automatically. The
example question is a test prompt, not a guarantee that the model answers it.

## Useful runs

### Inspect one step in the playground, without an API call

```bash
python typesafe_llm.py 'What is the opposite of hot? One lowercase word.' --dry-run > request.json
```

The result contains the full HTTP body: `model`, `state`, and `questions`. In the
playground, copy its `state` and `questions` values into the corresponding fields.
No key is needed for `--dry-run`. A ready-made example is in
`examples/first_request.json`.

### See each character's alternatives

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
`ascii` has 97: all printable ASCII characters, newline, and STOP. The live
service's acceptance/performance with this many options has not been verified.
Use `lower` or a smaller custom alphabet when your deployment rejects a larger
candidate set. Even successful requests may become less reliable as the option
set changes; comparing them is an experiment, not a guaranteed upgrade.

### Use a smaller custom alphabet

```bash
python typesafe_llm.py 'Compute three times four. Return only the number.' --characters-json examples/math_characters.json --max-calls 8
```

A character file is a JSON string, or a JSON array of one-character strings.
Duplicates are removed. STOP is always added. Unicode characters are supported
by the wrapper, but live model support is unverified. Terminal escape/control
characters other than tab/newline are rejected.

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

## Saved files

Each run creates a unique subfolder under `runs/` (change the parent using
`--out-dir`). Existing runs are never overwritten.

| File | Contents |
|---|---|
| `answer.txt` | Initial prefix plus all generated characters; checkpointed after each character. |
| `trace.jsonl` | Exact request bodies, response bodies, request IDs, latency, and every local character decision. |
| `config.json` | Prompt, initial prefix, context, alphabet, instructions, decoding parameters, and seed. |
| `summary.json` | Stop reason, attempted calls, received responses, characters, reported token totals, and observed model versions. |

The trace distinguishes the API-selected label from the locally selected label,
and raw Choice probabilities from the normalized/filtered decoding policy.
Rounding can make raw values sum to something other than exactly one. That sum
is retained; no corrected value silently replaces the original data.

Usage totals sum only reported values. `responses_missing_usage` identifies how
many otherwise received responses omitted each token count. Failed/timed-out
requests may not report usage. These totals are **not a billing reconciliation**.

**These are plaintext research logs containing your prompt, context, and generated
text.** API keys and Authorization headers are not written to them. Share or
commit traces only after checking their contents.

## Inspect a trace without more API calls

```bash
python inspect_trace.py RUN_FOLDER
python inspect_trace.py RUN_FOLDER/trace.jsonl --top 8 --limit 0
```

The table shows each picked label, the API's original selection, raw probability,
local selection probability, entropy, and the top alternatives. Entropy is over
the normalized Choice distribution. It is **not native next-token entropy**.

## Budgets, stopping, and errors

One emitted character requires one successful evaluation request. STOP also
requires a request but adds no character. Producing N characters and then
observing STOP can therefore take N+1 requests. Each request resends the growing
prefix and all options. Start with tiny answers.

`--max-calls` is a hard cap on attempted generation HTTP calls, including failed
calls. There are **no automatic retries or redirects**. A timeout, HTTP error,
malformed probability distribution, or network error ends the run and preserves
previous output. `--timeout` limits network operations, not the wall-clock time
for the whole run.

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

`INSTRUCTIONS` in `typesafe_llm.py` holds the next-character question. Change it
in code, or supply a UTF-8 file through `--instructions-file`. The full text is
recorded for every run. This can change the behavior substantially.

## Tests and verification

```bash
python -m unittest discover -s tests -v
```

The included suite has **41 passing offline tests** in the supplied version.
It covers request construction, authentication-header placement, non-retry
behavior, redirects, sampling, temperature, top-k/top-p, invalid distributions,
character/HTTP caps, STOP, interruption, saved-prefix continuation, traces, and
the trace inspector.

Tests use mocked HTTP responses and a scripted fake evaluator. Any `cold`
output in the fixtures demonstrates decoder plumbing only; it is **not a live
Jev result**. No authenticated live integration test was performed. HTTP
contracts were checked against TypeSafe's official SDK source, recorded in
`SOURCES.md`.
