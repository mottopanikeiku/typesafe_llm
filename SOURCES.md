# TypeSafe model and protocol sources

This project uses direct HTTPS requests and the Python standard library. The
model/protocol sources below are official TypeSafe documentation, schemas, and SDK code.
Links to `main` and live documentation can change; saved run payloads, decoder
versions, and returned model names record what an experiment actually used.

## Company and model

- [TypeSafe AI](https://typesafe.ai/) is the company/lab.
- [System One](https://docs.typesafe.ai/concepts/system-one.md) describes its
  structured-decision paradigm. Jev is its flagship and first System One model.
- [AI primer](https://docs.typesafe.ai/introduction/machine-learning-primer.md)
  describes Reinforcement Learning for Calibrated Decisions (RLCD), rather than
  optimizing generated text.
- [How to build with System One](https://docs.typesafe.ai/concepts/how-to-build-with-system-one.md)
  recommends narrow questions, relevant state, and evaluation on real domain data.

The documented API provides evaluation and model listing. It does not provide
training/fine-tuning endpoints, weights, hidden activations, a native tokenizer,
or native vocabulary logits. This decoder constructs an autoregressive process
outside Jev by repeatedly asking a new prefix-conditioned question. It neither
trains Jev nor creates a separately trained model from Jev outputs.

## Public API contract

- [HTTP reference](https://docs.typesafe.ai/api.md): authenticated
  `POST https://api.typesafe.ai/v1/systemone`; JSON fields `model`, `state`, and
  `questions`; typed `answers`, model name, and usage in the response.
- [Live OpenAPI schema](https://api.typesafe.ai/openapi.json): evaluation and
  authenticated `GET /v1/models`. The returned model can differ from a requested
  alias, so the decoder records both instead of assuming an immutable alias.
- [Quick start](https://docs.typesafe.ai/introduction/quickstart.md): API-key
  setup, bearer authentication, and the currently recommended `jev-latest` alias.
- [Choice](https://docs.typesafe.ai/primitives/choice.md): up to **255 options**;
  model-visible option names and descriptions; `choice`, `probabilities`, and
  `confidence`. Question IDs are response-routing keys, not model input.
- [Primitives](https://docs.typesafe.ai/primitives.md): all questions in a request
  share the same state and are evaluated independently; the documented request
  budget is approximately 32,000 tokens. The wrapper cannot enforce a precise
  token-size limit without TypeSafe's tokenizer; request errors are preserved.
- [Confidence](https://docs.typesafe.ai/confidence.md): confidence is derived from
  distribution shape and differs from an individual option's probability. It is
  not proof that a generated answer is correct.

Choice probabilities are over the supplied options. They are not the native
next-token probabilities of a generative LLM. Rounding can make the returned
values sum imperfectly; traces preserve the raw values separately from local
normalization, temperature, top-k, and top-p.

## Composition boundaries

[Speculative fan-out](https://docs.typesafe.ai/patterns/fan-out.md) supports many
independent questions over one shared state. It does not establish that asking
about several different candidate prefixes in that state is identical to
separate requests with isolated states. This decoder conditions every emission on
the actual selected prefix. Lexical ranking groups all share that same prefix,
prompt, and context; they do not speculate over different future states.

Lexical mode ranks disjoint groups of dictionary continuations, then makes a
separate final Choice over the shortlisted candidates plus the original vocabulary
and STOP. Probabilities from different groups are not treated as globally
comparable. The final distribution is conditional on the shortlist; even a score
of one does not prove that the answer is factually correct.

[Noul](https://docs.typesafe.ai/primitives/noul.md) returns an independent yes
probability. A set of Noul scores is not a mutually exclusive Choice distribution;
normalizing it locally does not make the two primitives equivalent. The shipped
decoder retains Choice rather than presenting those scores as token probabilities.

[Hierarchical classification](https://docs.typesafe.ai/cookbooks/hierarchical_classification.md)
shows Choice composition and beam search through a taxonomy. That is useful
research context, not evidence that an arbitrary language vocabulary will work
well or that text-generation probabilities are calibrated.

## Independent lexical data

`data/english_words.json` is adapted from Hermit Dave's
[FrequencyWords English 2018 list](https://github.com/hermitdave/FrequencyWords/blob/master/content/2018/en/en_50k.txt),
derived from OpenSubtitles2018. The repository's
[license declaration](https://github.com/hermitdave/FrequencyWords#license)
assigns **CC BY-SA 4.0 to content** (its MIT license applies to code, not the lists).

This adaptation retains the first 20,000 lowercase ASCII alphabetic words in
source frequency order and removes counts and other entries. The data file
records creator, source, changes, and the
[CC BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/).
This data license does not relicense the Python code. Run configuration and
campaign manifests identify the lexicon by source path and SHA-256.

The dictionary was selected independently of evaluation targets and contains no
TypeSafe-generated training examples. It supplies candidate text, not answers or
a hidden language model. English frequency bias, limited candidate coverage,
case handling, and shortlist truncation remain limitations.

## Usage and pricing

The [SDK response implementation](https://github.com/typesafe-ai/typesafe-sdk-python/blob/main/src/typesafe_sdk/_core/response_types.py)
allows input/output token counts to be absent and notes that `billing_units`
required by an older schema are not returned by the API. The decoder does not
invent missing counts or derive charges from that obsolete field.

The live OpenAPI schema describes input tokens as billable and output tokens as
currently free. It does not give a current input-token price. Historical official
cookbooks such as [parallel questions](https://docs.typesafe.ai/cookbooks/parallel_questions.md)
and [Choice consistency](https://docs.typesafe.ai/cookbooks/consistency_choice_cookbook.md)
contain different date/version-specific prices. Those examples do not establish
this account's current rates, credit conversion, failed-request charges, or
numeric rate limits. Verify billing in the account console before large runs.

The program records attempted requests, received responses, reported usage, and
missing usage separately. Request caps are hard; token totals are observations,
not a guaranteed USD spending cap or billing reconciliation.

## SDK source checks

- [API root and environment variables](https://raw.githubusercontent.com/typesafe-ai/typesafe-sdk-python/main/src/typesafe_sdk/constants.py)
- [Endpoint paths and request-ID header](https://raw.githubusercontent.com/typesafe-ai/typesafe-sdk-python/main/src/typesafe_sdk/_core/constants.py)
- [Request fields](https://raw.githubusercontent.com/typesafe-ai/typesafe-sdk-python/main/src/typesafe_sdk/_core/endpoints.py)
- [Authentication and transport](https://raw.githubusercontent.com/typesafe-ai/typesafe-sdk-python/main/src/typesafe_sdk/_core/transport.py)
- [Question types](https://raw.githubusercontent.com/typesafe-ai/typesafe-sdk-python/main/src/typesafe_sdk/_core/question_types.py)
- [Response types and usage](https://raw.githubusercontent.com/typesafe-ai/typesafe-sdk-python/main/src/typesafe_sdk/_core/response_types.py)
- [Generated model metadata](https://github.com/typesafe-ai/typesafe-sdk-python/blob/main/src/typesafe_sdk/_schemas/models.py)

## Applicable terms

The [public preview terms](https://typesafe.ai/terms), last updated November 19,
2025, limit access to evaluation and restrict distillation, training an imitator,
development of similar/competing products or services, and publication of
benchmarks/performance information. They also say preview access may not be
suitable for production. The repository contains evaluation code and fixed task
suites, not a commercial service, a distilled model, or published live benchmark
results. Keep private run artifacts out of Git and clarify any newer written
account agreement before changing that scope.
