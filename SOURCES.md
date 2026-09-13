# Protocol sources

The program uses direct HTTPS requests. These are the official TypeSafe SDK
source files used to verify the HTTP contract, rather than guessing endpoints
or relying on an unverified installed package version.

API root and environment-variable names:

```text
https://raw.githubusercontent.com/typesafe-ai/typesafe-sdk-python/main/src/typesafe_sdk/constants.py
```

POST `/v1/systemone`, GET `/v1/models`, and request-ID header:

```text
https://raw.githubusercontent.com/typesafe-ai/typesafe-sdk-python/main/src/typesafe_sdk/_core/constants.py
```

JSON body fields (`state`, `model`, `questions`):

```text
https://raw.githubusercontent.com/typesafe-ai/typesafe-sdk-python/main/src/typesafe_sdk/_core/endpoints.py
```

Bearer authentication and JSON headers:

```text
https://raw.githubusercontent.com/typesafe-ai/typesafe-sdk-python/main/src/typesafe_sdk/_core/transport.py
```

Choice question shape and optional/undescribed criteria:

```text
https://raw.githubusercontent.com/typesafe-ai/typesafe-sdk-python/main/src/typesafe_sdk/_core/question_types.py
```

Choice response, model metadata, and optional token-usage fields:

```text
https://raw.githubusercontent.com/typesafe-ai/typesafe-sdk-python/main/src/typesafe_sdk/_core/response_types.py
```

These sources establish request/response mechanics, not the model's training
objective, native next-token distribution, or expected text-generation quality.
