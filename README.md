# AYInference

**Version:** 0.1.0
**Date:** 2026-10-24
**Status:** Active — experimental public surface
**Owner:** AYInference

Optional, headless model invocation. Enable `AY_ENABLE_INFERENCE`, link
`AYInference`, include `AYInference/Inference.h`. This is a separate local Git
repository with no configured remote; the root records its exact commit. A fresh
clone needs this source supplied explicitly until a remote is assigned. The option
defaults OFF and missing sources fail explicitly when enabled.

Supports Mock, Chat Completions-compatible endpoints and OpenAI Responses wire
protocols. Each profile declares model, origin, capability flags, limits and the
environment variable name for a credential. Remote endpoints require verified
HTTPS plus a credential; literal loopback model servers may omit credentials and
use HTTP. No model name, account entitlement, endpoint compatibility or capability
is inferred. `legacyMaxTokens` selects `max_tokens` for compatible local providers;
the default Chat wire field is `max_completion_tokens`.

Requests contain caller messages, an optional output schema and an output-token
limit. Profile request/response byte limits, per-call output-token cap and timeout
are enforced. Token usage is optional and provider-reported; Mock does not invent
token counts. Schema support must be explicitly declared. Unsupported schema
keywords fail before networking; provider schema restrictions may be stricter.

`onText` enables streaming. Deltas are provisional: UI code may preview them, but
must not execute their contents. Only a completed, non-refused, untruncated result
passing local validation returns `Ok`. Chat streaming requires a successful finish
and DONE marker; Responses requires the completed event. Errors/refusals never
return executable output. No hidden reasoning is requested, no tools execute and
no retries or autonomous task creation occur in this module.

`assembleContext` retains identifiers/sources and chooses stable priority fragments
under a byte budget. Inputs must already be authorized. Byte budgets are not model
token-window guarantees. Character state, memory meanings, prompts, permissions,
stage transitions, business schemas and review policy remain outside this module.

```cpp
ayt::inference::ModelProfile model; // Mock, no API/key required
model.mockOutput = R"({"value":"ready"})";
ayt::inference::Request request;
request.messages = {{"user", "Return the requested JSON object"}};
request.outputSchema = R"({"type":"object","properties":{"value":{"type":"string"}},"required":["value"],"additionalProperties":false})";
auto result = ayt::inference::invoke(model, request);
if (result) { /* validate business policy, then commit */ }
```

Tests: `AYInference_Tests`, local fake providers for both protocols and split SSE.
No live provider/key usage is included in the offline gate.

Protocol sources, accessed 2026-10-24:
[Structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs),
[Responses migration](https://developers.openai.com/api/docs/guides/migrate-to-responses).
