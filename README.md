# AYInference

Response wire-size limits include SSE event overhead; `Limit` with an issue at
`/limits/response_bytes` distinguishes this from a caller input/token limit.
No partial text or function proposals are returned on that failure.

**Version:** 0.3.0
**Date:** 2026-10-27
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
token counts. Schema support must be explicitly declared. Profiles without strict
Schema support can opt into `supportsJsonObject`: an object `outputSchema` selects
`json_object` wire mode for Chat or Responses, while the same caller schema remains
mandatory local validation. Strict `supportsJsonSchema` takes precedence. No
capability inference, automatic downgrade or parsing of tool-like prose occurs.
JSON mode requires an explicit JSON instruction in caller messages.
See [DeepSeek JSON Output](https://api-docs.deepseek.com/guides/json_mode/)
(accessed 2026-10-26); provider/model support must still be configured explicitly. Unsupported schema
keywords fail before networking; provider schema restrictions may be stricter.

`onText` enables streaming. Deltas are provisional: UI code may preview them, but
must not execute their contents. Only a completed, non-refused, untruncated result
passing local validation returns `Ok`. Chat streaming requires a successful finish
and DONE marker; Responses requires the completed event. Errors/refusals never
return executable output. No hidden reasoning is requested, no tools execute and
no retries or autonomous task creation occur in this module.

Standard function proposals are separately opt-in through `supportsToolCalls`
and caller-owned `Request::tools`. A definition supplies a portable name,
description and object parameter schema. Chat Completions and Responses each
encode their own function wire shape. Completed calls return as owned
`Result::toolCalls` only after declaration, identity and argument validation;
they are never executed or authorized here. `outputSchema` validates final text
when no call is returned. Unknown tools, malformed arguments, incomplete/refused
outputs and duplicate call IDs fail without returning any proposal. Tool-looking
prose, DSML/XML tags and custom/server-side tool types are not interpreted.

Native history uses assistant `Message::toolCalls`, followed by `role="tool"`
messages with matching `toolCallId`. Results remain data, not system instructions.
History names/JSON/identities and balanced correlations are checked before HTTP.
Images remain separate user observations. Limits are128 declarations,16 calls per
reply/history message,64-character portable names,128-byte IDs and64KiB arguments.
Function arguments are not sent through `onText`; Responses proposals come from
the completed response. Its complete final text is authoritative over tentative
deltas, including provisional whitespace. Hosts still validate their own business
permissions, persistence and execution results.

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

`ModelProfile::supportsVision` explicitly opts in image input. User `Message`
values may include owned `ImageInput{dataUrl, detail}` entries. Only bounded
base64 JPEG/PNG/WebP/GIF data URLs are accepted; remote URLs, file paths and images
on system/assistant messages reject before networking. Chat uses image_url content
parts; Responses uses input_image. The engine does not fetch, interpret or retain
images automatically. Text-only profiles remain unchanged; request byte limits
also include image data. Fixture tests verify both wire shapes and rejection.

Image protocol reference, accessed2026-10-25:
[Images and vision](https://developers.openai.com/api/docs/guides/images-vision).

ModelProfile.proxyOrigin/useSystemProxy forward explicit HTTP proxy or opt-in
Windows manual proxy selection to AYNetworkHttp. Defaults remain direct in this
module; the host chooses its own default. Model credentials stay in the target
HTTPS request, not a proxy URL or plaintext settings field. Proxy authentication
and automatic PAC resolution are outside the current transport.

Protocol sources, accessed 2026-10-24:
[Structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs),
[Responses migration](https://developers.openai.com/api/docs/guides/migrate-to-responses).

`ModelProfile.reasoningEffort` optionally sends a validated effort label through
`reasoning.effort` (Responses) or `reasoning_effort` (Chat). Empty preserves the
provider default. The caller must verify model support and label semantics;
no provider is detected and no hidden reasoning text is requested, returned as
application content, or replayed. In particular a Chat effort label is not a
provider-specific thinking-mode toggle. Supported portable labels are none,
minimal, low, medium, high, xhigh and max; unknown labels fail before I/O.
