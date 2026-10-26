# AYInference design

**Version:** 0.2.0
**Date:** 2026-10-27
**Status:** Active
**Owner:** AYInference

## 1. Dependency boundary

AYInference -> AYNetworkHttp + AYSerializer. Public types contain owned messages,
profile metadata, optional usage and validation issues. No dependency on World,
Scene, AYUI, gameplay behavior or persistent character data. Composition is owned
by applications; each invocation creates isolated transport/parser state.

## 2. Invocation

Check budgets/capabilities/schema -> read environment credential -> encode the
selected protocol -> bounded verified transport -> bounded envelope/SSE decoding
-> completion/refusal checks -> local schema validation -> owned final Result.
Failure clears final text and function proposals. Streaming preview cannot serve as authorization.
Mock follows the same output validation boundary with no network or fake usage.

Wire output selection is explicit: strict Schema when supported, otherwise
`json_object` only when `supportsJsonObject` is declared and the caller schema
has object type. Both modes keep the same local schema validation. No capability
inference, implicit fallback, native-tool execution or permissive prose parsing.

## 3. Ownership and failure

Invocation blocks its caller and may run in an owned AYTask job. Cancellation
predicates must be thread-safe; preview callbacks run on that invoking thread.
Callers own UI marshaling. Provider error bodies and credentials are not exposed
as diagnostics or written to logs. Business proposals are never parsed/executed
by this module. Generic standard function envelopes are parsed only when explicitly
declared, with parameter-schema checks; the application decides business approval,
execution, retries and persistent call audit records.

## 4. Validation and limits

See README and JsonSchema's documented subset. Requests and envelopes are bounded;
malformed UTF-8/JSON, incomplete streams and unknown validation keywords fail.
Compatible endpoints must match the declared protocol; non-compatible providers
need an additional adapter, not ad-hoc business-side HTTP code. This release has
no native Anthropic/Gemini adapter, embedding index or automatic tokenizer.

## 5. Explicit image input

ImageInput is owned request data, authorized by the application and gated by
supportsVision. Validation bounds image count and request bytes, validates the
data URL media type/base64 shape and restricts images to user messages. Only
Chat image_url and Responses input_image are encoded; no URL downloading or
provider capability inference. Callers may discard older frames from their prompt
while retaining a separately governed archive. Tests use loopback providers;
public model vision quality and provider-specific limits remain deployment checks.

## 6. Standard native function envelopes

`supportsToolCalls` and `Request.tools` jointly declare the callable names and
schemas. Capability absence fails before HTTP. Input history keeps assistant call
IDs and tool receipts balanced; no historical function is implicitly made callable
again. Responses uses function_call/function_call_output items; Chat uses assistant
tool_calls and tool-role messages. Only declared, bounded JSON arguments reach
owned Result.toolCalls, with no execution. A batch fails as a whole on invalid
output. Final text Schema applies only when no function proposal was returned.

Streaming function arguments never enter the text preview. Chat deltas assemble
IDs/names/arguments and need a successful tool_calls finish plus DONE; Responses
reads the authoritative completed output rather than provisional arguments. Final
response text can replace mismatching tentative text. Cancellation/refusal and
incomplete completion clear every candidate. Unstructured tags are ordinary text,
not an alternate execution channel.

Wire response bytes include SSE framing, repeated completed-event payloads and non-visible provider channels. A transport byte-limit failure retains a response_bytes diagnostic at /limits/response_bytes and clears all partial text/proposals. The caller chooses its bounded raw-wire allowance independently from output-token and visible-text budgets.
