# AYInference design

**Version:** 0.1.0
**Date:** 2026-10-24
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
Failure clears final text. Streaming preview cannot serve as authorization.
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
by this module. The application decides retries and persistent call audit records.

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
