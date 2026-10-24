#pragma once
#include <AYSerializer/JsonSchema.h>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ayt::inference {
enum class Protocol { Mock, ChatCompletions, Responses };
enum class Status { Ok, Invalid, Unsupported, CredentialMissing, Cancelled, Timeout, Transport, Provider, Refused, Incomplete, InvalidOutput, Limit };
struct ModelProfile {
    Protocol protocol = Protocol::Mock;
    std::string origin;
    std::string basePath = "/v1";
    std::string model;
    std::string credentialEnvironment;
    bool supportsJsonSchema = false;
    bool supportsStreaming = true;
    bool legacyMaxTokens = false;
    std::chrono::milliseconds timeout{30000};
    std::size_t maxRequestBytes = 2 * 1024 * 1024;
    std::size_t maxResponseBytes = 4 * 1024 * 1024;
    std::uint32_t maxOutputTokensPerCall = 8192;
    std::string mockOutput = "{}";
};
struct Message { std::string role, text; };
struct Request {
    std::vector<Message> messages;
    std::string outputSchema;
    std::string schemaName = "output";
    std::uint32_t maxOutputTokens = 1024;
    /// Must be safe to call concurrently from the invoking/watchdog threads.
    std::function<bool()> cancelled;
    /// Tentative text only. Must not execute proposals; consume Result after validation.
    std::function<bool(std::string_view)> onText;
};
struct Usage {
    std::optional<std::uint64_t> inputTokens, outputTokens;
};
struct Result {
    Status status = Status::Invalid;
    int httpStatus = 0;
    std::string text, providerRequestId, model;
    Usage usage;
    std::vector<serializer::JsonValidationIssue> issues;
    explicit operator bool() const noexcept { return status == Status::Ok; }
};
/// Stateless model invocation. Profiles own protocol/capability choices; business
/// schemas and context remain caller-owned. No automatic retries or tool execution.
/// Remote providers require HTTPS and an environment credential. Literal loopback
/// endpoints may use HTTP and omit authentication for local model servers.
Result invoke(const ModelProfile& profile, const Request& request);
struct ContextFragment { std::string id, text, source; int priority = 0; };
struct ContextAssembly { std::vector<ContextFragment> selected; std::vector<std::string> omitted; std::size_t bytes = 0; };
/// Stable priority selection over already-authorized material. Byte budget is not
/// a token/context-window guarantee; applications can impose model-specific limits.
ContextAssembly assembleContext(std::vector<ContextFragment> fragments, std::size_t maxBytes);
}
