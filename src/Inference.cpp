#include <AYInference/Inference.h>
#include <AYNetwork/HttpClient.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <set>

namespace ayt::inference {
namespace {
using Json = nlohmann::json;
Json boundedJson(std::string_view text) {
    serializer::JsonValidationLimits limits; limits.maxBytes = 64 * 1024 * 1024; limits.maxNodes = 262144;
    if (!serializer::validateJsonDocument(text,"true",limits)) throw std::invalid_argument("provider envelope");
    return Json::parse(text);
}
bool loopback(const std::string& origin) {
    for (const auto* host : {"http://127.0.0.1", "http://localhost", "http://[::1]", "https://127.0.0.1", "https://localhost", "https://[::1]"}) {
        const std::string_view prefix(host);
        if (origin == prefix || (origin.starts_with(prefix) && origin.size() > prefix.size() && origin[prefix.size()] == ':')) return true;
    }
    return false;
}
bool cancelled(const Request& request) {
    try { return request.cancelled && request.cancelled(); } catch (...) { return true; }
}
void metadata(const Json& json, Result& result, bool responses) {
    if (json.contains("id") && json["id"].is_string()) result.providerRequestId = json["id"].get<std::string>().substr(0,256);
    if (json.contains("model") && json["model"].is_string()) result.model = json["model"].get<std::string>().substr(0,256);
    if (json.contains("usage") && json["usage"].is_object()) {
        const auto& usage = json["usage"];
        const auto* in = responses ? "input_tokens" : "prompt_tokens";
        const auto* out = responses ? "output_tokens" : "completion_tokens";
        if (usage.contains(in) && usage[in].is_number_unsigned()) result.usage.inputTokens = usage[in].get<std::uint64_t>();
        if (usage.contains(out) && usage[out].is_number_unsigned()) result.usage.outputTokens = usage[out].get<std::uint64_t>();
    }
}
std::string responseText(const Json& response, bool& refused) {
    std::string text;
    for (const auto& item : response.at("output")) {
        if (item.value("type", "") != "message") continue;
        for (const auto& part : item.at("content")) {
            if (part.value("type", "") == "refusal") refused = true;
            else if (part.value("type", "") == "output_text") text += part.at("text").get<std::string>();
        }
    }
    return text;
}
struct Sse {
    std::string buffer, data;
    std::function<bool(std::string_view)> dispatch;
    bool feed(std::string_view chunk) {
        buffer.append(chunk);
        for (;;) {
            const auto end = buffer.find('\n'); if (end == std::string::npos) break;
            auto line = buffer.substr(0, end); buffer.erase(0, end + 1);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) {
                if (!data.empty()) { data.pop_back(); const bool ok = dispatch(data); data.clear(); if (!ok) return false; }
            } else if (line.starts_with("data:")) { auto value = std::string_view(line).substr(5); if (value.starts_with(' ')) value.remove_prefix(1); data.append(value); data += '\n'; }
        }
        return true;
    }
};
}
Result invoke(const ModelProfile& profile, const Request& request) {
    Result result; result.model = profile.model;
    auto failure = [&](Status status) { result.status = status; result.text.clear(); return result; };
    try {
    if (cancelled(request)) return failure(Status::Cancelled);
    if (!profile.maxOutputTokensPerCall || profile.maxOutputTokensPerCall > 1048576 || request.maxOutputTokens > profile.maxOutputTokensPerCall) return failure(Status::Limit);
    if (request.messages.empty() || request.messages.size() > 1024 || !request.maxOutputTokens || request.maxOutputTokens > 1048576 || !profile.maxRequestBytes || profile.maxRequestBytes > 16 * 1024 * 1024 || !profile.maxResponseBytes || profile.maxResponseBytes > 64 * 1024 * 1024 || profile.timeout.count() <= 0 || profile.timeout > std::chrono::hours(1)) return failure(Status::Invalid);
    std::size_t inputBytes = request.outputSchema.size();
    for (const auto& message : request.messages) {
        if (message.role != "system" && message.role != "developer" && message.role != "user" && message.role != "assistant") return failure(Status::Invalid);
        inputBytes += message.text.size(); if (inputBytes > profile.maxRequestBytes) return failure(Status::Limit);
        if(!message.images.empty()&&(!profile.supportsVision||message.role!="user"))return failure(Status::Unsupported);
        if(message.images.size()>8)return failure(Status::Limit);
        for(const auto& image:message.images){
            inputBytes+=image.dataUrl.size();if(inputBytes>profile.maxRequestBytes)return failure(Status::Limit);
            bool prefix=false;for(const auto* mime:{"jpeg","png","webp","gif"})if(image.dataUrl.starts_with(std::string("data:image/")+mime+";base64,"))prefix=true;
            const auto comma=image.dataUrl.find(',');if(!prefix||comma==std::string::npos||comma+1==image.dataUrl.size()||(image.detail!="auto"&&image.detail!="low"&&image.detail!="high"))return failure(Status::Invalid);
            const auto data=std::string_view(image.dataUrl).substr(comma+1);if(data.size()%4||!std::all_of(data.begin(),data.end(),[](unsigned char c){return (c>='A'&&c<='Z')||(c>='a'&&c<='z')||(c>='0'&&c<='9')||c=='+'||c=='/'||c=='=';}))return failure(Status::Invalid);
            const auto padding=data.find('=');if(padding!=std::string_view::npos&&(padding+2<data.size()||!std::all_of(data.begin()+padding,data.end(),[](char c){return c=='=';})))return failure(Status::Invalid);
        }
    }
    if (!request.outputSchema.empty()) {
        auto validation = serializer::validateJsonSchema(request.outputSchema);
        if (!validation) { result.issues = std::move(validation.issues); return failure(Status::Invalid); }
        if (request.schemaName.empty() || request.schemaName.size() > 64 || !std::all_of(request.schemaName.begin(), request.schemaName.end(), [](unsigned char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-'; })) return failure(Status::Invalid);
        if (profile.protocol != Protocol::Mock && !profile.supportsJsonSchema) {
            if (!profile.supportsJsonObject) return failure(Status::Unsupported);
            if (Json::parse(request.outputSchema).value("type",Json()) != "object") return failure(Status::Unsupported);
        }
    }
    if (request.onText && !profile.supportsStreaming) return failure(Status::Unsupported);
    if (profile.protocol == Protocol::Mock) {
        if (profile.mockOutput.size() > profile.maxResponseBytes) return failure(Status::Limit);
        result.text = profile.mockOutput;
        if (request.onText) { try { if (!request.onText(result.text)) return failure(Status::Cancelled); } catch (...) { return failure(Status::Cancelled); } }
    } else {
        if (profile.model.empty() || profile.model.size() > 256 || !net::http::validOrigin(profile.origin) || (!profile.origin.starts_with("https://") && !loopback(profile.origin)) || profile.basePath.empty() || profile.basePath.front() != '/' || profile.basePath.back() == '/' || profile.basePath.find_first_of("?#\r\n") != std::string::npos) return failure(Status::Invalid);
        const auto credential = net::http::environmentCredential(profile.credentialEnvironment);
        if ((!profile.credentialEnvironment.empty() && credential.empty()) || (!loopback(profile.origin) && credential.empty())) return failure(Status::CredentialMissing);
        const bool responses = profile.protocol == Protocol::Responses;
        Json body{{"model", profile.model}, {"stream", static_cast<bool>(request.onText)}};
        Json messages = Json::array();for(const auto& message:request.messages){
            if(message.images.empty()){messages.push_back({{"role",message.role},{"content",message.text}});continue;}
            Json parts=Json::array();if(!message.text.empty())parts.push_back({{"type",responses?"input_text":"text"},{"text",message.text}});
            for(const auto& image:message.images)if(responses)parts.push_back({{"type","input_image"},{"image_url",image.dataUrl},{"detail",image.detail}});else parts.push_back({{"type","image_url"},{"image_url",{{"url",image.dataUrl},{"detail",image.detail}}}});
            messages.push_back({{"role",message.role},{"content",parts}});
        }
        body[responses ? "input" : "messages"] = std::move(messages);
        body[responses ? "max_output_tokens" : profile.legacyMaxTokens ? "max_tokens" : "max_completion_tokens"] = request.maxOutputTokens;
        if (responses) body["store"] = false;
        else if (request.onText) body["stream_options"] = {{"include_usage",true}};
        if (!request.outputSchema.empty()) {
            auto schema = Json::parse(request.outputSchema);
            if (!profile.supportsJsonSchema) {
                if (responses) body["text"]["format"] = {{"type","json_object"}};
                else body["response_format"] = {{"type","json_object"}};
            } else if (responses) body["text"]["format"] = {{"type","json_schema"},{"name",request.schemaName},{"strict",true},{"schema",schema}};
            else body["response_format"] = {{"type","json_schema"},{"json_schema",{{"name",request.schemaName},{"strict",true},{"schema",schema}}}};
        }
        net::http::Request wire; wire.origin = profile.origin; wire.path = profile.basePath + (responses ? "/responses" : "/chat/completions");
        wire.body = body.dump(); if (wire.body.size() > profile.maxRequestBytes) return failure(Status::Limit);
        wire.headers = {{"Content-Type","application/json"}};
        if (!credential.empty()) wire.headers.emplace_back("Authorization", "Bearer " + credential);
        wire.timeout = profile.timeout; wire.maxResponseBytes = profile.maxResponseBytes; wire.cancelled = request.cancelled;
        wire.proxyOrigin=profile.proxyOrigin;wire.useSystemProxy=profile.useSystemProxy;
        bool finished = false, doneMarker = false, refused = false, malformed = false, userStopped = false;
        Sse sse;
        sse.dispatch = [&](std::string_view data) {
            if (data == "[DONE]") { doneMarker = true; return true; }
            if (doneMarker) { malformed = true; return false; }
            try {
                const auto event = boundedJson(data);
                std::string delta;
                if (responses) {
                    const auto type = event.value("type", "");
                    if (type == "response.output_text.delta") delta = event.at("delta").get<std::string>();
                    else if (type == "response.refusal.delta" || type == "response.refusal.done") refused = true;
                    else if (type == "response.completed") { const auto& r = event.at("response"); finished = r.value("status", "") == "completed"; metadata(r,result,true); bool finalRefused = false; const auto full = responseText(r,finalRefused); refused = refused || finalRefused; if (result.text.empty()) delta = full; }
                    else if (type == "response.incomplete" || type == "response.failed" || type == "error") { malformed = true; return false; }
                } else {
                    if (event.contains("error")) { malformed = true; return false; }
                    metadata(event,result,false);
                    for (const auto& choice : event.at("choices")) {
                        if (choice.value("index",0) != 0) continue;
                        const auto& value = choice.at("delta");
                        if (value.contains("refusal") && !value["refusal"].is_null()) refused = true;
                        if (value.contains("tool_calls")) { malformed = true; return false; }
                        if (value.contains("content") && !value["content"].is_null()) delta += value["content"].get<std::string>();
                        if (choice.contains("finish_reason") && !choice["finish_reason"].is_null()) finished = choice["finish_reason"] == "stop";
                    }
                }
                result.text += delta;
                if (!delta.empty()) { if (!request.onText(delta)) { userStopped = true; return false; } }
                return true;
            } catch (...) { malformed = true; return false; }
        };
        if (request.onText) wire.onChunk = [&](std::string_view chunk) { return sse.feed(chunk); };
        auto response = net::http::perform(wire); result.httpStatus = response.httpStatus;
        if (userStopped || cancelled(request) || response.status == net::http::Status::Cancelled) return failure(Status::Cancelled);
        if (response.status == net::http::Status::Timeout) return failure(Status::Timeout);
        if (response.status == net::http::Status::Limit) return failure(Status::Limit);
        if (malformed) return failure(Status::Provider);
        if (!response) return failure(Status::Transport);
        if (response.httpStatus < 200 || response.httpStatus >= 300) return failure(Status::Provider);
        if (request.onText) {
            if (!sse.buffer.empty() || !sse.data.empty() || !finished || (!responses && !doneMarker)) return failure(Status::Incomplete);
        } else {
            try {
                const auto value = boundedJson(response.body);
                metadata(value,result,responses);
                if (responses) { finished = value.value("status", "") == "completed"; result.text = responseText(value, refused); }
                else {
                    const auto& choice = value.at("choices").at(0); const auto& message = choice.at("message");
                    refused = message.contains("refusal") && !message["refusal"].is_null();
                    finished = choice.value("finish_reason", "") == "stop";
                    if (message.contains("tool_calls")) return failure(Status::Unsupported);
                    if (message.contains("content") && message["content"].is_string()) result.text = message["content"].get<std::string>();
                }
            } catch (...) { return failure(Status::Provider); }
        }
        if (refused) return failure(Status::Refused);
        if (!finished) return failure(Status::Incomplete);
    }
    if (cancelled(request)) return failure(Status::Cancelled);
    if (!request.outputSchema.empty()) {
        auto validation = serializer::validateJsonDocument(result.text, request.outputSchema);
        if (!validation) { result.issues = std::move(validation.issues); return failure(Status::InvalidOutput); }
    }
    result.status = Status::Ok; return result;
    } catch (...) { return failure(Status::Invalid); }
}
ContextAssembly assembleContext(std::vector<ContextFragment> fragments, std::size_t maxBytes) {
    std::stable_sort(fragments.begin(),fragments.end(),[](const auto& a,const auto& b) { return a.priority > b.priority; });
    ContextAssembly out; std::set<std::string> seen;
    for (auto& f : fragments) {
        if (f.id.empty() || !seen.insert(f.id).second || f.text.size() > maxBytes - out.bytes) out.omitted.push_back(f.id);
        else { out.bytes += f.text.size(); out.selected.push_back(std::move(f)); }
    }
    return out;
}
}
