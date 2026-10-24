#include <AYTestFixtures.h>
#include <AYInference/Inference.h>
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <thread>
#include <atomic>
using namespace ayt::inference;
namespace {
using Json = nlohmann::json;
const std::string schema = R"({"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false})";
struct ProviderFixture {
    httplib::Server server; std::jthread thread; int port; std::atomic<int> calls{0};
    ProviderFixture() {
        server.Post("/v1/chat/completions",[this](const auto& req,auto& res) {
            ++calls; const auto body = Json::parse(req.body); const auto model = body.at("model").template get<std::string>();
            if (model == "http-error") { res.status = 429; res.set_content("sensitive-provider-body","text/plain"); return; }
            const auto output = model == "invalid" ? R"({"wrong":true})" : R"({"answer":"ok"})";
            if (body.value("stream",false)) {
                const auto event = Json{{"choices",Json::array({{{"index",0},{"delta",{{"content",output}}},{"finish_reason",nullptr}}})}}.dump();
                const auto end = Json{{"choices",Json::array({{{"index",0},{"delta",Json::object()},{"finish_reason","stop"}}})}}.dump();
                const auto usage = Json{{"choices",Json::array()},{"usage",{{"prompt_tokens",7},{"completion_tokens",4}}}}.dump();
                const auto sse = ": keepalive\r\ndata: " + event + "\r\n\r\ndata: " + end + "\n\ndata: " + usage + "\n\n" + (model == "cut" ? "" : "data: [DONE]\n\n");
                res.set_chunked_content_provider("text/event-stream",[sse](std::size_t offset,httplib::DataSink& sink) {
                    if (offset >= sse.size()) { sink.done(); return true; }
                    return sink.write(sse.data()+offset,(std::min)(std::size_t{3},sse.size()-offset));
                }); return;
            }
            Json message{{"content",output}};
            if (model == "refused") message = {{"content",nullptr},{"refusal","declined"}};
            res.set_content(Json{{"id","request-1"},{"model",model},{"choices",Json::array({{{"message",message},{"finish_reason",model == "cut" ? "length" : "stop"}}})},{"usage",{{"prompt_tokens",7},{"completion_tokens",4}}}}.dump(),"application/json");
        });
        server.Post("/v1/responses",[this](const auto& req,auto& res) {
            ++calls; const auto body = Json::parse(req.body);
            if (!body.contains("input") || body.value("store",true) || !body.contains("text")) { res.status = 400; return; }
            const Json response{{"id","response-1"},{"status","completed"},{"output",Json::array({{{"type","message"},{"content",Json::array({{{"type","output_text"},{"text",R"({"answer":"ok"})"}}})}}})},{"usage",{{"input_tokens",3},{"output_tokens",2}}}};
            if (body.value("stream",false)) {
                const auto delta = Json{{"type","response.output_text.delta"},{"delta",R"({"answer":"ok"})"}};
                const auto complete = Json{{"type","response.completed"},{"response",response}};
                res.set_content("event: response.output_text.delta\ndata: "+delta.dump()+"\n\nevent: response.completed\ndata: "+complete.dump()+"\n\n","text/event-stream");
            } else res.set_content(response.dump(),"application/json");
        });
        port = server.bind_to_any_port("127.0.0.1");
        thread = std::jthread([this] { server.listen_after_bind(); }); server.wait_until_ready();
    }
    ~ProviderFixture() { server.stop(); thread.join(); }
    ModelProfile profile(Protocol protocol = Protocol::ChatCompletions) const {
        ModelProfile p; p.protocol = protocol; p.origin = "http://127.0.0.1:"+std::to_string(port); p.model = "test"; p.supportsJsonSchema = true; return p;
    }
};
Request request() { Request r; r.messages = {{"system","Return JSON"},{"user","hello"}}; r.outputSchema = schema; return r; }
}
TEST_SUITE(InferenceTests)
TEST_CASE(mock_validates_callers_schema_and_does_not_invent_usage) {
    ModelProfile p; p.mockOutput = R"({"answer":"ok"})";
    auto r = invoke(p,request()); CHECK(static_cast<bool>(r)); CHECK(!r.usage.inputTokens.has_value());
    p.mockOutput = R"({"answer":"ok","unexpected":1})";
    r = invoke(p,request()); CHECK(r.status == Status::InvalidOutput); CHECK(r.text.empty()); CHECK(!r.issues.empty());
    auto req = request(); req.cancelled = [] { return true; }; CHECK(invoke(p,req).status == Status::Cancelled);
    req = request(); req.outputSchema = R"({"$ref":"unsupported"})"; CHECK(invoke(p,req).status == Status::Invalid);
    req=request(); req.maxOutputTokens=p.maxOutputTokensPerCall+1; CHECK(invoke(p,req).status==Status::Limit);
    req=request(); req.messages[0].text=std::string("\xff",1); p.protocol=Protocol::ChatCompletions; p.model="test"; p.origin="http://127.0.0.1:1"; p.supportsJsonSchema=true;
    CHECK(invoke(p,req).status==Status::Invalid);
}
TEST_CASE(chat_protocol_stream_and_nonstream_have_validated_results) {
    ProviderFixture fixture; auto p = fixture.profile(); auto req = request();
    auto result = invoke(p,req); CHECK(static_cast<bool>(result)); CHECK(result.providerRequestId == "request-1"); CHECK(result.usage.inputTokens == 7);
    std::string provisional; req.onText = [&](std::string_view text) { provisional += text; return true; };
    result = invoke(p,req); CHECK(static_cast<bool>(result)); CHECK(result.text == provisional); CHECK(result.usage.outputTokens == 4);
    p.model = "cut"; result = invoke(p,req); CHECK(result.status == Status::Incomplete); CHECK(result.text.empty());
    p.model = "invalid"; CHECK(invoke(p,req).status == Status::InvalidOutput);
    req.onText = [](std::string_view) { return false; }; p.model = "test"; CHECK(invoke(p,req).status == Status::Cancelled);
}
TEST_CASE(responses_protocol_and_capability_errors_are_explicit) {
    ProviderFixture fixture; auto p = fixture.profile(Protocol::Responses); auto req = request();
    auto result = invoke(p,req); CHECK(static_cast<bool>(result)); CHECK(result.usage.inputTokens == 3);
    req.onText = [](std::string_view) { return true; }; CHECK(static_cast<bool>(invoke(p,req)));
    const int calls = fixture.calls;
    p.supportsJsonSchema = false; CHECK(invoke(p,req).status == Status::Unsupported); CHECK(fixture.calls == calls);
    p.supportsJsonSchema = true; p.credentialEnvironment = "AY_INFERENCE_ABSENT_TEST_CREDENTIAL_20261024";
    CHECK(invoke(p,req).status == Status::CredentialMissing); CHECK(fixture.calls == calls);
    p.credentialEnvironment.clear(); p.origin = "http://example.com"; CHECK(invoke(p,req).status == Status::Invalid);
}
TEST_CASE(refusals_truncation_and_provider_errors_never_become_output) {
    ProviderFixture fixture; auto p = fixture.profile(); auto req = request();
    p.model = "refused"; CHECK(invoke(p,req).status == Status::Refused);
    p.model = "cut"; CHECK(invoke(p,req).status == Status::Incomplete);
    p.model = "http-error"; auto r = invoke(p,req); CHECK(r.status == Status::Provider); CHECK(r.httpStatus == 429); CHECK(r.text.empty());
}
TEST_CASE(context_selection_is_stable_bounded_and_retains_sources) {
    auto result = assembleContext({{"a","1234","source-a",1},{"b","12","source-b",2},{"c","12","source-c",1},{"b","dup","source-b",0}},4);
    CHECK(result.selected.size() == 2); CHECK(result.selected[0].id == "b"); CHECK(result.selected[1].id == "c");
    CHECK(result.selected[1].source == "source-c"); CHECK(result.bytes == 4); CHECK(result.omitted.size() == 2);
}
TEST_SUITE_END
int main(int argc,char** argv) { return ayt::test::runTests("AYInference",argc,argv); }
