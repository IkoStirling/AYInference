#include <AYInference/Inference.h>
#include <AYTestFixtures.h>
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <atomic>
#include <mutex>
#include <thread>
using namespace ayt::inference;
namespace {
using Json=nlohmann::json;
const std::string parameters=R"({"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false})";
struct NativeFixture {
    httplib::Server server;std::jthread host;int port=0;std::atomic<int> calls=0;std::mutex mutex;Json body;
    NativeFixture(){
        auto serve=[this](const httplib::Request& req,httplib::Response& res,bool responses){
            ++calls;const auto request=Json::parse(req.body);{std::lock_guard guard(mutex);body=request;}
            const auto model=request.at("model").get<std::string>();
            bool previous=false;for(const auto& message:request.at(responses?"input":"messages"))if(message.value(responses?"type":"role",std::string{})==(responses?"function_call_output":"tool"))previous=true;
            const std::string callId=previous?"call-2":"call-1";
            const std::string name=model=="unlisted"?"erase_everything":"read_file";
            const std::string args=model=="bad-args"?R"({"path":42})":R"({"path":"safe.txt"})";
            if(responses){
                Json output=Json::array({{{"type","function_call"},{"call_id",callId},{"name",name},{"arguments",args}}});
                if(model=="duplicate")output.push_back(output[0]);
                if(model=="refused")output.push_back({{"type","message"},{"content",Json::array({{{"type","refusal"},{"refusal","not permitted"}}})}});
                if(model=="final-text"||model=="text-tags")output=Json::array({{{"type","message"},{"content",Json::array({{{"type","output_text"},{"text",model=="text-tags"?"<tool_call>read_file</tool_call>":R"({"answer":"final"})"}}})}}});
                const Json result={{"id","native-response"},{"status",model=="cut"?"incomplete":"completed"},{"output",output}};
                if(request.value("stream",false)){
                    const Json delta=model=="final-text"?Json{{"type","response.output_text.delta"},{"delta","   "}}:Json{{"type","response.function_call_arguments.delta"},{"delta","partial-not-a-call"}};
                    const Json complete={{"type",model=="cut"?"response.incomplete":"response.completed"},{"response",result}};
                    res.set_content("data: "+Json{{"type","response.reasoning_text.delta"},{"delta","not visible output"}}.dump()+"\n\ndata: "+delta.dump()+"\n\ndata: "+complete.dump()+"\n\n","text/event-stream");
                }else res.set_content(result.dump(),"application/json");return;
            }
            Json message={{"role","assistant"},{"content",nullptr},{"tool_calls",Json::array({{{"id",callId},{"type","function"},{"function",{{"name",name},{"arguments",args}}}}})}};
            if(model=="duplicate")message["tool_calls"].push_back(message["tool_calls"][0]);
            if(model=="refused")message["refusal"]="not permitted";
            if(model=="text-tags"){message.erase("tool_calls");message["content"]="<tool_call>read_file</tool_call>";}
            if(request.value("stream",false)){
                auto frame=[](Json delta,Json reason=nullptr){return "data: "+Json{{"choices",Json::array({{{"index",0},{"delta",delta},{"finish_reason",reason}}})}}.dump()+"\n\n";};
                const Json first={{"tool_calls",Json::array({{{"index",0},{"id","call-"},{"type","function"},{"function",{{"name",name.substr(0,4)},{"arguments",args.substr(0,5)}}}}})}};
                Json second={{"tool_calls",Json::array({{{"index",0},{"id",callId.substr(5)},{"function",{{"name",name.substr(4)},{"arguments",args.substr(5)}}}}})}};
                if(model=="refused")second["refusal"]="not permitted";
                auto wire=frame(first)+frame(second)+frame(Json::object(),model=="cut"?"length":"tool_calls")+"data: [DONE]\n\n";
                res.set_content(wire,"text/event-stream");
            }else res.set_content(Json{{"choices",Json::array({{{"message",message},{"finish_reason",model=="cut"?"length":model=="text-tags"?"stop":"tool_calls"}}})}}.dump(),"application/json");
        };
        server.Post("/v1/chat/completions",[serve](const auto& req,auto& res){serve(req,res,false);});
        server.Post("/v1/responses",[serve](const auto& req,auto& res){serve(req,res,true);});
        port=server.bind_to_any_port("127.0.0.1");host=std::jthread([this]{server.listen_after_bind();});server.wait_until_ready();
    }
    ~NativeFixture(){server.stop();host.join();}
    ModelProfile profile(Protocol protocol)const{ModelProfile p;p.protocol=protocol;p.origin="http://127.0.0.1:"+std::to_string(port);p.model="native";p.supportsToolCalls=true;p.supportsJsonSchema=true;return p;}
    Json captured(){std::lock_guard guard(mutex);return body;}
};
Request nativeRequest(){Request req;req.messages={{"system","Return JSON or call a declared function"},{"user","read this"}};req.tools={{"read_file","Read an approved file",parameters}};req.outputSchema=R"({"type":"object","required":["answer"],"properties":{"answer":{"type":"string"}},"additionalProperties":false})";return req;}
}
TEST_SUITE(NativeFunctionTools)
TEST_CASE(reasoning_effort_is_explicit_validated_and_does_not_expose_reasoning){
    NativeFixture fixture;
    for(const auto protocol:{Protocol::ChatCompletions,Protocol::Responses}){
        auto profile=fixture.profile(protocol);auto request=nativeRequest();auto result=invoke(profile,request);CHECK(static_cast<bool>(result));
        CHECK(!fixture.captured().contains("reasoning")&&!fixture.captured().contains("reasoning_effort"));
        profile.reasoningEffort="low";result=invoke(profile,request);CHECK(static_cast<bool>(result));
        const auto wire=fixture.captured();if(protocol==Protocol::Responses)CHECK(wire.at("reasoning").at("effort")=="low");else CHECK(wire.at("reasoning_effort")=="low");
        const auto before=fixture.calls.load();profile.reasoningEffort="arbitrary-provider-json";CHECK(invoke(profile,request).status==Status::Invalid);CHECK_INT_EQ(fixture.calls.load(),before);
    }
}
TEST_CASE(wire_response_limit_is_distinguished_and_never_returns_partial_proposals){
    NativeFixture fixture;
    for(const auto protocol:{Protocol::ChatCompletions,Protocol::Responses})for(const bool stream:{false,true}){
        auto profile=fixture.profile(protocol);profile.maxResponseBytes=16;auto request=nativeRequest();if(stream)request.onText=[](auto){return true;};
        const auto result=invoke(profile,request);CHECK(result.status==Status::Limit);CHECK(result.text.empty()&&result.toolCalls.empty());
        CHECK(!result.issues.empty());CHECK(result.issues[0].code=="response_bytes");
    }
}
TEST_CASE(native_calls_do_not_require_final_text_schema_support_or_parse_tool_like_prose){
    NativeFixture fixture;for(const auto protocol:{Protocol::ChatCompletions,Protocol::Responses}){
        auto profile=fixture.profile(protocol);profile.supportsJsonSchema=false;auto req=nativeRequest();req.outputSchema.clear();
        auto result=invoke(profile,req);CHECK(static_cast<bool>(result));CHECK_INT_EQ(result.toolCalls.size(),1);
        profile.model="text-tags";result=invoke(profile,req);CHECK(static_cast<bool>(result));CHECK(result.toolCalls.empty());CHECK(result.text=="<tool_call>read_file</tool_call>");
    }
}
TEST_CASE(invalid_function_declarations_fail_before_any_provider_request){
    NativeFixture fixture;auto profile=fixture.profile(Protocol::ChatCompletions);auto req=nativeRequest();
    req.tools[0].name="workspace.read";CHECK(invoke(profile,req).status==Status::Invalid);
    req=nativeRequest();req.tools[0].parameters=R"({"$ref":"external"})";CHECK(invoke(profile,req).status==Status::Invalid);
    req=nativeRequest();req.tools.push_back(req.tools[0]);CHECK(invoke(profile,req).status==Status::Invalid);
    req=nativeRequest();profile.supportsJsonSchema=false;req.outputSchema.clear();req.tools[0].strict=true;CHECK(invoke(profile,req).status==Status::Unsupported);
    CHECK_INT_EQ(fixture.calls.load(),0);
}
TEST_CASE(function_proposals_are_opt_in_and_validated_in_both_protocols){
    NativeFixture fixture;
    for(const auto protocol:{Protocol::ChatCompletions,Protocol::Responses})for(const bool streaming:{false,true}){
        auto profile=fixture.profile(protocol);auto req=nativeRequest();std::string preview;if(streaming)req.onText=[&](std::string_view value){preview+=value;return true;};
        auto result=invoke(profile,req);CHECK(static_cast<bool>(result));CHECK_INT_EQ(result.toolCalls.size(),1);CHECK(result.toolCalls[0].name=="read_file");CHECK(result.toolCalls[0].id=="call-1");CHECK(result.toolCalls[0].arguments==R"({"path":"safe.txt"})");CHECK(preview.empty());
        const auto wire=fixture.captured();CHECK(wire.at("tool_choice")=="auto");const auto tool=protocol==Protocol::Responses?wire.at("tools")[0]:wire.at("tools")[0].at("function");CHECK(tool.at("name")=="read_file");CHECK(tool.at("parameters")==Json::parse(parameters));
    }
    auto profile=fixture.profile(Protocol::ChatCompletions);profile.supportsToolCalls=false;const int calls=fixture.calls.load();CHECK(invoke(profile,nativeRequest()).status==Status::Unsupported);CHECK_INT_EQ(fixture.calls.load(),calls);
}
TEST_CASE(function_history_is_correlated_without_turning_it_into_an_instruction){
    NativeFixture fixture;
    for(const auto protocol:{Protocol::ChatCompletions,Protocol::Responses}){
        auto profile=fixture.profile(protocol);auto req=nativeRequest();Message call{"assistant","",{},{{"call-1","read_file",R"({"path":"safe.txt"})"}}};Message receipt{"tool",R"({"content":"untrusted document"})"};receipt.toolCallId="call-1";req.messages.push_back(call);req.messages.push_back(receipt);
        CHECK(static_cast<bool>(invoke(profile,req)));const auto wire=fixture.captured();
        if(protocol==Protocol::Responses){CHECK(wire.at("input")[2].at("type")=="function_call");CHECK(wire.at("input")[3].at("type")=="function_call_output");CHECK(wire.at("input")[3].at("call_id")=="call-1");}
        else{CHECK(wire.at("messages")[2].at("tool_calls")[0].at("id")=="call-1");CHECK(wire.at("messages")[3].at("role")=="tool");CHECK(wire.at("messages")[3].at("tool_call_id")=="call-1");}
        const auto count=fixture.calls.load();req.messages.back().toolCallId="unmatched";CHECK(invoke(profile,req).status==Status::Invalid);CHECK_INT_EQ(fixture.calls.load(),count);
    }
}
TEST_CASE(invalid_or_refused_native_outputs_never_return_executable_proposals){
    NativeFixture fixture;
    for(const auto protocol:{Protocol::ChatCompletions,Protocol::Responses})for(const bool streaming:{false,true})for(const auto* model:{"bad-args","unlisted","refused","cut"}){
        auto profile=fixture.profile(protocol);profile.model=model;auto req=nativeRequest();if(streaming)req.onText=[](std::string_view){return true;};const auto result=invoke(profile,req);
        CHECK(!result);CHECK(result.text.empty()&&result.toolCalls.empty());
        if(std::string_view(model)=="refused")CHECK(result.status==Status::Refused);
        else if(std::string_view(model)=="cut")CHECK(result.status==Status::Incomplete);
        else CHECK(result.status==Status::InvalidOutput);
    }
    auto profile=fixture.profile(Protocol::ChatCompletions);profile.model="duplicate";CHECK(invoke(profile,nativeRequest()).status==Status::InvalidOutput);
}
TEST_CASE(completed_response_text_is_authoritative_over_tentative_whitespace){
    NativeFixture fixture;auto profile=fixture.profile(Protocol::Responses);profile.model="final-text";auto req=nativeRequest();req.tools.clear();std::string preview;req.onText=[&](std::string_view text){preview+=text;return true;};const auto result=invoke(profile,req);CHECK(static_cast<bool>(result));CHECK(result.text==R"({"answer":"final"})");CHECK(result.toolCalls.empty());
}
TEST_SUITE_END
