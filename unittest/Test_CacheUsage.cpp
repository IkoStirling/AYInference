#include <AYTestFixtures.h>
#include <AYInference/Inference.h>
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <thread>
#include <mutex>
using namespace ayt::inference;
namespace {
using Json=nlohmann::json;
struct CacheUsageFixture {
    httplib::Server server;std::jthread worker;std::mutex mutex;Json usage;int port;
    void set(Json value){std::lock_guard lock(mutex);usage=std::move(value);}
    CacheUsageFixture(){
        const auto handler=[this](const httplib::Request& req,httplib::Response& res){
            const auto body=Json::parse(req.body);const bool responses=req.path=="/v1/responses";Json metrics;{std::lock_guard lock(mutex);metrics=usage;}
            Json envelope=responses?Json{{"status","completed"},{"output",Json::array({Json{{"type","message"},{"content",Json::array({Json{{"type","output_text"},{"text","ok"}}})}}})}}:Json{{"choices",Json::array({Json{{"message",{{"content","ok"}}},{"finish_reason","stop"}}})}};
            if(!metrics.is_null())envelope["usage"]=metrics;
            if(!body.value("stream",false)){res.set_content(envelope.dump(),"application/json");return;}
            if(responses){res.set_content("data: "+Json{{"type","response.completed"},{"response",envelope}}.dump()+"\n\n","text/event-stream");return;}
            const Json end={{"choices",Json::array({Json{{"delta",{{"content","ok"}}},{"finish_reason","stop"}}})}};
            res.set_content("data: "+end.dump()+"\n\ndata: "+Json{{"choices",Json::array()},{"usage",metrics}}.dump()+"\n\ndata: [DONE]\n\n","text/event-stream");
        };
        server.Post("/v1/chat/completions",handler);server.Post("/v1/responses",handler);port=server.bind_to_any_port("127.0.0.1");worker=std::jthread([this]{server.listen_after_bind();});server.wait_until_ready();
    }
    ~CacheUsageFixture(){server.stop();worker.join();}
    Result call(Protocol protocol,bool streaming){ModelProfile profile;profile.protocol=protocol;profile.origin="http://127.0.0.1:"+std::to_string(port);profile.model="cache-fixture";Request request;request.messages={{"user","hello"}};if(streaming)request.onText=[](std::string_view){return true;};return invoke(profile,request);}
};
}
TEST_SUITE(InferenceCacheUsage)
TEST_CASE(cache_usage_aliases_work_in_both_protocols_and_streaming_modes){
    CacheUsageFixture fixture;
    for(auto protocol:{Protocol::ChatCompletions,Protocol::Responses})for(bool stream:{false,true}){
        const bool responses=protocol==Protocol::Responses;
        Json metrics={{responses?"input_tokens":"prompt_tokens",100},{responses?"output_tokens":"completion_tokens",5},{responses?"input_tokens_details":"prompt_tokens_details",{{"cached_tokens",80}}}};
        fixture.set(metrics);auto result=fixture.call(protocol,stream);CHECK(static_cast<bool>(result));CHECK(result.usage.inputTokens==100);CHECK(result.usage.cachedInputTokens==80);CHECK(result.usage.uncachedInputTokens==20);
        metrics.erase(responses?"input_tokens_details":"prompt_tokens_details");metrics["prompt_cache_hit_tokens"]=64;metrics["prompt_cache_miss_tokens"]=36;fixture.set(metrics);result=fixture.call(protocol,stream);CHECK(static_cast<bool>(result));CHECK(result.usage.cachedInputTokens==64&&result.usage.uncachedInputTokens==36);
    }
}
TEST_CASE(unknown_cache_usage_stays_unknown_and_invalid_counts_do_not_invent_hits){
    CacheUsageFixture fixture;fixture.set({{"prompt_tokens",100},{"completion_tokens",5}});auto result=fixture.call(Protocol::ChatCompletions,false);CHECK(static_cast<bool>(result));CHECK(!result.usage.cachedInputTokens&&!result.usage.uncachedInputTokens);
    for(auto metrics:{Json{{"prompt_tokens",100},{"prompt_cache_hit_tokens",0}},Json{{"prompt_tokens",100},{"prompt_cache_miss_tokens",100}}}){fixture.set(metrics);result=fixture.call(Protocol::ChatCompletions,false);CHECK(result.usage.cachedInputTokens==0&&result.usage.uncachedInputTokens==100);}
    for(auto metrics:{Json{{"prompt_tokens",100},{"prompt_cache_hit_tokens",101}},Json{{"prompt_tokens",100},{"prompt_cache_hit_tokens",90},{"prompt_cache_miss_tokens",30}},Json{{"prompt_tokens",100},{"prompt_cache_hit_tokens",-1}},Json{{"prompt_tokens",100},{"prompt_cache_hit_tokens","80"}}}){fixture.set(metrics);result=fixture.call(Protocol::ChatCompletions,false);CHECK(static_cast<bool>(result));CHECK(!result.usage.cachedInputTokens&&!result.usage.uncachedInputTokens);}
    ModelProfile mock;mock.mockOutput="ok";Request req;req.messages={{"user","hello"}};result=invoke(mock,req);CHECK(!result.usage.cachedInputTokens&&!result.usage.uncachedInputTokens);
}
TEST_SUITE_END
