bool toolName(std::string_view name){return !name.empty()&&name.size()<=64&&std::all_of(name.begin(),name.end(),[](unsigned char c){return (c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='_'||c=='-';});}
bool callIdentity(std::string_view value){return !value.empty()&&value.size()<=128&&std::none_of(value.begin(),value.end(),[](unsigned char c){return c<32||c==127;});}
Status toolDeclarations(const ModelProfile& profile,const Request& request,std::size_t& bytes){
    if(request.tools.empty())return Status::Ok;if(!profile.supportsToolCalls)return Status::Unsupported;
    if(request.tools.size()>128)return Status::Limit;std::set<std::string> names;
    for(const auto& tool:request.tools){
        if(!toolName(tool.name)||!names.insert(tool.name).second||tool.description.size()>4096||tool.parameters.size()>65536)return Status::Invalid;
        if(tool.strict&&!profile.supportsJsonSchema)return Status::Unsupported;
        if(!serializer::validateJsonSchema(tool.parameters))return Status::Invalid;
        if(Json::parse(tool.parameters).value("type",Json())!="object")return Status::Unsupported;
        bytes+=tool.name.size()+tool.description.size()+tool.parameters.size();if(bytes>profile.maxRequestBytes)return Status::Limit;
    }return Status::Ok;
}
Status toolHistory(const ModelProfile& profile,const Request& request,std::size_t& bytes){
    std::set<std::string> pending,seen;
    for(const auto& message:request.messages){
        const bool native=!message.toolCalls.empty()||message.role=="tool"||!message.toolCallId.empty();
        if(native&&!profile.supportsToolCalls)return Status::Unsupported;
        if(message.role=="tool"){
            if(!message.toolCalls.empty()||!message.images.empty()||!callIdentity(message.toolCallId)||!pending.erase(message.toolCallId))return Status::Invalid;
            bytes+=message.toolCallId.size();
        }else{
            if(!message.toolCallId.empty()||!pending.empty())return Status::Invalid;
            if(!message.toolCalls.empty()){
                if(message.role!="assistant"||!message.images.empty()||message.toolCalls.size()>16)return Status::Invalid;
                for(const auto& call:message.toolCalls){
                    if(!callIdentity(call.id)||!toolName(call.name)||!seen.insert(call.id).second||!serializer::validateJsonDocument(call.arguments,R"({"type":"object"})"))return Status::Invalid;
                    if(call.arguments.size()>65536)return Status::Limit;
                    pending.insert(call.id);bytes+=call.id.size()+call.name.size()+call.arguments.size();
                }
            }
        }
        if(bytes>profile.maxRequestBytes)return Status::Limit;
    }
    return pending.empty()?Status::Ok:Status::Invalid;
}
void responseCalls(const Json& response,std::vector<ToolCall>& calls){
    calls.clear();for(const auto& item:response.at("output"))if(item.value("type",std::string{})=="function_call")calls.push_back({item.at("call_id"),item.at("name"),item.at("arguments")});
}
Status validateToolCalls(const ModelProfile& profile,const Request& request,Result& result){
    if(result.toolCalls.empty())return Status::Ok;
    if(!profile.supportsToolCalls||request.tools.empty())return Status::Unsupported;
    if(result.toolCalls.size()>16)return Status::Limit;std::set<std::string> ids,historical;
    for(const auto& message:request.messages)for(const auto& call:message.toolCalls)historical.insert(call.id);
    for(std::size_t i=0;i<result.toolCalls.size();++i){
        const auto& call=result.toolCalls[i];const auto pointer="/tool_calls/"+std::to_string(i);
        if(!callIdentity(call.id)||!ids.insert(call.id).second||historical.contains(call.id)||!toolName(call.name)){result.issues.push_back({pointer,"tool_identity"});return Status::InvalidOutput;}
        const auto definition=std::find_if(request.tools.begin(),request.tools.end(),[&](const auto& tool){return tool.name==call.name;});
        if(definition==request.tools.end()){result.issues.push_back({pointer+"/name","undeclared_tool"});return Status::InvalidOutput;}
        if(call.arguments.size()>65536)return Status::Limit;
        const auto validation=serializer::validateJsonDocument(call.arguments,definition->parameters);
        if(!validation){for(const auto& issue:validation.issues)result.issues.push_back({pointer+"/arguments"+issue.pointer,issue.code});return Status::InvalidOutput;}
    }return Status::Ok;
}
