from enum import Enum

from intellifun.backend import LLMBackend, LLMRequest
from intellifun.message import AIMessage, Function, ToolCalling, ToolMessageGroup, UserMessage, UserVisionMessage, MessageUsage


__anthropic_client = None
__async_anthropic_client = None

def get_anthropic_client():
    global __anthropic_client
    if __anthropic_client is None:
        from anthropic import Anthropic
        __anthropic_client = Anthropic()
    return __anthropic_client

def get_async_anthropic_client():
    global __async_anthropic_client
    if __async_anthropic_client is None:
        from anthropic import AsyncAnthropic
        __async_anthropic_client = AsyncAnthropic()
    return __async_anthropic_client


class AnthropicModels(str, Enum):
    CLAUDE_3_5_SONNET = 'claude-3.5-sonnet-20241022'


class AnthropicBackend(LLMBackend):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._register_message_encoders()

    def _register_message_encoders(self):
        self.register_message_encoder(UserVisionMessage, enc_anthropic_user_vision)
        self.register_message_encoder(UserMessage, enc_anthropic_user)
        self.register_message_encoder(AIMessage, enc_anthropic_ai)
        self.register_message_encoder(ToolMessageGroup, enc_anthropic_tool_group)
    
    def _prepare_request_params(self, req: LLMRequest):
        '''Prepare the request parameters for the Anthropic API'''
        msgs = []
        for m in req.messages:
            msgs.extend(self.encode_message(m))

        params = {
            'model': self.model,
            'temperature': req.temperature,
            'messages': msgs,
            'system': req.system_message.content,
            'max_tokens': req.max_tokens or 5000,
        }

        if req.tools:
            tools = [self.encode_tool(t) for t in req.tools]
            params['tools'] = tools
            
        return params
    
    def call(self, req: LLMRequest) -> AIMessage | None:
        '''Call the Anthropic model with the request and return the response as an AIMessage'''
        params = self._prepare_request_params(req)
        client = get_anthropic_client()
        resp = client.messages.create(**params)
        return self.decode_result(resp)
    
    async def async_call(self, req: LLMRequest) -> AIMessage | None:
        '''Async call to the Anthropic model with the request and return the response as an AIMessage'''
        params = self._prepare_request_params(req)
        client = get_async_anthropic_client()
        resp = await client.messages.create(**params)
        return self.decode_result(resp)

    def decode_result(self, resp):
        '''decode the result from the Anthropic API'''
        resp_message = None
        tool_calls = []
        for msg in resp.content:
            if msg.type == 'text':
                resp_message = msg.text
                break
            
            if msg.type == 'tool_use':
                tool_calls.append(self.decode_toolcalling(msg))
                break

        # Extract usage information from response
        usage = MessageUsage(
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
            cached_tokens=resp.usage.cache_read_input_tokens if resp.usage.cache_read_input_tokens else 0,
            total_tokens=resp.usage.input_tokens + resp.usage.output_tokens
        )

        return AIMessage(content=resp_message,
                         tool_calls=tool_calls,
                         usage=usage,
                         model=resp.model)

    def encode_toolcalling(self, tool_call):
        '''encode a tool call as a dictionary for the Anthropic API'''
        return {
            'id': tool_call.id,
            'type': tool_call.type,
            'name': tool_call.function.name,
            'input': tool_call.function.arguments,
        }

    def decode_toolcalling(self, m):
        '''decode a tool call from the Anthropic API response'''
        fc = Function(name=m.name, arguments=m.input)
        return ToolCalling(id=m.id, type=m.type, function=fc)

    def encode_tool(self, tool):
        '''encode a tool as a dictionary for the Anthropic API'''
        return {
            'name': tool.name,
            'description': tool.description,
            'input_schema': tool.parameters,
        }

for m in AnthropicModels:
    LLMBackend.register_backend(m, AnthropicBackend(m))

# --- Pure encoder functions for Anthropic ---
def enc_anthropic_user_vision(msg: UserVisionMessage):
    raise ValueError('Vision messages are not supported by the Anthropic API')

def enc_anthropic_user(msg: UserMessage):
    return {'role': 'user', 'content': msg.build_content()}

def enc_anthropic_ai(msg: AIMessage):
    txt = {'type': 'text', 'text': msg.content}
    msgs = [txt]
    if msg.tool_calls:
        for c in msg.tool_calls:
            msgs.append(encode_toolcalling_anthropic(c))
    return {'role': 'assistant', 'content': msgs}

def enc_anthropic_tool_group(msg: ToolMessageGroup):
    msgs = []
    for tm in msg.tool_messages:
        msgs.append({
            'type': 'tool_result',
            'content': tm.content,
            'tool_call_id': tm.tool_call_id
        })
    return {'role': 'user', 'content': msgs}

def encode_toolcalling_anthropic(tool_call):
    return {
        'id': tool_call.id,
        'type': tool_call.type,
        'name': tool_call.function.name,
        'input': tool_call.function.arguments,
    }
