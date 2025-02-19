from enum import Enum

from intellifun.backend import LLMBackend, LLMRequest
from intellifun.message import AIMessage, Function, ToolCalling, ToolMessageGroup, UserMessage, UserVisionMessage, MessageUsage


__anthropic_client = None

def get_anthropic_client():
    global __anthropic_client
    if __anthropic_client is None:
        from anthropic import Anthropic
        __anthropic_client = Anthropic()
    return __anthropic_client


class AnthropicModels(str, Enum):
    CLAUDE_3_5_SONNET = 'claude-3.5-sonnet-20241022'


class AnthropicBackend(LLMBackend):
    def __init__(self, model):
        self.model = model
    
    def call(self, req: LLMRequest) -> AIMessage | None:
        '''Call the Anthropic model with the request and return the response as an AIMessage'''
        msgs = [self.encode_msg(m) for m in req.messages]

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
        
        client = get_anthropic_client()
        resp = client.messages.create(**params)

        return self.decode_result(resp)

    def encode_msg(self, msg):
        '''encode a message as a dictionary for the Anthropic API'''
        if isinstance(msg, UserVisionMessage):
            raise ValueError('Vision messages are not supported by the Anthropic API')
        
        if isinstance(msg, UserMessage):
            return {'role': 'user', 'content': msg.build_content() }
        
        if isinstance(msg, AIMessage):
            txt = {'type': 'text', 'text': msg.content}
            msgs = [txt]
            if msg.tool_calls:
                for c in msg.tool_calls:
                    msgs.append(self.encode_toolcalling(c))
            return {'role': 'assistant', 'content': msgs}
        
        if isinstance(msg, ToolMessageGroup):
            msgs = []
            for tm in msg.tool_messages:
                msgs.append({
                    'type': 'tool_result',
                    'content': tm.content,
                    'tool_call_id': tm.tool_call_id
                })
            
            return {'role': 'user',
                    'content': msgs }
        
        return {'role': 'user', 'content': msg.content}

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
