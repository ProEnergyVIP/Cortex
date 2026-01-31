from enum import Enum

import logging
from typing import Any, AsyncIterable, Iterable, Optional

from cortex.backend import LLMBackend, LLMRequest
from cortex.message import AIMessage, DeveloperMessage, FunctionCall, ToolMessageGroup, UserMessage, UserVisionMessage, MessageUsage

logger = logging.getLogger(__name__)


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
        self.register_message_encoder(DeveloperMessage, enc_anthropic_developer)
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

    def _event_type(self, event: Any) -> Optional[str]:
        if isinstance(event, dict):
            return event.get('type')
        return getattr(event, 'type', None)

    def _event_text_delta(self, event: Any) -> Optional[str]:
        etype = self._event_type(event)
        if etype != 'content_block_delta':
            return None

        if isinstance(event, dict):
            delta = event.get('delta')
            delta_type = None if delta is None else delta.get('type')
            if delta_type != 'text_delta':
                return None
            return delta.get('text')

        delta = getattr(event, 'delta', None)
        if delta is None:
            return None
        if getattr(delta, 'type', None) != 'text_delta':
            return None
        return getattr(delta, 'text', None)

    def stream(self, req: LLMRequest) -> Iterable[str]:
        params = self._prepare_request_params(req)
        client = get_anthropic_client()

        stream_method = getattr(client.messages, 'stream', None)
        if callable(stream_method):
            try:
                with stream_method(**params) as stream:
                    for event in stream:
                        delta = self._event_text_delta(event)
                        if delta:
                            yield delta
                return
            except Exception as e:
                logger.error('Anthropic streaming request failed: %s', e)
                raise e

        yield from super().stream(req)

    async def async_stream(self, req: LLMRequest) -> AsyncIterable[str]:
        params = self._prepare_request_params(req)
        client = get_async_anthropic_client()

        stream_method = getattr(client.messages, 'stream', None)
        if callable(stream_method):
            try:
                async with stream_method(**params) as stream:
                    async for event in stream:
                        delta = self._event_text_delta(event)
                        if delta:
                            yield delta
                return
            except Exception as e:
                logger.error('Anthropic streaming request failed: %s', e)
                raise e

        async for chunk in super().async_stream(req):
            yield chunk

    def decode_result(self, resp):
        '''decode the result from the Anthropic API'''
        resp_message = None
        tool_calls = []
        for msg in resp.content:
            if msg.type == 'text':
                resp_message = msg.text
                break
            
            if msg.type == 'tool_use':
                tool_calls.append(self.decode_function_call(msg))
                break

        # Extract usage information from response
        usage = MessageUsage(
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
            cached_tokens=resp.usage.cache_read_input_tokens if resp.usage.cache_read_input_tokens else 0,
            total_tokens=resp.usage.input_tokens + resp.usage.output_tokens
        )

        return AIMessage(content=resp_message,
                         function_calls=tool_calls,
                         usage=usage,
                         model=resp.model)

    def decode_function_call(self, m):
        '''decode a tool call from the Anthropic API response'''
        return FunctionCall(
            id=m.id,
            type=m.type,
            name=m.name,
            arguments=m.input,
            call_id=m.id
        )

    def encode_tool(self, tool):
        '''encode a tool as a dictionary for the Anthropic API'''
        return {
            'name': tool.name,
            'description': tool.description,
            'input_schema': tool.parameters,
        }

for m in AnthropicModels:
    LLMBackend.register_backend(m, AnthropicBackend)

# --- Pure encoder functions for Anthropic ---
def enc_anthropic_user_vision(msg: UserVisionMessage):
    raise ValueError('Vision messages are not supported by the Anthropic API')

def enc_anthropic_user(msg: UserMessage):
    return {'role': 'user', 'content': msg.build_content()}

def enc_anthropic_developer(msg: DeveloperMessage):
    return {'role': 'user', 'content': f'<<developer>>{msg.content}<<developer>>'}

def enc_anthropic_ai(msg: AIMessage):
    txt = {'type': 'text', 'text': msg.content}
    msgs = [txt]
    if msg.function_calls:
        for c in msg.function_calls:
            msgs.append(encode_function_call_anthropic(c))
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

def encode_function_call_anthropic(tool_call):
    return {
        'id': tool_call.id,
        'type': tool_call.type,
        'name': tool_call.name,
        'input': tool_call.arguments,
    }
