from enum import Enum
from dataclasses import is_dataclass, asdict
from openai import OpenAI, AsyncOpenAI

from intellifun.backend import LLMBackend
from intellifun.tool import (
    FunctionTool,
    WebSearchTool,
    CodeInterpreterTool,
    FileSearchTool,
    MCPTool,
)
from intellifun.message import AIMessage, Function, MessageUsage, SystemMessage, ToolCalling, ToolMessage, ToolMessageGroup, UserMessage, UserVisionMessage


__openai_client = None
__async_openai_client = None

def get_openai_client():
    global __openai_client
    if __openai_client is None:
        __openai_client = OpenAI()
    return __openai_client

def get_async_openai_client():
    global __async_openai_client
    if __async_openai_client is None:
        __async_openai_client = AsyncOpenAI()
    return __async_openai_client


class GPTModels(str, Enum):
    '''OpenAI GPT models'''
    GPT_4 = 'gpt-4'

    GPT_4O = 'gpt-4o'

    GPT_4O_MINI = 'gpt-4o-mini'

    GPT_4_1 = 'gpt-4.1'

    GPT_4_1_MINI = 'gpt-4.1-mini'

    GPT_4_1_NANO = 'gpt-4.1-nano'

    GPT_4_TURBO = 'gpt-4-turbo-preview'  # for automatically the latest version of GPT-4 Turbo

    GPT_35_TURBO = 'gpt-3.5-turbo-0125'

    GPT_35_FINE_TUNED = 'ft:gpt-3.5-turbo-0613:pro-energy::7w4wKQiR'


class OpenAIBackend(LLMBackend):
    '''OpenAI backend for the LLM'''
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._register_message_encoders()
        self._register_tool_encoders()

    def _register_message_encoders(self):
        # Register pure functions
        self.register_message_encoder(SystemMessage, enc_openai_system)
        self.register_message_encoder(UserMessage, enc_openai_user)
        self.register_message_encoder(UserVisionMessage, enc_openai_uservision)
        self.register_message_encoder(AIMessage, enc_openai_ai)
        self.register_message_encoder(ToolMessage, enc_openai_tool)
        self.register_message_encoder(ToolMessageGroup, enc_openai_tool_group)
    
    def _register_tool_encoders(self):
        # Function tools (locally executed; provider receives function schema)
        self.register_tool_encoder(FunctionTool, enc_openai_function_tool)
        # Hosted tools intended for the Responses API. These encoders will produce
        # native tool payloads. Note: chat.completions may not accept these types.
        self.register_tool_encoder(WebSearchTool, enc_openai_web_search_tool)
        self.register_tool_encoder(CodeInterpreterTool, enc_openai_code_interpreter_tool)
        self.register_tool_encoder(FileSearchTool, enc_openai_file_search_tool)
        self.register_tool_encoder(MCPTool, enc_openai_mcp_tool)
    
    def _prepare_request_params(self, req):
        '''Prepare the request parameters for the OpenAI API'''
        msgs = []
        # system message first
        msgs.extend(self.encode_message(req.system_message))
        # then conversation messages
        for m in req.messages:
            msgs.extend(self.encode_message(m))

        params = {
            'model': self.model,
            'temperature': req.temperature,
            'messages': msgs,
        }
        if req.max_tokens:
            params['max_tokens'] = req.max_tokens

        if req.tools:
            tools = [self.encode_tool(t) for t in req.tools]
            params['tools'] = tools
            
        return params
    
    def _process_response(self, chat):
        '''Process the response from the OpenAI API'''
        resp = chat.choices[0]
        resp_msg = resp.message
        
        if resp_msg.tool_calls:
            tool_calls = [self.decode_toolcalling(t) for t in resp_msg.tool_calls]
        else:
            tool_calls = None

        # Create usage information if available
        usage = MessageUsage(
            prompt_tokens=chat.usage.prompt_tokens,
            completion_tokens=chat.usage.completion_tokens,
            cached_tokens=chat.usage.prompt_tokens_details.cached_tokens,
            total_tokens=chat.usage.total_tokens
        )

        return AIMessage(content=resp_msg.content,
                        tool_calls=tool_calls,
                        usage=usage,
                        model=chat.model)
    
    def call(self, req):
        '''Call the OpenAI model with the request and return the response as an AIMessage'''
        params = self._prepare_request_params(req)
        client = get_openai_client()
        chat = client.chat.completions.create(**params)
        return self._process_response(chat)
    
    async def async_call(self, req):
        '''Async call to the OpenAI model with the request and return the response as an AIMessage'''
        params = self._prepare_request_params(req)
        client = get_async_openai_client()
        chat = await client.chat.completions.create(**params)
        return self._process_response(chat)

    def encode_toolcalling(self, tool_call):
        '''encode a ToolCalling object as a dictionary for the OpenAI API'''
        f = tool_call.function
        return {'id': tool_call.id,
                'type': tool_call.type,
                'function': {'name': f.name, 'arguments': f.arguments}
                }
    
    def decode_toolcalling(self, m):
        '''decode a tool call message from openAI to a ToolCalling object'''
        func = m.function
        fc = Function(name=func.name, arguments=func.arguments)
        return ToolCalling(id=m.id, type=m.type, function=fc)


for m in GPTModels:
    LLMBackend.register_backend(m, OpenAIBackend(m))

# --- Pure encoder functions for OpenAI ---
def enc_openai_system(msg: SystemMessage):
    return {'role': 'system', 'content': msg.content}

def enc_openai_user(msg: UserMessage):
    return {'role': 'user', 'content': msg.build_content()}

def enc_openai_uservision(msg: UserVisionMessage):
    message = msg.build_content()
    if not msg.image_urls:
        return {'role': 'user', 'content': message}
    msgs = [{'type': 'text', 'text': message}]
    for url in msg.image_urls:
        msgs.append({'type': 'image_url',
                     'image_url': {'url': url,
                                   'detail': 'low'
                                   }
                     })
    return {'role': 'user', 'content': msgs}

def enc_openai_ai(msg: AIMessage):
    m = {'role': 'assistant', 'content': msg.content}
    if msg.tool_calls:
        m['tool_calls'] = [encode_toolcalling_openai(t) for t in msg.tool_calls]
    return m

def enc_openai_tool(msg: ToolMessage):
    return {'role': 'tool',
            'content': msg.content,
            'tool_call_id': msg.tool_call_id}

def enc_openai_tool_group(msg: ToolMessageGroup):
    # Expand into multiple tool messages
    return [enc_openai_tool(tm) for tm in msg.tool_messages]

def encode_toolcalling_openai(tool_call):
    f = tool_call.function
    return {'id': tool_call.id,
            'type': tool_call.type,
            'function': {'name': f.name, 'arguments': f.arguments}
            }

# --- Tool encoder pure functions for OpenAI ---

def _strip_none(obj):
    """Recursively remove keys with None values from dicts and process lists.

    Dataclasses are converted to dicts via asdict, then cleaned.
    """
    if is_dataclass(obj):
        obj = asdict(obj)
    if isinstance(obj, dict):
        return {k: _strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none(v) for v in obj]
    return obj

def enc_openai_function_tool(tool: FunctionTool):
    return {
        'type': 'function',
        'strict': tool.strict,
        'name': tool.name,
        'description': tool.description,
        'parameters': tool.parameters,
    }


def enc_openai_web_search_tool(tool: WebSearchTool):
    payload = _strip_none(tool)
    payload['type'] = 'web_search'
    return payload


def enc_openai_code_interpreter_tool(tool: CodeInterpreterTool):
    payload = _strip_none(tool)
    payload['type'] = 'code_interpreter'
    return payload


def enc_openai_file_search_tool(tool: FileSearchTool):
    payload = _strip_none(tool)
    payload['type'] = 'file_search'
    return payload


def enc_openai_mcp_tool(tool: MCPTool):
    payload = _strip_none(tool)
    payload['type'] = 'mcp'
    return payload
