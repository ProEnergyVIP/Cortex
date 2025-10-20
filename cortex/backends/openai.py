from enum import Enum
from dataclasses import is_dataclass, asdict
import logging
from openai import OpenAI, AsyncOpenAI

from cortex.backend import LLMBackend
from cortex.tool import (
    FunctionTool,
    WebSearchTool,
    CodeInterpreterTool,
    FileSearchTool,
    MCPTool,
)
from cortex.message import AIMessage, DeveloperMessage, MessageUsage, SystemMessage, FunctionCall, ToolCalling, ToolMessage, ToolMessageGroup, UserMessage, UserVisionMessage

logger = logging.getLogger(__name__)

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
    GPT_5 = 'gpt-5'

    GPT_5_MINI = 'gpt-5-mini'

    GPT_5_NANO = 'gpt-5-nano'

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
        self.register_message_encoder(DeveloperMessage, enc_openai_developer)
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
        msgs = [payload
                for m in req.messages
                for payload in (self.encode_message(m) or [])
                if payload is not None]

        params = {
            'model': self.model,
            'instructions': req.system_message.content,
            'input': msgs,
        }

        if req.temperature is not None:
            params['temperature'] = req.temperature
        
        if req.max_tokens:
            params['max_tokens'] = req.max_tokens

        if req.tools:
            tools = [self.encode_tool(t) for t in req.tools]
            params['tools'] = tools
        
        if req.reasoning_effort:
            params['reasoning'] = {
                'effort': req.reasoning_effort.value
            }
        
        logger.debug('OpenAI request params: %s', params)
        
        return params
    
    def _process_response(self, response):
        '''Process the response from the OpenAI API'''
        logger.debug('OpenAI response output: %s', response.output)

        content = None
        tool_calls = []
        output_dicts = []

        for m in response.output:
            if m.type == "message" and content is None:
                content_obj = m.content[0]
                if content_obj.type == 'output_text':
                    content = content_obj.text
                output_dicts.append(m.model_dump(exclude_none=True))
            elif m.type == "function_call":
                val = m.model_dump(exclude_none=True)
                tool_calls.append(FunctionCall(**val))
                output_dicts.append(val)
            elif m.type == "reasoning":
                output_dicts.append(m.model_dump(exclude_none=True))
        
        logger.debug('OpenAI response content: %s', content)
        logger.debug('OpenAI response tool calls: %s', tool_calls)
        logger.debug('Actual output: %s', output_dicts)
        
        # Create usage information if available
        usg_obj = response.usage
        usage = MessageUsage(
            prompt_tokens=usg_obj.input_tokens,
            cached_tokens=usg_obj.input_tokens_details.cached_tokens,
            completion_tokens=usg_obj.output_tokens,
            total_tokens=usg_obj.total_tokens,
        )

        return AIMessage(model=response.model,
                         content=content,
                         original_output=output_dicts,
                         function_calls=tool_calls,
                         usage=usage,
                         )
    
    def call(self, req):
        '''Call the OpenAI model with the request and return the response as an AIMessage'''
        params = self._prepare_request_params(req)
        client = get_openai_client()
        try:
            response = client.responses.create(**params)
        except Exception as e:
            logger.error('OpenAI request failed: %s', e)
            raise e
        return self._process_response(response)
            
    async def async_call(self, req):
        '''Async call to the OpenAI model with the request and return the response as an AIMessage'''
        params = self._prepare_request_params(req)
        client = get_async_openai_client()
        try:
            response = await client.responses.create(**params)
        except Exception as e:
            logger.error('OpenAI request failed: %s', e)
            raise e
        return self._process_response(response)

for m in GPTModels:
    LLMBackend.register_backend(m, OpenAIBackend)

# --- Pure encoder functions for OpenAI ---
def enc_openai_system(msg: SystemMessage):
    return {'role': 'system', 'content': msg.content}

def enc_openai_developer(msg: DeveloperMessage):
    return {'role': 'developer', 'content': msg.content}

def enc_openai_user(msg: UserMessage):
    if msg.images is not None and msg.files is not None:
        return {'role': 'user', 'content': msg.content}
    
    msgs = []

    if msg.content:
        msgs.append({'type': 'input_text', 'text': msg.content})

    if msg.images:
        msgs.extend([_strip_none(img) for img in msg.images])
    
    if msg.files:
        msgs.extend([_strip_none(file) for file in msg.files])
    
    return {'role': 'user', 'content': msgs}


def enc_openai_uservision(msg: UserVisionMessage):
    if not msg.image_urls:
        return {'role': 'user', 'content': msg.content}
    
    msgs = [{'type': 'input_text', 'text': msg.content}]
    for url in msg.image_urls:
        msgs.append({'type': 'input_image',
                     'image_url': url,
                     'detail': 'low',
                     })
    return {'role': 'user', 'content': msgs}

def enc_openai_ai(msg: AIMessage):
    # if there're old tool calls data
    if msg.tool_calls:
        # old tool calls data won't work anymore. just ignore them
        return None
    return msg.original_output

def enc_openai_old_toolcall(tc: ToolCalling):
    # convert the old ToolCalling format to new FunctionCall format
    return {'type': 'function_call',
            'name': tc.function.name,
            'arguments': tc.function.arguments,
            'call_id': tc.id
            }

def enc_openai_tool(msg: ToolMessage):
    return {'type': 'function_call_output',
            'output': msg.content,
            'call_id': msg.tool_call_id}

def enc_openai_tool_group(msg: ToolMessageGroup):
    # Expand into multiple tool messages
    return [enc_openai_tool(tm) for tm in msg.tool_messages]

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
    params = tool.parameters

    if 'additionalProperties' not in params:
        params['additionalProperties'] = False

    return {
        'type': 'function',
        'strict': tool.strict,
        'name': tool.name,
        'description': tool.description,
        'parameters': params,
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
