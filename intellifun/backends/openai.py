from enum import Enum
from openai import OpenAI, AsyncOpenAI

from intellifun.backend import LLMBackend
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
        self.model = model
    
    def _prepare_request_params(self, req):
        '''Prepare the request parameters for the OpenAI API'''
        msgs = []
        # flatten the messages
        for m in req.messages:
            if isinstance(m, ToolMessageGroup):
                for tm in m.tool_messages:
                    msgs.append(tm)
            else:
                msgs.append(m)
    
        msgs.insert(0, req.system_message)
        msgs = [self.encode_msg(m) for m in msgs]

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

    def encode_msg(self, msg):
        '''encode a message as a dictionary for the OpenAI API'''
        if isinstance(msg, SystemMessage):
            return {'role': 'system', 'content': msg.content}
        
        if isinstance(msg, UserVisionMessage):
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
            return {'role': 'user', 'content': msgs }
        
        if isinstance(msg, UserMessage):
            return {'role': 'user', 'content': msg.build_content() }
        
        if isinstance(msg, AIMessage):
            m = {'role': 'assistant', 'content': msg.content}

            if msg.tool_calls:
                m['tool_calls'] = [self.encode_toolcalling(t) for t in msg.tool_calls]
            
            return m
        
        if isinstance(msg, ToolMessage):
            return {'role': 'tool',
                    'content': msg.content,
                    'tool_call_id': msg.tool_call_id
                   }
        
        return {'role': 'user', 'content': msg.content}

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
    
    def encode_tool(self, tool):
        '''encode an Agent's tool as a dictionary for the OpenAI API'''
        return {'type': 'function', 
                'function': {
                    'name': tool.name,
                    'description': tool.description,
                    'parameters': tool.parameters,
                    }
                }

for m in GPTModels:
    LLMBackend.register_backend(m, OpenAIBackend(m))
