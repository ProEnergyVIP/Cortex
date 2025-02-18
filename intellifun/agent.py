from dataclasses import dataclass, field
import inspect
import json
from typing import Callable, List, Optional
from pydantic import BaseModel

from intellifun.LLM import get_random_error_message
from intellifun.debug import is_debug
from intellifun.message import Function, SystemMessage, ToolMessage, ToolMessageGroup, UserMessage, print_message


class Context(BaseModel):
    '''Base class for context used in the agent'''
    send_response: Optional[Callable] = None


@dataclass
class Tool:
    name: str
    func: callable
    description: str
    parameters: dict
    prompt: str = None

    __called_times = 0

    def check_call_limit(self, limit=10):
        '''Check if the tool has been called too many times'''
        if self.__called_times >= limit:
            return False
        return True
    
    def increment_call_count(self):
        '''Increment the call count of the tool'''
        self.__called_times += 1


@dataclass
class ToolFuncResult:
    '''Result of running a tool function'''
    content: str
    frontend_content: str = None
    tools: List[Tool] = field(default_factory=list)


MAX_RECENT_CALLS = 5  # Only track the last 5 calls

class Agent:
    def __init__(self, llm, tools=None, sys_prompt='', memory=None, context=None, json_reply=False):
        self.llm = llm
        self.tools = tools or []
        self.sys_msg = sys_prompt if isinstance(sys_prompt, SystemMessage) else SystemMessage(content=sys_prompt)
        self.memory = memory
        self.context = context
        self.json_reply = json_reply
        # Track recent tool calls to detect repetition
        self._recent_tool_calls = []

    def _is_repeated_tool_call(self, func: Function) -> bool:
        '''Check if this exact tool call was made recently'''
        current_call = (func.name, func.arguments)
        # Look for the same tool name and arguments in recent calls
        return current_call in self._recent_tool_calls

    def _add_tool_call(self, func: Function):
        '''Add a tool call to the recent calls list'''
        current_call = (func.name, func.arguments)
        self._recent_tool_calls.append(current_call)
        # Keep only the most recent calls
        self._recent_tool_calls = self._recent_tool_calls[-MAX_RECENT_CALLS:]

    def get_response_sender(self):
        '''Get the response sender function'''
        if self.context is not None:
            if not isinstance(self.context, Context):
                raise ValueError('context must be an instance of Context')
            return self.context.send_response
        return None
    
    def ask(self, message, user_name=None):
        '''Ask a question to the agent'''
        def err_func(msg):
            send_resp = self.get_response_sender()
            if send_resp:
                reply = {'message': msg} if self.json_reply else msg
                send_resp(reply)

        history_msgs = self.memory.load_memory()

        if isinstance(message, str):
            message = UserMessage(content=message, user_name=user_name)
        
        conversation = [message]

        i = 0
        reply = None

        while i < 10:
            i += 1

            msgs = [*history_msgs, *conversation]

            print_message(self.sys_msg)
            for m in msgs:
                print_message(m)
            
            # call the model
            try:
                ai_msg = self.llm.call(self.sys_msg, msgs, tools=self.tools)
            except Exception as e:
                err_msg = get_random_error_message()
                err_func(err_msg)
                raise e
            
            print_message(ai_msg)
            
            conversation.append(ai_msg)
            # check if we need to run a tool
            if ai_msg.tool_calls is not None:
                tool_msgs = self.process_func_call(ai_msg)
                conversation.append(tool_msgs)
            elif ai_msg.content:
                try:
                    reply = json.loads(ai_msg.content) if self.json_reply else ai_msg.content
                except json.JSONDecodeError as e:
                    err_msg = f'Error processing JSON message: {e}. Make sure your response is a valid JSON string, without the `json` tag.'
                    conversation.append(UserMessage(content=err_msg))
                    continue

                self.memory.add_messages(conversation)
                return reply

        self.memory.add_messages(conversation)

        return reply if reply is not None else 'Sorry, I am not sure how to answer that.'


    def process_func_call(self, ai_msg):
        '''Process the function call in the LLM result'''
        actions = []
        tools = []
        
        msgs = []
        for fc in ai_msg.tool_calls:
            # Check if this is a repeated tool call
            if self._is_repeated_tool_call(fc.function):
                msg = f'Tool "{fc.function.name}" was just called with the same arguments again. To prevent loops, please try a different approach or different arguments.'
                msgs.append(ToolMessage(content=msg, tool_call_id=fc.id))
                continue

            func_res = self.run_tool_func(fc.function)
            tool_res_msg = ToolMessage(content=func_res.content, tool_call_id=fc.id)
            msgs.append(tool_res_msg)
            
            if func_res.frontend_content is not None:
                actions.append(func_res.frontend_content)
            
            if func_res.tools:
                tools.extend(func_res.tools)

            # Track this tool call
            self._add_tool_call(fc.function)
        
        msg_group = ToolMessageGroup(tool_messages=msgs)

        # accumulate actions and send them to the frontend
        send_resp = self.get_response_sender()
        if actions and send_resp:
            send_resp({'actions': actions})
        
        # accumulate tools and add them to the agent
        if tools:
            self.tools.extend(tools)
        
        return msg_group

    def run_tool_func(self, func: Function):
        '''Run the given tool function and return the result'''
        tool_name = func.name
        
        for tool in self.tools:
            if tool.name == tool_name:
                if not tool.check_call_limit():
                    self.tools.remove(tool)
                    return ToolFuncResult(content=f'Tool "{tool_name}" has been called too many times, it will be removed from the list of available tools.')
                
                try:
                    tool_input = json.loads(func.arguments)
                    
                    sig = inspect.signature(tool.func)
                    num_params = len(sig.parameters)
  
                    if num_params == 0:
                        res = tool.func()
                    elif num_params == 1:
                        res = tool.func(tool_input)
                    elif num_params == 2:
                        res = tool.func(tool_input, self.context)
                    elif num_params == 3:
                        res = tool.func(tool_input, self.context, self)
                    else:
                        return ToolFuncResult(content=f'Invalid number of parameters for tool function {tool_name}: {num_params}')
                    
                    tool.increment_call_count()

                    # check result data type and wrap it into a ToolFuncResult
                    if isinstance(res, dict):
                        msg = res['message'] if 'message' in res else f'tool function {tool_name} finished'
                        func_res = ToolFuncResult(content=msg)
                        if 'tools' in res:
                            func_res.tools=res['tools']
                            del res['tools']
                        if 'action' in res:
                            func_res.frontend_content = res
                        return func_res
                    
                    if isinstance(res, str):
                        return ToolFuncResult(content=res)

                    return ToolFuncResult(content=f'tool function {tool_name} finished')
                except json.JSONDecodeError as e:
                    return ToolFuncResult(content=f'Error decoding JSON parameter for "{tool_name}": {e}. Use valid JSON string without the `json` tag.')
                except Exception as e:
                    if is_debug:
                        import traceback
                        traceback.print_exc()

                    return ToolFuncResult(content=f'Error running tool "{tool_name}": {e}')
        
        return ToolFuncResult(content=f'No tool named "{tool_name}" found. Do not call it again.')
