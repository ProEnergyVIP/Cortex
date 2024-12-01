from dataclasses import dataclass, field
import inspect
import json
from typing import Callable, List, Optional
from pydantic import BaseModel

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

@dataclass
class ToolFuncResult:
    '''Result of running a tool function'''
    content: str
    frontend_content: str = None
    tools: List[Tool] = field(default_factory=list)


class Agent:
    def __init__(self, llm, tools=[], sys_prompt='', memory=None, context=None, json_reply=False):
        self.llm = llm
        self.tools = tools
        self.sys_msg = sys_prompt if isinstance(sys_prompt, SystemMessage) else SystemMessage(content=sys_prompt)
        self.memory = memory
        self.context = context
        self.json_reply = json_reply
    
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
            msgs = [*history_msgs, *conversation]

            print_message(self.sys_msg)
            for m in msgs:
                print_message(m)
            
            # call the model
            ai_msg = self.llm.call(self.sys_msg, msgs, tools=self.tools, error_func=err_func)
            
            print_message(ai_msg)
            
            conversation.append(ai_msg)
            # check if we need to run a tool
            if ai_msg.tool_calls is not None:
                self.process_func_call(ai_msg, conversation)
            elif ai_msg.content:
                try:
                    reply = self.process_ai_message(ai_msg)
                except Exception as e:
                    conversation.append(UserMessage(content=f'Error processing JSON message: {e}. Please make sure your response is a valid JSON string that can be loaded with python json.loads() function directly.'))
                    continue

                self.memory.add_messages(conversation)
                return reply
        
        self.memory.add_messages(conversation)

        return reply if reply is not None else 'Sorry, I am not sure how to answer that.'
    

    def process_ai_message(self, ai_msg):
        '''Process the result message from LLM'''
        return json.loads(ai_msg.content) if self.json_reply else ai_msg.content

    def process_func_call(self, ai_msg, conversation):
        '''Process the function call in the LLM result'''
        actions = []
        tools = []
        
        msgs = []
        for fc in ai_msg.tool_calls:
            func_res = self.run_tool_func(fc.function)
            tool_res_msg = ToolMessage(content=func_res.content, tool_call_id=fc.id)
            msgs.append(tool_res_msg)
            
            if func_res.frontend_content is not None:
                actions.append(func_res.frontend_content)
            if func_res.tools:
                tools.extend(func_res.tools)
        
        msg_group = ToolMessageGroup(tool_messages=msgs)
        conversation.append(msg_group)

        # accumulate actions and send them to the frontend
        send_resp = self.get_response_sender()
        if actions and send_resp:
            send_resp({'actions': actions})
        
        # accumulate tools and add them to the agent
        if tools:
            self.tools.extend(tools)

    def run_tool_func(self, func: Function):
        '''Run the given tool function and return the result'''
        tool_name = func.name
        
        for tool in self.tools:
            if tool.name == tool_name:
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
                    elif isinstance(res, str):
                        return ToolFuncResult(content=res)
                    else:
                        return ToolFuncResult(content=f'tool function {tool_name} finished')
                except Exception as e:
                    if is_debug:
                        import traceback
                        traceback.print_exc()

                    return ToolFuncResult(f'Error running tool "{tool_name}": {e}')
        
        return ToolFuncResult(f'No tool named "{tool_name}" found')
