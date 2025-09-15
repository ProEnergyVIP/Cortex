# Tool classes now live in intellifun.tool
from intellifun.tool import BaseTool, FunctionTool, Tool
import json
import rich

from intellifun.LLM import get_random_error_message
from intellifun.debug import is_debug
from intellifun.message import (FunctionCall, SystemMessage,
                                ToolMessage, ToolMessageGroup, UserMessage, AgentUsage, print_message)

from intellifun.logging_config import get_default_logging_config

# Re-export Tool for backward compatibility (old imports from agent)
__all__ = ['Agent', 'Tool']


MAX_RECENT_CALLS = 5  # Only track the last 5 calls

START_DELIM = '-' * 80
END_DELIM = '^' * 80

class Agent:
    def __init__(self, llm, tools=None, sys_prompt='', memory=None, context=None, json_reply=False, 
                 name=None, logging_config=None, tool_call_limit=10):
        self.llm = llm
        self.tools = tools or []

        # If there are more than 5 tools, use a dictionary for faster lookup
        if len(self.tools) > 5:
            tmp = {}
            for tool in self.tools:
                name = getattr(tool, 'name', None)
                if name:
                    tmp[name] = tool
            self.tools_dict = tmp if tmp else None
        else:
            self.tools_dict = None
        
        self.sys_msg = sys_prompt if isinstance(sys_prompt, SystemMessage) else SystemMessage(content=sys_prompt)
        self.memory = memory
        self.context = context
        self.json_reply = json_reply
        # Track recent tool calls to detect repetition
        self._recent_tool_calls = []
        # Agent name and logging configuration
        self.name = name
        # If no logging config is provided, use the global default
        self.logging_config = logging_config or get_default_logging_config()
        self.tool_call_limit = tool_call_limit
    
    def _find_tool(self, tool_name: str) -> BaseTool | None:
        if self.tools_dict:
            return self.tools_dict.get(tool_name)

        for tool in self.tools:
            if getattr(tool, 'name', None) == tool_name:
                return tool
        return None

    def _is_repeated_tool_call(self, func: FunctionCall) -> bool:
        '''Check if this exact tool call was made recently'''
        current_call = (func.name, func.arguments)
        # Look for the same tool name and arguments in recent calls
        return current_call in self._recent_tool_calls

    def _add_tool_call(self, func: FunctionCall):
        '''Add a tool call to the recent calls list'''
        current_call = (func.name, func.arguments)
        self._recent_tool_calls.append(current_call)
        # Keep only the most recent calls
        self._recent_tool_calls = self._recent_tool_calls[-MAX_RECENT_CALLS:]

    def _prepare_conversation(self, message, user_name, history_msgs):
        '''Prepare the conversation with the user message'''
        # Add the user's message to the conversation
        if isinstance(message, str):
            message = UserMessage(content=message, user_name=user_name)
        
        conversation = [message]
        
        # Print agent name if available
        self.print_name()

        show_sys_prompt = self.logging_config.print_system_prompt
        show_msgs = self.logging_config.print_messages

        # Print system prompt if enabled
        if show_sys_prompt:
            print_message(self.sys_msg)

        if show_msgs:
            print(START_DELIM)

            # log history when showing system prompt
            if show_sys_prompt:
                for m in history_msgs:
                    print_message(m)

            print_message(message)
            
        return message, conversation, show_msgs
    
    def _handle_response(self, conversation, agent_usage, usage, show_msgs):
        '''Handle the response after the conversation is complete'''
        if self.memory:
            self.memory.add_messages(conversation)
        
        if self.logging_config.print_usage_report:
            print(agent_usage.format())
        if usage:
            usage.merge(agent_usage)
        
        if show_msgs:
            print(END_DELIM)
            
    async def _async_handle_response(self, conversation, agent_usage, usage, show_msgs):
        '''Handle the response after the conversation is complete (async version)'''
        if self.memory:
            await self.memory.add_messages(conversation)
        
        if self.logging_config.print_usage_report:
            print(agent_usage.format())
        if usage:
            usage.merge(agent_usage)
        
        if show_msgs:
            print(END_DELIM)

    def ask(self, message, user_name=None, usage=None, loop_limit=10):
        '''Ask a question to the agent, and get a response

        Args:
            message (str or Message): The message to ask
            user_name (str, optional): The name of the user. Defaults to None.
            usage (AgentUsage, optional): Object to accumulate token usage across models.
                You can pass an AgentUsage object to track usage across multiple calls.
            loop_limit (int, optional): The maximum number of times to call the model.
                Defaults to 10.

        Returns:
            str: The response from the agent
        '''
        reply = None
        agent_usage = AgentUsage()  # Track total usage across all calls
        
        # Get history messages from memory
        history_msgs = self.memory.load_memory() if self.memory else []
        
        message, conversation, show_msgs = self._prepare_conversation(message, user_name, history_msgs)
        
        # Main conversation loop
        for _ in range(loop_limit):
            msgs = [*history_msgs, *conversation]
            
            # call the model
            try:
                ai_msg = self.llm.call(self.sys_msg, msgs, tools=self.tools)
                # Add usage to AgentUsage if available
                if ai_msg.usage and ai_msg.model:
                    agent_usage.add_usage(ai_msg.model, ai_msg.usage)
            except Exception as _:
                err_msg = get_random_error_message()
                reply = {'message': err_msg} if self.json_reply else err_msg
                break

            reply = self._process_ai_message(ai_msg, conversation, show_msgs)
            if reply is not None:
                self._handle_response(conversation, agent_usage, usage, show_msgs)
                return reply

        self._handle_response(conversation, agent_usage, usage, show_msgs)
        return reply if reply is not None else 'Sorry, I am not sure how to answer that.'

    async def async_ask(self, message, user_name=None, usage=None, loop_limit=10):
        '''Ask a question to the agent asynchronously, and get a response

        Args:
            message (str or Message): The message to ask
            user_name (str, optional): The name of the user. Defaults to None.
            usage (AgentUsage, optional): Object to accumulate token usage across models.
                You can pass an AgentUsage object to track usage across multiple calls.
            loop_limit (int, optional): The maximum number of times to call the model.
                Defaults to 10.
        
        Returns:
            str: The response from the agent
        '''
        reply = None
        agent_usage = AgentUsage()  # Track total usage across all calls
        
        # Get history messages from memory asynchronously
        history_msgs = await self.memory.load_memory() if self.memory else []
        
        message, conversation, show_msgs = self._prepare_conversation(message, user_name, history_msgs)
        
        # Main conversation loop
        for _ in range(loop_limit):
            msgs = [*history_msgs, *conversation]
            
            # call the model asynchronously
            try:
                ai_msg = await self.llm.async_call(self.sys_msg, msgs, tools=self.tools)
                # Add usage to AgentUsage if available
                if ai_msg.usage and ai_msg.model:
                    agent_usage.add_usage(ai_msg.model, ai_msg.usage)
            except Exception as _:
                err_msg = get_random_error_message()
                reply = {'message': err_msg} if self.json_reply else err_msg
                break

            # Process the AI message
            reply = await self._async_process_ai_message(ai_msg, conversation, show_msgs)
            if reply is not None:
                await self._async_handle_response(conversation, agent_usage, usage, show_msgs)
                return reply

        await self._async_handle_response(conversation, agent_usage, usage, show_msgs)
        return reply if reply is not None else 'Sorry, I am not sure how to answer that.'

    def print_name(self):
        '''Print the agent name if available'''        
        if self.name:
            rich.print(f"[bold cyan]Agent: {self.name}[/bold cyan]")

    def _process_ai_message(self, ai_msg, conversation, show_msgs):
        """Process the AI message and determine next steps"""
        conversation.append(ai_msg)

        # check if we need to run a tool
        if ai_msg.function_calls is not None:
            tool_msgs = self.process_func_call(ai_msg, show_msgs)
            conversation.append(tool_msgs)
            return None  # Continue the conversation
        elif ai_msg.content:
            if show_msgs:
                print_message(ai_msg)
            try:
                return json.loads(ai_msg.content) if self.json_reply else ai_msg.content
            except json.JSONDecodeError as e:
                err_msg = f"Error processing JSON message: {e}. Make sure your response is a valid JSON string and do not include the `json` tag."
                conversation.append(UserMessage(content=err_msg))
                return None  # Continue the conversation with error message

        return None  # Continue the conversation

    async def _async_process_ai_message(self, ai_msg, conversation, show_msgs):
        """Process the AI message asynchronously and determine next steps"""
        conversation.append(ai_msg)

        # check if we need to run a tool
        if ai_msg.function_calls is not None:
            tool_msgs = await self.async_process_func_call(ai_msg, show_msgs)
            conversation.append(tool_msgs)
            return None  # Continue the conversation
        elif ai_msg.content:
            if show_msgs:
                print_message(ai_msg)
            try:
                return json.loads(ai_msg.content) if self.json_reply else ai_msg.content
            except json.JSONDecodeError as e:
                err_msg = f"Error processing JSON message: {e}. Make sure your response is a valid JSON string and do not include the `json` tag."
                conversation.append(UserMessage(content=err_msg))
                return None  # Continue the conversation with error message

        return None  # Continue the conversation

    def process_func_call(self, ai_msg, show_msgs):
        '''Process the function call in the LLM result'''
        msgs = []
        for fc in ai_msg.function_calls:
            # Check if this is a repeated tool call
            if show_msgs:
                rich.print(f'[bold purple]Tool: {fc.name}[/bold purple]')
            
            if self._is_repeated_tool_call(fc):
                msg = f'Tool "{fc.name}" was just called with the same arguments again. To prevent loops, please try a different approach or different arguments.'
                msgs.append(ToolMessage(content=msg, tool_call_id=fc.call_id or fc.id))
                continue

            func_res = self.run_tool_func(fc)
            tool_res_msg = ToolMessage(content=func_res, tool_call_id=fc.call_id or fc.id)
            msgs.append(tool_res_msg)

            if show_msgs:
                print_message(tool_res_msg)

            # Track this tool call
            self._add_tool_call(fc)
        
        msg_group = ToolMessageGroup(tool_messages=msgs)
        
        return msg_group

    async def async_process_func_call(self, ai_msg, show_msgs):
        '''Process the function call in the LLM result asynchronously'''
        msgs = []
        for fc in ai_msg.function_calls:
            # Check if this is a repeated tool call
            if show_msgs:
                rich.print(f'[bold purple]Tool: {fc.name}[/bold purple]')
            
            if self._is_repeated_tool_call(fc):
                msg = f'Tool "{fc.name}" was just called with the same arguments again. To prevent loops, please try a different approach or different arguments.'
                msgs.append(ToolMessage(content=msg, tool_call_id=fc.call_id or fc.id))
                continue

            func_res = await self.async_run_tool_func(fc)
            tool_res_msg = ToolMessage(content=func_res, tool_call_id=fc.call_id or fc.id)
            msgs.append(tool_res_msg)

            if show_msgs:
                print_message(tool_res_msg)

            # Track this tool call
            self._add_tool_call(fc)
        
        msg_group = ToolMessageGroup(tool_messages=msgs)
        
        return msg_group

    def run_tool_func(self, func: FunctionCall):
        '''Run the given tool function and return the result'''
        tool_name = func.name
        
        tool = self._find_tool(tool_name)
        if tool is None:
            return f'No tool named "{tool_name}" found. Do not call it again.'
        # Only FunctionTool can run locally
        if not isinstance(tool, FunctionTool):
            return f'Tool "{tool_name}" is not a function tool and cannot be executed locally. Do not call it directly.'
        
        if not tool.check_call_limit(self.tool_call_limit):
            self.tools.remove(tool)
            return f'Tool "{tool_name}" has been called too many times, it will be removed from the list of available tools.'
        
        try:
            tool_input = json.loads(func.arguments) if isinstance(func.arguments, str) else func.arguments
            
            res = tool.run(tool_input, self.context, self)
            
            if isinstance(res, str):
                return res

            return json.dumps(res)
        except json.JSONDecodeError as e:
            return f'Error decoding JSON parameter for "{tool_name}": {e}. Use valid JSON string without the `json` tag.'
        except Exception as e:
            if is_debug:
                import traceback
                traceback.print_exc()

            return f'Error running tool "{tool_name}": {e}'

    async def async_run_tool_func(self, func: FunctionCall):
        '''Run the given tool function asynchronously and return the result'''
        tool_name = func.name
        
        tool = self._find_tool(tool_name)
        if tool is None:
            return f'No tool named "{tool_name}" found. Do not call it again.'
        # Only FunctionTool can run locally
        if not isinstance(tool, FunctionTool):
            return f'Tool "{tool_name}" is not a function tool and cannot be executed locally. Do not call it directly.'
        
        if not tool.check_call_limit(self.tool_call_limit):
            self.tools.remove(tool)
            return f'Tool "{tool_name}" has been called too many times, it will be removed from the list of available tools.'
        
        try:
            tool_input = json.loads(func.arguments) if isinstance(func.arguments, str) else func.arguments
            
            res = await tool.async_run(tool_input, self.context, self)
            
            if isinstance(res, str):
                return res
            
            return json.dumps(res)
        except json.JSONDecodeError as e:
            return f'Error decoding JSON parameter for "{tool_name}": {e}. Use valid JSON string without the `json` tag.'
        except Exception as e:
            if is_debug:
                import traceback
                traceback.print_exc()

            return f'Error running tool "{tool_name}": {e}'