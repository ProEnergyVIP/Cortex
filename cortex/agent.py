# Standard library
import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

# Local imports
from cortex.debug import is_debug
from cortex.LLM import get_random_error_message
from cortex.message import (DeveloperMessage, FunctionCall, Message, SystemMessage,
                                ToolMessage, ToolMessageGroup, UserMessage, AgentUsage)
from cortex.tool import BaseTool, FunctionTool, Tool

# Re-export Tool for backward compatibility (old imports from agent)
__all__ = ['Agent', 'Tool']

logger = logging.getLogger(__name__)

MAX_RECENT_CALLS = 5  # Only track the last 5 calls
DEFAULT_FALLBACK_MESSAGE = 'Sorry, I am not sure how to answer that.'

START_DELIM = '-' * 80
END_DELIM = '^' * 80

class Agent:
    def __init__(self, llm, tools=None, sys_prompt='', memory=None, context=None, json_reply=False, 
                 name=None, tool_call_limit=10, save_error_to_memory=False, mode='async', 
                 enable_parallel_tools=True, max_parallel_tools=None):
        '''
        Initialize the Agent.
        
        Args:
            mode: 'sync' or 'async' - determines which ask method to use and validates tools accordingly.
                  'sync' mode requires all tools to be sync functions (use with ask()).
                  'async' mode requires all tools to be async functions (use with async_ask()).
                  Default is 'async' as most users work with async LLM calls.
            enable_parallel_tools: If True, run multiple tool calls in parallel/concurrently.
                  Default is True for better performance.
            max_parallel_tools: Maximum number of tools to run in parallel. None means unlimited.
                  Only applies when enable_parallel_tools=True.
        '''
        if mode not in ('sync', 'async'):
            raise ValueError(f"mode must be 'sync' or 'async', got '{mode}'")
        
        self.mode = mode
        self.llm = llm
        self.tools = tools or []
        
        # Process and validate all tools
        self.tools_dict = self._process_and_validate_tools()
        
        self.sys_msg = sys_prompt if isinstance(sys_prompt, SystemMessage) else SystemMessage(content=sys_prompt)
        self.memory = memory
        self.context = context
        self.json_reply = json_reply
        self.name = name
        self.tool_call_limit = tool_call_limit
        self.save_error_to_memory = save_error_to_memory
        self.enable_parallel_tools = enable_parallel_tools
        self.max_parallel_tools = max_parallel_tools
        # Track recent tool calls to detect repetition
        self._recent_tool_calls = []
    
    def _process_and_validate_tools(self) -> dict:
        '''Process and validate all tools, returning a dictionary for O(1) lookup
        
        This method:
        1. Validates all tools have names
        2. Validates FunctionTools match the agent mode (sync/async)
        3. Builds a dictionary for O(1) tool lookup
        
        Returns:
            dict: Dictionary mapping tool names to tool objects
        '''
        tools_dict = {}

        is_async = self.mode == 'async'
        
        for tool in self.tools:
            # Validate tool has a name
            tool_name = getattr(tool, 'name', None)
            if not tool_name:
                raise ValueError(f"Tool {tool} must have a 'name' attribute")
            
            # Validate FunctionTools match the agent mode
            if isinstance(tool, FunctionTool):
                if not is_async and tool.is_async:
                    raise TypeError(
                        f"Tool '{tool.name}' is async but agent mode is 'sync'. "
                        f"Either set mode='async' or convert the tool to a sync function."
                    )
                elif is_async and tool.is_sync:
                    raise TypeError(
                        f"Tool '{tool.name}' is sync but agent mode is 'async'. "
                        f"Either set mode='sync' or convert the tool to an async function."
                    )
            
            # Add to dictionary
            tools_dict[tool_name] = tool
        
        return tools_dict
    
    def _find_tool(self, tool_name: str) -> Optional[BaseTool]:
        '''Find a tool by name using O(1) dictionary lookup'''
        return self.tools_dict.get(tool_name)
    
    def _remove_tool(self, tool: BaseTool) -> None:
        '''Safely remove a tool from both the list and dictionary'''
        # Remove from list
        if tool in self.tools:
            self.tools.remove(tool)
        # Remove from dictionary if it exists
        if self.tools_dict and hasattr(tool, 'name'):
            self.tools_dict.pop(tool.name, None)

    def _serialize_arguments(self, arguments) -> str:
        '''Serialize arguments to a consistent string format for comparison'''
        if isinstance(arguments, dict):
            return json.dumps(arguments, sort_keys=True)
        return str(arguments)

    def _is_repeated_tool_call(self, func: FunctionCall) -> bool:
        '''Check if this exact tool call was made recently'''
        args_str = self._serialize_arguments(func.arguments)
        current_call = (func.name, args_str)
        return current_call in self._recent_tool_calls

    def _add_tool_call(self, func: FunctionCall) -> None:
        '''Add a tool call to the recent calls list'''
        args_str = self._serialize_arguments(func.arguments)
        current_call = (func.name, args_str)
        self._recent_tool_calls.append(current_call)
        # Keep only the most recent calls
        self._recent_tool_calls = self._recent_tool_calls[-MAX_RECENT_CALLS:]

    def _prepare_conversation(self, message, user_name, history_msgs) -> List[Message]:
        '''Prepare the conversation with the user message'''
        # Convert string to UserMessage
        if isinstance(message, str):
            message = UserMessage(content=message, user_name=user_name)
        
        # Convert to list if needed
        conversation = message if isinstance(message, list) else [message]
        
        # Log conversation
        self._log_conversation_start(message, history_msgs)
        
        return conversation
    
    def _log_conversation_start(self, message, history_msgs) -> None:
        '''Log the start of a conversation'''
        # Print agent name if available
        if self.name:
            logger.info(f"[bold cyan]Agent: {self.name}[/bold cyan]")

        logger.debug(self.sys_msg.decorate())
        logger.info(START_DELIM)

        # Log history when showing system prompt
        for msg in history_msgs:
            logger.debug(msg.decorate())

        # Log current message(s)
        messages = message if isinstance(message, list) else [message]
        for msg in messages:
            logger.info(msg.decorate())
    
    def _handle_response(self, conversation, agent_usage, usage, is_error=False) -> None:
        '''Handle the response after the conversation is complete'''
        if self.memory and (not is_error or self.save_error_to_memory):
            self.memory.add_messages(conversation)
        
        logger.info(agent_usage.format())
        
        if usage:
            usage.merge(agent_usage)
        
        logger.info(END_DELIM)
            
    async def _async_handle_response(self, conversation, agent_usage, usage, is_error=False) -> None:
        '''Handle the response after the conversation is complete (async version)'''
        if self.memory and (not is_error or self.save_error_to_memory):
            await self.memory.add_messages(conversation)
        
        logger.info(agent_usage.format())
        
        if usage:
            usage.merge(agent_usage)
        
        logger.info(END_DELIM)

    def ask(self, message: str | Message | List[Message], user_name=None, usage=None, loop_limit=10):
        '''Ask a question to the agent, and get a response

        Args:
            message (str or Message or List[Message]): The message to ask
            user_name (str, optional): The name of the user. Defaults to None.
            usage (AgentUsage, optional): Object to accumulate token usage across models.
                You can pass an AgentUsage object to track usage across multiple calls.
            loop_limit (int, optional): The maximum number of times to call the model.
                Defaults to 10.

        Returns:
            str: The response from the agent
        '''
        if self.mode != 'sync':
            raise RuntimeError(
                f"Agent mode is '{self.mode}' but ask() is for sync mode. "
                f"Use async_ask() instead or set mode='sync' at initialization."
            )
        
        reply = None
        agent_usage = AgentUsage()  # Track total usage across all calls
        
        # Get history messages from memory
        history_msgs = self.memory.load_memory() if self.memory else []
        conversation = self._prepare_conversation(message, user_name, history_msgs)

        is_error = False

        # Main conversation loop
        for _ in range(loop_limit):
            msgs = [*history_msgs, *conversation]
            
            # call the model
            try:
                ai_msg = self.llm.call(self.sys_msg, msgs, tools=self.tools)
                # Add usage to AgentUsage if available
                if ai_msg.usage and ai_msg.model:
                    agent_usage.add_usage(ai_msg.model, ai_msg.usage)
            except Exception as e:
                logger.error('error calling LLM model: %s', e)

                err_msg = get_random_error_message()
                reply = {'message': err_msg} if self.json_reply else err_msg
                is_error = True
                break

            reply = self._process_ai_message(ai_msg, conversation)
            if reply is not None:
                self._handle_response(conversation, agent_usage, usage, is_error)
                return reply

        self._handle_response(conversation, agent_usage, usage, is_error)
        return reply if reply is not None else DEFAULT_FALLBACK_MESSAGE

    async def async_ask(self, message: str | Message | List[Message], user_name=None, usage=None, loop_limit=10):
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
        if self.mode != 'async':
            raise RuntimeError(
                f"Agent mode is '{self.mode}' but async_ask() is for async mode. "
                f"Use ask() instead or set mode='async' at initialization."
            )
        
        reply = None
        agent_usage = AgentUsage()  # Track total usage across all calls
        
        # Get history messages from memory asynchronously
        history_msgs = await self.memory.load_memory() if self.memory else []
        conversation = self._prepare_conversation(message, user_name, history_msgs)
        
        is_error = False
        # Main conversation loop
        for _ in range(loop_limit):
            msgs = [*history_msgs, *conversation]
            
            # call the model asynchronously
            try:
                ai_msg = await self.llm.async_call(self.sys_msg, msgs, tools=self.tools)
                # Add usage to AgentUsage if available
                if ai_msg.usage and ai_msg.model:
                    agent_usage.add_usage(ai_msg.model, ai_msg.usage)
            except Exception as e:
                logger.error('error calling LLM model: %s', e)

                err_msg = get_random_error_message()
                reply = {'message': err_msg} if self.json_reply else err_msg
                is_error = True
                break

            # Process the AI message
            reply = await self._async_process_ai_message(ai_msg, conversation)
            if reply is not None:
                await self._async_handle_response(conversation, agent_usage, usage, is_error)
                return reply

        await self._async_handle_response(conversation, agent_usage, usage, is_error)
        return reply if reply is not None else DEFAULT_FALLBACK_MESSAGE
    
    def _parse_json_reply(self, content: str):
        '''Parse JSON reply if json_reply mode is enabled'''
        if self.json_reply:
            return json.loads(content)
        return content
    
    def _process_ai_message(self, ai_msg, conversation):
        '''Process the AI message and determine next steps (sync mode)'''
        conversation.append(ai_msg)

        # Check if we need to run a tool
        if ai_msg.function_calls:
            logger.info('function calls: %s', ai_msg.function_calls)
            tool_msgs = self.process_func_call(ai_msg)
            conversation.append(tool_msgs)
            return None  # Continue the conversation
        elif ai_msg.content:
            logger.info(ai_msg.decorate())
            try:
                return self._parse_json_reply(ai_msg.content)
            except json.JSONDecodeError as e:
                err_msg = f"Error processing JSON message: {e}. Make sure your response is a valid JSON string and do not include the `json` tag."
                conversation.append(DeveloperMessage(content=err_msg))
                return None  # Continue the conversation with error message

        return None  # Continue the conversation
    
    async def _async_process_ai_message(self, ai_msg, conversation):
        '''Process the AI message and determine next steps (async mode)'''
        conversation.append(ai_msg)

        # Check if we need to run a tool
        if ai_msg.function_calls:
            logger.info('function calls: %s', ai_msg.function_calls)
            tool_msgs = await self.async_process_func_call(ai_msg)
            conversation.append(tool_msgs)
            return None  # Continue the conversation
        elif ai_msg.content:
            logger.info(ai_msg.decorate())
            try:
                return self._parse_json_reply(ai_msg.content)
            except json.JSONDecodeError as e:
                err_msg = f"Error processing JSON message: {e}. Make sure your response is a valid JSON string and do not include the `json` tag."
                conversation.append(DeveloperMessage(content=err_msg))
                return None  # Continue the conversation with error message

        return None  # Continue the conversation

    def _get_tool_call_id(self, func_call: FunctionCall) -> str:
        '''Get the tool call ID, with fallback'''
        return func_call.call_id or func_call.id
    
    def _process_single_tool_call(self, func_call: FunctionCall) -> ToolMessage:
        '''Process a single tool call (sync mode) - extracted for reuse in parallel execution'''
        logger.info(f'[bold purple]Tool: {func_call.name}[/bold purple]')
        
        # Check if this is a repeated tool call
        if self._is_repeated_tool_call(func_call):
            msg = f'Tool "{func_call.name}" was just called with the same arguments again. To prevent loops, please try a different approach or different arguments.'
            return ToolMessage(content=msg, tool_call_id=self._get_tool_call_id(func_call))
        
        func_result = self.run_tool_func(func_call)
        tool_res_msg = ToolMessage(content=func_result, tool_call_id=self._get_tool_call_id(func_call))
        
        logger.info(tool_res_msg.decorate())
        
        # Track this tool call
        self._add_tool_call(func_call)
        
        return tool_res_msg
    
    async def _async_process_single_tool_call(self, func_call: FunctionCall) -> ToolMessage:
        '''Process a single tool call (async mode) - extracted for reuse in parallel execution'''
        logger.info(f'[bold purple]Tool: {func_call.name}[/bold purple]')
        
        # Check if this is a repeated tool call
        if self._is_repeated_tool_call(func_call):
            msg = f'Tool "{func_call.name}" was just called with the same arguments again. To prevent loops, please try a different approach or different arguments.'
            return ToolMessage(content=msg, tool_call_id=self._get_tool_call_id(func_call))
        
        func_result = await self.async_run_tool_func(func_call)
        tool_res_msg = ToolMessage(content=func_result, tool_call_id=self._get_tool_call_id(func_call))
        
        logger.info(tool_res_msg.decorate())
        
        # Track this tool call
        self._add_tool_call(func_call)
        
        return tool_res_msg
    
    def process_func_call(self, ai_msg):
        '''Process function calls in the LLM result (sync mode)'''
        func_calls = ai_msg.function_calls
        
        # Use parallel execution if enabled and multiple tools
        if self.enable_parallel_tools and len(func_calls) > 1:
            messages = self._process_func_calls_parallel(func_calls)
        else:
            # Sequential execution
            messages = []
            for func_call in func_calls:
                tool_msg = self._process_single_tool_call(func_call)
                messages.append(tool_msg)
        
        return ToolMessageGroup(tool_messages=messages)
    
    def _process_func_calls_parallel(self, func_calls: List[FunctionCall]) -> List[ToolMessage]:
        '''Process multiple function calls in parallel using ThreadPoolExecutor (sync mode)'''
        max_workers = self.max_parallel_tools or len(func_calls)
        
        if self.max_parallel_tools:
            logger.info(f'Running {len(func_calls)} tools in parallel (max_workers={max_workers})')
        else:
            logger.info(f'Running {len(func_calls)} tools in parallel (unlimited)')
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks and maintain order
            messages = list(executor.map(self._process_single_tool_call, func_calls))
        
        return messages
    
    async def async_process_func_call(self, ai_msg):
        '''Process function calls in the LLM result (async mode)'''
        func_calls = ai_msg.function_calls
        
        # Use concurrent execution if enabled and multiple tools
        if self.enable_parallel_tools and len(func_calls) > 1:
            messages = await self._async_process_func_calls_concurrent(func_calls)
        else:
            # Sequential execution
            messages = []
            for func_call in func_calls:
                tool_msg = await self._async_process_single_tool_call(func_call)
                messages.append(tool_msg)
        
        return ToolMessageGroup(tool_messages=messages)
    
    async def _async_process_func_calls_concurrent(self, func_calls: List[FunctionCall]) -> List[ToolMessage]:
        '''Process multiple function calls concurrently using asyncio.gather (async mode)'''
        if self.max_parallel_tools:
            logger.info(f'Running {len(func_calls)} tools concurrently (max={self.max_parallel_tools})')
            # Use semaphore to limit concurrency
            semaphore = asyncio.Semaphore(self.max_parallel_tools)
            
            async def limited_call(func_call):
                async with semaphore:
                    return await self._async_process_single_tool_call(func_call)
            
            tasks = [limited_call(fc) for fc in func_calls]
        else:
            logger.info(f'Running {len(func_calls)} tools concurrently (unlimited)')
            # No limit on concurrency
            tasks = [self._async_process_single_tool_call(fc) for fc in func_calls]
        
        # gather returns results in order
        messages = await asyncio.gather(*tasks)
        
        return list(messages)

    def _parse_tool_arguments(self, arguments) -> dict:
        '''Parse tool arguments from string or dict'''
        if isinstance(arguments, str):
            return json.loads(arguments)
        return arguments
    
    def _format_tool_result(self, result) -> str:
        '''Format tool result as string'''
        if isinstance(result, str):
            return result
        return json.dumps(result)
    
    def _validate_and_get_tool(self, tool_name: str) -> Tuple[Optional[FunctionTool], Optional[str]]:
        '''Validate and get tool, returning (tool, error_message)
        
        Returns:
            tuple: (tool, error_msg) where tool is None if error, error_msg is None if success
        '''
        tool = self._find_tool(tool_name)
        if tool is None:
            return None, f'No tool named "{tool_name}" found. Do not call it again.'
        
        # Only FunctionTool can run locally
        if not isinstance(tool, FunctionTool):
            return None, f'Tool "{tool_name}" is not a function tool and cannot be executed locally. Do not call it directly.'
        
        if not tool.check_call_limit(self.tool_call_limit):
            self._remove_tool(tool)
            return None, f'Tool "{tool_name}" has been called too many times and has been removed from available tools.'
        
        return tool, None
    
    def run_tool_func(self, func: FunctionCall) -> str:
        '''Run the given tool function and return the result (sync mode)'''
        tool_name = func.name
        
        tool, error_msg = self._validate_and_get_tool(tool_name)
        if error_msg:
            return error_msg
        
        try:
            tool_input = self._parse_tool_arguments(func.arguments)
            result = tool.run(tool_input, self.context, self)
            return self._format_tool_result(result)
        except json.JSONDecodeError as e:
            return f'Error decoding JSON parameter for "{tool_name}": {e}. Use valid JSON string without the `json` tag.'
        except Exception as e:
            if is_debug:
                import traceback
                traceback.print_exc()
            return f'Error running tool "{tool_name}": {e}'
    
    async def async_run_tool_func(self, func: FunctionCall) -> str:
        '''Run the given tool function and return the result (async mode)'''
        tool_name = func.name
        
        tool, error_msg = self._validate_and_get_tool(tool_name)
        if error_msg:
            return error_msg
        
        try:
            tool_input = self._parse_tool_arguments(func.arguments)
            result = await tool.async_run(tool_input, self.context, self)
            return self._format_tool_result(result)
        except json.JSONDecodeError as e:
            return f'Error decoding JSON parameter for "{tool_name}": {e}. Use valid JSON string without the `json` tag.'
        except Exception as e:
            if is_debug:
                import traceback
                traceback.print_exc()
            return f'Error running tool "{tool_name}": {e}'