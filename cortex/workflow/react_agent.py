"""ReAct Agent built on WorkflowAgent.

A ReActAgent behaves like the original Agent - it accepts a system prompt, tools,
and runs in a loop to reason, choose tools, execute them, and give a final response.
"""

import asyncio
import json
import logging
from typing import List, Optional

from cortex.LLM import get_random_error_message
from cortex.message import FunctionCall, SystemMessage, ToolMessage, ToolMessageGroup, UserMessage, AgentUsage
from cortex.tool import BaseTool, FunctionTool
from cortex.workflow import WorkflowAgent

logger = logging.getLogger(__name__)

DEFAULT_FALLBACK_MESSAGE = "Sorry, I am not sure how to answer that."
MAX_RECENT_CALLS = 5


def _serialize_arguments(arguments) -> str:
    if isinstance(arguments, dict):
        return json.dumps(arguments, sort_keys=True)
    return str(arguments)


class ReActAgent:
    """A ReAct-style agent built on WorkflowAgent.
    
    It runs a reasoning loop:
    1. Think - reason about the current state
    2. Act - call a tool or respond
    3. Observe - get tool results and continue
    """

    def __init__(
        self,
        llm,
        tools=None,
        sys_prompt="",
        name=None,
        context=None,
        memory=None,
        json_reply=False,
        tool_call_limit=10,
        loop_limit=10,
        enable_parallel_tools=True,
        max_parallel_tools=None,
    ):
        self.llm = llm
        self.tools = tools or []
        self.name = name
        self.context = context
        self.memory = memory
        self.json_reply = json_reply
        self.tool_call_limit = tool_call_limit
        self.loop_limit = loop_limit
        self.enable_parallel_tools = enable_parallel_tools
        self.max_parallel_tools = max_parallel_tools

        self.sys_msg = (
            sys_prompt if isinstance(sys_prompt, SystemMessage) else SystemMessage(content=sys_prompt)
        )

        self.tools_dict = self._process_and_validate_tools()
        self._recent_tool_calls: List[tuple] = []
        self._wf: Optional[WorkflowAgent] = None

    def _process_and_validate_tools(self) -> dict:
        tools_dict = {}
        for tool in self.tools:
            tool_name = getattr(tool, "name", None)
            if not tool_name:
                raise ValueError(f"Tool {tool} must have a 'name' attribute")
            tools_dict[tool_name] = tool
        return tools_dict

    def _find_tool(self, tool_name: str) -> Optional[BaseTool]:
        return self.tools_dict.get(tool_name)

    def _is_repeated_tool_call(self, func: FunctionCall) -> bool:
        args_str = _serialize_arguments(func.arguments)
        current_call = (func.name, args_str)
        return current_call in self._recent_tool_calls

    def _add_tool_call(self, func: FunctionCall) -> None:
        args_str = _serialize_arguments(func.arguments)
        current_call = (func.name, args_str)
        self._recent_tool_calls.append(current_call)
        self._recent_tool_calls = self._recent_tool_calls[-MAX_RECENT_CALLS:]

    def _build_workflow(self) -> WorkflowAgent:
        """Build the internal workflow that implements the ReAct loop."""
        wf = WorkflowAgent(
            name=self.name or "ReActAgent",
            context=self.context,
            memory=self.memory,
        )

        def prepare_fn(data, context):
            """Initialize conversation state."""
            user_input = data.get("input", "")
            history = data.get("_history", [])
            return {
                "_conversation": history + [UserMessage(content=user_input)],
                "_iteration": 0,
                "_loop_limit": self.loop_limit,
            }

        async def think_fn(data, context):
            """Reason about the current state and decide next action."""
            conversation = data.get("_conversation", [])
            sys_msg = self.sys_msg

            try:
                ai_msg = await self.llm.async_call(sys_msg, conversation, tools=self.tools)
            except Exception as e:
                logger.error("error calling LLM model: %s", e)
                return {
                    "_final_answer": get_random_error_message(),
                    "_next_node": "respond",
                }

            if ai_msg.usage and ai_msg.model:
                data["_usage"] = ai_msg.usage
                data["_model"] = ai_msg.model

            conversation.append(ai_msg)

            if ai_msg.function_calls:
                return {
                    "_conversation": conversation,
                    "_ai_msg": {
                        "content": ai_msg.content,
                        "function_calls": [
                            {"name": fc.name, "arguments": fc.arguments, "id": fc.call_id or fc.id}
                            for fc in ai_msg.function_calls
                        ],
                    },
                    "_action": "act",
                    "_next_node": "check_loop",
                }
            elif ai_msg.content:
                return {
                    "_conversation": conversation,
                    "_ai_msg": {"content": ai_msg.content},
                    "_action": "respond",
                    "_next_node": "check_loop",
                }

            return {
                "_conversation": conversation,
                "_action": "continue",
                "_next_node": "check_loop",
            }

        def act_fn(data, context):
            """Execute tool calls from the model."""
            ai_msg_data = data.get("_ai_msg", {})
            func_calls_data = ai_msg_data.get("function_calls", [])

            if not func_calls_data:
                return {"_action": "think", "_next_node": "think"}

            func_calls = [
                FunctionCall(
                    name=fc["name"],
                    arguments=fc["arguments"],
                    call_id=fc.get("id"),
                )
                for fc in func_calls_data
            ]

            conversation = data.get("_conversation", [])

            if self.enable_parallel_tools and len(func_calls) > 1:
                tool_messages = asyncio.run(
                    self._run_tools_concurrent(func_calls)
                )
            else:
                tool_messages = asyncio.run(
                    self._run_tools_sequential(func_calls)
                )

            conversation.append(ToolMessageGroup(tool_messages=tool_messages))

            iteration = data.get("_iteration", 0)
            return {
                "_conversation": conversation,
                "_iteration": iteration + 1,
                "_next_node": "think",
            }

        def respond_fn(data, context):
            """Return final response when no tools needed."""
            ai_msg_data = data.get("_ai_msg", {})
            content = ai_msg_data.get("content", "")

            if self.json_reply:
                try:
                    content = json.loads(content)
                except json.JSONDecodeError as e:
                    content = f"Error processing JSON message: {e}"

            return {"_final_answer": content, "_output": content}

        def check_loop_fn(data, context):
            """Check if we've exceeded loop limit."""
            iteration = data.get("_iteration", 0)
            loop_limit = data.get("_loop_limit", self.loop_limit)
            action = data.get("_action", "think")

            if iteration >= loop_limit:
                return {"_next_node": "respond", "_final_answer": DEFAULT_FALLBACK_MESSAGE, "_output": DEFAULT_FALLBACK_MESSAGE}
            
            if action == "respond":
                return {"_next_node": "respond"}
            if action == "act":
                return {"_next_node": "act"}
            return {"_next_node": "think"}

        wf.add_node("prepare", prepare_fn, start=True)
        wf.add_node("think", think_fn)
        wf.add_node("act", act_fn)
        wf.add_node("respond", respond_fn, is_final=True)
        wf.add_node("check_loop", check_loop_fn)

        wf.add_edge("prepare", "think")
        wf.add_edge("think", "check_loop")
        wf.add_edge("check_loop", "respond")
        wf.add_edge("check_loop", "act")
        wf.add_edge("act", "think")

        return wf

    def _run_tools_sequential(self, func_calls: List[FunctionCall]) -> List[ToolMessage]:
        """Run tools sequentially (sync)."""
        messages = []
        for func_call in func_calls:
            msg = self._process_single_tool_call(func_call)
            messages.append(msg)
        return messages

    def _run_tools_concurrent(self, func_calls: List[FunctionCall]) -> List[ToolMessage]:
        """Run tools concurrently (sync wrapper)."""
        return asyncio.run(self._run_tools_async_concurrent(func_calls))

    async def _run_tools_async_concurrent(
        self, func_calls: List[FunctionCall]
    ) -> List[ToolMessage]:
        """Run tools concurrently (async)."""
        if self.max_parallel_tools:
            semaphore = asyncio.Semaphore(self.max_parallel_tools)

            async def limited_call(func_call):
                async with semaphore:
                    return await self._async_process_single_tool_call(func_call)

            tasks = [limited_call(fc) for fc in func_calls]
        else:
            tasks = [self._async_process_single_tool_call(fc) for fc in func_calls]

        return await asyncio.gather(*tasks)

    def _process_single_tool_call(self, func_call: FunctionCall) -> ToolMessage:
        """Process a single tool call (sync)."""
        logger.info(f"[bold purple]Tool: {func_call.name}[/bold purple]")

        if self._is_repeated_tool_call(func_call):
            msg = f'Tool "{func_call.name}" was just called with the same arguments again. To prevent loops, please try a different approach or different arguments.'
            return ToolMessage(content=msg, tool_call_id=func_call.call_id or func_call.id)

        func_result = self._run_tool_func(func_call)
        tool_msg = ToolMessage(content=func_result, tool_call_id=func_call.call_id or func_call.id)

        logger.info(tool_msg.decorate())
        self._add_tool_call(func_call)

        return tool_msg

    async def _async_process_single_tool_call(self, func_call: FunctionCall) -> ToolMessage:
        """Process a single tool call (async)."""
        logger.info(f"[bold purple]Tool: {func_call.name}[/bold purple]")

        if self._is_repeated_tool_call(func_call):
            msg = f'Tool "{func_call.name}" was just called with the same arguments again. To prevent loops, please try a different approach or different arguments.'
            return ToolMessage(content=msg, tool_call_id=func_call.call_id or func_call.id)

        func_result = await self._async_run_tool_func(func_call)
        tool_msg = ToolMessage(content=func_result, tool_call_id=func_call.call_id or func_call.id)

        logger.info(tool_msg.decorate())
        self._add_tool_call(func_call)

        return tool_msg

    def _parse_tool_arguments(self, arguments) -> dict:
        if isinstance(arguments, str):
            return json.loads(arguments)
        return arguments

    def _format_tool_result(self, result) -> str:
        if isinstance(result, str):
            return result
        return json.dumps(result)

    def _run_tool_func(self, func: FunctionCall) -> str:
        """Run tool synchronously."""
        tool = self._find_tool(func.name)
        if tool is None:
            return f'No tool named "{func.name}" found. Do not call it again.'
        if not isinstance(tool, FunctionTool):
            return f'Tool "{func.name}" is not a function tool and cannot be executed locally.'

        try:
            tool_input = self._parse_tool_arguments(func.arguments)
            result = tool.run(tool_input, self.context, None)
            return self._format_tool_result(result)
        except json.JSONDecodeError as e:
            return f'Error decoding JSON parameter for "{func.name}": {e}.'
        except Exception as e:
            return f'Error running tool "{func.name}": {e}'

    async def _async_run_tool_func(self, func: FunctionCall) -> str:
        """Run tool asynchronously."""
        tool = self._find_tool(func.name)
        if tool is None:
            return f'No tool named "{func.name}" found. Do not call it again.'
        if not isinstance(tool, FunctionTool):
            return f'Tool "{func.name}" is not a function tool and cannot be executed locally.'

        try:
            tool_input = self._parse_tool_arguments(func.arguments)
            result = await tool.async_run(tool_input, self.context, None)
            return self._format_tool_result(result)
        except json.JSONDecodeError as e:
            return f'Error decoding JSON parameter for "{func.name}": {e}.'
        except Exception as e:
            return f'Error running tool "{func.name}": {e}'

    def _ensure_workflow(self) -> WorkflowAgent:
        """Ensure the internal workflow is built."""
        if self._wf is None:
            self._wf = self._build_workflow()
            self._wf.build()
        return self._wf

    async def async_ask(self, message, usage=None, loop_limit=None):
        """Ask the agent a question asynchronously.
        
        Args:
            message: The message to ask
            usage: Optional AgentUsage to accumulate token usage
            loop_limit: Override default loop limit
        
        Returns:
            str: The response from the agent
        """
        wf = self._ensure_workflow()

        run = await wf.async_run(message)

        final = run.final_output
        if final is None:
            final = run.state.data.get("_final_answer", DEFAULT_FALLBACK_MESSAGE)

        if usage and run.state.data.get("_usage"):
            agent_usage = AgentUsage()
            agent_usage.add_usage(
                run.state.data.get("_model", "unknown"),
                run.state.data.get("_usage"),
            )
            usage.merge(agent_usage)

        return final

    def ask(self, message, usage=None, loop_limit=None):
        """Ask the agent a question synchronously.
        
        Args:
            message: The message to ask
            usage: Optional AgentUsage to accumulate token usage
            loop_limit: Override default loop limit
        
        Returns:
            str: The response from the agent
        """
        return asyncio.run(self.async_ask(message, usage, loop_limit))
