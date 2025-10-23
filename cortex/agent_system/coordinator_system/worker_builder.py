from asyncio import iscoroutine
from typing import Callable, Optional
import re
from cortex import LLM, Agent, Tool
from cortex.message import DeveloperMessage, UserMessage

from ..core.context import AgentSystemContext
from ..core.builder import AgentBuilder

# A generic worker agent prompt for a worker agent collaborating within a multi-agent team.
# It intentionally avoids any product- or organization-specific wording.
WORKER_PROMPT = """
You are {agent_role}, a specialized worker agent operating under {coordinator_name},
the coordinator of your team.

[CORE RESPONSIBILITIES]
1. Understand the user’s intent and task context as relayed by {coordinator_name}.
2. Decide the correct action path:
   (a) Call tools or functions to perform specific operations or retrieve data.
   (b) Generate a message if a human-readable reply or update is required.
3. When information is missing, ambiguous, or contradictory, compose a concise
   clarifying question addressed to the user, but route it through {coordinator_name}.

[COLLABORATION PROTOCOL]
- You do not communicate directly with the end user.
- All outgoing messages must follow this structured format:
  - "to_user": the message {coordinator_name} should forward to the user.
  - "{coordinator_key}": internal notes, clarifications, or handoffs back to {coordinator_name}.
- Keep internal messages factual, brief, and actionable — avoid redundant summaries.

[REASONING AND EXECUTION]
- Before acting, verify that you have all required inputs to proceed.
- When multiple independent operations are needed, call all relevant tools simultaneously
  rather than sequentially — the system executes them in parallel for efficiency.
- If a tool call fails:
  - Analyze the error once.
  - Retry only if a meaningful correction or alternative input is available.
  - Otherwise, report the issue via "{coordinator_key}" with a clear diagnostic explanation.

[QUALITY AND SAFETY]
- Base every conclusion and response on verified data, tool results, or explicit context.
- Do not fabricate IDs, data, or assumptions.
- Avoid speculation; if uncertain, escalate the uncertainty through "{coordinator_key}".
- Be concise, accurate, and transparent in both reasoning and results.

[TASK CONTEXT AND OBJECTIVES]
{task_desc}

[BEHAVIORAL SUMMARY]
You are a dependable domain specialist who:
- Works independently while communicating clearly through {coordinator_name}.
- Uses tools effectively and reports outcomes succinctly.
- Produces clear, factual, and user-ready information suitable for delivery through {coordinator_name}.
"""

# Response formatting definitions.
# The coordinator field key ("{coordinator_key}") is configurable to maintain compatibility
# with existing systems. The default is "to_coordinator", but callers can override it.

NORMAL_FORMAT = """
[OUTPUT FORMAT — STANDARD MODE]
When you are NOT calling a tool, output a single valid JSON object.
Do not include code fences, Markdown, or extra text.

Allowed top-level keys:
  - "to_user": the message intended for the end user.
      • Keep it clear, actionable, and concise.
      • Avoid technical jargon unless explicitly requested.
  - "{coordinator_key}": a short note for {coordinator_name}.
      • Summarize decisions made, blockers, or the next step.

Rules:
- Only include the keys listed above; no additional fields.
- The entire response must be directly parsable by `json.loads()`.
- Keep "{coordinator_key}" short and factual (1–3 sentences).
"""

THOUGHT_FORMAT = """
[OUTPUT FORMAT — WITH INTERNAL THOUGHT]
When you are NOT calling a tool, output a single valid JSON object.
Do not include code fences, Markdown, or extra text.

Allowed top-level keys:
  - "thought": your brief internal reasoning note.
      • Use it to outline logic or next actions concisely.
      • Do not expose private system logic or sensitive data.
  - "to_user": the message intended for the end user.
      • Keep it clear, actionable, and relevant.
  - "{coordinator_key}": a short coordination note for {coordinator_name}.
      • State what was done, what’s needed next, or why the task paused.

Rules:
- Only include the keys listed above; no additional fields.
- The response must be valid JSON (parsable by `json.loads()`).
- Keep both "thought" and "{coordinator_key}" brief (1–3 sentences each).
"""


class WorkerAgentBuilder(AgentBuilder):
    """Builder for a generic worker agent that collaborates with a coordinator.

    This builder defers agent construction until invoked, allowing the coordinating system
    to inject runtime details (e.g., coordinator identity, memory, and tools) via context.
    It uses the generic worker prompts and is independent of any product or organization.

    Args:
        name: worker agent name; also used to derive the default Tool name ("{name}_agent").
        llm: LLM instance used by the worker agent.
        prompt_builder: Callable to build the prompt for the worker agent.
        tools_builder: Optional callable to load tools for the worker agent.
        memory_k: Optional memory object to attach to the worker agent.
        thinking: If True, use the thought-enabled response format block.
        introduction: introduction for the worker agent to the coordinator.
        before_agent: Optional callable to run before the agent is built.
    """

    def __init__(
        self,
        *,
        name: str,
        llm: LLM,
        prompt_builder: Callable,
        tools_builder: Optional[Callable] = None,
        memory_k: Optional[int] = 5,
        thinking: bool = True,
        introduction: Optional[str] = None,
        before_agent: Optional[Callable] = None,
    ):
        super().__init__(
            name=name,
            llm=llm,
            prompt_builder=prompt_builder,
            tools_builder=tools_builder,
            memory_k=memory_k,
        )
        # Agent-side settings
        self.thinking = thinking
        # Tool exposure settings
        self.introduction = introduction
        self.before_agent = before_agent
        # Define parameters schema for the installed Tool
        self.parameters = {
            "type": "object",
            "properties": {
                "user_input": {
                    "type": "string",
                    "description": "User's message, verbatim",
                },
                "context_instructions": {
                    "type": ["string", "null"],
                    "description": "Optional extra context instructions for the agent",
                },
            },
            "required": ["user_input", "context_instructions"],
            "additionalProperties": False,
        }
    
    @classmethod
    def compose_prompt(cls, agent_name, task_desc, coordinator_name, coordinator_key, thinking=True):
        """
        Compose the prompt for the worker agent.
        This is a class method to allow for easy reuse of the prompt composition logic.
        Users might just want to use the prompt instead of the builder.
        """
        format_block = THOUGHT_FORMAT if thinking else NORMAL_FORMAT
        prompt_parts = [WORKER_PROMPT, format_block]
        
        prompt = "".join(prompt_parts)
        
        fmt_kwargs = {
            "agent_role": agent_name,
            "task_desc": task_desc,
            "coordinator_name": coordinator_name,
            "coordinator_key": coordinator_key,
        }
        
        return prompt.format(**fmt_kwargs)
    
    async def build_agent(self, *, context: AgentSystemContext, coordinator_name: Optional[str] = None) -> Agent:
        # derive coordinator display and key
        display_name = coordinator_name or "the coordinator"
        if coordinator_name:
            base = re.sub(r"\W+", "_", coordinator_name.strip().lower()).strip("_")
            coordinator_key = f"to_{base}" if base else "to_coordinator"
        else:
            coordinator_key = "to_coordinator"
        
        task_desc = await self.build_prompt(context)

        sys_prompt = self.compose_prompt(self.name, task_desc, display_name, coordinator_key, self.thinking)

        bank = await context.get_memory_bank()
        memory = await bank.get_agent_memory(self.name_key, k=self.memory_k)
        
        tools = await self.load_tools(context)

        return Agent(
            name=self.name,
            llm=self.llm,
            tools=tools,
            sys_prompt=sys_prompt,
            memory=memory,
            context=context,
            json_reply=True,
        )

    def install(
        self,
        *,
        coordinator_name: Optional[str] = None,
    ) -> Tool:
        """Install this worker as a Tool.

        Args:
            coordinator_name: Optional coordinator name. If provided, it is embedded
                directly into the worker's system prompt at build time.

        Returns:
            Tool: Named "{self.name}_agent" with parameters requiring
            "user_input" and "context_instructions".
        """
        async def func(args, context: AgentSystemContext):
            if self.before_agent:
                res = self.before_agent(context)
                if iscoroutine(res):
                    res = await res
                
                if res is None or res is True:
                    pass
                else:
                    return res
            
            agent = await self.build_agent(context=context, coordinator_name=coordinator_name)
            
            user_input = args["user_input"]
            ctx_instructions = args.get("context_instructions")

            msgs = [UserMessage(content=user_input)]
            if ctx_instructions:
                msgs.append(DeveloperMessage(content=ctx_instructions))

            return await agent.async_ask(msgs, usage=getattr(context, "usage", None))

        tool_name = self.name_key + "_agent"

        return Tool(
            name=tool_name,
            func=func,
            description=self.introduction,
            parameters=self.parameters,
        )
