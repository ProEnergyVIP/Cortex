from asyncio import iscoroutine
from typing import Callable, Optional
import re
import json
from cortex import LLM, Agent, Tool
from cortex.message import DeveloperMessage, UserMessage

from ..core.context import AgentSystemContext
from ..core.whiteboard import WhiteboardUpdateType
from ..core.builder import AgentBuilder

# A generic worker agent prompt for a worker agent collaborating within a multi-agent team.
# It intentionally avoids any product- or organization-specific wording.
WORKER_PROMPT = """
You are {agent_role}, a specialized worker agent operating under {coordinator_name},
the coordinator of your team.

[CORE RESPONSIBILITIES]
1. Understand the REAL TASK DESCRIPTION section provided below. It defines your primary objective
   and the tasks you must perform.
2. Understand the user’s request and optional helpful context as relayed by {coordinator_name}.
3. Think carefully about the task and the user’s request, make a plan of actions.
   And perform the actions outlined in the plan, and answer the user at the end.
4. If information is missing, ambiguous, or contradictory, compose a concise
   clarifying question addressed to the user, but route it through {coordinator_name}.

[COLLABORATION PROTOCOL]
- You do not communicate directly with the user.
- Your final message must be a valid JSON object with the following keys:
  - "to_user": the message {coordinator_name} should forward to the user.
  - "{coordinator_key}": internal notes, clarifications, or handoffs back to {coordinator_name}.
- Keep internal messages factual, brief, and actionable — avoid redundant summaries.
- Do not include code fences, Markdown, or extra text. The final message must be
  able to be parsed by `json.loads()` in Python.

[REASONING AND EXECUTION]
- Before acting, verify that you have all required inputs to proceed.
- Analyze the task to identify if multiple independent operations are needed:
  • If multiple tools can work independently, call ALL of them simultaneously
    (the system executes them in parallel for efficiency).
  • If operations depend on each other, execute them sequentially.
  • Example: fetching data from multiple sources → call all fetch tools at once.
- If a tool call fails:
  - Analyze the error once.
  - Retry only if a meaningful correction or alternative input is available.
  - Otherwise, report the issue via "{coordinator_key}" with a clear diagnostic explanation.

[QUALITY AND SAFETY]
- Base every conclusion and response on verified data, tool results, or explicit context.
- Do not fabricate IDs, data, or assumptions.
- Avoid speculation; if uncertain, escalate the uncertainty through "{coordinator_key}".
- Be concise, accurate, and transparent in both reasoning and results.

[BEHAVIORAL SUMMARY]
You are a dependable domain specialist who:
- Works independently while communicating clearly through {coordinator_name}.
- Uses tools effectively and reports outcomes succinctly.
- Produces clear, factual, and user-ready information suitable for delivery through {coordinator_name}.

[REAL TASK DESCRIPTION]
{task_desc}
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

NORMAL_FORMAT_WITH_CONTEXT = """
[OUTPUT FORMAT — STANDARD MODE]
When you are NOT calling a tool, output a single valid JSON object.
Do not include code fences, Markdown, or extra text.

Allowed top-level keys:
  - "to_user": the message intended for the end user.
      • Keep it clear, actionable, and concise.
      • Avoid technical jargon unless explicitly requested.
  - "{coordinator_key}": a short note for {coordinator_name}.
      • Summarize decisions made, blockers, or the next step.
  - "whiteboard_suggestion": OPTIONAL structured suggestions for updating the whiteboard.
      • If present, it MUST be a JSON object with these optional keys:
          • "progress": string summary of overall progress.
          • "blockers_add": array of strings describing blockers to add.
          • "blockers_remove": array of strings describing blockers that are resolved.
          • "decisions": array of objects with keys "decision" (string) and optional "rationale" (string).
      • The coordinator decides whether and how to apply these suggestions.

Rules:
- Only include the keys listed above; no additional fields.
- The entire response must be directly parsable by `json.loads()`.
- Keep "{coordinator_key}" short and factual (1–3 sentences).
"""

THOUGHT_FORMAT_WITH_CONTEXT = """
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
  - "whiteboard_suggestion": OPTIONAL structured suggestions for updating the whiteboard.
      • If present, it MUST be a JSON object with these optional keys:
          • "progress": string summary of overall progress.
          • "blockers_add": array of strings describing blockers to add.
          • "blockers_remove": array of strings describing blockers that are resolved.
          • "decisions": array of objects with keys "decision" (string) and optional "rationale" (string).
      • The coordinator decides whether and how to apply these suggestions.

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
        enable_context_suggestions: bool = True,
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
        self.enable_context_suggestions = enable_context_suggestions
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
    def compose_prompt(cls, agent_name, task_desc, coordinator_name, coordinator_key, thinking=True, enable_context_suggestions: bool = True):
        """
        Compose the prompt for the worker agent.
        This is a class method to allow for easy reuse of the prompt composition logic.
        Users might just want to use the prompt instead of the builder.
        """
        if enable_context_suggestions:
            format_block = THOUGHT_FORMAT_WITH_CONTEXT if thinking else NORMAL_FORMAT_WITH_CONTEXT
        else:
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

        enable_suggestions = self.enable_context_suggestions and bool(getattr(context, "whiteboard", None))
        sys_prompt = self.compose_prompt(
            self.name,
            task_desc,
            display_name,
            coordinator_key,
            self.thinking,
            enable_suggestions,
        )

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
            
            # STEP 1: Get agent-specific view of the whiteboard before routing (if available)
            if getattr(context, "whiteboard", None):
                agent_view = await context.whiteboard.get_agent_view(self.name)
            else:
                agent_view = {}
            
            # STEP 2: Build context summary to include in message
            context_parts = []
            if agent_view.get("mission"):
                context_parts.append(f"Mission: {agent_view['mission']}")
            if agent_view.get("current_focus"):
                context_parts.append(f"Current Focus: {agent_view['current_focus']}")
            if agent_view.get("my_role"):
                context_parts.append(f"Your Role: {agent_view['my_role']}")
            if agent_view.get("active_blockers"):
                context_parts.append(f"Active Blockers: {', '.join(agent_view['active_blockers'])}")
            
            # Include recent updates from other agents
            if agent_view.get("recent_updates"):
                recent = agent_view["recent_updates"][:5]  # Last 5 updates
                if recent:
                    updates_summary = "\n".join([
                        f"  - [{u['type']}] {u['agent_name']}: {str(u['content'])[:100]}"
                        for u in recent
                    ])
                    context_parts.append(f"Recent Team Updates:\n{updates_summary}")
            
            # Combine with existing context_instructions
            whiteboard_info = "\n\n".join(context_parts) if context_parts else None

            if whiteboard_info:
                if ctx_instructions:
                    combined_context = f"{ctx_instructions}\n\n[Whiteboard]\n{whiteboard_info}"
                else:
                    combined_context = f"[Whiteboard]\n{whiteboard_info}"
            else:
                combined_context = ctx_instructions

            msgs = [UserMessage(content=user_input)]
            if combined_context:
                msgs.append(DeveloperMessage(content=combined_context))

            # STEP 3: Execute worker agent
            response = await agent.async_ask(msgs, usage=getattr(context, "usage", None))

            # STEP 3.5: Apply any whiteboard suggestions from the worker
            suggestion = None
            if isinstance(response, str):
                try:
                    parsed = json.loads(response)
                    if isinstance(parsed, dict):
                        suggestion = parsed.get("whiteboard_suggestion")
                except Exception:
                    suggestion = None
            elif isinstance(response, dict):
                suggestion = response.get("whiteboard_suggestion")

            if suggestion and getattr(context, "whiteboard", None):
                await context.whiteboard.apply_suggestion(
                    suggestion, source_agent=self.name
                )
            
            # STEP 4: Log worker's response to the whiteboard and persist if store attached
            if getattr(context, "whiteboard", None):
                await context.whiteboard.add_update(
                    agent_name=self.name,
                    update_type=WhiteboardUpdateType.FINDING,
                    content={
                        "task": user_input[:200],  # Truncate long inputs
                        "response_summary": response[:500] if isinstance(response, str) else str(response)[:500],
                        "status": "completed"
                    },
                    tags=["worker_response", self.name_key]
                )
            # Do not auto-save here; user or coordinator controls persistence policy
            
            return response

        tool_name = self.name_key + "_agent"

        return Tool(
            name=tool_name,
            func=func,
            description=self.introduction,
            parameters=self.parameters,
        )
