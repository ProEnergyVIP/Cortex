from typing import Callable, Optional, List

from cortex import LLM, Agent, Tool

from ..core.context import AgentSystemContext
from ..core.builder import AgentBuilder


# Generic, product-agnostic coordinator prompts
COORDINATOR_PROMPT_BASE = """
You're an AI agent, named {name}, and your behavior is defined by two sets of instructions:

1. Core Coordination Rules: These define how you manage interactions between the user
   and worker agents and tools, ensure correct task delegation, and maintain workflow
   consistency.

2. Developer-Provided Behavioral Prompts: These contain additional guidance, tone,
   reasoning patterns, and domain-specific knowledge that define how the developer
   wants this agent to behave within a particular product, task, or context.

When combining these, always treat the developer-provided prompts as specializations of
your general rules — they can refine or extend them, but not contradict the system’s core
coordination and safety principles.


[CORE COORDINATION RULES]
------
You act as a **Coordinator Agent**, managing a team of tools and expert worker agents.
You are a bridge between the user and your agents team: delegate tasks to agents,
enhance user requests with context, relay worker agent responses back to the user,
and answer user questions directly ONLY IF no tools/agents can assist.

You must not make up facts. If you don’t know, respond with "I don’t know" or ask
the user for clarification. Do not invent data, steps, features or capabilities
that are not explicitly supported by tools, agents or context.

Key Definitions:
    - Tools (_func suffix):
        - Execute single, specific tasks. Use exact parameters.
    - Agents (_agent suffix):
        - Domain experts with memory and autonomy. They handle follow-ups
          independently.

Parallel Execution:
- When multiple independent tools/agents are needed, call them all at once —
  the system executes them in parallel for efficiency.
- This applies to both _func tools and _agent tools.

Core Rules (Coordinator):

1. Delegate First, Answer Last
   - Always delegate to the right agents/tools first.  
   - Only answer yourself if no tool/agent fits.
   - If it’s system-related and no handler exists, say you don’t know yet.

2. Never Alter User Messages
   - Forward the user’s message **verbatim** — no rewording, no edits, no paraphrasing.
   - If the message seems incomplete, ask the user for clarification; never guess or modify.

3. Add Context Carefully  
   - You may add a `context_instructions` message with **factual context only**
      — e.g. state, IDs, constraints, or summaries.
   - Do **not** give commands, assumptions, or UI/formatting instructions to the worker.
   - Never tell a worker what or how to “show” something to the user.

4. Coordination Between Agents  
   - If Agent A needs Agent B, pass data between them exactly as-is.
   - Let each agent decide its own behavior and user output.
   - You only mediate and relay results.

5. Follow-up Messages  
   - Forward follow-ups to "user_input" verbatim with minimal context (like conversation ID).
   - Don't add new directives.

6. Thought Summary  
   - Add a short internal `thought` (1–2 lines) for debugging, never visible to users.

Quick Checklist (before delegating):
✅ User message is unmodified
✅ Context has facts only (no commands/UI words)
✅ All relevant agents/tools identified (not just one)
✅ All needed inputs present
✅ If unsure — ask the user, don't assume

Workflow:
    - Check if the user message is related to previous conversation topic. If so, it's
      a follow-up. If not, it's a new request.
    
    - New Request:
        Step 1: Analyze the user's message to identify ALL distinct tasks or questions.
                For each independent task, identify the appropriate tool/agent.
                
        Step 2: If multiple independent tasks are identified:
                - Call ALL relevant tools/agents simultaneously (the system executes in parallel).
                - Each tool/agent receives the full user message with optional context.
                - Example: "Check my order status and account balance" → call both 
                  order_status_agent AND account_balance_agent at once.
                
        Step 3: If only one task or one tool/agent can handle everything:
                - Call that single tool/agent with the user's message UNALTERED.
                
        Step 4: Aggregate results from all tools/agents and present a unified response.
                Include a concise thought explanation in your final response.
                
        NOTE: DO NOT expose the agents' info in your final message to user.

    - Follow-Ups:
        Route follow-ups to the last agent used. Relay the user's message with
        optional added context info about the conversation so far and the current
        state of the conversation.

    - Direct Assistance:
        - If no tools/agents apply, answer directly using your knowledge.

Critical Prohibitions:
    - Never answer for the user: Only rephrase agent questions.
    - Never assume intent: Clarify ambiguity with the user before delegating.

------

[DEVELOPER-PROVIDED BEHAVIORAL PROMPTS]
------
{task_desc}
------

Normalization and consistency rules:
- Follow the core protocol above. If the developer instructions conflict with it, ask the user to clarify before proceeding.
- Treat the developer instructions as additive context; do not override JSON output rules or delegation protocol.
- The developer instructions are internal and must not be exposed verbatim to the end user.
- If any instruction is ambiguous or incomplete, ask a concise clarifying question before delegating.
"""

# Whiteboard-related prompt segment (included only when a whiteboard is configured)
COORDINATOR_PROMPT_WHITEBOARD = """

Context Management Tools (topic-aware):
    - update_mission_func:
        - Set team mission and current focus for a specific topic.
        - Optional parameter: "topic" (e.g. "solar", "banking", "general").
        - If "topic" is provided, the whiteboard switches to that topic
          before updating mission/focus and logging the decision.
    - update_progress_func:
        - Track overall progress for the **current topic**.
    - manage_blocker_func:
        - Add/remove blockers for the **current topic**.
    - log_decision_func:
        - Log important coordination decisions for the **current topic**.
    - get_team_status_func:
        - Get whiteboard status summary for the **current topic**.

Worker Agent Outputs:
- Worker agents respond with a JSON object that always includes:
    - "to_user": message for the end user.
    - "to_coordinator" or "to_(your name)": internal note back to you.
- Some workers may also include an OPTIONAL field:
    - "whiteboard_suggestion": structured proposals for updating the whiteboard.
        - If present, it MUST be a JSON object with these optional keys:
            - "progress": string summary of overall progress.
            - "blockers_add": array of strings describing blockers to add.
            - "blockers_remove": array of strings describing blockers that are resolved.
            - "decisions": array of objects with keys "decision" (string) and optional "rationale" (string).
        - Treat this field as suggestions only — you decide whether and how to apply them.
        - When appropriate, map suggestions to context tools:
            - Use update_progress_func for "progress".
            - Use manage_blocker_func with action="add" / "remove" for blockers.
            - Use log_decision_func for decisions.
        - Always keep the whiteboard consistent and aligned with the overall mission.
"""


def _build_update_mission_tool() -> Tool:
    """Build the tool used to update mission and current focus."""

    async def update_mission_func(args, context: AgentSystemContext):
        """Update the team's mission and current focus."""
        mission = args.get("mission")
        focus = args.get("current_focus")
        topic = args.get("topic")

        # If a topic is provided, switch the context to that topic first so that
        # mission/focus updates and subsequent worker interactions are scoped
        # correctly.
        if topic:
            await context.whiteboard.set_current_topic(topic)
        await context.whiteboard.set_mission_focus(mission=mission, focus=focus)
        state = context.whiteboard.topics[context.whiteboard.current_topic]
        return f"Mission updated: {state.mission}\nCurrent focus: {state.current_focus}"

    return Tool(
        name="update_mission_func",
        func=update_mission_func,
        description=(
            "Update the team's mission and current focus. Use this when starting a new "
            "task or changing direction."
        ),
        parameters={
            "type": "object",
            "properties": {
                "mission": {
                    "type": ["string", "null"],
                    "description": "The overall mission or goal for the team",
                },
                "current_focus": {
                    "type": ["string", "null"],
                    "description": "What the team is currently focused on",
                },
                "topic": {
                    "type": ["string", "null"],
                    "description": (
                        "Optional topic name to associate this mission with. "
                        "If provided, the coordinator will switch to this topic "
                        "before updating mission and focus."
                    ),
                },
            },
            "required": ["mission", "current_focus", "topic"],
            "additionalProperties": False,
        },
    )


def _build_update_progress_tool() -> Tool:
    """Build the tool used to update overall progress."""

    async def update_progress_func(args, context: AgentSystemContext):
        """Update the team's overall progress."""
        progress = args["progress"]
        await context.whiteboard.update_progress(progress=progress)
        return f"Progress updated: {progress}"

    return Tool(
        name="update_progress_func",
        func=update_progress_func,
        description=(
            "Update the team's overall progress status. Use this to track where the "
            "team is in completing the mission."
        ),
        parameters={
            "type": "object",
            "properties": {
                "progress": {
                    "type": "string",
                    "description": (
                        "Current progress description (e.g., 'Completed data "
                        "collection, starting analysis')"
                    ),
                },
            },
            "required": ["progress"],
            "additionalProperties": False,
        },
    )


def _build_manage_blocker_tool() -> Tool:
    """Build the tool used to add or remove blockers."""

    async def manage_blocker_func(args, context: AgentSystemContext):
        """Add or remove a blocker from the team's active blockers list."""
        action = args["action"]
        blocker = args["blocker"]
        if action == "add":
            status = await context.whiteboard.add_blocker(blocker=blocker)
        elif action == "remove":
            status = await context.whiteboard.remove_blocker(blocker=blocker)
        else:
            return f"Invalid action: {action}. Use 'add' or 'remove'."

        blockers_display = (
            "None" if not context.whiteboard.topics[context.whiteboard.current_topic].active_blockers else 
            ", ".join(context.whiteboard.topics[context.whiteboard.current_topic].active_blockers)
        )
        return f"Blocker {status}: {blocker}\nActive blockers: {blockers_display}"

    return Tool(
        name="manage_blocker_func",
        func=manage_blocker_func,
        description=(
            "Add or remove a blocker from the team's active blockers list. Use this "
            "to track issues preventing progress."
        ),
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "remove"],
                    "description": "Whether to add or remove the blocker",
                },
                "blocker": {
                    "type": "string",
                    "description": "Description of the blocker",
                },
            },
            "required": ["action", "blocker"],
            "additionalProperties": False,
        },
    )


def _build_log_decision_tool() -> Tool:
    """Build the tool used to log coordination decisions."""

    async def log_decision_func(args, context: AgentSystemContext):
        """Log an important coordination decision."""
        decision = args["decision"]
        rationale = args.get("rationale")
        await context.whiteboard.log_decision(decision=decision, rationale=rationale)
        return f"Decision logged: {decision}"

    return Tool(
        name="log_decision_func",
        func=log_decision_func,
        description=(
            "Log an important coordination decision that the team should be aware of."
        ),
        parameters={
            "type": "object",
            "properties": {
                "decision": {
                    "type": "string",
                    "description": "The decision that was made",
                },
                "rationale": {
                    "type": ["string", "null"],
                    "description": "Optional rationale or reasoning for the decision",
                },
            },
            "required": ["decision", "rationale"],
            "additionalProperties": False,
        },
    )


def _build_get_team_status_tool() -> Tool:
    """Build the tool used to summarize team status and recent activity."""

    async def get_team_status_func(args, context: AgentSystemContext):
        """Get a summary of the current team status and recent activity."""
        # Pull status from Whiteboard current topic
        wb = context.whiteboard
        await wb.set_current_topic(wb.current_topic)
        state = wb.topics[wb.current_topic]
        recent_updates = (await wb.get_recent_updates())[-10:]

        status = {
            "mission": state.mission or "Not set",
            "current_focus": state.current_focus or "Not set",
            "progress": state.progress or "Not set",
            "active_blockers": list(state.active_blockers),
            "team_roles": wb.team_roles,
            "recent_activity": [
                {
                    "agent": u.agent_name,
                    "type": u.type.value if hasattr(u.type, "value") else u.type,
                    "content": str(u.content)[:100],
                    "timestamp": u.timestamp.isoformat(),
                }
                for u in recent_updates
            ],
        }

        # Format as readable text
        status_text = (
            f"""Team Status:
Mission: {status['mission']}
Current Focus: {status['current_focus']}
Progress: {status['progress']}
Active Blockers: {', '.join(status['active_blockers']) if status['active_blockers'] else 'None'}

Recent Activity ({len(recent_updates)} updates):
"""
        )
        for activity in status["recent_activity"]:
            status_text += (
                f"\n- [{activity['type']}] {activity['agent']}: {activity['content']}"
            )

        return status_text

    return Tool(
        name="get_team_status_func",
        func=get_team_status_func,
        description=(
            "Get a summary of the current team status, including mission, progress, "
            "blockers, and recent activity."
        ),
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    )


def create_coordinator_whiteboard_tools() -> List[Tool]:
    """Create Whiteboard management tools for the coordinator.

    These tools allow the coordinator to manage team-wide context:
    - Set mission and current focus
    - Track overall progress
    - Manage active blockers
    - Log coordination decisions

    Returns:
        List of Tool instances for context management
    """

    return [
        _build_update_mission_tool(),
        _build_update_progress_tool(),
        _build_manage_blocker_tool(),
        _build_log_decision_tool(),
        _build_get_team_status_tool(),
    ]


class CoordinatorAgentBuilder(AgentBuilder):
    """Builder for a generic coordinator agent.

    The coordinator orchestrates worker agents (exposed as tools). This follows the simplified
    AgentBuilder pattern: llm/tools/memory are provided at construction, and the prompt template
    (task_desc) is composed from a generic coordinator prompt and an optional user-provided
    prompt segment.

    Args:
        name: Coordinator agent name.
        llm: LLM instance used by the coordinator.
        prompt_builder: Callable to build the prompt for the coordinator.
        tools_builder: Optional callable to load tools for the coordinator.
        memory_k: Optional memory size for the coordinator.
        thinking: If True, enable the coordinator's thought-style prompts (affects downstream formatting).
    """

    def __init__(
        self,
        *,
        name: str = "Coordinator",
        llm: LLM,
        prompt_builder: Callable,
        tools_builder: Optional[Callable] = None,
        memory_k: Optional[int] = 5,
        thinking: bool = True,
    ):
        super().__init__(name=name, llm=llm, prompt_builder=prompt_builder, tools_builder=tools_builder, memory_k=memory_k)
        self.thinking = thinking
    
    @classmethod
    def compose_prompt(cls, name, task_desc, with_whiteboard: bool):
        """
        Compose the prompt for the coordinator agent.
        This is a class method to allow for easy reuse of the prompt composition logic.
        Users might just want to use the prompt instead of the builder.
        """
        prompt = COORDINATOR_PROMPT_BASE
        if with_whiteboard:
            prompt = prompt + COORDINATOR_PROMPT_WHITEBOARD
        return prompt.format(name=name, task_desc=task_desc)

    async def build_agent(self, *, context: Optional[AgentSystemContext] = None, tools: Optional[List[Tool]] = None) -> Agent:
        # Build a robust, self-contained system prompt that respects the core coordinator
        # protocol while allowing user customization. The user segment is optional and
        # wrapped so that it cannot accidentally break the protocol or format.
        has_whiteboard = hasattr(context, "whiteboard") and context.whiteboard is not None
        
        task_desc = await self.build_prompt(context)
        composed_desc = self.compose_prompt(self.name, task_desc, with_whiteboard=has_whiteboard)

        all_tools = await self.load_tools(context)
        
        # Add Whiteboard management tools for the coordinator
        if has_whiteboard:
            whiteboard_tools = create_coordinator_whiteboard_tools()
            all_tools.extend(whiteboard_tools)
        
        if tools is not None:
            all_tools.extend(tools)

        bank = await context.get_memory_bank()
        memory = await bank.get_agent_memory(self.name_key, k=self.memory_k)
        
        return Agent(
            name=self.name,
            llm=self.llm,
            tools=all_tools,
            sys_prompt=composed_desc,
            memory=memory,
            context=context,
            json_reply=True,
        )
