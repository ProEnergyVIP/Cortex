# Whiteboard Integration in Message Routing

## Overview

The coordinator-to-worker message routing has been augmented with Whiteboard capabilities. Workers now receive relevant team context before execution and automatically log their findings after completion.

## Changes Made

### File: `cortex/agent_system/coordinator_system/worker_builder.py`

#### 1. Added Import

```python
from ..core.context import AgentSystemContext, UpdateType
```

#### 2. Modified `install()` Method's `func` Implementation

The worker tool function now follows this enhanced flow:

**BEFORE (Original Flow):**
```python
async def func(args, context: AgentSystemContext):
    # 1. Run before_agent hook
    # 2. Build agent
    # 3. Extract user_input and context_instructions
    # 4. Create messages
    # 5. Call agent.async_ask()
    # 6. Return response
```

**AFTER (Enhanced Flow with Whiteboard):**
```python
async def func(args, context: AgentSystemContext):
    # 1. Run before_agent hook (unchanged)
    # 2. Build agent (unchanged)
    # 3. Extract user_input and context_instructions (unchanged)
    
    # ✨ NEW: STEP 1 - Get agent-specific view of the Whiteboard
    agent_view = context.get_agent_view(self.name)
    
    # ✨ NEW: STEP 2 - Build context summary from agent view
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
    
    # ✨ NEW: Combine Whiteboard with existing context_instructions
    shared_context_info = "\n\n".join(context_parts) if context_parts else None
    
    if shared_context_info:
        if ctx_instructions:
            combined_context = f"{ctx_instructions}\n\n[Whiteboard]\n{shared_context_info}"
        else:
            combined_context = f"[Whiteboard]\n{shared_context_info}"
    else:
        combined_context = ctx_instructions
    
    # 4. Create messages with enhanced context
    msgs = [UserMessage(content=user_input)]
    if combined_context:
        msgs.append(DeveloperMessage(content=combined_context))
    
    # ✨ STEP 3 - Execute worker agent
    response = await agent.async_ask(msgs, usage=getattr(context, "usage", None))
    
    # ✨ NEW: STEP 4 - Log worker's response to the Whiteboard
    context.add_update(
        agent_name=self.name,
        update_type=UpdateType.FINDING,
        content={
            "task": user_input[:200],  # Truncate long inputs
            "response_summary": response[:500] if isinstance(response, str) else str(response)[:500],
            "status": "completed"
        },
        tags=["worker_response", self.name_key]
    )
    
    # 6. Return response (unchanged)
    return response
```

## How It Works

### 1. **Before Routing to Worker** (Lines 244-277)

When the coordinator calls a worker agent tool:

```python
# Coordinator calls worker tool
await data_analyst_agent(
    user_input="Analyze Q4 sales trends",
    context_instructions="Focus on customer satisfaction metrics"
)
```

The worker tool function:
1. Calls `context.get_agent_view(worker_name)` to get relevant context
2. Extracts key information:
   - Mission
   - Current focus
   - Worker's role
   - Active blockers
   - Recent updates from other agents (last 5)
3. Formats this into a readable context summary
4. Combines it with any existing `context_instructions`

### 2. **Context Included in Message** (Lines 279-281)

The worker receives a `DeveloperMessage` with combined context:

```
[Whiteboard]
Mission: Build recommendation system
Current Focus: Feature extraction phase
Your Role: Data Analyst
Active Blockers: API rate limit issue

Recent Team Updates:
  - [progress] Data Engineer: Data pipeline deployed successfully
  - [decision] ML Engineer: Choosing gradient boosting for initial model
  - [blocker] Data Engineer: API rate limit reached, need to implement backoff
```

This gives the worker:
- ✅ Awareness of the overall mission
- ✅ Knowledge of their specific role
- ✅ Visibility into what other agents are doing
- ✅ Awareness of current blockers

### 3. **After Worker Responds** (Lines 286-296)

After the worker completes its task:

```python
context.add_update(
    agent_name="Data Analyst",
    update_type=UpdateType.FINDING,
    content={
        "task": "Analyze Q4 sales trends",
        "response_summary": "Customer satisfaction increased 15% in Q4...",
        "status": "completed"
    },
    tags=["worker_response", "data_analyst"]
)
```

This update:
- ✅ Records what the worker did
- ✅ Captures the response summary
- ✅ Makes it available to other agents via `get_agent_view()`
- ✅ Enables coordination between workers

## Benefits

### 1. **Automatic Whiteboard Sharing**
Workers automatically receive relevant context without coordinator having to manually pass it.

### 2. **Team Awareness**
Each worker can see what other workers have done recently, enabling better coordination.

### 3. **Maintains Existing Structure**
- ✅ No changes to coordinator logic
- ✅ No changes to message passing structure
- ✅ Backward compatible (works even if context fields are empty)
- ✅ Existing `context_instructions` still work and are preserved

### 4. **Automatic Logging**
All worker responses are automatically logged to shared context for team visibility.

## Example Flow

```python
# Setup
context = AgentSystemContext(
    memory_bank=memory_bank,
    mission="Build recommendation system",
    current_focus="Feature extraction",
    team_roles={
        "Data Engineer": "Infrastructure & Data",
        "ML Engineer": "Model Development",
        "Data Analyst": "Analysis & Insights"
    }
)

# Worker 1: Data Engineer deploys pipeline
# → Automatically logs: [progress] Data pipeline deployed

# Worker 2: ML Engineer makes decision
# → Receives context showing Data Engineer's progress
# → Automatically logs: [decision] Using gradient boosting

# Worker 3: Data Analyst analyzes data
# → Receives context showing both previous workers' updates
# → Knows pipeline is ready and model choice is made
# → Can provide analysis aligned with team decisions
# → Automatically logs: [finding] Customer satisfaction trends
```

## Configuration

### Customizing Context Information

You can control what context is shared by modifying the `context_parts` building logic (lines 248-266):

```python
# Add custom fields
if agent_view.get("protocols"):
    context_parts.append(f"Protocols: {', '.join(agent_view['protocols'])}")

# Adjust number of recent updates
recent = agent_view["recent_updates"][:10]  # Show last 10 instead of 5

# Filter updates by type
recent_blockers = [u for u in agent_view["recent_updates"] if u['type'] == 'blocker']
```

### Customizing Update Logging

You can change what gets logged after worker execution (lines 287-296):

```python
# Use different update types based on response
update_type = UpdateType.PROGRESS if "in progress" in response else UpdateType.FINDING

# Add more metadata
context.add_update(
    agent_name=self.name,
    update_type=update_type,
    content={
        "task": user_input[:200],
        "response_summary": response[:500],
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
        "execution_time_ms": execution_time
    },
    tags=["worker_response", self.name_key, "automated"]
)
```

## Testing

To verify the integration works:

```python
# 1. Set up context with mission and roles
context = AgentSystemContext(
    memory_bank=memory_bank,
    mission="Test mission",
    team_roles={"Worker1": "Tester"}
)

# 2. Call a worker
system = CoordinatorSystem(coordinator_builder, [worker_builder], context)
response = await system.async_ask("Test task")

# 3. Check that update was logged
assert len(context.updates) > 0
assert context.updates[-1].agent_name == "Worker1"
assert context.updates[-1].type == UpdateType.FINDING

# 4. Verify next worker receives context
view = context.get_agent_view("Worker2")
assert len(view["recent_updates"]) > 0
```

## Migration Notes

### Existing Code Compatibility

✅ **No breaking changes** - existing coordinator systems work without modification:
- If context fields are empty, no shared context is added
- Workers still receive their original `context_instructions`
- Response format unchanged

### Gradual Adoption

You can adopt shared context incrementally:

1. **Phase 1**: Just set `mission` and `team_roles` - workers get basic context
2. **Phase 2**: Add `current_focus` and `protocols` - workers get more guidance
3. **Phase 3**: Use `active_blockers` - workers aware of issues
4. **Phase 4**: Leverage `recent_updates` - full team coordination

## See Also

- `AgentSystemContext` - Main context class
- `ContextUpdate` and `UpdateType` - Update model and types
- `CONTEXT_UPDATE_REFERENCE.md` - Detailed update patterns
- `examples/shared_context_example.py` - Usage examples
