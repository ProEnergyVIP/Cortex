# Coordinator Context Management

## Overview

The coordinator agent now actively manages the Whiteboard for the entire team. This enables:
- **Mission Setting**: Coordinator defines team goals at task start
- **Progress Tracking**: Coordinator monitors and updates team progress
- **Blocker Management**: Coordinator tracks issues preventing progress
- **Decision Logging**: Coordinator records important coordination decisions
- **Team Status**: Coordinator can query current team state

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER REQUEST                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  COORDINATOR AGENT                          │
│                                                             │
│  1. Receives request                                        │
│  2. Sets mission/focus (update_mission_func)                │
│  3. Delegates to workers                                    │
│  4. Updates progress (update_progress_func)                 │
│  5. Manages blockers (manage_blocker_func)                  │
│  6. Logs decisions (log_decision_func)                      │
│  7. Returns response to user                                │
└─────────────────────────────────────────────────────────────┘
                            ↓
        ┌───────────────────┼───────────────────┐
        ↓                   ↓                   ↓
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  WORKER 1    │   │  WORKER 2    │   │  WORKER 3    │
│              │   │              │   │              │
│ Gets context │   │ Gets context │   │ Gets context │
│ from the     │   │ from the     │   │ from the     │
│ Whiteboard   │   │ Whiteboard   │   │ Whiteboard   │
│              │   │              │   │              │
│ Executes     │   │ Executes     │   │ Executes     │
│ task         │   │ task         │   │ task         │
│              │   │              │   │              │
│ Logs result  │   │ Logs result  │   │ Logs result  │
│ to           │   │ to           │   │ to           │
│ Whiteboard   │   │ Whiteboard   │   │ Whiteboard   │
└──────────────┘   └──────────────┘   └──────────────┘
```

## Coordinator Context Management Tools

The coordinator automatically gets 5 context management tools:

### 1. `update_mission_func`
**Purpose**: Set or update the team's mission and current focus

**When to use**: 
- At the start of a new project or major task
- When changing direction or priorities

**Parameters**:
```python
{
    "mission": "Overall goal for the team",
    "current_focus": "What the team is currently focused on"
}
```

**Example**:
```python
# Coordinator automatically calls this
update_mission_func({
    "mission": "Build customer churn prediction system",
    "current_focus": "Data collection and pipeline setup"
})
```

**Effect**:
- Updates `context.mission` and `context.current_focus`
- Logs decision to the Whiteboard
- Workers automatically receive this in their context view

### 2. `update_progress_func`
**Purpose**: Track the team's overall progress

**When to use**:
- After workers complete significant milestones
- When transitioning between project phases

**Parameters**:
```python
{
    "progress": "Current progress description"
}
```

**Example**:
```python
update_progress_func({
    "progress": "Completed data collection, starting feature extraction"
})
```

**Effect**:
- Updates `context.progress`
- Logs progress update to the Whiteboard
- Workers see current progress in their context view

### 3. `manage_blocker_func`
**Purpose**: Add or remove blockers from the team's active blockers list

**When to use**:
- When a worker reports an issue
- When a blocker is resolved

**Parameters**:
```python
{
    "action": "add" | "remove",
    "blocker": "Description of the blocker"
}
```

**Example**:
```python
# Add a blocker
manage_blocker_func({
    "action": "add",
    "blocker": "API rate limit reached, need backoff logic"
})

# Remove a blocker
manage_blocker_func({
    "action": "remove",
    "blocker": "API rate limit reached, need backoff logic"
})
```

**Effect**:
- Updates `context.active_blockers` list
- Logs blocker change to the Whiteboard
- Workers see active blockers in their context view

### 4. `log_decision_func`
**Purpose**: Log important coordination decisions

**When to use**:
- When making architectural or strategic decisions
- When choosing between alternatives
- When establishing team protocols

**Parameters**:
```python
{
    "decision": "The decision that was made",
    "rationale": "Optional reasoning for the decision"
}
```

**Example**:
```python
log_decision_func({
    "decision": "Using gradient boosting for initial model",
    "rationale": "Best performance on validation set with limited training time"
})
```

**Effect**:
- Logs decision to the Whiteboard
- Workers see decisions in recent updates

### 5. `get_team_status_func`
**Purpose**: Get a summary of current team status and recent activity

**When to use**:
- When user asks for status update
- Before making coordination decisions
- To check team progress

**Parameters**: None

**Example**:
```python
get_team_status_func({})
```

**Returns**:
```
Team Status:
Mission: Build recommendation system
Current Focus: Feature extraction phase
Progress: Completed data collection, starting analysis
Active Blockers: API rate limit issue

Recent Activity (10 updates):
- [progress] Data Engineer: Data pipeline deployed successfully
- [decision] ML Engineer: Choosing gradient boosting for initial model
- [blocker] Data Engineer: API rate limit reached, need to implement backoff
...
```

## Coordinator Workflow with Context Management

The coordinator prompt now includes guidance on using these tools:

### New Request Workflow

```
Step 0: Set Mission & Focus
    ↓
    update_mission_func({
        mission: "Overall goal",
        current_focus: "Current phase"
    })

Step 1-3: Analyze & Delegate
    ↓
    Identify tasks and call worker agents

Step 4: Track Progress & Blockers
    ↓
    update_progress_func({progress: "Status"})
    
    If blockers identified:
    manage_blocker_func({action: "add", blocker: "Issue"})

Step 5: Return Response
    ↓
    Aggregate results and respond to user
```

## Complete Example

```python
from cortex import LLM, GPTModels
from cortex.agent_memory import AsyncAgentMemoryBank
from cortex.agent_system import (
    AgentSystemContext,
    CoordinatorSystem,
    CoordinatorAgentBuilder,
    WorkerAgentBuilder,
)

# Initialize context with team roles
memory_bank = AsyncAgentMemoryBank()
context = AgentSystemContext(
    memory_bank=memory_bank,
    team_roles={
        "Data Engineer": "Infrastructure & Data",
        "Data Analyst": "Analysis & Insights",
        "ML Engineer": "Model Development"
    }
)

# Define workers
data_engineer = WorkerAgentBuilder(
    name="Data Engineer",
    llm=LLM(model=GPTModels.GPT_4O_MINI),
    prompt_builder=lambda ctx: "You handle data infrastructure...",
    introduction="Data Engineer - handles data pipelines"
)

data_analyst = WorkerAgentBuilder(
    name="Data Analyst",
    llm=LLM(model=GPTModels.GPT_4O_MINI),
    prompt_builder=lambda ctx: "You analyze data patterns...",
    introduction="Data Analyst - performs analysis"
)

ml_engineer = WorkerAgentBuilder(
    name="ML Engineer",
    llm=LLM(model=GPTModels.GPT_4O_MINI),
    prompt_builder=lambda ctx: "You build ML models...",
    introduction="ML Engineer - builds models"
)

# Define coordinator
coordinator = CoordinatorAgentBuilder(
    name="Team Lead",
    llm=LLM(model=GPTModels.GPT_4O_MINI),
    prompt_builder=lambda ctx: """
    You coordinate the data science team.
    Use context management tools to track team progress.
    """
)

# Build system
system = CoordinatorSystem(
    coordinator_builder=coordinator,
    workers=[data_engineer, data_analyst, ml_engineer],
    context=context
)

# Use the system
response = await system.async_ask(
    "Build a customer churn prediction system"
)

# Coordinator automatically:
# 1. Calls update_mission_func to set mission
# 2. Delegates to data_engineer and data_analyst
# 3. Workers receive the Whiteboard automatically
# 4. Workers log their findings
# 5. Coordinator calls update_progress_func
# 6. Returns response

# Check Whiteboard
print(f"Mission: {context.mission}")
print(f"Progress: {context.progress}")
print(f"Updates: {len(context.updates)}")
```

## What Happens Behind the Scenes

### 1. Coordinator Sets Mission
```python
# Coordinator receives: "Build a customer churn prediction system"

# Coordinator thinks: "This is a new major task, I should set the mission"

# Coordinator calls:
update_mission_func({
    "mission": "Build customer churn prediction system",
    "current_focus": "Data collection and pipeline setup"
})

# Result:
# - context.mission = "Build customer churn prediction system"
# - context.current_focus = "Data collection and pipeline setup"
# - Update logged with type=DECISION, agent_name="Coordinator"
```

### 2. Coordinator Delegates to Workers
```python
# Coordinator calls workers in parallel:
await data_engineer_agent({
    "user_input": "Set up data pipeline for customer data",
    "context_instructions": "Focus on churn-related features"
})

await data_analyst_agent({
    "user_input": "Analyze customer behavior patterns",
    "context_instructions": "Identify churn indicators"
})
```

### 3. Workers Receive Context Automatically
```python
# Inside data_engineer_agent tool:

# STEP 1: Get agent view
agent_view = context.get_agent_view("Data Engineer")
# Returns:
# {
#     "mission": "Build customer churn prediction system",
#     "current_focus": "Data collection and pipeline setup",
#     "my_role": "Infrastructure & Data",
#     "recent_updates": [],
#     ...
# }

# STEP 2: Build context message
context_message = """
[Whiteboard]
Mission: Build customer churn prediction system
Current Focus: Data collection and pipeline setup
Your Role: Infrastructure & Data
"""

# STEP 3: Send to worker with context
messages = [
    UserMessage("Set up data pipeline for customer data"),
    DeveloperMessage(context_message)
]

# Worker receives full context and executes task
```

### 4. Workers Log Results
```python
# After data_engineer completes:
context.add_update(
    agent_name="Data Engineer",
    update_type=UpdateType.FINDING,
    content={
        "task": "Set up data pipeline for customer data",
        "response_summary": "Data pipeline deployed successfully...",
        "status": "completed"
    },
    tags=["worker_response", "data_engineer"]
)

# This update is now available to other workers!
```

### 5. Coordinator Tracks Progress
```python
# After workers complete, coordinator calls:
update_progress_func({
    "progress": "Data pipeline deployed, analysis in progress"
})

# If issues found:
manage_blocker_func({
    "action": "add",
    "blocker": "API rate limit reached"
})
```

### 6. Next Worker Sees Full Context
```python
# When ML Engineer is called later:
agent_view = context.get_agent_view("ML Engineer")
# Returns:
# {
#     "mission": "Build customer churn prediction system",
#     "current_focus": "Data collection and pipeline setup",
#     "my_role": "Model Development",
#     "progress": "Data pipeline deployed, analysis in progress",
#     "active_blockers": ["API rate limit reached"],
#     "recent_updates": [
#         {"agent_name": "Coordinator", "type": "decision", ...},
#         {"agent_name": "Data Engineer", "type": "finding", ...},
#         {"agent_name": "Data Analyst", "type": "finding", ...},
#         {"agent_name": "Coordinator", "type": "progress", ...}
#     ]
# }

# ML Engineer now knows:
# - What the team is building
# - What's been done already
# - Current issues/blockers
# - Their role in the team
```

## Benefits

### 1. **Automatic Team Coordination**
- Coordinator sets mission → All workers know the goal
- Workers log findings → Other workers see progress
- Coordinator tracks blockers → Workers aware of issues

### 2. **No Manual Context Passing**
- Coordinator doesn't need to manually tell each worker what others did
- Workers automatically receive relevant context
- Context stays synchronized across the team

### 3. **Better Decision Making**
- Workers can see what other workers have done
- Workers can avoid duplicate work
- Workers can build on each other's findings

### 4. **Progress Visibility**
- User can ask "What's the status?" → Coordinator uses get_team_status_func
- Clear audit trail of all decisions and progress
- Easy to understand what happened and when

## Implementation Details

### Files Modified

1. **`coordinator_builder.py`**
   - Added `UpdateType` import
   - Added `create_coordinator_context_tools()` function (5 tools)
   - Modified `build_agent()` to automatically add context tools
   - Updated coordinator prompt with tool descriptions and workflow

2. **`worker_builder.py`**
   - Added `UpdateType` import
   - Modified `install()` function to:
     - Get agent view before routing
     - Include context in worker message
     - Log worker response after completion

### No Changes Needed

- ✅ `CoordinatorSystem` - works as-is
- ✅ `AgentSystem` - works as-is
- ✅ User code - backward compatible

## Testing

```python
# Test coordinator context management
async def test_coordinator_context():
    context = AgentSystemContext(
        memory_bank=memory_bank,
        team_roles={"Worker1": "Role1"}
    )
    
    system = CoordinatorSystem(coordinator, [worker], context)
    
    # Make request
    await system.async_ask("Build something")
    
    # Verify coordinator set mission
    assert context.mission != ""
    
    # Verify coordinator tracked progress
    assert context.progress != ""
    
    # Verify workers logged findings
    assert len(context.updates) > 0
    
    # Verify coordinator updates are logged
    coordinator_updates = [
        u for u in context.updates 
        if u.agent_name == "Coordinator"
    ]
    assert len(coordinator_updates) > 0
```

## Migration Guide

### Existing Systems

✅ **No code changes required** - existing coordinator systems work without modification

The context management tools are automatically added to all coordinators. If the coordinator doesn't use them, they're simply ignored.

### Recommended Adoption

1. **Phase 1**: Let coordinator automatically use tools
   - Coordinator will naturally start using tools based on prompt guidance
   - No code changes needed

2. **Phase 2**: Initialize team roles
   ```python
   context = AgentSystemContext(
       memory_bank=memory_bank,
       team_roles={
           "Worker1": "Role description",
           "Worker2": "Role description"
       }
   )
   ```

3. **Phase 3**: Customize coordinator prompt
   ```python
   coordinator = CoordinatorAgentBuilder(
       name="Lead",
       llm=llm,
       prompt_builder=lambda ctx: """
       You coordinate the team.
       
       Always use update_mission_func at task start.
       Always use update_progress_func after major milestones.
       Track blockers with manage_blocker_func.
       """
   )
   ```

## See Also

- `AgentSystemContext` - Core context class
- `ContextUpdate` and `UpdateType` - Update model
- `WHITEBOARD_ROUTING_CHANGES.md` - Worker routing details
- `examples/coordinator_whiteboard_example.py` - Complete examples
