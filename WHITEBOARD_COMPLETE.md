# Whiteboard System - Complete Implementation

## Overview

The Whiteboard enables seamless coordination between the coordinator and worker agents through automatic context sharing. The coordinator actively manages team-wide state, and workers automatically receive and contribute to this shared state.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER REQUEST                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    COORDINATOR AGENT                            │
│                                                                 │
│  Context Management Tools (Automatic):                          │
│  • update_mission_func - Set team mission/focus                 │
│  • update_progress_func - Track progress                        │
│  • manage_blocker_func - Manage blockers                        │
│  • log_decision_func - Log decisions                            │
│  • get_team_status_func - Get status                            │
│                                                                 │
│  Workflow:                                                      │
│  1. Set mission (update_mission_func)                           │
│  2. Delegate to workers                                         │
│  3. Track progress (update_progress_func)                       │
│  4. Manage blockers (manage_blocker_func)                       │
│  5. Return response                                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
              ┌───────────────┼───────────────┐
              ↓               ↓               ↓
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │  WORKER 1   │  │  WORKER 2   │  │  WORKER 3   │
    │             │  │             │  │             │
    │ BEFORE:     │  │ BEFORE:     │  │ BEFORE:     │
    │ Get context │  │ Get context │  │ Get context │
    │ view        │  │ view        │  │ view        │
    │             │  │             │  │             │
    │ DURING:     │  │ DURING:     │  │ DURING:     │
    │ Execute     │  │ Execute     │  │ Execute     │
    │ with        │  │ with        │  │ with        │
    │ context     │  │ context     │  │ context     │
    │             │  │             │  │             │
    │ AFTER:      │  │ AFTER:      │  │ AFTER:      │
    │ Log result  │  │ Log result  │  │ Log result  │
    │ to Whiteboard│  │ to Whiteboard│  │ to Whiteboard│
    │ context     │  │ context     │  │ context     │
    └─────────────┘  └─────────────┘  └─────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      WHITEBOARD STATE                           │
│                                                                 │
│  • mission: "Build customer churn prediction system"            │
│  • current_focus: "Data collection phase"                       │
│  • progress: "Pipeline deployed, analysis in progress"          │
│  • team_roles: {"Data Engineer": "Infrastructure", ...}         │
│  • active_blockers: ["API rate limit"]                          │
│  • updates: [                                                   │
│      {agent: "Coordinator", type: "decision", ...},             │
│      {agent: "Data Engineer", type: "finding", ...},            │
│      {agent: "Data Analyst", type: "finding", ...}              │
│    ]                                                            │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. AgentSystemContext (Core)
**File**: `cortex/agent_system/core/context.py`

**Whiteboard Fields**:
```python
class AgentSystemContext(BaseModel):
    # Existing fields
    usage: Optional[AgentUsage] = None
    memory_bank: Optional[object] = None
    
    # Whiteboard fields
    mission: str = ""                              # Team's overall goal
    current_focus: str = ""                        # Current phase/focus
    progress: str = ""                             # Overall progress
    team_roles: Dict[str, str] = {}                # agent_name -> role
    protocols: List[str] = []                      # Team protocols
    updates: List[ContextUpdate] = []              # All updates
    artifacts: Dict[str, List[Dict]] = {}          # Shared artifacts
    last_activity: datetime = Field(...)           # Last update time
    active_blockers: List[str] = []                # Current blockers
```

**Methods**:
```python
def add_update(
    agent_name: str,
    update_type: UpdateType,
    content: Dict[str, Any],
    tags: Optional[List[str]] = None
) -> ContextUpdate

def get_agent_view(agent_name: str) -> Dict

def get_recent_updates(
    since: Optional[datetime] = None,
    agent_name: Optional[str] = None,
    update_type: Optional[UpdateType] = None
) -> List[ContextUpdate]
```

### 2. ContextUpdate Model
**File**: `cortex/agent_system/core/context.py`

```python
class UpdateType(str, Enum):
    PROGRESS = "progress"
    FINDING = "finding"
    DECISION = "decision"
    ARTIFACT = "artifact"
    BLOCKER = "blocker"

class ContextUpdate(BaseModel):
    id: str                                        # Short UUID
    agent_name: str                                # Agent that created update
    type: UpdateType                               # Type of update
    content: Dict[str, Any]                        # Flexible content
    timestamp: datetime                            # When created
    tags: List[str] = []                           # Tags for filtering
```

### 3. Coordinator Context Management
**File**: `cortex/agent_system/coordinator_system/coordinator_builder.py`

**Tools Added**:
1. `update_mission_func` - Set mission and focus
2. `update_progress_func` - Track progress
3. `manage_blocker_func` - Add/remove blockers
4. `log_decision_func` - Log decisions
5. `get_team_status_func` - Get status summary

**Prompt Updated**:
- Added tool descriptions
- Added workflow guidance (Step 0: Set mission, Step 4: Track progress)
- Instructs coordinator to use tools for team coordination

### 4. Worker Context Integration
**File**: `cortex/agent_system/coordinator_system/worker_builder.py`

**Enhanced Flow**:
```python
async def func(args, context: AgentSystemContext):
    # STEP 1: Get agent view
    agent_view = context.get_agent_view(worker_name)
    
    # STEP 2: Build context summary
    context_parts = [
        f"Mission: {agent_view['mission']}",
        f"Current Focus: {agent_view['current_focus']}",
        f"Your Role: {agent_view['my_role']}",
        f"Active Blockers: {agent_view['active_blockers']}",
        "Recent Team Updates: ..."
    ]
    
    # STEP 3: Execute worker with context
    response = await agent.async_ask([
        UserMessage(user_input),
        DeveloperMessage(context_summary)
    ])
    
    # STEP 4: Log worker response
    context.add_update(
        agent_name=worker_name,
        update_type=UpdateType.FINDING,
        content={
            "task": user_input,
            "response_summary": response,
            "status": "completed"
        },
        tags=["worker_response", worker_key]
    )
    
    return response
```

## Complete Flow Example

### User Request
```python
await system.async_ask("Build a customer churn prediction system")
```

### Step-by-Step Execution

#### 1. Coordinator Receives Request
```python
# Coordinator analyzes request
# Identifies: New major task requiring data engineering and analysis
```

#### 2. Coordinator Sets Mission
```python
# Coordinator calls update_mission_func
context.mission = "Build customer churn prediction system"
context.current_focus = "Data collection and pipeline setup"

# Logged to context:
context.updates.append(ContextUpdate(
    agent_name="Coordinator",
    type=UpdateType.DECISION,
    content={
        "action": "mission_updated",
        "mission": "Build customer churn prediction system",
        "focus": "Data collection and pipeline setup"
    },
    tags=["coordinator", "mission"]
))
```

#### 3. Coordinator Delegates to Workers
```python
# Coordinator calls workers in parallel
await data_engineer_agent({
    "user_input": "Set up data pipeline for customer churn data",
    "context_instructions": "Focus on churn-related features"
})

await data_analyst_agent({
    "user_input": "Analyze customer behavior patterns for churn",
    "context_instructions": "Identify key churn indicators"
})
```

#### 4. Worker 1 (Data Engineer) Executes

**4a. Get Context View**
```python
agent_view = context.get_agent_view("Data Engineer")
# Returns:
{
    "mission": "Build customer churn prediction system",
    "current_focus": "Data collection and pipeline setup",
    "my_role": "Infrastructure & Data",
    "progress": "",
    "active_blockers": [],
    "recent_updates": [
        {
            "agent_name": "Coordinator",
            "type": "decision",
            "content": {"action": "mission_updated", ...},
            ...
        }
    ]
}
```

**4b. Build Context Message**
```python
context_message = """
Focus on churn-related features

[Whiteboard]
Mission: Build customer churn prediction system
Current Focus: Data collection and pipeline setup
Your Role: Infrastructure & Data

Recent Team Updates:
  - [decision] Coordinator: Mission updated
"""
```

**4c. Execute with Context**
```python
response = await data_engineer.async_ask([
    UserMessage("Set up data pipeline for customer churn data"),
    DeveloperMessage(context_message)
])
# Response: "Data pipeline deployed successfully. Connected to customer database..."
```

**4d. Log Result**
```python
context.add_update(
    agent_name="Data Engineer",
    update_type=UpdateType.FINDING,
    content={
        "task": "Set up data pipeline for customer churn data",
        "response_summary": "Data pipeline deployed successfully...",
        "status": "completed"
    },
    tags=["worker_response", "data_engineer"]
)
```

#### 5. Worker 2 (Data Analyst) Executes

**5a. Get Context View**
```python
agent_view = context.get_agent_view("Data Analyst")
# Returns:
{
    "mission": "Build customer churn prediction system",
    "current_focus": "Data collection and pipeline setup",
    "my_role": "Analysis & Insights",
    "recent_updates": [
        {
            "agent_name": "Coordinator",
            "type": "decision",
            "content": {"action": "mission_updated", ...}
        },
        {
            "agent_name": "Data Engineer",
            "type": "finding",
            "content": {
                "task": "Set up data pipeline...",
                "response_summary": "Data pipeline deployed successfully...",
                "status": "completed"
            }
        }
    ]
}
```

**5b. Build Context Message**
```python
context_message = """
Identify key churn indicators

[Whiteboard]
Mission: Build customer churn prediction system
Current Focus: Data collection and pipeline setup
Your Role: Analysis & Insights

Recent Team Updates:
  - [decision] Coordinator: Mission updated
  - [finding] Data Engineer: Data pipeline deployed successfully...
"""
```

**Note**: Data Analyst now knows the pipeline is ready!

**5c. Execute and Log**
```python
response = await data_analyst.async_ask([...])
# Response: "Analysis complete. Key churn indicators: usage frequency < 2/week..."

context.add_update(
    agent_name="Data Analyst",
    update_type=UpdateType.FINDING,
    content={...},
    tags=["worker_response", "data_analyst"]
)
```

#### 6. Coordinator Tracks Progress
```python
# Coordinator calls update_progress_func
context.progress = "Data pipeline deployed, churn analysis complete"

context.add_update(
    agent_name="Coordinator",
    update_type=UpdateType.PROGRESS,
    content={
        "action": "progress_updated",
        "progress": "Data pipeline deployed, churn analysis complete"
    },
    tags=["coordinator", "progress"]
)
```

#### 7. Coordinator Returns Response
```python
# Coordinator aggregates worker responses
return """
I've initiated the customer churn prediction system:

1. Data Pipeline: Successfully deployed and connected to customer database
2. Analysis: Identified key churn indicators including usage frequency and support tickets

Next steps: Build the prediction model using these insights.
"""
```

### Final Context State
```python
context.mission = "Build customer churn prediction system"
context.current_focus = "Data collection and pipeline setup"
context.progress = "Data pipeline deployed, churn analysis complete"
context.updates = [
    ContextUpdate(agent_name="Coordinator", type="decision", ...),
    ContextUpdate(agent_name="Data Engineer", type="finding", ...),
    ContextUpdate(agent_name="Data Analyst", type="finding", ...),
    ContextUpdate(agent_name="Coordinator", type="progress", ...)
]
```

## Key Benefits

### 1. Automatic Coordination
- ✅ Coordinator sets mission → Workers know the goal
- ✅ Workers log findings → Other workers see progress
- ✅ Coordinator tracks blockers → Workers aware of issues
- ✅ No manual context passing needed

### 2. Team Awareness
- ✅ Each worker sees what others have done
- ✅ Workers can build on each other's work
- ✅ Workers avoid duplicate effort
- ✅ Workers stay aligned with team goals

### 3. Progress Visibility
- ✅ Clear audit trail of all work
- ✅ Easy to see what happened and when
- ✅ User can ask for status updates
- ✅ Coordinator can query team state

### 4. Backward Compatible
- ✅ Existing systems work without changes
- ✅ Tools automatically added to coordinators
- ✅ Workers get context automatically
- ✅ No breaking changes

## Usage

### Basic Setup
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
context = AgentSystemContext(
    memory_bank=AsyncAgentMemoryBank(),
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

# Define coordinator
coordinator = CoordinatorAgentBuilder(
    name="Team Lead",
    llm=LLM(model=GPTModels.GPT_4O_MINI),
    prompt_builder=lambda ctx: "You coordinate the team..."
)

# Build system
system = CoordinatorSystem(
    coordinator_builder=coordinator,
    workers=[data_engineer, data_analyst, ml_engineer],
    context=context
)

# Use the system - context management is automatic!
response = await system.async_ask("Build a recommendation system")

# Check Whiteboard
print(f"Mission: {context.mission}")
print(f"Progress: {context.progress}")
print(f"Updates: {len(context.updates)}")
```

### Advanced: Custom Context Management
```python
# Customize coordinator to emphasize context management
coordinator = CoordinatorAgentBuilder(
    name="Team Lead",
    llm=LLM(model=GPTModels.GPT_4O_MINI),
    prompt_builder=lambda ctx: """
    You coordinate the data science team.
    
    IMPORTANT: Always use context management tools:
    - Start every new project with update_mission_func
    - Track progress after each major milestone with update_progress_func
    - Log all important decisions with log_decision_func
    - Add blockers immediately when workers report issues
    - Use get_team_status_func when user asks for status
    """
)
```

## Files Modified

### Core Context
- ✅ `cortex/agent_system/core/context.py` - Added Whiteboard fields and methods
- ✅ `cortex/agent_system/core/__init__.py` - Exported ContextUpdate and UpdateType
- ✅ `cortex/agent_system/__init__.py` - Exported ContextUpdate and UpdateType

### Coordinator
- ✅ `cortex/agent_system/coordinator_system/coordinator_builder.py`
  - Added UpdateType import
  - Added create_coordinator_context_tools() function
  - Modified build_agent() to add context tools
  - Updated COORDINATOR_PROMPT with tool descriptions and workflow

### Workers
- ✅ `cortex/agent_system/coordinator_system/worker_builder.py`
  - Added UpdateType import
  - Modified install() to get context, include in message, and log results

### Documentation
- ✅ `CONTEXT_UPDATE_REFERENCE.md` - ContextUpdate model reference
- ✅ `WHITEBOARD_MIGRATION.md` - Migration guide
- ✅ `WHITEBOARD_ROUTING_CHANGES.md` - Worker routing details
- ✅ `COORDINATOR_CONTEXT_MANAGEMENT.md` - Coordinator tools guide
- ✅ `WHITEBOARD_COMPLETE.md` - This complete overview

### Examples
- ✅ `examples/context_update_example.py` - ContextUpdate usage
- ✅ `examples/whiteboard_example.py` - Whiteboard patterns
- ✅ `examples/coordinator_whiteboard_example.py` - Full team coordination

## Testing

See `examples/coordinator_whiteboard_example.py` for complete working examples.

## Next Steps

1. **Try the examples**: Run `examples/coordinator_whiteboard_example.py`
2. **Update your systems**: Add team_roles to your AgentSystemContext
3. **Monitor context**: Check context.updates to see team activity
4. **Customize**: Adjust coordinator prompts to emphasize context management

## Summary

The Whiteboard provides automatic team coordination through:
- **Coordinator**: Actively manages mission, progress, and blockers
- **Workers**: Automatically receive and contribute to the Whiteboard
- **Context**: Central state synchronized across all agents
- **Updates**: Audit trail of all team activity

No manual context passing needed - it all happens automatically! 🎉
