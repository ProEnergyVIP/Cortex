# ContextUpdate Model Reference

## Overview

The `ContextUpdate` Pydantic model provides a structured, type-safe way for agents to share updates in a multi-agent system. It uses an enum for update types and flexible dictionary content for extensibility.

## Model Definition

```python
from cortex.agent_system import ContextUpdate, UpdateType

class UpdateType(str, Enum):
    """Types of updates that can be made to the Whiteboard."""
    PROGRESS = "progress"
    FINDING = "finding"
    DECISION = "decision"
    ARTIFACT = "artifact"
    BLOCKER = "blocker"

class ContextUpdate(BaseModel):
    """Represents an update to the Whiteboard by an agent."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    agent_name: str
    type: UpdateType
    content: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    tags: List[str] = Field(default_factory=list)
```

## Fields

### `id: str`
- **Auto-generated** 8-character unique identifier
- Used to reference specific updates
- Generated from UUID

### `agent_name: str`
- Name of the agent creating the update
- Required field
- Used for filtering updates by agent

### `type: UpdateType`
- Enum defining the update category
- Required field
- Values: `PROGRESS`, `FINDING`, `DECISION`, `ARTIFACT`, `BLOCKER`

### `content: Dict[str, Any]`
- Flexible dictionary for update data
- Required field
- Structure depends on update type (see patterns below)

### `timestamp: datetime`
- Auto-generated timestamp
- When the update was created
- Used for time-based filtering

### `tags: List[str]`
- Optional categorization tags
- Default: empty list
- Used for filtering and organization

## Update Type Patterns

### PROGRESS
Track task or workflow progress.

```python
context.add_update(
    agent_name="worker_1",
    update_type=UpdateType.PROGRESS,
    content={
        "task": "data_processing",
        "stage": "cleaning",
        "percentage": 75,
        "records_processed": 7500,
        "records_total": 10000,
        "eta_minutes": 15
    },
    tags=["data", "processing"]
)
```

### FINDING
Share discoveries, insights, or analysis results.

```python
context.add_update(
    agent_name="analyst",
    update_type=UpdateType.FINDING,
    content={
        "title": "Customer Satisfaction Trend",
        "summary": "15% increase in Q4 satisfaction scores",
        "confidence": 0.95,
        "data_source": "Q4 2024 surveys",
        "key_metrics": {
            "satisfaction_score": 8.5,
            "response_rate": 0.82
        }
    },
    tags=["analysis", "positive", "customer"]
)
```

### DECISION
Document decisions made by agents.

```python
context.add_update(
    agent_name="coordinator",
    update_type=UpdateType.DECISION,
    content={
        "decision": "Adopt gradient boosting model",
        "rationale": "Best performance on validation set (AUC: 0.94)",
        "alternatives_considered": ["random_forest", "neural_network"],
        "approval_required": False,
        "decided_by": "ml_engineer",
        "effective_date": "2024-11-15"
    },
    tags=["modeling", "decision", "ml"]
)
```

### ARTIFACT
Track created artifacts (code, documents, models, etc.).

```python
context.add_update(
    agent_name="developer",
    update_type=UpdateType.ARTIFACT,
    content={
        "artifact_type": "code",
        "name": "data_pipeline.py",
        "location": "/src/pipelines/data_pipeline.py",
        "status": "completed",
        "version": "1.0.0",
        "size_bytes": 15420,
        "checksum": "abc123..."
    },
    tags=["code", "pipeline", "completed"]
)
```

### BLOCKER
Report issues blocking progress.

```python
context.add_update(
    agent_name="devops",
    update_type=UpdateType.BLOCKER,
    content={
        "blocker": "API rate limit exceeded",
        "severity": "high",  # low, medium, high, critical
        "impact": "Data ingestion paused",
        "affected_tasks": ["data_collection", "feature_extraction"],
        "estimated_resolution": "2 hours",
        "workaround": "Implement exponential backoff",
        "assigned_to": "infrastructure_team"
    },
    tags=["blocker", "infrastructure", "urgent"]
)
```

## Usage Examples

### Creating Updates

```python
from cortex.agent_system import AgentSystemContext, UpdateType

# Create context
context = AgentSystemContext(memory_bank=memory_bank)

# Add an update (returns the created ContextUpdate)
update = context.add_update(
    agent_name="my_agent",
    update_type=UpdateType.PROGRESS,
    content={"task": "analysis", "percentage": 50},
    tags=["analysis"]
)

print(f"Created update with ID: {update.id}")
```

### Filtering Updates

```python
# Get all progress updates
progress_updates = context.get_recent_updates(
    update_type=UpdateType.PROGRESS
)

# Get updates from specific agent
agent_updates = context.get_recent_updates(
    agent_name="analyst"
)

# Get recent updates (last hour)
from datetime import datetime, timedelta
one_hour_ago = datetime.now() - timedelta(hours=1)
recent = context.get_recent_updates(since=one_hour_ago)

# Combine filters
recent_findings = context.get_recent_updates(
    agent_name="analyst",
    update_type=UpdateType.FINDING,
    since=one_hour_ago
)
```

### Getting Agent View

```python
# Get filtered view for specific agent
view = context.get_agent_view("my_agent")

# Access recent updates
for update in view['recent_updates']:
    print(f"[{update['type']}] {update['agent_name']}: {update['content']}")
```

### Direct ContextUpdate Creation

```python
from cortex.agent_system import ContextUpdate, UpdateType

# Create directly (not added to context automatically)
update = ContextUpdate(
    agent_name="test_agent",
    type=UpdateType.FINDING,
    content={"result": "important discovery"},
    tags=["test", "discovery"]
)

# Manually add to context if needed
context.updates.append(update)
```

## Serialization

ContextUpdate is fully serializable thanks to Pydantic:

```python
# To dictionary
update_dict = update.dict()
# {'id': 'a1b2c3d4', 'agent_name': 'test', 'type': 'finding', ...}

# To JSON
update_json = update.json()
# '{"id": "a1b2c3d4", "agent_name": "test", "type": "finding", ...}'

# From dictionary
reconstructed = ContextUpdate(**update_dict)

# From JSON
import json
reconstructed = ContextUpdate(**json.loads(update_json))
```

Note: The `UpdateType` enum is automatically serialized as its string value (e.g., `"progress"`) due to `Config.use_enum_values = True`.

## Enum Usage

```python
from cortex.agent_system import UpdateType

# Access enum values
UpdateType.PROGRESS  # UpdateType.PROGRESS
UpdateType.FINDING   # UpdateType.FINDING

# Get string value
UpdateType.PROGRESS.value  # "progress"

# Get enum name
UpdateType.PROGRESS.name   # "PROGRESS"

# Iterate all types
for update_type in UpdateType:
    print(f"{update_type.name}: {update_type.value}")

# Compare
if update.type == UpdateType.PROGRESS:
    print("This is a progress update")
```

## Best Practices

### 1. Use Consistent Content Structure
Define standard schemas for each update type in your application:

```python
# Good: Consistent structure
PROGRESS_SCHEMA = {
    "task": str,
    "percentage": int,
    "status": str
}

# Use it
context.add_update(
    agent_id="worker",
    update_type=UpdateType.PROGRESS,
    content={
        "task": "data_cleaning",
        "percentage": 75,
        "status": "in_progress"
    }
)
```

### 2. Use Descriptive Tags
Tags help with filtering and organization:

```python
# Good: Specific, hierarchical tags
tags=["data", "processing", "etl", "completed"]

# Avoid: Too generic
tags=["update", "info"]
```

### 3. Include Context in Content
Make updates self-contained:

```python
# Good: Includes context
content={
    "task": "model_training",
    "model_type": "gradient_boosting",
    "dataset": "customer_data_v2",
    "accuracy": 0.94
}

# Avoid: Requires external context
content={"accuracy": 0.94}  # What model? What data?
```

### 4. Use Appropriate Update Types
Choose the right type for your update:

- **PROGRESS**: Ongoing work status
- **FINDING**: Completed analysis or discovery
- **DECISION**: Choice made that affects workflow
- **ARTIFACT**: Created output or deliverable
- **BLOCKER**: Issue preventing progress

### 5. Handle Blockers Promptly
When adding blockers, include actionable information:

```python
context.add_update(
    agent_name="worker",
    update_type=UpdateType.BLOCKER,
    content={
        "blocker": "Database connection timeout",
        "severity": "high",
        "impact": "Cannot process new records",
        "workaround": "Use cached data for now",
        "assigned_to": "devops_team",
        "ticket_id": "INFRA-1234"
    },
    tags=["blocker", "database", "urgent"]
)
```

## Integration with Tools

Create tools that log updates:

```python
from cortex import Tool
from cortex.agent_system import UpdateType

async def log_progress_func(args, ctx: AgentSystemContext):
    """Tool for agents to log progress."""
    update = ctx.add_update(
        agent_name=args["agent_name"],
        update_type=UpdateType.PROGRESS,
        content={
            "task": args["task"],
            "percentage": args["percentage"],
            "status": args.get("status", "in_progress")
        },
        tags=args.get("tags", [])
    )
    return f"Progress logged: {update.id}"

log_progress_tool = Tool(
    name="log_progress",
    func=log_progress_func,
    description="Log progress on a task",
    parameters={
        "type": "object",
        "properties": {
            "agent_name": {"type": "string"},
            "task": {"type": "string"},
            "percentage": {"type": "integer", "minimum": 0, "maximum": 100},
            "status": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["agent_name", "task", "percentage"]
    }
)
```

## See Also

- `AgentSystemContext` - Main context class
- `examples/context_update_example.py` - Comprehensive examples
- `examples/whiteboard_example.py` - Multi-agent coordination examples
