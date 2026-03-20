# Simplified Workflow Design Proposal

## 1. Simplified State Contract

### Current EngineState fields (audit results)
```python
class EngineState:
    input: Any = None              # Keep: original workflow input
    data: dict[str, Any] = field(default_factory=dict)  # Keep: user-mutable bag
    context: Any = None            # Keep: shared runtime context
    usage: Any = None              # Keep: usage tracking
    memory: Any = None             # Keep: conversation memory
    last_output: Any = None        # Keep: last node output
    final_output: Any = None       # Keep: terminal workflow output
    current_node: Optional[str] = None  # Keep: execution tracking
    completed_nodes: list[str] = field(default_factory=list)  # Keep: execution tracking
    metadata: dict[str, Any] = field(default_factory=dict)  # Keep: engine metadata
```

### Proposed simplified EngineState
```python
@dataclass
class EngineState:
    """Minimal workflow state with only essential engine fields."""
    
    # User-mutable data bag (the only field user code should modify)
    data: dict[str, Any] = field(default_factory=dict)
    
    # Essential engine fields (read-only for user code)
    context: Any = None
    usage: Any = None
    memory: Any = None
    current_node: Optional[str] = None
    completed_nodes: list[str] = field(default_factory=list)
    
    # Derived fields that can be computed from data
    @property
    def input(self) -> Any:
        return self.data.get("input")
    
    @property
    def last_output(self) -> Any:
        return self.data.get("_last_output")
    
    @property
    def final_output(self) -> Any:
        return self.data.get("_final_output")
    
    # Engine-only methods
    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)
    
    def update(self, updates: dict[str, Any]) -> None:
        self.data.update(updates)
```

### Key simplifications
- **Removed**: `input`, `last_output`, `final_output`, `metadata` as top-level fields
- **Derived**: These values are now stored in `state.data` with reserved keys:
  - `"input"` for original workflow input
  - `"_last_output"` for last node output  
  - `"_final_output"` for terminal workflow output
- **User contract**: Only `state.data` is mutable; all other fields are engine-managed

## 2. Function-Only Node Contract

### Current node contract
```python
async def run(self, user_input: Any = None, *, context: Any = None, state: Any = None, workflow: Any = None) -> WorkflowNodeResult
```

### Proposed simplified node contract
```python
# Node function signature
async def node_function(data: dict[str, Any], context: Any) -> dict[str, Any]:
    """
    Process workflow data and return updates.
    
    Args:
        data: Current state.data dictionary (mutable copy)
        context: Shared runtime context
        
    Returns:
        Dictionary of updates to merge into state.data
    """
    # Example implementation
    result = process_data(data, context)
    return {"result": result, "_last_output": result}
```

### Engine-side execution
```python
# Engine calls node functions like this:
async def run_node(node_func, state):
    # Create a mutable copy of current data
    current_data = dict(state.data)
    
    # Call node function
    updates = await node_function(current_data, state.context)
    
    # Merge updates back into state
    state.data.update(updates)
    
    # Track execution
    state.completed_nodes.append(node.name)
    state.current_node = None
    
    return WorkflowNodeResult(updates=updates)
```

## 3. Runnable Removal Plan

### Current Runnable abstractions to remove
- `RunnableNode` class
- `RunnableAdapter` class
- `FunctionRunnable` class
- `AskCapableRunnableLike`, `RunCapableRunnableLike`, `RunnableLike` protocols
- `RunnableInvocation` result wrapper
- `invoke_runnable` function
- `adapt_runnable`, `function_runnable` helpers
- All runnable-related imports in `__init__.py`

### Migration strategy
1. Replace `RunnableNode` with `FunctionNode` that wraps plain functions
2. Update `llm_node` and `function_node` helpers to use the new contract
3. Remove `runtime.py` entirely (all runnable adaptation utilities)
4. Update `parallel_node` to work with function nodes instead of runnable nodes

## 4. State Update Merging Logic

### Before node execution
```python
# Engine ensures reserved keys are available
if "input" not in state.data:
    state.data["input"] = user_input
```

### After node execution
```python
# Engine merges node updates with conflict resolution
def merge_updates(state_data: dict, updates: dict) -> dict:
    """
    Merge node updates into state.data with reserved key handling.
    
    Reserved keys:
    - "input": never overwritten (original workflow input)
    - "_last_output": always overwritten (last node output)
    - "_final_output": only overwritten by terminal nodes
    """
    merged = dict(state_data)
    
    for key, value in updates.items():
        if key == "input":
            # Never overwrite original input
            continue
        elif key == "_final_output":
            # Only allow final nodes to set this
            merged[key] = value
        else:
            # Normal merge for all other keys including "_last_output"
            merged[key] = value
    
    return merged
```

## 5. Implementation Steps

1. **Implement simplified state**
   - Update `EngineState` and `WorkflowStateProtocol`
   - Add property getters for derived fields
   - Update serialization/cloning

2. **Implement function-only nodes**
   - Create `FunctionNode` class
   - Update node execution contract
   - Update helpers to use new contract

3. **Remove runnable abstractions**
   - Delete `runtime.py`
   - Remove `RunnableNode`
   - Update imports and exports

4. **Implement state merging**
   - Add merge logic to engine
   - Handle reserved keys properly
   - Update node result handling

5. **Verify and migrate examples**
   - Update examples to use new contract
   - Run compilation tests
   - Update documentation
