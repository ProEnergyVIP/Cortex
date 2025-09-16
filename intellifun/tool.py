from dataclasses import dataclass, field
from functools import cached_property
import inspect

# Base marker class for any tool
@dataclass
class BaseTool:
    """Abstract marker for any tool (function tools and hosted tools).

    This base class intentionally carries no fields. Each subclass defines
    its own configuration. Only FunctionTool executes locally in the Agent.
    """
    pass


# Function tool (locally executable)
@dataclass
class FunctionTool(BaseTool):
    """A locally executable function tool.

    Backward compatible replacement for the old Tool class.
    """
    name: str
    func: callable
    description: str
    parameters: dict
    prompt: str = None
    strict: bool = True

    __called_times = 0

    def check_call_limit(self, limit=10):
        '''Check if the tool has been called too many times'''
        if self.__called_times >= limit:
            return False
        return True
    
    def increment_call_count(self):
        '''Increment the call count of the tool'''
        self.__called_times += 1

    @cached_property
    def is_async(self):
        '''Check if the tool is async'''
        return inspect.iscoroutinefunction(self.func)

    @cached_property
    def is_sync(self):
        '''Check if the tool is sync'''
        return not self.is_async

    async def async_run(self, tool_input, context, agent):
        '''Run the tool function asynchronously
        
        This method will handle both async and sync functions correctly:
        - If the function is async, it will be awaited
        - If the function is sync, it will be run in a thread pool
        
        Returns:
            The result of the function call
        '''
        self.increment_call_count()
        
        # Check the number of parameters the function expects
        sig = inspect.signature(self.func)
        num_params = len(sig.parameters)

        if num_params == 0:
            args = []
        elif num_params == 1:
            args = [tool_input]
        elif num_params == 2:
            args = [tool_input, context]
        elif num_params == 3:
            args = [tool_input, context, agent]
        else:
            raise ValueError(f"Tool function {self.name} expects 0, 1, 2, or 3 parameters but received {num_params} parameters")

        if self.is_async:
            # If the function is already async, just await it
            return await self.func(*args)
        else:
            # If the function is sync, run it in a thread pool
            import asyncio
            return await asyncio.to_thread(self.func, *args)
    
    def run(self, tool_input, context, agent):
        '''Run the tool function synchronously
        
        This method will handle both async and sync functions correctly:
        - If the function is sync, it will be called directly
        - If the function is async, it will be run in an event loop
        
        Returns:
            The result of the function call
        '''
        self.increment_call_count()
        
        # Check the number of parameters the function expects
        sig = inspect.signature(self.func)
        num_params = len(sig.parameters)

        if num_params == 0:
            args = []
        elif num_params == 1:
            args = [tool_input]
        elif num_params == 2:
            args = [tool_input, context]
        elif num_params == 3:
            args = [tool_input, context, agent]
        else:
            raise ValueError(f"Tool function {self.name} expects 0, 1, 2, or 3 parameters but received {num_params} parameters")
        
        if self.is_sync:
            # If the function is sync, just call it
            return self.func(*args)
        else:
            # If the function is async, run it in an event loop
            import asyncio
            
            # Get or create an event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # If there's no event loop in this thread, create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the async function in the event loop
            if loop.is_running():
                # If the loop is already running, we need to create a new one
                # This is a bit of a hack, but it's the best we can do
                return asyncio.run_coroutine_threadsafe(
                    self.func(*args), loop
                ).result()
            else:
                # If the loop is not running, we can just run the coroutine
                return loop.run_until_complete(self.func(*args))


# Web Search tool
@dataclass
class WebSearchUserLocation:
    city: str | None = None
    country: str | None = None
    region: str | None = None
    timezone: str | None = None
    # Per spec, this is always an approximation; default to 'approximate'
    type: str | None = 'approximate'


@dataclass
class WebSearchFilters:
    # Allowed domains for the search; if empty, all domains are allowed
    allowed_domains: list[str] = field(default_factory=list)


@dataclass
class WebSearchTool(BaseTool):
    # Optional filters object passed through to the provider
    filters: WebSearchFilters | None = None
    # Guidance for how much context window to use: one of 'low', 'medium', 'high'
    search_context_size: str | None = 'medium'
    # Approximate user location information
    user_location: WebSearchUserLocation | None = None


# Code Interpreter tool
@dataclass
class CodeInterpreterContainerAuto:
    # Configuration for an auto-managed code interpreter container
    # Always type 'auto' per spec; optional list of uploaded file IDs
    type: str = 'auto'
    file_ids: list[str] | None = None


@dataclass
class CodeInterpreterTool(BaseTool):
    # Required container: either a container ID (string) or an auto container config
    container: str | CodeInterpreterContainerAuto


# MCP tool
@dataclass
class MCPToolsFilter:
    # Filter object to specify which tools are allowed or require approval
    read_only: bool | None = None
    tool_names: list[str] | None = None


@dataclass
class MCPApprovalFilter:
    # Approval policy can be specified as filters for 'always' and/or 'never'
    always: MCPToolsFilter | None = None
    never: MCPToolsFilter | None = None


@dataclass
class MCPTool(BaseTool):
    # Required label identifying this MCP server
    server_label: str
    # One of server_url or connector_id must be provided
    server_url: str | None = None
    connector_id: str | None = None
    # Optional OAuth access token
    authorization: str | None = None
    # Optional HTTP headers for the MCP server
    headers: dict | None = None
    # Allowed tools can be a list of names or a filter object
    allowed_tools: list[str] | MCPToolsFilter | None = None
    # Require approval can be 'always' | 'never' (string), a filter object, or an approval filter object
    require_approval: str | MCPApprovalFilter | None = None
    # Optional server description
    server_description: str | None = None


# File Search tool
@dataclass
class FileSearchRankingOptions:
    ranker: str | None = None
    score_threshold: float | None = None  # 0.0 - 1.0


@dataclass
class FileSearchTool(BaseTool):
    # Required: IDs of the vector stores to search
    vector_store_ids: list[str]
    # Optional filter object (pass-through)
    filters: dict | None = None
    # Optional: 1..50
    max_num_results: int | None = None
    # Optional ranking options
    ranking_options: FileSearchRankingOptions | None = None


# Backward compatibility export
Tool = FunctionTool
