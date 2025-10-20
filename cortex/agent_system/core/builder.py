from asyncio import iscoroutine
from inspect import signature
from typing import Callable, Optional
from cortex import LLM

from .context import AgentSystemContext


class AgentBuilder:
    """Core base class to accumulate information needed to build an Agent and expose it as a Tool.

    Subclasses should implement build_agent(context) to construct and return an Agent instance
    using the runtime context (e.g., to fetch llm, memory, project info).

    The install() method returns an cortex.Tool that, when executed, builds the agent lazily
    and forwards the user's message (plus optional developer context) to the agent.
    """

    def __init__(
        self,
        *,
        name: str,
        llm: LLM,
        prompt_builder: Callable,
        tools_builder: Optional[Callable] = None,
        memory_k: Optional[int] = 5
    ):
        self.name = name
        self.name_key = name.lower().replace(" ", "_")
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.tools_builder = tools_builder
        self.memory_k = memory_k

    def install(self, *args, **kwargs):
        raise NotImplementedError

    async def load_tools(self, context: AgentSystemContext):
        '''
        Load tools for the agent.
        '''
        if self.tools_builder:
            # check if there's a context in the function signature
            sig = signature(self.tools_builder)
            if 'context' in sig.parameters or 'ctx' in sig.parameters or len(sig.parameters) > 0:
                res = self.tools_builder(context)
            else:
                res = self.tools_builder()
            
            if iscoroutine(res):
                return await res
            return res
        
        return []
    
    async def build_prompt(self, context: AgentSystemContext):
        '''
        Build the prompt for the agent.
        '''
        # check if there's a context in the function signature
        sig = signature(self.prompt_builder)
        if 'context' in sig.parameters or 'ctx' in sig.parameters or len(sig.parameters) > 0:
            p = self.prompt_builder(context)
        else:
            p = self.prompt_builder()
        
        if iscoroutine(p):
            return await p
        return p
