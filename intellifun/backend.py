from dataclasses import dataclass, field

from intellifun.message import AIMessage, SystemMessage


@dataclass
class LLMRequest:
    system_message: SystemMessage
    messages: list
    temperature: float = 0.5
    max_tokens: int = None
    tools: list = field(default_factory=list)

class LLMBackend:
    backend_registry = {}

    def call(self, req: LLMRequest) -> AIMessage | None:
        return None
    
    async def async_call(self, req: LLMRequest) -> AIMessage | None:
        """Async version of call method. By default falls back to synchronous call."""
        return None

    @classmethod
    def get_backend(cls, model):
        return cls.backend_registry.get(model, None)
    
    @classmethod
    def register_backend(cls, model, backend):
        cls.backend_registry[model] = backend
