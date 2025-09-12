from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Type

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

    def __init__(self) -> None:
        # Map of Message subclass -> encoder function
        # Encoder returns either a single backend-native payload (dict) or a list of them
        self._message_encoders: Dict[Type, Callable[[object], Any]] = {}

    # --- Message encoding registry API ---
    def register_message_encoder(self, message_type: Type, encoder: Callable[[object], Any]) -> None:
        """Register an encoder for a specific Message subclass.

        The encoder should accept a Message instance and return either:
        - a single backend-native payload (dict), or
        - a list of such payloads when one Message expands to multiple entries
        """
        self._message_encoders[message_type] = encoder

    def _find_encoder_for(self, msg: object) -> Callable[[object], Any] | None:
        """Find the most specific registered encoder for the given message instance.
        Walk the class MRO to allow subclass matching.
        """
        for cls in type(msg).mro():
            enc = self._message_encoders.get(cls)
            if enc is not None:
                return enc
        return None

    def default_message_encoder(self, msg: object) -> Dict[str, Any]:
        """Fallback encoder if no specific encoder is registered.
        Default to a user message with plain content.
        """
        # Avoid importing Message here to prevent circular deps; duck-typing on 'content'.
        content = getattr(msg, 'content', str(msg))
        return {"role": "user", "content": content}

    def encode_message(self, msg: object) -> List[Dict[str, Any]]:
        """Encode a Message instance using the registered encoder.

        Always returns a list of backend-native payloads. If the specific encoder
        returns a single payload, it will be wrapped into a list.
        """
        encoder = self._find_encoder_for(msg)
        payload = encoder(msg) if encoder else self.default_message_encoder(msg)
        if isinstance(payload, list):
            return payload  # type: ignore[return-value]
        return [payload]  # type: ignore[list-item]

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
