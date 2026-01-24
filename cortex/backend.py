from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type
from enum import Enum
import fnmatch

from cortex.message import AIMessage, SystemMessage


class ReasoningEffort(Enum):
    """Level of reasoning effort to request from the model.

    This generally maps to provider settings like OpenAI's
    reasoning effort (e.g., "low", "medium", "high").
    """
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class LLMRequest:
    system_message: SystemMessage
    messages: list
    temperature: Optional[float] = None
    max_tokens: int = None
    tools: list = field(default_factory=list)
    reasoning_effort: Optional[ReasoningEffort] = None

class LLMBackend:
    backend_registry = {}
    backend_instance_cache = {}

    def __init__(self) -> None:
        # Map of Message subclass -> encoder function
        # Encoder returns either a single backend-native payload (dict) or a list of them
        self._message_encoders: Dict[Type, Callable[[object], Any]] = {}
        # Map of Tool subclass -> encoder function
        # Encoder returns a single backend-native tool payload (dict)
        self._tool_encoders: Dict[Type, Callable[[object], Any]] = {}

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

    # --- Tool encoding registry API ---
    def register_tool_encoder(self, tool_type: Type, encoder: Callable[[object], Dict[str, Any]]) -> None:
        """Register an encoder for a specific Tool subclass.

        The encoder should accept a Tool instance and return a single backend-native tool payload (dict).
        """
        self._tool_encoders[tool_type] = encoder

    def _find_tool_encoder_for(self, tool: object) -> Callable[[object], Dict[str, Any]] | None:
        """Find the most specific registered encoder for the given tool instance.
        Walk the class MRO to allow subclass matching.
        """
        for cls in type(tool).mro():
            enc = self._tool_encoders.get(cls)
            if enc is not None:
                return enc
        return None

    def default_tool_encoder(self, tool: object) -> Dict[str, Any]:
        """Fallback encoder if no specific tool encoder is registered.
        Assumes a function-style tool with name/description/parameters.
        """
        name = getattr(tool, 'name', None)
        description = getattr(tool, 'description', None)
        parameters = getattr(tool, 'parameters', None)
        if name and description is not None and parameters is not None:
            return {
                'type': 'function',
                'strict': tool.strict,
                'name': name,
                'description': description,
                'parameters': parameters,
            }
        # As a last resort, return an empty function tool with a generic name
        return {
            'type': 'function',
            'strict': tool.strict,
            'name': name or 'unknown_tool',
            'description': description or 'No description provided.',
            'parameters': parameters or {'type': 'object', 'properties': {}, 'additionalProperties': True},
        }

    def encode_tool(self, tool: object) -> Dict[str, Any]:
        """Encode a Tool instance using the registered tool encoder or fallback."""
        encoder = self._find_tool_encoder_for(tool)
        return encoder(tool) if encoder else self.default_tool_encoder(tool)

    def call(self, req: LLMRequest) -> AIMessage | None:
        return None
    
    async def async_call(self, req: LLMRequest) -> AIMessage | None:
        """Async version of call method. By default falls back to synchronous call."""
        return None

    @classmethod
    def get_backend(cls, model):
        """Get a backend instance for the given model.
        The backend instance will be cached for future use.

        Args:
            model: The model identifier this backend handles
        """
        model_key = str(model)

        if model_key in cls.backend_instance_cache:
            return cls.backend_instance_cache[model_key]
        
        backend_cls = cls.backend_registry.get(model_key, None)
        if backend_cls is None:
            # Fallback to wildcard/pattern matches (exact match always preferred).
            # Patterns are registered as strings (e.g. "gpt-*")
            for pattern, candidate_backend_cls in cls.backend_registry.items():
                print(f'pattern: {pattern}')
                print(f'model_key: {model_key}')
                print(f'matches: {fnmatch.fnmatchcase(model_key, pattern)}')
                if isinstance(pattern, str) and fnmatch.fnmatchcase(model_key, pattern):
                    backend_cls = candidate_backend_cls
                    break
        if backend_cls is None:
            return None
        
        backend = backend_cls(model_key)
        cls.backend_instance_cache[model_key] = backend

        return backend
    
    @classmethod
    def register_backend(cls, model, backend_cls):
        """Register a backend class for a model.
        
        Args:
            model: The model identifier this backend handles
            backend_cls: The backend class to instantiate
        """
        cls.backend_registry[str(model)] = backend_cls
