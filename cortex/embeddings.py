from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union
from abc import ABC, abstractmethod
from threading import Lock
import numpy as np
import logging

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class EmbeddingRequest:
    """Request for generating embeddings"""
    text: Union[str, List[str]]  # Input text or list of texts to embed
    model: str  # Model to use for embedding
    metadata: Optional[Dict[str, Any]] = None  # Optional metadata

class EmbeddingBackend(ABC):
    """Base class for all embedding backends"""
    
    @abstractmethod
    def embed(self, request: EmbeddingRequest) -> Union[np.ndarray, List[np.ndarray]]:
        """Generate embedding(s) synchronously"""
        pass
    
    @abstractmethod
    async def async_embed(self, request: EmbeddingRequest) -> Union[np.ndarray, List[np.ndarray]]:
        """Generate embedding(s) asynchronously"""
        pass

class Embedding:
    """Main embedding class following similar patterns to LLM class"""
    
    _backup_backends = {}  # Maps model -> backup_model
    _failed_models = set()  # Set of models that have failed
    _runtime_lock = Lock()  # Lock for protecting runtime state
    backend_registry = {}  # Registry for backends

    @staticmethod
    def _normalize_model(model: Any) -> str:
        """Normalize model identifiers (e.g., Enum values) into plain strings."""
        if isinstance(model, str):
            return model
        value = getattr(model, "value", None)
        if value is not None:
            return str(value)
        return str(model)
    
    def __init__(self, model: str, **kwargs):
        """
        Initialize the Embedding instance.
        
        Args:
            model: The model identifier to use
            **kwargs: Additional arguments to pass to the backend
        """
        self.model = self._normalize_model(model)
        self.kwargs = kwargs
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize the backend using the effective model"""
        effective_model = self._get_effective_model(self._normalize_model(self.model))
        backend_class = self.backend_registry.get(effective_model)
        if not backend_class:
            raise ValueError(f"No backend registered for model: {effective_model}")
        self.backend = backend_class(**self.kwargs)
    
    @classmethod
    def _get_effective_model(cls, model: str) -> str:
        """Get the effective model to use, considering failures and backups"""
        model = cls._normalize_model(model)
        if model in cls._failed_models and model in cls._backup_backends:
            backup = cls._backup_backends[model]
            logger.warning(f"Model {model} failed, falling back to {backup}")
            return backup
        return model
    
    @classmethod
    def set_backup_backend(cls, model: str, backup_model: str) -> None:
        """Set a backup backend model to use if the primary model fails.
        
        Args:
            model: The primary model identifier
            backup_model: The backup model identifier to use if primary fails
            
        Raises:
            ValueError: If setting this backup would create a cycle in the backup chain
        """
        model = cls._normalize_model(model)
        backup_model = cls._normalize_model(backup_model)
        if model == backup_model:
            raise ValueError("Cannot set a model as its own backup")
        cls._backup_backends[model] = backup_model
    
    @classmethod
    def reset_failed_models(cls) -> None:
        """Reset the failed models state, allowing previously failed models to be tried again."""
        with cls._runtime_lock:
            cls._failed_models.clear()
    
    @classmethod
    def register_backend(cls, model: str, backend_class: type) -> None:
        """Register a new backend for a model.
        
        Args:
            model: The model identifier this backend handles
            backend_class: The backend class to instantiate
        """
        model = cls._normalize_model(model)
        cls.backend_registry[model] = backend_class
    
    def embed(self, text: Union[str, List[str]], metadata: Optional[Dict] = None) -> np.ndarray:
        """Generate embedding(s) synchronously.
        
        Args:
            text: Input text or list of texts to embed
            metadata: Optional metadata to include with the request
            
        Returns:
            numpy.ndarray: For single text input, returns a 1D array.
                          For multiple texts, returns a 2D array where each row is an embedding.
        """
        request = EmbeddingRequest(
            text=text,
            model=self.model,
            metadata=metadata
        )
        
        try:
            result = self.backend.embed(request)
            return np.array(result) if not isinstance(result, np.ndarray) else result
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            self._handle_failure()
            return self.embed(text, metadata)  # Retry with backup if available
    
    async def async_embed(self, text: Union[str, List[str]], metadata: Optional[Dict] = None) -> np.ndarray:
        """Generate embedding(s) asynchronously.
        
        Args:
            text: Input text or list of texts to embed
            metadata: Optional metadata to include with the request
            
        Returns:
            numpy.ndarray: For single text input, returns a 1D array.
                          For multiple texts, returns a 2D array where each row is an embedding.
        """
        request = EmbeddingRequest(
            text=text,
            model=self.model,
            metadata=metadata
        )
        
        try:
            result = await self.backend.async_embed(request)
            return np.array(result) if not isinstance(result, np.ndarray) else result
        except Exception as e:
            logger.error(f"Error generating async embeddings: {e}")
            self._handle_failure()
            return await self.async_embed(text, metadata)  # Retry with backup if available
    
    def _handle_failure(self):
        """Handle model failure by switching to backup if available"""
        with self._runtime_lock:
            if self.model not in self._failed_models and self.model in self._backup_backends:
                logger.warning(f"Marking model {self.model} as failed")
                self._failed_models.add(self.model)
                self._initialize_backend()
