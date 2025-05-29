"""Factory for creating vector store instances."""
import logging
from enum import Enum
from typing import Any, Union

from .base import VectorStore
from .memory import InMemoryVectorStore

try:
    from .chroma_store import ChromaVectorStore
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    ChromaVectorStore = None

logger = logging.getLogger(__name__)


class VectorStoreType(str, Enum):
    """Supported vector store types."""
    MEMORY = "memory"  # In-memory store (for testing/development)
    PINECONE = "pinecone"  # Pinecone vector database
    CHROMA = "chroma"     # Chroma vector database
    MILVUS = "milvus"     # Milvus vector database
    WEAVIATE = "weaviate"  # Weaviate vector database


def get_vector_store(
    store_type: Union[str, VectorStoreType],
    **kwargs: Any
) -> VectorStore:
    """Get a vector store instance by type.
    
    Args:
        store_type: Type of vector store to create
        **kwargs: Additional arguments to pass to the vector store constructor
        
    Returns:
        An instance of the requested vector store
        
    Raises:
        ValueError: If the store type is not supported
    """
    if isinstance(store_type, str):
        try:
            store_type = VectorStoreType(store_type.lower())
        except ValueError as e:
            raise ValueError(
                f"Unknown vector store type: {store_type}. "
                f"Available types: {', '.join(t.value for t in VectorStoreType)}"
            ) from e
    
    if store_type == VectorStoreType.MEMORY:
        return InMemoryVectorStore(**kwargs)
    elif store_type == VectorStoreType.CHROMA:
        if not CHROMA_AVAILABLE:
            raise ImportError(
                "ChromaDB is not installed. Please install it with: pip install chromadb"
            )
        # ChromaDB specific parameter validation
        if "persist_directory" not in kwargs:
            logger.warning(
                "No persist_directory provided for ChromaDB. "
                "Data will be stored in memory only and will not persist between sessions."
            )
        return ChromaVectorStore(**kwargs)
    
    # Add other store types as they are implemented
    elif store_type == VectorStoreType.PINECONE:
        raise NotImplementedError("Pinecone vector store is not yet implemented")
    elif store_type == VectorStoreType.MILVUS:
        raise NotImplementedError("Milvus vector store is not yet implemented")
    elif store_type == VectorStoreType.WEAVIATE:
        raise NotImplementedError("Weaviate vector store is not yet implemented")
    else:
        raise ValueError(f"Unsupported vector store type: {store_type}")
