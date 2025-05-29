"""Factory for creating vector store instances."""
from enum import Enum
from typing import Any, Union

from .base import VectorStore
from .memory import InMemoryVectorStore


class VectorStoreType(str, Enum):
    """Supported vector store types."""
    MEMORY = "memory"  # In-memory store (for testing/development)
    REDIS = "redis"    # Redis with RediSearch
    PINECONE = "pinecone"  # Pinecone vector database
    PGVECTOR = "pgvector"  # PostgreSQL with pgvector
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
    
    # Add other store types as they are implemented
    elif store_type == VectorStoreType.REDIS:
        raise NotImplementedError("Redis vector store is not yet implemented")
    elif store_type == VectorStoreType.PINECONE:
        raise NotImplementedError("Pinecone vector store is not yet implemented")
    elif store_type == VectorStoreType.PGVECTOR:
        raise NotImplementedError("PGVector store is not yet implemented")
    elif store_type == VectorStoreType.CHROMA:
        raise NotImplementedError("Chroma vector store is not yet implemented")
    elif store_type == VectorStoreType.MILVUS:
        raise NotImplementedError("Milvus vector store is not yet implemented")
    elif store_type == VectorStoreType.WEAVIATE:
        raise NotImplementedError("Weaviate vector store is not yet implemented")
    else:
        raise ValueError(f"Unsupported vector store type: {store_type}")
