"""Vector store implementations for similarity search and storage."""

# Import the main classes for easier access
from .base import VectorStore, VectorSearchResult
from .memory import InMemoryVectorStore
from .factory import VectorStoreType, get_vector_store

__all__ = [
    'VectorStore',
    'VectorSearchResult',
    'InMemoryVectorStore',
    'VectorStoreType',
    'get_vector_store'
]
