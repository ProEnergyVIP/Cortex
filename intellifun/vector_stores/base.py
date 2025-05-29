"""Base classes for vector store implementations."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class VectorSearchResult:
    """Result of a vector similarity search.
    
    Attributes:
        id: Unique identifier of the stored vector
        content: The original text content
        metadata: Additional metadata associated with the vector
        score: Similarity score (higher is more similar)
        vector: Optional, the actual vector if requested
    """
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    vector: Optional[np.ndarray] = None


class VectorStore(ABC):
    """Abstract base class for vector storage and retrieval.
    
    All vector store implementations should inherit from this class and implement
    the required methods.
    """
    
    @abstractmethod
    async def add(
        self,
        texts: List[str],
        vectors: List[np.ndarray],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add texts and their vector embeddings to the store.
        
        Args:
            texts: List of text contents to store
            vectors: List of vector embeddings corresponding to the texts
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of IDs for the texts. If not provided, will be auto-generated.
            
        Returns:
            List of IDs for the stored texts
        """
        pass
    
    @abstractmethod
    async def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        include_vectors: bool = False
    ) -> List[VectorSearchResult]:
        """Search for similar vectors in the store.
        
        Args:
            query_vector: The query vector to search with
            k: Number of results to return
            filter: Optional filter to apply to metadata
            include_vectors: Whether to include vectors in the results
            
        Returns:
            List of VectorSearchResult objects sorted by relevance (highest first)
        """
        pass
    
    @abstractmethod
    async def delete(self, ids: List[str]) -> bool:
        """Delete entries by their IDs.
        
        Args:
            ids: List of IDs to delete
            
        Returns:
            True if deletion was successful
        """
        pass
    
    @abstractmethod
    async def get(self, ids: List[str], include_vectors: bool = False) -> List[Optional[Dict[str, Any]]]:
        """Retrieve entries by their IDs.
        
        Args:
            ids: List of IDs to retrieve
            include_vectors: Whether to include vectors in the results
            
        Returns:
            List of dictionaries containing the stored data (or None for missing IDs)
        """
        pass
    
    async def __aenter__(self):
        """Support async context manager."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Support async context manager."""
        pass
