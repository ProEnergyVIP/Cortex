"""In-memory implementation of the vector store.

This is useful for testing and development, but not suitable for production use.
"""
from typing import Dict, List, Optional, Any
import uuid
import numpy as np
from .base import VectorStore, VectorSearchResult


class InMemoryVectorStore(VectorStore):
    """In-memory implementation of the vector store.
    
    This implementation uses simple Python data structures and is not persistent.
    It's useful for testing and development purposes.
    """
    
    def __init__(self, **kwargs):
        """Initialize the in-memory vector store."""
        self._store: Dict[str, Dict[str, Any]] = {}
        self._vectors: Dict[str, np.ndarray] = {}
    
    async def add(
        self,
        texts: List[str],
        vectors: List[np.ndarray],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add texts and their vector embeddings to the store."""
        if len(texts) != len(vectors):
            raise ValueError("Number of texts must match number of vectors")
        
        if metadatas is None:
            metadatas = [{} for _ in texts]
        elif len(metadatas) != len(texts):
            raise ValueError("Number of metadatas must match number of texts")
        
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        elif len(ids) != len(texts):
            raise ValueError("Number of ids must match number of texts")
        
        for idx, (text, vector, metadata, doc_id) in enumerate(zip(texts, vectors, metadatas, ids)):
            if not isinstance(vector, np.ndarray):
                raise ValueError(f"Vector at index {idx} is not a numpy array")
                
            self._store[doc_id] = {
                'text': text,
                'metadata': metadata or {}
            }
            self._vectors[doc_id] = np.array(vector, dtype=np.float32)
        
        return ids
    
    async def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        include_vectors: bool = False
    ) -> List[VectorSearchResult]:
        """Search for similar vectors in the store."""
        if not self._vectors:
            return []
        
        # Convert query vector to numpy array if it isn't already
        query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        
        # Get all vectors and their IDs
        ids = list(self._vectors.keys())
        vectors = np.stack([self._vectors[_id] for _id in ids])
        
        # Calculate cosine similarity
        query_norm = np.linalg.norm(query_vector, axis=1, keepdims=True)
        vectors_norm = np.linalg.norm(vectors, axis=1, keepdims=True)
        
        # Avoid division by zero
        query_norm[query_norm == 0] = 1e-10
        vectors_norm[vectors_norm == 0] = 1e-10
        
        # Normalize vectors
        query_vector_norm = query_vector / query_norm
        vectors_normed = vectors / vectors_norm
        
        # Calculate similarity scores
        scores = np.dot(vectors_normed, query_vector_norm.T).flatten()
        
        # Combine with IDs and filter if needed
        results = []
        for i, score in enumerate(scores):
            doc_id = ids[i]
            doc = self._store[doc_id]
            
            # Apply metadata filter if provided
            if filter:
                if not all(
                    doc['metadata'].get(k) == v 
                    for k, v in filter.items()
                ):
                    continue
            
            results.append((score, doc_id, doc))
        
        # Sort by score (descending)
        results.sort(key=lambda x: x[0], reverse=True)
        
        # Return top k results
        return [
            VectorSearchResult(
                id=doc_id,
                content=doc['text'],
                metadata=doc['metadata'],
                score=float(score),
                vector=self._vectors[doc_id] if include_vectors else None
            )
            for score, doc_id, doc in results[:k]
        ]
    
    async def delete(self, ids: List[str]) -> bool:
        """Delete entries by their IDs."""
        for doc_id in ids:
            if doc_id in self._store:
                del self._store[doc_id]
            if doc_id in self._vectors:
                del self._vectors[doc_id]
        return True
    
    async def get(self, ids: List[str], include_vectors: bool = False) -> List[Optional[Dict[str, Any]]]:
        """Retrieve entries by their IDs."""
        results = []
        for doc_id in ids:
            if doc_id not in self._store:
                results.append(None)
                continue
                
            doc = self._store[doc_id].copy()
            if include_vectors and doc_id in self._vectors:
                doc['vector'] = self._vectors[doc_id]
            results.append(doc)
        
        return results
