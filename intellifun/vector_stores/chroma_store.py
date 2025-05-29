"""ChromaDB implementation of the vector store.

This module provides a ChromaDB implementation of the VectorStore interface.
Chromadb is an optional dependency and will only be imported if used.
"""
import asyncio
import logging
import uuid
from typing import Dict, List, Optional, Any, TYPE_CHECKING

import numpy as np

from .base import VectorStore, VectorSearchResult

# Only import chromadb when actually used
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

if TYPE_CHECKING:
    import chromadb
    from chromadb.config import Settings

class ChromaNotAvailableError(ImportError):
    """Raised when ChromaDB is not installed."""
    def __init__(self):
        super().__init__(
            "ChromaDB is not installed. Please install it with: "
            "pip install chromadb"
        )

logger = logging.getLogger(__name__)


def run_in_thread(fn, *args, **kwargs):
    """Run a synchronous function in a thread pool."""
    return asyncio.to_thread(fn, *args, **kwargs)

class ChromaVectorStore(VectorStore):
    """ChromaDB implementation of the VectorStore interface.
    
    This implementation uses ChromaDB for vector storage and retrieval.
    It supports both in-memory and persistent storage.
    """
    
    def __init__(
        self,
        collection_name: str = "default",
        persist_directory: Optional[str] = None,
        **kwargs
    ):
        """Initialize the ChromaDB store.
        
        Args:
            collection_name: Name of the collection to use
            persist_directory: If provided, persist data to this directory.
                              If None, data will be stored in memory only.
            **kwargs: Additional arguments to pass to the ChromaDB client
        """
        if not CHROMA_AVAILABLE:
            raise ChromaNotAvailableError()
            
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize ChromaDB client
        settings = Settings(anonymized_telemetry=False)
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory, settings=settings, **kwargs)
        else:
            self.client = chromadb.Client(settings=settings, **kwargs)
        
        # Get or create the collection
        try:
            self.collection = self.client.get_collection(collection_name)
        except ValueError:
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(collection_name)
    
    async def add(
        self,
        texts: List[str],
        vectors: List[np.ndarray],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs
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
        
        # Convert numpy arrays to lists for ChromaDB
        embeddings = [v.tolist() for v in vectors]
        
        # Add to collection in a thread
        await run_in_thread(
            self.collection.add,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        return ids
    
    async def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        include_vectors: bool = False,
        **kwargs
    ) -> List[VectorSearchResult]:
        """Search for similar vectors in the store."""
        # Convert query vector to list
        query_embedding = query_vector.tolist()
        
        # Query the collection in a thread
        results = await run_in_thread(
            self.collection.query,
            query_embeddings=[query_embedding],
            n_results=k,
            where=filter,
            include=["metadatas", "documents", "distances"]
        )
        
        # Process results
        search_results = []
        for i in range(len(results["ids"][0])):  # For each result
            doc_id = results["ids"][0][i]
            content = results["documents"][0][i]
            metadata = results["metadatas"][0][i] or {}
            distance = results["distances"][0][i] if results["distances"] else 0.0
            
            # Convert distance to similarity score (higher is better)
            # Chroma returns L2 distance, so we convert to similarity
            score = 1.0 / (1.0 + distance) if distance is not None else 0.0
            
                # Get vector if requested
            vector = None
            if include_vectors:
                # Need to fetch the document to get the vector
                doc = await run_in_thread(
                    self.collection.get,
                    ids=[doc_id],
                    include=["embeddings"]
                )
                if doc and doc["embeddings"]:
                    vector = np.array(doc["embeddings"][0], dtype=np.float32)
            
            search_results.append(
                VectorSearchResult(
                    id=doc_id,
                    content=content,
                    metadata=metadata,
                    score=score,
                    vector=vector
                )
            )
        
        return search_results
    
    async def delete(self, ids: List[str], **kwargs) -> bool:
        """Delete entries by their IDs."""
        if not ids:
            return True
            
        await run_in_thread(self.collection.delete, ids=ids)
        return True
    
    async def get(
        self, 
        ids: List[str], 
        include_vectors: bool = False,
        **kwargs
    ) -> List[Optional[Dict[str, Any]]]:
        """Retrieve entries by their IDs."""
        if not ids:
            return []
            
        include = ["metadatas", "documents"]
        if include_vectors:
            include.append("embeddings")
            
        results = await run_in_thread(self.collection.get, ids=ids, include=include)
        
        output = []
        for i, doc_id in enumerate(ids):
            if doc_id not in results["ids"]:
                output.append(None)
                continue
                
            idx = results["ids"].index(doc_id)
            doc = {
                "id": doc_id,
                "content": results["documents"][idx],
                "metadata": results["metadatas"][idx] or {}
            }
            
            if include_vectors and "embeddings" in results and results["embeddings"]:
                doc["vector"] = np.array(results["embeddings"][idx], dtype=np.float32)
                
            output.append(doc)
            
        return output
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Chroma handles its own cleanup
        # Ensure any pending operations are completed
        if hasattr(self, 'client'):
            await run_in_thread(self.client.heartbeat)  # Force any pending operations to complete
