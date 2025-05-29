"""OpenAI embedding backend for the Embedding class."""
from typing import List, Union
import numpy as np
import openai

from intellifun.embeddings import EmbeddingBackend, EmbeddingRequest
from intellifun.backends.openai import get_openai_client, get_async_openai_client

class OpenAIEmbeddingBackend(EmbeddingBackend):
    """OpenAI embedding backend implementation"""
    
    def __init__(self, **kwargs):
        """Initialize the OpenAI embedding backend.
        
        Args:
            **kwargs: Additional arguments to pass to the OpenAI client
        """
        self.client_kwargs = kwargs
        super().__init__()
    
    def embed(self, request: EmbeddingRequest) -> Union[np.ndarray, List[np.ndarray]]:
        """Generate embeddings synchronously.
        
        Args:
            request: The embedding request
            
        Returns:
            numpy.ndarray: For single text, returns a 1D array.
                         For multiple texts, returns a 2D array where each row is an embedding.
        """
        is_batch = isinstance(request.text, list)
        texts = request.text if is_batch else [request.text]
        
        try:
            response = get_openai_client().embeddings.create(
                input=texts,
                model=request.model,
                **self.client_kwargs
            )
            
            # Extract embeddings from response
            embeddings = [np.array(item.embedding) for item in response.data]
            
            # Sort embeddings to match input order (important for async)
            if len(embeddings) > 1:
                indices = [item.index for item in response.data]
                embeddings = [embeddings[i] for i in sorted(indices)]
            
            return np.array(embeddings[0]) if not is_batch else np.array(embeddings)
            
        except openai.APIError as e:
            # Handle specific API errors
            error_msg = f"OpenAI API error: {str(e)}"
            if "rate limit" in str(e).lower():
                error_msg += " (Rate limit exceeded)"
            raise RuntimeError(error_msg) from e
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings: {str(e)}") from e
    
    async def async_embed(self, request: EmbeddingRequest) -> Union[np.ndarray, List[np.ndarray]]:
        """Generate embeddings asynchronously.
        
        Args:
            request: The embedding request
            
        Returns:
            numpy.ndarray: For single text, returns a 1D array.
                         For multiple texts, returns a 2D array where each row is an embedding.
        """
        is_batch = isinstance(request.text, list)
        texts = request.text if is_batch else [request.text]
        
        try:
            response = await get_async_openai_client().embeddings.create(
                input=texts,
                model=request.model,
                **self.client_kwargs
            )
            
            # Extract embeddings from response
            embeddings = [np.array(item.embedding) for item in response.data]
            
            # Sort embeddings to match input order (important for async)
            if len(embeddings) > 1:
                indices = [item.index for item in response.data]
                embeddings = [embeddings[i] for i in sorted(indices)]
            
            return np.array(embeddings[0]) if not is_batch else np.array(embeddings)
            
        except openai.APIError as e:
            # Handle specific API errors
            error_msg = f"OpenAI API error: {str(e)}"
            if "rate limit" in str(e).lower():
                error_msg += " (Rate limit exceeded)"
            raise RuntimeError(error_msg) from e
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate async embeddings: {str(e)}") from e

# Register the OpenAI embedding models
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# Import here to avoid circular imports
from intellifun.embeddings import Embedding  # noqa: E402

# Register the backend with common OpenAI embedding models
for model in [
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002"
]:
    Embedding.register_backend(model, OpenAIEmbeddingBackend)
