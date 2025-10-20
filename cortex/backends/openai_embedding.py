"""OpenAI embedding backend for the Embedding class."""
from enum import Enum
from typing import List, Union
import numpy as np
import openai

from cortex.embeddings import Embedding, EmbeddingBackend, EmbeddingRequest
from cortex.backends.openai import get_openai_client, get_async_openai_client


class OpenAIEmbeddingModels(str, Enum):
    """OpenAI embedding models.
    
    Attributes:
        TEXT_EMBED_3_SMALL: Most recent model with strong performance, 1536 dimensions
        TEXT_EMBED_3_LARGE: Larger model with better performance, 3072 dimensions
        TEXT_EMBED_ADA_002: Older model, 1536 dimensions (legacy)
    """
    TEXT_EMBED_3_SMALL = "text-embedding-3-small"
    TEXT_EMBED_3_LARGE = "text-embedding-3-large"
    TEXT_EMBED_ADA_002 = "text-embedding-ada-002"

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

# Register the backend with all OpenAI embedding models
for model in OpenAIEmbeddingModels:
    Embedding.register_backend(model.value, OpenAIEmbeddingBackend)
