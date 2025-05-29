#!/usr/bin/env python3
"""
Example usage of the Embedding API with OpenAI backend.

Make sure to set your OPENAI_API_KEY environment variable before running this example.
"""
import os
import asyncio
import numpy as np
from intellifun.embeddings import Embedding
from intellifun.backends import DEFAULT_OPENAI_EMBEDDING_MODEL

def sync_example():
    """Synchronous embedding example"""
    print("=== Synchronous Embedding Example ===")
    
    # Initialize the embedder with default model
    embedder = Embedding(model=DEFAULT_OPENAI_EMBEDDING_MODEL)
    
    # Single text embedding
    text = "This is a test sentence."
    embedding = embedder.embed(text)
    print(f"\nSingle text embedding shape: {embedding.shape}")
    print(f"First 5 values: {embedding[:5]}")
    
    # Batch embedding
    texts = ["First text", "Second text", "Third text"]
    embeddings = embedder.embed(texts)
    print(f"\nBatch embeddings shape: {embeddings.shape}")
    print(f"First embedding first 5 values: {embeddings[0][:5]}")

async def async_example():
    """Asynchronous embedding example"""
    print("\n=== Asynchronous Embedding Example ===")
    
    # Initialize the embedder with default model
    embedder = Embedding(model=DEFAULT_OPENAI_EMBEDDING_MODEL)
    
    # Single text embedding
    text = "This is an async test sentence."
    embedding = await embedder.async_embed(text)
    print(f"\nAsync single text embedding shape: {embedding.shape}")
    print(f"First 5 values: {embedding[:5]}")
    
    # Batch embedding
    texts = ["Async first text", "Async second text", "Async third text"]
    embeddings = await embedder.async_embed(texts)
    print(f"\nAsync batch embeddings shape: {embeddings.shape}")
    print(f"First embedding first 5 values: {embeddings[0][:5]}")

def backup_model_example():
    """Example showing backup model functionality"""
    print("\n=== Backup Model Example ===")
    
    # Set up backup model (this would typically be in your app initialization)
    Embedding.set_backup_backend("text-embedding-3-small", "text-embedding-ada-002")
    
    # This will try text-embedding-3-small first, and if it fails, fall back to text-embedding-ada-002
    embedder = Embedding(model="text-embedding-3-small")
    
    try:
        # This will use the primary model if available, otherwise fall back to the backup
        embedding = embedder.embed("Test with fallback")
        print(f"\nGenerated embedding with shape: {embedding.shape}")
    except Exception as e:
        print(f"\nError: {e}")
    
    # Reset failed models if needed
    # Embedding.reset_failed_models()

if __name__ == "__main__":
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-api-key'")
        exit(1)
    
    # Run synchronous example
    sync_example()
    
    # Run asynchronous example
    asyncio.run(async_example())
    
    # Run backup model example
    backup_model_example()
