"""Example usage of the vector store functionality."""
import asyncio
import numpy as np
from cortex import get_vector_store

async def main():
    # Create an in-memory vector store
    store = get_vector_store("memory")
    
    # Sample texts and their embeddings
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Python is a popular programming language",
        "Vectors can be used for semantic search",
        "Natural language processing helps computers understand human language"
    ]
    
    # Generate some random embeddings (in a real app, use a proper embedding model)
    # Each vector has 384 dimensions (like sentence-transformers/all-MiniLM-L6-v2)
    vectors = [np.random.rand(384).astype(np.float32) for _ in range(len(texts))]
    
    # Add documents to the store
    print("Adding documents to the vector store...")
    ids = await store.add(
        texts=texts,
        vectors=vectors,
        metadatas=[{"source": "example", "index": i} for i in range(len(texts))]
    )
    print(f"Added {len(ids)} documents with IDs: {ids}")
    
    # Perform a search
    query_vector = np.random.rand(384).astype(np.float32)  # Random query vector
    print("\nSearching for similar vectors...")
    results = await store.search(query_vector, k=2)
    
    print("\nTop 2 results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result.score:.4f}")
        print(f"   Text: {result.content[:80]}...")
        print(f"   Metadata: {result.metadata}")
    
    # Get documents by ID
    print("\nRetrieving documents by ID...")
    docs = await store.get([ids[0], ids[1]])
    for i, doc in enumerate(docs):
        if doc:
            print(f"Document {i+1}: {doc['text'][:80]}...")
        else:
            print(f"Document {i+1}: Not found")
    
    # Clean up
    await store.delete(ids)
    print("\nCleaned up: Deleted all documents")

if __name__ == "__main__":
    asyncio.run(main())
