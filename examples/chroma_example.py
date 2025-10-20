"""Example of using the ChromaDB vector store.

This example demonstrates how to:
1. Set up a ChromaDB vector store with persistent storage
2. Add documents with embeddings
3. Perform similarity searches
4. Filter results by metadata
5. Clean up resources

Prerequisites:
1. chromadb package installed (automatically installed with cortex)
2. numpy for generating example embeddings
"""
import asyncio
import logging
import numpy as np
from dotenv import load_dotenv
from cortex.vector_stores import get_vector_store, VectorStoreType

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    # Create a temporary directory for this example
    import tempfile
    import shutil
    
    # Create a temporary directory for ChromaDB
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Using temporary directory for ChromaDB: {temp_dir}")
    
    try:
        # Initialize the ChromaDB store with persistent storage
        logger.info("Initializing ChromaDB store...")
        store = get_vector_store(
            VectorStoreType.CHROMA,
            collection_name="example_collection",
            persist_directory=temp_dir  # Omit this for in-memory only
        )
        
        # Sample documents and their embeddings
        documents = [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning is a subset of artificial intelligence",
            "Python is a popular programming language for data science",
            "Vectors can be used for semantic search and recommendations",
            "Natural language processing enables computers to understand human language"
        ]
        
        # Generate some random embeddings (in a real app, use a proper embedding model)
        logger.info("Generating sample embeddings...")
        embeddings = [np.random.rand(384).astype(np.float32) for _ in documents]
        
        # Add metadata (optional)
        metadatas = [
            {"category": "example", "length": len(doc.split()), "source": "example"} 
            for doc in documents
        ]
        
        # Add documents to the store
        logger.info("Adding documents to the store...")
        doc_ids = await store.add(
            texts=documents,
            vectors=embeddings,
            metadatas=metadatas
        )
        logger.info(f"Added {len(doc_ids)} documents with IDs: {doc_ids}")
        
        # Perform a search
        logger.info("\nPerforming similarity search...")
        query_embedding = np.random.rand(384).astype(np.float32)  # Random query
        
        # Basic search
        print("\nTop 3 most similar documents:")
        results = await store.search(query_embedding, k=3)
        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result.score:.4f}")
            print(f"   Text: {result.content}")
            print(f"   Metadata: {result.metadata}")
        
        # Search with metadata filter
        print("\nSearching for documents with length > 5 words:")
        filtered_results = await store.search(
            query_embedding,
            k=3,
            filter={"length": {"$gt": 5}}
        )
        for i, result in enumerate(filtered_results, 1):
            print(f"{i}. Score: {result.score:.4f}")
            print(f"   Text: {result.content}")
            print(f"   Length: {result.metadata.get('length')} words")
    
    finally:
        # Clean up the temporary directory
        logger.info("\nCleaning up temporary directory...")
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info("Cleanup complete.")

if __name__ == "__main__":
    asyncio.run(main())
