#!/usr/bin/env python3
"""
Check ChromaDB collection details
"""

import chromadb
from musequill.agents.researcher.researcher_agent_config import ResearcherConfig

def check_collection_details():
    """Check collection metadata and dimension requirements."""
    
    config = ResearcherConfig()
    client = chromadb.HttpClient(
        host=config.chroma_host,
        port=config.chroma_port
    )
    
    try:
        collection = client.get_collection(name=config.chroma_collection_name)
        
        # Get collection metadata
        print(f"Collection name: {collection.name}")
        print(f"Collection metadata: {collection.metadata}")
        print(f"Collection count: {collection.count()}")
        
        # Check if there are any existing documents to see their embedding dimensions
        results = collection.peek(limit=1)
        if results['embeddings'] and results['embeddings'][0]:
            embedding_dim = len(results['embeddings'][0])
            print(f"Existing embedding dimensions: {embedding_dim}")
        else:
            print("No existing embeddings found")
        
        print(f"\nAgent configuration:")
        print(f"Embedding model: {config.embedding_model}")
        print(f"Expected dimensions: {config.embedding_dimensions}")
        
        # Try to delete the collection and recreate it
        print(f"\nDeleting existing collection to fix dimension mismatch...")
        client.delete_collection(name=config.chroma_collection_name)
        print(f"✓ Deleted collection")
        
        # Recreate with correct metadata
        new_collection = client.create_collection(
            name=config.chroma_collection_name,
            metadata={
                "description": "Research materials for book writing",
                "embedding_model": config.embedding_model,
                "embedding_dimensions": config.embedding_dimensions,
                "created_at": "2025-06-21T23:30:00Z",
                "recreated_reason": "dimension_mismatch_fix"
            }
        )
        print(f"✓ Created new collection with correct dimensions")
        print(f"New collection metadata: {new_collection.metadata}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_collection_details()