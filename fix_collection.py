#!/usr/bin/env python3
"""
Fix ChromaDB collection dimension mismatch
"""

import chromadb
from musequill.agents.researcher.researcher_agent_config import ResearcherConfig

def fix_collection():
    """Fix collection dimension mismatch."""
    
    config = ResearcherConfig()
    client = chromadb.HttpClient(
        host=config.chroma_host,
        port=config.chroma_port
    )
    
    try:
        # Check if collection exists
        try:
            collection = client.get_collection(name=config.chroma_collection_name)
            print(f"Found existing collection: {collection.name}")
            print(f"Collection metadata: {collection.metadata}")
            print(f"Collection count: {collection.count()}")
            
            # Delete the existing collection
            client.delete_collection(name=config.chroma_collection_name)
            print(f"✓ Deleted existing collection")
            
        except Exception as e:
            print(f"Collection doesn't exist or error getting it: {e}")
        
        # Create new collection with correct configuration
        new_collection = client.create_collection(
            name=config.chroma_collection_name,
            metadata={
                "description": "Research materials for book writing",
                "embedding_model": config.embedding_model,
                "embedding_dimensions": config.embedding_dimensions,
                "created_at": "2025-06-21T23:30:00Z",
                "fixed_dimension_mismatch": True
            }
        )
        print(f"✓ Created new collection: {new_collection.name}")
        print(f"New collection metadata: {new_collection.metadata}")
        print(f"Expected embedding dimensions: {config.embedding_dimensions}")
        
        return True
        
    except Exception as e:
        print(f"Error fixing collection: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = fix_collection()
    if success:
        print("\n✅ Collection fixed successfully!")
    else:
        print("\n❌ Failed to fix collection!")