#!/usr/bin/env python3
"""
Debug ChromaDB connection issues
"""

import os
import chromadb
from chromadb.config import Settings
from musequill.agents.researcher.researcher_agent_config import ResearcherConfig

def test_chroma_connection():
    """Test ChromaDB connection with various configurations."""
    
    print("=== ChromaDB Connection Debug ===")
    
    # Test 1: Check environment variables
    print(f"Environment CHROMA_PORT: {os.getenv('CHROMA_PORT', 'not set')}")
    print(f"Environment CHROMA_HOST: {os.getenv('CHROMA_HOST', 'not set')}")
    
    # Test 2: Check config loading
    config = ResearcherConfig()
    print(f"Config chroma_host: {config.chroma_host}")
    print(f"Config chroma_port: {config.chroma_port}")
    print(f"Config collection_name: {config.chroma_collection_name}")
    
    # Test 3: Try to connect
    try:
        client = chromadb.HttpClient(
            host=config.chroma_host,
            port=config.chroma_port,
            settings=Settings(
                chroma_server_authn_credentials=None,
                chroma_server_authn_provider=None
            )
        )
        print(f"✓ Client created successfully")
        
        # Test heartbeat
        heartbeat = client.heartbeat()
        print(f"✓ Heartbeat successful: {heartbeat}")
        
        # Test collection operations
        collections = client.list_collections()
        print(f"✓ Existing collections: {[c.name for c in collections]}")
        
        # Try to get or create collection
        try:
            collection = client.get_collection(name=config.chroma_collection_name)
            print(f"✓ Found existing collection: {config.chroma_collection_name}")
            count = collection.count()
            print(f"✓ Collection has {count} items")
        except Exception as e:
            print(f"⚠ Collection doesn't exist, trying to create: {e}")
            try:
                collection = client.create_collection(
                    name=config.chroma_collection_name,
                    metadata={
                        "description": "Test collection for debugging",
                        "created_by": "debug_script"
                    }
                )
                print(f"✓ Created new collection: {config.chroma_collection_name}")
            except Exception as create_e:
                print(f"✗ Failed to create collection: {create_e}")
                return False
        
        # Test simple add/query operation
        try:
            test_docs = ["This is a test document for debugging ChromaDB connection."]
            test_ids = ["debug_test_1"]
            test_metadata = [{"source": "debug_script", "test": True}]
            
            collection.add(
                ids=test_ids,
                documents=test_docs,
                metadatas=test_metadata
            )
            print(f"✓ Successfully added test document")
            
            # Query back
            results = collection.get(ids=test_ids)
            if results['documents']:
                print(f"✓ Successfully retrieved test document: {results['documents'][0][:50]}...")
            
            # Clean up
            collection.delete(ids=test_ids)
            print(f"✓ Cleaned up test document")
            
        except Exception as e:
            print(f"✗ Error with add/query operations: {e}")
            return False
        
        print(f"\n✅ All ChromaDB tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        print(f"✗ Error type: {type(e).__name__}")
        
        # Try alternative connection methods
        print(f"\n--- Trying alternative connections ---")
        
        # Try direct port connection
        try:
            client_direct = chromadb.HttpClient(host="localhost", port=18000)
            heartbeat = client_direct.heartbeat()
            print(f"✓ Direct connection to localhost:18000 works: {heartbeat}")
        except Exception as direct_e:
            print(f"✗ Direct connection failed: {direct_e}")
        
        return False

if __name__ == "__main__":
    test_chroma_connection()