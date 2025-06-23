#!/usr/bin/env python3
"""
Debug script to investigate ChromaDB data for book_id: 315e2d0a-d58c-4da7-9944-7c798a246c82
"""

import chromadb
from chromadb.config import Settings
from musequill.config.logging import get_logger

logger = get_logger(__name__)

def debug_chromadb():
    """Debug ChromaDB contents for the specific book_id."""
    
    target_book_id = '315e2d0a-d58c-4da7-9944-7c798a246c82'
    
    try:
        # Initialize Chroma client with default config
        chroma_client = chromadb.HttpClient(
            host='localhost',
            port=18000,
            settings=Settings(
                chroma_server_authn_credentials=None,
                chroma_server_authn_provider=None
            )
        )
        
        # List collections
        collections = chroma_client.list_collections()
        print(f"Available collections: {[c.name for c in collections]}")
        
        # Try to get the default collection
        collection_name = 'book_research'
        try:
            collection = chroma_client.get_collection(name=collection_name)
            print(f"Successfully connected to collection: {collection_name}")
        except Exception as e:
            print(f"Could not connect to collection {collection_name}: {e}")
            return
        
        # Get all documents to see what book_ids exist
        print("\n=== All documents in collection ===")
        all_results = collection.get(
            include=["metadatas"],
            limit=100  # Limit to avoid too much output
        )
        
        if all_results.get('metadatas'):
            all_book_ids = set()
            for metadata in all_results['metadatas']:
                book_id = metadata.get('book_id')
                all_book_ids.add(book_id)
                print(f"Found book_id: '{book_id}' (type: {type(book_id)})")
            
            print(f"\nUnique book_ids in collection: {list(all_book_ids)}")
            print(f"Total documents: {len(all_results['metadatas'])}")
            
            # Check if target book_id exists (exact match)
            if target_book_id in all_book_ids:
                print(f"\n✅ Target book_id '{target_book_id}' found in collection!")
            else:
                print(f"\n❌ Target book_id '{target_book_id}' NOT found in collection")
                
                # Check for similar book_ids
                similar_ids = [bid for bid in all_book_ids if bid and target_book_id.replace('-', '') in str(bid).replace('-', '')]
                if similar_ids:
                    print(f"Similar book_ids found: {similar_ids}")
        
        # Try direct query for target book_id
        print(f"\n=== Direct query for book_id: '{target_book_id}' ===")
        target_results = collection.get(
            where={"book_id": target_book_id},
            include=["documents", "metadatas"]
        )
        
        print(f"Direct query returned {len(target_results.get('documents', []))} documents")
        
        if target_results.get('metadatas'):
            print("Sample metadata from target query:")
            for i, metadata in enumerate(target_results['metadatas'][:3]):
                print(f"  Document {i+1}:")
                print(f"    book_id: '{metadata.get('book_id')}' (type: {type(metadata.get('book_id'))})")
                print(f"    query: '{metadata.get('query', 'N/A')}'")
                print(f"    query_type: '{metadata.get('query_type', 'N/A')}'")
                print(f"    source_domain: '{metadata.get('source_domain', 'N/A')}'")
        
        # Try different variations of the book_id query
        print(f"\n=== Testing query variations ===")
        
        # Test with string conversion
        string_results = collection.get(
            where={"book_id": str(target_book_id)},
            include=["metadatas"]
        )
        print(f"Query with str(book_id) returned: {len(string_results.get('metadatas', []))} documents")
        
        # Test case variations
        lower_results = collection.get(
            where={"book_id": target_book_id.lower()},
            include=["metadatas"]
        )
        print(f"Query with lower() returned: {len(lower_results.get('metadatas', []))} documents")
        
        upper_results = collection.get(
            where={"book_id": target_book_id.upper()},
            include=["metadatas"]
        )
        print(f"Query with upper() returned: {len(upper_results.get('metadatas', []))} documents")
        
        print("\n=== Collection Statistics ===")
        try:
            count = collection.count()
            print(f"Total documents in collection: {count}")
        except Exception as e:
            print(f"Could not get collection count: {e}")
            
    except Exception as e:
        print(f"Error debugging ChromaDB: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_chromadb()