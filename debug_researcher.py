#!/usr/bin/env python3
"""
Debug Researcher Agent storage issues
"""

import asyncio
from musequill.agents.researcher.researcher_agent import ResearcherAgent, ProcessedChunk
from musequill.agents.researcher.researcher_agent_config import ResearcherConfig

async def test_storage_process():
    """Test the storage process specifically."""
    
    print("=== Researcher Agent Storage Debug ===")
    
    try:
        # Initialize researcher agent
        config = ResearcherConfig()
        researcher = ResearcherAgent(config)
        
        print(f"✓ Researcher agent initialized")
        print(f"Collection: {researcher.chroma_collection.name}")
        print(f"Collection count before: {researcher.chroma_collection.count()}")
        
        # Create some test chunks manually
        test_chunks = []
        for i in range(3):
            # Create a simple embedding (normally this would come from OpenAI)
            test_embedding = [0.1] * config.embedding_dimensions
            
            chunk = ProcessedChunk(
                chunk_id=f"debug_test_chunk_{i}",
                content=f"This is test content for chunk {i}. It contains enough text to be meaningful.",
                embedding=test_embedding,
                metadata={
                    'book_id': 'debug_book_123',
                    'query': 'debug test query',
                    'query_type': 'test',
                    'query_priority': 5,
                    'source_url': f'http://example.com/test{i}',
                    'source_title': f'Test Source {i}',
                    'source_domain': 'example.com',
                    'source_score': 0.8,
                    'chunk_index': i,
                    'chunk_size': len(f"This is test content for chunk {i}. It contains enough text to be meaningful."),
                    'processed_at': '2025-06-21T23:00:00Z'
                },
                quality_score=0.7,
                source_info={
                    'url': f'http://example.com/test{i}',
                    'title': f'Test Source {i}',
                    'domain': 'example.com',
                    'tavily_score': 0.8
                }
            )
            test_chunks.append(chunk)
        
        print(f"✓ Created {len(test_chunks)} test chunks")
        
        # Test the storage method
        book_id = 'debug_book_123'
        stored_count = await researcher._store_chunks_in_chroma(test_chunks, book_id)
        
        print(f"Storage result: {stored_count} chunks stored")
        print(f"Collection count after: {researcher.chroma_collection.count()}")
        
        # Verify the chunks were stored
        results = researcher.chroma_collection.get(
            where={"book_id": book_id},
            include=["documents", "metadatas"]
        )
        
        print(f"✓ Retrieved {len(results['documents'])} documents from ChromaDB")
        if results['documents']:
            for i, (doc, meta) in enumerate(zip(results['documents'], results['metadatas'])):
                print(f"  Document {i}: {doc[:50]}... (book_id: {meta.get('book_id')})")
        
        # Clean up
        if results['ids']:
            researcher.chroma_collection.delete(ids=results['ids'])
            print(f"✓ Cleaned up {len(results['ids'])} test documents")
        
        return stored_count > 0
        
    except Exception as e:
        print(f"✗ Error in storage test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_storage_process())
    if result:
        print("\n✅ Storage test passed!")
    else:
        print("\n❌ Storage test failed!")