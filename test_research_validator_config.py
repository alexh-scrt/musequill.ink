#!/usr/bin/env python3
"""
Test script to verify ResearchValidatorAgent configuration and ChromaDB connectivity
"""

import os
from musequill.agents.research_validator.research_validator_config import ResearchValidatorConfig
from musequill.agents.research_validator.research_validator_agent import ResearchValidatorAgent
from musequill.agents.agent_state import BookWritingState, ProcessingStage, ResearchQuery, Chapter
from datetime import datetime, timezone

def test_config():
    """Test configuration loading."""
    print("=== Testing Configuration ===")
    
    # Check environment variables
    chroma_port_env = os.getenv('CHROMA_PORT')
    print(f"CHROMA_PORT environment variable: {chroma_port_env}")
    
    # Load config
    config = ResearchValidatorConfig()
    print(f"Config chroma_port: {config.chroma_port}")
    print(f"Config chroma_host: {config.chroma_host}")
    print(f"Config chroma_collection_name: {config.chroma_collection_name}")

def test_research_validator():
    """Test ResearchValidatorAgent with the specific book_id."""
    print("\n=== Testing ResearchValidatorAgent ===")
    
    target_book_id = '315e2d0a-d58c-4da7-9944-7c798a246c82'
    
    # Create a minimal test state
    test_state = BookWritingState(
        book_id=target_book_id,
        orchestration_id="test_orch",
        thread_id="test_thread",
        title="Test Book",
        description="Test description",
        genre="Science Fiction",
        target_word_count=50000,
        target_audience="Test audience",
        author_preferences={},
        outline={},
        chapters=[],
        current_stage=ProcessingStage.RESEARCH_COMPLETE,
        processing_started_at=datetime.now(timezone.utc).isoformat(),
        processing_updated_at=datetime.now(timezone.utc).isoformat(),
        research_queries=[
            ResearchQuery(
                query="test query",
                priority=5,
                query_type="background_information",
                status="completed",
                results_count=10,
                created_at=datetime.now(timezone.utc).isoformat()
            )
        ],
        research_strategy="Test strategy",
        total_research_chunks=0,
        research_completed_at=None,
        current_chapter=0,
        writing_strategy=None,
        writing_style_guide=None,
        total_word_count=0,
        writing_started_at=None,
        writing_completed_at=None,
        review_notes=None,
        revision_count=0,
        quality_score=None,
        errors=[],
        retry_count=0,
        last_error_at=None,
        progress_percentage=0.0,
        estimated_completion_time=None,
        final_book_content=None,
        metadata={}
    )
    
    try:
        # Create validator agent
        validator = ResearchValidatorAgent()
        print(f"Validator chroma_port: {validator.config.chroma_port}")
        
        # Test the retrieve_research_data method directly
        print(f"\nTesting _retrieve_research_data for book_id: {target_book_id}")
        research_data = validator._retrieve_research_data(target_book_id)
        
        print(f"Research data retrieved:")
        print(f"  Total chunks: {research_data['total_chunks']}")
        print(f"  Query groups: {len(research_data['query_groups'])}")
        print(f"  Category groups: {len(research_data['category_groups'])}")
        print(f"  Unique sources: {len(research_data['unique_sources'])}")
        print(f"  Unique domains: {len(research_data['unique_domains'])}")
        
        if research_data['chunks']:
            print(f"\nSample chunk metadata:")
            sample_chunk = research_data['chunks'][0]
            print(f"  Query: {sample_chunk['metadata'].get('query', 'N/A')}")
            print(f"  Query type: {sample_chunk['metadata'].get('query_type', 'N/A')}")
            print(f"  Source domain: {sample_chunk['metadata'].get('source_domain', 'N/A')}")
        
        # Test full validation
        print(f"\nTesting full validation...")
        validation_result = validator.validate_research(test_state)
        
        print(f"Validation results:")
        print(f"  Is sufficient: {validation_result['is_sufficient']}")
        print(f"  Confidence score: {validation_result['confidence_score']}")
        print(f"  Additional queries needed: {len(validation_result['additional_queries'])}")
        
        return True
        
    except Exception as e:
        print(f"Error during validation test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_config()
    success = test_research_validator()
    
    if success:
        print("\n✅ Research validator test completed successfully!")
    else:
        print("\n❌ Research validator test failed!")