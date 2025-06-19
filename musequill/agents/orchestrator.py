"""
Book Writing Orchestrator

Defines the LangGraph workflow for the book writing pipeline.
Coordinates research and writing agents through a structured state machine.

Pipeline Flow:
1. Research Planning -> Generate research queries
2. Research Execution -> Conduct research and store in vector DB
3. Research Validation -> Validate research completeness
4. Writing Planning -> Plan writing strategy and structure
5. Chapter Writing -> Write chapters iteratively
6. Quality Review -> Review and refine content
7. Final Assembly -> Compile final book
8. Storage -> Store completed book in MongoDB

Based on the POC pattern but extended for book writing workflow.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from typing import Dict, Any, Literal
from enum import Enum

from musequill.config.logging import get_logger
from musequill.agents.agent_state import BookWritingState, ProcessingStage

logger = get_logger(__name__)


class NodeType(str, Enum):
    """Node types in the book writing orchestration."""
    RESEARCH_PLANNER = "research_planner"
    RESEARCHER = "researcher"
    RESEARCH_VALIDATOR = "research_validator"
    WRITING_PLANNER = "writing_planner"
    CHAPTER_WRITER = "chapter_writer"
    QUALITY_REVIEWER = "quality_reviewer"
    FINAL_ASSEMBLER = "final_assembler"
    BOOK_STORER = "book_storer"


def research_planning_node(state: BookWritingState) -> BookWritingState:
    """
    Research Planning Node
    
    Analyzes the book outline and generates comprehensive research queries.
    This node creates a research strategy and specific queries for gathering information.
    """
    try:
        logger.info(f"Starting research planning for book {state['book_id']}")
        
        # Import here to avoid circular imports
        from musequill.agents.research.research_planner import ResearchPlannerAgent
        
        # Create research planner instance
        planner = ResearchPlannerAgent()
        
        # Generate research plan and queries
        research_plan = planner.create_research_plan(state)
        
        # Update state with research plan
        updated_state = state.copy()
        updated_state['current_stage'] = ProcessingStage.RESEARCH_PLANNING
        updated_state['research_strategy'] = research_plan['strategy']
        updated_state['research_queries'] = research_plan['queries']
        updated_state['progress_percentage'] = 10.0
        
        logger.info(f"Research planning completed for book {state['book_id']}, generated {len(research_plan['queries'])} queries")
        
        return updated_state
        
    except Exception as e:
        logger.error(f"Error in research planning for book {state['book_id']}: {e}")
        updated_state = state.copy()
        updated_state['errors'].append(f"Research planning failed: {str(e)}")
        updated_state['current_stage'] = ProcessingStage.FAILED
        return updated_state


def research_execution_node(state: BookWritingState) -> BookWritingState:
    """
    Research Execution Node
    
    Executes the research queries and stores results in the vector database.
    Uses web search and other sources to gather comprehensive information.
    """
    try:
        logger.info(f"Starting research execution for book {state['book_id']}")
        
        # Import here to avoid circular imports
        from musequill.agents.researcher.researcher_agent import ResearcherAgent
        
        # Create researcher instance
        researcher = ResearcherAgent()
        
        # Execute research queries
        research_results = researcher.execute_research(state)
        
        # Update state with research results
        updated_state = state.copy()
        updated_state['current_stage'] = ProcessingStage.RESEARCHING
        updated_state['total_research_chunks'] = research_results['total_chunks']
        updated_state['research_queries'] = research_results['updated_queries']
        updated_state['progress_percentage'] = 30.0
        
        logger.info(f"Research execution completed for book {state['book_id']}, stored {research_results['total_chunks']} chunks")
        
        return updated_state
        
    except Exception as e:
        logger.error(f"Error in research execution for book {state['book_id']}: {e}")
        updated_state = state.copy()
        updated_state['errors'].append(f"Research execution failed: {str(e)}")
        updated_state['current_stage'] = ProcessingStage.FAILED
        return updated_state


def research_validation_node(state: BookWritingState) -> BookWritingState:
    """
    Research Validation Node
    
    Validates that sufficient research has been gathered and identifies any gaps.
    May trigger additional research if needed.
    """
    try:
        logger.info(f"Starting research validation for book {state['book_id']}")
        
        # Import here to avoid circular imports
        from musequill.agents.research_validator.research_validator_agent import ResearchValidatorAgent
        
        # Create validator instance
        validator = ResearchValidatorAgent()
        
        # Validate research completeness
        validation_results = validator.validate_research(state)
        
        # Update state with validation results
        updated_state = state.copy()
        
        if validation_results['is_sufficient']:
            updated_state['current_stage'] = ProcessingStage.RESEARCH_COMPLETE
            updated_state['research_completed_at'] = validation_results['completed_at']
            updated_state['progress_percentage'] = 40.0
            logger.info(f"Research validation passed for book {state['book_id']}")
        else:
            # Research insufficient, add more queries
            updated_state['research_queries'].extend(validation_results['additional_queries'])
            updated_state['current_stage'] = ProcessingStage.RESEARCHING
            logger.info(f"Research validation failed for book {state['book_id']}, added {len(validation_results['additional_queries'])} additional queries")
        
        return updated_state
        
    except Exception as e:
        logger.error(f"Error in research validation for book {state['book_id']}: {e}")
        updated_state = state.copy()
        updated_state['errors'].append(f"Research validation failed: {str(e)}")
        updated_state['current_stage'] = ProcessingStage.FAILED
        return updated_state


def writing_planning_node(state: BookWritingState) -> BookWritingState:
    """
    Writing Planning Node
    
    Creates a comprehensive writing strategy, style guide, and chapter-by-chapter plan.
    """
    try:
        logger.info(f"Starting writing planning for book {state['book_id']}")
        
        # Import here to avoid circular imports
        from musequill.agents.writing_planner import WritingPlannerAgent
        
        # Create writing planner instance
        planner = WritingPlannerAgent()
        
        # Generate writing plan
        writing_plan = planner.create_writing_plan(state)
        
        # Update state with writing plan
        updated_state = state.copy()
        updated_state['current_stage'] = ProcessingStage.WRITING_PLANNING
        updated_state['writing_strategy'] = writing_plan['strategy']
        updated_state['writing_style_guide'] = writing_plan['style_guide']
        updated_state['chapters'] = writing_plan['updated_chapters']
        updated_state['writing_started_at'] = writing_plan['started_at']
        updated_state['progress_percentage'] = 50.0
        
        logger.info(f"Writing planning completed for book {state['book_id']}")
        
        return updated_state
        
    except Exception as e:
        logger.error(f"Error in writing planning for book {state['book_id']}: {e}")
        updated_state = state.copy()
        updated_state['errors'].append(f"Writing planning failed: {str(e)}")
        updated_state['current_stage'] = ProcessingStage.FAILED
        return updated_state


def chapter_writing_node(state: BookWritingState) -> BookWritingState:
    """
    Chapter Writing Node
    
    Writes individual chapters using research materials from the vector database.
    Processes chapters iteratively until all are complete.
    """
    try:
        logger.info(f"Starting chapter writing for book {state['book_id']}, chapter {state['current_chapter'] + 1}")
        
        # Import here to avoid circular imports
        from musequill.agents.chapter_writer import ChapterWriterAgent
        
        # Create chapter writer instance
        writer = ChapterWriterAgent()
        
        # Write the current chapter
        writing_results = writer.write_chapter(state)
        
        # Update state with writing results
        updated_state = state.copy()
        updated_state['chapters'] = writing_results['updated_chapters']
        updated_state['current_chapter'] = writing_results['next_chapter']
        updated_state['total_word_count'] = writing_results['total_word_count']
        updated_state['current_stage'] = ProcessingStage.WRITING
        
        # Calculate progress based on chapters completed
        chapters_completed = sum(1 for ch in updated_state['chapters'] if ch['status'] == 'complete')
        total_chapters = len(updated_state['chapters'])
        chapter_progress = (chapters_completed / total_chapters) * 40  # 40% of total progress for writing
        updated_state['progress_percentage'] = 50.0 + chapter_progress
        
        logger.info(f"Chapter {state['current_chapter'] + 1} completed for book {state['book_id']}, {chapters_completed}/{total_chapters} chapters done")
        
        return updated_state
        
    except Exception as e:
        logger.error(f"Error in chapter writing for book {state['book_id']}: {e}")
        updated_state = state.copy()
        updated_state['errors'].append(f"Chapter writing failed: {str(e)}")
        updated_state['current_stage'] = ProcessingStage.FAILED
        return updated_state


def quality_review_node(state: BookWritingState) -> BookWritingState:
    """
    Quality Review Node
    
    Reviews the completed book for consistency, quality, and completeness.
    May trigger revisions if quality standards are not met.
    """
    try:
        logger.info(f"Starting quality review for book {state['book_id']}")
        
        # Import here to avoid circular imports
        from musequill.agents.quality_reviewer import QualityReviewerAgent
        
        # Create quality reviewer instance
        reviewer = QualityReviewerAgent()
        
        # Review the book quality
        review_results = reviewer.review_book(state)
        
        # Update state with review results
        updated_state = state.copy()
        updated_state['current_stage'] = ProcessingStage.REVIEW
        updated_state['quality_score'] = review_results['quality_score']
        updated_state['review_notes'] = review_results['review_notes']
        updated_state['progress_percentage'] = 95.0
        
        # Check if revisions are needed
        if review_results['needs_revision']:
            updated_state['revision_count'] += 1
            updated_state['chapters'] = review_results['revised_chapters']
            logger.info(f"Quality review for book {state['book_id']} requires revisions (revision #{updated_state['revision_count']})")
        else:
            updated_state['writing_completed_at'] = review_results['completed_at']
            logger.info(f"Quality review passed for book {state['book_id']}, quality score: {review_results['quality_score']}")
        
        return updated_state
        
    except Exception as e:
        logger.error(f"Error in quality review for book {state['book_id']}: {e}")
        updated_state = state.copy()
        updated_state['errors'].append(f"Quality review failed: {str(e)}")
        updated_state['current_stage'] = ProcessingStage.FAILED
        return updated_state


def final_assembly_node(state: BookWritingState) -> BookWritingState:
    """
    Final Assembly Node
    
    Assembles all chapters into the final book format and prepares for storage.
    """
    try:
        logger.info(f"Starting final assembly for book {state['book_id']}")
        
        # Import here to avoid circular imports
        from musequill.agents.final_assembler import FinalAssemblerAgent
        
        # Create final assembler instance
        assembler = FinalAssemblerAgent()
        
        # Assemble the final book
        assembly_results = assembler.assemble_book(state)
        
        # Update state with final book content
        updated_state = state.copy()
        updated_state['final_book_content'] = assembly_results['final_content']
        updated_state['metadata'].update(assembly_results['metadata'])
        updated_state['progress_percentage'] = 99.0
        
        logger.info(f"Final assembly completed for book {state['book_id']}, total words: {assembly_results['total_words']}")
        
        return updated_state
        
    except Exception as e:
        logger.error(f"Error in final assembly for book {state['book_id']}: {e}")
        updated_state = state.copy()
        updated_state['errors'].append(f"Final assembly failed: {str(e)}")
        updated_state['current_stage'] = ProcessingStage.FAILED
        return updated_state


def book_storage_node(state: BookWritingState) -> BookWritingState:
    """
    Book Storage Node
    
    Stores the completed book in MongoDB and updates status.
    """
    try:
        logger.info(f"Starting book storage for book {state['book_id']}")
        
        # Import here to avoid circular imports
        from musequill.agents.book_storer import BookStorerAgent
        
        # Create book storer instance
        storer = BookStorerAgent()
        
        # Store the book
        storage_results = storer.store_book(state)
        
        # Update state with storage confirmation
        updated_state = state.copy()
        updated_state['current_stage'] = ProcessingStage.COMPLETE
        updated_state['progress_percentage'] = 100.0
        updated_state['metadata'].update(storage_results['metadata'])
        
        logger.info(f"Book storage completed for book {state['book_id']}, stored as document ID: {storage_results['document_id']}")
        
        return updated_state
        
    except Exception as e:
        logger.error(f"Error in book storage for book {state['book_id']}: {e}")
        updated_state = state.copy()
        updated_state['errors'].append(f"Book storage failed: {str(e)}")
        updated_state['current_stage'] = ProcessingStage.FAILED
        return updated_state


# Edge condition functions
def should_continue_research(state: BookWritingState) -> Literal["research_execution", "writing_planning"]:
    """Determine if more research is needed or if we can proceed to writing."""
    if state['current_stage'] == ProcessingStage.RESEARCHING:
        # Check if there are pending research queries
        pending_queries = [q for q in state['research_queries'] if q['status'] == 'pending']
        if pending_queries:
            return "research_execution"
    
    return "writing_planning"


def should_continue_writing(state: BookWritingState) -> Literal["chapter_writer", "quality_reviewer"]:
    """Determine if more chapters need to be written or if we can proceed to review."""
    # Check if there are incomplete chapters
    incomplete_chapters = [ch for ch in state['chapters'] if ch['status'] != 'complete']
    
    if incomplete_chapters and state['current_chapter'] < len(state['chapters']):
        return "chapter_writer"
    
    return "quality_reviewer"


def should_revise_or_complete(state: BookWritingState) -> Literal["chapter_writer", "final_assembler", "END"]:
    """Determine if revisions are needed or if we can proceed to final assembly."""
    # Check if maximum revisions reached
    MAX_REVISIONS = 3  # Configurable limit
    
    if state['revision_count'] >= MAX_REVISIONS:
        logger.warning(f"Book {state['book_id']} reached maximum revisions ({MAX_REVISIONS}), proceeding to final assembly")
        return "final_assembler"
    
    # Check if quality review indicated revisions needed
    if state.get('review_notes') and any('revision' in note.lower() for note in state['review_notes']):
        return "chapter_writer"
    
    # Check quality score threshold
    quality_threshold = 0.8  # Configurable threshold
    if state.get('quality_score', 0) < quality_threshold:
        return "chapter_writer"
    
    return "final_assembler"


def is_processing_complete(state: BookWritingState) -> Literal["END"]:
    """Always end after book storage."""
    return END


def create_book_writing_graph(checkpointer: BaseCheckpointSaver) -> StateGraph:
    """
    Create the LangGraph workflow for book writing orchestration.
    
    Args:
        checkpointer: LangGraph checkpointer for state persistence
        
    Returns:
        Compiled StateGraph for book writing orchestration
    """
    logger.info("Creating book writing orchestration graph")
    
    # Create the state graph
    builder = StateGraph(BookWritingState)
    
    # Add all nodes
    builder.add_node(str(NodeType.RESEARCH_PLANNER), research_planning_node)
    builder.add_node(str(NodeType.RESEARCHER), research_execution_node)
    builder.add_node(str(NodeType.RESEARCH_VALIDATOR), research_validation_node)
    builder.add_node(str(NodeType.WRITING_PLANNER), writing_planning_node)
    builder.add_node(str(NodeType.CHAPTER_WRITER), chapter_writing_node)
    builder.add_node(str(NodeType.QUALITY_REVIEWER), quality_review_node)
    builder.add_node(str(NodeType.FINAL_ASSEMBLER), final_assembly_node)
    builder.add_node(str(NodeType.BOOK_STORER), book_storage_node)
    
    # Set entry point
    builder.set_entry_point(str(NodeType.RESEARCH_PLANNER))
    
    # Add edges for the main flow
    builder.add_edge(str(NodeType.RESEARCH_PLANNER), str(NodeType.RESEARCHER))
    builder.add_edge(str(NodeType.RESEARCHER), str(NodeType.RESEARCH_VALIDATOR))
    
    # Conditional edge after research validation
    builder.add_conditional_edges(
        str(NodeType.RESEARCH_VALIDATOR),
        should_continue_research,
        {
            "research_execution": str(NodeType.RESEARCHER),
            "writing_planning": str(NodeType.WRITING_PLANNER)
        }
    )
    
    builder.add_edge(str(NodeType.WRITING_PLANNER), str(NodeType.CHAPTER_WRITER))
    
    # Conditional edge for chapter writing loop
    builder.add_conditional_edges(
        str(NodeType.CHAPTER_WRITER),
        should_continue_writing,
        {
            "chapter_writer": str(NodeType.CHAPTER_WRITER),
            "quality_reviewer": str(NodeType.QUALITY_REVIEWER)
        }
    )
    
    # Conditional edge for quality review and revisions
    builder.add_conditional_edges(
        str(NodeType.QUALITY_REVIEWER),
        should_revise_or_complete,
        {
            "chapter_writer": str(NodeType.CHAPTER_WRITER),
            "final_assembler": str(NodeType.FINAL_ASSEMBLER),
            END: END
        }
    )
    
    builder.add_edge(str(NodeType.FINAL_ASSEMBLER), str(NodeType.BOOK_STORER))
    
    # Final edge to end
    builder.add_conditional_edges(
        str(NodeType.BOOK_STORER),
        is_processing_complete,
        {END: END}
    )
    
    # Compile the graph with checkpointer
    graph = builder.compile(checkpointer=checkpointer, debug=True)
    
    logger.info("Book writing orchestration graph created successfully")
    
    return graph


def main():
    """Test function for the orchestrator."""
    from langgraph.checkpoint.memory import MemorySaver
    from musequill.agents.agent_state import BookWritingState, ProcessingStage
    from datetime import datetime, timezone
    
    print("Testing Book Writing Orchestrator...")
    
    # Create test state
    test_state = BookWritingState(
        book_id="test_book_123",
        orchestration_id="test_orch_456",
        thread_id="test_thread_789",
        title="Test Book",
        description="A test book for the orchestrator",
        genre="Fiction",
        target_word_count=50000,
        target_audience="General",
        author_preferences={},
        outline={"summary": "Test outline"},
        chapters=[],
        current_stage=ProcessingStage.INITIALIZED,
        processing_started_at=datetime.now(timezone.utc).isoformat(),
        processing_updated_at=datetime.now(timezone.utc).isoformat(),
        research_queries=[],
        research_strategy=None,
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
        # Create memory checkpointer
        memory = MemorySaver()
        
        # Create the orchestration graph
        graph = create_book_writing_graph(memory)
        
        print("Graph created successfully!")
        print("Graph nodes:", list(graph.nodes.keys()))
        print("Graph edges:", list(graph.edges))
        
        # Test configuration
        thread_config = {"configurable": {"thread_id": "test_thread_789"}}
        
        print("\nOrchestrator test completed successfully!")
        
    except Exception as e:
        print(f"Error during orchestrator test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()