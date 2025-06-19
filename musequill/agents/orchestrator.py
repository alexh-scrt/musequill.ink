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
        logger.info(f"Starting chapter writing for book {state['book_id']}")
        
        # Import here to avoid circular imports
        from musequill.agents.chapter_writer.chapter_writer_agent import ChapterWriterAgent
        
        # Create chapter writer instance
        writer = ChapterWriterAgent()
        
        # Get current chapter status
        current_chapter_num = state['current_chapter']
        total_chapters = len(state['chapters'])
        
        logger.info(f"Writing chapter {current_chapter_num + 1} of {total_chapters}")
        
        # Check if all chapters are complete
        if current_chapter_num >= total_chapters:
            logger.info(f"All chapters completed for book {state['book_id']}")
            updated_state = state.copy()
            updated_state['current_stage'] = ProcessingStage.WRITING_COMPLETE
            updated_state['writing_completed_at'] = datetime.now(timezone.utc).isoformat()
            updated_state['progress_percentage'] = 90.0  # Ready for quality review
            return updated_state
        
        # Write the next chapter
        writing_result = writer.write_next_chapter(state)
        
        # Update state based on writing result
        updated_state = state.copy()
        
        if writing_result['status'] == 'success':
            # Update the chapter in the state
            chapter_index = writing_result['chapter_number'] - 1
            updated_state['chapters'][chapter_index] = writing_result['updated_chapter']
            
            # Update overall state
            updated_state['current_chapter'] = current_chapter_num + 1
            updated_state['total_word_count'] += writing_result['words_written']
            updated_state['current_stage'] = ProcessingStage.WRITING
            
            # Calculate progress (50% base + 40% for writing completion)
            writing_progress = (current_chapter_num + 1) / total_chapters * 40
            updated_state['progress_percentage'] = 50.0 + writing_progress
            
            # Update writing started timestamp if this is the first chapter
            if current_chapter_num == 0 and not state.get('writing_started_at'):
                updated_state['writing_started_at'] = datetime.now(timezone.utc).isoformat()
            
            logger.info(f"Successfully wrote Chapter {writing_result['chapter_number']} for book {state['book_id']}")
            
        elif writing_result['status'] == 'complete':
            # All chapters are done
            updated_state['current_stage'] = ProcessingStage.WRITING_COMPLETE
            updated_state['writing_completed_at'] = datetime.now(timezone.utc).isoformat()
            updated_state['progress_percentage'] = 90.0
            logger.info(f"Chapter writing phase completed for book {state['book_id']}")
            
        else:
            # Handle writing error
            error_message = writing_result.get('error_message', 'Unknown chapter writing error')
            retry_count = writing_result.get('retry_count', 0)
            
            if retry_count < writer.config.max_retry_attempts:
                # Retry the chapter
                logger.warning(f"Chapter {current_chapter_num + 1} failed, will retry. Error: {error_message}")
                
                retry_result = writer.retry_failed_chapter(state, current_chapter_num + 1, retry_count)
                
                if retry_result['status'] == 'success':
                    # Update with retry success
                    chapter_index = retry_result['chapter_number'] - 1
                    updated_state['chapters'][chapter_index] = retry_result['updated_chapter']
                    updated_state['current_chapter'] = current_chapter_num + 1
                    updated_state['total_word_count'] += retry_result['updated_chapter'].get('word_count', 0)
                    
                    # Calculate progress
                    writing_progress = (current_chapter_num + 1) / total_chapters * 40
                    updated_state['progress_percentage'] = 50.0 + writing_progress
                    
                    logger.info(f"Chapter {retry_result['chapter_number']} completed on retry for book {state['book_id']}")
                else:
                    # Retry also failed
                    updated_state['errors'].append(f"Chapter {current_chapter_num + 1} writing failed after {retry_count + 1} attempts: {error_message}")
                    updated_state['retry_count'] = retry_count + 1
                    updated_state['last_error_at'] = datetime.now(timezone.utc).isoformat()
                    
                    # If max retries exceeded, mark as failed
                    if retry_count >= writer.config.max_retry_attempts - 1:
                        updated_state['current_stage'] = ProcessingStage.FAILED
                        logger.error(f"Chapter writing failed for book {state['book_id']} after maximum retries")
            else:
                # Max retries exceeded
                updated_state['errors'].append(f"Chapter writing failed: {error_message}")
                updated_state['current_stage'] = ProcessingStage.FAILED
                logger.error(f"Chapter writing failed for book {state['book_id']}: {error_message}")
        
        return updated_state
        
    except Exception as e:
        logger.error(f"Error in chapter writing for book {state['book_id']}: {e}")
        updated_state = state.copy()
        updated_state['errors'].append(f"Chapter writing node failed: {str(e)}")
        updated_state['current_stage'] = ProcessingStage.FAILED
        updated_state['last_error_at'] = datetime.now(timezone.utc).isoformat()
        return updated_state


from datetime import datetime, timezone
from typing import Literal
from musequill.config.logging import get_logger
from musequill.agents.agent_state import BookWritingState, ProcessingStage

logger = get_logger(__name__)


def quality_review_node(state: BookWritingState) -> BookWritingState:
    """
    Quality Review Node
    
    Reviews the completed book for consistency, quality, and completeness.
    May trigger revisions if quality standards are not met.
    """
    try:
        logger.info(f"Starting quality review for book {state['book_id']}")
        
        # Import here to avoid circular imports
        from musequill.agents.quality_reviewer.quality_reviewer_agent import QualityReviewerAgent
        
        # Create quality reviewer instance
        reviewer = QualityReviewerAgent()
        
        # Validate that we have completed chapters to review
        completed_chapters = [ch for ch in state['chapters'] if ch.get('status') == 'complete' and ch.get('content')]
        
        if not completed_chapters:
            logger.error(f"No completed chapters found for quality review of book {state['book_id']}")
            updated_state = state.copy()
            updated_state['errors'].append("No completed chapters available for quality review")
            updated_state['current_stage'] = ProcessingStage.FAILED
            return updated_state
        
        logger.info(f"Reviewing {len(completed_chapters)} completed chapters for book {state['book_id']}")
        
        # Conduct comprehensive quality review
        review_results = reviewer.review_book_quality(state)
        
        # Update state based on review results
        updated_state = state.copy()
        
        if review_results['status'] == 'success':
            # Update state with review results
            updated_state['quality_score'] = review_results['overall_quality_score']
            updated_state['current_stage'] = ProcessingStage.REVIEW
            
            # Store detailed review information
            if 'review_notes' not in updated_state:
                updated_state['review_notes'] = []
            
            # Add review summary to notes
            review_summary = (
                f"Quality Review Complete - Score: {review_results['overall_quality_score']:.2f}/1.0, "
                f"Meets Threshold: {review_results['meets_quality_threshold']}, "
                f"Requires Revision: {review_results['requires_revision']}"
            )
            updated_state['review_notes'].append(review_summary)
            
            # Add detailed feedback if available
            if review_results.get('assessment_summary'):
                updated_state['review_notes'].append(f"Assessment: {review_results['assessment_summary']}")
            
            # Store quality assessment metadata
            updated_state['metadata']['quality_review'] = {
                'overall_score': review_results['overall_quality_score'],
                'meets_threshold': review_results['meets_quality_threshold'],
                'requires_revision': review_results['requires_revision'],
                'revision_urgency': review_results.get('revision_urgency', 'medium'),
                'approval_recommendation': review_results['approval_recommendation'],
                'review_date': review_results['quality_assessment'].created_at,
                'chapters_reviewed': len(review_results['chapter_metrics']),
                'consistency_score': review_results['consistency_metrics']['overall_consistency_score'],
                'priority_revision_areas': review_results.get('priority_revision_areas', [])
            }
            
            # Determine next steps based on review outcome
            if review_results['requires_revision']:
                # Book requires revision
                logger.info(f"Book {state['book_id']} requires revision - Score: {review_results['overall_quality_score']:.2f}, Urgency: {review_results.get('revision_urgency', 'medium')}")
                
                # Check revision count limits
                current_revisions = updated_state.get('revision_count', 0)
                max_revisions = 3  # Could be configurable
                
                if current_revisions >= max_revisions:
                    # Maximum revisions reached
                    logger.warning(f"Book {state['book_id']} reached maximum revisions ({max_revisions})")
                    
                    if review_results['approval_recommendation'] == 'escalate_to_human_review':
                        updated_state['current_stage'] = ProcessingStage.FAILED
                        updated_state['errors'].append(f"Quality review failed after {max_revisions} revision cycles - human review required")
                        updated_state['review_notes'].append("ESCALATED: Maximum revisions reached - requires human intervention")
                    else:
                        # Proceed to final assembly despite quality issues
                        updated_state['current_stage'] = ProcessingStage.REVIEW
                        updated_state['review_notes'].append("ACCEPTED: Proceeding despite quality concerns due to revision limit")
                        logger.info(f"Accepting book {state['book_id']} despite quality concerns - revision limit reached")
                
                else:
                    # Set up for revision cycle
                    updated_state['revision_count'] = current_revisions + 1
                    updated_state['current_stage'] = ProcessingStage.REVIEW
                    
                    # Store revision guidance
                    revision_guidance = {
                        'revision_strategy': review_results.get('revision_strategy', 'General quality improvement needed'),
                        'priority_areas': review_results.get('priority_revision_areas', []),
                        'expected_improvements': review_results['quality_assessment'].revision_priority_areas,
                        'urgency': review_results.get('revision_urgency', 'medium'),
                        'detailed_feedback': review_results.get('detailed_feedback', '')
                    }
                    updated_state['metadata']['revision_guidance'] = revision_guidance
                    
                    # Add specific revision notes
                    updated_state['review_notes'].append(f"REVISION REQUIRED: Cycle {current_revisions + 1}/{max_revisions}")
                    updated_state['review_notes'].append(f"Priority Areas: {', '.join(review_results.get('priority_revision_areas', []))}")
                    
                    logger.info(f"Book {state['book_id']} scheduled for revision cycle {current_revisions + 1}")
            
            else:
                # Book approved - proceed to final assembly
                logger.info(f"Book {state['book_id']} approved - Quality score: {review_results['overall_quality_score']:.2f}")
                updated_state['current_stage'] = ProcessingStage.REVIEW
                updated_state['review_notes'].append("APPROVED: Quality review passed - proceeding to final assembly")
            
            # Update progress
            if updated_state['current_stage'] == ProcessingStage.REVIEW:
                updated_state['progress_percentage'] = 95.0  # Near completion
            
            logger.info(f"Quality review completed for book {state['book_id']} - Status: {updated_state['current_stage']}")
            
        else:
            # Review process failed
            error_message = review_results.get('error_message', 'Unknown quality review error')
            logger.error(f"Quality review failed for book {state['book_id']}: {error_message}")
            
            updated_state['errors'].append(f"Quality review failed: {error_message}")
            updated_state['current_stage'] = ProcessingStage.FAILED
            updated_state['last_error_at'] = datetime.now(timezone.utc).isoformat()
            
            # Add fallback review notes
            updated_state['review_notes'] = updated_state.get('review_notes', [])
            updated_state['review_notes'].append("ERROR: Quality review process failed - manual review required")
        
        return updated_state
        
    except Exception as e:
        logger.error(f"Error in quality review for book {state['book_id']}: {e}")
        updated_state = state.copy()
        updated_state['errors'].append(f"Quality review node failed: {str(e)}")
        updated_state['current_stage'] = ProcessingStage.FAILED
        updated_state['last_error_at'] = datetime.now(timezone.utc).isoformat()
        
        # Add emergency review notes
        updated_state['review_notes'] = updated_state.get('review_notes', [])
        updated_state['review_notes'].append("CRITICAL ERROR: Quality review system failure - immediate manual intervention required")
        
        return updated_state


def should_revise_or_complete(state: BookWritingState) -> Literal["chapter_writer", "final_assembler"]:
    """
    Determine if revisions are needed or if we can proceed to final assembly.
    This function is used by the orchestrator to route after quality review.
    """
    try:
        # Check if we're in review stage and have review results
        if state['current_stage'] != ProcessingStage.REVIEW:
            logger.warning(f"Unexpected stage for revision decision: {state['current_stage']}")
            return "final_assembler"
        
        # Check revision guidance in metadata
        revision_guidance = state.get('metadata', {}).get('revision_guidance')
        quality_review = state.get('metadata', {}).get('quality_review')
        
        # If we have explicit revision guidance, use it
        if revision_guidance:
            priority_areas = revision_guidance.get('priority_areas', [])
            urgency = revision_guidance.get('urgency', 'medium')
            
            logger.info(f"Book {state['book_id']} revision decision - Priority areas: {priority_areas}, Urgency: {urgency}")
            
            # If there are priority areas that need revision, go back to chapter writer
            if priority_areas and urgency in ['high', 'critical', 'medium']:
                logger.info(f"Routing book {state['book_id']} back to chapter writer for revision")
                return "chapter_writer"
        
        # Check quality score threshold
        quality_score = state.get('quality_score', 0.0)
        quality_threshold = 0.8  # Could be configurable
        
        if quality_score < quality_threshold:
            # Check revision count to avoid infinite loops
            revision_count = state.get('revision_count', 0)
            max_revisions = 3  # Could be configurable
            
            if revision_count < max_revisions:
                logger.info(f"Book {state['book_id']} quality score {quality_score:.2f} below threshold {quality_threshold} - routing for revision")
                return "chapter_writer"
            else:
                logger.warning(f"Book {state['book_id']} reached max revisions ({max_revisions}) - proceeding to final assembly")
        
        # Check review notes for explicit revision requirements
        review_notes = state.get('review_notes', [])
        if any('REVISION REQUIRED' in note for note in review_notes):
            revision_count = state.get('revision_count', 0)
            max_revisions = 3
            
            if revision_count < max_revisions:
                logger.info(f"Book {state['book_id']} has explicit revision requirement - routing to chapter writer")
                return "chapter_writer"
        
        # Default to final assembly
        logger.info(f"Book {state['book_id']} approved for final assembly - Quality score: {quality_score:.2f}")
        return "final_assembler"
        
    except Exception as e:
        logger.error(f"Error in revision decision for book {state['book_id']}: {e}")
        # Default to final assembly in case of error
        return "final_assembler"


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