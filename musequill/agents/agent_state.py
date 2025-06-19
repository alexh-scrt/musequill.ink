"""
Agent State Definition for Book Writing Orchestration

Defines the state structure that flows through the research and writing pipeline.
"""

from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class ProcessingStage(str, Enum):
    """Stages in the book writing pipeline."""
    INITIALIZED = "initialized"
    RESEARCH_PLANNING = "research_planning"
    RESEARCHING = "researching"
    RESEARCH_COMPLETE = "research_complete"
    WRITING_PLANNING = "writing_planning"
    WRITING = "writing"
    WRITING_COMPLETE = "writing_complete"
    REVIEW = "review"
    COMPLETE = "complete"
    FAILED = "failed"


class ResearchQuery(TypedDict):
    """Structure for individual research queries."""
    query: str
    priority: int  # 1-5, 5 being highest priority
    query_type: str  # e.g., "background", "technical", "examples", "expert_opinion"
    status: str  # "pending", "completed", "failed"
    results_count: Optional[int]
    created_at: str


class Chapter(TypedDict):
    """Structure for individual chapters."""
    chapter_number: int
    title: str
    description: str
    target_word_count: int
    status: str  # "planned", "researching", "writing", "complete", "review"
    content: Optional[str]
    research_chunks_used: Optional[List[str]]  # IDs of research chunks used
    word_count: Optional[int]
    created_at: Optional[str]
    completed_at: Optional[str]


class BookWritingState(TypedDict):
    """
    Complete state for the book writing orchestration pipeline.
    This state flows through all agents in the pipeline.
    """
    
    # Book Identification
    book_id: str
    orchestration_id: str  # Unique ID for this orchestration run
    thread_id: str  # LangGraph thread ID
    
    # Book Metadata
    title: str
    description: str
    genre: str
    target_word_count: int
    target_audience: Optional[str]
    author_preferences: Optional[Dict[str, Any]]
    
    # Planning Information
    outline: Dict[str, Any]  # Original book outline/plan
    chapters: List[Chapter]  # Detailed chapter information
    
    # Processing Status
    current_stage: ProcessingStage
    processing_started_at: str
    processing_updated_at: str
    
    # Research Phase
    research_queries: List[ResearchQuery]
    research_strategy: Optional[str]  # LLM-generated research strategy
    total_research_chunks: int
    research_completed_at: Optional[str]
    
    # Writing Phase
    current_chapter: int  # Currently being written chapter (0-based)
    writing_strategy: Optional[str]  # LLM-generated writing strategy
    writing_style_guide: Optional[str]  # Style guidelines for consistency
    total_word_count: int
    writing_started_at: Optional[str]
    writing_completed_at: Optional[str]
    
    # Quality Control
    review_notes: Optional[List[str]]
    revision_count: int
    quality_score: Optional[float]  # 0-1 quality assessment
    
    # Error Handling
    errors: List[str]  # Any errors encountered during processing
    retry_count: int
    last_error_at: Optional[str]
    
    # Progress Tracking
    progress_percentage: float  # 0-100 overall progress
    estimated_completion_time: Optional[str]
    
    # Final Output
    final_book_content: Optional[str]  # Complete book when finished
    metadata: Dict[str, Any]  # Additional metadata for storage