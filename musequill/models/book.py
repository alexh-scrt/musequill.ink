"""
Complete Book Model Classes for MuseQuill.ink
A comprehensive data model system that can represent any book from conception to publication.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
from pydantic import BaseModel, Field, validator, root_validator
from datetime import datetime, date
from enum import Enum
from uuid import UUID, uuid4
from decimal import Decimal


# ============================================================================
# Core Enums and Base Types
# ============================================================================

class BookStatus(str, Enum):
    """Book development status."""
    CONCEPT = "concept"
    REQUIREMENTS_DEFINED = "requirements_defined"
    OUTLINED = "outlined"
    IN_PROGRESS = "in_progress"
    FIRST_DRAFT_COMPLETE = "first_draft_complete"
    REVISING = "revising"
    EDITING = "editing"
    FINAL_DRAFT = "final_draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class ContentType(str, Enum):
    """Types of content within a book."""
    FRONT_MATTER = "front_matter"
    CHAPTER = "chapter"
    SECTION = "section"
    SCENE = "scene"
    APPENDIX = "appendix"
    GLOSSARY = "glossary"
    INDEX = "index"
    BIBLIOGRAPHY = "bibliography"


class WritingPhase(str, Enum):
    """Current writing phase."""
    PLANNING = "planning"
    RESEARCH = "research"
    OUTLINING = "outlining"
    DRAFTING = "drafting"
    REVISING = "revising"
    EDITING = "editing"
    PROOFREADING = "proofreading"
    FORMATTING = "formatting"
    PUBLISHING = "publishing"


class PriorityLevel(str, Enum):
    """Priority levels for content elements."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    OPTIONAL = "optional"


# ============================================================================
# Author and Project Models
# ============================================================================

class Author(BaseModel):
    """Author information and credentials."""
    id: UUID = Field(default_factory=uuid4)
    name: str
    pen_name: Optional[str] = None
    email: str
    bio: Optional[str] = None
    
    # Professional Information
    credentials: List[str] = Field(default_factory=list)
    expertise_areas: List[str] = Field(default_factory=list)
    previous_works: List[str] = Field(default_factory=list)
    awards: List[str] = Field(default_factory=list)
    
    # Platform Information
    website: Optional[str] = None
    social_media: Dict[str, str] = Field(default_factory=dict)
    mailing_list_size: Optional[int] = None
    platform_reach: Optional[str] = None
    
    # Writing Preferences
    preferred_genres: List[str] = Field(default_factory=list)
    writing_style_notes: Optional[str] = None
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None


class Project(BaseModel):
    """Top-level project container (can contain multiple books)."""
    id: UUID = Field(default_factory=uuid4)
    name: str
    description: Optional[str] = None
    project_type: str = Field(default="single_book")  # single_book, series, collection
    
    # Project Management
    status: str = Field(default="active")
    priority: PriorityLevel = Field(default=PriorityLevel.MEDIUM)
    start_date: Optional[date] = None
    target_completion_date: Optional[date] = None
    
    # Team
    primary_author_id: UUID
    collaborators: List[UUID] = Field(default_factory=list)
    
    # Project Settings
    default_settings: Dict[str, Any] = Field(default_factory=dict)
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None


# ============================================================================
# Core Book Models
# ============================================================================

class Book(BaseModel):
    """Main book entity - the central model for any book."""
    id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    
    # Basic Information
    title: str
    subtitle: Optional[str] = None
    working_title: Optional[str] = None
    title_alternatives: List[str] = Field(default_factory=list)
    
    # Content Classification
    genre: str
    sub_genres: List[str] = Field(default_factory=list)
    category: str  # Fiction, Non-fiction, etc.
    subject_areas: List[str] = Field(default_factory=list)
    
    # Physical Specifications
    target_word_count: int
    current_word_count: int = 0
    target_page_count: Optional[int] = None
    current_page_count: int = 0
    
    # Status and Progress
    status: BookStatus = Field(default=BookStatus.CONCEPT)
    completion_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    current_phase: WritingPhase = Field(default=WritingPhase.PLANNING)
    
    # Content Structure
    structure_type: str = Field(default="standard")  # standard, academic, technical, narrative
    number_of_parts: int = Field(default=1)
    target_chapter_count: int
    current_chapter_count: int = 0
    
    # Requirements Reference
    requirements_id: Optional[UUID] = None  # Links to BookRequirements
    
    # Publishing Information
    isbn: Optional[str] = None
    publication_date: Optional[date] = None
    publisher: Optional[str] = None
    edition: str = Field(default="1st")
    copyright_year: Optional[int] = None
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    last_edited_at: Optional[datetime] = None
    
    # Calculated Properties
    @property
    def estimated_reading_time_minutes(self) -> int:
        """Estimate reading time based on current word count."""
        # Average reading speed: 200-250 words per minute
        return int(self.current_word_count / 225) if self.current_word_count > 0 else 0
    
    @property
    def words_remaining(self) -> int:
        """Calculate words remaining to reach target."""
        return max(0, self.target_word_count - self.current_word_count)
    
    @property
    def progress_ratio(self) -> float:
        """Calculate progress as ratio (0.0 to 1.0)."""
        if self.target_word_count == 0:
            return 0.0
        return min(1.0, self.current_word_count / self.target_word_count)


class BookPart(BaseModel):
    """Book parts/sections for multi-part books."""
    id: UUID = Field(default_factory=uuid4)
    book_id: UUID
    
    # Structure
    part_number: int
    title: str
    subtitle: Optional[str] = None
    description: Optional[str] = None
    
    # Content
    target_word_count: int
    current_word_count: int = 0
    target_chapter_count: int
    current_chapter_count: int = 0
    
    # Status
    status: str = Field(default="planned")
    completion_percentage: float = Field(default=0.0)
    
    # Purpose and Goals
    purpose: Optional[str] = None
    learning_objectives: List[str] = Field(default_factory=list)
    key_concepts: List[str] = Field(default_factory=list)
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None


class Chapter(BaseModel):
    """Individual chapter within a book."""
    id: UUID = Field(default_factory=uuid4)
    book_id: UUID
    part_id: Optional[UUID] = None
    
    # Structure
    chapter_number: int
    title: str
    subtitle: Optional[str] = None
    
    # Content Planning
    summary: Optional[str] = None
    outline: Optional[str] = None
    key_points: List[str] = Field(default_factory=list)
    learning_objectives: List[str] = Field(default_factory=list)
    
    # Specifications
    target_word_count: int = Field(default=3000)
    current_word_count: int = 0
    estimated_reading_time: Optional[int] = None
    
    # Status and Progress
    status: str = Field(default="planned")  # planned, outlined, drafted, revised, final
    completion_percentage: float = Field(default=0.0)
    priority: PriorityLevel = Field(default=PriorityLevel.MEDIUM)
    
    # Content Elements
    includes_exercises: bool = False
    includes_examples: bool = False
    includes_case_studies: bool = False
    includes_code: bool = False
    includes_images: bool = False
    includes_tables: bool = False
    
    # Dependencies
    prerequisite_chapters: List[UUID] = Field(default_factory=list)
    dependent_chapters: List[UUID] = Field(default_factory=list)
    
    # Author Notes
    author_notes: Optional[str] = None
    research_needed: List[str] = Field(default_factory=list)
    technical_challenges: List[str] = Field(default_factory=list)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    last_edited_at: Optional[datetime] = None


class Section(BaseModel):
    """Sections within chapters."""
    id: UUID = Field(default_factory=uuid4)
    chapter_id: UUID
    
    # Structure
    section_number: Optional[int] = None
    title: str
    level: int = Field(default=1)  # Heading level (H1, H2, H3, etc.)
    
    # Content
    content_type: ContentType = Field(default=ContentType.SECTION)
    summary: Optional[str] = None
    target_word_count: int = Field(default=500)
    current_word_count: int = 0
    
    # Purpose
    purpose: Optional[str] = None
    key_concepts: List[str] = Field(default_factory=list)
    
    # Status
    status: str = Field(default="planned")
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None


# ============================================================================
# Content and Writing Models
# ============================================================================

class Content(BaseModel):
    """Actual written content for any book element."""
    id: UUID = Field(default_factory=uuid4)
    
    # Content Identity
    content_type: ContentType
    parent_id: UUID  # ID of chapter, section, etc.
    
    # Content Data
    title: Optional[str] = None
    text_content: str = ""
    formatted_content: Optional[str] = None  # HTML, Markdown, etc.
    
    # Version Control
    version: str = Field(default="1.0")
    is_current_version: bool = True
    
    # Metrics
    word_count: int = 0
    character_count: int = 0
    paragraph_count: int = 0
    sentence_count: int = 0
    
    # Status
    status: str = Field(default="draft")  # draft, review, approved, published
    quality_score: Optional[float] = None  # AI-generated quality assessment
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    
    @validator('word_count', pre=True, always=True)
    @classmethod
    def calculate_word_count(cls, v, values):
        """Auto-calculate word count from text content."""
        text = values.get('text_content', '')
        return len(text.split()) if text else 0


class ContentVersion(BaseModel):
    """Version history for content changes."""
    id: UUID = Field(default_factory=uuid4)
    content_id: UUID
    
    # Version Information
    version_number: str
    version_name: Optional[str] = None
    change_summary: str
    
    # Content Snapshot
    text_content: str
    word_count: int
    
    # Change Metadata
    change_type: str  # create, edit, revision, major_rewrite
    changed_by: UUID  # Author ID
    change_reason: Optional[str] = None
    
    # AI Assistance
    ai_assistance_used: bool = False
    ai_suggestions_applied: List[str] = Field(default_factory=list)
    
    created_at: datetime = Field(default_factory=datetime.now)


class Outline(BaseModel):
    """Hierarchical outline for book structure."""
    id: UUID = Field(default_factory=uuid4)
    book_id: UUID
    
    # Outline Metadata
    outline_type: str = Field(default="chapter")  # chapter, detailed, scene
    outline_format: str = Field(default="hierarchical")  # hierarchical, linear, mind_map
    
    # Structure
    items: List['OutlineItem'] = Field(default_factory=list)
    
    # Status
    status: str = Field(default="draft")
    completion_level: str = Field(default="basic")  # basic, detailed, comprehensive
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None


class OutlineItem(BaseModel):
    """Individual item in an outline."""
    id: UUID = Field(default_factory=uuid4)
    outline_id: UUID
    parent_item_id: Optional[UUID] = None
    
    # Structure
    sequence: int
    level: int  # Nesting level
    item_type: str  # chapter, section, point, note
    
    # Content
    title: str
    description: Optional[str] = None
    notes: Optional[str] = None
    
    # Specifications
    estimated_word_count: Optional[int] = None
    priority: PriorityLevel = Field(default=PriorityLevel.MEDIUM)
    
    # Child Items
    children: List['OutlineItem'] = Field(default_factory=list)
    
    created_at: datetime = Field(default_factory=datetime.now)


# ============================================================================
# Research and Reference Models
# ============================================================================

class ResearchTopic(BaseModel):
    """Research topics needed for the book."""
    id: UUID = Field(default_factory=uuid4)
    book_id: UUID
    
    # Topic Information
    topic: str
    description: Optional[str] = None
    research_question: Optional[str] = None
    
    # Scope and Priority
    scope: str = Field(default="chapter")  # book, part, chapter, section
    priority: PriorityLevel = Field(default=PriorityLevel.MEDIUM)
    complexity: int = Field(default=3, ge=1, le=5)
    
    # Status
    status: str = Field(default="identified")  # identified, in_progress, completed, verified
    completion_percentage: float = Field(default=0.0)
    
    # Research Plan
    research_methods: List[str] = Field(default_factory=list)
    sources_needed: List[str] = Field(default_factory=list)
    expert_interviews_needed: List[str] = Field(default_factory=list)
    
    # Dependencies
    applies_to_chapters: List[UUID] = Field(default_factory=list)
    prerequisite_research: List[UUID] = Field(default_factory=list)
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None


class Source(BaseModel):
    """Research sources and references."""
    id: UUID = Field(default_factory=uuid4)
    
    # Source Information
    title: str
    author: Optional[str] = None
    source_type: str  # book, article, website, interview, video, podcast
    url: Optional[str] = None
    
    # Publication Details
    publication_date: Optional[date] = None
    publisher: Optional[str] = None
    journal: Optional[str] = None
    volume: Optional[str] = None
    pages: Optional[str] = None
    doi: Optional[str] = None
    isbn: Optional[str] = None
    
    # Content Assessment
    relevance_score: int = Field(default=5, ge=1, le=10)
    credibility_score: int = Field(default=5, ge=1, le=10)
    bias_assessment: Optional[str] = None
    
    # Usage
    key_insights: List[str] = Field(default_factory=list)
    quotes: List[str] = Field(default_factory=list)
    summary: Optional[str] = None
    
    # Application
    applies_to_topics: List[UUID] = Field(default_factory=list)
    used_in_chapters: List[UUID] = Field(default_factory=list)
    
    # Metadata
    access_date: Optional[date] = None
    status: str = Field(default="identified")  # identified, accessed, reviewed, applied
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None


class Citation(BaseModel):
    """Citations and references within content."""
    id: UUID = Field(default_factory=uuid4)
    source_id: UUID
    content_id: UUID
    
    # Citation Details
    citation_type: str = Field(default="reference")  # reference, quote, paraphrase, statistic
    page_reference: Optional[str] = None
    specific_location: Optional[str] = None
    
    # Context
    context_description: str
    purpose: str  # support_argument, provide_evidence, show_example, etc.
    
    # Formatting
    citation_style: str = Field(default="APA")  # APA, MLA, Chicago, etc.
    formatted_citation: Optional[str] = None
    
    created_at: datetime = Field(default_factory=datetime.now)


# ============================================================================
# Quality and Review Models
# ============================================================================

class QualityMetrics(BaseModel):
    """Quality assessment metrics for content."""
    id: UUID = Field(default_factory=uuid4)
    content_id: UUID
    
    # Readability Metrics
    reading_level: Optional[str] = None
    flesch_score: Optional[float] = None
    gunning_fog_index: Optional[float] = None
    
    # Content Quality
    clarity_score: Optional[float] = None
    coherence_score: Optional[float] = None
    engagement_score: Optional[float] = None
    
    # Technical Quality
    grammar_score: Optional[float] = None
    spelling_accuracy: Optional[float] = None
    style_consistency: Optional[float] = None
    
    # AI Assessment
    ai_quality_score: Optional[float] = None
    ai_suggestions: List[str] = Field(default_factory=list)
    
    # Human Assessment
    human_review_score: Optional[float] = None
    reviewer_comments: Optional[str] = None
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None


class Review(BaseModel):
    """Reviews and feedback on content."""
    id: UUID = Field(default_factory=uuid4)
    content_id: UUID
    reviewer_id: UUID
    
    # Review Details
    review_type: str = Field(default="content")  # content, technical, copy_edit, proof
    review_stage: str = Field(default="draft")  # outline, draft, revision, final
    
    # Assessment
    overall_rating: int = Field(ge=1, le=10)
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    
    # Specific Feedback
    feedback_comments: str
    line_comments: List[Dict[str, Any]] = Field(default_factory=list)  # Position-specific comments
    
    # Status
    review_status: str = Field(default="pending")  # pending, completed, addressed
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None


# ============================================================================
# Production and Publishing Models
# ============================================================================

class Manuscript(BaseModel):
    """Complete manuscript compilation."""
    id: UUID = Field(default_factory=uuid4)
    book_id: UUID
    
    # Manuscript Details
    title: str
    version: str = Field(default="1.0")
    manuscript_type: str = Field(default="complete")  # complete, partial, sample
    
    # Compilation
    compiled_content: str = ""
    content_order: List[UUID] = Field(default_factory=list)  # Ordered content IDs
    
    # Formatting
    format_type: str = Field(default="manuscript")  # manuscript, ebook, print, web
    style_guide: Optional[str] = None
    formatting_notes: Optional[str] = None
    
    # Metrics
    total_word_count: int = 0
    total_page_count: int = 0
    total_chapters: int = 0
    
    # Status
    status: str = Field(default="draft")  # draft, review, final, published
    last_compiled: datetime = Field(default_factory=datetime.now)
    
    # Export Information
    export_formats: List[str] = Field(default_factory=list)  # pdf, docx, epub, html
    file_paths: Dict[str, str] = Field(default_factory=dict)
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None


class PublishingPlan(BaseModel):
    """Publishing strategy and timeline."""
    id: UUID = Field(default_factory=uuid4)
    book_id: UUID
    
    # Publishing Strategy
    publishing_route: str  # traditional, self_published, hybrid
    target_publishers: List[str] = Field(default_factory=list)
    target_platforms: List[str] = Field(default_factory=list)
    
    # Market Positioning
    target_market: str
    competitive_titles: List[str] = Field(default_factory=list)
    unique_selling_points: List[str] = Field(default_factory=list)
    marketing_hooks: List[str] = Field(default_factory=list)
    
    # Timeline
    manuscript_deadline: Optional[date] = None
    editing_deadline: Optional[date] = None
    design_deadline: Optional[date] = None
    publication_date: Optional[date] = None
    marketing_launch_date: Optional[date] = None
    
    # Requirements
    word_count_requirements: Optional[Tuple[int, int]] = None  # min, max
    format_requirements: List[str] = Field(default_factory=list)
    
    # Budget and Resources
    budget_estimate: Optional[Decimal] = None
    resource_requirements: List[str] = Field(default_factory=list)
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None


class MarketingPlan(BaseModel):
    """Marketing and promotion strategy."""
    id: UUID = Field(default_factory=uuid4)
    book_id: UUID
    
    # Target Audience
    primary_audience: str
    secondary_audiences: List[str] = Field(default_factory=list)
    audience_demographics: Dict[str, Any] = Field(default_factory=dict)
    
    # Marketing Strategy
    key_messages: List[str] = Field(default_factory=list)
    marketing_channels: List[str] = Field(default_factory=list)
    promotional_activities: List[str] = Field(default_factory=list)
    
    # Content Marketing
    blog_post_topics: List[str] = Field(default_factory=list)
    social_media_strategy: Dict[str, Any] = Field(default_factory=dict)
    speaking_opportunities: List[str] = Field(default_factory=list)
    
    # Metrics and Goals
    sales_goals: Dict[str, int] = Field(default_factory=dict)  # platform -> target sales
    engagement_goals: Dict[str, int] = Field(default_factory=dict)
    success_metrics: List[str] = Field(default_factory=list)
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None


# ============================================================================
# Analytics and Progress Models
# ============================================================================

class WritingSession(BaseModel):
    """Individual writing session tracking."""
    id: UUID = Field(default_factory=uuid4)
    book_id: UUID
    author_id: UUID
    
    # Session Details
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_minutes: Optional[int] = None
    
    # Work Completed
    words_written: int = 0
    content_created: List[UUID] = Field(default_factory=list)  # Content IDs
    sections_completed: List[UUID] = Field(default_factory=list)
    
    # Focus and Goals
    session_goal: Optional[str] = None
    session_type: str = Field(default="writing")  # writing, editing, research, planning
    focus_areas: List[str] = Field(default_factory=list)
    
    # AI Assistance
    ai_interactions: int = 0
    ai_words_generated: int = 0
    ai_suggestions_accepted: int = 0
    
    # Session Assessment
    productivity_rating: Optional[int] = Field(None, ge=1, le=10)
    mood_rating: Optional[int] = Field(None, ge=1, le=10)
    energy_level: Optional[int] = Field(None, ge=1, le=10)
    
    # Notes
    session_notes: Optional[str] = None
    challenges_faced: List[str] = Field(default_factory=list)
    achievements: List[str] = Field(default_factory=list)
    
    created_at: datetime = Field(default_factory=datetime.now)


class ProgressTracker(BaseModel):
    """Progress tracking and analytics."""
    id: UUID = Field(default_factory=uuid4)
    book_id: UUID
    
    # Current Status
    current_word_count: int = 0
    target_word_count: int
    completion_percentage: float = 0.0
    
    # Progress Metrics
    daily_word_averages: Dict[str, int] = Field(default_factory=dict)  # date -> words
    weekly_progress: Dict[str, int] = Field(default_factory=dict)  # week -> words
    monthly_progress: Dict[str, int] = Field(default_factory=dict)  # month -> words
    
    # Velocity Tracking
    current_velocity: float = 0.0  # words per day
    projected_completion_date: Optional[date] = None
    
    # Milestones
    milestones_achieved: List[Dict[str, Any]] = Field(default_factory=list)
    upcoming_milestones: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Quality Trends
    quality_trend: List[float] = Field(default_factory=list)
    consistency_score: Optional[float] = None
    
    last_updated: datetime = Field(default_factory=datetime.now)


# Update forward references
Outline.model_rebuild()
OutlineItem.model_rebuild()


# ============================================================================
# Factory Functions and Utilities
# ============================================================================

def create_book_from_requirements(
    requirements: 'BookRequirements',  # From the provided class
    project_id: UUID,
    author_id: UUID
) -> Book:
    """Create a Book instance from BookRequirements."""
    
    return Book(
        project_id=project_id,
        title=requirements.title_ideas[0] if requirements.title_ideas else requirements.topic,
        genre=requirements.genre.value,
        sub_genres=[g.value for g in requirements.marketing.keywords_for_discovery[:3]],
        category="Fiction" if requirements.genre in ["Fiction", "Fantasy", "Science Fiction", "Romance", "Mystery", "Thriller", "Horror"] else "Non-fiction",
        target_word_count=requirements.target_word_count,
        target_chapter_count=requirements.number_of_chapters,
        structure_type=requirements.content_structure.value,
        requirements_id=None,  # Would link to stored requirements
        status=BookStatus.REQUIREMENTS_DEFINED,
        current_phase=WritingPhase.PLANNING
    )


def create_chapter_outline_from_book(book: Book) -> List[Chapter]:
    """Generate initial chapter outline from book specifications."""
    chapters = []
    words_per_chapter = book.target_word_count // book.target_chapter_count
    
    for i in range(1, book.target_chapter_count + 1):
        chapter = Chapter(
            book_id=book.id,
            chapter_number=i,
            title=f"Chapter {i}",
            target_word_count=words_per_chapter,
            status="planned"
        )
        chapters.append(chapter)
    
    return chapters


def calculate_book_metrics(book: Book, chapters: List[Chapter]) -> Dict[str, Any]:
    """Calculate comprehensive metrics for a book."""
    total_words = sum(c.current_word_count for c in chapters)
    completed_chapters = len([c for c in chapters if c.status == "final"])
    
    return {
        "word_count": {
            "current": total_words,
            "target": book.target_word_count,
            "percentage": (total_words / book.target_word_count * 100) if book.target_word_count > 0 else 0
        },
        "chapters": {
            "completed": completed_chapters,
            "total": len(chapters),
            "percentage": (completed_chapters / len(chapters) * 100) if chapters else 0
        },
        "estimated_pages": total_words // 250,  # ~250 words per page
        "estimated_reading_time": total_words // 225,  # ~225 words per minute
        "status_distribution": {
            status: len([c for c in chapters if c.status == status])
            for status in ["planned", "outlined", "drafted", "revised", "final"]
        }
    }


# Example usage and validation
if __name__ == "__main__":
    # Create a sample project and book
    project = Project(
        name="My First Book Project",
        description="A technical book about Python programming",
        primary_author_id=uuid4()
    )
    
    book = Book(
        project_id=project.id,
        title="Mastering Python Development",
        subtitle="A Comprehensive Guide",
        genre="Technical",
        target_word_count=75000,
        target_chapter_count=15,
        status=BookStatus.OUTLINED
    )
    
    # Create chapter outline
    chapters = create_chapter_outline_from_book(book)
    
    # Calculate metrics
    metrics = calculate_book_metrics(book, chapters)
    
    print(f"Created book: {book.title}")
    print(f"Chapters: {len(chapters)}")
    print(f"Target words: {book.target_word_count:,}")
    print(f"Metrics: {metrics}")


# ============================================================================
# Advanced Model Extensions
# ============================================================================

class BookSeries(BaseModel):
    """Model for book series management."""
    id: UUID = Field(default_factory=uuid4)
    name: str
    description: Optional[str] = None
    
    # Series Information
    planned_books: int
    published_books: int = 0
    series_status: str = Field(default="planning")  # planning, active, completed, abandoned
    
    # Continuity Management
    main_characters: List[UUID] = Field(default_factory=list)
    recurring_themes: List[str] = Field(default_factory=list)
    world_building_elements: List[UUID] = Field(default_factory=list)
    
    # Publication Strategy
    publication_order: List[UUID] = Field(default_factory=list)  # Book IDs in order
    release_schedule: Dict[str, date] = Field(default_factory=dict)  # Book ID -> release date
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None


class Character(BaseModel):
    """Character model for fiction books."""
    id: UUID = Field(default_factory=uuid4)
    book_id: Optional[UUID] = None
    series_id: Optional[UUID] = None
    
    # Basic Information
    name: str
    full_name: Optional[str] = None
    nicknames: List[str] = Field(default_factory=list)
    age: Optional[int] = None
    
    # Character Details
    role: str = Field(default="supporting")  # protagonist, antagonist, supporting, minor
    character_type: str = Field(default="human")  # human, animal, fantasy_creature, ai, etc.
    
    # Physical Description
    physical_description: Optional[str] = None
    distinguishing_features: List[str] = Field(default_factory=list)
    
    # Personality
    personality_traits: List[str] = Field(default_factory=list)
    motivations: List[str] = Field(default_factory=list)
    fears: List[str] = Field(default_factory=list)
    goals: List[str] = Field(default_factory=list)
    
    # Background
    backstory: Optional[str] = None
    occupation: Optional[str] = None
    education: Optional[str] = None
    family_background: Optional[str] = None
    
    # Story Arc
    character_arc: Optional[str] = None
    development_notes: Optional[str] = None
    
    # Relationships
    relationships: Dict[UUID, str] = Field(default_factory=dict)  # Character ID -> relationship type
    
    # Appearance Tracking
    first_appearance: Optional[UUID] = None  # Chapter/Section ID
    major_scenes: List[UUID] = Field(default_factory=list)  # Scene IDs
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None


class WorldBuilding(BaseModel):
    """World building elements for fiction."""
    id: UUID = Field(default_factory=uuid4)
    book_id: Optional[UUID] = None
    series_id: Optional[UUID] = None
    
    # World Information
    world_name: Optional[str] = None
    world_type: str = Field(default="realistic")  # realistic, fantasy, sci_fi, alternate_history
    
    # Geography and Environment
    locations: List[Dict[str, Any]] = Field(default_factory=list)
    climate_description: Optional[str] = None
    natural_features: List[str] = Field(default_factory=list)
    
    # Society and Culture
    cultures: List[Dict[str, Any]] = Field(default_factory=list)
    governments: List[Dict[str, Any]] = Field(default_factory=list)
    religions: List[Dict[str, Any]] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    
    # Technology and Magic
    technology_level: Optional[str] = None
    magic_system: Optional[Dict[str, Any]] = None
    supernatural_elements: List[str] = Field(default_factory=list)
    
    # History and Timeline
    historical_events: List[Dict[str, Any]] = Field(default_factory=list)
    current_timeline: Optional[str] = None
    
    # Rules and Constraints
    world_rules: List[str] = Field(default_factory=list)
    physical_laws: List[str] = Field(default_factory=list)
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None


class PlotStructure(BaseModel):
    """Plot structure and story beats."""
    id: UUID = Field(default_factory=uuid4)
    book_id: UUID
    
    # Structure Type
    structure_template: str = Field(default="three_act")  # three_act, hero_journey, save_the_cat, custom
    
    # Story Elements
    premise: Optional[str] = None
    inciting_incident: Optional[str] = None
    plot_points: List[Dict[str, Any]] = Field(default_factory=list)
    climax: Optional[str] = None
    resolution: Optional[str] = None
    
    # Conflict Structure
    main_conflict: Optional[str] = None
    subplot_conflicts: List[str] = Field(default_factory=list)
    internal_conflicts: List[str] = Field(default_factory=list)
    external_conflicts: List[str] = Field(default_factory=list)
    
    # Themes
    primary_theme: Optional[str] = None
    secondary_themes: List[str] = Field(default_factory=list)
    thematic_questions: List[str] = Field(default_factory=list)
    
    # Pacing
    act_breakdowns: List[Dict[str, Any]] = Field(default_factory=list)
    tension_curve: List[int] = Field(default_factory=list)  # Tension levels by chapter
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None


class ContentTemplate(BaseModel):
    """Reusable content templates."""
    id: UUID = Field(default_factory=uuid4)
    
    # Template Information
    name: str
    description: Optional[str] = None
    template_type: str = Field(default="chapter")  # chapter, section, exercise, case_study
    category: str = Field(default="general")  # technical, business, fiction, academic
    
    # Template Structure
    template_structure: Dict[str, Any] = Field(default_factory=dict)
    required_elements: List[str] = Field(default_factory=list)
    optional_elements: List[str] = Field(default_factory=list)
    
    # Content Guidelines
    writing_guidelines: Optional[str] = None
    style_notes: Optional[str] = None
    word_count_guidance: Optional[str] = None
    
    # Usage Tracking
    usage_count: int = 0
    last_used: Optional[datetime] = None
    
    # Customization Options
    customizable_fields: List[str] = Field(default_factory=list)
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None


class EditingWorkflow(BaseModel):
    """Editing and revision workflow management."""
    id: UUID = Field(default_factory=uuid4)
    book_id: UUID
    
    # Workflow Configuration
    workflow_type: str = Field(default="standard")  # standard, academic, technical, fiction
    editing_stages: List[str] = Field(default_factory=list)  # content_edit, line_edit, copy_edit, proofread
    
    # Current Stage
    current_stage: str = Field(default="content_edit")
    stage_progress: Dict[str, float] = Field(default_factory=dict)  # stage -> completion %
    
    # Quality Checkpoints
    quality_gates: List[Dict[str, Any]] = Field(default_factory=list)
    quality_standards: Dict[str, Any] = Field(default_factory=dict)
    
    # Reviewer Assignment
    assigned_editors: Dict[str, UUID] = Field(default_factory=dict)  # stage -> editor_id
    review_schedule: Dict[str, date] = Field(default_factory=dict)  # stage -> deadline
    
    # Workflow Status
    workflow_status: str = Field(default="not_started")  # not_started, in_progress, completed
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None


class Collaboration(BaseModel):
    """Collaboration and team management."""
    id: UUID = Field(default_factory=uuid4)
    book_id: UUID
    
    # Team Structure
    primary_author: UUID
    co_authors: List[UUID] = Field(default_factory=list)
    editors: List[UUID] = Field(default_factory=list)
    reviewers: List[UUID] = Field(default_factory=list)
    beta_readers: List[UUID] = Field(default_factory=list)
    
    # Permissions and Access
    access_levels: Dict[UUID, str] = Field(default_factory=dict)  # user_id -> permission_level
    content_assignments: Dict[UUID, List[UUID]] = Field(default_factory=dict)  # user_id -> chapter_ids
    
    # Communication
    team_communications: List[Dict[str, Any]] = Field(default_factory=list)
    shared_documents: List[UUID] = Field(default_factory=list)
    
    # Workflow Coordination
    approval_workflows: List[Dict[str, Any]] = Field(default_factory=list)
    review_cycles: List[Dict[str, Any]] = Field(default_factory=list)
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None


class AIAssistance(BaseModel):
    """AI assistance tracking and configuration."""
    id: UUID = Field(default_factory=uuid4)
    book_id: UUID
    
    # AI Configuration
    enabled_features: List[str] = Field(default_factory=list)  # writing_assist, research, editing, fact_check
    ai_writing_style: Optional[str] = None
    customization_preferences: Dict[str, Any] = Field(default_factory=dict)
    
    # Usage Statistics
    total_interactions: int = 0
    words_generated: int = 0
    suggestions_accepted: int = 0
    suggestions_rejected: int = 0
    
    # AI Performance
    user_satisfaction_ratings: List[int] = Field(default_factory=list)
    most_helpful_features: List[str] = Field(default_factory=list)
    
    # Learning and Adaptation
    user_feedback_patterns: Dict[str, Any] = Field(default_factory=dict)
    style_learning_data: Dict[str, Any] = Field(default_factory=dict)
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None


# ============================================================================
# Comprehensive Model Registry
# ============================================================================

class BookModelRegistry:
    """Registry of all book-related models."""
    
    # Core Models
    CORE_MODELS = [
        Author,
        Project,
        Book,
        BookPart,
        Chapter,
        Section,
    ]
    
    # Content Models
    CONTENT_MODELS = [
        Content,
        ContentVersion,
        Outline,
        OutlineItem,
    ]
    
    # Research Models
    RESEARCH_MODELS = [
        ResearchTopic,
        Source,
        Citation,
    ]
    
    # Quality Models
    QUALITY_MODELS = [
        QualityMetrics,
        Review,
    ]
    
    # Production Models
    PRODUCTION_MODELS = [
        Manuscript,
        PublishingPlan,
        MarketingPlan,
    ]
    
    # Analytics Models
    ANALYTICS_MODELS = [
        WritingSession,
        ProgressTracker,
    ]
    
    # Fiction-Specific Models
    FICTION_MODELS = [
        BookSeries,
        Character,
        WorldBuilding,
        PlotStructure,
    ]
    
    # Workflow Models
    WORKFLOW_MODELS = [
        ContentTemplate,
        EditingWorkflow,
        Collaboration,
        AIAssistance,
    ]
    
    @classmethod
    def get_all_models(cls):
        """Get all model classes."""
        return (
            cls.CORE_MODELS +
            cls.CONTENT_MODELS +
            cls.RESEARCH_MODELS +
            cls.QUALITY_MODELS +
            cls.PRODUCTION_MODELS +
            cls.ANALYTICS_MODELS +
            cls.FICTION_MODELS +
            cls.WORKFLOW_MODELS
        )
    
    @classmethod
    def get_model_by_name(cls, model_name: str):
        """Get model class by name."""
        for model in cls.get_all_models():
            if model.__name__ == model_name:
                return model
        return None
    
    @classmethod
    def get_models_by_category(cls, category: str):
        """Get models by category."""
        category_map = {
            "core": cls.CORE_MODELS,
            "content": cls.CONTENT_MODELS,
            "research": cls.RESEARCH_MODELS,
            "quality": cls.QUALITY_MODELS,
            "production": cls.PRODUCTION_MODELS,
            "analytics": cls.ANALYTICS_MODELS,
            "fiction": cls.FICTION_MODELS,
            "workflow": cls.WORKFLOW_MODELS,
        }
        return category_map.get(category.lower(), [])


# ============================================================================
# Database Schema Generator
# ============================================================================

def generate_database_schema():
    """Generate database schema information for all models."""
    schema_info = {}
    
    for model in BookModelRegistry.get_all_models():
        schema_info[model.__name__] = {
            "table_name": model.__name__.lower(),
            "fields": list(model.__fields__.keys()),
            "relationships": [],  # Would be populated based on UUID fields
            "indexes": [],  # Would be defined based on query patterns
        }
    
    return schema_info


# ============================================================================
# Validation and Integrity Checks
# ============================================================================

def validate_book_structure(book: Book, chapters: List[Chapter]) -> List[str]:
    """Validate book structure integrity."""
    issues = []
    
    # Check chapter numbering
    chapter_numbers = [c.chapter_number for c in chapters]
    expected_numbers = list(range(1, len(chapters) + 1))
    if sorted(chapter_numbers) != expected_numbers:
        issues.append("Chapter numbering is not sequential")
    
    # Check word count consistency
    total_chapter_words = sum(c.current_word_count for c in chapters)
    if abs(book.current_word_count - total_chapter_words) > 100:
        issues.append("Book word count doesn't match sum of chapter word counts")
    
    # Check target vs actual chapter count
    if len(chapters) != book.target_chapter_count:
        issues.append(f"Chapter count ({len(chapters)}) doesn't match target ({book.target_chapter_count})")
    
    return issues


# ============================================================================
# Example Usage and Testing
# ============================================================================

if __name__ == "__main__":
    # Demonstrate comprehensive book modeling
    
    # 1. Create a fiction book with full structure
    fiction_project = Project(
        name="Epic Fantasy Series",
        project_type="series",
        primary_author_id=uuid4()
    )
    
    fiction_book = Book(
        project_id=fiction_project.id,
        title="The Dragon's Legacy",
        subtitle="Book One of the Realm Chronicles",
        genre="Fantasy",
        sub_genres=["Epic Fantasy", "Adventure"],
        category="Fiction",
        target_word_count=120000,
        target_chapter_count=24,
        status=BookStatus.IN_PROGRESS
    )
    
    # 2. Create world building
    world = WorldBuilding(
        book_id=fiction_book.id,
        world_name="Aethermoor",
        world_type="fantasy",
        locations=[
            {"name": "Dragon's Peak", "type": "mountain", "importance": "high"},
            {"name": "Silverwood Forest", "type": "forest", "importance": "medium"}
        ],
        magic_system={
            "type": "elemental",
            "elements": ["fire", "water", "earth", "air"],
            "rules": ["Cannot create matter", "Requires mental focus", "Physical toll on user"]
        }
    )
    
    # 3. Create main characters
    protagonist = Character(
        book_id=fiction_book.id,
        name="Lyra Stormwind",
        age=22,
        role="protagonist",
        personality_traits=["brave", "impulsive", "loyal"],
        motivations=["save her village", "master her powers"],
        fears=["losing control", "failing her people"],
        backstory="Raised by her grandmother after her parents died in a dragon attack"
    )
    
    # 4. Create plot structure
    plot = PlotStructure(
        book_id=fiction_book.id,
        structure_template="hero_journey",
        premise="A young woman discovers she has dragon-bonding abilities",
        main_conflict="Ancient evil threatens to destroy the realm",
        primary_theme="Power comes with responsibility",
        secondary_themes=["friendship", "sacrifice", "growing up"]
    )
    
    # 5. Create non-fiction book for comparison
    nonfiction_book = Book(
        project_id=Project(
            name="Technical Writing Project",
            primary_author_id=uuid4()
        ).id,
        title="Modern Python Development",
        genre="Technical",
        category="Non-fiction",
        target_word_count=75000,
        target_chapter_count=15,
        status=BookStatus.OUTLINED
    )
    
    # 6. Create research topics for non-fiction
    research_topics = [
        ResearchTopic(
            book_id=nonfiction_book.id,
            topic="Python 3.12 New Features",
            priority=PriorityLevel.HIGH,
            research_methods=["documentation_review", "code_testing"]
        ),
        ResearchTopic(
            book_id=nonfiction_book.id,
            topic="FastAPI vs Flask Performance",
            priority=PriorityLevel.MEDIUM,
            research_methods=["benchmarking", "expert_interviews"]
        )
    ]
    
    # 7. Demonstrate model validation
    fiction_chapters = create_chapter_outline_from_book(fiction_book)
    validation_issues = validate_book_structure(fiction_book, fiction_chapters)
    
    print("=== MuseQuill Book Model Demonstration ===")
    print(f"Fiction Book: {fiction_book.title}")
    print(f"World: {world.world_name}")
    print(f"Protagonist: {protagonist.name}")
    print(f"Plot Theme: {plot.primary_theme}")
    print(f"Non-fiction Book: {nonfiction_book.title}")
    print(f"Research Topics: {len(research_topics)}")
    print(f"Model Registry: {len(BookModelRegistry.get_all_models())} total models")
    print(f"Validation Issues: {len(validation_issues)}")
    
    # 8. Show model relationships
    print("\n=== Model Categories ===")
    for category in ["core", "content", "fiction", "workflow"]:
        models = BookModelRegistry.get_models_by_category(category)
        print(f"{category.title()}: {[m.__name__ for m in models]}")
    
    print("\n=== Schema Information ===")
    schema = generate_database_schema()
    print(f"Total tables: {len(schema)}")
    print("Sample table fields:")
    for table_name, info in list(schema.items())[:3]:
        print(f"  {table_name}: {info['fields'][:5]}...")
    
    print("\n✅ Complete book modeling system ready for implementation!")