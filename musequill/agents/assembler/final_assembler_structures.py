"""
Final Assembler Agent Data Structures

Comprehensive data structures, models, and type definitions for the Final Assembler Agent.
Separated from the main agent file to improve maintainability and readability.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path
from enum import Enum

from langchain_core.pydantic_v1 import BaseModel


class DocumentFormat(str, Enum):
    """Supported document output formats."""
    PDF = "pdf"
    EPUB = "epub"
    DOCX = "docx"
    HTML = "html"
    MARKDOWN = "markdown"
    LATEX = "latex"
    TXT = "txt"


class ValidationStatus(str, Enum):
    """Document validation status."""
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    UNKNOWN = "unknown"
    FAILED = "failed"


class AssemblyPhase(str, Enum):
    """Assembly process phases."""
    VALIDATION = "validation"
    METADATA_GENERATION = "metadata_generation"
    CONTENT_ASSEMBLY = "content_assembly"
    FORMAT_GENERATION = "format_generation"
    QUALITY_VALIDATION = "quality_validation"
    FINALIZATION = "finalization"


@dataclass
class BookMetadata:
    """Complete book metadata for final assembly."""
    # Basic Information
    title: str
    author: str
    genre: str
    description: str
    isbn: Optional[str] = None
    publication_date: Optional[str] = None
    language: str = "en"
    
    # Content Statistics
    word_count: int = 0
    chapter_count: int = 0
    page_count: Optional[int] = None
    estimated_reading_time: Optional[int] = None  # minutes
    
    # Technical Metadata
    generation_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "1.0"
    format_version: str = "1.0"
    
    # Quality Information
    quality_score: Optional[float] = None
    content_rating: Optional[str] = None
    target_audience: Optional[str] = None
    reading_level: Optional[str] = None
    
    # Publisher Information
    publisher: str = "MuseQuill AI"
    publisher_url: Optional[str] = None
    copyright_year: Optional[int] = None
    copyright_holder: Optional[str] = None
    
    # Subject and Classification
    subjects: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    dewey_decimal: Optional[str] = None
    lcc_classification: Optional[str] = None
    
    # Technical Generation Info
    generation_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TableOfContentsEntry:
    """Enhanced table of contents entry."""
    level: int  # 1=chapter, 2=section, 3=subsection, etc.
    number: Optional[str]  # Chapter/section number
    title: str
    page_number: Optional[int] = None
    anchor_id: Optional[str] = None  # For HTML/digital formats
    subsections: List['TableOfContentsEntry'] = field(default_factory=list)
    
    # Content hints
    word_count: Optional[int] = None
    estimated_reading_time: Optional[int] = None  # minutes
    content_type: Optional[str] = None  # "chapter", "appendix", "bibliography", etc.


@dataclass
class IndexEntry:
    """Enhanced index entry with cross-references."""
    term: str
    page_references: List[int] = field(default_factory=list)
    see_also: List[str] = field(default_factory=list)
    see_instead: Optional[str] = None
    subentries: Dict[str, List[int]] = field(default_factory=dict)
    importance: int = 1  # 1-5, higher = more important
    category: Optional[str] = None  # "person", "concept", "place", etc.


@dataclass
class BibliographyEntry:
    """Enhanced bibliography entry with multiple citation styles."""
    # Core Information
    entry_id: str
    title: str
    authors: List[str] = field(default_factory=list)
    
    # Publication Details
    publication_type: str = "article"  # article, book, website, etc.
    journal: Optional[str] = None
    publisher: Optional[str] = None
    publication_date: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    
    # Digital Information
    url: Optional[str] = None
    doi: Optional[str] = None
    isbn: Optional[str] = None
    access_date: Optional[str] = None
    
    # Citation Styles
    apa_citation: Optional[str] = None
    mla_citation: Optional[str] = None
    chicago_citation: Optional[str] = None
    
    # Metadata
    reliability_score: Optional[float] = None
    usage_count: int = 0
    notes: Optional[str] = None


@dataclass
class FormattedDocument:
    """Enhanced formatted document with validation and metadata."""
    # Basic Information
    format_type: Union[DocumentFormat, str]  # Allow string for backward compatibility
    file_path: Union[Path, str]  # Allow string for backward compatibility
    file_size: int
    
    # Generation Information
    generation_time: float = 0.0
    generator_version: str = "1.0"
    template_used: Optional[str] = None
    
    # Validation (with backward compatibility)
    validation_status: Union[ValidationStatus, str] = ValidationStatus.UNKNOWN
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    
    # Content Statistics
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    character_count: Optional[int] = None
    
    # Technical Details
    file_hash: Optional[str] = None
    compression_used: bool = False
    compression_ratio: Optional[float] = None
    
    # Format-specific metadata
    format_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Handle backward compatibility conversions."""
        # Convert string format_type to enum if needed
        if isinstance(self.format_type, str):
            try:
                self.format_type = DocumentFormat(self.format_type.lower())
            except ValueError:
                # Keep as string if not a valid enum value
                pass
        
        # Convert string file_path to Path if needed
        if isinstance(self.file_path, str):
            self.file_path = Path(self.file_path)
        
        # Convert string validation_status to enum if needed
        if isinstance(self.validation_status, str):
            try:
                # Map common string values to enum
                status_mapping = {
                    'valid': ValidationStatus.VALID,
                    'invalid': ValidationStatus.INVALID,
                    'warning': ValidationStatus.WARNING,
                    'unknown': ValidationStatus.UNKNOWN,
                    'failed': ValidationStatus.FAILED
                }
                self.validation_status = status_mapping.get(self.validation_status.lower(), ValidationStatus.UNKNOWN)
            except (AttributeError, KeyError):
                self.validation_status = ValidationStatus.UNKNOWN


@dataclass
class AssemblyPhaseResult:
    """Result of an individual assembly phase."""
    phase: AssemblyPhase
    success: bool
    start_time: datetime
    end_time: datetime
    duration: float
    
    # Results
    output_data: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    # Metrics
    items_processed: int = 0
    bytes_processed: int = 0
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssemblyResults:
    """Complete assembly results with enhanced tracking."""
    # Basic Results (Required)
    book_id: str
    success: bool
    total_word_count: int
    
    # Timing Information (with defaults for backward compatibility)
    overall_start_time: datetime = field(default_factory=lambda: datetime.now())
    overall_end_time: datetime = field(default_factory=lambda: datetime.now())
    total_duration: float = 0.0
    
    # Backward compatibility fields
    assembly_time: Optional[float] = None  # Maps to total_duration
    error_messages: List[str] = field(default_factory=list)  # Maps to critical_errors
    
    # Content Information
    total_character_count: int = 0
    total_pages: Optional[int] = None
    
    # Generated Outputs
    generated_formats: List[FormattedDocument] = field(default_factory=list)
    failed_formats: List[str] = field(default_factory=list)
    
    # Document Components
    metadata: Optional[BookMetadata] = None
    table_of_contents: List[TableOfContentsEntry] = field(default_factory=list)
    index_entries: List[IndexEntry] = field(default_factory=list)
    bibliography: List[BibliographyEntry] = field(default_factory=list)
    
    # Process Tracking
    phase_results: List[AssemblyPhaseResult] = field(default_factory=list)
    
    # Validation and Quality
    validation_results: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Issues and Warnings
    critical_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Assembly Statistics
    assembly_statistics: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Handle backward compatibility mapping."""
        # Map assembly_time to total_duration if provided
        if self.assembly_time is not None and self.total_duration == 0.0:
            self.total_duration = self.assembly_time
        
        # Map error_messages to critical_errors
        if self.error_messages:
            self.critical_errors.extend(self.error_messages)
        
        # Set end time based on duration if not provided
        if self.total_duration > 0 and self.overall_end_time == self.overall_start_time:
            self.overall_end_time = datetime.now()


# Pydantic Models for LLM Interactions

class BookStructureValidationModel(BaseModel):
    """Pydantic model for LLM book structure validation."""
    structure_valid: bool
    overall_coherence_score: float
    issues_found: List[str]
    critical_issues: List[str]
    warnings: List[str]
    improvements_suggested: List[str]
    chapter_flow_assessment: str
    missing_elements: List[str]
    structural_strengths: List[str]
    recommended_fixes: List[str]
    publication_readiness: str  # "ready", "needs_minor_fixes", "needs_major_revision"


class TableOfContentsEnhancementModel(BaseModel):
    """Pydantic model for LLM-enhanced table of contents."""
    enhanced_entries: List[Dict[str, Any]]
    structure_improvements: List[str]
    navigation_enhancements: List[str]
    accessibility_improvements: List[str]
    formatting_suggestions: List[str]
    overall_quality_score: float
    recommended_modifications: List[str]


class IndexGenerationModel(BaseModel):
    """Pydantic model for LLM-generated index."""
    index_terms: List[Dict[str, Any]]
    cross_references: List[Dict[str, Any]]
    term_categories: Dict[str, List[str]]
    importance_rankings: Dict[str, int]
    coverage_assessment: str
    quality_score: float
    missing_important_terms: List[str]
    recommended_additions: List[str]


class BibliographyEnhancementModel(BaseModel):
    """Pydantic model for LLM-enhanced bibliography."""
    enhanced_entries: List[Dict[str, Any]]
    citation_style_corrections: List[str]
    missing_information: List[str]
    duplicate_entries: List[str]
    reliability_assessments: Dict[str, float]
    formatting_improvements: List[str]
    completeness_score: float


class ContentFormattingModel(BaseModel):
    """Pydantic model for LLM content formatting."""
    formatted_content: str
    formatting_changes_made: List[str]
    style_consistency_improvements: List[str]
    readability_enhancements: List[str]
    accessibility_improvements: List[str]
    technical_corrections: List[str]
    formatting_quality_score: float


# Type Aliases for Complex Types

ChapterData = Dict[str, Any]
FormatConfig = Dict[str, Any]
ValidationRules = Dict[str, Any]
ProcessingOptions = Dict[str, Any]
QualityMetrics = Dict[str, float]
PerformanceMetrics = Dict[str, Union[int, float, str]]
ErrorReport = Dict[str, Any]


# Constants and Configuration

DEFAULT_FORMATTING_OPTIONS = {
    "pdf": {
        "page_size": "A4",
        "margin_inches": 1.0,
        "font_family": "Times New Roman",
        "font_size": 12,
        "line_spacing": 1.5,
        "include_page_numbers": True,
        "include_headers": True,
        "include_footers": False,
    },
    "epub": {
        "cover_required": True,
        "toc_depth": 3,
        "include_navigation": True,
        "metadata_complete": True,
    },
    "html": {
        "include_css": True,
        "responsive_design": True,
        "include_navigation": True,
        "accessibility_compliant": True,
    },
    "docx": {
        "include_toc": True,
        "include_page_numbers": True,
        "style_template": "professional",
        "track_changes": False,
    },
    "markdown": {
        "include_yaml_frontmatter": True,
        "table_of_contents": True,
        "github_flavored": True,
    }
}

QUALITY_THRESHOLDS = {
    "minimum_word_count": 1000,
    "maximum_word_count": 500000,
    "minimum_chapters": 1,
    "maximum_chapters": 100,
    "minimum_quality_score": 0.6,
    "structure_coherence_threshold": 0.7,
    "content_completeness_threshold": 0.8,
    "formatting_consistency_threshold": 0.75,
}

VALIDATION_RULES = {
    "required_metadata_fields": [
        "title", "genre", "total_word_count"
    ],
    "required_chapter_fields": [
        "chapter_number", "title", "content"
    ],
    "maximum_file_sizes": {
        "pdf": 50 * 1024 * 1024,  # 50MB
        "epub": 25 * 1024 * 1024,  # 25MB
        "docx": 30 * 1024 * 1024,  # 30MB
        "html": 10 * 1024 * 1024,  # 10MB
    },
    "content_validation": {
        "check_for_placeholders": True,
        "validate_chapter_numbering": True,
        "check_cross_references": True,
        "validate_citations": True,
    }
}