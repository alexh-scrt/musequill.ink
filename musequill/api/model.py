from uuid import UUID
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator

from musequill.models.presets import (
    GenreType,
    SubGenre,
    BookLength,
    StoryStructure,
    PlotType,
    NarrativePOV,
    PacingType,
    ConflictType,
    CharacterRole,
    CharacterArchetype,
    WorldType,
    MagicSystemType,
    TechnologyLevel,
    WritingStyle,
    ToneType,
    AgeGroup,
    AudienceType,
    ReadingLevel,
    PublicationRoute,
    ContentWarning,
    AIAssistanceLevel,
    ResearchPriority,
    WritingSchedule
)
from musequill.models.subgenre import (
    validate_book_genre_subgenre
)

class BookCreationRequest(BaseModel):
    """Request model for creating a new book plan."""
    title: str = Field(..., min_length=1, max_length=200)
    subtitle: Optional[str] = Field(None, max_length=200)
    genre: GenreType
    sub_genre: Optional[str] = None  # Changed from SubGenre enum to string for validation
    length: BookLength
    description: Optional[str] = Field(None, max_length=1000)
    
    # ... rest of the fields remain the same ...
    structure: StoryStructure
    plot_type: Optional[PlotType] = None
    pov: NarrativePOV
    pacing: Optional[PacingType] = None
    conflict_types: List[ConflictType] = Field(default_factory=list, max_items=3)
    
    main_character_role: CharacterRole
    character_archetype: Optional[CharacterArchetype] = None
    world_type: WorldType
    magic_system: Optional[MagicSystemType] = None
    technology_level: Optional[TechnologyLevel] = None
    
    writing_style: WritingStyle
    tone: ToneType
    complexity: Optional[str] = None
    
    age_group: AgeGroup
    audience_type: AudienceType
    reading_level: Optional[ReadingLevel] = None
    publication_route: PublicationRoute
    content_warnings: List[ContentWarning] = Field(default_factory=list)
    
    ai_assistance_level: AIAssistanceLevel
    research_priority: Optional[ResearchPriority] = None
    writing_schedule: Optional[WritingSchedule] = None
    additional_notes: Optional[str] = Field(None, max_length=500)

    @field_validator('sub_genre')
    @classmethod
    def validate_subgenre_for_genre(cls, v, info):
        """Validate that sub-genre is appropriate for the selected genre."""
        if v is None or v == "":
            return None
            
        # Get genre from context
        genre = info.data.get('genre')
        if genre:
            validation_result = validate_book_genre_subgenre(genre.value, v)
            if not validation_result['valid']:
                raise ValueError(f"Sub-genre '{v}' is not valid for genre '{genre.value}'. {validation_result.get('error', '')}")
        
        return v

    @field_validator('conflict_types')
    @classmethod
    def validate_conflict_types(cls, v):
        if len(v) > 3:
            raise ValueError('Maximum 3 conflict types allowed')
        return v

    @field_validator('magic_system', 'technology_level', 'character_archetype', 'plot_type', 'pacing', 'reading_level', 'research_priority', 'writing_schedule', mode='before')
    @classmethod
    def convert_empty_strings_to_none(cls, v):
        """Convert empty strings to None for optional enum fields."""
        if v == "" or v is None:
            return None
        return v

    @field_validator('subtitle', 'description', 'complexity', 'additional_notes', mode='before')
    @classmethod
    def convert_empty_strings_for_text_fields(cls, v):
        """Convert empty strings to None for optional text fields."""
        if v == "" or v is None:
            return None
        return v


class BookCreationResponse(BaseModel):
    """Response model for book creation."""
    book_id: UUID
    title: str
    status: str
    message: str
    estimated_word_count: int
    estimated_chapters: int
    created_at: datetime
    
    # New fields for agent integration
    agent_id: Optional[str] = None
    planning_status: str = "initializing"
    next_steps: List[str] = Field(default_factory=list)

class BookStatusResponse(BaseModel):
    """Response model for book status."""
    book_id: UUID
    title: str
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    # Planning information
    planning_status: Optional[str] = None
    agent_id: Optional[str] = None
    planning_results: Optional[Dict[str, Any]] = None
    
    # Progress tracking
    estimated_word_count: int
    estimated_chapters: int
    completion_percentage: float = 0.0
    
    # Messages and next steps
    status_message: Optional[str] = None
    next_steps: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None

class EnumChoice(BaseModel):
    """Model for enum choice with metadata."""
    value: str
    label: str
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class EnumData(BaseModel):
    """Complete enum data for frontend."""
    enums: Dict[str, List[List[str]]]
    metadata: Dict[str, Any]
    recommendations: Dict[str, Dict[str, List[str]]]

class GenreSubgenreValidationRequest(BaseModel):
    """Request model for genre-subgenre validation."""
    genre: str
    subgenre: str
