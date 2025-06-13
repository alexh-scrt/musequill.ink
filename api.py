"""
Book Planner Backend API - Enhanced Version
FastAPI server providing enum data and book creation endpoints with proper metadata
"""
import sys
import time
import logging
import traceback
from typing import Dict, List, Optional, Any
from datetime import datetime
from uuid import UUID, uuid4
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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
    WritingSchedule,
    get_genre_recommendations,
    get_enum_with_metadata,
    validate_enum_combination,
    get_default_values
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('musequill_api.log')
    ]
)

logger = logging.getLogger(__name__)

class BookCreationRequest(BaseModel):
    """Request model for creating a new book plan."""
    
    # Step 1: Book Basics
    title: str = Field(..., min_length=1, max_length=200)
    subtitle: Optional[str] = Field(None, max_length=200)
    genre: GenreType
    sub_genre: Optional[SubGenre] = None
    length: BookLength
    description: Optional[str] = Field(None, max_length=1000)
    
    # Step 2: Story Structure
    structure: StoryStructure
    plot_type: Optional[PlotType] = None
    pov: NarrativePOV
    pacing: Optional[PacingType] = None
    conflict_types: List[ConflictType] = Field(default_factory=list, max_items=3)
    
    # Step 3: Characters & World
    main_character_role: CharacterRole
    character_archetype: Optional[CharacterArchetype] = None
    world_type: WorldType
    magic_system: Optional[MagicSystemType] = None
    technology_level: Optional[TechnologyLevel] = None
    
    # Step 4: Writing Style
    writing_style: WritingStyle
    tone: ToneType
    complexity: Optional[str] = None
    
    # Step 5: Audience & Publishing
    age_group: AgeGroup
    audience_type: AudienceType
    reading_level: Optional[ReadingLevel] = None
    publication_route: PublicationRoute
    content_warnings: List[ContentWarning] = Field(default_factory=list)
    
    # Step 6: AI Assistance
    ai_assistance_level: AIAssistanceLevel
    research_priority: Optional[ResearchPriority] = None
    writing_schedule: Optional[WritingSchedule] = None
    additional_notes: Optional[str] = Field(None, max_length=500)

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

# Pydantic models for API requests/responses
# class BookCreationRequest(BaseModel):
#     """Request model for creating a new book plan."""
    
#     # Step 1: Book Basics
#     title: str = Field(..., min_length=1, max_length=200)
#     subtitle: Optional[str] = Field(None, max_length=200)
#     genre: GenreType
#     sub_genre: Optional[SubGenre] = None
#     length: BookLength
#     description: Optional[str] = Field(None, max_length=1000)
    
#     # Step 2: Story Structure
#     structure: StoryStructure
#     plot_type: Optional[PlotType] = None
#     pov: NarrativePOV
#     pacing: Optional[PacingType] = None
#     conflict_types: List[ConflictType] = Field(default_factory=list, max_items=3)
    
#     # Step 3: Characters & World
#     main_character_role: CharacterRole
#     character_archetype: Optional[CharacterArchetype] = None
#     world_type: WorldType
#     magic_system: Optional[MagicSystemType] = None
#     technology_level: Optional[TechnologyLevel] = None
    
#     # Step 4: Writing Style
#     writing_style: WritingStyle
#     tone: ToneType
#     complexity: Optional[str] = None
    
#     # Step 5: Audience & Publishing
#     age_group: AgeGroup
#     audience_type: AudienceType
#     reading_level: Optional[ReadingLevel] = None
#     publication_route: PublicationRoute
#     content_warnings: List[ContentWarning] = Field(default_factory=list)
    
#     # Step 6: AI Assistance
#     ai_assistance_level: AIAssistanceLevel
#     research_priority: Optional[ResearchPriority] = None
#     writing_schedule: Optional[WritingSchedule] = None
#     additional_notes: Optional[str] = Field(None, max_length=500)

#     @field_validator('conflict_types')
#     @classmethod
#     def validate_conflict_types(cls, v):
#         if len(v) > 3:
#             raise ValueError('Maximum 3 conflict types allowed')
#         return v


class BookCreationResponse(BaseModel):
    """Response model for successful book creation."""
    book_id: UUID
    title: str
    status: str
    message: str
    estimated_word_count: int
    estimated_chapters: int
    created_at: datetime


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


# FastAPI application
app = FastAPI(
    title="MuseQuill Book Planner API",
    description="API for book parameter selection and AI-assisted book planning",
    version="0.1.0"
)

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start_time = time.time()
    
    # Log request details
    logger.info(f"🔵 {request.method} {request.url}")
    logger.debug(f"Headers: {dict(request.headers)}")
    
    # For POST requests, log the body (be careful with sensitive data)
    if request.method == "POST":
        body = await request.body()
        logger.debug(f"Request body: {body.decode('utf-8')[:1000]}...")  # First 1000 chars
        
        # Re-create request with body for downstream processing
        from fastapi import Request
        async def receive():
            return {"type": "http.request", "body": body}
        request._receive = receive
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(f"🟢 {request.method} {request.url} - {response.status_code} ({process_time:.3f}s)")
    
    return response


# CORS middleware for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (our HTML frontend)
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

# In-memory storage for demo (use database in production)
books_db: Dict[UUID, Dict] = {}

# Enhanced exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors with detailed logging."""
    logger.error(f"Validation error for {request.method} {request.url}")
    logger.error(f"Request body: {await request.body()}")
    logger.error(f"Validation errors: {exc.errors()}")
    
    # Format error details for frontend
    error_details = []
    for error in exc.errors():
        field_path = " -> ".join(str(x) for x in error["loc"])
        error_details.append({
            "field": field_path,
            "message": error["msg"],
            "type": error["type"],
            "input": error.get("input")
        })
    
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Validation Error",
            "message": "Please check the form fields and try again",
            "errors": error_details,
            "request_id": id(request)
        }
    )

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle value errors with logging."""
    logger.error(f"Value error for {request.method} {request.url}: {str(exc)}")
    return JSONResponse(
        status_code=400,
        content={
            "detail": "Invalid Value",
            "message": str(exc),
            "request_id": id(request)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions."""
    logger.error(f"Unhandled exception for {request.method} {request.url}")
    logger.error(f"Exception: {type(exc).__name__}: {str(exc)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal Server Error",
            "message": "An unexpected error occurred",
            "request_id": id(request)
        }
    )



def enum_to_choices(enum_class) -> List[List[str]]:
    """Convert enum to choices format."""
    return [[item.value, item.value.replace('_', ' ').title()] for item in enum_class]


def get_enum_metadata():
    """Get comprehensive metadata for enums."""
    return {
        "GenreType": {
            "description": "Primary genre classification for the book",
            "popularity": {
                "fantasy": 9, "romance": 10, "mystery": 8, "science_fiction": 7,
                "thriller": 8, "business": 7, "self_help": 9, "literary_fiction": 6,
                "young_adult": 8, "horror": 6, "historical_fiction": 7, "memoir": 6,
                "biography": 5, "science": 5, "technology": 6, "health": 7,
                "true_crime": 8, "children": 7, "poetry": 3
            },
            "difficulty": {
                "literary_fiction": 9, "science_fiction": 8, "fantasy": 7,
                "mystery": 7, "romance": 5, "thriller": 6, "business": 6,
                "self_help": 4, "memoir": 6, "biography": 7, "children": 3
            },
            "market_viability": {
                "romance": 10, "fantasy": 9, "mystery": 8, "thriller": 8,
                "self_help": 9, "business": 7, "young_adult": 9, "children": 8,
                "literary_fiction": 4, "poetry": 2, "science": 4
            }
        },
        "BookLength": {
            "description": "Target length category for the book",
            "word_counts": {
                "flash_fiction": [100, 1000],
                "short_story": [1000, 7500],
                "novelette": [7500, 17500],
                "novella": [17500, 40000],
                "short_novel": [40000, 60000],
                "standard_novel": [60000, 90000],
                "long_novel": [90000, 120000],
                "epic_novel": [120000, 200000],
                "article": [500, 2000],
                "essay": [1000, 5000],
                "guide": [5000, 15000],
                "manual": [15000, 50000],
                "comprehensive_book": [50000, 150000]
            },
            "estimated_pages": {
                "short_story": [4, 30], "novella": [70, 160], "short_novel": [160, 240],
                "standard_novel": [240, 360], "long_novel": [360, 480], "epic_novel": [480, 800]
            }
        },
        "StoryStructure": {
            "description": "Narrative structure framework for organizing the story",
            "complexity": {
                "three_act": 5, "hero_journey": 7, "save_the_cat": 6,
                "seven_point": 8, "freytag_pyramid": 6, "snowflake": 9
            },
            "best_for_genres": {
                "three_act": ["mystery", "thriller", "romance", "drama"],
                "hero_journey": ["fantasy", "science_fiction", "adventure"],
                "save_the_cat": ["comedy", "action", "thriller"],
                "seven_point": ["character_driven", "literary_fiction"]
            }
        },
        "WorldType": {
            "description": "Setting and world-building approach",
            "research_required": {
                "realistic": 8, "alternate_history": 9, "low_fantasy": 6,
                "high_fantasy": 4, "science_fiction": 7, "urban_fantasy": 6
            }
        },
        "WritingStyle": {
            "description": "Approach to prose and narrative voice",
            "target_audiences": {
                "academic": ["academics", "students"], "conversational": ["general_readers"],
                "technical": ["professionals", "experts"], "narrative": ["general_readers", "genre_fans"]
            }
        },
        "AgeGroup": {
            "description": "Primary target age demographic",
            "content_guidelines": {
                "children": "Simple language, positive themes, age-appropriate content",
                "middle_grade": "Coming-of-age themes, friendship, family",
                "young_adult": "Identity, relationships, challenging themes",
                "adult": "Complex themes, mature content allowed"
            }
        }
    }


def get_comprehensive_recommendations():
    """Get comprehensive recommendation mappings from backend models."""
    recommendations = {}
    
    # Generate recommendations for each genre
    for genre in GenreType:
        genre_recs = get_genre_recommendations(genre)
        if genre_recs:
            recommendations[genre.value] = genre_recs
    
    return recommendations


@app.get("/")
async def serve_frontend():
    """Serve the main frontend page."""
    return FileResponse(static_dir / "index.html")


@app.get("/api/enums")
async def get_enums() -> EnumData:
    """Get all enum data for frontend form population with comprehensive metadata."""
    
    all_enums = {
        "GenreType": enum_to_choices(GenreType),
        "SubGenre": enum_to_choices(SubGenre),
        "BookLength": enum_to_choices(BookLength),
        "StoryStructure": enum_to_choices(StoryStructure),
        "PlotType": enum_to_choices(PlotType),
        "NarrativePOV": enum_to_choices(NarrativePOV),
        "PacingType": enum_to_choices(PacingType),
        "ConflictType": enum_to_choices(ConflictType),
        "CharacterRole": enum_to_choices(CharacterRole),
        "CharacterArchetype": enum_to_choices(CharacterArchetype),
        "WorldType": enum_to_choices(WorldType),
        "MagicSystemType": enum_to_choices(MagicSystemType),
        "TechnologyLevel": enum_to_choices(TechnologyLevel),
        "WritingStyle": enum_to_choices(WritingStyle),
        "ToneType": enum_to_choices(ToneType),
        "AgeGroup": enum_to_choices(AgeGroup),
        "AudienceType": enum_to_choices(AudienceType),
        "ReadingLevel": enum_to_choices(ReadingLevel),
        "PublicationRoute": enum_to_choices(PublicationRoute),
        "ContentWarning": enum_to_choices(ContentWarning),
        "AIAssistanceLevel": enum_to_choices(AIAssistanceLevel),
        "ResearchPriority": enum_to_choices(ResearchPriority),
        "WritingSchedule": enum_to_choices(WritingSchedule)
    }
    
    return EnumData(
        enums=all_enums,
        metadata=get_enum_metadata(),
        recommendations=get_comprehensive_recommendations()
    )


@app.get("/api/enums/{enum_name}")
async def get_specific_enum(enum_name: str) -> Dict[str, Any]:
    """Get specific enum data with detailed metadata."""
    enum_mapping = {
        "GenreType": GenreType,
        "SubGenre": SubGenre,
        "BookLength": BookLength,
        "StoryStructure": StoryStructure,
        "PlotType": PlotType,
        "NarrativePOV": NarrativePOV,
        "PacingType": PacingType,
        "ConflictType": ConflictType,
        "CharacterRole": CharacterRole,
        "CharacterArchetype": CharacterArchetype,
        "WorldType": WorldType,
        "MagicSystemType": MagicSystemType,
        "TechnologyLevel": TechnologyLevel,
        "WritingStyle": WritingStyle,
        "ToneType": ToneType,
        "AgeGroup": AgeGroup,
        "AudienceType": AudienceType,
        "ReadingLevel": ReadingLevel,
        "PublicationRoute": PublicationRoute,
        "ContentWarning": ContentWarning,
        "AIAssistanceLevel": AIAssistanceLevel,
        "ResearchPriority": ResearchPriority,
        "WritingSchedule": WritingSchedule
    }
    
    if enum_name not in enum_mapping:
        raise HTTPException(status_code=404, detail=f"Enum '{enum_name}' not found")
    
    enum_class = enum_mapping[enum_name]
    metadata = get_enum_metadata().get(enum_name, {})
    
    return {
        "name": enum_name,
        "choices": enum_to_choices(enum_class),
        "metadata": get_enum_with_metadata(enum_class),
        "description": metadata.get("description", ""),
        "additional_metadata": metadata
    }


@app.get("/api/recommendations/{genre}")
async def get_recommendations(genre: str) -> Dict[str, List[str]]:
    """Get recommendations for a specific genre from backend models."""
    try:
        # Validate genre exists
        genre_enum = GenreType(genre)
        recommendations = get_genre_recommendations(genre_enum)
        return recommendations
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Genre '{genre}' not found")


@app.get("/api/validation/check")
async def validate_selections(
    genre: Optional[str] = None,
    age_group: Optional[str] = None,
    content_warnings: Optional[str] = None,
    world_type: Optional[str] = None,
    structure: Optional[str] = None
) -> Dict[str, Any]:
    """Validate enum combinations and return warnings/recommendations."""
    selections = {}
    
    if genre:
        selections['genre'] = genre
    if age_group:
        selections['age_group'] = age_group
    if content_warnings:
        selections['content_warnings'] = content_warnings.split(',') if content_warnings else []
    if world_type:
        selections['world_type'] = world_type
    if structure:
        selections['structure'] = structure
    
    warnings = validate_enum_combination(selections)
    
    return {
        "valid": len(warnings) == 0,
        "warnings": warnings,
        "recommendations": get_genre_recommendations(GenreType(genre)) if genre else {}
    }


@app.get("/api/defaults")
async def get_default_selections() -> Dict[str, str]:
    """Get default values for form fields from backend models."""
    return get_default_values()


@app.get("/api/metadata/word-counts")
async def get_word_count_data() -> Dict[str, Any]:
    """Get detailed word count information for all book lengths."""
    metadata = get_enum_metadata()
    word_counts = metadata.get("BookLength", {}).get("word_counts", {})
    
    result = {}
    for length_key, range_data in word_counts.items():
        if isinstance(range_data, list) and len(range_data) == 2:
            result[length_key] = {
                "min_words": range_data[0],
                "max_words": range_data[1],
                "average_words": (range_data[0] + range_data[1]) // 2,
                "estimated_pages": {
                    "min": range_data[0] // 250,  # ~250 words per page
                    "max": range_data[1] // 250
                },
                "estimated_reading_time_hours": {
                    "min": round(range_data[0] / 13500, 1),  # ~225 words/min * 60 min
                    "max": round(range_data[1] / 13500, 1)
                }
            }
    
    return result


@app.get("/api/metadata/genre-analysis")
async def get_genre_analysis() -> Dict[str, Any]:
    """Get comprehensive genre analysis data."""
    metadata = get_enum_metadata()
    genre_meta = metadata.get("GenreType", {})
    
    result = {}
    for genre in GenreType:
        genre_key = genre.value
        result[genre_key] = {
            "display_name": genre.value.replace('_', ' ').title(),
            "is_fiction": genre.is_fiction,
            "popularity_score": genre_meta.get("popularity", {}).get(genre_key, 5),
            "difficulty_score": genre_meta.get("difficulty", {}).get(genre_key, 5),
            "market_viability": genre_meta.get("market_viability", {}).get(genre_key, 5),
            "recommendations": get_genre_recommendations(genre),
            "typical_length": get_typical_length_for_genre(genre_key),
            "common_structures": get_common_structures_for_genre(genre_key)
        }
    
    return result


def get_typical_length_for_genre(genre: str) -> List[str]:
    """Get typical book lengths for a genre."""
    length_mapping = {
        "children": ["short_story", "novella"],
        "picture_book": ["flash_fiction", "short_story"],
        "young_adult": ["short_novel", "standard_novel"],
        "romance": ["novella", "standard_novel"],
        "fantasy": ["standard_novel", "long_novel", "epic_novel"],
        "science_fiction": ["standard_novel", "long_novel"],
        "mystery": ["novella", "standard_novel"],
        "thriller": ["standard_novel"],
        "business": ["guide", "manual"],
        "self_help": ["guide", "manual"],
        "memoir": ["standard_novel"],
        "biography": ["standard_novel", "long_novel"]
    }
    return length_mapping.get(genre, ["standard_novel"])


def get_common_structures_for_genre(genre: str) -> List[str]:
    """Get common story structures for a genre."""
    structure_mapping = {
        "fantasy": ["hero_journey", "three_act"],
        "science_fiction": ["three_act", "seven_point"],
        "mystery": ["three_act", "fichtean_curve"],
        "romance": ["three_act", "story_circle"],
        "thriller": ["three_act", "save_the_cat"],
        "business": ["instructional", "case_study"],
        "self_help": ["problem_solution", "step_by_step"]
    }
    return structure_mapping.get(genre, ["three_act"])


@app.post("/api/books/create")
async def create_book(
    request: BookCreationRequest,
    background_tasks: BackgroundTasks
) -> BookCreationResponse:
    """Create a new book plan based on user parameters."""
    
    # Generate unique book ID
    book_id = uuid4()
    
    # Calculate estimated metrics based on length
    word_count_mapping = {
        BookLength.FLASH_FICTION: 500,
        BookLength.SHORT_STORY: 4000,
        BookLength.NOVELETTE: 12500,
        BookLength.NOVELLA: 30000,
        BookLength.SHORT_NOVEL: 50000,
        BookLength.STANDARD_NOVEL: 75000,
        BookLength.LONG_NOVEL: 105000,
        BookLength.EPIC_NOVEL: 150000,
        BookLength.ARTICLE: 1250,
        BookLength.ESSAY: 3000,
        BookLength.GUIDE: 10000,
        BookLength.MANUAL: 32500,
        BookLength.COMPREHENSIVE_BOOK: 100000
    }
    
    estimated_word_count = word_count_mapping.get(request.length, 75000)
    estimated_chapters = max(1, estimated_word_count // 3000)  # ~3000 words per chapter
    
    # Store book data
    book_data = {
        "id": book_id,
        "created_at": datetime.now(),
        "parameters": request.dict(),
        "status": "planning",
        "estimated_word_count": estimated_word_count,
        "estimated_chapters": estimated_chapters,
        "validation_warnings": validate_enum_combination(request.dict())
    }
    
    books_db[book_id] = book_data
    
    # Add background task to start AI planning process
    background_tasks.add_task(start_book_planning, book_id, request)
    
    return BookCreationResponse(
        book_id=book_id,
        title=request.title,
        status="created",
        message="Book plan created successfully. AI agents are now analyzing your requirements.",
        estimated_word_count=estimated_word_count,
        estimated_chapters=estimated_chapters,
        created_at=datetime.now()
    )


async def start_book_planning(book_id: UUID, request: BookCreationRequest):
    """Background task to start the AI book planning process."""
    # This is where we would integrate with our AI agents
    # For now, just simulate the planning process
    
    print(f"Starting AI planning for book {book_id}")
    print(f"Title: {request.title}")
    print(f"Genre: {request.genre}")
    print(f"Structure: {request.structure}")
    print(f"AI Assistance Level: {request.ai_assistance_level}")
    
    # Update book status
    if book_id in books_db:
        books_db[book_id]["status"] = "ai_planning"
        books_db[book_id]["planning_started_at"] = datetime.now()


@app.get("/api/books/{book_id}")
async def get_book(book_id: UUID) -> Dict[str, Any]:
    """Get book details by ID."""
    if book_id not in books_db:
        raise HTTPException(status_code=404, detail="Book not found")
    
    return books_db[book_id]


@app.get("/api/books")
async def list_books() -> List[Dict[str, Any]]:
    """List all books (for demo purposes)."""
    return list(books_db.values())


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "book-planner-api",
        "version": "0.1.0",
        "books_created": len(books_db),
        "enums_available": len([
            GenreType, SubGenre, BookLength, StoryStructure, PlotType,
            NarrativePOV, PacingType, ConflictType, CharacterRole,
            CharacterArchetype, WorldType, MagicSystemType, TechnologyLevel,
            WritingStyle, ToneType, AgeGroup, AudienceType, ReadingLevel,
            PublicationRoute, ContentWarning, AIAssistanceLevel,
            ResearchPriority, WritingSchedule
        ])
    }


@app.get("/dashboard")
async def serve_dashboard():
    """Serve a simple dashboard page (placeholder)."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MuseQuill Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .book-card { background: #f5f5f5; padding: 20px; margin: 20px 0; border-radius: 8px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📚 MuseQuill Dashboard</h1>
            <p>Welcome to your writing dashboard! Your book planning is in progress.</p>
            <div class="book-card">
                <h3>🎉 Book Plan Created</h3>
                <p>Your AI agents are analyzing your preferences and creating a personalized outline.</p>
                <p><strong>Next steps:</strong></p>
                <ul>
                    <li>✅ Parameters collected</li>
                    <li>🔄 AI analysis in progress</li>
                    <li>⏳ Chapter outline generation</li>
                    <li>⏳ Character development</li>
                    <li>⏳ Research plan creation</li>
                </ul>
            </div>
            <a href="/" style="color: #3498db;">← Back to Book Planner</a>
        </div>
    </body>
    </html>
    """
    return FileResponse(static_dir / "dashboard.html") if (static_dir / "dashboard.html").exists() else html_content


if __name__ == "__main__":
    import uvicorn
    import argparse
    
    # Command line argument parsing
    parser = argparse.ArgumentParser(description='MuseQuill Book Planner API')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8055, help='Port to bind to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--log-level', default='DEBUG', help='Log level')
    
    args = parser.parse_args()
    
    # Set log level based on arguments
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("uvicorn").setLevel(logging.DEBUG)
        logging.getLogger("fastapi").setLevel(logging.DEBUG)
    
    logger.info("🚀 Starting MuseQuill Book Planner API server...")
    logger.info(f"🌐 API Server: http://{args.host}:{args.port}")
    logger.info(f"📝 Frontend: http://localhost:8088")
    logger.info(f"🔧 API docs: http://{args.host}:{args.port}/docs")
    logger.info(f"💾 Dashboard: http://{args.host}:{args.port}/dashboard")
    logger.info(f"🐛 Debug endpoint: http://{args.host}:{args.port}/api/debug/book-data")
    logger.info("📊 API endpoints:")
    logger.info(f"   - GET http://{args.host}:{args.port}/api/enums")
    logger.info(f"   - GET http://{args.host}:{args.port}/api/recommendations/{{genre}}")
    logger.info(f"   - POST http://{args.host}:{args.port}/api/books/create")
    
    # Run the server
    uvicorn.run(
        app, 
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level.lower(),
        access_log=True
    )