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

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# pylint: disable=unused-import
import bootstrap
from musequill.models.presets import (
    GenreType,
#    SubGenre,
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
#    get_enum_with_metadata,
    validate_enum_combination,
    get_default_values
)
from musequill.api.model import (
    BookCreationRequest,
    BookCreationResponse,
    BookStatusResponse,
    EnumChoice,
    EnumData,
    GenreSubgenreValidationRequest,
#    validate_book_genre_subgenre
)
from musequill.models.subgenre import (
    SubGenreRegistry, 
    SubGenre,
    get_subgenres_for_frontend,
    validate_book_genre_subgenre
)
from musequill.agents.factory import AgentFactory, get_agent_factory
from musequill.routers.planning import planning_router
from musequill.core.openai_client.client import OpenAIClient
from musequill.config.settings import Settings
from musequill.config.logging import setup_logging, get_logger
from musequill.database.book import books_db
from musequill.workers.planning import (
    enum_to_choices,
    get_enum_metadata,
    get_comprehensive_recommendations,
    get_enum_with_metadata,
    determine_next_steps,
    start_book_planning_with_error_handling,
    update_book_status_in_db
)
from musequill.models.word_count import WORD_COUNT_MAPPING
        
# Create settings
settings = Settings()
settings.LOG_LEVEL = "DEBUG"
settings.STRUCTURED_LOGGING = True
settings.LOG_FORMAT = "json"
settings.LOG_FILE_PATH = None  # No file logging for test
        
# Setup logging
setup_logging(settings)

# Get logger and test
logger = get_logger(__name__)

# Dependency injection for agent system
async def get_openai_client() -> OpenAIClient:
    """Get OpenAI client instance."""
    return OpenAIClient()

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

def get_agent_factory_dependency(
    openai_client: OpenAIClient = Depends(get_openai_client)
) -> AgentFactory:
    """Get agent factory instance."""
    return get_agent_factory(openai_client)

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

@app.get("/")
async def serve_frontend():
    """Serve the main frontend page."""
    return FileResponse(static_dir / "index.html")


@app.get("/api/enums")
async def get_enums() -> EnumData:
    """Get all enum data for frontend form population with comprehensive metadata."""
    
    all_enums = {
        "GenreType": enum_to_choices(GenreType),
        "SubGenre": SubGenreRegistry.get_all_subgenres(),  # Use the new system
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

@app.get("/api/subgenres/{genre}")
async def get_subgenres_for_genre(genre: str) -> Dict[str, Any]:
    """Get available subgenres for a specific genre."""
    
    try:
        subgenres_data = get_subgenres_for_frontend(genre)
        return {
            "genre": genre,
            "subgenres": subgenres_data["subgenres"],
            "count": subgenres_data["count"],
            "enum_class": subgenres_data["enum_class"]
        }
    except Exception as e:
        logger.error(f"Error getting subgenres for {genre}: {str(e)}")
        return {
            "genre": genre,
            "subgenres": [],
            "count": 0,
            "error": str(e)
        }

@app.post("/api/validate/genre-subgenre")
async def validate_genre_subgenre(request: GenreSubgenreValidationRequest) -> Dict[str, Any]:
    """Validate genre and subgenre combination via JSON body."""  

    try:
        # Validate the combination
        validation_result = validate_book_genre_subgenre(request.genre, request.subgenre)
        
        return {
            "valid": validation_result["valid"],
            "genre": request.genre,
            "subgenre": request.subgenre,
            "message": "Valid combination" if validation_result["valid"] else validation_result.get("error", "Invalid combination"),
            "suggestions": validation_result.get("suggestions", []) if not validation_result["valid"] else []
        }
        
    except Exception as e:
        logger.error(f"Error validating genre-subgenre: {str(e)}")
        return {
            "valid": False,
            "genre": request.genre,
            "subgenre": request.subgenre,
            "message": f"Validation error: {str(e)}",
            "suggestions": []
        }

@app.get("/api/subgenres")
async def get_all_subgenres() -> Dict[str, Any]:
    """Get all available sub-genres across all genres."""
    all_subgenres = SubGenreRegistry.get_all_subgenres()
    
    return {
        "subgenres": all_subgenres,
        "total_count": len(all_subgenres),
        "by_genre": {
            genre: SubGenreRegistry.get_subgenre_choices(genre)
            for genre in SubGenreRegistry.GENRE_SUBGENRE_MAP.keys()
        }
    }

@app.get("/api/genres/{genre}/info")
async def get_genre_info(genre: str) -> Dict[str, Any]:
    """Get comprehensive information about a specific genre."""
    try:
        # Validate genre exists
        genre_enum = GenreType(genre)
        
        # Get sub-genres
        subgenres = SubGenreRegistry.get_subgenre_choices(genre)
        
        # Get metadata if available
        metadata = get_enum_metadata()
        genre_metadata = metadata.get("GenreType", {})
        
        return {
            "genre": genre,
            "display_name": genre.replace('_', ' ').title(),
            "is_fiction": genre_enum.is_fiction,
            "subgenres": {
                "available": subgenres,
                "count": len(subgenres),
                "enum_class": SubGenreRegistry.get_subgenres_for_genre(genre).__name__ if SubGenreRegistry.get_subgenres_for_genre(genre) else None
            },
            "metadata": {
                "popularity": genre_metadata.get("popularity", {}).get(genre, 5),
                "difficulty": genre_metadata.get("difficulty", {}).get(genre, 5),
                "market_viability": genre_metadata.get("market_viability", {}).get(genre, 5)
            },
            "recommendations": get_genre_recommendations(genre_enum) if hasattr(genre_enum, '__call__') else {}
        }
        
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Genre '{genre}' not found")

@app.get("/api/debug/subgenres")
async def debug_subgenres() -> Dict[str, Any]:
    """Debug endpoint to test the sub-genre system."""
    debug_info = {}
    
    # Test a few genres
    test_genres = ["fantasy", "science_fiction", "business", "self_help", "mystery"]
    
    for genre in test_genres:
        try:
            subgenres = SubGenreRegistry.get_subgenre_choices(genre)
            debug_info[genre] = {
                "subgenre_count": len(subgenres),
                "subgenres": subgenres[:3],  # First 3 for brevity
                "enum_class": SubGenreRegistry.get_subgenres_for_genre(genre).__name__ if SubGenreRegistry.get_subgenres_for_genre(genre) else None
            }
        except Exception as e:
            debug_info[genre] = {"error": str(e)}
    
    # Test validation
    debug_info["validation_tests"] = {
        "fantasy_high_fantasy": validate_book_genre_subgenre("fantasy", "high_fantasy"),
        "fantasy_entrepreneurship": validate_book_genre_subgenre("fantasy", "entrepreneurship"),
        "business_marketing": validate_book_genre_subgenre("business", "marketing")
    }
    
    # Master enum info
    debug_info["master_enum"] = {
        "total_subgenres": len(SubGenre),
        "sample_values": [item.value for item in list(SubGenre)[:5]]
    }
    
    return debug_info

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


@app.post("/api/books/create", response_model=BookCreationResponse)
async def create_book(
    request: BookCreationRequest,
    background_tasks: BackgroundTasks,
    factory: AgentFactory = Depends(get_agent_factory_dependency)
) -> BookCreationResponse:
    """Create a new book plan based on user parameters with AI agent planning."""
    
    try:
        if request.sub_genre:
            validation_result = validate_book_genre_subgenre(request.genre.value, request.sub_genre)
            if not validation_result['valid']:
                raise HTTPException(
                    status_code=422, 
                    detail={
                        "message": "Invalid genre/sub-genre combination",
                        "error": validation_result['error'],
                        "valid_subgenres": validation_result.get('valid_subgenres', [])
                    }
                )

        # Generate unique book ID
        book_id = uuid4()
        
        logger.info(
            "Creating new book",
            book_id=str(book_id),
            title=request.title,
            genre=request.genre.value,
            ai_assistance=request.ai_assistance_level.value
        )
        
        default_word_count:int = WORD_COUNT_MAPPING.get(BookLength.NOVELLA)
        estimated_word_count = WORD_COUNT_MAPPING.get(request.length, default_word_count)
        estimated_chapters = max(1, estimated_word_count // 3000)  # ~3000 words per chapter
        
        # Validate enum combinations
        validation_warnings = validate_enum_combination(request.dict())
        
        # Initialize book data with enhanced structure
        book_data = {
            "id": book_id,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "parameters": request.dict(),
            "status": "initializing",
            "planning_status": "pending",
            "estimated_word_count": estimated_word_count,
            "estimated_chapters": estimated_chapters,
            "completion_percentage": 0.0,
            "validation_warnings": validation_warnings,
            "genre_info": {
                "genre": request.genre.value,
                "sub_genre": request.sub_genre,
                "is_fiction": request.genre.is_fiction,
                "available_subgenres": SubGenreRegistry.get_subgenre_choices(request.genre.value)
            },
            "validation_info": {
                "genre_subgenre_valid": True if not request.sub_genre else validate_book_genre_subgenre(request.genre.value, request.sub_genre)['valid'],
                "warnings": validate_enum_combination(request.dict())
            },         
            # Agent-related fields
            "agent_id": None,
            "planning_results": None,
            "status_message": "Book created, preparing for AI analysis...",
            "next_steps": [
                "AI agents are analyzing your requirements",
                "Story outline will be created",
                "Chapter structure will be planned",
                "Research requirements will be identified"
            ],
            
            # Error tracking
            "error_message": None,
            "retry_count": 0
        }
        
        # Store book data
        books_db[book_id] = book_data
        
        # Determine next steps based on AI assistance level
        next_steps = determine_next_steps(request)
        
        # Start the AI planning process in background
        background_tasks.add_task(
            start_book_planning_with_error_handling,
            book_id,
            request,
            factory
        )
        
        logger.info(
            "Book created successfully, starting AI planning",
            book_id=str(book_id),
            estimated_word_count=estimated_word_count,
            estimated_chapters=estimated_chapters
        )
        
        return BookCreationResponse(
            book_id=book_id,
            title=request.title,
            status="created",
            message="Book plan created successfully. AI agents are now analyzing your requirements.",
            estimated_word_count=estimated_word_count,
            estimated_chapters=estimated_chapters,
            created_at=datetime.now(),
            planning_status="starting",
            next_steps=next_steps
        )
        
    except Exception as e:
        logger.error("Failed to create book", error=str(e), title=request.title)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create book: {str(e)}"
        ) from e

@app.get("/api/books/{book_id}", response_model=BookStatusResponse)
async def get_book(book_id: UUID) -> BookStatusResponse:
    """Get book details and current status."""
    
    try:
        if book_id not in books_db:
            raise HTTPException(status_code=404, detail="Book not found")
        
        book_data = books_db[book_id]
        
        return BookStatusResponse(
            book_id=book_id,
            title=book_data["parameters"]["title"],
            status=book_data["status"],
            created_at=book_data["created_at"],
            updated_at=book_data.get("updated_at"),
            planning_status=book_data.get("planning_status"),
            agent_id=book_data.get("agent_id"),
            planning_results=book_data.get("planning_results"),
            estimated_word_count=book_data["estimated_word_count"],
            estimated_chapters=book_data["estimated_chapters"],
            completion_percentage=book_data.get("completion_percentage", 0.0),
            status_message=book_data.get("status_message"),
            next_steps=book_data.get("next_steps", []),
            error_message=book_data.get("error_message")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get book", book_id=str(book_id), error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/books", response_model=List[BookStatusResponse])
async def list_books() -> List[BookStatusResponse]:
    """List all books with their current status."""
    
    try:
        books = []
        for book_id, book_data in books_db.items():
            try:
                book_response = BookStatusResponse(
                    book_id=book_id,
                    title=book_data["parameters"]["title"],
                    status=book_data["status"],
                    created_at=book_data["created_at"],
                    updated_at=book_data.get("updated_at"),
                    planning_status=book_data.get("planning_status"),
                    agent_id=book_data.get("agent_id"),
                    planning_results=book_data.get("planning_results"),
                    estimated_word_count=book_data["estimated_word_count"],
                    estimated_chapters=book_data["estimated_chapters"],
                    completion_percentage=book_data.get("completion_percentage", 0.0),
                    status_message=book_data.get("status_message"),
                    next_steps=book_data.get("next_steps", []),
                    error_message=book_data.get("error_message")
                )
                books.append(book_response)
            except Exception as e:
                logger.warning("Failed to process book in list", book_id=str(book_id), error=str(e))
                continue
        
        return books
        
    except Exception as e:
        logger.error("Failed to list books", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/books/{book_id}/planning")
async def get_book_planning_details(book_id: UUID) -> Dict[str, Any]:
    """Get detailed planning information for a book."""
    
    try:
        if book_id not in books_db:
            raise HTTPException(status_code=404, detail="Book not found")
        
        book_data = books_db[book_id]
        planning_results = book_data.get("planning_results")
        
        if not planning_results:
            return {
                "book_id": str(book_id),
                "planning_status": book_data.get("planning_status", "not_started"),
                "message": "Planning has not been completed yet"
            }
        
        return {
            "book_id": str(book_id),
            "planning_status": "completed",
            "planning_results": planning_results,
            "agent_id": book_data.get("agent_id"),
            "created_at": book_data["created_at"],
            "updated_at": book_data.get("updated_at")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get planning details", book_id=str(book_id), error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/books/{book_id}/retry-planning")
async def retry_book_planning(
    book_id: UUID,
    background_tasks: BackgroundTasks,
    factory: AgentFactory = Depends(get_agent_factory_dependency)
) -> Dict[str, Any]:
    """Retry planning for a failed book."""
    
    try:
        if book_id not in books_db:
            raise HTTPException(status_code=404, detail="Book not found")
        
        book_data = books_db[book_id]
        
        # Check if retry is appropriate
        current_status = book_data["status"]
        if current_status not in ["planning_failed", "planning_retrying"]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot retry planning for book with status: {current_status}"
            )
        
        # Reset status and retry
        await update_book_status_in_db(book_id, "planning_retry_requested", {
            "status_message": "Planning retry requested, starting new attempt...",
            "retry_count": 0,
            "error_message": None
        })
        
        # Get original request from stored parameters
        original_request = BookCreationRequest(**book_data["parameters"])
        
        # Start planning again
        background_tasks.add_task(
            start_book_planning_with_error_handling,
            book_id,
            original_request,
            factory
        )
        
        logger.info("Planning retry initiated", book_id=str(book_id))
        
        return {
            "success": True,
            "book_id": str(book_id),
            "message": "Planning retry initiated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to retry planning", book_id=str(book_id), error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/agents/status")
async def get_agents_status(
    factory: AgentFactory = Depends(get_agent_factory_dependency)
) -> Dict[str, Any]:
    """Get status of all agents in the factory."""
    
    try:
        agents = await factory.list_agents()
        health_check = await factory.health_check()
        factory_stats = factory.get_factory_stats()
        
        return {
            "factory_healthy": health_check["factory_healthy"],
            "total_agents": len(agents),
            "healthy_agents": health_check["healthy_agents"],
            "agents": agents,
            "factory_stats": factory_stats,
            "health_details": health_check
        }
        
    except Exception as e:
        logger.error("Failed to get agents status", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/agents/cleanup")
async def cleanup_agents(
    factory: AgentFactory = Depends(get_agent_factory_dependency)
) -> Dict[str, Any]:
    """Cleanup all agents (admin endpoint)."""
    
    try:
        cleanup_result = await factory.cleanup_all_agents()
        
        logger.info("Agents cleanup completed", result=cleanup_result)
        
        return {
            "success": True,
            "message": "Agent cleanup completed",
            "cleanup_result": cleanup_result
        }
        
    except Exception as e:
        logger.error("Failed to cleanup agents", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check(
    factory: AgentFactory = Depends(get_agent_factory_dependency)
) -> Dict[str, Any]:
    """Enhanced health check including agent system."""
    
    try:
        # Get agent system health
        agent_health = await factory.health_check()
        
        return {
            "status": "healthy",
            "service": "musequill-book-planner-api",
            "version": "2.0.0",
            "timestamp": datetime.now().isoformat(),
            
            # Database status
            "books_created": len(books_db),
            "database_healthy": True,
            
            # Agent system status
            "agent_system": {
                "healthy": agent_health["factory_healthy"],
                "total_agents": agent_health["total_agents"],
                "healthy_agents": agent_health["healthy_agents"],
                "factory_uptime": agent_health["factory_uptime_seconds"]
            },
            
            # API features
            "features": {
                "book_creation": True,
                "ai_planning": True,
                "agent_management": True,
                "background_processing": True
            },
            
            # Enum system
            "enums_available": len([
                GenreType, SubGenre, BookLength, StoryStructure, PlotType,
                NarrativePOV, PacingType, ConflictType, CharacterRole,
                CharacterArchetype, WorldType, MagicSystemType, TechnologyLevel,
                WritingStyle, ToneType, AgeGroup, AudienceType, ReadingLevel,
                PublicationRoute, ContentWarning, AIAssistanceLevel,
                ResearchPriority, WritingSchedule
            ])
        }
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "service": "musequill-book-planner-api",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# Add this to your existing FastAPI app setup
def setup_agent_routes(app: FastAPI) -> None:
    """Setup agent-related routes."""
    
    # Import and include planning routes
    app.include_router(planning_router, prefix="/api/v1")
    
    logger.info("Agent routes configured successfully")


# Call this when setting up your FastAPI app
setup_agent_routes(app)

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