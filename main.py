"""
Updated API endpoint integration for book creation with Planning Agent.
This replaces your existing create_book endpoint and start_book_planning function.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from typing import Dict, List, Optional, Any
from datetime import datetime
from uuid import UUID, uuid4
import structlog

# Your existing imports
from musequill.models.presets import (
    GenreType, SubGenre, BookLength, StoryStructure, PlotType,
    NarrativePOV, PacingType, ConflictType, CharacterRole,
    CharacterArchetype, WorldType, MagicSystemType, TechnologyLevel,
    WritingStyle, ToneType, AgeGroup, AudienceType, ReadingLevel,
    PublicationRoute, ContentWarning, AIAssistanceLevel,
    ResearchPriority, WritingSchedule
)

# New agent system imports
from musequill.core.openai_client import OpenAIClient
from musequill.agents.factory.agent_factory import get_agent_factory, AgentFactory
from musequill.agents.planning.integration import start_book_planning
from musequill.core.base.agent import AgentType

logger = structlog.get_logger(__name__)

# Your existing BookCreationRequest model (keeping it the same)
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

    # ... (your existing validators)


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


# Dependency injection for agent system
async def get_openai_client() -> OpenAIClient:
    """Get OpenAI client instance."""
    return OpenAIClient()


async def get_agent_factory(
    openai_client: OpenAIClient = Depends(get_openai_client)
) -> AgentFactory:
    """Get agent factory instance."""
    return get_agent_factory(openai_client)


# Your existing books_db (you can replace this with real database later)
books_db: Dict[UUID, Dict[str, Any]] = {}


@app.post("/api/books/create", response_model=BookCreationResponse)
async def create_book(
    request: BookCreationRequest,
    background_tasks: BackgroundTasks,
    factory: AgentFactory = Depends(get_agent_factory)
) -> BookCreationResponse:
    """Create a new book plan based on user parameters with AI agent planning."""
    
    try:
        # Generate unique book ID
        book_id = uuid4()
        
        logger.info(
            "Creating new book",
            book_id=str(book_id),
            title=request.title,
            genre=request.genre.value,
            ai_assistance=request.ai_assistance_level.value
        )
        
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
        )


async def start_book_planning_with_error_handling(
    book_id: UUID,
    request: BookCreationRequest,
    factory: AgentFactory
) -> None:
    """
    Wrapper for start_book_planning with comprehensive error handling.
    """
    
    max_retries = 2
    retry_count = 0
    
    while retry_count <= max_retries:
        try:
            # Call the actual planning function
            result = await start_book_planning(book_id, request)
            
            # Update book with success
            if book_id in books_db:
                books_db[book_id].update({
                    "planning_completed": True,
                    "retry_count": retry_count,
                    "last_attempt": datetime.now()
                })
            
            logger.info(
                "Book planning completed successfully",
                book_id=str(book_id),
                retry_count=retry_count,
                success=result.get("success", False)
            )
            return
            
        except Exception as e:
            retry_count += 1
            error_msg = str(e)
            
            logger.error(
                "Book planning attempt failed",
                book_id=str(book_id),
                retry_count=retry_count,
                max_retries=max_retries,
                error=error_msg
            )
            
            # Update book with retry information
            if book_id in books_db:
                books_db[book_id].update({
                    "retry_count": retry_count,
                    "last_error": error_msg,
                    "last_attempt": datetime.now()
                })
            
            if retry_count <= max_retries:
                # Update status for retry
                await update_book_status_in_db(book_id, "planning_retrying", {
                    "status_message": f"Planning failed, retrying ({retry_count}/{max_retries})...",
                    "error_message": error_msg
                })
                
                # Wait before retry (exponential backoff)
                import asyncio
                await asyncio.sleep(min(2 ** retry_count, 10))
            else:
                # Final failure
                await update_book_status_in_db(book_id, "planning_failed", {
                    "status_message": f"Planning failed after {max_retries} attempts",
                    "error_message": error_msg,
                    "next_steps": [
                        "Review error message",
                        "Try creating book again with different parameters",
                        "Contact support if issue persists"
                    ]
                })
                break


async def update_book_status_in_db(
    book_id: UUID,
    status: str,
    additional_data: Optional[Dict[str, Any]] = None
) -> None:
    """Update book status in database."""
    try:
        if book_id in books_db:
            books_db[book_id]["status"] = status
            books_db[book_id]["updated_at"] = datetime.now()
            
            if additional_data:
                books_db[book_id].update(additional_data)
                
    except Exception as e:
        logger.error("Failed to update book status", book_id=str(book_id), error=str(e))


def determine_next_steps(request: BookCreationRequest) -> List[str]:
    """Determine next steps based on AI assistance level and request parameters."""
    
    steps = []
    assistance_level = request.ai_assistance_level.value.lower()
    
    if assistance_level in ['full_automation', 'high_assistance']:
        steps = [
            "AI agents are analyzing your requirements",
            "Story outline and structure will be created automatically",
            "Chapter breakdown will be generated",
            "Research requirements will be identified",
            "Writing process will begin automatically"
        ]
    elif assistance_level == 'medium_assistance':
        steps = [
            "AI agents are creating your story outline",
            "Chapter structure will be planned",
            "Research needs will be identified",
            "You'll review and approve the plan before writing begins"
        ]
    else:  # minimal or guided assistance
        steps = [
            "AI will create a basic story outline",
            "You'll receive suggestions for improvement",
            "Manual review and refinement will be required",
            "Writing guidance will be provided as needed"
        ]
    
    # Add genre-specific steps
    if request.genre in [GenreType.FANTASY, GenreType.SCIENCE_FICTION]:
        steps.append("World-building elements will be analyzed")
    
    if request.research_priority in [ResearchPriority.HIGH, ResearchPriority.CRITICAL]:
        steps.append("Detailed research plan will be created")
    
    return steps


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
    factory: AgentFactory = Depends(get_agent_factory)
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
    factory: AgentFactory = Depends(get_agent_factory)
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
    factory: AgentFactory = Depends(get_agent_factory)
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
    factory: AgentFactory = Depends(get_agent_factory)
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
    from musequill.agents.planning.api import planning_router
    app.include_router(planning_router, prefix="/api/v1")
    
    logger.info("Agent routes configured successfully")


# Call this when setting up your FastAPI app
# setup_agent_routes(app)