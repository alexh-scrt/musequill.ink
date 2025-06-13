"""
Book Planner Backend API
FastAPI server providing enum data and book creation endpoints
"""
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
)
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
from datetime import datetime
from uuid import UUID, uuid4
import json
import os
from pathlib import Path


# Pydantic models for API requests/responses
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

    @validator('conflict_types')
    @classmethod
    def validate_conflict_types(cls, v):
        if len(v) > 3:
            raise ValueError('Maximum 3 conflict types allowed')
        return v

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

def enum_to_choices(enum_class) -> List[List[str]]:
    """Convert enum to choices format."""
    return [[item.value, item.value.replace('_', ' ').title()] for item in enum_class]

def get_enum_metadata():
    """Get metadata for enums."""
    return {
        "GenreType": {
            "description": "Primary genre classification",
            "popularity": {"fantasy": 9, "romance": 10, "mystery": 8}
        },
        "BookLength": {
            "word_counts": {
                "short_story": [1000, 7500],
                "novella": [17500, 40000],
                "standard_novel": [60000, 90000],
                "long_novel": [90000, 120000]
            }
        }
    }

def get_genre_recommendations():
    """Get recommendations based on genre selection."""
    return {
        "fantasy": {
            "sub_genres": ["high_fantasy", "urban_fantasy"],
            "structures": ["hero_journey", "three_act"],
            "character_roles": ["protagonist", "mentor"],
            "world_types": ["high_fantasy", "low_fantasy"]
        },
        "science_fiction": {
            "sub_genres": ["space_opera", "cyberpunk"],
            "structures": ["three_act", "seven_point"],
            "world_types": ["science_fiction"],
            "tech_levels": ["near_future", "far_future"]
        },
        "mystery": {
            "sub_genres": ["cozy_mystery", "hard_boiled"],
            "structures": ["three_act"],
            "pacing": ["moderate_paced", "fast_paced"]
        },
        "business": {
            "writing_styles": ["conversational", "technical"],
            "audiences": ["professionals"],
            "publication_routes": ["self_published", "traditional"]
        }
    }

@app.get("/")
async def serve_frontend():
    """Serve the main frontend page."""
    return FileResponse(static_dir / "index.html")

@app.get("/api/enums")
async def get_enums() -> EnumData:
    """Get all enum data for frontend form population."""
    
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
        recommendations=get_genre_recommendations()
    )

@app.get("/api/recommendations/{genre}")
async def get_recommendations(genre: str) -> Dict[str, List[str]]:
    """Get recommendations for a specific genre."""
    recommendations = get_genre_recommendations()
    return recommendations.get(genre, {})

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
        BookLength.SHORT_STORY: 5000,
        BookLength.NOVELLA: 30000,
        BookLength.SHORT_NOVEL: 50000,
        BookLength.STANDARD_NOVEL: 75000,
        BookLength.LONG_NOVEL: 105000,
        BookLength.EPIC_NOVEL: 150000
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
        "estimated_chapters": estimated_chapters
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
        "books_created": len(books_db)
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
    
    # Create static directory and save frontend files
    static_dir.mkdir(exist_ok=True)
    
    # Save a simple dashboard page
    dashboard_html = """<!DOCTYPE html>
<html>
<head>
    <title>MuseQuill Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; }
        .status { background: #e8f6f3; padding: 15px; border-radius: 6px; margin: 20px 0; }
        .progress { background: #ecf0f1; height: 10px; border-radius: 5px; margin: 10px 0; }
        .progress-bar { background: #3498db; height: 100%; border-radius: 5px; width: 60%; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📚 MuseQuill Dashboard</h1>
        <div class="status">
            <h3>🎉 Book Plan Created Successfully!</h3>
            <p>Your AI agents are working on your book plan...</p>
            <div class="progress"><div class="progress-bar"></div></div>
            <p><small>60% complete - Generating chapter outline...</small></p>
        </div>
        <a href="/">← Create Another Book</a>
    </div>
</body>
</html>"""
    
    with open(static_dir / "dashboard.html", "w") as f:
        f.write(dashboard_html)
    
    print("🚀 Starting MuseQuill Book Planner API server...")
    print("📝 Frontend available at: http://localhost:8000")
    print("🔧 API docs available at: http://localhost:8000/docs")
    print("💾 Dashboard at: http://localhost:8000/dashboard")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)