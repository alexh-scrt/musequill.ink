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
from musequill.api.model import (
    BookCreationRequest,
    BookCreationResponse,
    BookStatusResponse,
    EnumChoice,
    EnumData,
    GenreSubgenreValidationRequest,
    validate_book_genre_subgenre
)
from musequill.models.subgenre import (
    SubGenreRegistry, 
    SubGenre,
    get_subgenres_for_frontend,
    validate_book_genre_subgenre
)
from musequill.agents.factory import AgentFactory
from musequill.routers.planning import planning_router
from musequill.core.openai_client.client import OpenAIClient
from musequill.config.logging import get_logger
from musequill.database import book as book_db
from musequill.agents.integration import start_book_planning as agent_start_book_planning

# Get logger and test
logger = get_logger("test_logger")

def enum_to_choices(enum_class) -> List[List[str]]:
    """Convert enum to choices format - handles both regular enums and dynamic SubGenre."""
    if enum_class == SubGenre:
        # For the master SubGenre enum, get all sub-genres
        return SubGenreRegistry.get_all_subgenres()
    else:
        # For regular enums
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
            
            # Add null checking
            if result is None:
                raise ValueError("Planning function returned None - check function implementation")
            
            if not isinstance(result, dict):
                raise ValueError(f"Planning function returned {type(result)}, expected dict")
            
            # Check for success indicator
            success = result.get("success", False)
            if not success:
                error_msg = result.get("error", "Unknown planning error")
                raise ValueError(f"Planning failed: {error_msg}")
            
            # Update book with success
            if book_db.book_exists(str(book_id)):
                book_db.update_book(str(book_id), {
                    "planning_completed": True,
                    "retry_count": retry_count,
                    "last_attempt": datetime.now(),
                    "planning_results": result.get("planning_result"),
                    "agent_id": result.get("agent_id")
                })
            
            logger.info(
                "Book planning completed successfully",
                book_id=str(book_id),
                retry_count=retry_count,
                success=success
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
            if book_db.book_exists(book_id):
                book_db.update_book(book_id, {
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
        if book_db.book_exists(book_id):
            if additional_data:
                book_db.update_book_status(book_id, status, additional_data)            
                
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


async def start_book_planning(book_id: UUID, request: BookCreationRequest) -> Dict[str, Any]:
    """
    Main book planning function.
    """
    try:
        
        # Call the agent integration function
        result = await agent_start_book_planning(book_id, request)
        
        # Ensure we always return a dictionary
        if result is None:
            return {
                "success": False,
                "error": "Planning function returned None",
                "book_id": str(book_id)
            }
        
        return result
        
    except ImportError as e:
        logger.error("Failed to import planning function", error=str(e))
        return {
            "success": False,
            "error": f"Import error: {str(e)}",
            "book_id": str(book_id)
        }
    except Exception as e:
        logger.error("Planning function failed", book_id=str(book_id), error=str(e))
        return {
            "success": False,
            "error": str(e),
            "book_id": str(book_id)
        }