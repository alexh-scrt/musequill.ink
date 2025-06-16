"""
Integration module for book planning with agent pipeline.
This connects the API endpoint with the Planning Agent.
"""

import asyncio
from typing import Dict, Any, Optional, List
from uuid import UUID
from datetime import datetime
import structlog

from musequill.core.openai_client import OpenAIClient
from musequill.agents.factory import get_agent_factory, AgentFactory
from musequill.core.base.agent import AgentType
from musequill.agents.planning import PlanningAgent
from musequill.models.presets import GenreType

logger = structlog.get_logger(__name__)

# This would be imported from your existing code
# from your_api_module import BookCreationRequest, books_db

async def start_book_planning(book_id: UUID, request) -> Dict[str, Any]:
    """
    Background task to start the AI book planning process.
    This integrates with the existing API endpoint.
    """
    
    planning_result = None
    error_message = None
    
    try:
        genre:Optional[GenreType] = getattr(request, 'genre', None)
        logger.info(
            "Starting AI book planning",
            book_id=str(book_id),
            title=getattr(request, 'title', 'Unknown'),
            genre=genre.value if genre else 'Unknown'
        )
        
        # Check if request has required attributes
        if not hasattr(request, 'title'):
            raise ValueError("Request missing required 'title' attribute")
        
        # Update book status to indicate planning has started
        await update_book_status(book_id, "ai_planning_started", {
            "planning_started_at": datetime.now(),
            "agent_type": "planning",
            "status_message": "AI planning agent is analyzing your requirements..."
        })
        
        # Initialize OpenAI client and agent factory
        openai_client = OpenAIClient()
        factory = get_agent_factory(openai_client)
        
        # Validate factory
        if factory is None:
            raise ValueError("Failed to get agent factory")
        
        # Create or get planning agent
        planning_agent = await factory.get_or_create_agent(
            AgentType.PLANNING,
            agent_id=f"planning_{book_id}"
        )
        
        # Validate agent
        if planning_agent is None:
            raise ValueError("Failed to create planning agent")
        
        logger.info(
            "Planning agent ready",
            agent_id=planning_agent.agent_id,
            book_id=str(book_id)
        )
        
        # Update status
        await update_book_status(book_id, "ai_planning_active", {
            "agent_id": planning_agent.agent_id,
            "status_message": "Planning agent is creating your book outline..."
        })
        
        # Create the book plan using the planning agent
        planning_result = await planning_agent.create_book_plan_from_request(request)
        
        # Validate planning result
        if planning_result is None:
            raise ValueError("Planning agent returned None result")
        
        # Process and store the planning results
        processed_results = await process_planning_results(book_id, planning_result, request)
        
        # Update book with planning results
        await update_book_status(book_id, "planning_completed", {
            "planning_completed_at": datetime.now(),
            "agent_id": planning_agent.agent_id,
            "planning_results": processed_results,
            "status_message": f"Planning completed successfully! Created {len(getattr(planning_result, 'chapters', []))} chapters.",
            "next_steps": getattr(planning_result, 'next_steps', [])
        })
        
        logger.info(
            "Book planning completed successfully",
            book_id=str(book_id),
            chapters_created=len(getattr(planning_result, 'chapters', [])),
            agent_id=planning_agent.agent_id
        )
        
        return {
            "success": True,
            "book_id": str(book_id),
            "planning_result": processed_results,
            "agent_id": planning_agent.agent_id,
            "message": "Book planning completed successfully"
        }
        
    except Exception as e:
        error_message = str(e)
        logger.error(
            "Book planning failed",
            book_id=str(book_id),
            error=error_message,
            title=getattr(request, 'title', 'Unknown')
        )
        
        # Update book status with error
        try:
            await update_book_status(book_id, "planning_failed", {
                "planning_failed_at": datetime.now(),
                "error_message": error_message,
                "status_message": f"Planning failed: {error_message}"
            })
        except Exception as status_error:
            logger.error("Failed to update book status", error=str(status_error))
        
        return {
            "success": False,
            "book_id": str(book_id),
            "error": error_message,
            "message": f"Book planning failed: {error_message}"
        }
    
    finally:
        # Cleanup resources if needed
        try:
            if 'openai_client' in locals() and openai_client:
                # Note: Don't close the client if it's shared
                pass
        except Exception as cleanup_error:
            logger.warning("Cleanup warning", error=str(cleanup_error))

async def process_planning_results(
    book_id: UUID, 
    planning_result, 
    original_request
) -> Dict[str, Any]:
    """
    Process and format planning results for storage and API response.
    
    Args:
        book_id: The book UUID
        planning_result: PlanningResult from the planning agent
        original_request: Original BookCreationRequest
    
    Returns:
        Processed planning data ready for storage
    """
    
    try:
        # Extract core information
        outline = planning_result.outline
        chapters = planning_result.chapters
        research_reqs = planning_result.research_requirements
        
        # Create processed results structure
        processed = {
            "planning_metadata": {
                "story_id": outline.story_id,
                "planning_confidence": planning_result.planning_confidence,
                "planning_date": datetime.now().isoformat(),
                "agent_version": "1.0.0",
                "total_estimated_time": estimate_total_time(planning_result)
            },
            
            "story_outline": {
                "title": outline.title or original_request.title,
                "genre": outline.genre.value,
                "premise": outline.premise,
                "synopsis": outline.synopsis,
                "central_theme": outline.central_theme,
                "tone": outline.tone,
                "setting_overview": outline.setting_overview,
                "estimated_word_count": outline.estimated_word_count,
                "estimated_chapters": outline.estimated_chapters,
                "structure_type": outline.structure.structure_type.value,
                "themes": outline.structure.themes,
                "main_characters": [
                    {
                        "name": char.name,
                        "role": char.role,
                        "description": char.description
                    } for char in outline.main_characters
                ],
                "secondary_themes": outline.secondary_themes
            },
            
            "chapter_structure": [
                {
                    "chapter_number": chapter.chapter_number,
                    "title": chapter.title or f"Chapter {chapter.chapter_number}",
                    "summary": chapter.summary,
                    "purpose": chapter.purpose,
                    "plot_advancement": chapter.plot_advancement,
                    "word_count_target": chapter.word_count_target,
                    "act": getattr(chapter, 'act', None),
                    "story_percentage": chapter.story_percentage,
                    "characters_present": getattr(chapter, 'characters_present', []),
                    "settings": getattr(chapter, 'settings', []),
                    "key_scenes": getattr(chapter, 'key_scenes', [])
                } for chapter in chapters
            ],
            
            "research_requirements": {
                "total_requirements": len(research_reqs.requirements),
                "estimated_research_time": research_reqs.estimated_research_time,
                "categories": research_reqs.research_categories,
                "requirements": [
                    {
                        "topic": req.topic,
                        "priority": req.priority.value,
                        "description": req.description,
                        "reason": req.reason,
                        "keywords": req.keywords,
                        "estimated_time": getattr(req, 'estimated_time', 2)
                    } for req in research_reqs.requirements
                ]
            },
            
            "quality_assessment": {
                "potential_plot_holes": planning_result.potential_plot_holes,
                "suggestions": extract_suggestions_from_result(planning_result),
                "strengths": extract_strengths_from_result(planning_result),
                "areas_for_improvement": extract_improvements_from_result(planning_result)
            },
            
            "next_steps": planning_result.next_steps,
            
            "original_request_metadata": {
                "title": original_request.title,
                "subtitle": original_request.subtitle,
                "description": original_request.description,
                "additional_notes": original_request.additional_notes,
                "ai_assistance_level": original_request.ai_assistance_level.value,
                "publication_route": original_request.publication_route.value,
                "content_warnings": [cw.value for cw in original_request.content_warnings]
            }
        }
        
        # Add book metadata if available from planning result
        if hasattr(planning_result, 'book_metadata'):
            processed["book_metadata"] = planning_result.book_metadata
        
        return processed
        
    except Exception as e:
        logger.error("Failed to process planning results", book_id=str(book_id), error=str(e))
        
        # Return minimal processed results in case of error
        return {
            "planning_metadata": {
                "planning_date": datetime.now().isoformat(),
                "error": str(e)
            },
            "story_outline": {
                "title": original_request.title,
                "genre": original_request.genre.value,
                "error": "Failed to process planning results"
            },
            "chapter_structure": [],
            "research_requirements": {"total_requirements": 0},
            "next_steps": ["Review planning error and retry"]
        }


async def update_book_status(
    book_id: UUID, 
    status: str, 
    additional_data: Optional[Dict[str, Any]] = None
) -> None:
    """
    Update book status in the database.
    
    Args:
        book_id: Book UUID
        status: New status string
        additional_data: Additional data to merge into book record
    """
    
    try:
        # This assumes you have a books_db dict or database
        # Replace this with your actual database update logic
        
        # For the in-memory dict approach from your existing code:
        # if book_id in books_db:
        #     books_db[book_id]["status"] = status
        #     books_db[book_id]["updated_at"] = datetime.now()
        #     
        #     if additional_data:
        #         books_db[book_id].update(additional_data)
        
        # For a real database, you might do something like:
        # await database.update_book(book_id, {"status": status, **additional_data})
        
        logger.info(
            "Book status updated",
            book_id=str(book_id),
            status=status,
            additional_fields=list(additional_data.keys()) if additional_data else []
        )
        
    except Exception as e:
        logger.error("Failed to update book status", book_id=str(book_id), error=str(e))


def should_start_next_phase(request) -> bool:
    """
    Determine if the next phase should start automatically.
    
    Args:
        request: BookCreationRequest
    
    Returns:
        Boolean indicating if next phase should start
    """
    
    # Check AI assistance level
    if hasattr(request, 'ai_assistance_level'):
        assistance_level = request.ai_assistance_level.value.lower()
        
        # High assistance levels should continue automatically
        if assistance_level in ['full_automation', 'high_assistance']:
            return True
        
        # Medium assistance might continue based on other factors
        if assistance_level == 'medium_assistance':
            # Could check other factors like research requirements
            return False
    
    return False


async def schedule_next_phase(
    book_id: UUID, 
    planning_result, 
    original_request
) -> None:
    """
    Schedule the next phase of book creation (research or writing).
    
    Args:
        book_id: Book UUID
        planning_result: Results from planning phase
        original_request: Original book creation request
    """
    
    try:
        logger.info("Scheduling next phase", book_id=str(book_id))
        
        # Determine next phase based on planning results
        research_reqs = planning_result.research_requirements
        
        if research_reqs.requirements and len(research_reqs.requirements) > 0:
            # Schedule research phase
            await update_book_status(book_id, "research_scheduled", {
                "next_phase": "research",
                "research_scheduled_at": datetime.now(),
                "status_message": "Research phase scheduled to begin..."
            })
            
            # Here you would typically:
            # 1. Queue a background task for research
            # 2. Or trigger the research agent
            # await start_book_research(book_id, planning_result, original_request)
            
        else:
            # Skip to writing phase
            await update_book_status(book_id, "writing_scheduled", {
                "next_phase": "writing",
                "writing_scheduled_at": datetime.now(),
                "status_message": "Writing phase scheduled to begin..."
            })
            
            # Here you would typically:
            # 1. Queue a background task for writing
            # 2. Or trigger the writing agent
            # await start_book_writing(book_id, planning_result, original_request)
        
    except Exception as e:
        logger.error("Failed to schedule next phase", book_id=str(book_id), error=str(e))


def estimate_total_time(planning_result) -> Dict[str, int]:
    """Estimate total time for book completion."""
    
    try:
        # Basic time estimation
        research_time = planning_result.research_requirements.estimated_research_time
        
        # Estimate writing time (words per hour varies, but use conservative estimate)
        word_count = planning_result.outline.estimated_word_count
        writing_time = word_count // 500  # ~500 words per hour
        
        # Estimate editing time (usually 20-30% of writing time)
        editing_time = int(writing_time * 0.25)
        
        return {
            "research_hours": research_time,
            "writing_hours": writing_time,
            "editing_hours": editing_time,
            "total_hours": research_time + writing_time + editing_time
        }
        
    except Exception:
        return {"total_hours": 0, "error": "Could not estimate time"}


def extract_suggestions_from_result(planning_result) -> List[str]:
    """Extract suggestions from planning result."""
    suggestions = []
    
    # Extract from next_steps
    if hasattr(planning_result, 'next_steps'):
        suggestions.extend(planning_result.next_steps)
    
    # Could extract from other fields if available
    # if hasattr(planning_result, 'recommendations'):
    #     suggestions.extend(planning_result.recommendations)
    
    return suggestions


def extract_strengths_from_result(planning_result) -> List[str]:
    """Extract strengths from planning result."""
    strengths = []
    
    # Analyze confidence level
    if planning_result.planning_confidence > 0.8:
        strengths.append("High confidence in story structure")
    
    if len(planning_result.chapters) > 0:
        strengths.append("Well-structured chapter outline")
    
    if planning_result.outline.central_theme != "To be identified":
        strengths.append("Clear central theme identified")
    
    return strengths


def extract_improvements_from_result(planning_result) -> List[str]:
    """Extract areas for improvement from planning result."""
    improvements = []
    
    # Analyze potential issues
    if planning_result.potential_plot_holes:
        improvements.extend([f"Address: {hole}" for hole in planning_result.potential_plot_holes])
    
    if planning_result.planning_confidence < 0.7:
        improvements.append("Consider refining story structure for higher confidence")
    
    return improvements