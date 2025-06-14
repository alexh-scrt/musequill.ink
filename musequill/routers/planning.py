"""
Planning Router - Dedicated API routes for Planning Agent functionality.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from uuid import UUID, uuid4
import structlog
import json

from musequill.core.openai_client import OpenAIClient
from musequill.agents.factory import get_agent_factory, AgentFactory
from musequill.core.base.agent import AgentType, AgentState
from musequill.models.planning import (
    PlanningRequest, PlanningResult, StoryOutline, ChapterPlan,
    ResearchRequirements, GenreType, StoryStructure, QuickPlanResponse,
    PlanningStatusResponse, ValidationResult, PlanningMetrics
)

logger = structlog.get_logger(__name__)

# Create the router
planning_router = APIRouter(prefix="/planning", tags=["planning"])

# ============================================================================
# Dependency Injection
# ============================================================================

async def get_openai_client() -> OpenAIClient:
    """Dependency to get OpenAI client."""
    return OpenAIClient()


def get_factory_dependency(openai_client: OpenAIClient = Depends(get_openai_client)) -> AgentFactory:
    """Dependency to get agent factory."""
    return get_agent_factory(openai_client)

async def get_planning_agent(
    agent_id: Optional[str] = None,
    factory: AgentFactory = Depends(get_factory_dependency)
):
    """Dependency to get or create a planning agent."""
    try:
        if agent_id:
            agent = await factory.get_agent(agent_id)
            if not agent:
                raise HTTPException(status_code=404, detail="Planning agent not found")
            if agent.agent_type != AgentType.PLANNING:
                raise HTTPException(status_code=400, detail="Agent is not a planning agent")
            return agent
        else:
            # Create or get a planning agent
            return await factory.get_or_create_agent(AgentType.PLANNING)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get planning agent", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get planning agent: {str(e)}")


# ============================================================================
# Core Planning Endpoints
# ============================================================================

@planning_router.post("/create-outline", response_model=Dict[str, Any])
async def create_story_outline(
    request: PlanningRequest,
    background_tasks: BackgroundTasks,
    agent = Depends(get_planning_agent)
):
    """Create a comprehensive story outline."""
    try:
        start_time = datetime.now()
        
        logger.info(
            "Creating story outline",
            genre=request.genre.value if request.genre else "unknown",
            target_length=request.target_length,
            agent_id=agent.agent_id
        )
        
        # Create the outline
        outline = await agent.create_story_outline(
            request.description,
            request.genre or GenreType.OTHER,
            request
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Convert outline to dict for response
        outline_dict = outline.dict()
        outline_dict['processing_time_seconds'] = processing_time
        
        logger.info(
            "Story outline created successfully",
            story_id=outline.story_id,
            estimated_chapters=outline.estimated_chapters,
            processing_time=processing_time,
            agent_id=agent.agent_id
        )
        
        return {
            "success": True,
            "outline": outline_dict,
            "agent_id": agent.agent_id,
            "processing_time_seconds": processing_time,
            "message": f"Story outline created successfully with {outline.estimated_chapters} chapters"
        }
        
    except Exception as e:
        logger.error("Failed to create story outline", error=str(e), agent_id=agent.agent_id)
        raise HTTPException(status_code=500, detail=f"Failed to create story outline: {str(e)}")


@planning_router.post("/create-chapters/{story_id}")
async def create_chapter_structure(
    story_id: str,
    outline: StoryOutline,
    agent = Depends(get_planning_agent)
):
    """Create detailed chapter structure for a story outline."""
    try:
        logger.info("Creating chapter structure", story_id=story_id, agent_id=agent.agent_id)
        
        chapters = await agent.plan_chapter_structure(outline)
        
        chapters_dict = [chapter.dict() for chapter in chapters]
        
        logger.info(
            "Chapter structure created",
            story_id=story_id,
            chapters_created=len(chapters),
            agent_id=agent.agent_id
        )
        
        return {
            "success": True,
            "story_id": story_id,
            "chapters": chapters_dict,
            "total_chapters": len(chapters),
            "agent_id": agent.agent_id,
            "message": f"Created {len(chapters)} chapter plans"
        }
        
    except Exception as e:
        logger.error("Failed to create chapters", story_id=story_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to create chapters: {str(e)}")


@planning_router.post("/analyze-research/{story_id}")
async def analyze_research_requirements(
    story_id: str,
    outline: StoryOutline,
    chapters: Optional[List[ChapterPlan]] = None,
    agent = Depends(get_planning_agent)
):
    """Analyze research requirements for a story."""
    try:
        logger.info("Analyzing research requirements", story_id=story_id, agent_id=agent.agent_id)
        
        research_requirements = await agent.analyze_story_requirements(outline, chapters)
        
        logger.info(
            "Research analysis completed",
            story_id=story_id,
            requirements_found=len(research_requirements.requirements),
            estimated_hours=research_requirements.estimated_research_time,
            agent_id=agent.agent_id
        )
        
        return {
            "success": True,
            "story_id": story_id,
            "research_requirements": research_requirements.dict(),
            "agent_id": agent.agent_id,
            "message": f"Found {len(research_requirements.requirements)} research requirements"
        }
        
    except Exception as e:
        logger.error("Failed to analyze research", story_id=story_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to analyze research: {str(e)}")


@planning_router.post("/complete-plan", response_model=Dict[str, Any])
async def create_complete_planning_result(
    request: PlanningRequest,
    background_tasks: BackgroundTasks,
    agent = Depends(get_planning_agent)
):
    """Create a complete planning result with outline, chapters, and research."""
    try:
        start_time = datetime.now()
        
        logger.info(
            "Creating complete planning result",
            genre=request.genre.value if request.genre else "unknown",
            target_length=request.target_length,
            agent_id=agent.agent_id
        )
        
        # Create complete plan
        planning_result = await agent.create_complete_plan(request)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Convert to dict for response
        result_dict = {
            "outline": planning_result.outline.dict(),
            "chapters": [chapter.dict() for chapter in planning_result.chapters],
            "research_requirements": planning_result.research_requirements.dict(),
            "planning_confidence": planning_result.planning_confidence,
            "potential_plot_holes": planning_result.potential_plot_holes,
            "story_strengths": planning_result.story_strengths,
            "areas_for_improvement": planning_result.areas_for_improvement,
            "next_steps": planning_result.next_steps,
            "processing_time_seconds": processing_time,
            "agent_version": planning_result.agent_version,
            "created_at": planning_result.created_at.isoformat()
        }
        
        logger.info(
            "Complete planning result created",
            story_id=planning_result.outline.story_id,
            chapters=len(planning_result.chapters),
            research_items=len(planning_result.research_requirements.requirements),
            confidence=planning_result.planning_confidence,
            processing_time=processing_time,
            agent_id=agent.agent_id
        )
        
        return {
            "success": True,
            "planning_result": result_dict,
            "agent_id": agent.agent_id,
            "processing_time_seconds": processing_time,
            "message": "Complete planning result created successfully"
        }
        
    except Exception as e:
        logger.error("Failed to create complete plan", error=str(e), agent_id=agent.agent_id)
        raise HTTPException(status_code=500, detail=f"Failed to create complete plan: {str(e)}")


# ============================================================================
# Quick Planning Endpoints
# ============================================================================

@planning_router.post("/quick-plan", response_model=QuickPlanResponse)
async def create_quick_plan(
    description: str,
    genre: Optional[GenreType] = GenreType.OTHER,
    target_length: Optional[int] = 50000,
    agent = Depends(get_planning_agent)
):
    """Create a quick story plan for rapid development."""
    try:
        logger.info("Creating quick plan", genre=genre.value, target_length=target_length)
        
        # Create minimal request
        quick_request = PlanningRequest(
            description=description,
            genre=genre,
            target_length=target_length
        )
        
        # Create outline only (faster)
        outline = await agent.create_story_outline(description, genre, quick_request)
        
        return QuickPlanResponse(
            success=True,
            story_id=outline.story_id,
            title=outline.title or "Untitled Story",
            chapters_planned=outline.estimated_chapters,
            estimated_word_count=outline.estimated_word_count,
            research_requirements=0,  # Quick plan doesn't include research
            confidence_score=0.8,  # Default for quick plans
            message="Quick plan created successfully",
            genre=genre.value,
            structure=outline.structure.structure_type.value,
            main_theme=outline.central_theme
        )
        
    except Exception as e:
        logger.error("Failed to create quick plan", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to create quick plan: {str(e)}")


# ============================================================================
# Agent Management Endpoints
# ============================================================================

@planning_router.get("/agents", response_model=List[Dict[str, Any]])
async def list_planning_agents(factory: AgentFactory = Depends(get_factory_dependency)):
    """List all active planning agents."""
    try:
        all_agents = await factory.list_agents()
        planning_agents = [
            agent for agent in all_agents
            if agent.get("agent_type") == "planning"
        ]
        
        logger.info("Listed planning agents", count=len(planning_agents))
        
        return planning_agents
        
    except Exception as e:
        logger.error("Failed to list planning agents", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list planning agents: {str(e)}")


@planning_router.get("/agents/{agent_id}/status", response_model=PlanningStatusResponse)
async def get_planning_agent_status(
    agent_id: str,
    factory: AgentFactory = Depends(get_factory_dependency)
):
    """Get detailed status of a specific planning agent."""
    try:
        agent = await factory.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Planning agent not found")
        
        if agent.agent_type != AgentType.PLANNING:
            raise HTTPException(status_code=400, detail="Agent is not a planning agent")
        
        # Get planning-specific status
        planning_status = await agent.get_planning_status()
        
        return PlanningStatusResponse(
            agent_id=agent_id,
            status=agent.state.value,
            current_task=planning_status.get("current_task"),
            progress_percentage=planning_status.get("progress_percentage", 0.0),
            estimated_completion=None,  # Could be calculated based on current task
            message=f"Agent {agent_id} is {agent.state.value}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get agent status", agent_id=agent_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get agent status: {str(e)}")


@planning_router.post("/agents/{agent_id}/restart")
async def restart_planning_agent(
    agent_id: str,
    factory: AgentFactory = Depends(get_factory_dependency)
):
    """Restart a planning agent."""
    try:
        logger.info("Restarting planning agent", agent_id=agent_id)
        
        new_agent = await factory.restart_agent(agent_id)
        
        if new_agent:
            return {
                "success": True,
                "agent_id": agent_id,
                "new_agent_id": new_agent.agent_id,
                "message": f"Agent {agent_id} restarted successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Agent not found or restart failed")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to restart agent", agent_id=agent_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to restart agent: {str(e)}")


@planning_router.delete("/agents/{agent_id}")
async def remove_planning_agent(
    agent_id: str,
    factory: AgentFactory = Depends(get_factory_dependency)
):
    """Remove a planning agent."""
    try:
        success = await factory.remove_agent(agent_id)
        
        if success:
            logger.info("Planning agent removed", agent_id=agent_id)
            return {
                "success": True,
                "message": f"Agent {agent_id} removed successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Agent not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to remove agent", agent_id=agent_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to remove agent: {str(e)}")


# ============================================================================
# Story Management Endpoints
# ============================================================================

@planning_router.get("/stories/{story_id}")
async def get_story_plan(
    story_id: str,
    agent_id: Optional[str] = None,
    factory: AgentFactory = Depends(get_factory_dependency)
):
    """Get a stored story plan by ID."""
    try:
        if agent_id:
            agent = await factory.get_agent(agent_id)
        else:
            # Try to find any planning agent with this story
            agents = await factory.get_agents_by_type(AgentType.PLANNING)
            agent = agents[0] if agents else None
        
        if not agent:
            raise HTTPException(status_code=404, detail="No planning agent available")
        
        # Try to get the story from agent context
        outline_key = f"outline_{story_id}"
        chapters_key = f"chapters_{story_id}"
        research_key = f"research_{story_id}"
        
        outline = agent.get_context(outline_key)
        chapters = agent.get_context(chapters_key)
        research = agent.get_context(research_key)
        
        if not outline:
            raise HTTPException(status_code=404, detail="Story plan not found")
        
        return {
            "success": True,
            "story_id": story_id,
            "outline": outline.dict() if hasattr(outline, 'dict') else outline,
            "chapters": [ch.dict() if hasattr(ch, 'dict') else ch for ch in chapters] if chapters else [],
            "research": research.dict() if hasattr(research, 'dict') else research if research else {},
            "agent_id": agent.agent_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get story plan", story_id=story_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get story plan: {str(e)}")


@planning_router.post("/stories/{story_id}/refine")
async def refine_story_plan(
    story_id: str,
    improvements: List[str],
    agent_id: Optional[str] = None,
    factory: AgentFactory = Depends(get_factory_dependency)
):
    """Refine an existing story plan based on feedback."""
    try:
        logger.info("Refining story plan", story_id=story_id, improvements_count=len(improvements))
        
        if agent_id:
            agent = await factory.get_agent(agent_id)
        else:
            agents = await factory.get_agents_by_type(AgentType.PLANNING)
            agent = agents[0] if agents else None
        
        if not agent:
            raise HTTPException(status_code=404, detail="No planning agent available")
        
        # Get current story plan
        outline_key = f"outline_{story_id}"
        outline = agent.get_context(outline_key)
        
        if not outline:
            raise HTTPException(status_code=404, detail="Story plan not found")
        
        # Create refinement prompt
        improvement_text = "\n".join([f"- {imp}" for imp in improvements])
        refinement_request = f"""
Please refine this story plan based on the following improvements:

{improvement_text}

Current story outline: {outline.dict() if hasattr(outline, 'dict') else str(outline)}
"""
        
        # Create new request for refinement
        refine_request = PlanningRequest(
            description=refinement_request,
            genre=outline.genre if hasattr(outline, 'genre') else GenreType.OTHER,
            target_length=outline.estimated_word_count if hasattr(outline, 'estimated_word_count') else 50000
        )
        
        # Create refined plan
        refined_result = await agent.create_complete_plan(refine_request)
        
        return {
            "success": True,
            "story_id": story_id,
            "refined_plan": {
                "outline": refined_result.outline.dict(),
                "chapters": [ch.dict() for ch in refined_result.chapters],
                "research": refined_result.research_requirements.dict()
            },
            "improvements_applied": improvements,
            "agent_id": agent.agent_id,
            "message": "Story plan refined successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to refine story plan", story_id=story_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to refine story plan: {str(e)}")


# ============================================================================
# Validation and Analysis Endpoints
# ============================================================================

@planning_router.post("/validate-plan")
async def validate_story_plan(
    outline: StoryOutline,
    chapters: List[ChapterPlan],
    research_requirements: Optional[ResearchRequirements] = None,
    agent = Depends(get_planning_agent)
):
    """Validate a complete story plan for quality and coherence."""
    try:
        logger.info("Validating story plan", story_id=outline.story_id, agent_id=agent.agent_id)
        
        # Create default research requirements if not provided
        if not research_requirements:
            research_requirements = ResearchRequirements(story_id=outline.story_id)
        
        # Validate the plan
        validation_result = await agent.validate_plan(outline, chapters, research_requirements)
        
        # Convert to validation response format
        validation_response = ValidationResult(
            overall_score=validation_result.planning_confidence,
            plot_coherence=0.8,  # Could be calculated from validation
            character_consistency=0.8,
            pacing_assessment=0.8,
            theme_development=0.8,
            critical_issues=validation_result.potential_plot_holes,
            warnings=[],  # Could extract from validation
            suggestions=validation_result.next_steps,
            strengths=validation_result.story_strengths,
            weaknesses=validation_result.areas_for_improvement,
            improvement_priorities=validation_result.next_steps[:3]  # Top 3 priorities
        )
        
        return {
            "success": True,
            "story_id": outline.story_id,
            "validation_result": validation_response.dict(),
            "agent_id": agent.agent_id,
            "message": f"Validation completed with score: {validation_response.overall_score:.2f}"
        }
        
    except Exception as e:
        logger.error("Failed to validate story plan", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to validate story plan: {str(e)}")


# ============================================================================
# Utility and Testing Endpoints
# ============================================================================

@planning_router.post("/test-connection")
async def test_planning_connection(factory: AgentFactory = Depends(get_factory_dependency)):
    """Test the planning system connection and functionality."""
    try:
        # Test agent creation
        test_agent = await factory.create_agent(
            AgentType.PLANNING,
            agent_id=f"test_agent_{uuid4().hex[:8]}"
        )
        
        # Test basic functionality
        test_request = PlanningRequest(
            description="A short test story about a robot learning to paint",
            genre=GenreType.SCIENCE_FICTION,
            target_length=1000
        )
        
        outline = await test_agent.create_story_outline(
            test_request.description,
            test_request.genre,
            test_request
        )
        
        # Cleanup test agent
        await factory.remove_agent(test_agent.agent_id)
        
        return {
            "success": True,
            "message": "Planning system connection test successful",
            "test_results": {
                "agent_creation": True,
                "outline_creation": True,
                "agent_cleanup": True,
                "test_story_id": outline.story_id
            }
        }
        
    except Exception as e:
        logger.error("Planning connection test failed", error=str(e))
        return {
            "success": False,
            "message": f"Planning system test failed: {str(e)}",
            "test_results": {
                "agent_creation": False,
                "outline_creation": False,
                "agent_cleanup": False
            }
        }


@planning_router.get("/metrics")
async def get_planning_metrics(factory: AgentFactory = Depends(get_factory_dependency)):
    """Get planning system metrics and statistics."""
    try:
        # Get agent factory stats
        factory_stats = factory.get_factory_stats()
        
        # Get planning agents
        planning_agents = await factory.get_agents_by_type(AgentType.PLANNING)
        
        # Calculate metrics
        total_planning_agents = len(planning_agents)
        healthy_agents = sum(1 for agent in planning_agents if agent.is_healthy())
        
        # Could aggregate metrics from agents if they track them
        metrics = PlanningMetrics(
            total_plans_created=factory_stats.get("total_agents_created", 0),
            average_planning_time=0.0,  # Would need to track this
            success_rate=healthy_agents / max(total_planning_agents, 1),
            average_confidence=0.8,  # Would aggregate from completed plans
            plots_with_holes=0,  # Would track from validations
            research_requirements_avg=5.0,  # Would calculate from plans
            average_chapters_per_book=20.0,
            average_words_per_chapter=3000.0
        )
        
        return {
            "success": True,
            "metrics": metrics.dict(),
            "factory_stats": factory_stats,
            "agent_counts": {
                "total_planning_agents": total_planning_agents,
                "healthy_agents": healthy_agents,
                "unhealthy_agents": total_planning_agents - healthy_agents
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get planning metrics", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get planning metrics: {str(e)}")


# ============================================================================
# Batch Operations
# ============================================================================

@planning_router.post("/batch/create-outlines")
async def create_multiple_outlines(
    requests: List[PlanningRequest],
    background_tasks: BackgroundTasks,
    max_concurrent: int = Query(default=3, le=10),
    factory: AgentFactory = Depends(get_factory_dependency)
):
    """Create multiple story outlines concurrently."""
    try:
        if len(requests) > 20:
            raise HTTPException(status_code=400, detail="Maximum 20 requests allowed per batch")
        
        logger.info("Creating batch outlines", batch_size=len(requests), max_concurrent=max_concurrent)
        
        # Create multiple agents for concurrent processing
        agents = []
        for i in range(min(max_concurrent, len(requests))):
            agent = await factory.create_agent(
                AgentType.PLANNING,
                agent_id=f"batch_agent_{uuid4().hex[:8]}"
            )
            agents.append(agent)
        
        # Process requests concurrently
        import asyncio
        
        async def process_request(request, agent):
            try:
                outline = await agent.create_story_outline(
                    request.description,
                    request.genre or GenreType.OTHER,
                    request
                )
                return {"success": True, "outline": outline.dict()}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        # Distribute requests among agents
        tasks = []
        for i, request in enumerate(requests):
            agent = agents[i % len(agents)]
            tasks.append(process_request(request, agent))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Cleanup agents
        for agent in agents:
            await factory.remove_agent(agent.agent_id)
        
        successful = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
        
        return {
            "success": True,
            "batch_size": len(requests),
            "successful": successful,
            "failed": len(requests) - successful,
            "results": results,
            "message": f"Batch processing completed: {successful}/{len(requests)} successful"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Batch outline creation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")


# ============================================================================
# Health Check
# ============================================================================

@planning_router.get("/health")
async def planning_health_check(factory: AgentFactory = Depends(get_factory_dependency)):
    """Health check specifically for planning system."""
    try:
        health = await factory.health_check()
        planning_agents = await factory.get_agents_by_type(AgentType.PLANNING)
        
        return {
            "status": "healthy" if health["factory_healthy"] else "unhealthy",
            "planning_system": {
                "factory_healthy": health["factory_healthy"],
                "total_planning_agents": len(planning_agents),
                "healthy_planning_agents": sum(1 for agent in planning_agents if agent.is_healthy()),
                "factory_uptime_seconds": health["factory_uptime_seconds"]
            },
            "capabilities": {
                "outline_creation": True,
                "chapter_planning": True,
                "research_analysis": True,
                "plan_validation": True,
                "batch_processing": True
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Planning health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }