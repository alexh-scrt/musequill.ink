"""
Agent factory for creating and managing MuseQuill agents.
"""

from typing import Dict, Optional, Type, Any, List, Union
from enum import Enum
import uuid
import asyncio
import structlog
from datetime import datetime

from musequill.core.openai_client import OpenAIClient
from musequill.config import get_settings
from musequill.core.base.agent import BaseAgent, AgentType, AgentConfig, AgentState

# Import agent classes (only PlanningAgent for now)
from musequill.agents.planning import PlanningAgent

logger = structlog.get_logger(__name__)


class AgentFactory:
    """Factory for creating and managing MuseQuill agents."""
    
    def __init__(self, openai_client: OpenAIClient):
        self.openai_client = openai_client
        self.settings = get_settings()
        self._agents: Dict[str, BaseAgent] = {}
        self._agent_classes: Dict[AgentType, Type[BaseAgent]] = {
            AgentType.PLANNING: PlanningAgent,
            # TODO: Add other agents as they're implemented
            # AgentType.WRITING: WritingAgent,
            # AgentType.CHARACTER: CharacterAgent,
            # AgentType.PLOT: PlotAgent,
            # AgentType.EDITOR: EditorAgent,
            # AgentType.RESEARCH: ResearchAgent,
            # AgentType.CRITIC: CriticAgent,
            # AgentType.PROPONENT: ProponentAgent,
            # AgentType.MEMORY_MANAGER: MemoryManagerAgent,
        }
        self._created_at = datetime.now()
        self._total_agents_created = 0
    
    async def create_agent(
        self, 
        agent_type: AgentType, 
        agent_id: Optional[str] = None,
        **kwargs
    ) -> BaseAgent:
        """Create a new agent instance."""
        try:
            if agent_type not in self._agent_classes:
                available_types = list(self._agent_classes.keys())
                raise ValueError(
                    f"Unknown agent type: {agent_type}. "
                    f"Available types: {[t.value for t in available_types]}"
                )
            
            agent_class = self._agent_classes[agent_type]
            
            # Generate agent ID if not provided
            if not agent_id:
                agent_id = f"{agent_type.value}_{uuid.uuid4().hex[:8]}"
            
            # Get model configuration for this agent type
            model = self.settings.get_openai_model_for_agent(agent_type.value)
            
            # Create agent configuration
            config_kwargs = {
                "model": model,
                "agent_id": agent_id,
                **kwargs
            }
            
            # Create the agent
            agent = agent_class(self.openai_client, **config_kwargs)
            await agent.initialize()
            
            # Store the agent
            self._agents[agent.agent_id] = agent
            self._total_agents_created += 1
            
            logger.info(
                "Agent created successfully",
                agent_type=agent_type.value,
                agent_id=agent.agent_id,
                model=model,
                total_agents=len(self._agents)
            )
            
            return agent
            
        except Exception as e:
            logger.error(
                "Failed to create agent",
                agent_type=agent_type.value,
                error=str(e)
            )
            raise
    
    async def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an existing agent by ID."""
        return self._agents.get(agent_id)
    
    async def get_or_create_agent(
        self, 
        agent_type: AgentType,
        agent_id: Optional[str] = None,
        **kwargs
    ) -> BaseAgent:
        """Get existing agent or create new one if it doesn't exist."""
        
        # If agent_id provided, try to get existing agent
        if agent_id and agent_id in self._agents:
            existing_agent = self._agents[agent_id]
            if existing_agent.agent_type == agent_type:
                return existing_agent
            else:
                logger.warning(
                    "Agent ID exists but type mismatch",
                    requested_type=agent_type.value,
                    existing_type=existing_agent.agent_type.value
                )
        
        # Look for any existing agent of the requested type
        for agent in self._agents.values():
            if agent.agent_type == agent_type and agent.is_healthy():
                logger.info(
                    "Reusing existing healthy agent",
                    agent_type=agent_type.value,
                    agent_id=agent.agent_id
                )
                return agent
        
        # Create new agent
        return await self.create_agent(agent_type, agent_id, **kwargs)
    
    async def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent by ID."""
        try:
            if agent_id not in self._agents:
                logger.warning("Attempted to remove non-existent agent", agent_id=agent_id)
                return False
            
            agent = self._agents[agent_id]
            await agent.cleanup()
            del self._agents[agent_id]
            
            logger.info(
                "Agent removed successfully",
                agent_id=agent_id,
                agent_type=agent.agent_type.value,
                remaining_agents=len(self._agents)
            )
            
            return True
            
        except Exception as e:
            logger.error("Failed to remove agent", agent_id=agent_id, error=str(e))
            return False
    
    async def list_agents(self) -> List[Dict[str, Any]]:
        """List all active agents with their status."""
        agents_info = []
        
        for agent in self._agents.values():
            try:
                metrics = agent.get_metrics()
                agents_info.append({
                    "agent_id": agent.agent_id,
                    "agent_type": agent.agent_type.value,
                    "name": agent.name,
                    "state": agent.state.value,
                    "healthy": agent.is_healthy(),
                    "created_at": getattr(agent, 'created_at', None),
                    "metrics": metrics.dict() if hasattr(metrics, 'dict') else metrics
                })
            except Exception as e:
                logger.warning("Failed to get agent info", agent_id=agent.agent_id, error=str(e))
                agents_info.append({
                    "agent_id": agent.agent_id,
                    "agent_type": agent.agent_type.value,
                    "name": getattr(agent, 'name', 'Unknown'),
                    "state": "error",
                    "healthy": False,
                    "error": str(e)
                })
        
        return agents_info
    
    async def get_agents_by_type(self, agent_type: AgentType) -> List[BaseAgent]:
        """Get all agents of a specific type."""
        return [
            agent for agent in self._agents.values() 
            if agent.agent_type == agent_type
        ]
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all agents."""
        total_agents = len(self._agents)
        healthy_agents = 0
        unhealthy_agents = []
        
        for agent in self._agents.values():
            try:
                if agent.is_healthy():
                    healthy_agents += 1
                else:
                    unhealthy_agents.append({
                        "agent_id": agent.agent_id,
                        "agent_type": agent.agent_type.value,
                        "state": agent.state.value
                    })
            except Exception as e:
                unhealthy_agents.append({
                    "agent_id": agent.agent_id,
                    "agent_type": getattr(agent, 'agent_type', 'unknown'),
                    "error": str(e)
                })
        
        return {
            "factory_healthy": True,
            "total_agents": total_agents,
            "healthy_agents": healthy_agents,
            "unhealthy_agents": len(unhealthy_agents),
            "unhealthy_details": unhealthy_agents,
            "factory_uptime_seconds": (datetime.now() - self._created_at).total_seconds(),
            "total_agents_created": self._total_agents_created,
            "openai_client_healthy": self.openai_client is not None
        }
    
    async def cleanup_all_agents(self) -> Dict[str, int]:
        """Cleanup all agents."""
        successful_cleanups = 0
        failed_cleanups = 0
        
        agent_ids = list(self._agents.keys())
        
        for agent_id in agent_ids:
            try:
                success = await self.remove_agent(agent_id)
                if success:
                    successful_cleanups += 1
                else:
                    failed_cleanups += 1
            except Exception as e:
                logger.error("Failed to cleanup agent", agent_id=agent_id, error=str(e))
                failed_cleanups += 1
        
        logger.info(
            "Agent cleanup completed",
            successful=successful_cleanups,
            failed=failed_cleanups
        )
        
        return {
            "successful_cleanups": successful_cleanups,
            "failed_cleanups": failed_cleanups,
            "remaining_agents": len(self._agents)
        }
    
    async def restart_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Restart an agent (cleanup and recreate)."""
        try:
            if agent_id not in self._agents:
                logger.warning("Cannot restart non-existent agent", agent_id=agent_id)
                return None
            
            old_agent = self._agents[agent_id]
            agent_type = old_agent.agent_type
            
            # Remove old agent
            await self.remove_agent(agent_id)
            
            # Create new agent with same ID
            new_agent = await self.create_agent(agent_type, agent_id)
            
            logger.info("Agent restarted successfully", agent_id=agent_id)
            return new_agent
            
        except Exception as e:
            logger.error("Failed to restart agent", agent_id=agent_id, error=str(e))
            return None
    
    def get_factory_stats(self) -> Dict[str, Any]:
        """Get factory statistics."""
        return {
            "total_agents": len(self._agents),
            "agents_by_type": {
                agent_type.value: len([
                    a for a in self._agents.values() 
                    if a.agent_type == agent_type
                ]) for agent_type in AgentType
            },
            "factory_uptime_seconds": (datetime.now() - self._created_at).total_seconds(),
            "total_agents_created": self._total_agents_created,
            "available_agent_types": [t.value for t in self._agent_classes.keys()]
        }


# Global factory instance (singleton pattern)
_factory_instance: Optional[AgentFactory] = None


def get_agent_factory(openai_client: OpenAIClient) -> AgentFactory:
    """Get or create the global agent factory instance."""
    global _factory_instance
    
    if _factory_instance is None:
        _factory_instance = AgentFactory(openai_client)
        logger.info("Agent factory instance created")
    
    return _factory_instance


async def reset_agent_factory():
    """Reset the global factory instance (useful for testing)."""
    global _factory_instance
    
    if _factory_instance:
        await _factory_instance.cleanup_all_agents()
        _factory_instance = None
        logger.info("Agent factory instance reset")


# Convenience functions for common operations
async def create_planning_agent(openai_client: OpenAIClient, **kwargs) -> PlanningAgent:
    """Convenience function to create a planning agent."""
    factory = get_agent_factory(openai_client)
    agent = await factory.create_agent(AgentType.PLANNING, **kwargs)
    return agent


async def get_or_create_planning_agent(openai_client: OpenAIClient, **kwargs) -> PlanningAgent:
    """Convenience function to get or create a planning agent."""
    factory = get_agent_factory(openai_client)
    agent = await factory.get_or_create_agent(AgentType.PLANNING, **kwargs)
    return agent