"""Base agent class and interfaces."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field
import uuid

from ..openai_client import OpenAIClient


class AgentType(str, Enum):
    """Types of agents in the system."""
    
    PLANNING = "planning"
    WRITING = "writing"
    CHARACTER = "character"
    PLOT = "plot"
    EDITOR = "editor"
    RESEARCH = "research"
    CRITIC = "critic"
    PROPONENT = "proponent"
    MEMORY_MANAGER = "memory_manager"


class AgentState(str, Enum):
    """Agent execution states."""
    
    IDLE = "idle"
    THINKING = "thinking"
    WORKING = "working"
    WAITING = "waiting"
    COMPLETED = "completed"
    ERROR = "error"


class AgentMetrics(BaseModel):
    """Agent performance metrics."""
    
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens_used: int = 0
    total_cost_usd: float = 0.0
    avg_response_time_ms: float = 0.0
    last_activity: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100


class AgentConfig(BaseModel):
    """Agent configuration."""
    
    agent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_type: AgentType
    name: str
    description: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    system_prompt: str = ""
    enabled: bool = True
    timeout_seconds: int = 300
    max_retries: int = 3
    

class BaseAgent(ABC):
    """Base class for all agents in the MuseQuill system."""
    
    def __init__(
        self,
        config: AgentConfig,
        openai_client: OpenAIClient,
        **kwargs
    ):
        self.config = config
        self.openai_client = openai_client
        self.state = AgentState.IDLE
        self.metrics = AgentMetrics()
        self._context: Dict[str, Any] = {}
        self._history: List[Dict[str, Any]] = []
        
    @property
    def agent_id(self) -> str:
        """Get agent ID."""
        return self.config.agent_id
    
    @property
    def agent_type(self) -> AgentType:
        """Get agent type."""
        return self.config.agent_type
    
    @property
    def name(self) -> str:
        """Get agent name."""
        return self.config.name
    
    def set_context(self, key: str, value: Any) -> None:
        """Set context variable."""
        self._context[key] = value
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """Get context variable."""
        return self._context.get(key, default)
    
    def clear_context(self) -> None:
        """Clear all context variables."""
        self._context.clear()
    
    def add_to_history(self, entry: Dict[str, Any]) -> None:
        """Add entry to agent history."""
        entry["timestamp"] = datetime.now().isoformat()
        entry["agent_id"] = self.agent_id
        self._history.append(entry)
    
    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get agent history."""
        if limit:
            return self._history[-limit:]
        return self._history.copy()
    
    def clear_history(self) -> None:
        """Clear agent history."""
        self._history.clear()
    
    @abstractmethod
    async def process(self, input_data: Any, **kwargs) -> Any:
        """Process input and return output."""
        pass
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the agent."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup agent resources."""
        pass
    
    async def _execute_llm_request(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """Execute LLM request with error handling and metrics."""
        start_time = datetime.now()
        
        try:
            self.state = AgentState.THINKING
            
            # Prepare messages with system prompt
            if self.config.system_prompt and messages[0].get("role") != "system":
                messages.insert(0, {
                    "role": "system",
                    "content": self.config.system_prompt
                })
            
            # Make the request
            response = await self.openai_client.chat_completion(
                messages=messages,
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                **kwargs
            )
            
            content = response.choices[0].message.content
            
            # Update metrics
            self.metrics.successful_requests += 1
            self.metrics.total_requests += 1
            
            if response.usage:
                self.metrics.total_tokens_used += response.usage.total_tokens
            
            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            if self.metrics.avg_response_time_ms == 0:
                self.metrics.avg_response_time_ms = response_time
            else:
                self.metrics.avg_response_time_ms = (
                    (self.metrics.avg_response_time_ms * (self.metrics.successful_requests - 1) + response_time) /
                    self.metrics.successful_requests
                )
            
            self.metrics.last_activity = datetime.now()
            
            # Add to history
            self.add_to_history({
                "type": "llm_request",
                "messages": messages,
                "response": content,
                "model": self.config.model,
                "tokens_used": response.usage.total_tokens if response.usage else 0,
                "response_time_ms": response_time
            })
            
            self.state = AgentState.IDLE
            return content
            
        except Exception as e:
            self.metrics.failed_requests += 1
            self.metrics.total_requests += 1
            self.state = AgentState.ERROR
            
            # Add error to history
            self.add_to_history({
                "type": "error",
                "error": str(e),
                "messages": messages
            })
            
            raise
    
    async def stream_response(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Any:
        """Stream LLM response."""
        try:
            self.state = AgentState.THINKING
            
            # Prepare messages with system prompt
            if self.config.system_prompt and messages[0].get("role") != "system":
                messages.insert(0, {
                    "role": "system",
                    "content": self.config.system_prompt
                })
            
            # Stream the response
            async for chunk in await self.openai_client.chat_completion(
                messages=messages,
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=True,
                **kwargs
            ):
                yield chunk
            
            self.state = AgentState.IDLE
            
        except Exception as e:
            self.metrics.failed_requests += 1
            self.metrics.total_requests += 1
            self.state = AgentState.ERROR
            raise
    
    def get_metrics(self) -> AgentMetrics:
        """Get agent metrics."""
        return self.metrics.copy()
    
    def reset_metrics(self) -> None:
        """Reset agent metrics."""
        self.metrics = AgentMetrics()
    
    def is_healthy(self) -> bool:
        """Check if agent is healthy."""
        return (
            self.config.enabled and
            self.state != AgentState.ERROR and
            self.metrics.success_rate >= 50.0  # At least 50% success rate
        )
    
    def __str__(self) -> str:
        return f"{self.config.name} ({self.agent_type.value})"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.agent_id} type={self.agent_type.value} state={self.state.value}>"