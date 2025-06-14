"""OpenAI configuration management."""
from dataclasses import dataclass
from typing import Dict, Optional
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class OpenAIConfig(BaseSettings):
    """OpenAI API configuration."""
    
    api_key: str = Field(..., description="OpenAI API key")
    base_url: str = Field(default="https://api.openai.com/v1", description="OpenAI API base URL")
    default_model: str = Field(default="gpt-4o", description="Default model to use")
    max_tokens: int = Field(default=4096, description="Maximum tokens per request")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    request_timeout: int = Field(default=60, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retries")
    
    # Model-specific configurations
    planning_model: str = Field(default="gpt-4", description="Model for planning agent")
    writing_model: str = Field(default="gpt-4o", description="Model for writing agent")
    character_model: str = Field(default="gpt-4", description="Model for character agent")
    plot_model: str = Field(default="gpt-4", description="Model for plot agent")
    editor_model: str = Field(default="gpt-4", description="Model for editor agent")
    research_model: str = Field(default="gpt-3.5-turbo", description="Model for research agent")
    critic_model: str = Field(default="gpt-4", description="Model for critic agent")
    proponent_model: str = Field(default="gpt-4", description="Model for proponent agent")
    memory_model: str = Field(default="gpt-3.5-turbo", description="Model for memory management")
    
    # Cost management
    daily_budget_usd: float = Field(default=100.0, description="Daily budget in USD")
    monthly_budget_usd: float = Field(default=3000.0, description="Monthly budget in USD")
    cost_tracking_enabled: bool = Field(default=True, description="Enable cost tracking")
    alert_threshold_pct: float = Field(default=80.0, ge=0.0, le=100.0, description="Budget alert threshold percentage")
    
    class Config:
        env_prefix = "OPENAI_"
        case_sensitive = False

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate OpenAI API key format."""
        if not v.startswith("sk-"):
            raise ValueError("OpenAI API key must start with 'sk-'")
        if len(v) < 20:
            raise ValueError("OpenAI API key is too short")
        return v

    def get_model_for_agent(self, agent_type: str) -> str:
        """Get the configured model for a specific agent type."""
        model_mapping = {
            "planning": self.planning_model,
            "writing": self.writing_model,
            "character": self.character_model,
            "plot": self.plot_model,
            "editor": self.editor_model,
            "research": self.research_model,
            "critic": self.critic_model,
            "proponent": self.proponent_model,
            "memory": self.memory_model,
        }
        return model_mapping.get(agent_type, self.default_model)


class ModelConfiguration(BaseModel):
    """Configuration for a specific model."""
    
    name: str
    max_tokens: int
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[list[str]] = None
    
    class Config:
        extra = "allow"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""
    requests_per_minute: int = 20
    requests_per_hour: int = 100
    storage_uri: str = "memory://"  # Use "redis://localhost:6379" for Redis
