"""OpenAI model definitions and utilities."""

from enum import Enum
from typing import Dict, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class ModelType(str, Enum):
    """Available OpenAI model types."""
    
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4 = "gpt-4"
    GPT_35_TURBO = "gpt-3.5-turbo"
    GPT_35_TURBO_16K = "gpt-3.5-turbo-16k"
    
    # Embedding models
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"


class ModelPricing(BaseModel):
    """Model pricing information."""
    
    input_cost_per_1k_tokens: float = Field(..., description="Cost per 1K input tokens in USD")
    output_cost_per_1k_tokens: float = Field(..., description="Cost per 1K output tokens in USD")
    context_window: int = Field(..., description="Maximum context window size")
    max_output_tokens: int = Field(..., description="Maximum output tokens")


class OpenAIModel(BaseModel):
    """OpenAI model information and configuration."""
    
    name: str = Field(..., description="Model name")
    model_type: ModelType = Field(..., description="Model type enum")
    pricing: ModelPricing = Field(..., description="Pricing information")
    capabilities: list[str] = Field(default_factory=list, description="Model capabilities")
    recommended_for: list[str] = Field(default_factory=list, description="Recommended use cases")
    
    class Config:
        use_enum_values = True


class TokenUsage(BaseModel):
    """Token usage tracking."""
    
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    @property
    def efficiency_ratio(self) -> float:
        """Calculate efficiency ratio (output/input)."""
        if self.prompt_tokens == 0:
            return 0.0
        return self.completion_tokens / self.prompt_tokens


class RequestMetrics(BaseModel):
    """Request performance metrics."""
    
    model: str
    timestamp: datetime
    duration_ms: float
    token_usage: TokenUsage
    cost_usd: float
    success: bool
    error_message: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# Model pricing data (as of 2024)
MODEL_PRICING: Dict[str, ModelPricing] = {
    ModelType.GPT_4O: ModelPricing(
        input_cost_per_1k_tokens=0.005,
        output_cost_per_1k_tokens=0.015,
        context_window=128000,
        max_output_tokens=4096
    ),
    ModelType.GPT_4O_MINI: ModelPricing(
        input_cost_per_1k_tokens=0.00015,
        output_cost_per_1k_tokens=0.0006,
        context_window=128000,
        max_output_tokens=16384
    ),
    ModelType.GPT_4_TURBO: ModelPricing(
        input_cost_per_1k_tokens=0.01,
        output_cost_per_1k_tokens=0.03,
        context_window=128000,
        max_output_tokens=4096
    ),
    ModelType.GPT_4: ModelPricing(
        input_cost_per_1k_tokens=0.03,
        output_cost_per_1k_tokens=0.06,
        context_window=8192,
        max_output_tokens=4096
    ),
    ModelType.GPT_35_TURBO: ModelPricing(
        input_cost_per_1k_tokens=0.0015,
        output_cost_per_1k_tokens=0.002,
        context_window=16385,
        max_output_tokens=4096
    ),
    ModelType.TEXT_EMBEDDING_3_LARGE: ModelPricing(
        input_cost_per_1k_tokens=0.00013,
        output_cost_per_1k_tokens=0.0,
        context_window=8191,
        max_output_tokens=0
    ),
    ModelType.TEXT_EMBEDDING_3_SMALL: ModelPricing(
        input_cost_per_1k_tokens=0.00002,
        output_cost_per_1k_tokens=0.0,
        context_window=8191,
        max_output_tokens=0
    ),
}


# Predefined model configurations
AVAILABLE_MODELS: Dict[str, OpenAIModel] = {
    ModelType.GPT_4O: OpenAIModel(
        name="GPT-4o",
        model_type=ModelType.GPT_4O,
        pricing=MODEL_PRICING[ModelType.GPT_4O],
        capabilities=["text_generation", "function_calling", "structured_output", "vision"],
        recommended_for=["creative_writing", "complex_reasoning", "multimodal_tasks"]
    ),
    ModelType.GPT_4: OpenAIModel(
        name="GPT-4",
        model_type=ModelType.GPT_4,
        pricing=MODEL_PRICING[ModelType.GPT_4],
        capabilities=["text_generation", "function_calling", "structured_output"],
        recommended_for=["complex_reasoning", "analysis", "planning"]
    ),
    ModelType.GPT_35_TURBO: OpenAIModel(
        name="GPT-3.5 Turbo",
        model_type=ModelType.GPT_35_TURBO,
        pricing=MODEL_PRICING[ModelType.GPT_35_TURBO],
        capabilities=["text_generation", "function_calling"],
        recommended_for=["general_tasks", "summarization", "research"]
    ),
}


def get_model_pricing(model_name: str) -> Optional[ModelPricing]:
    """Get pricing information for a model."""
    return MODEL_PRICING.get(model_name)


def calculate_cost(model_name: str, token_usage: TokenUsage) -> float:
    """Calculate the cost for a given token usage."""
    pricing = get_model_pricing(model_name)
    if not pricing:
        return 0.0
    
    input_cost = (token_usage.prompt_tokens / 1000) * pricing.input_cost_per_1k_tokens
    output_cost = (token_usage.completion_tokens / 1000) * pricing.output_cost_per_1k_tokens
    
    return input_cost + output_cost


def get_recommended_model(task_type: str) -> str:
    """Get recommended model for a task type."""
    task_model_mapping = {
        "creative_writing": ModelType.GPT_4O,
        "complex_reasoning": ModelType.GPT_4,
        "analysis": ModelType.GPT_4,
        "planning": ModelType.GPT_4,
        "research": ModelType.GPT_35_TURBO,
        "summarization": ModelType.GPT_35_TURBO,
        "general": ModelType.GPT_4O_MINI,
    }
    
    return task_model_mapping.get(task_type, ModelType.GPT_4O)