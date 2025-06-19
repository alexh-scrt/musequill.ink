"""
Configuration management for the Research Planner Agent
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ResearchPlannerConfig(BaseSettings):
    """Configuration settings for the research planner agent."""
    
    # LLM settings
    openai_api_key: str = Field(
        default="",
        validation_alias="OPENAI_API_KEY",
        description="OpenAI API key for LLM operations"
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        validation_alias="RESEARCH_PLANNER_LLM_MODEL",
        description="LLM model to use for research planning"
    )
    llm_temperature: float = Field(
        default=0.3,
        validation_alias="RESEARCH_PLANNER_LLM_TEMPERATURE",
        description="Temperature for LLM responses (higher for more creativity)",
        ge=0.0,
        le=2.0
    )
    llm_max_tokens: int = Field(
        default=4000,
        validation_alias="RESEARCH_PLANNER_MAX_TOKENS",
        description="Maximum tokens for LLM responses",
        ge=500,
        le=16000
    )
    
    # Research planning settings
    max_research_queries: int = Field(
        default=20,
        validation_alias="MAX_RESEARCH_QUERIES",
        description="Maximum number of research queries to generate",
        ge=5,
        le=100
    )
    min_research_queries: int = Field(
        default=8,
        validation_alias="MIN_RESEARCH_QUERIES",
        description="Minimum number of research queries to generate",
        ge=3,
        le=50
    )
    
    # Query categorization
    query_categories: list = Field(
        default=[
            "background_information",
            "technical_details", 
            "expert_opinions",
            "case_studies",
            "current_trends",
            "historical_context",
            "statistical_data",
            "examples_and_illustrations"
        ],
        description="Categories for organizing research queries"
    )
    
    # Query priority settings
    high_priority_categories: list = Field(
        default=["background_information", "technical_details", "expert_opinions"],
        description="Query categories that should be marked as high priority"
    )
    medium_priority_categories: list = Field(
        default=["case_studies", "current_trends", "statistical_data"],
        description="Query categories that should be marked as medium priority"
    )
    low_priority_categories: list = Field(
        default=["historical_context", "examples_and_illustrations"],
        description="Query categories that should be marked as low priority"
    )
    
    # Content analysis settings
    min_chapter_analysis_length: int = Field(
        default=10,
        validation_alias="MIN_CHAPTER_ANALYSIS_LENGTH",
        description="Minimum characters in chapter description for detailed analysis",
        ge=5,
        le=100
    )
    
    # Research strategy settings
    strategy_depth_levels: list = Field(
        default=["surface", "intermediate", "deep"],
        description="Available depth levels for research strategy"
    )
    default_research_depth: str = Field(
        default="intermediate",
        validation_alias="DEFAULT_RESEARCH_DEPTH",
        description="Default research depth level"
    )
    
    # Quality control
    query_uniqueness_threshold: float = Field(
        default=0.8,
        validation_alias="QUERY_UNIQUENESS_THRESHOLD",
        description="Similarity threshold for query deduplication (0-1)",
        ge=0.5,
        le=1.0
    )
    enable_query_validation: bool = Field(
        default=True,
        validation_alias="ENABLE_QUERY_VALIDATION",
        description="Enable validation of generated queries"
    )
    
    # Retry and error handling
    max_generation_retries: int = Field(
        default=3,
        validation_alias="MAX_GENERATION_RETRIES",
        description="Maximum retries for query generation",
        ge=1,
        le=10
    )
    
    # Output formatting
    include_research_rationale: bool = Field(
        default=True,
        validation_alias="INCLUDE_RESEARCH_RATIONALE",
        description="Include rationale for each research query"
    )
    include_estimated_sources: bool = Field(
        default=True,
        validation_alias="INCLUDE_ESTIMATED_SOURCES",
        description="Include estimated number of sources per query"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )