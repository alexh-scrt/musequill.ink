"""
Configuration management for the Research Validator Agent
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ResearchValidatorConfig(BaseSettings):
    """Configuration settings for the research validator agent."""
    
    # LLM settings for validation analysis
    openai_api_key: str = Field(
        default="",
        validation_alias="OPENAI_API_KEY",
        description="OpenAI API key for LLM operations"
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        validation_alias="RESEARCH_VALIDATOR_LLM_MODEL",
        description="LLM model to use for research validation"
    )
    llm_temperature: float = Field(
        default=0.2,
        validation_alias="RESEARCH_VALIDATOR_LLM_TEMPERATURE",
        description="Temperature for LLM responses (lower for more analytical)",
        ge=0.0,
        le=2.0
    )
    llm_max_tokens: int = Field(
        default=3000,
        validation_alias="RESEARCH_VALIDATOR_MAX_TOKENS",
        description="Maximum tokens for LLM responses",
        ge=500,
        le=16000
    )
    
    # Chroma Vector Store settings (for analyzing stored research)
    chroma_host: str = Field(
        default="localhost",
        validation_alias="CHROMA_HOST",
        description="Chroma database host"
    )
    chroma_port: int = Field(
        default=8000,
        validation_alias="CHROMA_PORT",
        description="Chroma database port"
    )
    chroma_collection_name: str = Field(
        default="book_research",
        validation_alias="CHROMA_COLLECTION_NAME",
        description="Chroma collection name for research materials"
    )
    
    # Validation criteria
    min_chunks_per_query: int = Field(
        default=3,
        validation_alias="MIN_CHUNKS_PER_QUERY",
        description="Minimum chunks required per research query",
        ge=1,
        le=20
    )
    min_total_chunks: int = Field(
        default=15,
        validation_alias="MIN_TOTAL_CHUNKS",
        description="Minimum total chunks required for book",
        ge=5,
        le=200
    )
    min_unique_sources: int = Field(
        default=5,
        validation_alias="MIN_UNIQUE_SOURCES",
        description="Minimum unique sources required",
        ge=2,
        le=50
    )
    min_source_diversity: float = Field(
        default=0.6,
        validation_alias="MIN_SOURCE_DIVERSITY",
        description="Minimum source diversity ratio (unique domains / total sources)",
        ge=0.1,
        le=1.0
    )
    
    # Quality thresholds
    min_avg_quality_score: float = Field(
        default=0.5,
        validation_alias="MIN_AVG_QUALITY_SCORE",
        description="Minimum average quality score for research chunks",
        ge=0.1,
        le=1.0
    )
    min_high_quality_percentage: float = Field(
        default=0.3,
        validation_alias="MIN_HIGH_QUALITY_PERCENTAGE",
        description="Minimum percentage of high-quality chunks required",
        ge=0.1,
        le=1.0
    )
    high_quality_threshold: float = Field(
        default=0.7,
        validation_alias="HIGH_QUALITY_THRESHOLD",
        description="Quality score threshold for high-quality classification",
        ge=0.5,
        le=1.0
    )
    
    # Coverage analysis
    required_query_categories: list = Field(
        default=[
            "background_information",
            "technical_details",
            "expert_opinions"
        ],
        description="Query categories that must have research results"
    )
    min_coverage_per_category: int = Field(
        default=2,
        validation_alias="MIN_COVERAGE_PER_CATEGORY",
        description="Minimum chunks required per required category",
        ge=1,
        le=10
    )
    
    # Gap analysis settings
    enable_gap_analysis: bool = Field(
        default=True,
        validation_alias="ENABLE_GAP_ANALYSIS",
        description="Enable LLM-based gap analysis"
    )
    gap_analysis_sample_size: int = Field(
        default=20,
        validation_alias="GAP_ANALYSIS_SAMPLE_SIZE",
        description="Number of chunks to sample for gap analysis",
        ge=5,
        le=100
    )
    max_additional_queries: int = Field(
        default=5,
        validation_alias="MAX_ADDITIONAL_QUERIES",
        description="Maximum additional queries to generate",
        ge=1,
        le=20
    )
    
    # Content analysis
    analyze_content_distribution: bool = Field(
        default=True,
        validation_alias="ANALYZE_CONTENT_DISTRIBUTION",
        description="Analyze distribution of content across topics"
    )
    min_content_per_chapter: int = Field(
        default=2,
        validation_alias="MIN_CONTENT_PER_CHAPTER",
        description="Minimum research chunks recommended per chapter",
        ge=1,
        le=20
    )
    
    # Validation strictness levels
    validation_strictness: str = Field(
        default="medium",
        validation_alias="VALIDATION_STRICTNESS",
        description="Validation strictness level (low/medium/high)"
    )
    
    # Retry and fallback settings
    enable_automatic_query_generation: bool = Field(
        default=True,
        validation_alias="ENABLE_AUTO_QUERY_GENERATION",
        description="Enable automatic generation of additional queries"
    )
    fallback_validation_mode: bool = Field(
        default=True,
        validation_alias="FALLBACK_VALIDATION_MODE",
        description="Use fallback validation when LLM analysis fails"
    )
    
    # Performance settings
    max_analysis_time: int = Field(
        default=120,
        validation_alias="MAX_ANALYSIS_TIME",
        description="Maximum time for validation analysis in seconds",
        ge=30,
        le=600
    )
    enable_parallel_analysis: bool = Field(
        default=True,
        validation_alias="ENABLE_PARALLEL_ANALYSIS",
        description="Enable parallel analysis of different validation aspects"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )