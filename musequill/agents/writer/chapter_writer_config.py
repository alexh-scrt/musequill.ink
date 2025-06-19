"""
Configuration management for the Chapter Writer Agent
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChapterWriterConfig(BaseSettings):
    """Configuration settings for the chapter writer agent."""
    
    # LLM settings for chapter writing
    openai_api_key: str = Field(
        default="",
        validation_alias="OPENAI_API_KEY",
        description="OpenAI API key for LLM operations"
    )
    llm_model: str = Field(
        default="gpt-4o",
        validation_alias="CHAPTER_WRITER_LLM_MODEL",
        description="LLM model to use for chapter writing"
    )
    llm_temperature: float = Field(
        default=0.7,
        validation_alias="CHAPTER_WRITER_LLM_TEMPERATURE",
        description="Temperature for LLM responses (higher for creativity)",
        ge=0.0,
        le=2.0
    )
    llm_max_tokens: int = Field(
        default=8000,
        validation_alias="CHAPTER_WRITER_MAX_TOKENS",
        description="Maximum tokens for LLM responses",
        ge=2000,
        le=16000
    )
    
    # Chroma Vector Store settings (for research retrieval)
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
    chroma_tenant: str = Field(
        default="default_tenant",
        validation_alias="CHROMA_TENANT",
        description="Chroma tenant name"
    )
    chroma_database: str = Field(
        default="default_database",
        validation_alias="CHROMA_DATABASE",
        description="Chroma database name"
    )
    
    # OpenAI Embeddings for research retrieval
    embedding_model: str = Field(
        default="text-embedding-3-small",
        validation_alias="EMBEDDING_MODEL",
        description="OpenAI embedding model for research queries"
    )
    embedding_dimensions: int = Field(
        default=1536,
        validation_alias="EMBEDDING_DIMENSIONS",
        description="Embedding vector dimensions",
        ge=256,
        le=3072
    )
    
    # Research retrieval settings
    max_research_chunks_per_chapter: int = Field(
        default=15,
        validation_alias="MAX_RESEARCH_CHUNKS_PER_CHAPTER",
        description="Maximum research chunks to retrieve per chapter",
        ge=5,
        le=50
    )
    research_relevance_threshold: float = Field(
        default=0.7,
        validation_alias="RESEARCH_RELEVANCE_THRESHOLD",
        description="Minimum relevance score for research chunks",
        ge=0.0,
        le=1.0
    )
    enable_research_diversity: bool = Field(
        default=True,
        validation_alias="ENABLE_RESEARCH_DIVERSITY",
        description="Ensure diversity in research chunk selection"
    )
    research_query_expansion: bool = Field(
        default=True,
        validation_alias="RESEARCH_QUERY_EXPANSION",
        description="Expand chapter topics for broader research retrieval"
    )
    
    # Writing settings
    enable_iterative_writing: bool = Field(
        default=True,
        validation_alias="ENABLE_ITERATIVE_WRITING",
        description="Enable iterative chapter writing and refinement"
    )
    writing_approach: str = Field(
        default="research_driven",
        validation_alias="WRITING_APPROACH",
        description="Writing approach (research_driven/creative/hybrid)"
    )
    enforce_style_consistency: bool = Field(
        default=True,
        validation_alias="ENFORCE_STYLE_CONSISTENCY",
        description="Enforce style guide consistency"
    )
    include_chapter_connections: bool = Field(
        default=True,
        validation_alias="INCLUDE_CHAPTER_CONNECTIONS",
        description="Include connections to previous and next chapters"
    )
    
    # Content organization settings
    enforce_target_word_count: bool = Field(
        default=True,
        validation_alias="ENFORCE_TARGET_WORD_COUNT",
        description="Attempt to meet target word count"
    )
    word_count_tolerance: float = Field(
        default=0.2,
        validation_alias="WORD_COUNT_TOLERANCE",
        description="Acceptable word count variance (20% = 0.2)",
        ge=0.0,
        le=0.5
    )
    min_words_per_chapter: int = Field(
        default=800,
        validation_alias="MIN_WORDS_PER_CHAPTER",
        description="Minimum words per chapter",
        ge=200,
        le=5000
    )
    max_words_per_chapter: int = Field(
        default=8000,
        validation_alias="MAX_WORDS_PER_CHAPTER",
        description="Maximum words per chapter",
        ge=1000,
        le=20000
    )
    
    # Quality control settings
    enable_quality_checks: bool = Field(
        default=True,
        validation_alias="ENABLE_QUALITY_CHECKS",
        description="Enable automated quality checks"
    )
    min_quality_score: float = Field(
        default=0.7,
        validation_alias="MIN_QUALITY_SCORE",
        description="Minimum quality score to accept chapter",
        ge=0.0,
        le=1.0
    )
    quality_metrics: list = Field(
        default=[
            "coherence",
            "research_integration",
            "style_consistency",
            "argument_structure",
            "readability"
        ],
        description="Quality metrics to evaluate"
    )
    
    # Progress tracking settings
    enable_progress_tracking: bool = Field(
        default=True,
        validation_alias="ENABLE_PROGRESS_TRACKING",
        description="Enable detailed progress tracking"
    )
    track_research_usage: bool = Field(
        default=True,
        validation_alias="TRACK_RESEARCH_USAGE",
        description="Track which research chunks are used"
    )
    generate_writing_analytics: bool = Field(
        default=True,
        validation_alias="GENERATE_WRITING_ANALYTICS",
        description="Generate analytics on writing process"
    )
    
    # Error handling and retry settings
    max_retry_attempts: int = Field(
        default=3,
        validation_alias="MAX_RETRY_ATTEMPTS",
        description="Maximum retry attempts for failed chapters",
        ge=1,
        le=10
    )
    retry_with_reduced_context: bool = Field(
        default=True,
        validation_alias="RETRY_WITH_REDUCED_CONTEXT",
        description="Retry with reduced context on failure"
    )
    fallback_to_basic_writing: bool = Field(
        default=True,
        validation_alias="FALLBACK_TO_BASIC_WRITING",
        description="Fallback to basic writing approach on repeated failures"
    )
    
    # Performance optimization settings
    max_writing_time_per_chapter: int = Field(
        default=300,
        validation_alias="MAX_WRITING_TIME_PER_CHAPTER",
        description="Maximum time per chapter in seconds",
        ge=60,
        le=1800
    )
    enable_async_processing: bool = Field(
        default=True,
        validation_alias="ENABLE_ASYNC_PROCESSING",
        description="Enable asynchronous processing where possible"
    )
    cache_research_retrievals: bool = Field(
        default=True,
        validation_alias="CACHE_RESEARCH_RETRIEVALS",
        description="Cache research retrievals for efficiency"
    )
    
    # Output formatting settings
    include_chapter_metadata: bool = Field(
        default=True,
        validation_alias="INCLUDE_CHAPTER_METADATA",
        description="Include metadata with chapter content"
    )
    generate_chapter_summaries: bool = Field(
        default=True,
        validation_alias="GENERATE_CHAPTER_SUMMARIES",
        description="Generate brief summaries for each chapter"
    )
    include_source_citations: bool = Field(
        default=True,
        validation_alias="INCLUDE_SOURCE_CITATIONS",
        description="Include research source citations"
    )
    citation_style: str = Field(
        default="academic",
        validation_alias="CITATION_STYLE",
        description="Citation style (academic/informal/numbered)"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )