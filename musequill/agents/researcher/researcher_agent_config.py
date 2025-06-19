"""
Configuration management for the Researcher Agent
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ResearcherConfig(BaseSettings):
    """Configuration settings for the researcher agent."""
    
    # Tavily API settings
    tavily_api_key: str = Field(
        default="",
        validation_alias="TAVILY_API_KEY",
        description="Tavily API key for web search"
    )
    tavily_search_depth: str = Field(
        default="advanced",
        validation_alias="TAVILY_SEARCH_DEPTH",
        description="Depth of Tavily search (basic/advanced)"
    )
    tavily_max_results: int = Field(
        default=10,
        validation_alias="TAVILY_MAX_RESULTS",
        description="Maximum search results per query",
        ge=1,
        le=20
    )
    tavily_include_answer: bool = Field(
        default=True,
        validation_alias="TAVILY_INCLUDE_ANSWER",
        description="Include Tavily's answer summary"
    )
    tavily_include_raw_content: bool = Field(
        default=True,
        validation_alias="TAVILY_INCLUDE_RAW_CONTENT", 
        description="Include raw content from sources"
    )
    tavily_include_images: bool = Field(
        default=False,
        validation_alias="TAVILY_INCLUDE_IMAGES",
        description="Include image results from Tavily"
    )
    
    # Chroma Vector Store settings
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
        description="Chroma collection name for storing research materials"
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
    
    # OpenAI Embeddings settings
    openai_api_key: str = Field(
        default="",
        validation_alias="OPENAI_API_KEY",
        description="OpenAI API key for embeddings"
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        validation_alias="EMBEDDING_MODEL",
        description="OpenAI embedding model"
    )
    embedding_dimensions: int = Field(
        default=1536,
        validation_alias="EMBEDDING_DIMENSIONS",
        description="Embedding vector dimensions",
        ge=256,
        le=3072
    )
    
    # Text Processing settings
    chunk_size: int = Field(
        default=1000,
        validation_alias="RESEARCH_CHUNK_SIZE",
        description="Text chunk size for embedding",
        ge=100,
        le=8000
    )
    chunk_overlap: int = Field(
        default=200,
        validation_alias="RESEARCH_CHUNK_OVERLAP",
        description="Overlap between text chunks",
        ge=0,
        le=500
    )
    min_chunk_size: int = Field(
        default=100,
        validation_alias="MIN_CHUNK_SIZE",
        description="Minimum chunk size to store",
        ge=50,
        le=500
    )
    max_content_length: int = Field(
        default=50000,
        validation_alias="MAX_CONTENT_LENGTH",
        description="Maximum content length to process per result",
        ge=1000,
        le=200000
    )
    
    # Processing settings
    max_concurrent_queries: int = Field(
        default=3,
        validation_alias="MAX_CONCURRENT_RESEARCH_QUERIES",
        description="Maximum concurrent research queries",
        ge=1,
        le=10
    )
    query_retry_attempts: int = Field(
        default=3,
        validation_alias="QUERY_RETRY_ATTEMPTS",
        description="Number of retry attempts for failed queries",
        ge=1,
        le=5
    )
    retry_delay_seconds: int = Field(
        default=5,
        validation_alias="RETRY_DELAY_SECONDS",
        description="Delay between retry attempts",
        ge=1,
        le=60
    )
    rate_limit_delay: float = Field(
        default=1.0,
        validation_alias="RATE_LIMIT_DELAY",
        description="Delay between API calls to respect rate limits",
        ge=0.1,
        le=10.0
    )
    
    # Content Quality settings
    min_content_quality_score: float = Field(
        default=0.3,
        validation_alias="MIN_CONTENT_QUALITY_SCORE",
        description="Minimum quality score for content inclusion",
        ge=0.0,
        le=1.0
    )
    enable_content_filtering: bool = Field(
        default=True,
        validation_alias="ENABLE_CONTENT_FILTERING",
        description="Enable content quality filtering"
    )
    filter_duplicate_content: bool = Field(
        default=True,
        validation_alias="FILTER_DUPLICATE_CONTENT",
        description="Filter out duplicate content"
    )
    content_similarity_threshold: float = Field(
        default=0.85,
        validation_alias="CONTENT_SIMILARITY_THRESHOLD",
        description="Threshold for considering content duplicate",
        ge=0.5,
        le=1.0
    )
    
    # Source Quality settings
    trusted_domains: list = Field(
        default=[
            "edu", "gov", "org", "wikipedia.org", "scholar.google.com",
            "nature.com", "science.org", "ieee.org", "acm.org", 
            "reuters.com", "bbc.com", "nytimes.com", "washingtonpost.com"
        ],
        description="List of trusted domain patterns"
    )
    blocked_domains: list = Field(
        default=[
            "example.com", "test.com", "spam.com"
        ],
        description="List of blocked domain patterns"
    )
    min_source_score: float = Field(
        default=0.1,
        validation_alias="MIN_SOURCE_SCORE",
        description="Minimum Tavily source score to include",
        ge=0.0,
        le=1.0
    )
    
    # Storage settings
    batch_size: int = Field(
        default=50,
        validation_alias="CHROMA_BATCH_SIZE",
        description="Batch size for Chroma insertions",
        ge=1,
        le=1000
    )
    enable_metadata_indexing: bool = Field(
        default=True,
        validation_alias="ENABLE_METADATA_INDEXING",
        description="Enable metadata indexing in Chroma"
    )
    
    # Monitoring and logging
    log_search_results: bool = Field(
        default=True,
        validation_alias="LOG_SEARCH_RESULTS",
        description="Log detailed search results"
    )
    log_chunk_details: bool = Field(
        default=False,
        validation_alias="LOG_CHUNK_DETAILS",
        description="Log detailed chunk processing information"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )