"""
Configuration management for the Book Retriever Component
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BookRetrieverConfig(BaseSettings):
    """Configuration settings for the book retriever component."""
    
    # Redis settings (for retrieving books from queue)
    redis_url: str = Field(
        default="redis://localhost:6380/0",
        validation_alias="REDIS_URL",
        description="Redis connection URL"
    )
    queue_name: str = Field(
        default="book_writing_queue",
        validation_alias="BOOK_QUEUE_NAME",
        description="Redis queue name for book writing jobs"
    )
    
    # MongoDB settings (for book validation and updates)
    mongodb_url: str = Field(
        default="mongodb://localhost:27017",
        validation_alias="MONGODB_URL",
        description="MongoDB connection URL"
    )
    database_name: str = Field(
        default="musequill",
        validation_alias="MONGODB_DATABASE",
        description="Database name"
    )
    database_username: str = Field(
        default="",
        validation_alias="MONGODB_USERNAME",
        description="Database username"
    )
    database_password: str = Field(
        default="",
        validation_alias="MONGODB_PASSWORD",
        description="Database password"
    )
    auth_database: str = Field(
        default="admin",
        validation_alias="MONGODB_AUTH_DATABASE",
        description="MongoDB authentication database"
    )
    books_collection: str = Field(
        default="books",
        validation_alias="MONGODB_BOOKS_COLLECTION",
        description="Books collection name"
    )
    
    # Processing settings
    poll_interval: int = Field(
        default=5,
        validation_alias="RETRIEVER_POLL_INTERVAL",
        description="Polling interval in seconds for checking queue",
        ge=1,
        le=300
    )
    max_concurrent_orchestrations: int = Field(
        default=3,
        validation_alias="MAX_CONCURRENT_ORCHESTRATIONS",
        description="Maximum orchestrations to run concurrently",
        ge=1,
        le=10
    )
    orchestration_timeout: int = Field(
        default=3600,
        validation_alias="ORCHESTRATION_TIMEOUT",
        description="Timeout for orchestration completion in seconds",
        ge=300,
        le=14400  # 4 hours max
    )
    
    # Book validation settings
    required_fields: list = Field(
        default=["_id", "title", "description", "genre_info"],
        description="Required fields for book validation"
    )
    min_title_length: int = Field(
        default=3,
        validation_alias="MIN_TITLE_LENGTH",
        description="Minimum title length for validation",
        ge=1,
        le=200
    )
    min_description_length: int = Field(
        default=10,
        validation_alias="MIN_DESCRIPTION_LENGTH",
        description="Minimum description length for validation",
        ge=5,
        le=1000
    )
    
    # Default values for missing fields
    default_genre: str = Field(
        default="General Fiction",
        validation_alias="DEFAULT_GENRE",
        description="Default genre if not specified"
    )
    default_target_length: int = Field(
        default=50000,
        validation_alias="DEFAULT_TARGET_LENGTH",
        description="Default target word count for books",
        ge=1000,
        le=500000
    )
    default_chapter_count: int = Field(
        default=10,
        validation_alias="DEFAULT_CHAPTER_COUNT",
        description="Default number of chapters if not specified",
        ge=1,
        le=100
    )
    
    # Orchestration settings
    langgraph_checkpointer_type: str = Field(
        default="memory",
        validation_alias="LANGGRAPH_CHECKPOINTER_TYPE",
        description="Type of checkpointer to use (memory/redis/postgres)"
    )
    redis_checkpointer_url: str = Field(
        default="",
        validation_alias="REDIS_CHECKPOINTER_URL",
        description="Redis URL for LangGraph checkpointer (if using redis)"
    )
    
    # Error handling settings
    max_retries: int = Field(
        default=3,
        validation_alias="MAX_BOOK_RETRIES",
        description="Maximum retries for failed book processing",
        ge=0,
        le=10
    )
    retry_delay: int = Field(
        default=60,
        validation_alias="RETRY_DELAY_SECONDS",
        description="Delay between retries in seconds",
        ge=1,
        le=3600
    )
    dead_letter_queue: str = Field(
        default="book_writing_failed",
        validation_alias="DEAD_LETTER_QUEUE",
        description="Queue for books that failed processing"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )