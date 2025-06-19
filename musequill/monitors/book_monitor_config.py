"""
Configuration management for the Book Pipeline Monitor
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class BookMonitorConfig(BaseSettings):
    """Configuration settings for the book pipeline monitor."""
    
    # MongoDB settings
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
    collection_name: str = Field(
        default="books",
        validation_alias="MONGODB_COLLECTION",
        description="Books collection name"
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

    # Redis settings
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
    
    # Monitor settings
    poll_interval: int = Field(
        default=10,
        validation_alias="BOOK_POLL_INTERVAL",
        description="Polling interval in seconds",
        ge=1,
        le=3600
    )
    max_books_per_batch: int = Field(
        default=5,
        validation_alias="MAX_BOOKS_PER_BATCH",
        description="Maximum books to process in one batch",
        ge=1,
        le=1000
    )
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )