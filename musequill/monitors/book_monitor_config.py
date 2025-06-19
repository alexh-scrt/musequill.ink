"""
Configuration management for the Book Pipeline Monitor
"""

from pydantic import Field
from pydantic_settings import BaseSettings

class BookMonitorConfig(BaseSettings):
    """Configuration settings for the book pipeline monitor."""
    
    # MongoDB settings
    mongodb_url: str = Field(
        default="mongodb://localhost:27017",
        env="MONGODB_URL",
        description="MongoDB connection URL"
    )
    database_name: str = Field(
        default="musequill",
        env="MONGODB_DATABASE",
        description="Database name"
    )
    collection_name: str = Field(
        default="books",
        env="MONGODB_COLLECTION",
        description="Books collection name"
    )
    database_username: str = Field(
        default="",
        env="MONGODB_USERNAME",
        description="Database username"
    )
    database_password: str = Field(
        default="",
        env="MONGODB_PASSWORD",
        description="Database password"
    )

    # Redis settings
    redis_url: str = Field(
        default="redis://localhost:6380/0",
        env="REDIS_URL",
        description="Redis connection URL"
    )
    queue_name: str = Field(
        default="book_writing_queue",
        env="BOOK_QUEUE_NAME",
        description="Redis queue name for book writing jobs"
    )
    
    # Monitor settings
    poll_interval: int = Field(
        default=10,
        env="BOOK_POLL_INTERVAL",
        description="Polling interval in seconds",
        ge=1,
        le=3600
    )
    max_books_per_batch: int = Field(
        default=5,
        env="MAX_BOOKS_PER_BATCH",
        description="Maximum books to process in one batch",
        ge=1,
        le=1000
    )
    model_config = {"extra": "ignore", "env_file": ".env", "env_file_encoding": "utf-8"}

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"