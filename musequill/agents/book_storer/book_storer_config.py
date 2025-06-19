"""
Configuration management for the Book Storer Agent
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BookStorerConfig(BaseSettings):
    """Configuration settings for the book storer agent."""
    
    # MongoDB settings
    mongodb_host: str = Field(
        default="localhost",
        validation_alias="MONGODB_HOST",
        description="MongoDB host"
    )
    mongodb_port: int = Field(
        default=27017,
        validation_alias="MONGODB_PORT",
        description="MongoDB port"
    )
    mongodb_database: str = Field(
        default="musequill",
        validation_alias="MONGODB_DATABASE",
        description="MongoDB database name"
    )
    mongodb_username: str = Field(
        default="",
        validation_alias="MONGODB_USERNAME",
        description="MongoDB username"
    )
    mongodb_password: str = Field(
        default="",
        validation_alias="MONGODB_PASSWORD",
        description="MongoDB password"
    )
    mongodb_auth_database: str = Field(
        default="admin",
        validation_alias="MONGODB_AUTH_DATABASE",
        description="MongoDB authentication database"
    )
    
    # Collection settings
    books_collection_name: str = Field(
        default="books",
        validation_alias="BOOKS_COLLECTION_NAME",
        description="MongoDB collection name for books"
    )
    book_files_collection_name: str = Field(
        default="book_files",
        validation_alias="BOOK_FILES_COLLECTION_NAME",
        description="MongoDB collection name for book files"
    )
    book_metadata_collection_name: str = Field(
        default="book_metadata",
        validation_alias="BOOK_METADATA_COLLECTION_NAME",
        description="MongoDB collection name for book metadata"
    )
    
    # Connection pool settings
    min_pool_size: int = Field(
        default=5,
        validation_alias="MONGODB_MIN_POOL_SIZE",
        description="Minimum MongoDB connection pool size",
        ge=1,
        le=100
    )
    max_pool_size: int = Field(
        default=50,
        validation_alias="MONGODB_MAX_POOL_SIZE",
        description="Maximum MongoDB connection pool size",
        ge=5,
        le=1000
    )
    max_idle_time_ms: int = Field(
        default=30000,
        validation_alias="MONGODB_MAX_IDLE_TIME_MS",
        description="Maximum idle time for MongoDB connections in milliseconds",
        ge=5000,
        le=300000
    )
    wait_queue_timeout_ms: int = Field(
        default=5000,
        validation_alias="MONGODB_WAIT_QUEUE_TIMEOUT_MS",
        description="Wait queue timeout for MongoDB connections in milliseconds",
        ge=1000,
        le=60000
    )
    
    # Storage optimization settings
    enable_compression: bool = Field(
        default=True,
        validation_alias="ENABLE_STORAGE_COMPRESSION",
        description="Enable compression for stored book content"
    )
    compression_level: int = Field(
        default=6,
        validation_alias="COMPRESSION_LEVEL",
        description="Compression level (1-9, higher = better compression)",
        ge=1,
        le=9
    )
    max_document_size_mb: int = Field(
        default=16,
        validation_alias="MAX_DOCUMENT_SIZE_MB",
        description="Maximum document size in MB for MongoDB",
        ge=1,
        le=16
    )
    
    # File storage settings
    store_file_attachments: bool = Field(
        default=True,
        validation_alias="STORE_FILE_ATTACHMENTS",
        description="Store generated book files as binary attachments"
    )
    max_file_size_mb: int = Field(
        default=50,
        validation_alias="MAX_FILE_SIZE_MB",
        description="Maximum file size in MB for attachments",
        ge=1,
        le=200
    )
    supported_file_formats: list = Field(
        default=["pdf", "epub", "docx", "html", "markdown"],
        validation_alias="SUPPORTED_FILE_FORMATS",
        description="List of supported file formats for storage"
    )
    
    # Metadata indexing settings
    enable_full_text_search: bool = Field(
        default=True,
        validation_alias="ENABLE_FULL_TEXT_SEARCH",
        description="Enable full-text search indexing"
    )
    index_chapter_content: bool = Field(
        default=True,
        validation_alias="INDEX_CHAPTER_CONTENT",
        description="Index individual chapter content for search"
    )
    create_content_embeddings: bool = Field(
        default=False,
        validation_alias="CREATE_CONTENT_EMBEDDINGS",
        description="Create vector embeddings for content search"
    )
    
    # Backup and versioning settings
    enable_versioning: bool = Field(
        default=True,
        validation_alias="ENABLE_BOOK_VERSIONING",
        description="Enable book version tracking"
    )
    max_versions_retained: int = Field(
        default=10,
        validation_alias="MAX_VERSIONS_RETAINED",
        description="Maximum number of book versions to retain",
        ge=1,
        le=100
    )
    auto_backup_enabled: bool = Field(
        default=True,
        validation_alias="AUTO_BACKUP_ENABLED",
        description="Enable automatic backup of stored books"
    )
    backup_retention_days: int = Field(
        default=30,
        validation_alias="BACKUP_RETENTION_DAYS",
        description="Number of days to retain backups",
        ge=1,
        le=365
    )
    
    # Performance and monitoring settings
    enable_performance_metrics: bool = Field(
        default=True,
        validation_alias="ENABLE_PERFORMANCE_METRICS",
        description="Enable performance metrics collection"
    )
    slow_query_threshold_ms: int = Field(
        default=1000,
        validation_alias="SLOW_QUERY_THRESHOLD_MS",
        description="Threshold for logging slow queries in milliseconds",
        ge=100,
        le=10000
    )
    connection_timeout_ms: int = Field(
        default=30000,
        validation_alias="CONNECTION_TIMEOUT_MS",
        description="MongoDB connection timeout in milliseconds",
        ge=5000,
        le=120000
    )
    operation_timeout_ms: int = Field(
        default=60000,
        validation_alias="OPERATION_TIMEOUT_MS",
        description="MongoDB operation timeout in milliseconds",
        ge=10000,
        le=300000
    )
    
    # Validation and quality assurance settings
    validate_before_storage: bool = Field(
        default=True,
        validation_alias="VALIDATE_BEFORE_STORAGE",
        description="Validate book data before storage"
    )
    require_quality_score: bool = Field(
        default=True,
        validation_alias="REQUIRE_QUALITY_SCORE",
        description="Require minimum quality score for storage"
    )
    min_quality_score: float = Field(
        default=0.7,
        validation_alias="MIN_QUALITY_SCORE",
        description="Minimum quality score required for storage",
        ge=0.0,
        le=1.0
    )
    validate_file_integrity: bool = Field(
        default=True,
        validation_alias="VALIDATE_FILE_INTEGRITY",
        description="Validate file integrity during storage"
    )
    
    # Error handling and retry settings
    enable_retry_on_failure: bool = Field(
        default=True,
        validation_alias="ENABLE_RETRY_ON_FAILURE",
        description="Enable retry on storage failures"
    )
    max_retry_attempts: int = Field(
        default=3,
        validation_alias="MAX_RETRY_ATTEMPTS",
        description="Maximum retry attempts for failed operations",
        ge=1,
        le=10
    )
    retry_delay_seconds: float = Field(
        default=1.0,
        validation_alias="RETRY_DELAY_SECONDS",
        description="Delay between retry attempts in seconds",
        ge=0.1,
        le=30.0
    )
    exponential_backoff: bool = Field(
        default=True,
        validation_alias="EXPONENTIAL_BACKOFF",
        description="Use exponential backoff for retries"
    )
    
    # Cleanup and maintenance settings
    enable_auto_cleanup: bool = Field(
        default=True,
        validation_alias="ENABLE_AUTO_CLEANUP",
        description="Enable automatic cleanup of old/failed books"
    )
    cleanup_failed_books_days: int = Field(
        default=7,
        validation_alias="CLEANUP_FAILED_BOOKS_DAYS",
        description="Days after which failed books are cleaned up",
        ge=1,
        le=365
    )
    cleanup_temp_data_hours: int = Field(
        default=24,
        validation_alias="CLEANUP_TEMP_DATA_HOURS",
        description="Hours after which temporary data is cleaned up",
        ge=1,
        le=168
    )
    
    @property
    def connection_string(self) -> str:
        """Generate MongoDB connection string."""
        if self.mongodb_username and self.mongodb_password:
            return (
                f"mongodb://{self.mongodb_username}:{self.mongodb_password}@"
                f"{self.mongodb_host}:{self.mongodb_port}/{self.mongodb_database}"
                f"?authSource={self.mongodb_auth_database}"
            )
        else:
            return f"mongodb://{self.mongodb_host}:{self.mongodb_port}/{self.mongodb_database}"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )