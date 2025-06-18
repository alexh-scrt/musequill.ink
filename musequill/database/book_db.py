"""
MuseQuill MongoDB Client
Python client for storing and managing book data in MongoDB
"""
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4
from dataclasses import dataclass
from contextlib import asynccontextmanager

import pymongo
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, OperationFailure
from bson import ObjectId
import motor.motor_asyncio
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from musequill.config.logging import get_logger

# Configure logging
logger = get_logger(__name__)

MONGODB_HOST=os.getenv('MONGODB_HOST')
MONGODB_PORT=int(os.getenv('MONGODB_PORT'))
MONGODB_DATABASE=os.getenv('MONGODB_DATABASE')
MONGODB_USERNAME=os.getenv('MONGODB_USERNAME')
MONGODB_PASSWORD=os.getenv('MONGODB_PASSWORD')
MONGODB_AUTH_DATABASE=os.getenv('MONGODB_AUTH_DATABASE')

# Connection Pool Settings
MONGODB_MIN_POOL_SIZE=int(os.getenv('MONGODB_MIN_POOL_SIZE'))
MONGODB_MAX_POOL_SIZE=int(os.getenv('MONGODB_MAX_POOL_SIZE'))
MONGODB_MAX_IDLE_TIME_MS=int(os.getenv('MONGODB_IDLE_TIME_MS'))
MONGODB_WAIT_QUEUE_TIMEOUT_MS=int(os.getenv('MONGODB_WAIT_QUEUE_TIMEOUT_MS'))

@dataclass
class MongoDBConfig:
    """MongoDB connection configuration."""
    host: str = MONGODB_HOST
    port: int = MONGODB_PORT
    database: str = MONGODB_DATABASE
    username: str = MONGODB_USERNAME
    password: str = MONGODB_PASSWORD
    auth_database: str = MONGODB_AUTH_DATABASE
    
    # Connection pool settings
    min_pool_size: int = MONGODB_MIN_POOL_SIZE
    max_pool_size: int = MONGODB_MAX_POOL_SIZE
    max_idle_time_ms: int = MONGODB_MAX_IDLE_TIME_MS
    wait_queue_timeout_ms: int = MONGODB_WAIT_QUEUE_TIMEOUT_MS
    
    @property
    def connection_string(self) -> str:
        """Generate MongoDB connection string."""
        return (
            f"mongodb://{self.username}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
            f"?authSource={self.auth_database}"
        )

    def __repr__(self) -> str:
        return (
            f"MongoDBConfig("
            f"host='{self.host}', port={self.port}, database='{self.database}', "
            f"username='{self.username}', password='***', "
            f"auth_database='{self.auth_database}', "
            f"min_pool_size={self.min_pool_size}, max_pool_size={self.max_pool_size}, "
            f"max_idle_time_ms={self.max_idle_time_ms}, wait_queue_timeout_ms={self.wait_queue_timeout_ms})"
        )

    def to_dict(self, mask_sensitive: bool = True) -> dict:
        """ Convert MongoDBConfig to dict for logging """
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "username": self.username,
            "password": "***" if mask_sensitive else self.password,
            "auth_database": self.auth_database,
            "min_pool_size": self.min_pool_size,
            "max_pool_size": self.max_pool_size,
            "max_idle_time_ms": self.max_idle_time_ms,
            "wait_queue_timeout_ms": self.wait_queue_timeout_ms,
        }


class BookStatus:
    """Book status constants."""
    INITIALIZING = "initializing"
    PLANNING = "planning"
    PLANNED = "planned"
    APPROVED = "approved"
    WRITING = "writing"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentType:
    """Agent type constants."""
    PLANNER = "planner"
    WRITER = "writer"
    RESEARCHER = "researcher"
    EDITOR = "editor"


class AgentSessionStatus:
    """Agent session status constants."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MongoDBClient:
    """Synchronous MongoDB client for MuseQuill."""
    
    def __init__(self, config: Optional[MongoDBConfig] = None):
        """Initialize MongoDB client."""
        self.config = config or MongoDBConfig()
        self._client: Optional[MongoClient] = None
        self._database = None
        self._books_collection = None
        self._agent_sessions_collection = None
    
    def connect(self) -> None:
        """Connect to MongoDB."""
        try:
            self._client = MongoClient(
                self.config.connection_string,
                minPoolSize=self.config.min_pool_size,
                maxPoolSize=self.config.max_pool_size,
                maxIdleTimeMS=self.config.max_idle_time_ms,
                waitQueueTimeoutMS=self.config.wait_queue_timeout_ms,
                retryWrites=True
            )
            
            # Test connection
            self._client.admin.command('ping')
            
            # Get database and collections
            self._database = self._client[self.config.database]
            self._books_collection = self._database.books
            self._agent_sessions_collection = self._database.agent_sessions
            
            logger.info(f"Connected to MongoDB at {self.config.host}:{self.config.port}")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def disconnect(self) -> None:
        """Disconnect from MongoDB."""
        if self._client:
            self._client.close()
            logger.info("Disconnected from MongoDB")
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
    
    # Book Operations
    
    def create_book(self, book_data: Dict[str, Any]) -> str:
        """
        Create a new book record.
        
        Args:
            book_data: Book data dictionary
            
        Returns:
            Book ID as string
        """
        book_doc = prepare_book_document(book_data)
        book_id = book_doc.get('_id')
        try:
            result = self._books_collection.insert_one(book_doc)
            logger.info(f"Created book with ID: {book_id}: {result}")
            return book_id
            
        except DuplicateKeyError as e:
            logger.error(f"Book with ID {book_id} already exists")
            raise ValueError(f"Book with ID {book_id} already exists") from e
        except Exception as e:
            logger.error(f"Failed to create book: {e}")
            raise
    
    def get_book(self, book_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a book by ID.
        
        Args:
            book_id: Book UUID as string
            
        Returns:
            Book document or None
        """
        try:
            book = self._books_collection.find_one({'_id': book_id})
            if book:
                logger.debug(f"Retrieved book: {book_id}")
            return book
        except Exception as e:
            logger.error(f"Failed to get book {book_id}: {e}")
            raise
    
    def update_book(self, book_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Update a book record.
        
        Args:
            book_id: Book UUID as string
            update_data: Data to update
            
        Returns:
            True if updated, False if not found
        """
        update_data['updated_at'] = datetime.now(timezone.utc)
        
        try:
            result = self._books_collection.update_one(
                {'_id': book_id},
                {'$set': update_data}
            )
            
            if result.matched_count > 0:
                logger.info(f"Updated book: {book_id}")
                return True
            else:
                logger.warning(f"Book not found for update: {book_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update book {book_id}: {e}")
            raise

    def book_exists(self, book_id: str) -> bool:
            """
            Check if a book exists in the collection.
            
            Args:
                book_id: Book UUID as string
                
            Returns:
                True if book exists, False otherwise
            """
            try:
                # Use count_documents with limit=1 for efficiency
                count = self._books_collection.count_documents({'_id': book_id}, limit=1)
                exists = count > 0
                
                logger.debug(f"Book {book_id} exists: {exists}")
                return exists
                
            except Exception as e:
                logger.error(f"Failed to check if book {book_id} exists: {e}")
                raise


    def update_book_status(self, book_id: str, status: str, additional_data: Optional[Dict] = None) -> bool:
        """
        Update book status with optional additional data.
        
        Args:
            book_id: Book UUID as string
            status: New status
            additional_data: Additional data to merge
            
        Returns:
            True if updated, False if not found
        """
        update_data = {
            'status': status,
            'updated_at': datetime.now(timezone.utc)
        }
        
        if additional_data:
            update_data.update(additional_data)
        
        return self.update_book(book_id, update_data)
    
    def list_books(self, 
                   status: Optional[str] = None,
                   limit: int = 100,
                   skip: int = 0) -> List[Dict[str, Any]]:
        """
        List books with optional filtering.
        
        Args:
            status: Filter by status
            limit: Maximum number of books to return
            skip: Number of books to skip
            
        Returns:
            List of book documents
        """
        try:
            query = {}
            if status:
                query['status'] = status
            
            cursor = self._books_collection.find(query).sort('created_at', -1).skip(skip).limit(limit)
            books = list(cursor)
            
            logger.debug(f"Listed {len(books)} books")
            return books
            
        except Exception as e:
            logger.error(f"Failed to list books: {e}")
            raise
    
    def delete_book(self, book_id: str) -> bool:
        """
        Delete a book record.
        
        Args:
            book_id: Book UUID as string
            
        Returns:
            True if deleted, False if not found
        """
        try:
            result = self._books_collection.delete_one({'_id': book_id})
            
            if result.deleted_count > 0:
                logger.info(f"Deleted book: {book_id}")
                return True
            else:
                logger.warning(f"Book not found for deletion: {book_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete book {book_id}: {e}")
            raise
    
    # Agent Session Operations
    
    def create_agent_session(self, session_data: Dict[str, Any]) -> str:
        """
        Create a new agent session.
        
        Args:
            session_data: Agent session data
            
        Returns:
            Session ID as string
        """
        session_id = session_data.get('_id', str(uuid4()))
        
        session_doc = {
            '_id': session_id,
            'book_id': session_data['book_id'],
            'agent_type': session_data['agent_type'],
            'agent_id': session_data.get('agent_id', ''),
            'status': session_data.get('status', AgentSessionStatus.RUNNING),
            'started_at': datetime.now(timezone.utc),
            'completed_at': session_data.get('completed_at'),
            
            'input_data': session_data.get('input_data', {}),
            'output_data': session_data.get('output_data', {}),
            'progress_updates': session_data.get('progress_updates', []),
            
            'lock_acquired_at': session_data.get('lock_acquired_at'),
            'next_agent': session_data.get('next_agent')
        }
        
        try:
            result = self._agent_sessions_collection.insert_one(session_doc)
            logger.info(f"Created agent session: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to create agent session: {e}")
            raise
    
    def get_agent_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get an agent session by ID."""
        try:
            return self._agent_sessions_collection.find_one({'_id': session_id})
        except Exception as e:
            logger.error(f"Failed to get agent session {session_id}: {e}")
            raise
    
    def update_agent_session(self, session_id: str, update_data: Dict[str, Any]) -> bool:
        """Update an agent session."""
        try:
            result = self._agent_sessions_collection.update_one(
                {'_id': session_id},
                {'$set': update_data}
            )
            return result.matched_count > 0
        except Exception as e:
            logger.error(f"Failed to update agent session {session_id}: {e}")
            raise
    
    def list_agent_sessions(self, 
                           book_id: Optional[str] = None,
                           agent_type: Optional[str] = None,
                           status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List agent sessions with optional filtering."""
        try:
            query = {}
            if book_id:
                query['book_id'] = book_id
            if agent_type:
                query['agent_type'] = agent_type
            if status:
                query['status'] = status
            
            cursor = self._agent_sessions_collection.find(query).sort('started_at', -1)
            return list(cursor)
            
        except Exception as e:
            logger.error(f"Failed to list agent sessions: {e}")
            raise
    
    # Utility Methods
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on database connection."""
        try:
            # Test connection
            ping_result = self._client.admin.command('ping')
            
            # Get database stats
            db_stats = self._database.command('dbStats')
            
            # Count documents
            books_count = self._books_collection.count_documents({})
            sessions_count = self._agent_sessions_collection.count_documents({})
            
            return {
                'status': 'healthy',
                'ping': ping_result['ok'] == 1,
                'database': self.config.database,
                'books_count': books_count,
                'sessions_count': sessions_count,
                'db_size_mb': round(db_stats['dataSize'] / (1024 * 1024), 2),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }


class AsyncMongoDBClient:
    """Asynchronous MongoDB client for MuseQuill."""
    
    def __init__(self, config: Optional[MongoDBConfig] = None):
        """Initialize async MongoDB client."""
        self.config = config or MongoDBConfig()
        self._client: Optional[AsyncIOMotorClient] = None
        self._database: Optional[AsyncIOMotorDatabase] = None
        self._books_collection: Optional[AsyncIOMotorCollection] = None
        self._agent_sessions_collection: Optional[AsyncIOMotorCollection] = None
    
    async def connect(self) -> None:
        """Connect to MongoDB asynchronously."""
        try:
            self._client = AsyncIOMotorClient(
                self.config.connection_string,
                minPoolSize=self.config.min_pool_size,
                maxPoolSize=self.config.max_pool_size,
                maxIdleTimeMS=self.config.max_idle_time_ms,
                waitQueueTimeoutMS=self.config.wait_queue_timeout_ms,
                retryWrites=True
            )
            
            # Test connection
            await self._client.admin.command('ping')
            
            # Get database and collections
            self._database = self._client[self.config.database]
            self._books_collection = self._database.books
            self._agent_sessions_collection = self._database.agent_sessions
            
            logger.info(f"Connected to MongoDB at {self.config.host}:{self.config.port}")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from MongoDB."""
        if self._client:
            self._client.close()
            logger.info("Disconnected from MongoDB")
    
    @asynccontextmanager
    async def session(self):
        """Async context manager for database sessions."""
        await self.connect()
        try:
            yield self
        finally:
            await self.disconnect()
    
    # Async versions of the main methods
    async def create_book(self, book_data: Dict[str, Any]) -> str:
        """Create a new book record asynchronously."""
        book_id = book_data.get('_id', str(uuid4()))
        
        book_doc = {
            '_id': book_id,
            'created_at': datetime.now(timezone.utc),
            'updated_at': datetime.now(timezone.utc),
            'status': book_data.get('status', BookStatus.INITIALIZING),
            'title': book_data.get('title', 'Untitled'),
            'genre': book_data.get('genre', ''),
            'estimated_word_count': book_data.get('estimated_word_count', 0),
            'completion_percentage': book_data.get('completion_percentage', 0.0),
            'parameters': book_data.get('parameters', {}),
            'planning_results': book_data.get('planning_results', {}),
            'agent_workflows': book_data.get('agent_workflows', {}),
            'validation_info': book_data.get('validation_info', {}),
            'tags': book_data.get('tags', []),
            'last_agent_id': book_data.get('last_agent_id'),
            'approval_status': book_data.get('approval_status', 'pending')
        }
        
        try:
            await self._books_collection.insert_one(book_doc)
            logger.info(f"Created book with ID: {book_id}")
            return book_id
        except DuplicateKeyError:
            raise ValueError(f"Book with ID {book_id} already exists")
        except Exception as e:
            logger.error(f"Failed to create book: {e}")
            raise
    
    async def get_book(self, book_id: str) -> Optional[Dict[str, Any]]:
        """Get a book by ID asynchronously."""
        try:
            return await self._books_collection.find_one({'_id': book_id})
        except Exception as e:
            logger.error(f"Failed to get book {book_id}: {e}")
            raise
    
    async def update_book(self, book_id: str, update_data: Dict[str, Any]) -> bool:
        """Update a book record asynchronously."""
        update_data['updated_at'] = datetime.now(timezone.utc)
        
        try:
            result = await self._books_collection.update_one(
                {'_id': book_id},
                {'$set': update_data}
            )
            return result.matched_count > 0
        except Exception as e:
            logger.error(f"Failed to update book {book_id}: {e}")
            raise


def prepare_book_document(book_data:Dict) -> Dict:
    """
    Prepare a complete book document for MongoDB storage from book_data.
    Handles enum serialization and ensures all fields are captured.
    """
    book_id = book_data.get('_id', str(uuid4()))
    
    # Function to safely extract enum values
    def safe_enum_value(value):
        if hasattr(value, 'value'):
            return value.value
        return value
    
    # Function to safely extract list of enum values
    def safe_enum_list(enum_list):
        if not enum_list:
            return []
        return [safe_enum_value(item) for item in enum_list]
    
    # Prepare complete book document
    book_doc = {
        # Core identification and timestamps
        '_id': str(book_id),
        'created_at': book_data.get('created_at', datetime.now(timezone.utc)),
        'updated_at': book_data.get('updated_at', datetime.now(timezone.utc)),
        
        # Status and progress
        'status': book_data.get('status', BookStatus.INITIALIZING),
        'planning_status': book_data.get('planning_status', 'pending'),
        'completion_percentage': book_data.get('completion_percentage', 0.0),
        'estimated_word_count': book_data.get('estimated_word_count', 0),
        'estimated_chapters': book_data.get('estimated_chapters', 0),
        'validation_warnings': book_data.get('validation_warnings', []),
        
        # Core book information
        'title': book_data.get('title', 'Untitled'),
        'subtitle': book_data.get('subtitle'),
        'description': book_data.get('description'),
        'additional_notes': book_data.get('additional_notes'),
        
        # Genre and world building
        'genre_info': {
            'genre': book_data.get('genre_info', {}).get('genre', safe_enum_value(book_data.get('parameters', {}).get('genre', ''))),
            'sub_genre': book_data.get('genre_info', {}).get('sub_genre'),
            'is_fiction': book_data.get('genre_info', {}).get('is_fiction', True),
            'available_subgenres': book_data.get('genre_info', {}).get('available_subgenres', []),
            'world_type': book_data.get('genre_info', {}).get('world_type'),
            'magic_system': book_data.get('genre_info', {}).get('magic_system'),
            'technology_level': book_data.get('genre_info', {}).get('technology_level'),
        },
        
        # Story structure and plot
        'story_info': {
            'length': book_data.get('story_info', {}).get('length'),
            'structure': book_data.get('story_info', {}).get('structure'),
            'plot_type': book_data.get('story_info', {}).get('plot_type'),
            'pov': book_data.get('story_info', {}).get('pov'),
            'pacing': book_data.get('story_info', {}).get('pacing'),
            'conflict_types': book_data.get('story_info', {}).get('conflict_types', []),
            'complexity': book_data.get('story_info', {}).get('complexity'),
        },
        
        # Character information
        'character_info': {
            'main_character_role': book_data.get('character_info', {}).get('main_character_role'),
            'character_archetype': book_data.get('character_info', {}).get('character_archetype'),
        },
        
        # Writing style and tone
        'style_info': {
            'writing_style': book_data.get('style_info', {}).get('writing_style'),
            'tone': book_data.get('style_info', {}).get('tone'),
        },
        
        # Target audience
        'audience_info': {
            'age_group': book_data.get('audience_info', {}).get('age_group'),
            'audience_type': book_data.get('audience_info', {}).get('audience_type'),
            'reading_level': book_data.get('audience_info', {}).get('reading_level'),
        },
        
        # Publication information
        'publication_info': {
            'publication_route': book_data.get('publication_info', {}).get('publication_route'),
            'content_warnings': book_data.get('publication_info', {}).get('content_warnings', []),
        },
        
        # AI and writing process
        'process_info': {
            'ai_assistance_level': book_data.get('process_info', {}).get('ai_assistance_level'),
            'research_priority': book_data.get('process_info', {}).get('research_priority'),
            'writing_schedule': book_data.get('process_info', {}).get('writing_schedule'),
        },
        
        # Validation information
        'validation_info': {
            'genre_subgenre_valid': book_data.get('validation_info', {}).get('genre_subgenre_valid', True),
            'warnings': book_data.get('validation_info', {}).get('warnings', []),
        },
        
        # Agent and workflow information
        'agent_id': book_data.get('agent_id'),
        'last_agent_id': book_data.get('last_agent_id'),
        'planning_results': book_data.get('planning_results', {}),
        'agent_workflows': book_data.get('agent_workflows', {}),
        'status_message': book_data.get('status_message'),
        'next_steps': book_data.get('next_steps', []),
        
        # Error tracking
        'error_message': book_data.get('error_message'),
        'retry_count': book_data.get('retry_count', 0),
        'approval_status': book_data.get('approval_status', 'pending'),
        
        # Raw parameters (for backward compatibility and debugging)
        'parameters': book_data.get('parameters', {}),
        
        # Additional metadata
        'tags': book_data.get('tags', []),
        'version': book_data.get('version', 1),
        'metadata': book_data.get('metadata', {}),
    }
    
    # Remove None values to keep the document clean (optional)
    book_doc = {k: v for k, v in book_doc.items() if v is not None}
    
    # Clean nested dictionaries of None values
    for key, value in book_doc.items():
        if isinstance(value, dict):
            book_doc[key] = {k: v for k, v in value.items() if v is not None}
    
    return book_doc