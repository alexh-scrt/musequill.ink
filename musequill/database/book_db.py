"""
MuseQuill MongoDB Client
Python client for storing and managing book data in MongoDB
"""

import os
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager

import pymongo
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, OperationFailure
from bson import ObjectId
import motor.motor_asyncio
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MongoDBConfig:
    """MongoDB connection configuration."""
    host: str = "localhost"
    port: int = 27017
    database: str = "musequill"
    username: str = "musequill"
    password: str = "musequill.ink.user"
    auth_database: str = "musequill"
    
    # Connection pool settings
    min_pool_size: int = 5
    max_pool_size: int = 50
    max_idle_time_ms: int = 30000
    wait_queue_timeout_ms: int = 5000
    
    @property
    def connection_string(self) -> str:
        """Generate MongoDB connection string."""
        return (
            f"mongodb://{self.username}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
            f"?authSource={self.auth_database}"
        )


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
        # Generate UUID if not provided
        book_id = book_data.get('_id', str(uuid4()))
        
        # Prepare book document
        book_doc = {
            '_id': book_id,
            'created_at': datetime.now(timezone.utc),
            'updated_at': datetime.now(timezone.utc),
            'status': book_data.get('status', BookStatus.INITIALIZING),
            'title': book_data.get('title', 'Untitled'),
            'genre': book_data.get('genre', ''),
            'estimated_word_count': book_data.get('estimated_word_count', 0),
            'completion_percentage': book_data.get('completion_percentage', 0.0),
            
            # Complex nested data
            'parameters': book_data.get('parameters', {}),
            'planning_results': book_data.get('planning_results', {}),
            'agent_workflows': book_data.get('agent_workflows', {}),
            'validation_info': book_data.get('validation_info', {}),
            
            # Additional fields
            'tags': book_data.get('tags', []),
            'last_agent_id': book_data.get('last_agent_id'),
            'approval_status': book_data.get('approval_status', 'pending')
        }
        
        try:
            result = self._books_collection.insert_one(book_doc)
            logger.info(f"Created book with ID: {book_id}")
            return book_id
            
        except DuplicateKeyError:
            logger.error(f"Book with ID {book_id} already exists")
            raise ValueError(f"Book with ID {book_id} already exists")
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


# Example usage and testing
if __name__ == "__main__":
    # Test the MongoDB client
    config = MongoDBConfig()
    
    # Test synchronous client
    with MongoDBClient(config) as client:
        # Health check
        health = client.health_check()
        print("Health check:", health)
        
        # Create a test book
        test_book = {
            'title': 'Test Book',
            'genre': 'fiction',
            'estimated_word_count': 50000,
            'parameters': {
                'genre': 'fantasy',
                'length': 'novel',
                'style': 'epic'
            }
        }
        
        book_id = client.create_book(test_book)
        print(f"Created book: {book_id}")
        
        # Get the book
        retrieved_book = client.get_book(book_id)
        print(f"Retrieved book: {retrieved_book['title']}")
        
        # Update book status
        client.update_book_status(book_id, BookStatus.PLANNING)
        
        # List books
        books = client.list_books()
        print(f"Total books: {len(books)}")
        
        # Clean up
        client.delete_book(book_id)
        print("Test book deleted")