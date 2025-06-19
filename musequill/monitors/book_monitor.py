"""
Book Pipeline DB Monitor

Monitors the MongoDB books collection for books ready to be written and sends them
to a writing pipeline using Redis as a simple, persistent message queue.

Key Features:
- Monitors for books with status='planned' and planning_completed=True
- Atomically updates book status to prevent duplicate processing
- Uses Redis for reliable message queuing
- Comprehensive logging and error handling
- Graceful shutdown handling
"""

import asyncio
import json
import atexit
import signal
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional
import threading

import redis
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ReturnDocument
from pymongo.errors import PyMongoError

from musequill.config.logging import get_logger

from musequill.monitors.book_monitor_config import BookMonitorConfig


logger = get_logger(__name__)


class BookMonitor:
    """
    Monitors the books collection and sends ready books to the writing pipeline.
    """
    
    def __init__(
        self,
        _config:Optional[BookMonitorConfig] = None
    ):
        if not _config:
            _config = BookMonitorConfig()
        self.mongodb_url = _config.mongodb_url
        self.database_name = _config.database_name
        self.database_username = _config.database_username
        self.database_password = _config.database_password
        self.auth_database = _config.auth_database
        self.collection_name = _config.collection_name
        self.redis_url = _config.redis_url
        self.queue_name = _config.queue_name
        self.poll_interval = _config.poll_interval
        
        # Initialize connections
        self.mongo_client: Optional[AsyncIOMotorClient] = None
        self.redis_client: Optional[redis.Redis] = None
        self.collection = None
        self.running = False
        self.shutdown_event = threading.Event()
        self.monitor_thread: Optional[threading.Thread] = None

        # Register cleanup handlers
        atexit.register(self.stop)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)


    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating monitor shutdown...")
        self.stop()


    async def initialize(self) -> None:
        """Initialize database and queue connections."""
        try:
            # Build MongoDB connection URL with authentication if credentials are provided
            if self.database_username and self.database_password:
                # Parse the URL to insert credentials with authSource
                if "://" in self.mongodb_url:
                    protocol, rest = self.mongodb_url.split("://", 1)
                    mongodb_url = f"{protocol}://{self.database_username}:{self.database_password}@{rest}?authSource={self.auth_database}"
                else:
                    mongodb_url = self.mongodb_url
            else:
                mongodb_url = self.mongodb_url
            
            # Initialize MongoDB connection
            self.mongo_client = AsyncIOMotorClient(mongodb_url)
            self.collection = self.mongo_client[self.database_name][self.collection_name]
            
            # Test MongoDB connection
            await self.mongo_client.admin.command('ping')
            logger.info("Connected to MongoDB successfully")
            
            # Initialize Redis connection
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            
            # Test Redis connection
            self.redis_client.ping()
            logger.info("Connected to Redis successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize connections: {e}")
            raise


    def start(self) -> None:
        """
        Start the book monitor in a separate daemon thread.
        This method returns immediately after starting the thread.
        """
        if self.running:
            logger.warning("Book monitor is already running")
            return
        
        logger.info("Starting threaded book monitor...")
        
        # Create and start the monitor thread
        self.monitor_thread = threading.Thread(
            target=self._run_async_loop,
            name="BookMonitorThread",
            daemon=True  # Dies when main thread dies
        )
        
        self.monitor_thread.start()
        
        # Give the thread a moment to start up
        time.sleep(1)
        
        if self.monitor_thread.is_alive():
            logger.info("Book monitor thread started successfully")
        else:
            logger.error("Failed to start book monitor thread")
            self.running = False


    def stop(self, timeout: float = 10.0) -> None:
        """
        Stop the book monitor gracefully.
        
        Args:
            timeout: Maximum time to wait for graceful shutdown
        """
        if not self.running:
            return
        
        logger.info("Stopping book monitor...")
        
        # Signal shutdown
        self.shutdown_event.set()
        self.running = False
        
        # Wait for the monitor thread to finish
        if self.monitor_thread and self.monitor_thread.is_alive():
            logger.info(f"Waiting up to {timeout}s for monitor thread to stop...")
            self.monitor_thread.join(timeout=timeout)
            
            if self.monitor_thread.is_alive():
                logger.warning("Monitor thread did not stop gracefully within timeout")
            else:
                logger.info("Monitor thread stopped successfully")

    
    def get_ready_books_filter(self) -> Dict:
        """
        Define the MongoDB filter for books ready to be written.
        
        A book is ready for writing when:
        1. status == 'planned' 
        2. planning_completed == True
        3. planning_status != 'error' (avoid error states)
        4. Not already in writing status
        """
        return {
            "status": "planned",
            "planning_completed": True,
            "planning_status": {"$ne": "error"},
            # Ensure we don't pick up books already being processed
            "$or": [
                {"writing_status": {"$exists": False}},
                {"writing_status": {"$nin": ["in_progress", "writing", "processing"]}}
            ]
        }
    
    def validate_book_for_writing(self, book: Dict) -> tuple[bool, List[str]]:
        """
        Validate that a book has all required fields for writing.
        
        Returns:
            tuple: (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required top-level fields
        required_fields = ['_id', 'title', 'genre_info', 'planning_results']
        for field in required_fields:
            if field not in book:
                issues.append(f"Missing required field: {field}")
        
        # Check planning results structure
        if 'planning_results' in book:
            planning_results = book['planning_results']
            
            # Check for story outline
            if 'story_outline' not in planning_results:
                issues.append("Missing story_outline in planning_results")
            
            # Check for chapter structure
            if 'chapter_structure' not in planning_results:
                issues.append("Missing chapter_structure in planning_results")
            elif not isinstance(planning_results['chapter_structure'], list):
                issues.append("chapter_structure must be a list")
            elif len(planning_results['chapter_structure']) == 0:
                issues.append("chapter_structure cannot be empty")
        
        # Validate genre info
        if 'genre_info' in book:
            genre_info = book['genre_info']
            if 'genre' not in genre_info:
                issues.append("Missing genre in genre_info")
        
        return len(issues) == 0, issues
    
    async def mark_book_as_writing(self, book_id: str) -> bool:
        """
        Atomically mark a book as being written to prevent duplicate processing.
        
        Returns:
            bool: True if successfully marked, False if already taken by another process
        """
        try:
            result = await self.collection.find_one_and_update(
                {
                    "_id": book_id,
                    "status": "planned",
                    "$or": [
                        {"writing_status": {"$exists": False}},
                        {"writing_status": {"$nin": ["in_progress", "writing", "processing"]}}
                    ]
                },
                {
                    "$set": {
                        "status": "writing",
                        "writing_status": "in_progress",
                        "writing_started_at": datetime.now(timezone.utc),
                        "updated_at": datetime.now(timezone.utc)
                    }
                },
                return_document=ReturnDocument.AFTER
            )
            
            return result is not None
            
        except PyMongoError as e:
            logger.error(f"Failed to mark book {book_id} as writing: {e}")
            return False
    
    def create_pipeline_message(self, book: Dict) -> Dict:
        """
        Create a standardized message for the writing pipeline.
        """
        return {
            "book_id": book["_id"],
            "title": book["title"],
            "genre": book["genre_info"]["genre"],
            "sub_genre": book["genre_info"].get("sub_genre"),
            "estimated_chapters": book.get("estimated_chapters", 1),
            "estimated_word_count": book.get("estimated_word_count", 2500),
            "planning_results": book["planning_results"],
            "queued_at": datetime.now(timezone.utc).isoformat(),
            "priority": "normal"  # Could be enhanced with priority logic
        }
    
    async def send_to_pipeline(self, message: Dict) -> bool:
        """
        Send a book to the writing pipeline via Redis queue.
        
        Returns:
            bool: True if successfully queued, False otherwise
        """
        try:
            # Use Redis LPUSH for FIFO queue behavior
            queue_length = self.redis_client.lpush(
                self.queue_name, 
                json.dumps(message, default=str)
            )
            
            logger.info(
                f"Book {message['book_id']} queued for writing. "
                f"Queue length: {queue_length}"
            )
            return True
            
        except redis.RedisError as e:
            logger.error(f"Failed to queue book {message['book_id']}: {e}")
            return False
    
    async def process_ready_books(self) -> int:
        """
        Find and process all books ready for writing.
        
        Returns:
            int: Number of books successfully queued
        """
        processed_count = 0
        
        try:
            # Find books ready for writing
            filter_query = self.get_ready_books_filter()
            ready_books = await self.collection.find(filter_query).to_list(length=100)
            
            if not ready_books:
                logger.debug("No books ready for writing found")
                return 0
            
            logger.info(f"Found {len(ready_books)} books ready for writing")
            
            for book in ready_books:
                book_id = book["_id"]
                
                try:
                    # Validate book structure
                    is_valid, issues = self.validate_book_for_writing(book)
                    if not is_valid:
                        logger.warning(
                            f"Book {book_id} failed validation: {', '.join(issues)}"
                        )
                        # Mark book with validation error
                        await self.collection.update_one(
                            {"_id": book_id},
                            {
                                "$set": {
                                    "validation_status": "failed",
                                    "validation_errors": issues,
                                    "updated_at": datetime.now(timezone.utc)
                                }
                            }
                        )
                        continue
                    
                    # Atomically claim the book for writing
                    if not await self.mark_book_as_writing(book_id):
                        logger.info(f"Book {book_id} already claimed by another process")
                        continue
                    
                    # Create pipeline message
                    message = self.create_pipeline_message(book)
                    
                    # Send to pipeline
                    if await self.send_to_pipeline(message):
                        processed_count += 1
                        logger.info(f"Successfully queued book {book_id} for writing")
                    else:
                        # Rollback status change if queueing failed
                        await self.collection.update_one(
                            {"_id": book_id},
                            {
                                "$set": {
                                    "status": "planned",
                                    "writing_status": "queue_failed",
                                    "updated_at": datetime.now(timezone.utc)
                                },
                                "$unset": {"writing_started_at": ""}
                            }
                        )
                        logger.error(f"Failed to queue book {book_id}, status rolled back")
                
                except Exception as e:
                    logger.error(f"Error processing book {book_id}: {e}")
                    continue
            
            if processed_count > 0:
                logger.info(f"Successfully processed {processed_count} books")
            
        except Exception as e:
            logger.error(f"Error in process_ready_books: {e}")
        
        return processed_count
    
    async def _interruptible_sleep(self, duration: float) -> bool:
        """
        Sleep for the specified duration but wake up if shutdown is signaled.
        
        Returns:
            bool: True if shutdown was signaled, False if sleep completed normally
        """
        end_time = time.time() + duration
        
        while time.time() < end_time:
            if self.shutdown_event.is_set():
                return True
            
            # Sleep in small increments to check shutdown signal
            remaining = end_time - time.time()
            sleep_time = min(0.1, remaining)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        return False
    
    async def run(self) -> None:
        """Main monitoring loop."""
        self.running = True
        logger.info("Starting book pipeline monitor...")
        
        while self.running and not self.shutdown_event.is_set():
            try:
                await self.process_ready_books()
                
                # Sleep for the specified interval, but wake up if shutdown is signaled
                if await self._interruptible_sleep(self.poll_interval):
                    logger.info("Shutdown signal received during sleep, stopping...")
                    break
                
            except asyncio.CancelledError:
                logger.info("Monitor cancelled, shutting down...")
                break
            except Exception as e:
                logger.error(f"Unexpected error in monitor loop: {e}")
                # Use interruptible sleep for error recovery too
                if await self._interruptible_sleep(self.poll_interval):
                    logger.info("Shutdown signal received during error recovery, stopping...")
                    break
        
        logger.info("Monitor loop exited")
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the monitor."""
        logger.info("Shutting down book pipeline monitor...")
        self.running = False
        
        if self.mongo_client:
            self.mongo_client.close()
            logger.info("MongoDB connection closed")
        
        if self.redis_client:
            self.redis_client.close()
            logger.info("Redis connection closed")

    def _run_async_loop(self) -> None:
        """
        Run the async monitor loop in a separate thread.
        Creates a new event loop for this thread and runs the async run method.
        """
        try:
            # Initialize connections in the new thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run initialization and monitoring
            loop.run_until_complete(self.initialize())
            loop.run_until_complete(self.run())
            
        except Exception as e:
            logger.error(f"Error in monitor thread: {e}")
        finally:
            # Cleanup
            if hasattr(self, 'loop') and not loop.is_closed():
                loop.run_until_complete(self.shutdown())
                loop.close()
