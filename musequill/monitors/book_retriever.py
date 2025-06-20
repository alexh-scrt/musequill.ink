"""
Book Retriever Component

Retrieves books from Redis queue, validates and normalizes book data,
creates orchestration instances, and initiates the book writing pipeline.

Key Features:
- Polls Redis queue for new books
- Validates book data completeness and format
- Fills in missing values with sensible defaults
- Creates and manages LangGraph orchestrations
- Tracks concurrent orchestrations
- Error handling and dead letter queue support
"""

import asyncio
import json
import time
import threading
import signal
import atexit
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Set
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
import traceback

import redis
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import PyMongoError
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.redis import RedisSaver

from musequill.config.logging import get_logger
from musequill.monitors.book_retriever_config import BookRetrieverConfig
from musequill.agents.agent_state import BookWritingState, ProcessingStage, Chapter

logger = get_logger(__name__)


class BookRetriever:
    """
    Book Retriever that manages book queue processing and orchestration lifecycle.
    """
    
    def __init__(self, config: Optional[BookRetrieverConfig] = None):
        if not config:
            config = BookRetrieverConfig()
        
        self.config = config
        self.running = False
        self.shutdown_event = threading.Event()
        self.retriever_thread: Optional[threading.Thread] = None
        
        # Initialize clients
        self.redis_client: Optional[redis.Redis] = None
        self.mongo_client: Optional[AsyncIOMotorClient] = None
        self.books_collection = None
        
        # Orchestration management
        self.active_orchestrations: Set[str] = set()
        self.orchestration_futures: Dict[str, Any] = {}
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_orchestrations)
        
        # Statistics tracking
        self.stats = {
            'books_processed': 0,
            'books_failed': 0,
            'orchestrations_completed': 0,
            'orchestrations_failed': 0,
            'started_at': None
        }
        
        # Register cleanup handlers
        atexit.register(self.stop)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating book retriever shutdown...")
        self.stop()
    
    async def initialize(self) -> None:
        """Initialize all connections and clients."""
        try:
            # Initialize Redis connection
            self.redis_client = redis.from_url(self.config.redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info("Connected to Redis successfully")
            
            # Initialize MongoDB connection
            if self.config.database_username and self.config.database_password:
                if "://" in self.config.mongodb_url:
                    protocol, rest = self.config.mongodb_url.split("://", 1)
                    mongodb_url = f"{protocol}://{self.config.database_username}:{self.config.database_password}@{rest}?authSource={self.config.auth_database}"
                else:
                    mongodb_url = self.config.mongodb_url
            else:
                mongodb_url = self.config.mongodb_url
            
            self.mongo_client = AsyncIOMotorClient(mongodb_url)
            self.books_collection = self.mongo_client[self.config.database_name][self.config.books_collection]
            
            # Test MongoDB connection
            await self.mongo_client.admin.command('ping')
            logger.info("Connected to MongoDB successfully")
            
            self.stats['started_at'] = datetime.now(timezone.utc)
            logger.info("Book retriever initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize book retriever: {e}")
            raise
    
    async def get_book_from_queue(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single book from the Redis queue.
        
        Returns:
            Dict containing book data or None if queue is empty
        """
        try:
            # Use BLPOP with timeout to avoid blocking indefinitely
            result = self.redis_client.blpop(self.config.queue_name, timeout=1)
            if result:
                queue_name, book_data = result
                book = json.loads(book_data)
                logger.info(f"Retrieved book from queue: {book.get('book_id', book.get('_id', 'unknown'))}")
                return book
            return None
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in queue message: {e}")
            # Move malformed message to dead letter queue
            self.redis_client.rpush(self.config.dead_letter_queue, book_data if 'book_data' in locals() else "Invalid JSON")
            return None
        except Exception as e:
            logger.error(f"Error retrieving book from queue: {e}")
            return None
    
    def validate_book_data(self, book: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate book data for required fields and format.
        
        Args:
            book: Book data dictionary
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required fields
        for field in self.config.required_fields:
            if field not in book or book[field] is None:
                errors.append(f"Missing required field: {field}")
        
        # Validate specific fields if present
        if 'title' in book and book['title']:
            if len(book['title'].strip()) < self.config.min_title_length:
                errors.append(f"Title too short (minimum {self.config.min_title_length} characters)")
        
        if 'description' in book and book['description']:
            if len(book['description'].strip()) < self.config.min_description_length:
                errors.append(f"Description too short (minimum {self.config.min_description_length} characters)")
        
        # Validate book_id format
        book_id = book.get('book_id') or book.get('_id')
        if not book_id:
            errors.append("Missing book_id or _id field")
        elif not isinstance(book_id, str) or len(book_id.strip()) == 0:
            errors.append("Invalid book_id format")
        
        # Validate outline structure if present
        if 'outline' in book and book['outline']:
            if not isinstance(book['outline'], dict):
                errors.append("Outline must be a dictionary/object")
        
        return len(errors) == 0, errors
    
    def normalize_book_data(self, book: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fill in missing values and normalize book data.
        
        Args:
            book: Original book data
            
        Returns:
            Normalized book data with defaults filled in
        """
        normalized = book.copy()
        
        # Ensure book_id is set
        if 'book_id' not in normalized and '_id' in normalized:
            normalized['book_id'] = str(normalized['_id'])
        
        # Fill in missing optional fields with defaults
        if not normalized.get('genre'):
            normalized['genre'] = self.config.default_genre
        
        if not normalized.get('target_word_count'):
            normalized['target_word_count'] = self.config.default_target_length
        
        if not normalized.get('target_audience'):
            normalized['target_audience'] = "General audience"
        
        # Normalize outline and create chapters
        if not normalized.get('outline'):
            normalized['outline'] = {}
        
        # Extract or create chapter information
        chapters = []
        outline = normalized.get('outline', {})
        
        if 'chapters' in outline and isinstance(outline['chapters'], list):
            # Use existing chapter structure
            for i, chapter_data in enumerate(outline['chapters']):
                chapter = Chapter(
                    chapter_number=i + 1,
                    title=chapter_data.get('title', f"Chapter {i + 1}"),
                    description=chapter_data.get('description', ''),
                    target_word_count=chapter_data.get('target_word_count', 
                                                     normalized['target_word_count'] // len(outline['chapters'])),
                    status="planned",
                    content=None,
                    research_chunks_used=None,
                    word_count=None,
                    created_at=None,
                    completed_at=None
                )
                chapters.append(chapter)
        else:
            # Create default chapters
            words_per_chapter = normalized['target_word_count'] // self.config.default_chapter_count
            for i in range(self.config.default_chapter_count):
                chapter = Chapter(
                    chapter_number=i + 1,
                    title=f"Chapter {i + 1}",
                    description="",
                    target_word_count=words_per_chapter,
                    status="planned",
                    content=None,
                    research_chunks_used=None,
                    word_count=None,
                    created_at=None,
                    completed_at=None
                )
                chapters.append(chapter)
        
        normalized['chapters'] = chapters
        
        # Add timestamps
        current_time = datetime.now(timezone.utc).isoformat()
        normalized['processing_queued_at'] = current_time
        
        return normalized
    
    def create_initial_agent_state(self, book: Dict[str, Any]) -> BookWritingState:
        """
        Create initial AgentState for orchestration.
        
        Args:
            book: Normalized book data
            
        Returns:
            BookWritingState for LangGraph orchestration
        """
        orchestration_id = f"orch_{uuid4().hex[:12]}"
        thread_id = f"thread_{book['book_id']}_{uuid4().hex[:8]}"
        current_time = datetime.now(timezone.utc).isoformat()
        
        state = BookWritingState(
            # Book Identification
            book_id=str(book['book_id']),
            orchestration_id=orchestration_id,
            thread_id=thread_id,
            
            # Book Metadata
            title=book['title'],
            description=book['description'],
            genre=book['genre'],
            target_word_count=book['target_word_count'],
            target_audience=book.get('target_audience'),
            author_preferences=book.get('author_preferences', {}),
            
            # Planning Information
            outline=book['outline'],
            chapters=book['chapters'],
            
            # Processing Status
            current_stage=ProcessingStage.INITIALIZED,
            processing_started_at=current_time,
            processing_updated_at=current_time,
            
            # Research Phase
            research_queries=[],
            research_strategy=None,
            total_research_chunks=0,
            research_completed_at=None,
            
            # Writing Phase
            current_chapter=0,
            writing_strategy=None,
            writing_style_guide=None,
            total_word_count=0,
            writing_started_at=None,
            writing_completed_at=None,
            
            # Quality Control
            review_notes=None,
            revision_count=0,
            quality_score=None,
            
            # Error Handling
            errors=[],
            retry_count=0,
            last_error_at=None,
            
            # Progress Tracking
            progress_percentage=0.0,
            estimated_completion_time=None,
            
            # Final Output
            final_book_content=None,
            metadata={
                'retriever_processed_at': current_time,
                'original_book_data': book,
                'processing_node': 'book_retriever'
            }
        )
        
        return state
    
    def create_checkpointer(self):
        """Create appropriate checkpointer based on configuration."""
        if self.config.langgraph_checkpointer_type == "redis" and self.config.redis_checkpointer_url:
            return RedisSaver.from_conn_string(self.config.redis_checkpointer_url)
        else:
            return MemorySaver()
    
    async def start_orchestration(self, state: BookWritingState) -> bool:
        """
        Start the LangGraph orchestration for a book.
        
        Args:
            state: Initial BookWritingState
            
        Returns:
            True if orchestration started successfully, False otherwise
        """
        try:
            # Import orchestrator here to avoid circular imports
            from musequill.agents.orchestrator import create_book_writing_graph
            
            orchestration_id = state['orchestration_id']
            
            # Check if we're at capacity
            if len(self.active_orchestrations) >= self.config.max_concurrent_orchestrations:
                logger.warning(f"At orchestration capacity ({self.config.max_concurrent_orchestrations}), queueing book {state['book_id']}")
                # Put book back in queue for later processing
                book_data = {
                    'book_id': state['book_id'],
                    'title': state['title'],
                    'description': state['description'],
                    'genre': state['genre'],
                    'target_word_count': state['target_word_count'],
                    'outline': state['outline'],
                    'chapters': state['chapters'],
                    'retry_count': state['retry_count']
                }
                self.redis_client.rpush(self.config.queue_name, json.dumps(book_data))
                return False
            
            # Create checkpointer
            checkpointer = self.create_checkpointer()
            
            # Create the orchestration graph
            graph = create_book_writing_graph(checkpointer)
            
            # Create thread config
            thread_config = {"configurable": {"thread_id": state['thread_id']}}
            
            # Track this orchestration
            self.active_orchestrations.add(orchestration_id)
            
            # Submit orchestration to executor
            future = self.executor.submit(
                self._run_orchestration_sync,
                graph,
                state,
                thread_config,
                orchestration_id
            )
            
            self.orchestration_futures[orchestration_id] = future
            
            logger.info(f"Started orchestration {orchestration_id} for book {state['book_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start orchestration for book {state['book_id']}: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def _run_orchestration_sync(self, graph, state: BookWritingState, thread_config: Dict, orchestration_id: str):
        """
        Run orchestration synchronously in executor thread.
        This method handles the complete orchestration lifecycle.
        """
        try:
            logger.info(f"Running orchestration {orchestration_id} for book {state['book_id']}")
            
            # Update book status in MongoDB
            asyncio.create_task(self._update_book_status(
                state['book_id'], 
                "processing", 
                {"orchestration_id": orchestration_id, "stage": "started"}
            ))
            
            # Run the orchestration
            result = graph.invoke(state, thread_config)
            
            # Process successful completion
            logger.info(f"Orchestration {orchestration_id} completed successfully")
            self.stats['orchestrations_completed'] += 1
            
            # Update book status
            asyncio.create_task(self._update_book_status(
                state['book_id'],
                "completed",
                {"completed_at": datetime.now(timezone.utc).isoformat()}
            ))
            
            return result
            
        except Exception as e:
            logger.error(f"Orchestration {orchestration_id} failed: {e}")
            logger.error(traceback.format_exc())
            self.stats['orchestrations_failed'] += 1
            
            # Update book status
            asyncio.create_task(self._update_book_status(
                state['book_id'],
                "failed",
                {
                    "error": str(e),
                    "failed_at": datetime.now(timezone.utc).isoformat()
                }
            ))
            
            # Handle retry logic
            if state['retry_count'] < self.config.max_retries:
                logger.info(f"Queueing book {state['book_id']} for retry (attempt {state['retry_count'] + 1})")
                retry_book = {
                    'book_id': state['book_id'],
                    'title': state['title'],
                    'description': state['description'],
                    'genre': state['genre'],
                    'target_word_count': state['target_word_count'],
                    'outline': state['outline'],
                    'retry_count': state['retry_count'] + 1
                }
                # Delay retry
                time.sleep(self.config.retry_delay)
                self.redis_client.rpush(self.config.queue_name, json.dumps(retry_book))
            else:
                logger.error(f"Book {state['book_id']} failed after {self.config.max_retries} retries, moving to dead letter queue")
                self.redis_client.rpush(self.config.dead_letter_queue, json.dumps({
                    'book_id': state['book_id'],
                    'error': str(e),
                    'failed_at': datetime.now(timezone.utc).isoformat(),
                    'retry_count': state['retry_count']
                }))
            
        finally:
            # Cleanup
            self.active_orchestrations.discard(orchestration_id)
            self.orchestration_futures.pop(orchestration_id, None)
    
    async def _update_book_status(self, book_id: str, status: str, metadata: Dict[str, Any]):
        """Update book status in MongoDB."""
        try:
            update_data = {
                "$set": {
                    "writing_status": status,
                    "updated_at": datetime.now(timezone.utc),
                    **metadata
                }
            }
            
            await self.books_collection.update_one(
                {"$or": [{"book_id": book_id}, {"_id": book_id}]},
                update_data
            )
            
        except Exception as e:
            logger.error(f"Failed to update book status for {book_id}: {e}")
    
    async def process_book(self, book: Dict[str, Any]) -> bool:
        """
        Process a single book through validation, normalization, and orchestration.
        
        Args:
            book: Raw book data from queue
            
        Returns:
            True if processing started successfully, False otherwise
        """
        book_id = book.get('book_id', book.get('_id', 'unknown'))
        
        try:
            # Validate book data
            is_valid, errors = self.validate_book_data(book)
            if not is_valid:
                logger.error(f"Book {book_id} validation failed: {errors}")
                self.stats['books_failed'] += 1
                
                # Move to dead letter queue
                error_data = {
                    'book_id': book_id,
                    'validation_errors': errors,
                    'failed_at': datetime.now(timezone.utc).isoformat(),
                    'original_data': book
                }
                self.redis_client.rpush(self.config.dead_letter_queue, json.dumps(error_data))
                return False
            
            # Normalize book data
            normalized_book = self.normalize_book_data(book)
            
            # Create initial agent state
            initial_state = self.create_initial_agent_state(normalized_book)
            
            # Start orchestration
            success = await self.start_orchestration(initial_state)
            
            if success:
                self.stats['books_processed'] += 1
                logger.info(f"Successfully started processing for book {book_id}")
            else:
                logger.warning(f"Failed to start orchestration for book {book_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing book {book_id}: {e}")
            logger.error(traceback.format_exc())
            self.stats['books_failed'] += 1
            return False
    
    def cleanup_completed_orchestrations(self):
        """Clean up completed orchestration futures."""
        completed_ids = []
        for orchestration_id, future in self.orchestration_futures.items():
            if future.done():
                completed_ids.append(orchestration_id)
        
        for orchestration_id in completed_ids:
            self.orchestration_futures.pop(orchestration_id, None)
            self.active_orchestrations.discard(orchestration_id)
    
    async def _interruptible_sleep(self, duration: float) -> bool:
        """Sleep with interrupt capability for graceful shutdown."""
        end_time = time.time() + duration
        
        while time.time() < end_time:
            if self.shutdown_event.is_set():
                return True
            
            remaining = end_time - time.time()
            sleep_time = min(0.1, remaining)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        return False
    
    async def run(self) -> None:
        """Main book retriever loop."""
        self.running = True
        logger.info("Starting book retriever...")
        
        while self.running and not self.shutdown_event.is_set():
            try:
                # Clean up completed orchestrations
                self.cleanup_completed_orchestrations()
                
                # Get book from queue
                book = await self.get_book_from_queue()
                
                if book:
                    # Process the book
                    success = await self.process_book(book)
                    if not success:
                        logger.warning(f"Failed to process book {book.get('book_id', 'unknown')}")
                else:
                    # No books in queue, sleep briefly
                    if await self._interruptible_sleep(self.config.poll_interval):
                        break
                
                # Log statistics periodically
                if self.stats['books_processed'] % 10 == 0 and self.stats['books_processed'] > 0:
                    self._log_statistics()
                
            except asyncio.CancelledError:
                logger.info("Book retriever cancelled, shutting down...")
                break
            except Exception as e:
                logger.error(f"Unexpected error in book retriever: {e}")
                logger.error(traceback.format_exc())
                if await self._interruptible_sleep(5):  # Error recovery delay
                    break
        
        logger.info("Book retriever loop exited")
    
    def _log_statistics(self):
        """Log current processing statistics."""
        uptime = datetime.now(timezone.utc) - self.stats['started_at']
        
        logger.info(
            f"Book Retriever Statistics - "
            f"Uptime: {uptime}, "
            f"Books Processed: {self.stats['books_processed']}, "
            f"Books Failed: {self.stats['books_failed']}, "
            f"Orchestrations Active: {len(self.active_orchestrations)}, "
            f"Orchestrations Completed: {self.stats['orchestrations_completed']}, "
            f"Orchestrations Failed: {self.stats['orchestrations_failed']}"
        )
    
    def start(self) -> None:
        """Start the book retriever in a separate thread."""
        if self.running:
            logger.warning("Book retriever is already running")
            return
        
        logger.info("Starting book retriever...")
        
        self.retriever_thread = threading.Thread(
            target=self._run_async_loop,
            name="BookRetrieverThread",
            daemon=True
        )
        
        self.retriever_thread.start()
        time.sleep(1)  # Give thread time to start
        
        if self.retriever_thread.is_alive():
            logger.info("Book retriever thread started successfully")
        else:
            logger.error("Failed to start book retriever thread")
            self.running = False
    
    def _run_async_loop(self) -> None:
        """Run the async loop in a separate thread."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            loop.run_until_complete(self.initialize())
            loop.run_until_complete(self.run())
            
        except Exception as e:
            logger.error(f"Error in book retriever thread: {e}")
            logger.error(traceback.format_exc())
        finally:
            if hasattr(self, 'loop') and not loop.is_closed():
                loop.close()
    
    def stop(self, timeout: float = 30.0) -> None:
        """Stop the book retriever gracefully."""
        if not self.running:
            return
        
        logger.info("Stopping book retriever...")
        
        self.shutdown_event.set()
        self.running = False
        
        # Wait for active orchestrations to complete
        if self.active_orchestrations:
            logger.info(f"Waiting for {len(self.active_orchestrations)} active orchestrations to complete...")
            
            # Wait for orchestrations with timeout
            start_time = time.time()
            while self.active_orchestrations and (time.time() - start_time) < timeout:
                time.sleep(1)
                self.cleanup_completed_orchestrations()
            
            if self.active_orchestrations:
                logger.warning(f"{len(self.active_orchestrations)} orchestrations still active after timeout")
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Wait for the retriever thread to finish
        if self.retriever_thread and self.retriever_thread.is_alive():
            logger.info(f"Waiting up to {timeout}s for retriever thread to stop...")
            self.retriever_thread.join(timeout=timeout)
            
            if self.retriever_thread.is_alive():
                logger.warning("Retriever thread did not stop gracefully within timeout")
            else:
                logger.info("Book retriever stopped successfully")
        
        # Final statistics
        self._log_statistics()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status and statistics."""
        uptime = None
        if self.stats['started_at']:
            uptime = (datetime.now(timezone.utc) - self.stats['started_at']).total_seconds()
        
        return {
            'running': self.running,
            'uptime_seconds': uptime,
            'active_orchestrations': len(self.active_orchestrations),
            'orchestration_capacity': self.config.max_concurrent_orchestrations,
            'statistics': self.stats.copy(),
            'queue_name': self.config.queue_name,
            'dead_letter_queue': self.config.dead_letter_queue
        }


def main():
    """Test function for BookRetriever."""
    import time
    
    print("Testing BookRetriever...")
    
    retriever = BookRetriever()
    
    try:
        print("Starting book retriever...")
        retriever.start()
        
        print("Retriever running for 30 seconds...")
        time.sleep(30)
        
        print("Stopping retriever...")
        retriever.stop(timeout=10.0)
        
        print("Test completed successfully!")
        print("Final status:", retriever.get_status())
        
    except KeyboardInterrupt:
        print("\nReceived Ctrl+C, stopping retriever...")
        retriever.stop(timeout=10.0)
    except Exception as e:
        print(f"Error during test: {e}")
        retriever.stop(timeout=10.0)


if __name__ == "__main__":
    main()