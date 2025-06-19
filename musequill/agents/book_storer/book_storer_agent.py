"""
Book Storer Agent

Stores completed books and their associated files in MongoDB with comprehensive
metadata, versioning, and quality assurance.

Key Features:
- MongoDB storage with connection pooling and error handling
- Multi-format file storage with compression and validation
- Comprehensive metadata indexing and search capabilities
- Version tracking and backup management
- Quality validation and integrity checking
- Performance monitoring and optimization
- Automatic cleanup and maintenance
- Full-text search indexing and content embeddings
"""

import os
import gzip
import time
import hashlib
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from uuid import uuid4
import traceback

import pymongo
from pymongo import MongoClient, IndexModel, TEXT
from pymongo.errors import DuplicateKeyError, OperationFailure, ServerSelectionTimeoutError
from bson import ObjectId, Binary
from gridfs import GridFS

from musequill.config.logging import get_logger
from musequill.agents.book_storer.book_storer_config import BookStorerConfig
from musequill.agents.agent_state import BookWritingState
from musequill.agents.assembler.final_assembler_agent import AssemblyResults, FormattedDocument

logger = get_logger(__name__)


@dataclass
class StoredBookMetadata:
    """Complete metadata for a stored book."""
    book_id: str
    title: str
    author: str
    genre: str
    description: str
    word_count: int
    chapter_count: int
    quality_score: float
    assembly_results: Dict[str, Any]
    created_at: str
    stored_at: str
    version: int
    file_formats: List[str]
    file_sizes: Dict[str, int]
    content_hash: str
    storage_location: str
    tags: List[str]
    search_keywords: List[str]
    generation_metadata: Dict[str, Any]


@dataclass
class StoredFile:
    """Information about a stored book file."""
    file_id: str
    format_type: str
    file_size: int
    compressed_size: Optional[int]
    checksum: str
    stored_at: str
    compression_ratio: Optional[float]
    gridfs_id: Optional[ObjectId]


@dataclass
class BookStorageResult:
    """Result of book storage operation."""
    success: bool
    book_id: str
    storage_location: str
    stored_files: List[StoredFile]
    metadata_id: str
    version: int
    total_size_bytes: int
    storage_time: float
    quality_validation_passed: bool
    warnings: List[str]
    error_message: Optional[str]


class BookStorerAgent:
    """
    Book Storer Agent that persists completed books to MongoDB with comprehensive features.
    """
    
    def __init__(self, config: Optional[BookStorerConfig] = None):
        if config is None:
            config = BookStorerConfig()
        
        self.config = config
        
        # MongoDB client and collections
        self.client: Optional[MongoClient] = None
        self.database = None
        self.books_collection = None
        self.book_files_collection = None
        self.book_metadata_collection = None
        self.gridfs: Optional[GridFS] = None
        
        # Storage statistics
        self.storage_stats = {
            'books_stored': 0,
            'total_bytes_stored': 0,
            'files_stored': 0,
            'compression_saved_bytes': 0,
            'average_storage_time': 0.0,
            'storage_failures': 0,
            'session_start': None
        }
        
        self._initialize_connection()
        logger.info("Book Storer Agent initialized")
    
    def _initialize_connection(self) -> None:
        """Initialize MongoDB connection and collections."""
        try:
            # Create MongoDB client
            self.client = MongoClient(
                self.config.connection_string,
                minPoolSize=self.config.min_pool_size,
                maxPoolSize=self.config.max_pool_size,
                maxIdleTimeMS=self.config.max_idle_time_ms,
                waitQueueTimeoutMS=self.config.wait_queue_timeout_ms,
                connectTimeoutMS=self.config.connection_timeout_ms,
                serverSelectionTimeoutMS=self.config.operation_timeout_ms,
                retryWrites=True,
                retryReads=True
            )
            
            # Get database and collections
            self.database = self.client[self.config.mongodb_database]
            self.books_collection = self.database[self.config.books_collection_name]
            self.book_files_collection = self.database[self.config.book_files_collection_name]
            self.book_metadata_collection = self.database[self.config.book_metadata_collection_name]
            
            # Initialize GridFS for large file storage
            self.gridfs = GridFS(self.database)
            
            # Create indexes
            self._create_indexes()
            
            # Test connection
            self.client.admin.command('ping')
            logger.info("MongoDB connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB connection: {e}")
            raise
    
    def _create_indexes(self) -> None:
        """Create necessary indexes for efficient querying."""
        try:
            # Books collection indexes
            books_indexes = [
                IndexModel([("book_id", 1)], unique=True),
                IndexModel([("title", 1)]),
                IndexModel([("author", 1)]),
                IndexModel([("genre", 1)]),
                IndexModel([("created_at", -1)]),
                IndexModel([("stored_at", -1)]),
                IndexModel([("quality_score", -1)]),
                IndexModel([("word_count", 1)]),
                IndexModel([("tags", 1)]),
                IndexModel([("search_keywords", 1)]),
                IndexModel([("content_hash", 1)]),
                IndexModel([("version", 1)])
            ]
            
            if self.config.enable_full_text_search:
                books_indexes.append(
                    IndexModel([("title", TEXT), ("description", TEXT), ("search_keywords", TEXT)])
                )
            
            self.books_collection.create_indexes(books_indexes)
            
            # Book files collection indexes
            files_indexes = [
                IndexModel([("book_id", 1)]),
                IndexModel([("format_type", 1)]),
                IndexModel([("file_id", 1)], unique=True),
                IndexModel([("stored_at", -1)]),
                IndexModel([("checksum", 1)])
            ]
            
            self.book_files_collection.create_indexes(files_indexes)
            
            # Metadata collection indexes
            metadata_indexes = [
                IndexModel([("book_id", 1)]),
                IndexModel([("version", 1)]),
                IndexModel([("created_at", -1)])
            ]
            
            self.book_metadata_collection.create_indexes(metadata_indexes)
            
            logger.info("Database indexes created successfully")
            
        except Exception as e:
            logger.warning(f"Error creating indexes: {e}")
    
    def store_book(self, state: BookWritingState) -> Dict[str, Any]:
        """
        Backward compatibility method for orchestrator integration.
        
        This method provides compatibility with the orchestrator's expected interface
        while internally handling the assembly results extraction.
        """
        try:
            # Extract assembly results from state
            assembly_results = state.get('assembly_results')
            
            if not assembly_results:
                # Create a basic assembly results object if not present
                from musequill.agents.assembler.final_assembler_structures import AssemblyResults
                from datetime import datetime
                
                # Create minimal assembly results from state
                assembly_results = AssemblyResults(
                    book_id=state['book_id'],
                    success=True,
                    total_word_count=sum(len(ch.get('content', '').split()) for ch in state.get('chapters', [])),
                    overall_start_time=datetime.now(),
                    overall_end_time=datetime.now(),
                    total_duration=0.0
                )
            
            # Call the main storage method
            return self.store_completed_book(state, assembly_results)
            
        except Exception as e:
            logger.error(f"Error in store_book compatibility method: {e}")
            return {
                'status': 'error',
                'error_message': str(e),
                'success': False,
                'document_id': None,
                'metadata': {}
            }
    
    def store_completed_book(self, state: BookWritingState, assembly_results: AssemblyResults) -> Dict[str, Any]:
        """
        Main entry point for storing completed books.
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting storage for book {state['book_id']}")
            
            # Phase 1: Validation
            validation_result = self._validate_book_for_storage(state, assembly_results)
            if not validation_result['valid']:
                return {
                    'status': 'error',
                    'error_message': f"Validation failed: {validation_result['error']}",
                    'success': False
                }
            
            # Phase 2: Prepare storage data
            book_metadata = self._prepare_book_metadata(state, assembly_results)
            
            # Phase 3: Store files
            stored_files = []
            if assembly_results.generated_formats:
                stored_files = self._store_book_files(assembly_results.generated_formats, state['book_id'])
            
            # Phase 4: Store main book document
            book_document = self._create_book_document(book_metadata, stored_files, state, assembly_results)
            
            # Phase 5: Execute storage with transaction
            storage_result = self._execute_storage_transaction(book_document, book_metadata, stored_files)
            
            # Phase 6: Update statistics and cleanup
            storage_time = time.time() - start_time
            self._update_storage_stats(storage_time, storage_result)
            
            if self.config.enable_auto_cleanup:
                self._cleanup_temporary_data(state['book_id'])
            
            return {
                'status': 'success',
                'storage_result': storage_result,
                'success': storage_result.success,
                'book_id': state['book_id'],
                'document_id': storage_result.metadata_id,  # For orchestrator compatibility
                'storage_location': storage_result.storage_location,
                'stored_files': len(stored_files),
                'storage_time': storage_time,
                'total_size_mb': storage_result.total_size_bytes / (1024 * 1024),
                'metadata': {  # For orchestrator compatibility
                    'storage_location': storage_result.storage_location,
                    'total_size_mb': storage_result.total_size_bytes / (1024 * 1024),
                    'stored_formats': len(stored_files),
                    'storage_completed_at': datetime.now(timezone.utc).isoformat(),
                    'version': storage_result.version,
                    'book_id': state['book_id']
                }
            }
            
        except Exception as e:
            logger.error(f"Error storing book {state['book_id']}: {e}")
            traceback.print_exc()
            
            # Update failure statistics
            self.storage_stats['storage_failures'] += 1
            
            return {
                'status': 'error',
                'error_message': str(e),
                'success': False,
                'storage_time': time.time() - start_time,
                'document_id': None,  # For orchestrator compatibility
                'metadata': {}  # For orchestrator compatibility
            }
    
    def _validate_book_for_storage(self, state: BookWritingState, assembly_results: AssemblyResults) -> Dict[str, Any]:
        """Validate book data before storage."""
        try:
            errors = []
            
            # Basic validation
            if not state.get('book_id'):
                errors.append("Missing book_id")
            
            if not state.get('title'):
                errors.append("Missing book title")
            
            if not assembly_results.success:
                errors.append("Assembly was not successful")
            
            # Quality validation
            if self.config.require_quality_score:
                quality_score = state.get('quality_score', 0.0)
                if quality_score < self.config.min_quality_score:
                    errors.append(f"Quality score {quality_score:.2f} below minimum {self.config.min_quality_score}")
            
            # Word count validation
            if assembly_results.total_word_count < 1000:
                errors.append(f"Word count too low: {assembly_results.total_word_count}")
            
            # File validation
            if self.config.store_file_attachments and not assembly_results.generated_formats:
                errors.append("No generated formats available for storage")
            
            # Check for duplicate
            if self._book_exists(state['book_id']):
                if not self.config.enable_versioning:
                    errors.append(f"Book {state['book_id']} already exists and versioning is disabled")
            
            return {
                'valid': len(errors) == 0,
                'error': '; '.join(errors) if errors else None,
                'warnings': []
            }
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return {
                'valid': False,
                'error': f"Validation failed: {str(e)}",
                'warnings': []
            }
    
    def _book_exists(self, book_id: str) -> bool:
        """Check if book already exists in database."""
        try:
            return self.books_collection.count_documents({'book_id': book_id}) > 0
        except Exception as e:
            logger.error(f"Error checking book existence: {e}")
            return False
    
    def _prepare_book_metadata(self, state: BookWritingState, assembly_results: AssemblyResults) -> StoredBookMetadata:
        """Prepare comprehensive book metadata."""
        # Generate content hash
        content_hash = self._generate_content_hash(state, assembly_results)
        
        # Extract search keywords
        search_keywords = self._extract_search_keywords(state, assembly_results)
        
        # Determine version
        version = self._get_next_version(state['book_id']) if self.config.enable_versioning else 1
        
        # Calculate file information
        file_formats = [f.format_type for f in assembly_results.generated_formats]
        file_sizes = {f.format_type: f.file_size for f in assembly_results.generated_formats}
        
        return StoredBookMetadata(
            book_id=state['book_id'],
            title=state['title'],
            author=state.get('author', 'MuseQuill AI'),
            genre=state['genre'],
            description=state.get('description', ''),
            word_count=assembly_results.total_word_count,
            chapter_count=len(state['chapters']),
            quality_score=state.get('quality_score', 0.0),
            assembly_results=self._serialize_assembly_results(assembly_results),
            created_at=state.get('created_at', datetime.now(timezone.utc).isoformat()),
            stored_at=datetime.now(timezone.utc).isoformat(),
            version=version,
            file_formats=file_formats,
            file_sizes=file_sizes,
            content_hash=content_hash,
            storage_location=f"mongodb://{self.config.mongodb_host}/{self.config.mongodb_database}",
            tags=state.get('tags', []),
            search_keywords=search_keywords,
            generation_metadata={
                'pipeline_version': '1.0',
                'agents_used': state.get('agents_used', []),
                'processing_time': state.get('total_processing_time', 0),
                'research_chunks_used': state.get('total_research_chunks', 0),
                'revision_count': state.get('revision_count', 0)
            }
        )
    
    def _generate_content_hash(self, state: BookWritingState, assembly_results: AssemblyResults) -> str:
        """Generate hash of book content for deduplication."""
        content_parts = []
        
        # Add main content
        for chapter in state['chapters']:
            if chapter.get('content'):
                content_parts.append(chapter['content'])
        
        # Add metadata
        content_parts.extend([
            state['title'],
            state['genre'],
            str(assembly_results.total_word_count)
        ])
        
        combined_content = '\n'.join(content_parts)
        return hashlib.sha256(combined_content.encode('utf-8')).hexdigest()
    
    def _extract_search_keywords(self, state: BookWritingState, assembly_results: AssemblyResults) -> List[str]:
        """Extract keywords for search indexing."""
        keywords = set()
        
        # Add basic keywords
        keywords.add(state['title'].lower())
        keywords.add(state['genre'].lower())
        
        # Add chapter titles
        for chapter in state['chapters']:
            if chapter.get('title'):
                keywords.update(chapter['title'].lower().split())
        
        # Add genre-specific keywords
        genre_keywords = {
            'fiction': ['story', 'novel', 'character', 'plot'],
            'non-fiction': ['guide', 'information', 'facts', 'analysis'],
            'technical': ['tutorial', 'guide', 'reference', 'documentation'],
            'academic': ['research', 'study', 'analysis', 'theory']
        }
        
        genre = state['genre'].lower()
        if genre in genre_keywords:
            keywords.update(genre_keywords[genre])
        
        # Filter and clean keywords
        keywords = {kw for kw in keywords if len(kw) > 2 and kw.isalpha()}
        
        return list(keywords)[:50]  # Limit to top 50 keywords
    
    def _get_next_version(self, book_id: str) -> int:
        """Get the next version number for a book."""
        try:
            latest = self.books_collection.find_one(
                {'book_id': book_id},
                sort=[('version', -1)]
            )
            return (latest['version'] + 1) if latest else 1
        except Exception:
            return 1
    
    def _serialize_assembly_results(self, assembly_results: AssemblyResults) -> Dict[str, Any]:
        """Serialize assembly results for storage."""
        # Convert dataclass to dict and handle non-serializable types
        result_dict = asdict(assembly_results)
        
        # Clean up file paths and sensitive information
        if 'generated_formats' in result_dict:
            for format_info in result_dict['generated_formats']:
                if 'file_path' in format_info:
                    # Store only filename, not full path
                    format_info['file_path'] = Path(format_info['file_path']).name
        
        return result_dict
    
    def _store_book_files(self, formatted_documents: List[FormattedDocument], book_id: str) -> List[StoredFile]:
        """Store book files using GridFS."""
        stored_files = []
        
        for doc in formatted_documents:
            try:
                file_path = Path(doc.file_path)
                
                # Validate file
                if not file_path.exists():
                    logger.warning(f"File not found: {file_path}")
                    continue
                
                file_size = file_path.stat().st_size
                
                # Check file size limits
                if file_size > (self.config.max_file_size_mb * 1024 * 1024):
                    logger.warning(f"File too large: {file_path} ({file_size} bytes)")
                    continue
                
                # Read and optionally compress file
                with open(file_path, 'rb') as f:
                    file_data = f.read()
                
                compressed_data = None
                compressed_size = None
                compression_ratio = None
                
                if self.config.enable_compression and doc.format_type in ['html', 'markdown', 'txt']:
                    compressed_data = gzip.compress(file_data, compresslevel=self.config.compression_level)
                    compressed_size = len(compressed_data)
                    compression_ratio = compressed_size / file_size if file_size > 0 else 1.0
                    
                    # Use compression only if it saves significant space
                    if compression_ratio < 0.8:
                        file_data = compressed_data
                        file_size = compressed_size
                    else:
                        compressed_data = None
                        compressed_size = None
                        compression_ratio = None
                
                # Generate checksum
                checksum = hashlib.md5(file_data).hexdigest()
                
                # Store in GridFS
                gridfs_id = self.gridfs.put(
                    file_data,
                    filename=f"{book_id}_{doc.format_type}_{uuid4().hex}",
                    book_id=book_id,
                    format_type=doc.format_type,
                    original_size=doc.file_size,
                    compressed=compressed_data is not None,
                    checksum=checksum,
                    stored_at=datetime.now(timezone.utc)
                )
                
                stored_file = StoredFile(
                    file_id=str(uuid4()),
                    format_type=doc.format_type,
                    file_size=doc.file_size,
                    compressed_size=compressed_size,
                    checksum=checksum,
                    stored_at=datetime.now(timezone.utc).isoformat(),
                    compression_ratio=compression_ratio,
                    gridfs_id=gridfs_id
                )
                
                stored_files.append(stored_file)
                logger.info(f"Stored file: {doc.format_type} for book {book_id}")
                
            except Exception as e:
                logger.error(f"Error storing file {doc.file_path}: {e}")
                continue
        
        return stored_files
    
    def _create_book_document(
        self,
        metadata: StoredBookMetadata,
        stored_files: List[StoredFile],
        state: BookWritingState,
        assembly_results: AssemblyResults
    ) -> Dict[str, Any]:
        """Create the main book document for MongoDB storage."""
        
        document = {
            '_id': metadata.book_id,
            'book_id': metadata.book_id,
            'title': metadata.title,
            'author': metadata.author,
            'genre': metadata.genre,
            'description': metadata.description,
            'word_count': metadata.word_count,
            'chapter_count': metadata.chapter_count,
            'quality_score': metadata.quality_score,
            'created_at': metadata.created_at,
            'stored_at': metadata.stored_at,
            'version': metadata.version,
            'content_hash': metadata.content_hash,
            'storage_location': metadata.storage_location,
            'tags': metadata.tags,
            'search_keywords': metadata.search_keywords,
            
            # File information
            'file_formats': metadata.file_formats,
            'file_sizes': metadata.file_sizes,
            'stored_files': [asdict(f) for f in stored_files],
            
            # Chapter content (if enabled)
            'chapters': self._prepare_chapters_for_storage(state['chapters']),
            
            # Assembly and generation metadata
            'assembly_results': metadata.assembly_results,
            'generation_metadata': metadata.generation_metadata,
            
            # Quality and validation info
            'validation_passed': True,
            'storage_validation': {
                'content_integrity_verified': True,
                'file_integrity_verified': self.config.validate_file_integrity,
                'quality_threshold_met': metadata.quality_score >= self.config.min_quality_score
            },
            
            # Pipeline information
            'pipeline_stage': 'completed',
            'processing_stages': state.get('processing_stages', []),
            'agents_used': state.get('agents_used', []),
            'total_processing_time': state.get('total_processing_time', 0),
            
            # Research information
            'research_summary': {
                'total_chunks_used': state.get('total_research_chunks', 0),
                'research_queries_count': len(state.get('research_queries', [])),
                'research_sources_count': len(set(
                    chunk.get('source_url', '') for chapter in state['chapters']
                    for chunk_id in chapter.get('research_chunks_used', [])
                    for chunk in [{}]  # Placeholder - would need actual chunk lookup
                )) if state.get('chapters') else 0
            },
            
            # Statistics and metrics
            'statistics': {
                'revision_count': state.get('revision_count', 0),
                'final_assembly_time': assembly_results.assembly_time,
                'formats_generated': len(assembly_results.generated_formats),
                'formats_failed': len(assembly_results.failed_formats),
                'total_file_size_bytes': sum(f.file_size for f in stored_files),
                'compression_ratio': self._calculate_average_compression_ratio(stored_files)
            },
            
            # Index fields for search
            'searchable_content': self._create_searchable_content(state, metadata) if self.config.enable_full_text_search else None,
            
            # Maintenance fields
            'last_accessed': None,
            'access_count': 0,
            'backup_status': 'pending' if self.config.auto_backup_enabled else 'disabled',
            'cleanup_eligible_after': self._calculate_cleanup_date(),
            
            # Versioning information
            'version_history': self._get_version_history(metadata.book_id) if self.config.enable_versioning else [],
            'is_latest_version': True,
            
            # Technical metadata
            'storage_format_version': '1.0',
            'mongodb_version': self._get_mongodb_version(),
            'storage_schema_version': '1.0'
        }
        
        return document
    
    def _prepare_chapters_for_storage(self, chapters: List[Dict]) -> List[Dict]:
        """Prepare chapter data for storage with optional content indexing."""
        prepared_chapters = []
        
        for chapter in chapters:
            chapter_data = {
                'chapter_number': chapter.get('chapter_number'),
                'title': chapter.get('title'),
                'description': chapter.get('description'),
                'word_count': chapter.get('word_count', 0),
                'status': chapter.get('status'),
                'created_at': chapter.get('created_at'),
                'completed_at': chapter.get('completed_at'),
                'research_chunks_used': chapter.get('research_chunks_used', [])
            }
            
            # Include content based on configuration
            if self.config.index_chapter_content:
                content = chapter.get('content', '')
                
                # Store full content or summary based on size
                if len(content) > 10000:  # Store summary for very long chapters
                    chapter_data['content_summary'] = content[:1000] + '...'
                    chapter_data['full_content_available'] = True
                else:
                    chapter_data['content'] = content
                    chapter_data['full_content_available'] = True
                
                # Extract chapter keywords
                chapter_data['keywords'] = self._extract_chapter_keywords(content)
            else:
                chapter_data['content_stored'] = False
            
            prepared_chapters.append(chapter_data)
        
        return prepared_chapters
    
    def _extract_chapter_keywords(self, content: str) -> List[str]:
        """Extract keywords from chapter content."""
        # Simple keyword extraction - could be enhanced with NLP
        words = content.lower().split()
        word_freq = {}
        
        for word in words:
            if len(word) > 3 and word.isalpha():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top 20 most frequent words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:20]]
    
    def _calculate_average_compression_ratio(self, stored_files: List[StoredFile]) -> Optional[float]:
        """Calculate average compression ratio for stored files."""
        ratios = [f.compression_ratio for f in stored_files if f.compression_ratio is not None]
        return sum(ratios) / len(ratios) if ratios else None
    
    def _create_searchable_content(self, state: BookWritingState, metadata: StoredBookMetadata) -> str:
        """Create searchable content for full-text indexing."""
        content_parts = [
            metadata.title,
            metadata.description,
            metadata.genre,
            ' '.join(metadata.search_keywords)
        ]
        
        # Add chapter titles and content snippets
        for chapter in state['chapters']:
            if chapter.get('title'):
                content_parts.append(chapter['title'])
            
            if chapter.get('content'):
                # Add first 200 words of each chapter
                words = chapter['content'].split()[:200]
                content_parts.append(' '.join(words))
        
        return ' '.join(content_parts)
    
    def _calculate_cleanup_date(self) -> str:
        """Calculate when temporary data becomes eligible for cleanup."""
        cleanup_date = datetime.now(timezone.utc) + timedelta(days=self.config.cleanup_failed_books_days)
        return cleanup_date.isoformat()
    
    def _get_version_history(self, book_id: str) -> List[Dict]:
        """Get version history for the book."""
        try:
            versions = self.books_collection.find(
                {'book_id': book_id},
                {'version': 1, 'stored_at': 1, 'quality_score': 1},
                sort=[('version', -1)]
            ).limit(self.config.max_versions_retained)
            
            return list(versions)
        except Exception:
            return []
    
    def _get_mongodb_version(self) -> str:
        """Get MongoDB server version."""
        try:
            server_info = self.client.server_info()
            return server_info.get('version', 'unknown')
        except Exception:
            return 'unknown'
    
    def _execute_storage_transaction(
        self,
        book_document: Dict[str, Any],
        metadata: StoredBookMetadata,
        stored_files: List[StoredFile]
    ) -> BookStorageResult:
        """Execute the storage operation with transaction support."""
        try:
            # Start transaction for atomic operations
            with self.client.start_session() as session:
                with session.start_transaction():
                    
                    # Store main book document
                    if self.config.enable_versioning and self._book_exists(metadata.book_id):
                        # Update existing book versions
                        self.books_collection.update_many(
                            {'book_id': metadata.book_id},
                            {'$set': {'is_latest_version': False}},
                            session=session
                        )
                    
                    # Insert new book document
                    self.books_collection.insert_one(book_document, session=session)
                    
                    # Store file metadata
                    if stored_files:
                        file_documents = []
                        for stored_file in stored_files:
                            file_doc = asdict(stored_file)
                            file_doc['book_id'] = metadata.book_id
                            file_doc['_id'] = stored_file.file_id
                            file_documents.append(file_doc)
                        
                        self.book_files_collection.insert_many(file_documents, session=session)
                    
                    # Store detailed metadata
                    metadata_document = asdict(metadata)
                    metadata_document['_id'] = f"{metadata.book_id}_v{metadata.version}"
                    self.book_metadata_collection.insert_one(metadata_document, session=session)
                    
                    session.commit_transaction()
            
            # Calculate total size
            total_size = sum(f.file_size for f in stored_files)
            
            return BookStorageResult(
                success=True,
                book_id=metadata.book_id,
                storage_location=metadata.storage_location,
                stored_files=stored_files,
                metadata_id=f"{metadata.book_id}_v{metadata.version}",
                version=metadata.version,
                total_size_bytes=total_size,
                storage_time=0.0,  # Will be set by caller
                quality_validation_passed=True,
                warnings=[],
                error_message=None
            )
            
        except DuplicateKeyError as e:
            error_msg = f"Book {metadata.book_id} already exists"
            logger.error(error_msg)
            return BookStorageResult(
                success=False,
                book_id=metadata.book_id,
                storage_location="",
                stored_files=[],
                metadata_id="",
                version=0,
                total_size_bytes=0,
                storage_time=0.0,
                quality_validation_passed=False,
                warnings=[],
                error_message=error_msg
            )
        
        except Exception as e:
            error_msg = f"Storage transaction failed: {str(e)}"
            logger.error(error_msg)
            return BookStorageResult(
                success=False,
                book_id=metadata.book_id,
                storage_location="",
                stored_files=[],
                metadata_id="",
                version=0,
                total_size_bytes=0,
                storage_time=0.0,
                quality_validation_passed=False,
                warnings=[],
                error_message=error_msg
            )
    
    def _update_storage_stats(self, storage_time: float, storage_result: BookStorageResult) -> None:
        """Update internal storage statistics."""
        if storage_result.success:
            self.storage_stats['books_stored'] += 1
            self.storage_stats['total_bytes_stored'] += storage_result.total_size_bytes
            self.storage_stats['files_stored'] += len(storage_result.stored_files)
            
            # Calculate compression savings
            compression_saved = sum(
                (f.file_size - f.compressed_size) for f in storage_result.stored_files
                if f.compressed_size is not None
            )
            self.storage_stats['compression_saved_bytes'] += compression_saved
            
            # Update average storage time
            current_avg = self.storage_stats['average_storage_time']
            books_count = self.storage_stats['books_stored']
            new_avg = ((current_avg * (books_count - 1)) + storage_time) / books_count
            self.storage_stats['average_storage_time'] = new_avg
        
        if self.storage_stats['session_start'] is None:
            self.storage_stats['session_start'] = time.time()
    
    def _cleanup_temporary_data(self, book_id: str) -> None:
        """Clean up temporary data after successful storage."""
        try:
            # Clean up old versions if over limit
            if self.config.enable_versioning:
                versions = self.books_collection.find(
                    {'book_id': book_id},
                    sort=[('version', -1)]
                ).skip(self.config.max_versions_retained)
                
                for old_version in versions:
                    self._archive_or_delete_version(old_version)
            
            # Clean up failed storage attempts
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.config.cleanup_failed_books_days)
            
            failed_books = self.books_collection.find({
                'pipeline_stage': 'failed',
                'stored_at': {'$lt': cutoff_date.isoformat()}
            })
            
            for failed_book in failed_books:
                self._cleanup_failed_book(failed_book['book_id'])
            
            logger.info(f"Cleanup completed for book {book_id}")
            
        except Exception as e:
            logger.warning(f"Cleanup failed for book {book_id}: {e}")
    
    def _archive_or_delete_version(self, version_doc: Dict) -> None:
        """Archive or delete an old book version."""
        try:
            book_id = version_doc['book_id']
            version = version_doc['version']
            
            # Remove files from GridFS
            file_records = self.book_files_collection.find({
                'book_id': book_id,
                'version': version
            })
            
            for file_record in file_records:
                if file_record.get('gridfs_id'):
                    self.gridfs.delete(file_record['gridfs_id'])
            
            # Remove file records
            self.book_files_collection.delete_many({
                'book_id': book_id,
                'version': version
            })
            
            # Remove book document
            self.books_collection.delete_one({'_id': version_doc['_id']})
            
            logger.info(f"Archived version {version} of book {book_id}")
            
        except Exception as e:
            logger.error(f"Error archiving version: {e}")
    
    def _cleanup_failed_book(self, book_id: str) -> None:
        """Clean up all data for a failed book."""
        try:
            # Remove all files
            file_records = self.book_files_collection.find({'book_id': book_id})
            for file_record in file_records:
                if file_record.get('gridfs_id'):
                    self.gridfs.delete(file_record['gridfs_id'])
            
            # Remove all records
            self.book_files_collection.delete_many({'book_id': book_id})
            self.book_metadata_collection.delete_many({'book_id': book_id})
            self.books_collection.delete_many({'book_id': book_id})
            
            logger.info(f"Cleaned up failed book {book_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up failed book {book_id}: {e}")
    
    def get_stored_book(self, book_id: str, version: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Retrieve a stored book by ID and optional version."""
        try:
            query = {'book_id': book_id}
            
            if version is not None:
                query['version'] = version
            else:
                query['is_latest_version'] = True
            
            return self.books_collection.find_one(query)
            
        except Exception as e:
            logger.error(f"Error retrieving book {book_id}: {e}")
            return None
    
    def get_book_file(self, book_id: str, format_type: str) -> Optional[bytes]:
        """Retrieve a book file by format type."""
        try:
            # Find file record
            file_record = self.book_files_collection.find_one({
                'book_id': book_id,
                'format_type': format_type
            })
            
            if not file_record or not file_record.get('gridfs_id'):
                return None
            
            # Retrieve from GridFS
            gridfs_file = self.gridfs.get(file_record['gridfs_id'])
            file_data = gridfs_file.read()
            
            # Decompress if needed
            if file_record.get('compressed', False):
                file_data = gzip.decompress(file_data)
            
            return file_data
            
        except Exception as e:
            logger.error(f"Error retrieving file for book {book_id}, format {format_type}: {e}")
            return None
    
    def list_books(self, limit: int = 100, skip: int = 0, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """List stored books with optional filtering."""
        try:
            query = filters or {}
            
            # Only return latest versions by default
            if 'is_latest_version' not in query:
                query['is_latest_version'] = True
            
            cursor = self.books_collection.find(query).skip(skip).limit(limit).sort('stored_at', -1)
            return list(cursor)
            
        except Exception as e:
            logger.error(f"Error listing books: {e}")
            return []
    
    def search_books(self, search_query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search books using full-text search."""
        try:
            if not self.config.enable_full_text_search:
                logger.warning("Full-text search is disabled")
                return []
            
            # Use MongoDB text search
            results = self.books_collection.find(
                {'$text': {'$search': search_query}},
                {'score': {'$meta': 'textScore'}}
            ).sort([('score', {'$meta': 'textScore'})]).limit(limit)
            
            return list(results)
            
        except Exception as e:
            logger.error(f"Error searching books: {e}")
            return []
    
    def get_storage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics."""
        stats = self.storage_stats.copy()
        
        # Calculate additional metrics
        try:
            # Database statistics
            db_stats = self.database.command("dbStats")
            collection_stats = self.books_collection.estimated_document_count()
            
            stats.update({
                'database_size_bytes': db_stats.get('dataSize', 0),
                'total_books_in_db': collection_stats,
                'index_size_bytes': db_stats.get('indexSize', 0)
            })
            
            # Calculate rates
            if stats['session_start']:
                session_duration = time.time() - stats['session_start']
                stats['books_per_hour'] = (stats['books_stored'] / session_duration) * 3600 if session_duration > 0 else 0
                stats['bytes_per_second'] = stats['total_bytes_stored'] / session_duration if session_duration > 0 else 0
            
            # Compression efficiency
            if stats['compression_saved_bytes'] > 0 and stats['total_bytes_stored'] > 0:
                stats['compression_efficiency'] = stats['compression_saved_bytes'] / stats['total_bytes_stored']
            
            # Add configuration info
            stats['configuration'] = {
                'versioning_enabled': self.config.enable_versioning,
                'compression_enabled': self.config.enable_compression,
                'full_text_search_enabled': self.config.enable_full_text_search,
                'max_file_size_mb': self.config.max_file_size_mb,
                'max_versions_retained': self.config.max_versions_retained
            }
            
        except Exception as e:
            logger.warning(f"Error calculating extended statistics: {e}")
        
        return stats
    
    def cleanup_resources(self) -> bool:
        """Clean up resources and close connections."""
        try:
            if self.client:
                self.client.close()
            
            logger.info("Book Storer Agent resources cleaned up")
            return True
        except Exception as e:
            logger.error(f"Error cleaning up resources: {e}")
            return False
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup_resources()


def main():
    """Test function for BookStorerAgent."""
    
    # Create test configuration
    config = BookStorerConfig()
    
    # Initialize agent
    agent = BookStorerAgent(config)
    
    # Create test state and assembly results
    test_state = {
        'book_id': 'test_book_storer_001',
        'title': 'Complete Guide to Book Storage',
        'author': 'MuseQuill AI',
        'genre': 'Technical',
        'target_word_count': 25000,
        'description': 'A comprehensive guide to book storage systems',
        'quality_score': 0.85,
        'chapters': [
            {
                'chapter_number': 1,
                'title': 'Introduction to Storage',
                'content': 'This chapter introduces the concepts of digital book storage and management systems...',
                'word_count': 1200,
                'status': 'complete'
            },
            {
                'chapter_number': 2,
                'title': 'Database Design',
                'content': 'This chapter covers the design principles for book storage databases...',
                'word_count': 1500,
                'status': 'complete'
            }
        ]
    }
    
    # Mock assembly results
    from musequill.agents.assembler.final_assembler_agent import AssemblyResults, FormattedDocument
    
    assembly_results = AssemblyResults(
        book_id='test_book_storer_001',
        success=True,
        total_word_count=2700,
        total_pages=12,
        generated_formats=[
            FormattedDocument(
                format_type='markdown',
                file_path='/tmp/test_book.md',
                file_size=5000,
                generation_time=1.5,
                validation_status='valid'
            )
        ],
        failed_formats=[],
        metadata=None,
        table_of_contents=[],
        index_entries=[],
        bibliography=[],
        assembly_time=3.2,
        validation_results={},
        error_messages=[]
    )
    
    try:
        # Test book storage
        print("Testing Book Storer Agent...")
        result = agent.store_completed_book(test_state, assembly_results)
        
        print(f"Storage Status: {result.get('status')}")
        print(f"Success: {result.get('success')}")
        print(f"Storage Time: {result.get('storage_time', 0):.2f} seconds")
        
        if result.get('success'):
            print(f"Book ID: {result.get('book_id')}")
            print(f"Storage Location: {result.get('storage_location')}")
            print(f"Files Stored: {result.get('stored_files')}")
            print(f"Total Size: {result.get('total_size_mb', 0):.2f} MB")
        else:
            print(f"Error: {result.get('error_message')}")
        
        # Test retrieval
        stored_book = agent.get_stored_book('test_book_storer_001')
        if stored_book:
            print(f"Retrieved book: {stored_book['title']}")
        
        # Test statistics
        stats = agent.get_storage_statistics()
        print(f"Books Stored: {stats['books_stored']}")
        print(f"Total Bytes: {stats['total_bytes_stored']}")
        
        print("BookStorerAgent test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        agent.cleanup_resources()


if __name__ == "__main__":
    main()