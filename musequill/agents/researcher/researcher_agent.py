"""
Researcher Agent

Executes research queries using Tavily web search and stores results in Chroma vector database.
Processes and chunks content with quality filtering and deduplication.

Key Features:
- Web search using Tavily API with advanced search capabilities
- Content processing with intelligent chunking and quality filtering
- Vector storage in hosted Chroma database with comprehensive metadata
- Concurrent query processing with rate limiting and retry logic
- Content deduplication and similarity filtering
- Source quality assessment and domain filtering
- Comprehensive error handling and monitoring
"""

import asyncio
import hashlib
import re
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Set
from urllib.parse import urlparse
from uuid import uuid4
from dataclasses import dataclass
import traceback

import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tavily import TavilyClient
from difflib import SequenceMatcher

from musequill.config.logging import get_logger
from musequill.agents.researcher.researcher_agent_config import ResearcherConfig
from musequill.agents.agent_state import BookWritingState, ResearchQuery

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """Structured search result from Tavily."""
    url: str
    title: str
    content: str
    raw_content: str
    score: float
    published_date: Optional[str]
    domain: str
    query: str
    tavily_answer: Optional[str] = None


@dataclass
class ProcessedChunk:
    """Processed content chunk ready for vector storage."""
    chunk_id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    quality_score: float
    source_info: Dict[str, str]


@dataclass
class ResearchResults:
    """Complete research results for a query."""
    query: str
    search_results: List[SearchResult]
    processed_chunks: List[ProcessedChunk]
    total_chunks_stored: int
    total_sources: int
    quality_stats: Dict[str, Any]
    execution_time: float
    status: str
    error_message: Optional[str] = None


class ResearcherAgent:
    """
    Researcher Agent that executes research queries and stores results in vector database.
    """
    
    def __init__(self, config: Optional[ResearcherConfig] = None):
        if not config:
            config = ResearcherConfig()
        
        self.config = config
        
        # Initialize clients
        self.tavily_client: Optional[TavilyClient] = None
        self.chroma_client: Optional[chromadb.HttpClient] = None
        self.chroma_collection = None
        self.embeddings: Optional[OpenAIEmbeddings] = None
        self.text_splitter: Optional[RecursiveCharacterTextSplitter] = None
        
        # Content tracking for deduplication
        self.content_hashes: Set[str] = set()
        self.processed_urls: Set[str] = set()
        
        # Statistics tracking
        self.stats = {
            'queries_processed': 0,
            'queries_failed': 0,
            'total_chunks_stored': 0,
            'total_sources_processed': 0,
            'duplicate_content_filtered': 0,
            'low_quality_filtered': 0,
            'processing_start_time': None
        }
        
        self._initialize_components()
        
        logger.info("Researcher Agent initialized")
    
    def _initialize_components(self) -> None:
        """Initialize all required components."""
        try:
            # Initialize Tavily client
            if self.config.tavily_api_key:
                self.tavily_client = TavilyClient(api_key=self.config.tavily_api_key)
                logger.info("Tavily client initialized")
            else:
                logger.error("Tavily API key not provided")
                raise ValueError("Tavily API key is required")
            
            # Initialize Chroma client
            self.chroma_client = chromadb.HttpClient(
                host=self.config.chroma_host,
                port=self.config.chroma_port,
                settings=Settings(
                    chroma_server_authn_credentials=None,
                    chroma_server_authn_provider=None
                )
            )
            
            # Get or create collection
            try:
                self.chroma_collection = self.chroma_client.get_collection(
                    name=self.config.chroma_collection_name
                )
                logger.info(f"Using existing Chroma collection: {self.config.chroma_collection_name}")
            except Exception:
                # Create collection with metadata for filtering
                self.chroma_collection = self.chroma_client.create_collection(
                    name=self.config.chroma_collection_name,
                    metadata={
                        "description": "Research materials for book writing",
                        "embedding_model": self.config.embedding_model,
                        "created_at": datetime.now(timezone.utc).isoformat()
                    }
                )
                logger.info(f"Created new Chroma collection: {self.config.chroma_collection_name}")
            
            # Initialize OpenAI embeddings
            self.embeddings = OpenAIEmbeddings(
                api_key=self.config.openai_api_key,
                model=self.config.embedding_model,
                dimensions=self.config.embedding_dimensions
            )
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
            )
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def execute_research(self, state: BookWritingState) -> Dict[str, Any]:
        """
        Execute research for all pending queries in the state.
        
        Args:
            state: BookWritingState containing research queries
            
        Returns:
            Dictionary with research results and updated queries
        """
        try:
            logger.info(f"Starting research execution for book {state['book_id']}")
            self.stats['processing_start_time'] = time.time()
            
            # Filter pending queries
            pending_queries = [q for q in state['research_queries'] if q['status'] == 'pending']
            
            if not pending_queries:
                logger.warning(f"No pending research queries found for book {state['book_id']}")
                return {
                    'updated_queries': state['research_queries'],
                    'total_chunks': 0,
                    'total_sources': 0,
                    'stats': self.stats
                }
            
            logger.info(f"Processing {len(pending_queries)} research queries")
            
            # Execute research queries with concurrency control
            research_results = self._execute_queries_concurrently(pending_queries, state['book_id'])
            
            # Update query statuses
            updated_queries = self._update_query_statuses(state['research_queries'], research_results)
            
            # Calculate totals
            total_chunks = sum(result.total_chunks_stored for result in research_results.values())
            total_sources = sum(result.total_sources for result in research_results.values())
            
            # Update statistics
            self.stats['total_chunks_stored'] += total_chunks
            self.stats['total_sources_processed'] += total_sources
            
            execution_time = time.time() - self.stats['processing_start_time']
            
            logger.info(
                f"Research execution completed for book {state['book_id']}: "
                f"{total_chunks} chunks stored, {total_sources} sources processed, "
                f"execution time: {execution_time:.2f}s"
            )
            
            return {
                'updated_queries': updated_queries,
                'total_chunks': total_chunks,
                'total_sources': total_sources,
                'execution_time': execution_time,
                'stats': self.stats,
                'detailed_results': research_results
            }
            
        except Exception as e:
            logger.error(f"Error executing research for book {state['book_id']}: {e}")
            logger.error(traceback.format_exc())
            self.stats['queries_failed'] += len(pending_queries)
            
            # Return failure result
            return {
                'updated_queries': self._mark_queries_failed(state['research_queries'], str(e)),
                'total_chunks': 0,
                'total_sources': 0,
                'error': str(e),
                'stats': self.stats
            }
    
    def _execute_queries_concurrently(self, queries: List[ResearchQuery], book_id: str) -> Dict[str, ResearchResults]:
        """
        Execute research queries with controlled concurrency.
        
        Args:
            queries: List of research queries to execute
            book_id: Book identifier for metadata
            
        Returns:
            Dictionary mapping query text to ResearchResults
        """
        results = {}
        
        # Process queries in batches to respect concurrency limits
        batch_size = self.config.max_concurrent_queries
        
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            
            logger.info(f"Processing batch {i//batch_size + 1} with {len(batch)} queries")
            
            # Execute batch concurrently
            batch_results = asyncio.run(self._execute_query_batch(batch, book_id))
            results.update(batch_results)
            
            # Rate limiting between batches
            if i + batch_size < len(queries):
                time.sleep(self.config.rate_limit_delay)
        
        return results
    
    async def _execute_query_batch(self, queries: List[ResearchQuery], book_id: str) -> Dict[str, ResearchResults]:
        """Execute a batch of queries concurrently."""
        tasks = []
        
        for query in queries:
            task = asyncio.create_task(
                self._research_single_query(query, book_id)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        batch_results = {}
        for query, result in zip(queries, results):
            if isinstance(result, Exception):
                logger.error(f"Query '{query['query']}' failed: {result}")
                batch_results[query['query']] = ResearchResults(
                    query=query['query'],
                    search_results=[],
                    processed_chunks=[],
                    total_chunks_stored=0,
                    total_sources=0,
                    quality_stats={},
                    execution_time=0.0,
                    status='failed',
                    error_message=str(result)
                )
                self.stats['queries_failed'] += 1
            else:
                batch_results[query['query']] = result
                self.stats['queries_processed'] += 1
        
        return batch_results
    
    async def _research_single_query(self, query: ResearchQuery, book_id: str) -> ResearchResults:
        """
        Research a single query with retries.
        
        Args:
            query: Research query to execute
            book_id: Book identifier
            
        Returns:
            ResearchResults for the query
        """
        start_time = time.time()
        last_error = None
        
        for attempt in range(self.config.query_retry_attempts):
            try:
                logger.info(f"Executing query '{query['query']}' (attempt {attempt + 1})")
                
                # Perform web search
                search_results = await self._perform_web_search(query['query'])
                
                # Filter and validate results
                filtered_results = self._filter_search_results(search_results, query)
                
                # Process content and create chunks
                processed_chunks = await self._process_search_results(filtered_results, query, book_id)
                
                # Store chunks in vector database
                chunks_stored = await self._store_chunks_in_chroma(processed_chunks, book_id)
                
                # Calculate quality statistics
                quality_stats = self._calculate_quality_stats(processed_chunks)
                
                execution_time = time.time() - start_time
                
                result = ResearchResults(
                    query=query['query'],
                    search_results=filtered_results,
                    processed_chunks=processed_chunks,
                    total_chunks_stored=chunks_stored,
                    total_sources=len(filtered_results),
                    quality_stats=quality_stats,
                    execution_time=execution_time,
                    status='completed'
                )
                
                logger.info(
                    f"Query '{query['query']}' completed: {chunks_stored} chunks stored, "
                    f"{len(filtered_results)} sources processed in {execution_time:.2f}s"
                )
                
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(f"Query '{query['query']}' attempt {attempt + 1} failed: {e}")
                
                if attempt < self.config.query_retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay_seconds)
        
        # All attempts failed
        execution_time = time.time() - start_time
        logger.error(f"Query '{query['query']}' failed after {self.config.query_retry_attempts} attempts")
        
        return ResearchResults(
            query=query['query'],
            search_results=[],
            processed_chunks=[],
            total_chunks_stored=0,
            total_sources=0,
            quality_stats={},
            execution_time=execution_time,
            status='failed',
            error_message=str(last_error)
        )
    
    async def _perform_web_search(self, query: str) -> List[SearchResult]:
        """
        Perform web search using Tavily API.
        
        Args:
            query: Search query string
            
        Returns:
            List of SearchResult objects
        """
        try:
            # Execute search with Tavily
            search_response = self.tavily_client.search(
                query=query,
                search_depth=self.config.tavily_search_depth,
                max_results=self.config.tavily_max_results,
                include_answer=self.config.tavily_include_answer,
                include_raw_content=self.config.tavily_include_raw_content,
                include_images=self.config.tavily_include_images
            )
            
            # Process search results
            search_results = []
            tavily_answer = search_response.get('answer', '') if self.config.tavily_include_answer else None
            
            for result in search_response.get('results', []):
                # Extract domain
                domain = urlparse(result.get('url', '')).netloc.lower()
                
                search_result = SearchResult(
                    url=result.get('url', ''),
                    title=result.get('title', ''),
                    content=result.get('content', ''),
                    raw_content=result.get('raw_content', ''),
                    score=result.get('score', 0.0),
                    published_date=result.get('published_date'),
                    domain=domain,
                    query=query,
                    tavily_answer=tavily_answer
                )
                
                search_results.append(search_result)
            
            if self.config.log_search_results:
                logger.info(f"Search for '{query}' returned {len(search_results)} results")
            
            return search_results
            
        except Exception as e:
            logger.error(f"Web search failed for query '{query}': {e}")
            raise
    
    def _filter_search_results(self, results: List[SearchResult], query: ResearchQuery) -> List[SearchResult]:
        """
        Filter search results based on quality criteria.
        
        Args:
            results: List of search results
            query: Original research query
            
        Returns:
            Filtered list of search results
        """
        filtered_results = []
        
        for result in results:
            # Check minimum score threshold
            if result.score < self.config.min_source_score:
                continue
            
            # Check domain filtering
            if self._is_domain_blocked(result.domain):
                logger.debug(f"Blocked domain: {result.domain}")
                continue
            
            # Check for duplicate URLs
            if result.url in self.processed_urls:
                logger.debug(f"Duplicate URL filtered: {result.url}")
                continue
            
            # Check content quality
            if not self._assess_content_quality(result):
                logger.debug(f"Low quality content filtered: {result.url}")
                continue
            
            filtered_results.append(result)
            self.processed_urls.add(result.url)
        
        logger.info(f"Filtered {len(results)} results to {len(filtered_results)} high-quality sources")
        return filtered_results
    
    def _is_domain_blocked(self, domain: str) -> bool:
        """Check if domain is in blocked list."""
        for blocked in self.config.blocked_domains:
            if blocked.lower() in domain.lower():
                return True
        return False
    
    def _is_domain_trusted(self, domain: str) -> bool:
        """Check if domain is in trusted list."""
        for trusted in self.config.trusted_domains:
            if trusted.lower() in domain.lower():
                return True
        return False
    
    def _assess_content_quality(self, result: SearchResult) -> bool:
        """
        Assess content quality based on various criteria.
        
        Args:
            result: Search result to assess
            
        Returns:
            True if content meets quality standards
        """
        if not self.config.enable_content_filtering:
            return True
        
        # Use raw_content if available, otherwise fall back to content
        content = result.raw_content or result.content
        
        if not content or len(content.strip()) < self.config.min_chunk_size:
            return False
        
        # Check content length (not too short, not too long)
        if len(content) > self.config.max_content_length:
            content = content[:self.config.max_content_length]
        
        # Basic quality indicators
        quality_score = 0.0
        
        # Domain trust score
        if self._is_domain_trusted(result.domain):
            quality_score += 0.3
        
        # Tavily score contribution
        quality_score += result.score * 0.4
        
        # Content structure indicators
        sentence_count = len(re.findall(r'[.!?]+', content))
        if sentence_count > 3:  # Multiple sentences indicate structured content
            quality_score += 0.2
        
        # Presence of meaningful content (not just navigation/ads)
        meaningful_words = len(re.findall(r'\b[a-zA-Z]{4,}\b', content))
        if meaningful_words > 50:
            quality_score += 0.1
        
        return quality_score >= self.config.min_content_quality_score
    
    async def _process_search_results(
        self, 
        results: List[SearchResult], 
        query: ResearchQuery, 
        book_id: str
    ) -> List[ProcessedChunk]:
        """
        Process search results into chunks with embeddings.
        
        Args:
            results: Filtered search results
            query: Original research query
            book_id: Book identifier
            
        Returns:
            List of ProcessedChunk objects
        """
        processed_chunks = []
        
        for result in results:
            try:
                # Get content (prefer raw_content)
                content = result.raw_content or result.content
                if not content:
                    continue
                
                # Truncate if too long
                if len(content) > self.config.max_content_length:
                    content = content[:self.config.max_content_length]
                
                # Split into chunks
                text_chunks = self.text_splitter.split_text(content)
                
                for i, chunk_text in enumerate(text_chunks):
                    if len(chunk_text.strip()) < self.config.min_chunk_size:
                        continue
                    
                    # Check for content duplication
                    if self.config.filter_duplicate_content:
                        content_hash = self._get_content_hash(chunk_text)
                        if content_hash in self.content_hashes:
                            self.stats['duplicate_content_filtered'] += 1
                            continue
                        self.content_hashes.add(content_hash)
                    
                    # Generate embedding
                    embedding = await self.embeddings.aembed_query(chunk_text)
                    
                    # Create unique chunk ID
                    chunk_id = f"{book_id}_{query['query_type']}_{uuid4().hex[:12]}"
                    
                    # Create comprehensive metadata
                    metadata = {
                        'book_id': book_id,
                        'query': query['query'],
                        'query_type': query['query_type'],
                        'query_priority': query['priority'],
                        'source_url': result.url,
                        'source_title': result.title,
                        'source_domain': result.domain,
                        'source_score': result.score,
                        'chunk_index': i,
                        'chunk_size': len(chunk_text),
                        'published_date': result.published_date,
                        'processed_at': datetime.now(timezone.utc).isoformat(),
                        'tavily_answer': result.tavily_answer[:500] if result.tavily_answer else None,
                        'total_chunks_from_source': len(text_chunks)
                    }
                    
                    # Calculate quality score for this chunk
                    quality_score = self._calculate_chunk_quality_score(chunk_text, result, query)
                    
                    source_info = {
                        'url': result.url,
                        'title': result.title,
                        'domain': result.domain,
                        'tavily_score': result.score
                    }
                    
                    processed_chunk = ProcessedChunk(
                        chunk_id=chunk_id,
                        content=chunk_text,
                        embedding=embedding,
                        metadata=metadata,
                        quality_score=quality_score,
                        source_info=source_info
                    )
                    
                    processed_chunks.append(processed_chunk)
                    
                    if self.config.log_chunk_details:
                        logger.debug(f"Processed chunk {chunk_id}: {len(chunk_text)} chars, quality: {quality_score:.2f}")
                
            except Exception as e:
                logger.error(f"Error processing result from {result.url}: {e}")
                continue
        
        logger.info(f"Processed {len(processed_chunks)} chunks from {len(results)} search results")
        return processed_chunks
    
    def _get_content_hash(self, content: str) -> str:
        """Generate hash for content deduplication."""
        # Normalize content for hashing
        normalized = re.sub(r'\s+', ' ', content.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _calculate_chunk_quality_score(self, chunk_text: str, result: SearchResult, query: ResearchQuery) -> float:
        """
        Calculate quality score for a content chunk.
        
        Args:
            chunk_text: Text content of the chunk
            result: Source search result
            query: Original research query
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        score = 0.0
        
        # Base score from Tavily
        score += result.score * 0.3
        
        # Domain trust factor
        if self._is_domain_trusted(result.domain):
            score += 0.2
        
        # Content length factor (optimal range)
        text_length = len(chunk_text)
        if 200 <= text_length <= 800:
            score += 0.15
        elif 100 <= text_length <= 1200:
            score += 0.1
        
        # Query relevance (simple keyword matching)
        query_words = set(query['query'].lower().split())
        chunk_words = set(chunk_text.lower().split())
        relevance = len(query_words.intersection(chunk_words)) / len(query_words) if query_words else 0
        score += relevance * 0.2
        
        # Content structure indicators
        if len(re.findall(r'[.!?]', chunk_text)) >= 2:  # Multiple sentences
            score += 0.1
        
        # Avoid promotional/spammy content
        spam_indicators = ['click here', 'subscribe now', 'buy now', '!!!', 'free trial']
        if not any(indicator in chunk_text.lower() for indicator in spam_indicators):
            score += 0.05
        
        return min(1.0, score)  # Cap at 1.0
    
    async def _store_chunks_in_chroma(self, chunks: List[ProcessedChunk], book_id: str) -> int:
        """
        Store processed chunks in Chroma vector database.
        
        Args:
            chunks: List of processed chunks to store
            book_id: Book identifier
            
        Returns:
            Number of chunks successfully stored
        """
        if not chunks:
            return 0
        
        try:
            stored_count = 0
            
            # Process chunks in batches
            batch_size = self.config.batch_size
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                try:
                    # Prepare batch data for Chroma
                    ids = [chunk.chunk_id for chunk in batch]
                    documents = [chunk.content for chunk in batch]
                    embeddings = [chunk.embedding for chunk in batch]
                    metadatas = [chunk.metadata for chunk in batch]
                    
                    # Store batch in Chroma
                    self.chroma_collection.add(
                        ids=ids,
                        documents=documents,
                        embeddings=embeddings,
                        metadatas=metadatas
                    )
                    
                    stored_count += len(batch)
                    
                    logger.debug(f"Stored batch of {len(batch)} chunks in Chroma")
                    
                    # Small delay between batches to avoid overwhelming the server
                    if i + batch_size < len(chunks):
                        await asyncio.sleep(0.1)
                
                except Exception as e:
                    logger.error(f"Failed to store batch starting at index {i}: {e}")
                    # Continue with next batch rather than failing completely
                    continue
            
            logger.info(f"Successfully stored {stored_count}/{len(chunks)} chunks in Chroma for book {book_id}")
            return stored_count
            
        except Exception as e:
            logger.error(f"Error storing chunks in Chroma: {e}")
            return 0
    
    def _calculate_quality_stats(self, chunks: List[ProcessedChunk]) -> Dict[str, Any]:
        """Calculate quality statistics for processed chunks."""
        if not chunks:
            return {}
        
        quality_scores = [chunk.quality_score for chunk in chunks]
        chunk_sizes = [len(chunk.content) for chunk in chunks]
        
        # Source diversity
        unique_domains = len(set(chunk.source_info['domain'] for chunk in chunks))
        unique_urls = len(set(chunk.source_info['url'] for chunk in chunks))
        
        # Quality distribution
        high_quality = sum(1 for score in quality_scores if score >= 0.7)
        medium_quality = sum(1 for score in quality_scores if 0.4 <= score < 0.7)
        low_quality = sum(1 for score in quality_scores if score < 0.4)
        
        return {
            'total_chunks': len(chunks),
            'avg_quality_score': sum(quality_scores) / len(quality_scores),
            'min_quality_score': min(quality_scores),
            'max_quality_score': max(quality_scores),
            'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'unique_domains': unique_domains,
            'unique_sources': unique_urls,
            'quality_distribution': {
                'high_quality': high_quality,
                'medium_quality': medium_quality,
                'low_quality': low_quality
            }
        }
    
    def _update_query_statuses(
        self, 
        original_queries: List[ResearchQuery], 
        results: Dict[str, ResearchResults]
    ) -> List[ResearchQuery]:
        """Update query statuses based on research results."""
        updated_queries = []
        
        for query in original_queries:
            updated_query = query.copy()
            
            if query['query'] in results:
                result = results[query['query']]
                updated_query['status'] = result.status
                updated_query['results_count'] = result.total_chunks_stored
                
                # Add additional metadata
                updated_query['execution_time'] = result.execution_time
                updated_query['sources_processed'] = result.total_sources
                updated_query['quality_stats'] = result.quality_stats
                
                if result.error_message:
                    updated_query['error_message'] = result.error_message
            
            updated_queries.append(updated_query)
        
        return updated_queries
    
    def _mark_queries_failed(self, queries: List[ResearchQuery], error_message: str) -> List[ResearchQuery]:
        """Mark all queries as failed with error message."""
        updated_queries = []
        
        for query in queries:
            updated_query = query.copy()
            if updated_query['status'] == 'pending':
                updated_query['status'] = 'failed'
                updated_query['error_message'] = error_message
                updated_query['results_count'] = 0
            updated_queries.append(updated_query)
        
        return updated_queries
    
    def search_similar_content(
        self, 
        query_text: str, 
        book_id: str, 
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for similar content in the vector database.
        
        Args:
            query_text: Text to search for
            book_id: Book identifier to filter by
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of similar content chunks with metadata
        """
        try:
            # Generate embedding for query
            query_embedding = self.embeddings.embed_query(query_text)
            
            # Search in Chroma
            results = self.chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where={"book_id": book_id},
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            similar_chunks = []
            
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0] if results['metadatas'] else []
                distances = results['distances'][0] if results['distances'] else []
                
                for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
                    # Convert distance to similarity (assuming cosine distance)
                    similarity = 1 - distance if distance else 0
                    
                    if similarity >= similarity_threshold:
                        similar_chunks.append({
                            'content': doc,
                            'metadata': metadata,
                            'similarity_score': similarity,
                            'rank': i + 1
                        })
            
            logger.info(f"Found {len(similar_chunks)} similar chunks for query '{query_text}'")
            return similar_chunks
            
        except Exception as e:
            logger.error(f"Error searching similar content: {e}")
            return []
    
    def get_research_summary(self, book_id: str) -> Dict[str, Any]:
        """
        Get comprehensive research summary for a book.
        
        Args:
            book_id: Book identifier
            
        Returns:
            Research summary with statistics and metadata
        """
        try:
            # Query all chunks for this book
            results = self.chroma_collection.get(
                where={"book_id": book_id},
                include=["metadatas"]
            )
            
            if not results['metadatas']:
                return {
                    'book_id': book_id,
                    'total_chunks': 0,
                    'error': 'No research data found'
                }
            
            metadatas = results['metadatas']
            
            # Calculate summary statistics
            total_chunks = len(metadatas)
            
            # Query type distribution
            query_types = {}
            for metadata in metadatas:
                query_type = metadata.get('query_type', 'unknown')
                query_types[query_type] = query_types.get(query_type, 0) + 1
            
            # Source diversity
            unique_domains = len(set(m.get('source_domain', '') for m in metadatas))
            unique_sources = len(set(m.get('source_url', '') for m in metadatas))
            
            # Priority distribution
            priorities = {}
            for metadata in metadatas:
                priority = metadata.get('query_priority', 0)
                priorities[priority] = priorities.get(priority, 0) + 1
            
            # Time range
            processed_times = [m.get('processed_at') for m in metadatas if m.get('processed_at')]
            earliest = min(processed_times) if processed_times else None
            latest = max(processed_times) if processed_times else None
            
            summary = {
                'book_id': book_id,
                'total_chunks': total_chunks,
                'unique_sources': unique_sources,
                'unique_domains': unique_domains,
                'query_type_distribution': query_types,
                'priority_distribution': priorities,
                'research_period': {
                    'earliest': earliest,
                    'latest': latest
                },
                'avg_chunk_size': sum(m.get('chunk_size', 0) for m in metadatas) / total_chunks if total_chunks else 0
            }
            
            logger.info(f"Generated research summary for book {book_id}: {total_chunks} chunks from {unique_sources} sources")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating research summary for book {book_id}: {e}")
            return {
                'book_id': book_id,
                'error': str(e)
            }
    
    def cleanup_book_research(self, book_id: str) -> bool:
        """
        Clean up research data for a specific book.
        
        Args:
            book_id: Book identifier
            
        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            # Get all chunk IDs for this book
            results = self.chroma_collection.get(
                where={"book_id": book_id},
                include=["metadatas"]
            )
            
            if results['ids']:
                chunk_ids = results['ids']
                
                # Delete chunks in batches
                batch_size = self.config.batch_size
                deleted_count = 0
                
                for i in range(0, len(chunk_ids), batch_size):
                    batch_ids = chunk_ids[i:i + batch_size]
                    
                    try:
                        self.chroma_collection.delete(ids=batch_ids)
                        deleted_count += len(batch_ids)
                    except Exception as e:
                        logger.error(f"Failed to delete batch of chunks: {e}")
                
                logger.info(f"Cleaned up {deleted_count} research chunks for book {book_id}")
                return deleted_count == len(chunk_ids)
            
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up research for book {book_id}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics for the researcher agent."""
        current_stats = self.stats.copy()
        
        if current_stats['processing_start_time']:
            current_stats['total_processing_time'] = time.time() - current_stats['processing_start_time']
        
        # Add collection statistics if available
        try:
            collection_count = self.chroma_collection.count()
            current_stats['total_chunks_in_collection'] = collection_count
        except:
            current_stats['total_chunks_in_collection'] = 'unavailable'
        
        return current_stats


def main():
    """Test function for ResearcherAgent."""
    from musequill.agents.agent_state import BookWritingState, ProcessingStage, ResearchQuery
    from datetime import datetime, timezone
    
    print("Testing ResearcherAgent...")
    
    # Create test research queries
    test_queries = [
        ResearchQuery(
            query="artificial intelligence machine learning basics",
            priority=5,
            query_type="technical_details",
            status="pending",
            results_count=None,
            created_at=datetime.now(timezone.utc).isoformat()
        ),
        ResearchQuery(
            query="AI ethics current debates 2024",
            priority=4,
            query_type="expert_opinions",
            status="pending",
            results_count=None,
            created_at=datetime.now(timezone.utc).isoformat()
        )
    ]
    
    # Create test state
    test_state = BookWritingState(
        book_id="test_book_ai_research",
        orchestration_id="test_orch_456",
        thread_id="test_thread_789",
        title="AI Research Test Book",
        description="Testing research capabilities",
        genre="Technology",
        target_word_count=50000,
        target_audience="Technology professionals",
        author_preferences={},
        outline={},
        chapters=[],
        current_stage=ProcessingStage.RESEARCHING,
        processing_started_at=datetime.now(timezone.utc).isoformat(),
        processing_updated_at=datetime.now(timezone.utc).isoformat(),
        research_queries=test_queries,
        research_strategy="Test research strategy",
        total_research_chunks=0,
        research_completed_at=None,
        current_chapter=0,
        writing_strategy=None,
        writing_style_guide=None,
        total_word_count=0,
        writing_started_at=None,
        writing_completed_at=None,
        review_notes=None,
        revision_count=0,
        quality_score=None,
        errors=[],
        retry_count=0,
        last_error_at=None,
        progress_percentage=0.0,
        estimated_completion_time=None,
        final_book_content=None,
        metadata={}
    )
    
    try:
        # Create researcher agent
        researcher = ResearcherAgent()
        
        print("Executing research...")
        research_results = researcher.execute_research(test_state)
        
        print(f"\nResearch Results:")
        print(f"Total chunks stored: {research_results['total_chunks']}")
        print(f"Total sources processed: {research_results['total_sources']}")
        print(f"Execution time: {research_results.get('execution_time', 0):.2f}s")
        
        # Show updated query statuses
        print(f"\nQuery Results:")
        for query in research_results['updated_queries']:
            print(f"- {query['query'][:50]}...")
            print(f"  Status: {query['status']}, Results: {query.get('results_count', 0)}")
        
        # Get research summary
        print(f"\nResearch Summary:")
        summary = researcher.get_research_summary(test_state['book_id'])
        print(f"Total chunks: {summary.get('total_chunks', 0)}")
        print(f"Unique sources: {summary.get('unique_sources', 0)}")
        print(f"Query types: {list(summary.get('query_type_distribution', {}).keys())}")
        
        # Get agent statistics
        stats = researcher.get_stats()
        print(f"\nAgent Statistics:")
        print(f"Queries processed: {stats['queries_processed']}")
        print(f"Queries failed: {stats['queries_failed']}")
        print(f"Total chunks stored: {stats['total_chunks_stored']}")
        
        print("\nResearcherAgent test completed successfully!")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()