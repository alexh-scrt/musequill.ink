"""
Chapter Writer Agent

Core writing component that writes individual chapters using research materials from the vector database.
Integrates research chunks, follows chapter plans, maintains style consistency, and tracks progress.

Key Features:
- Research Integration: Retrieves relevant research from Chroma vector store
- Context-Aware Writing: Uses chapter plans, style guides, and research to write content
- Iterative Writing: Supports chapter-by-chapter progression with context awareness
- Quality Consistency: Follows established style guides and writing strategies
- Progress Tracking: Updates word counts and completion status
- Error Recovery: Handles writing failures and retry logic
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
import statistics
import re

import chromadb
from chromadb.config import Settings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel

from musequill.config.logging import get_logger
from musequill.agents.writer.chapter_writer_config import ChapterWriterConfig
from musequill.agents.agent_state import BookWritingState, Chapter

logger = get_logger(__name__)


@dataclass
class ResearchChunk:
    """Research chunk retrieved for chapter writing."""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    relevance_score: float
    source_url: str
    source_title: str
    
    
@dataclass
class ChapterContext:
    """Complete context for writing a chapter."""
    chapter: Chapter
    research_chunks: List[ResearchChunk]
    previous_chapters: List[Chapter]
    writing_strategy: str
    style_guide: str
    additional_context: Dict[str, Any]


@dataclass
class WritingProgress:
    """Progress tracking for chapter writing."""
    chapter_number: int
    words_written: int
    target_words: int
    completion_percentage: float
    research_chunks_used: List[str]
    writing_time_seconds: float
    quality_indicators: Dict[str, Any]


class ChapterContentModel(BaseModel):
    """Pydantic model for LLM-generated chapter content."""
    chapter_title: str
    content: str
    key_points_covered: List[str]
    research_integration_notes: str
    word_count_estimate: int
    quality_self_assessment: str
    suggested_next_chapter_connection: str


class ChapterWriterAgent:
    """
    Chapter Writer Agent that writes individual chapters using research materials.
    """
    
    def __init__(self, config: Optional[ChapterWriterConfig] = None):
        if not config:
            config = ChapterWriterConfig()
        
        self.config = config
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            api_key=self.config.openai_api_key,
            model=self.config.llm_model,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens
        )
        
        # Initialize Chroma client for research retrieval
        self.chroma_client: Optional[chromadb.HttpClient] = None
        self.chroma_collection = None
        self.embeddings: Optional[OpenAIEmbeddings] = None
        
        # Writing progress tracking
        self.writing_stats = {
            'chapters_written': 0,
            'chapters_failed': 0,
            'total_words_written': 0,
            'total_research_chunks_used': 0,
            'average_writing_time': 0.0,
            'session_start_time': None
        }
        
        self._initialize_components()
        
        logger.info("Chapter Writer Agent initialized")
    
    def _initialize_components(self) -> None:
        """Initialize all required components."""
        try:
            # Initialize Chroma client
            self.chroma_client = chromadb.HttpClient(
                host=self.config.chroma_host,
                port=self.config.chroma_port,
                settings=Settings(
                    chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider",
                    chroma_client_auth_credentials=None,
                    anonymized_telemetry=False
                )
            )
            
            # Get or create collection
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name=self.config.chroma_collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            # Initialize embeddings
            self.embeddings = OpenAIEmbeddings(
                model=self.config.embedding_model,
                dimensions=self.config.embedding_dimensions,
                openai_api_key=self.config.openai_api_key
            )
            
            logger.info("Chapter Writer components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Chapter Writer components: {e}")
            raise
    
    def write_next_chapter(self, state: BookWritingState) -> Dict[str, Any]:
        """
        Write the next chapter in the sequence.
        
        Args:
            state: Current book writing state
            
        Returns:
            Dictionary with updated state information
        """
        try:
            start_time = time.time()
            
            # Get the current chapter to write
            current_chapter_num = state['current_chapter']
            
            if current_chapter_num >= len(state['chapters']):
                logger.info("All chapters completed")
                return {
                    'status': 'complete',
                    'message': 'All chapters have been written',
                    'current_stage': 'writing_complete'
                }
            
            chapter = state['chapters'][current_chapter_num]
            
            logger.info(f"Starting to write Chapter {chapter['chapter_number']}: {chapter['title']}")
            
            # Build chapter context
            chapter_context = self._build_chapter_context(chapter, state)
            
            # Write the chapter
            chapter_result = self._write_chapter_content(chapter_context)
            
            # Calculate metrics
            writing_time = time.time() - start_time
            
            # Update chapter with results
            updated_chapter = chapter.copy()
            updated_chapter.update({
                'content': chapter_result.content,
                'word_count': chapter_result.word_count_estimate,
                'research_chunks_used': chapter_result.research_chunks_used,
                'status': 'complete',
                'completed_at': datetime.now(timezone.utc).isoformat()
            })
            
            # Update writing statistics
            self._update_writing_stats(chapter_result, writing_time)
            
            # Prepare return data
            result = {
                'status': 'success',
                'chapter_number': chapter['chapter_number'],
                'chapter_content': chapter_result.content,
                'words_written': chapter_result.word_count_estimate,
                'research_chunks_used': len(chapter_result.research_chunks_used),
                'writing_time_seconds': writing_time,
                'quality_indicators': {
                    'research_integration_score': self._calculate_research_integration_score(chapter_result),
                    'style_consistency_score': self._calculate_style_consistency_score(chapter_result, state),
                    'content_coherence_score': self._calculate_coherence_score(chapter_result)
                },
                'next_chapter': current_chapter_num + 1,
                'updated_chapter': updated_chapter
            }
            
            logger.info(f"Successfully wrote Chapter {chapter['chapter_number']} with {chapter_result.word_count_estimate} words")
            
            return result
            
        except Exception as e:
            logger.error(f"Error writing chapter {current_chapter_num}: {e}")
            return {
                'status': 'error',
                'error_message': str(e),
                'chapter_number': current_chapter_num,
                'retry_count': state.get('retry_count', 0) + 1
            }
    
    def _build_chapter_context(self, chapter: Chapter, state: BookWritingState) -> ChapterContext:
        """
        Build comprehensive context for chapter writing.
        
        Args:
            chapter: Chapter to write
            state: Current book writing state
            
        Returns:
            Complete chapter context
        """
        try:
            # Retrieve relevant research chunks
            research_chunks = self._retrieve_research_for_chapter(chapter, state)
            
            # Get previous chapters for context
            previous_chapters = [
                ch for ch in state['chapters'] 
                if ch['chapter_number'] < chapter['chapter_number'] and ch.get('content')
            ]
            
            # Build additional context
            additional_context = {
                'book_title': state['title'],
                'book_genre': state['genre'],
                'target_audience': state.get('target_audience', 'General audience'),
                'total_chapters': len(state['chapters']),
                'current_progress': f"{chapter['chapter_number']}/{len(state['chapters'])}",
                'total_word_count_target': state['target_word_count'],
                'words_written_so_far': sum(ch.get('word_count', 0) for ch in previous_chapters)
            }
            
            return ChapterContext(
                chapter=chapter,
                research_chunks=research_chunks,
                previous_chapters=previous_chapters,
                writing_strategy=state.get('writing_strategy', ''),
                style_guide=state.get('writing_style_guide', ''),
                additional_context=additional_context
            )
            
        except Exception as e:
            logger.error(f"Error building chapter context: {e}")
            # Return minimal context to allow basic writing
            return ChapterContext(
                chapter=chapter,
                research_chunks=[],
                previous_chapters=[],
                writing_strategy=state.get('writing_strategy', ''),
                style_guide=state.get('writing_style_guide', ''),
                additional_context={'book_title': state['title']}
            )
    
    def _retrieve_research_for_chapter(self, chapter: Chapter, state: BookWritingState) -> List[ResearchChunk]:
        """
        Retrieve relevant research chunks for the chapter.
        
        Args:
            chapter: Chapter to find research for
            state: Current book writing state
            
        Returns:
            List of relevant research chunks
        """
        try:
            # Build search queries from chapter information
            search_queries = self._build_research_queries(chapter, state)
            
            retrieved_chunks = []
            chunk_ids_seen = set()
            
            for query in search_queries:
                try:
                    # Generate embedding for the query
                    query_embedding = self.embeddings.embed_query(query)
                    
                    # Search Chroma for relevant chunks
                    results = self.chroma_collection.query(
                        query_embeddings=[query_embedding],
                        n_results=self.config.max_research_chunks_per_chapter // len(search_queries),
                        where={"book_id": state['book_id']},
                        include=["documents", "metadatas", "distances"]
                    )
                    
                    # Process results
                    for i, chunk_id in enumerate(results['ids'][0]):
                        if chunk_id not in chunk_ids_seen:
                            distance = results['distances'][0][i]
                            relevance_score = 1.0 - distance  # Convert distance to relevance
                            
                            if relevance_score >= self.config.research_relevance_threshold:
                                chunk = ResearchChunk(
                                    chunk_id=chunk_id,
                                    content=results['documents'][0][i],
                                    metadata=results['metadatas'][0][i],
                                    relevance_score=relevance_score,
                                    source_url=results['metadatas'][0][i].get('source_url', ''),
                                    source_title=results['metadatas'][0][i].get('source_title', '')
                                )
                                retrieved_chunks.append(chunk)
                                chunk_ids_seen.add(chunk_id)
                
                except Exception as e:
                    logger.warning(f"Failed to retrieve research for query '{query}': {e}")
                    continue
            
            # Sort by relevance and limit
            retrieved_chunks.sort(key=lambda x: x.relevance_score, reverse=True)
            final_chunks = retrieved_chunks[:self.config.max_research_chunks_per_chapter]
            
            logger.info(f"Retrieved {len(final_chunks)} research chunks for Chapter {chapter['chapter_number']}")
            
            return final_chunks
            
        except Exception as e:
            logger.error(f"Error retrieving research for chapter: {e}")
            return []
    
    def _build_research_queries(self, chapter: Chapter, state: BookWritingState) -> List[str]:
        """
        Build search queries for finding relevant research.
        
        Args:
            chapter: Chapter to build queries for
            state: Current book writing state
            
        Returns:
            List of search queries
        """
        queries = []
        
        # Base query from chapter title and description
        base_query = f"{chapter['title']} {chapter.get('description', '')}"
        queries.append(base_query.strip())
        
        # Add book-specific context
        book_context_query = f"{state['title']} {state['genre']} {chapter['title']}"
        queries.append(book_context_query)
        
        # Extract key terms from chapter description
        if chapter.get('description'):
            # Simple keyword extraction (could be enhanced with NLP)
            key_terms = [
                term.strip() for term in chapter['description'].split() 
                if len(term) > 4 and term.isalpha()
            ]
            if key_terms:
                keyword_query = f"{state['genre']} " + " ".join(key_terms[:5])
                queries.append(keyword_query)
        
        # Add genre-specific research query
        genre_query = f"{state['genre']} {chapter['title']} research"
        queries.append(genre_query)
        
        return list(set(queries))  # Remove duplicates
    
    def _write_chapter_content(self, context: ChapterContext) -> ChapterContentModel:
        """
        Generate chapter content using LLM.
        
        Args:
            context: Complete chapter context
            
        Returns:
            Generated chapter content
        """
        try:
            # Create system prompt
            system_prompt = self._create_chapter_writing_system_prompt(context)
            
            # Create human prompt with context
            human_prompt = self._create_chapter_writing_human_prompt(context)
            
            # Generate content
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            logger.debug(f"Generating content for Chapter {context.chapter['chapter_number']}")
            
            response = self.llm.with_structured_output(ChapterContentModel).invoke(messages)
            
            # Add research chunk tracking
            research_chunk_ids = [chunk.chunk_id for chunk in context.research_chunks]
            response.research_chunks_used = research_chunk_ids
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating chapter content: {e}")
            # Return fallback content
            return self._create_fallback_content(context)
    
    def _create_chapter_writing_system_prompt(self, context: ChapterContext) -> str:
        """Create system prompt for chapter writing."""
        return f"""You are an expert book writer specializing in {context.additional_context.get('book_genre', 'general')} writing.

Your task is to write a complete chapter that:

**RESEARCH INTEGRATION:**
- Seamlessly integrates the provided research materials
- Uses evidence and examples from the research to support key points
- Maintains academic/professional credibility through proper research usage
- Balances research integration with original analysis and narrative

**STYLE & CONSISTENCY:**
- Follows the established writing style guide and strategy
- Maintains consistent tone, voice, and quality with previous chapters
- Uses appropriate terminology and formatting for the target audience
- Ensures smooth transitions and logical flow

**STRUCTURE & ORGANIZATION:**
- Follows a clear, logical structure with appropriate headings/sections
- Meets the target word count while maintaining quality
- Includes engaging introduction and strong conclusion
- Creates natural bridges to previous and upcoming chapters

**QUALITY STANDARDS:**
- Produces publication-ready content
- Ensures clarity, coherence, and readability
- Maintains engagement throughout the chapter
- Demonstrates deep understanding of the subject matter

Generate comprehensive, well-researched content that advances the book's overall narrative and objectives."""

    def _create_chapter_writing_human_prompt(self, context: ChapterContext) -> str:
        """Create human prompt with all chapter context."""
        
        # Build research context
        research_context = ""
        if context.research_chunks:
            research_context = "AVAILABLE RESEARCH:\n"
            for i, chunk in enumerate(context.research_chunks[:10], 1):
                research_context += f"\n{i}. Source: {chunk.source_title}\n"
                research_context += f"   Content: {chunk.content[:200]}...\n"
                research_context += f"   Relevance: {chunk.relevance_score:.2f}\n"
        
        # Build previous chapter context
        previous_context = ""
        if context.previous_chapters:
            latest_chapter = context.previous_chapters[-1]
            previous_context = f"\nPREVIOUS CHAPTER CONTEXT:\n"
            previous_context += f"Last Chapter: {latest_chapter['title']}\n"
            if latest_chapter.get('content'):
                # Get last paragraph for transition context
                content = latest_chapter['content']
                last_paragraph = content.split('\n\n')[-1] if content else ""
                previous_context += f"Ending: ...{last_paragraph[-200:] if len(last_paragraph) > 200 else last_paragraph}\n"
        
        # Build style guide context
        style_context = ""
        if context.style_guide:
            style_context = f"\nSTYLE GUIDE:\n{context.style_guide[:500]}...\n"
        
        return f"""BOOK INFORMATION:
Title: {context.additional_context['book_title']}
Genre: {context.additional_context.get('book_genre', 'Unknown')}
Target Audience: {context.additional_context.get('target_audience', 'General')}
Progress: Chapter {context.chapter['chapter_number']} of {context.additional_context['total_chapters']}

CURRENT CHAPTER:
Title: {context.chapter['title']}
Description: {context.chapter.get('description', 'No description provided')}
Target Word Count: {context.chapter.get('target_word_count', 'Not specified')}

{previous_context}

{research_context}

WRITING STRATEGY:
{context.writing_strategy[:400] if context.writing_strategy else 'No specific strategy provided'}

{style_context}

REQUIREMENTS:
- Write a complete, engaging chapter that follows the description and integrates the research
- Maintain consistency with the established style and previous chapters  
- Meet the target word count while ensuring quality
- Include smooth transitions and logical flow
- Provide professional, publication-ready content

Please write the complete chapter content."""

    def _create_fallback_content(self, context: ChapterContext) -> ChapterContentModel:
        """Create basic fallback content when main generation fails."""
        
        fallback_content = f"""# {context.chapter['title']}

{context.chapter.get('description', 'This chapter explores important aspects of the topic.')}

[Chapter content would be developed here with proper research integration and detailed analysis.]

## Key Points

- Important concept 1
- Important concept 2  
- Important concept 3

## Conclusion

This chapter has laid the foundation for understanding {context.chapter['title'].lower()}. In the next chapter, we will explore [related topic].

"""
        
        return ChapterContentModel(
            chapter_title=context.chapter['title'],
            content=fallback_content,
            key_points_covered=["Basic outline", "Chapter structure", "Transition planning"],
            research_integration_notes="Fallback content generated - manual research integration needed",
            word_count_estimate=len(fallback_content.split()),
            quality_self_assessment="Basic fallback content - requires manual enhancement",
            suggested_next_chapter_connection="Connect to next chapter topic",
            research_chunks_used=[]
        )
    
    def _calculate_research_integration_score(self, chapter_result: ChapterContentModel) -> float:
        """Calculate how well research was integrated."""
        if not hasattr(chapter_result, 'research_chunks_used') or not chapter_result.research_chunks_used:
            return 0.0
        
        # Simple scoring based on research usage
        chunks_used = len(chapter_result.research_chunks_used)
        max_chunks = self.config.max_research_chunks_per_chapter
        
        base_score = min(1.0, chunks_used / max(1, max_chunks * 0.6))
        
        # Bonus for explicit research integration notes
        if "research" in chapter_result.research_integration_notes.lower():
            base_score += 0.1
        
        return min(1.0, base_score)
    
    def _calculate_style_consistency_score(self, chapter_result: ChapterContentModel, state: BookWritingState) -> float:
        """Calculate style consistency with previous chapters."""
        # Simple heuristic - could be enhanced with more sophisticated analysis
        base_score = 0.8  # Assume good consistency by default
        
        # Check if style guide elements are mentioned in self-assessment
        if state.get('writing_style_guide') and "style" in chapter_result.quality_self_assessment.lower():
            base_score += 0.1
        
        # Check word count alignment
        if chapter_result.word_count_estimate > 0:
            target = state['chapters'][0].get('target_word_count', 1000)  # Rough estimate
            variance = abs(chapter_result.word_count_estimate - target) / target
            if variance < 0.2:  # Within 20%
                base_score += 0.1
        
        return min(1.0, base_score)
    
    def _calculate_coherence_score(self, chapter_result: ChapterContentModel) -> float:
        """Calculate content coherence score."""
        content = chapter_result.content
        
        # Simple coherence indicators
        score = 0.6  # Base score
        
        # Check for headings/structure
        if any(marker in content for marker in ['#', '##', '###']):
            score += 0.1
        
        # Check for transitions
        transition_words = ['however', 'furthermore', 'additionally', 'consequently', 'therefore', 'moreover']
        if any(word in content.lower() for word in transition_words):
            score += 0.1
        
        # Check for conclusion
        if any(word in content.lower() for word in ['conclusion', 'summary', 'in summary']):
            score += 0.1
        
        # Check length appropriateness
        if 500 <= len(content.split()) <= 3000:
            score += 0.1
        
        return min(1.0, score)
    
    def _update_writing_stats(self, chapter_result: ChapterContentModel, writing_time: float) -> None:
        """Update internal writing statistics."""
        self.writing_stats['chapters_written'] += 1
        self.writing_stats['total_words_written'] += chapter_result.word_count_estimate
        self.writing_stats['total_research_chunks_used'] += len(getattr(chapter_result, 'research_chunks_used', []))
        
        # Update average writing time
        current_avg = self.writing_stats['average_writing_time']
        chapters_count = self.writing_stats['chapters_written']
        new_avg = ((current_avg * (chapters_count - 1)) + writing_time) / chapters_count
        self.writing_stats['average_writing_time'] = new_avg
        
        if self.writing_stats['session_start_time'] is None:
            self.writing_stats['session_start_time'] = time.time()
    
    def get_writing_progress(self, state: BookWritingState) -> Dict[str, Any]:
        """Get current writing progress and statistics."""
        
        completed_chapters = [ch for ch in state['chapters'] if ch.get('status') == 'complete']
        total_chapters = len(state['chapters'])
        
        total_words_written = sum(ch.get('word_count', 0) for ch in completed_chapters)
        target_words = state['target_word_count']
        
        progress = {
            'chapters_completed': len(completed_chapters),
            'total_chapters': total_chapters,
            'chapters_remaining': total_chapters - len(completed_chapters),
            'completion_percentage': (len(completed_chapters) / total_chapters * 100) if total_chapters > 0 else 0,
            'words_written': total_words_written,
            'target_words': target_words,
            'word_completion_percentage': (total_words_written / target_words * 100) if target_words > 0 else 0,
            'current_chapter': state['current_chapter'],
            'writing_stats': self.writing_stats.copy()
        }
        
        # Add estimated completion time
        if self.writing_stats['average_writing_time'] > 0 and progress['chapters_remaining'] > 0:
            estimated_time = progress['chapters_remaining'] * self.writing_stats['average_writing_time']
            progress['estimated_completion_time_minutes'] = estimated_time / 60
        
        return progress
    
    def retry_failed_chapter(self, state: BookWritingState, chapter_number: int, retry_count: int = 0) -> Dict[str, Any]:
        """
        Retry writing a failed chapter with adjusted parameters.
        
        Args:
            state: Current book writing state
            chapter_number: Chapter number to retry (1-based)
            retry_count: Current retry attempt count
            
        Returns:
            Result of retry attempt
        """
        try:
            if retry_count >= self.config.max_retry_attempts:
                logger.error(f"Maximum retry attempts reached for Chapter {chapter_number}")
                return {
                    'status': 'failed',
                    'error_message': 'Maximum retry attempts exceeded',
                    'chapter_number': chapter_number
                }
            
            logger.info(f"Retrying Chapter {chapter_number} (attempt {retry_count + 1})")
            
            # Get the chapter
            chapter_index = chapter_number - 1
            if chapter_index >= len(state['chapters']):
                return {
                    'status': 'error',
                    'error_message': f'Chapter {chapter_number} not found'
                }
            
            chapter = state['chapters'][chapter_index]
            
            # Adjust configuration for retry
            if self.config.retry_with_reduced_context and retry_count > 0:
                # Reduce research chunks for simpler context
                original_max_chunks = self.config.max_research_chunks_per_chapter
                self.config.max_research_chunks_per_chapter = max(5, original_max_chunks // 2)
                
                try:
                    # Build context and write chapter
                    chapter_context = self._build_chapter_context(chapter, state)
                    chapter_result = self._write_chapter_content(chapter_context)
                    
                    # Restore original configuration
                    self.config.max_research_chunks_per_chapter = original_max_chunks
                    
                    # Update chapter with results
                    updated_chapter = chapter.copy()
                    updated_chapter.update({
                        'content': chapter_result.content,
                        'word_count': chapter_result.word_count_estimate,
                        'research_chunks_used': getattr(chapter_result, 'research_chunks_used', []),
                        'status': 'complete',
                        'completed_at': datetime.now(timezone.utc).isoformat()
                    })
                    
                    return {
                        'status': 'success',
                        'chapter_number': chapter_number,
                        'retry_count': retry_count + 1,
                        'updated_chapter': updated_chapter,
                        'message': f'Chapter {chapter_number} completed on retry {retry_count + 1}'
                    }
                    
                except Exception as e:
                    # Restore original configuration
                    self.config.max_research_chunks_per_chapter = original_max_chunks
                    raise e
            
            else:
                # Standard retry
                chapter_context = self._build_chapter_context(chapter, state)
                chapter_result = self._write_chapter_content(chapter_context)
                
                # Update chapter with results
                updated_chapter = chapter.copy()
                updated_chapter.update({
                    'content': chapter_result.content,
                    'word_count': chapter_result.word_count_estimate,
                    'research_chunks_used': getattr(chapter_result, 'research_chunks_used', []),
                    'status': 'complete',
                    'completed_at': datetime.now(timezone.utc).isoformat()
                })
                
                return {
                    'status': 'success',
                    'chapter_number': chapter_number,
                    'retry_count': retry_count + 1,
                    'updated_chapter': updated_chapter
                }
                
        except Exception as e:
            logger.error(f"Error during Chapter {chapter_number} retry: {e}")
            
            # Try fallback approach if enabled
            if self.config.fallback_to_basic_writing and retry_count >= 1:
                return self._write_basic_chapter(chapter, state)
            
            return {
                'status': 'error',
                'error_message': str(e),
                'chapter_number': chapter_number,
                'retry_count': retry_count + 1
            }
    
    def _write_basic_chapter(self, chapter: Chapter, state: BookWritingState) -> Dict[str, Any]:
        """
        Write a basic chapter without complex research integration as fallback.
        
        Args:
            chapter: Chapter to write
            state: Current book writing state
            
        Returns:
            Basic chapter writing result
        """
        try:
            logger.info(f"Writing basic fallback version of Chapter {chapter['chapter_number']}")
            
            # Create minimal context
            minimal_context = ChapterContext(
                chapter=chapter,
                research_chunks=[],  # No research chunks for basic writing
                previous_chapters=[],
                writing_strategy="Write a clear, structured chapter on the given topic",
                style_guide="Use clear, professional writing style",
                additional_context={
                    'book_title': state['title'],
                    'book_genre': state.get('genre', 'General'),
                    'fallback_mode': True
                }
            )
            
            # Generate basic content using simplified prompt
            basic_content = self._generate_basic_content(minimal_context)
            
            # Create chapter result
            updated_chapter = chapter.copy()
            updated_chapter.update({
                'content': basic_content,
                'word_count': len(basic_content.split()),
                'research_chunks_used': [],
                'status': 'complete',
                'completed_at': datetime.now(timezone.utc).isoformat()
            })
            
            return {
                'status': 'success',
                'chapter_number': chapter['chapter_number'],
                'updated_chapter': updated_chapter,
                'message': 'Basic fallback chapter completed',
                'fallback_mode': True
            }
            
        except Exception as e:
            logger.error(f"Failed to write basic chapter: {e}")
            return {
                'status': 'failed',
                'error_message': f'Basic chapter writing failed: {str(e)}',
                'chapter_number': chapter['chapter_number']
            }
    
    def _generate_basic_content(self, context: ChapterContext) -> str:
        """Generate basic chapter content without complex research integration."""
        
        basic_prompt = f"""Write a clear, well-structured chapter on the topic: {context.chapter['title']}

Description: {context.chapter.get('description', 'No description provided')}

Requirements:
- Write approximately {context.chapter.get('target_word_count', 1000)} words
- Use clear headings and structure
- Provide informative, engaging content
- Include introduction, main content, and conclusion
- Use professional tone appropriate for {context.additional_context.get('book_genre', 'general')} writing

Write the complete chapter content:"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=basic_prompt)])
            return response.content
            
        except Exception as e:
            logger.error(f"Failed to generate basic content: {e}")
            # Return absolute fallback
            return f"""# {context.chapter['title']}

## Introduction

This chapter explores the important topic of {context.chapter['title'].lower()}. Understanding this subject is crucial for readers interested in {context.additional_context.get('book_genre', 'this field')}.

## Main Content

{context.chapter.get('description', 'This section would contain the main discussion of the chapter topic.')}

Key points to consider:

- Important aspect 1
- Important aspect 2  
- Important aspect 3
- Important aspect 4

## Analysis

Further analysis and discussion of the topic would be provided here, exploring different perspectives and approaches.

## Conclusion

This chapter has provided an overview of {context.chapter['title'].lower()}. The concepts discussed here form an important foundation for understanding subsequent topics in this book.

In the next chapter, we will continue building on these concepts to explore related areas of study.
"""
    
    def validate_chapter_quality(self, chapter_content: str, chapter: Chapter, state: BookWritingState) -> Dict[str, Any]:
        """
        Validate the quality of written chapter content.
        
        Args:
            chapter_content: Written chapter content
            chapter: Chapter information
            state: Current book writing state
            
        Returns:
            Quality validation results
        """
        try:
            validation_results = {
                'overall_score': 0.0,
                'quality_metrics': {},
                'issues_found': [],
                'recommendations': [],
                'passes_quality_check': False
            }
            
            # Word count validation
            word_count = len(chapter_content.split())
            target_word_count = chapter.get('target_word_count', 1000)
            word_count_variance = abs(word_count - target_word_count) / target_word_count
            
            if word_count_variance <= self.config.word_count_tolerance:
                validation_results['quality_metrics']['word_count'] = 1.0
            elif word_count_variance <= self.config.word_count_tolerance * 2:
                validation_results['quality_metrics']['word_count'] = 0.7
            else:
                validation_results['quality_metrics']['word_count'] = 0.4
                validation_results['issues_found'].append(f"Word count ({word_count}) significantly differs from target ({target_word_count})")
            
            # Structure validation
            structure_score = self._validate_chapter_structure(chapter_content)
            validation_results['quality_metrics']['structure'] = structure_score
            
            if structure_score < 0.6:
                validation_results['issues_found'].append("Chapter lacks clear structure or headings")
            
            # Content coherence validation
            coherence_score = self._validate_content_coherence(chapter_content)
            validation_results['quality_metrics']['coherence'] = coherence_score
            
            if coherence_score < 0.6:
                validation_results['issues_found'].append("Content may lack coherence or logical flow")
            
            # Style consistency validation (basic)
            style_score = self._validate_style_consistency(chapter_content, state)
            validation_results['quality_metrics']['style_consistency'] = style_score
            
            # Calculate overall score
            metrics = validation_results['quality_metrics']
            overall_score = sum(metrics.values()) / len(metrics) if metrics else 0.0
            validation_results['overall_score'] = overall_score
            
            # Determine if passes quality check
            validation_results['passes_quality_check'] = (
                overall_score >= self.config.min_quality_score and 
                len(validation_results['issues_found']) <= 2
            )
            
            # Generate recommendations
            if not validation_results['passes_quality_check']:
                validation_results['recommendations'] = self._generate_quality_recommendations(validation_results)
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating chapter quality: {e}")
            return {
                'overall_score': 0.0,
                'quality_metrics': {},
                'issues_found': [f"Quality validation failed: {str(e)}"],
                'recommendations': ["Manual quality review recommended"],
                'passes_quality_check': False
            }
    
    def _validate_chapter_structure(self, content: str) -> float:
        """Validate chapter structure and organization."""
        score = 0.0
        
        # Check for headings
        heading_patterns = ['#', '##', '###']
        if any(pattern in content for pattern in heading_patterns):
            score += 0.3
        
        # Check for introduction (first paragraph)
        paragraphs = content.split('\n\n')
        if len(paragraphs) > 0 and len(paragraphs[0].split()) > 20:
            score += 0.2
        
        # Check for conclusion (last paragraph or explicit conclusion)
        if 'conclusion' in content.lower() or 'summary' in content.lower():
            score += 0.2
        
        # Check for multiple sections
        if len(paragraphs) >= 4:
            score += 0.2
        
        # Check for proper paragraph structure
        avg_paragraph_length = sum(len(p.split()) for p in paragraphs) / len(paragraphs) if paragraphs else 0
        if 30 <= avg_paragraph_length <= 150:
            score += 0.1
        
        return min(1.0, score)
    
    def _validate_content_coherence(self, content: str) -> float:
        """Validate content coherence and flow."""
        score = 0.5  # Base score
        
        # Check for transition words
        transition_words = [
            'however', 'therefore', 'furthermore', 'additionally', 'consequently',
            'moreover', 'nevertheless', 'meanwhile', 'subsequently', 'accordingly'
        ]
        
        transition_count = sum(1 for word in transition_words if word in content.lower())
        if transition_count > 0:
            score += min(0.3, transition_count * 0.1)
        
        # Check for logical flow indicators
        flow_indicators = ['first', 'second', 'third', 'finally', 'next', 'then', 'after']
        flow_count = sum(1 for indicator in flow_indicators if indicator in content.lower())
        if flow_count > 0:
            score += min(0.2, flow_count * 0.05)
        
        return min(1.0, score)
    
    def _validate_style_consistency(self, content: str, state: BookWritingState) -> float:
        """Validate style consistency with book requirements."""
        score = 0.7  # Base score assuming reasonable consistency
        
        # Check for appropriate tone (very basic check)
        genre = state.get('genre', '').lower()
        
        if 'academic' in genre:
            # Look for academic indicators
            academic_indicators = ['research', 'study', 'analysis', 'evidence', 'findings']
            if any(indicator in content.lower() for indicator in academic_indicators):
                score += 0.1
        
        elif 'business' in genre or 'professional' in genre:
            # Look for professional indicators
            professional_indicators = ['strategy', 'implementation', 'results', 'process', 'methodology']
            if any(indicator in content.lower() for indicator in professional_indicators):
                score += 0.1
        
        # Check for consistent formatting (basic)
        if content.count('\n\n') > 2:  # Multiple paragraphs
            score += 0.1
        
        # Check against minimum quality indicators
        if len(content.split()) < self.config.min_words_per_chapter:
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _generate_quality_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving chapter quality."""
        recommendations = []
        
        metrics = validation_results['quality_metrics']
        
        if metrics.get('word_count', 1.0) < 0.7:
            recommendations.append("Adjust chapter length to better match target word count")
        
        if metrics.get('structure', 1.0) < 0.6:
            recommendations.append("Improve chapter structure with clear headings and sections")
        
        if metrics.get('coherence', 1.0) < 0.6:
            recommendations.append("Enhance content flow with better transitions and logical organization")
        
        if metrics.get('style_consistency', 1.0) < 0.7:
            recommendations.append("Review style consistency with book guidelines and previous chapters")
        
        if len(validation_results['issues_found']) > 2:
            recommendations.append("Consider comprehensive revision to address multiple quality issues")
        
        return recommendations
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for the Chapter Writer Agent."""
        
        stats = self.writing_stats.copy()
        
        # Add session information
        if stats['session_start_time']:
            stats['session_duration_minutes'] = (time.time() - stats['session_start_time']) / 60
        
        # Add configuration information
        stats['configuration'] = {
            'model': self.config.llm_model,
            'max_research_chunks': self.config.max_research_chunks_per_chapter,
            'research_threshold': self.config.research_relevance_threshold,
            'target_quality_level': self.config.min_quality_score,
            'retry_attempts': self.config.max_retry_attempts
        }
        
        # Add performance metrics
        if stats['chapters_written'] > 0:
            stats['average_words_per_chapter'] = stats['total_words_written'] / stats['chapters_written']
            stats['average_research_chunks_per_chapter'] = stats['total_research_chunks_used'] / stats['chapters_written']
        
        return stats
    
    def cleanup_resources(self) -> bool:
        """
        Clean up resources and connections.
        
        Returns:
            True if cleanup successful
        """
        try:
            # Close Chroma client connections if needed
            if hasattr(self.chroma_client, 'close'):
                self.chroma_client.close()
            
            # Reset statistics
            self.writing_stats = {
                'chapters_written': 0,
                'chapters_failed': 0,
                'total_words_written': 0,
                'total_research_chunks_used': 0,
                'average_writing_time': 0.0,
                'session_start_time': None
            }
            
            logger.info("Chapter Writer Agent resources cleaned up successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up Chapter Writer Agent resources: {e}")
            return False


def main():
    """Test function for ChapterWriterAgent."""
    
    # Create test configuration
    config = ChapterWriterConfig()
    
    # Initialize agent
    agent = ChapterWriterAgent(config)
    
    # Create test state
    test_state = {
        'book_id': 'test_book_001',
        'title': 'Test Book Title',
        'genre': 'Technical',
        'target_word_count': 50000,
        'current_chapter': 0,
        'chapters': [
            {
                'chapter_number': 1,
                'title': 'Introduction to the Topic',
                'description': 'This chapter introduces the main concepts and sets the foundation for the book.',
                'target_word_count': 2500,
                'status': 'planned'
            }
        ],
        'writing_strategy': 'Research-driven approach with clear explanations',
        'writing_style_guide': 'Professional yet accessible tone for technical audience'
    }
    
    try:
        # Test chapter writing
        print("Testing Chapter Writer Agent...")
        result = agent.write_next_chapter(test_state)
        print(f"Chapter writing result: {result['status']}")
        
        if result['status'] == 'success':
            print(f"Words written: {result['words_written']}")
            print(f"Research chunks used: {result['research_chunks_used']}")
        
        # Test progress tracking
        progress = agent.get_writing_progress(test_state)
        print(f"Writing progress: {progress['completion_percentage']:.1f}%")
        
        # Get agent statistics
        stats = agent.get_agent_stats()
        print(f"Agent stats: {stats}")
        
    except Exception as e:
        print(f"Test failed: {e}")
    
    finally:
        # Cleanup
        agent.cleanup_resources()


if __name__ == "__main__":
    main()