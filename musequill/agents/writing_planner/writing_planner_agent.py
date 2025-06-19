"""
Writing Planner Agent

Creates comprehensive writing strategies and detailed chapter plans based on completed research.
Analyzes research content, optimizes chapter structure, and generates style guides for consistent writing.

Key Features:
- Research-driven writing strategy development
- Intelligent chapter planning with research integration
- Comprehensive style guide generation
- Narrative flow optimization and chapter dependencies
- Research clustering and content organization
- Timeline and milestone planning
- Quality consistency frameworks
"""

import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import statistics
import numpy as np

import chromadb
from chromadb.config import Settings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel

from musequill.config.logging import get_logger
from musequill.agents.writing_planner.writing_planner_config import WritingPlannerConfig
from musequill.agents.agent_state import BookWritingState, Chapter

logger = get_logger(__name__)


@dataclass
class ResearchCluster:
    """Research content cluster for organization."""
    cluster_id: int
    theme: str
    chunks: List[Dict[str, Any]]
    keywords: List[str]
    relevance_score: float
    chapter_assignments: List[int]


class WritingPlanModel(BaseModel):
    """Pydantic model for LLM-generated writing plans."""
    writing_approach: str
    key_themes: List[str]
    narrative_structure: str
    target_audience_considerations: str
    tone_and_style: str
    chapter_flow_strategy: str
    research_integration_approach: str
    quality_objectives: str


class ChapterEnhancement(BaseModel):
    """Pydantic model for chapter enhancement."""
    enhanced_title: str
    detailed_description: str
    key_points: List[str]
    structure_outline: str
    research_requirements: List[str]
    dependencies: List[int]
    complexity_assessment: str
    estimated_pages: int


class StyleGuideModel(BaseModel):
    """Pydantic model for style guide generation."""
    voice_description: str
    tone_guidelines: str
    writing_style_rules: List[str]
    formatting_standards: List[str]
    terminology_definitions: Dict[str, str]
    consistency_requirements: List[str]
    chapter_structure_template: str


class WritingPlannerAgent:
    """
    Writing Planner Agent that creates comprehensive writing strategies and plans.
    """
    
    def __init__(self, config: Optional[WritingPlannerConfig] = None):
        if not config:
            config = WritingPlannerConfig()
        
        self.config = config
        
        # Initialize clients
        self.llm: Optional[ChatOpenAI] = None
        self.embeddings: Optional[OpenAIEmbeddings] = None
        self.chroma_client: Optional[chromadb.HttpClient] = None
        self.chroma_collection = None
        
        # Processing caches
        self.research_cache: Dict[str, Any] = {}
        self.analysis_cache: Dict[str, Any] = {}
        
        self._initialize_components()
        
        logger.info("Writing Planner Agent initialized")
    
    def _initialize_components(self) -> None:
        """Initialize all required components."""
        try:
            # Initialize LLM
            self.llm = ChatOpenAI(
                api_key=self.config.openai_api_key,
                model=self.config.llm_model,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens
            )
            
            # Initialize embeddings for research analysis
            self.embeddings = OpenAIEmbeddings(
                api_key=self.config.openai_api_key,
                model=self.config.embedding_model
            )
            
            # Initialize Chroma client
            self.chroma_client = chromadb.HttpClient(
                host=self.config.chroma_host,
                port=self.config.chroma_port,
                settings=Settings(
                    chroma_server_authn_credentials=None,
                    chroma_server_authn_provider=None
                )
            )
            
            # Get collection
            self.chroma_collection = self.chroma_client.get_collection(
                name=self.config.chroma_collection_name
            )
            
            logger.info("Writing planner components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize writing planner components: {e}")
            raise
    
    def create_writing_plan(self, state: BookWritingState) -> Dict[str, Any]:
        """
        Create comprehensive writing plan based on research and book requirements.
        
        Args:
            state: BookWritingState with completed research
            
        Returns:
            Dictionary containing writing strategy, enhanced chapters, and style guide
        """
        try:
            logger.info(f"Creating writing plan for book {state['book_id']}")
            start_time = time.time()
            
            # Analyze completed research
            research_analysis = self._analyze_research_content(state['book_id'])
            
            # Create research clusters for better organization
            research_clusters = None
            if self.config.enable_research_clustering and research_analysis['chunks']:
                research_clusters = self._create_research_clusters(research_analysis)
            
            # Generate comprehensive writing strategy
            writing_strategy = asyncio.run(
                self._generate_writing_strategy(state, research_analysis, research_clusters)
            )
            
            # Enhance chapter plans with research integration
            enhanced_chapters = asyncio.run(
                self._enhance_chapter_plans(state, research_analysis, research_clusters, writing_strategy)
            )
            
            # Generate detailed style guide
            style_guide = None
            if self.config.generate_style_guide:
                style_guide = asyncio.run(
                    self._generate_style_guide(state, writing_strategy, research_analysis)
                )
            
            # Optimize chapter flow and dependencies
            if self.config.enable_narrative_flow_analysis:
                enhanced_chapters = self._optimize_narrative_flow(enhanced_chapters, state)
            
            # Calculate writing timeline and milestones
            timeline_info = self._calculate_writing_timeline(enhanced_chapters, state)
            
            # Create comprehensive writing plan
            writing_plan = {
                'strategy': writing_strategy,
                'updated_chapters': enhanced_chapters,
                'style_guide': style_guide,
                'research_analysis': research_analysis,
                'research_clusters': research_clusters,
                'timeline': timeline_info,
                'started_at': datetime.now(timezone.utc).isoformat(),
                'metadata': {
                    'planning_depth': self.config.planning_depth,
                    'research_chunks_analyzed': len(research_analysis.get('chunks', [])),
                    'clusters_created': len(research_clusters) if research_clusters else 0,
                    'planning_time': time.time() - start_time
                }
            }
            
            execution_time = time.time() - start_time
            
            logger.info(
                f"Writing plan created for book {state['book_id']}: "
                f"{len(enhanced_chapters)} chapters planned, "
                f"execution time: {execution_time:.2f}s"
            )
            
            return writing_plan
            
        except Exception as e:
            logger.error(f"Error creating writing plan for book {state['book_id']}: {e}")
            # Return basic fallback plan
            return self._create_fallback_writing_plan(state)
    
    def _analyze_research_content(self, book_id: str) -> Dict[str, Any]:
        """
        Analyze research content to understand themes, coverage, and quality.
        
        Args:
            book_id: Book identifier
            
        Returns:
            Research analysis results
        """
        try:
            # Check cache first
            cache_key = f"research_analysis_{book_id}"
            if self.config.cache_research_analysis and cache_key in self.analysis_cache:
                logger.info(f"Using cached research analysis for book {book_id}")
                return self.analysis_cache[cache_key]
            
            # Retrieve research data from Chroma
            results = self.chroma_collection.get(
                where={"book_id": book_id},
                include=["documents", "metadatas", "embeddings"]
            )
            
            if not results['documents']:
                logger.warning(f"No research data found for book {book_id}")
                return {
                    'chunks': [],
                    'total_chunks': 0,
                    'themes': [],
                    'quality_distribution': {},
                    'category_distribution': {},
                    'source_analysis': {}
                }
            
            # Process research chunks
            chunks = []
            for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
                chunk_data = {
                    'content': doc,
                    'metadata': metadata,
                    'embedding': results['embeddings'][i] if results['embeddings'] else None,
                    'chunk_id': f"chunk_{i}"
                }
                chunks.append(chunk_data)
            
            # Limit analysis to manageable size
            if len(chunks) > self.config.max_research_sample_size:
                chunks = self._sample_research_for_analysis(chunks)
            
            # Analyze themes and topics
            themes = self._extract_research_themes(chunks)
            
            # Analyze quality distribution
            quality_distribution = self._analyze_quality_distribution(chunks)
            
            # Analyze category distribution
            category_distribution = self._analyze_category_distribution(chunks)
            
            # Analyze source diversity
            source_analysis = self._analyze_source_diversity(chunks)
            
            analysis_result = {
                'chunks': chunks,
                'total_chunks': len(chunks),
                'themes': themes,
                'quality_distribution': quality_distribution,
                'category_distribution': category_distribution,
                'source_analysis': source_analysis,
                'analysis_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Cache the analysis
            if self.config.cache_research_analysis:
                self.analysis_cache[cache_key] = analysis_result
            
            logger.info(
                f"Research analysis completed for book {book_id}: "
                f"{len(chunks)} chunks, {len(themes)} themes identified"
            )
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing research content for book {book_id}: {e}")
            return {
                'chunks': [],
                'total_chunks': 0,
                'themes': [],
                'quality_distribution': {},
                'category_distribution': {},
                'source_analysis': {},
                'error': str(e)
            }
    
    def _sample_research_for_analysis(self, chunks: List[Dict]) -> List[Dict]:
        """Sample research chunks for analysis using stratified sampling."""
        target_size = self.config.max_research_sample_size
        
        # Group by category
        category_groups = defaultdict(list)
        for chunk in chunks:
            category = chunk['metadata'].get('query_type', 'unknown')
            category_groups[category].append(chunk)
        
        # Sample from each category proportionally
        sampled_chunks = []
        total_categories = len(category_groups)
        chunks_per_category = max(1, target_size // total_categories)
        
        for category, chunks_in_category in category_groups.items():
            # Sort by quality/score if available
            sorted_chunks = sorted(
                chunks_in_category,
                key=lambda x: x['metadata'].get('source_score', 0.5),
                reverse=True
            )
            
            # Take top chunks from this category
            sample_size = min(chunks_per_category, len(sorted_chunks))
            sampled_chunks.extend(sorted_chunks[:sample_size])
        
        # If we need more chunks, add highest-quality remaining ones
        remaining_needed = target_size - len(sampled_chunks)
        if remaining_needed > 0:
            remaining_chunks = [chunk for chunk in chunks if chunk not in sampled_chunks]
            sorted_remaining = sorted(
                remaining_chunks,
                key=lambda x: x['metadata'].get('source_score', 0.5),
                reverse=True
            )
            sampled_chunks.extend(sorted_remaining[:remaining_needed])
        
        return sampled_chunks[:target_size]
    
    def _extract_research_themes(self, chunks: List[Dict]) -> List[str]:
        """Extract major themes from research content."""
        try:
            # Combine all content for theme extraction
            all_content = " ".join([chunk['content'][:500] for chunk in chunks])
            
            themes = set()
            
            # Extract from query metadata
            for chunk in chunks:
                query = chunk['metadata'].get('query', '')
                query_type = chunk['metadata'].get('query_type', '')
                
                # Add query keywords as themes
                query_words = [word.lower() for word in query.split() if len(word) > 3]
                themes.update(query_words[:3])
                
                # Add category as theme
                if query_type:
                    themes.add(query_type.replace('_', ' '))
            
            # Extract from content (simple approach)
            content_words = all_content.lower().split()
            word_freq = Counter(content_words)
            
            # Get most common meaningful words
            common_words = [word for word, count in word_freq.most_common(20) 
                          if len(word) > 4 and word.isalpha()]
            themes.update(common_words[:10])
            
            return sorted(list(themes))[:15]
            
        except Exception as e:
            logger.error(f"Error extracting research themes: {e}")
            return []
    
    def _analyze_quality_distribution(self, chunks: List[Dict]) -> Dict[str, Any]:
        """Analyze quality distribution of research chunks."""
        quality_scores = []
        source_scores = []
        
        for chunk in chunks:
            quality_score = chunk['metadata'].get('quality_score', 0.5)
            source_score = chunk['metadata'].get('source_score', 0.5)
            
            quality_scores.append(quality_score)
            source_scores.append(source_score)
        
        if not quality_scores:
            return {}
        
        return {
            'avg_quality': statistics.mean(quality_scores),
            'quality_std': statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0,
            'avg_source_score': statistics.mean(source_scores),
            'high_quality_count': sum(1 for score in quality_scores if score >= 0.7),
            'low_quality_count': sum(1 for score in quality_scores if score < 0.4),
            'quality_range': {
                'min': min(quality_scores),
                'max': max(quality_scores)
            }
        }
    
    def _analyze_category_distribution(self, chunks: List[Dict]) -> Dict[str, Any]:
        """Analyze distribution of research across categories."""
        category_counts = Counter()
        category_quality = defaultdict(list)
        
        for chunk in chunks:
            category = chunk['metadata'].get('query_type', 'unknown')
            quality = chunk['metadata'].get('source_score', 0.5)
            
            category_counts[category] += 1
            category_quality[category].append(quality)
        
        # Calculate average quality per category
        category_avg_quality = {}
        for category, qualities in category_quality.items():
            category_avg_quality[category] = statistics.mean(qualities)
        
        return {
            'category_counts': dict(category_counts),
            'category_quality': category_avg_quality,
            'most_researched': category_counts.most_common(1)[0] if category_counts else None,
            'least_researched': category_counts.most_common()[-1] if category_counts else None,
            'total_categories': len(category_counts)
        }
    
    def _analyze_source_diversity(self, chunks: List[Dict]) -> Dict[str, Any]:
        """Analyze diversity and quality of research sources."""
        domains = set()
        urls = set()
        domain_counts = Counter()
        
        for chunk in chunks:
            domain = chunk['metadata'].get('source_domain', '')
            url = chunk['metadata'].get('source_url', '')
            
            if domain:
                domains.add(domain)
                domain_counts[domain] += 1
            if url:
                urls.add(url)
        
        return {
            'unique_domains': len(domains),
            'unique_urls': len(urls),
            'domain_distribution': dict(domain_counts.most_common()),
            'diversity_ratio': len(domains) / max(len(urls), 1),
            'avg_chunks_per_domain': len(chunks) / max(len(domains), 1)
        }
    
    def _create_research_clusters(self, research_analysis: Dict[str, Any]) -> List[ResearchCluster]:
        """Create thematic clusters from research content using embeddings."""
        if not self.config.enable_research_clustering:
            return []
        
        try:
            chunks = research_analysis['chunks']
            if len(chunks) < 3:
                return []
            
            # Extract embeddings
            embeddings = []
            valid_chunks = []
            
            for chunk in chunks:
                if chunk.get('embedding'):
                    embeddings.append(chunk['embedding'])
                    valid_chunks.append(chunk)
            
            if len(embeddings) < 3:
                logger.warning("Insufficient embeddings for clustering")
                return []
            
            # Determine optimal number of clusters
            n_clusters = min(max(2, len(valid_chunks) // 5), 8)
            
            # Simple clustering using numpy (instead of sklearn for simplicity)
            embeddings_array = np.array(embeddings)
            
            # Use simple k-means approach
            clusters = self._simple_kmeans(embeddings_array, n_clusters)
            
            # Group chunks by cluster
            cluster_groups = defaultdict(list)
            for chunk, label in zip(valid_chunks, clusters):
                cluster_groups[label].append(chunk)
            
            # Create ResearchCluster objects
            research_clusters = []
            for cluster_id, cluster_chunks in cluster_groups.items():
                theme = self._generate_cluster_theme(cluster_chunks)
                keywords = self._extract_cluster_keywords(cluster_chunks)
                relevance_score = self._calculate_cluster_relevance(cluster_chunks)
                
                cluster = ResearchCluster(
                    cluster_id=cluster_id,
                    theme=theme,
                    chunks=cluster_chunks,
                    keywords=keywords,
                    relevance_score=relevance_score,
                    chapter_assignments=[]
                )
                
                research_clusters.append(cluster)
            
            # Sort clusters by relevance
            research_clusters.sort(key=lambda x: x.relevance_score, reverse=True)
            
            logger.info(f"Created {len(research_clusters)} research clusters")
            return research_clusters
            
        except Exception as e:
            logger.error(f"Error creating research clusters: {e}")
            return []
    
    def _simple_kmeans(self, embeddings: np.ndarray, n_clusters: int, max_iters: int = 100) -> List[int]:
        """Simple k-means clustering implementation."""
        n_samples, n_features = embeddings.shape
        
        # Initialize centroids randomly
        centroids = embeddings[np.random.choice(n_samples, n_clusters, replace=False)]
        
        for _ in range(max_iters):
            # Assign points to nearest centroid
            distances = np.sqrt(((embeddings - centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            
            # Update centroids
            new_centroids = np.array([embeddings[labels == i].mean(axis=0) for i in range(n_clusters)])
            
            # Check for convergence
            if np.allclose(centroids, new_centroids):
                break
                
            centroids = new_centroids
        
        return labels.tolist()
    
    def _generate_cluster_theme(self, chunks: List[Dict]) -> str:
        """Generate a descriptive theme for a research cluster."""
        # Extract common query types and keywords
        query_types = [chunk['metadata'].get('query_type', '') for chunk in chunks]
        queries = [chunk['metadata'].get('query', '') for chunk in chunks]
        
        # Find most common query type
        type_counts = Counter(query_types)
        common_type = type_counts.most_common(1)[0][0] if type_counts else 'general'
        
        # Extract common keywords from queries
        all_query_words = []
        for query in queries:
            words = [word.lower() for word in query.split() if len(word) > 3 and word.isalpha()]
            all_query_words.extend(words)
        
        # Get most common keywords
        word_counts = Counter(all_query_words)
        common_words = [word for word, count in word_counts.most_common(3)]
        
        # Generate theme name
        if common_words:
            theme = f"{common_type.replace('_', ' ').title()}: {' & '.join(common_words).title()}"
        else:
            theme = common_type.replace('_', ' ').title()
        
        return theme
    
    def _extract_cluster_keywords(self, chunks: List[Dict]) -> List[str]:
        """Extract key terms that define the cluster."""
        all_content = " ".join([chunk['content'][:200] for chunk in chunks])
        
        words = [word.lower() for word in all_content.split() 
                if len(word) > 4 and word.isalpha()]
        
        word_counts = Counter(words)
        return [word for word, count in word_counts.most_common(10)]
    
    def _calculate_cluster_relevance(self, chunks: List[Dict]) -> float:
        """Calculate relevance score for a cluster based on quality and coherence."""
        if not chunks:
            return 0.0
        
        # Average quality score
        quality_scores = [chunk['metadata'].get('source_score', 0.5) for chunk in chunks]
        avg_quality = statistics.mean(quality_scores)
        
        # Cluster size factor
        size_factor = min(1.0, len(chunks) / 10)
        
        # Priority factor
        priorities = [chunk['metadata'].get('query_priority', 3) for chunk in chunks]
        avg_priority = statistics.mean(priorities) / 5.0
        
        # Combined relevance score
        relevance = (avg_quality * 0.5) + (size_factor * 0.3) + (avg_priority * 0.2)
        
        return min(1.0, relevance)
    
    async def _generate_writing_strategy(
        self,
        state: BookWritingState,
        research_analysis: Dict[str, Any],
        research_clusters: Optional[List[ResearchCluster]]
    ) -> str:
        """Generate comprehensive writing strategy using LLM."""
        try:
            if not self.config.enable_advanced_planning:
                return self._create_basic_writing_strategy(state)
            
            system_prompt = self._create_writing_strategy_prompt()
            human_prompt = self._create_writing_strategy_context(state, research_analysis, research_clusters)
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            structured_llm = self.llm.with_structured_output(WritingPlanModel)
            response = await structured_llm.ainvoke(messages)
            
            strategy = self._format_writing_strategy(response, state, research_analysis)
            
            logger.info(f"Generated comprehensive writing strategy for book {state['book_id']}")
            return strategy
            
        except Exception as e:
            logger.error(f"Error generating writing strategy: {e}")
            return self._create_basic_writing_strategy(state)
    
    def _create_writing_strategy_prompt(self) -> str:
        """Create system prompt for writing strategy generation."""
        return f"""You are an expert writing strategist specializing in {self.config.target_quality_level} publications.

Create a detailed writing strategy covering:

**WRITING APPROACH & METHODOLOGY:**
- Overall writing approach and framework
- Research integration methodology
- Quality assurance approach

**AUDIENCE & TONE STRATEGY:**
- Target audience analysis and needs
- Appropriate tone and writing style
- Complexity level and technical depth

**STRUCTURE & FLOW:**
- Narrative structure and progression
- Chapter flow and transitions
- Consistency frameworks

**RESEARCH INTEGRATION:**
- Strategy for incorporating research effectively
- Balance between depth and readability
- Credibility and authority maintenance

**QUALITY & CONSISTENCY:**
- Writing quality standards
- Key themes and messages
- Coherence maintenance methods

Provide specific, actionable guidance for professional-quality writing."""
    
    def _create_writing_strategy_context(
        self,
        state: BookWritingState,
        research_analysis: Dict[str, Any],
        research_clusters: Optional[List[ResearchCluster]]
    ) -> str:
        """Create context for writing strategy generation."""
        book_context = f"""BOOK PROJECT:
Title: {state['title']}
Genre: {state['genre']}
Target: {state['target_word_count']:,} words, {len(state['chapters'])} chapters
Audience: {state.get('target_audience', 'General audience')}
Description: {state['description']}"""
        
        # Add research context
        research_context = f"""
RESEARCH ANALYSIS:
Total Research: {research_analysis.get('total_chunks', 0)} chunks
Themes: {', '.join(research_analysis.get('themes', [])[:8])}
Quality: {research_analysis.get('quality_distribution', {}).get('avg_quality', 0):.2f}/1.0
Sources: {research_analysis.get('source_analysis', {}).get('unique_domains', 0)} domains"""
        
        # Add cluster information
        cluster_context = ""
        if research_clusters:
            cluster_context = f"""
RESEARCH CLUSTERS: {len(research_clusters)} identified"""
            for i, cluster in enumerate(research_clusters[:3]):
                cluster_context += f"\n- {cluster.theme} ({len(cluster.chunks)} chunks)"
        
        return f"""{book_context}{research_context}{cluster_context}

REQUIREMENTS:
Quality Level: {self.config.target_quality_level}
Planning Depth: {self.config.planning_depth}

Create a comprehensive writing strategy addressing all aspects of book development."""
    
    def _format_writing_strategy(
        self,
        response: WritingPlanModel,
        state: BookWritingState,
        research_analysis: Dict[str, Any]
    ) -> str:
        """Format the LLM response into a comprehensive writing strategy."""
        sections = [
            f"COMPREHENSIVE WRITING STRATEGY",
            f"Book: {state['title']}",
            f"Target: {state['target_word_count']:,} words | {len(state['chapters'])} chapters",
            f"Created: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "=" * 80,
            
            f"\n📝 WRITING APPROACH & METHODOLOGY",
            response.writing_approach,
            
            f"\n👥 TARGET AUDIENCE STRATEGY",
            response.target_audience_considerations,
            
            f"\n📚 NARRATIVE STRUCTURE & FLOW",
            response.narrative_structure,
            f"\nChapter Flow: {response.chapter_flow_strategy}",
            
            f"\n🎨 TONE & WRITING STYLE",
            response.tone_and_style,
            
            f"\n🔬 RESEARCH INTEGRATION",
            response.research_integration_approach,
            
            f"\n💡 KEY THEMES & MESSAGES"
        ]
        
        for i, theme in enumerate(response.key_themes, 1):
            sections.append(f"{i}. {theme}")
        
        sections.extend([
            f"\n🎯 QUALITY OBJECTIVES",
            response.quality_objectives,
            
            f"\n📊 RESEARCH FOUNDATION",
            f"• {research_analysis.get('total_chunks', 0)} research chunks available",
            f"• {research_analysis.get('source_analysis', {}).get('unique_domains', 0)} unique domains",
            f"• Quality score: {research_analysis.get('quality_distribution', {}).get('avg_quality', 0):.2f}/1.0"
        ])
        
        return "\n".join(sections)
    
    def _create_basic_writing_strategy(self, state: BookWritingState) -> str:
        """Create basic writing strategy when advanced planning is disabled."""
        return f"""BASIC WRITING STRATEGY

Book: {state['title']}
Target: {state['target_word_count']:,} words | {len(state['chapters'])} chapters

APPROACH:
- Sequential chapter writing based on outline
- Research integration as supporting evidence
- Consistent {self.config.target_quality_level} quality standards
- Appropriate tone for {state.get('target_audience', 'target audience')}

STRUCTURE:
- Follow existing chapter outline
- Maintain logical flow between chapters
- Use research to support arguments

QUALITY STANDARDS:
- Clear, engaging writing
- Proper research integration
- Consistent formatting and style
- Thorough review process"""
    
    async def _enhance_chapter_plans(
        self,
        state: BookWritingState,
        research_analysis: Dict[str, Any],
        research_clusters: Optional[List[ResearchCluster]],
        writing_strategy: str
    ) -> List[Chapter]:
        """Enhance chapter plans with detailed descriptions and research assignments."""
        try:
            enhanced_chapters = []
            
            # Process chapters
            for chapter in state['chapters']:
                enhanced_chapter = await self._enhance_single_chapter(
                    chapter, state, research_analysis, research_clusters, writing_strategy
                )
                enhanced_chapters.append(enhanced_chapter)
            
            # Optimize chapter lengths
            if self.config.auto_adjust_chapter_lengths:
                enhanced_chapters = self._optimize_chapter_lengths(enhanced_chapters, state)
            
            # Assign research to chapters
            enhanced_chapters = self._assign_research_to_chapters(
                enhanced_chapters, research_analysis, research_clusters
            )
            
            logger.info(f"Enhanced {len(enhanced_chapters)} chapters with detailed plans")
            return enhanced_chapters
            
        except Exception as e:
            logger.error(f"Error enhancing chapter plans: {e}")
            return self._create_fallback_chapter_plans(state['chapters'])
    
    async def _enhance_single_chapter(
        self,
        chapter: Chapter,
        state: BookWritingState,
        research_analysis: Dict[str, Any],
        research_clusters: Optional[List[ResearchCluster]],
        writing_strategy: str
    ) -> Chapter:
        """Enhance a single chapter with detailed planning."""
        try:
            system_prompt = self._create_chapter_enhancement_prompt()
            human_prompt = self._create_chapter_enhancement_context(
                chapter, state, research_analysis, writing_strategy
            )
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            structured_llm = self.llm.with_structured_output(ChapterEnhancement)
            response = await structured_llm.ainvoke(messages)
            
            # Create enhanced chapter
            enhanced_chapter = Chapter(
                chapter_number=chapter['chapter_number'],
                title=response.enhanced_title or chapter['title'],
                description=response.detailed_description or chapter.get('description', ''),
                target_word_count=self._calculate_target_word_count(response.estimated_pages, state),
                status=chapter.get('status', 'planned'),
                content=chapter.get('content'),
                research_chunks_used=None,
                word_count=chapter.get('word_count'),
                created_at=chapter.get('created_at'),
                completed_at=chapter.get('completed_at')
            )
            
            # Add enhancement metadata
            enhanced_chapter['enhanced_description'] = response.detailed_description
            enhanced_chapter['key_points'] = response.key_points
            enhanced_chapter['structure_template'] = response.structure_outline
            enhanced_chapter['dependencies'] = response.dependencies
            enhanced_chapter['complexity_score'] = self._assess_chapter_complexity(response.complexity_assessment)
            enhanced_chapter['estimated_writing_time'] = self._estimate_writing_time(
                enhanced_chapter['target_word_count'], 
                enhanced_chapter['complexity_score']
            )
            
            return enhanced_chapter
            
        except Exception as e:
            logger.error(f"Error enhancing chapter {chapter['chapter_number']}: {e}")
            return self._create_basic_enhanced_chapter(chapter, state)
    
    def _create_chapter_enhancement_prompt(self) -> str:
        """Create system prompt for chapter enhancement."""
        return f"""You are an expert book development editor specializing in {self.config.target_quality_level} publications.

Enhance the chapter outline with comprehensive planning details:

**ENHANCED TITLE & DESCRIPTION:**
- Refine title for clarity and engagement
- Expand description with specific objectives
- Define chapter's role in overall narrative

**KEY POINTS & STRUCTURE:**
- 5-8 specific key points to cover
- Logical structure for organizing content
- Recommended presentation approach

**RESEARCH REQUIREMENTS:**
- Types of research/evidence needed
- Specific examples or case studies
- Expert perspectives or data for credibility

**DEPENDENCIES & FLOW:**
- Other chapters this depends on (by number)
- Connection to overall book narrative
- Transition considerations

**COMPLEXITY ASSESSMENT:**
- Writing complexity (straightforward/moderate/complex)
- Potential writing challenges
- Relative difficulty estimation

**PRACTICAL DETAILS:**
- Estimated page count
- Structure template or format
- Special considerations

Provide specific, actionable guidance aligned with the book's strategy and audience."""
    
    def _create_chapter_enhancement_context(
        self,
        chapter: Chapter,
        state: BookWritingState,
        research_analysis: Dict[str, Any],
        writing_strategy: str
    ) -> str:
        """Create context for chapter enhancement."""
        context = f"""BOOK CONTEXT:
Title: {state['title']}
Genre: {state['genre']}
Target Audience: {state.get('target_audience', 'General audience')}
Total Chapters: {len(state['chapters'])}
Target Length: {state['target_word_count']:,} words

CURRENT CHAPTER:
Chapter {chapter['chapter_number']}: {chapter['title']}
Description: {chapter.get('description', 'No description provided')}
Target Words: {chapter.get('target_word_count', 'Not specified')}

SURROUNDING CHAPTERS:"""
        
        # Add adjacent chapters for context
        chapter_num = chapter['chapter_number']
        for ch in state['chapters']:
            if abs(ch['chapter_number'] - chapter_num) <= 2 and ch['chapter_number'] != chapter_num:
                context += f"\nChapter {ch['chapter_number']}: {ch['title']}"
                if ch.get('description'):
                    context += f" - {ch['description'][:80]}..."
        
        context += f"""

RESEARCH FOUNDATION:
Available Research: {research_analysis.get('total_chunks', 0)} chunks
Research Themes: {', '.join(research_analysis.get('themes', [])[:8])}
Quality Level: {research_analysis.get('quality_distribution', {}).get('avg_quality', 0):.2f}/1.0

WRITING STRATEGY EXCERPT:
{writing_strategy[:500]}...

Enhance this chapter with detailed planning that aligns with the overall strategy."""
        
        return context
    
    def _calculate_target_word_count(self, estimated_pages: int, state: BookWritingState) -> int:
        """Calculate target word count based on estimated pages."""
        words_per_page = 250
        base_word_count = estimated_pages * words_per_page
        
        target_count = max(
            self.config.min_chapter_word_count,
            min(self.config.max_chapter_word_count, base_word_count)
        )
        
        return target_count
    
    def _assess_chapter_complexity(self, complexity_assessment: str) -> float:
        """Convert complexity assessment to numeric score."""
        complexity_lower = complexity_assessment.lower()
        
        if 'complex' in complexity_lower or 'difficult' in complexity_lower:
            return 0.8
        elif 'moderate' in complexity_lower or 'medium' in complexity_lower:
            return 0.6
        elif 'straightforward' in complexity_lower or 'simple' in complexity_lower:
            return 0.4
        else:
            return 0.5
    
    def _estimate_writing_time(self, word_count: int, complexity_score: float) -> int:
        """Estimate writing time in hours for a chapter."""
        base_rate = 200 + (200 * (1 - complexity_score))
        estimated_hours = word_count / base_rate
        total_hours = estimated_hours * 1.5  # 50% overhead
        
        return max(2, round(total_hours))
    
    def _optimize_chapter_lengths(self, chapters: List[Chapter], state: BookWritingState) -> List[Chapter]:
        """Optimize chapter lengths for better balance."""
        if not self.config.auto_adjust_chapter_lengths:
            return chapters
        
        total_target = state['target_word_count']
        current_total = sum(ch['target_word_count'] for ch in chapters)
        
        adjustment_factor = total_target / max(current_total, 1)
        
        optimized_chapters = []
        for chapter in chapters:
            current_target = chapter['target_word_count']
            adjusted_target = int(current_target * adjustment_factor)
            
            final_target = max(
                self.config.min_chapter_word_count,
                min(self.config.max_chapter_word_count, adjusted_target)
            )
            
            optimized_chapter = chapter.copy()
            optimized_chapter['target_word_count'] = final_target
            optimized_chapters.append(optimized_chapter)
        
        logger.info(f"Optimized chapter lengths: {current_total} -> {sum(ch['target_word_count'] for ch in optimized_chapters)} words")
        return optimized_chapters
    
    def _assign_research_to_chapters(
        self,
        chapters: List[Chapter],
        research_analysis: Dict[str, Any],
        research_clusters: Optional[List[ResearchCluster]]
    ) -> List[Chapter]:
        """Assign relevant research chunks to each chapter."""
        if not research_analysis.get('chunks'):
            return chapters
        
        try:
            assigned_chapters = []
            research_chunks = research_analysis['chunks']
            
            for chapter in chapters:
                relevant_chunks = self._find_relevant_research(
                    chapter, research_chunks, research_clusters
                )
                
                assigned_chapter = chapter.copy()
                assigned_chapter['research_chunks_used'] = [
                    chunk['chunk_id'] for chunk in relevant_chunks
                ]
                assigned_chapter['supporting_evidence'] = [
                    f"{chunk['metadata'].get('source_title', 'Unknown')}: {chunk['content'][:100]}..."
                    for chunk in relevant_chunks[:3]
                ]
                
                assigned_chapters.append(assigned_chapter)
            
            logger.info(f"Assigned research to {len(assigned_chapters)} chapters")
            return assigned_chapters
            
        except Exception as e:
            logger.error(f"Error assigning research to chapters: {e}")
            return chapters
    
    def _find_relevant_research(
        self,
        chapter: Chapter,
        research_chunks: List[Dict],
        research_clusters: Optional[List[ResearchCluster]]
    ) -> List[Dict]:
        """Find research chunks most relevant to a specific chapter."""
        try:
            chapter_text = f"{chapter['title']} {chapter.get('description', '')} {chapter.get('enhanced_description', '')}"
            chapter_embedding = self.embeddings.embed_query(chapter_text)
            
            chunk_similarities = []
            for chunk in research_chunks:
                if chunk.get('embedding'):
                    # Simple cosine similarity calculation
                    similarity = self._cosine_similarity(chapter_embedding, chunk['embedding'])
                    chunk_similarities.append((chunk, similarity))
            
            # Sort by similarity and quality
            chunk_similarities.sort(
                key=lambda x: (x[1] * 0.7) + (x[0]['metadata'].get('source_score', 0.5) * 0.3),
                reverse=True
            )
            
            # Filter by similarity threshold
            relevant_chunks = [
                chunk for chunk, similarity in chunk_similarities
                if similarity >= self.config.research_similarity_threshold
            ][:self.config.research_chunks_per_chapter]
            
            return relevant_chunks
            
        except Exception as e:
            logger.error(f"Error finding relevant research for chapter {chapter['chapter_number']}: {e}")
            # Fallback: distribute evenly
            chunks_per_chapter = len(research_chunks) // len(chapter) if hasattr(chapter, '__len__') else 3
            start_idx = (chapter['chapter_number'] - 1) * chunks_per_chapter
            return research_chunks[start_idx:start_idx + chunks_per_chapter]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            vec1_array = np.array(vec1)
            vec2_array = np.array(vec2)
            
            dot_product = np.dot(vec1_array, vec2_array)
            norm1 = np.linalg.norm(vec1_array)
            norm2 = np.linalg.norm(vec2_array)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        except:
            return 0.0
    
    def _optimize_narrative_flow(self, chapters: List[Chapter], state: BookWritingState) -> List[Chapter]:
        """Optimize narrative flow and identify chapter dependencies."""
        if not self.config.enable_narrative_flow_analysis:
            return chapters
        
        try:
            optimized_chapters = []
            
            for i, chapter in enumerate(chapters):
                optimized_chapter = chapter.copy()
                dependencies = []
                
                # Check for dependencies
                if i > 0:
                    prev_chapter = chapters[i-1]
                    if self._chapters_are_dependent(prev_chapter, chapter):
                        dependencies.append(prev_chapter['chapter_number'])
                
                # Check for thematic dependencies
                for other_chapter in chapters[:i]:
                    if self._chapters_share_themes(other_chapter, chapter):
                        dependencies.append(other_chapter['chapter_number'])
                
                optimized_chapter['dependencies'] = list(set(dependencies))
                optimized_chapters.append(optimized_chapter)
            
            logger.info("Optimized narrative flow and chapter dependencies")
            return optimized_chapters
            
        except Exception as e:
            logger.error(f"Error optimizing narrative flow: {e}")
            return chapters
    
    def _chapters_are_dependent(self, prev_chapter: Chapter, current_chapter: Chapter) -> bool:
        """Check if current chapter depends on previous chapter."""
        prev_text = f"{prev_chapter['title']} {prev_chapter.get('description', '')}"
        current_text = f"{current_chapter['title']} {current_chapter.get('description', '')}"
        
        prev_words = set(word.lower() for word in prev_text.split() if len(word) > 3)
        current_words = set(word.lower() for word in current_text.split() if len(word) > 3)
        
        overlap = len(prev_words.intersection(current_words))
        return overlap >= 2
    
    def _chapters_share_themes(self, chapter1: Chapter, chapter2: Chapter) -> bool:
        """Check if chapters share significant thematic content."""
        text1 = f"{chapter1['title']} {chapter1.get('description', '')}"
        text2 = f"{chapter2['title']} {chapter2.get('description', '')}"
        
        words1 = set(word.lower() for word in text1.split() if len(word) > 4)
        words2 = set(word.lower() for word in text2.split() if len(word) > 4)
        
        overlap = len(words1.intersection(words2))
        total_unique = len(words1.union(words2))
        
        if total_unique == 0:
            return False
        
        overlap_ratio = overlap / total_unique
        return overlap_ratio >= 0.3
    
    def _calculate_writing_timeline(self, chapters: List[Chapter], state: BookWritingState) -> Dict[str, Any]:
        """Calculate estimated writing timeline and milestones."""
        if not self.config.include_writing_timeline:
            return {}
        
        try:
            total_estimated_hours = sum(
                ch.get('estimated_writing_time', 8) for ch in chapters
            )
            
            hours_per_day = 4
            estimated_days = total_estimated_hours / hours_per_day
            total_days_with_buffer = estimated_days * 1.3
            
            start_date = datetime.now(timezone.utc)
            estimated_completion = start_date + timedelta(days=total_days_with_buffer)
            
            # Generate milestones
            milestones = []
            if self.config.generate_progress_milestones:
                chapters_per_milestone = max(1, len(chapters) // 4)
                
                for i in range(0, len(chapters), chapters_per_milestone):
                    milestone_chapters = chapters[i:i + chapters_per_milestone]
                    milestone_hours = sum(
                        ch.get('estimated_writing_time', 8) for ch in milestone_chapters
                    )
                    milestone_days = milestone_hours / hours_per_day
                    milestone_date = start_date + timedelta(days=milestone_days * (i // chapters_per_milestone + 1))
                    
                    milestones.append({
                        'milestone': f"Chapters {milestone_chapters[0]['chapter_number']}-{milestone_chapters[-1]['chapter_number']} Complete",
                        'estimated_date': milestone_date.isoformat(),
                        'chapters_included': [ch['chapter_number'] for ch in milestone_chapters],
                        'estimated_words': sum(ch['target_word_count'] for ch in milestone_chapters)
                    })
            
            timeline = {
                'total_estimated_hours': total_estimated_hours,
                'estimated_writing_days': round(estimated_days),
                'estimated_completion_date': estimated_completion.isoformat(),
                'milestones': milestones,
                'daily_writing_target': round(state['target_word_count'] / total_days_with_buffer),
                'assumptions': {
                    'hours_per_day': hours_per_day,
                    'buffer_factor': 1.3,
                    'includes_revision_time': True
                }
            }
            
            logger.info(f"Calculated writing timeline: {round(estimated_days)} days, {len(milestones)} milestones")
            return timeline
            
        except Exception as e:
            logger.error(f"Error calculating writing timeline: {e}")
            return {'error': str(e)}
    
    async def _generate_style_guide(
        self,
        state: BookWritingState,
        writing_strategy: str,
        research_analysis: Dict[str, Any]
    ) -> str:
        """Generate comprehensive style guide for consistent writing."""
        try:
            system_prompt = self._create_style_guide_prompt()
            human_prompt = self._create_style_guide_context(state, writing_strategy, research_analysis)
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            structured_llm = self.llm.with_structured_output(StyleGuideModel)
            response = await structured_llm.ainvoke(messages)
            
            style_guide = self._format_style_guide(response, state)
            
            logger.info(f"Generated comprehensive style guide for book {state['book_id']}")
            return style_guide
            
        except Exception as e:
            logger.error(f"Error generating style guide: {e}")
            return self._create_basic_style_guide(state)
    
    def _create_style_guide_prompt(self) -> str:
        """Create system prompt for style guide generation."""
        return f"""You are a professional editorial consultant specializing in {self.config.target_quality_level} publications.

Create a detailed style guide ensuring consistency and quality:

**VOICE & TONE:**
- Author's voice characteristics
- Tone variations for content types
- Consistent personality maintenance

**WRITING STYLE RULES:**
- Sentence structure preferences
- Paragraph organization standards
- Transition techniques
- Technical terminology guidelines

**FORMATTING STANDARDS:**
- Heading and subheading conventions
- List formatting preferences
- Emphasis and highlighting
- Quote and citation integration

**TERMINOLOGY CONSISTENCY:**
- Key terms and preferred usage
- Industry-specific language
- Abbreviation guidelines
- Technical concept explanations

**CONSISTENCY REQUIREMENTS:**
- Elements to maintain across chapters
- Quality checkpoints
- Common pitfalls to avoid
- Editorial review criteria

**CHAPTER STRUCTURE TEMPLATE:**
- Standard organization pattern
- Introduction/conclusion guidelines
- Research integration methods
- Transition strategies

Support {self.config.style_consistency_level} consistency standards."""
    
    def _create_style_guide_context(
        self,
        state: BookWritingState,
        writing_strategy: str,
        research_analysis: Dict[str, Any]
    ) -> str:
        """Create context for style guide generation."""
        return f"""BOOK PROJECT:
Title: {state['title']}
Genre: {state['genre']}
Target Audience: {state.get('target_audience', 'General audience')}
Quality Level: {self.config.target_quality_level}
Consistency Level: {self.config.style_consistency_level}

WRITING STRATEGY SUMMARY:
{writing_strategy[:800]}...

RESEARCH CHARACTERISTICS:
Sources: {research_analysis.get('source_analysis', {}).get('unique_domains', 0)} unique domains
Quality: {research_analysis.get('quality_distribution', {}).get('avg_quality', 0):.2f}/1.0
Categories: {', '.join(list(research_analysis.get('category_distribution', {}).get('category_counts', {}).keys())[:5])}

REQUIREMENTS:
- Consistency across {len(state['chapters'])} chapters
- Target: {state['target_word_count']:,} words
- {self.config.target_quality_level} standards
- Clear research integration

Create a comprehensive style guide for consistent, high-quality writing."""
    
    def _format_style_guide(self, response: StyleGuideModel, state: BookWritingState) -> str:
        """Format the style guide response into a comprehensive document."""
        sections = [
            "COMPREHENSIVE STYLE GUIDE",
            f"Book: {state['title']}",
            f"Target: {self.config.target_quality_level.title()} Quality | {self.config.style_consistency_level.title()} Consistency",
            f"Created: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "=" * 80,
            
            "\n🎭 VOICE & TONE",
            response.voice_description,
            f"\nTone Guidelines:\n{response.tone_guidelines}",
            
            "\n✍️ WRITING STYLE RULES"
        ]
        
        for i, rule in enumerate(response.writing_style_rules, 1):
            sections.append(f"{i}. {rule}")
        
        sections.append("\n📝 FORMATTING STANDARDS")
        for i, standard in enumerate(response.formatting_standards, 1):
            sections.append(f"{i}. {standard}")
        
        sections.append("\n📚 TERMINOLOGY GLOSSARY")
        for term, definition in response.terminology_definitions.items():
            sections.append(f"• {term}: {definition}")
        
        sections.append("\n🎯 CONSISTENCY REQUIREMENTS")
        for requirement in response.consistency_requirements:
            sections.append(f"• {requirement}")
        
        sections.extend([
            "\n📖 CHAPTER STRUCTURE TEMPLATE",
            response.chapter_structure_template,
            
            "\n✅ QUALITY CHECKPOINTS",
            "Review each chapter for:"
        ])
        
        for checkpoint in self.config.consistency_check_points:
            sections.append(f"• {checkpoint.replace('_', ' ').title()}")
        
        return "\n".join(sections)
    
    def _create_basic_style_guide(self, state: BookWritingState) -> str:
        """Create basic style guide when advanced generation fails."""
        return f"""BASIC STYLE GUIDE

Book: {state['title']}
Quality Level: {self.config.target_quality_level.title()}

VOICE & TONE:
- Professional yet accessible tone for {state.get('target_audience', 'target audience')}
- Consistent voice throughout all chapters
- Clear, engaging writing style

WRITING STANDARDS:
- Clear, concise sentences with varied structure
- Logical paragraph organization
- Smooth transitions between sections
- Proper integration of research citations

FORMATTING:
- Consistent heading styles
- Standard bullet point and numbering formats
- Proper citation and reference formatting
- Professional presentation throughout

CONSISTENCY REQUIREMENTS:
- Maintain consistent terminology across chapters
- Use same citation style throughout
- Keep tone and voice consistent
- Regular quality reviews at chapter completion"""
    
    def _create_fallback_chapter_plans(self, chapters: List[Chapter]) -> List[Chapter]:
        """Create basic enhanced chapters when detailed planning fails."""
        enhanced_chapters = []
        
        for chapter in chapters:
            enhanced_chapter = chapter.copy()
            enhanced_chapter['enhanced_description'] = chapter.get('description', f"Content for {chapter['title']}")
            enhanced_chapter['key_points'] = [f"Key point for {chapter['title']}"]
            enhanced_chapter['structure_template'] = "Introduction -> Main Content -> Conclusion"
            enhanced_chapter['dependencies'] = []
            enhanced_chapter['complexity_score'] = 0.5
            enhanced_chapter['estimated_writing_time'] = 6
            
            enhanced_chapters.append(enhanced_chapter)
        
        return enhanced_chapters
    
    def _create_basic_enhanced_chapter(self, chapter: Chapter, state: BookWritingState) -> Chapter:
        """Create basic enhanced chapter when detailed enhancement fails."""
        enhanced_chapter = chapter.copy()
        
        total_chapters = len(state['chapters'])
        avg_words_per_chapter = state['target_word_count'] // total_chapters
        enhanced_chapter['target_word_count'] = max(
            self.config.min_chapter_word_count,
            min(self.config.max_chapter_word_count, avg_words_per_chapter)
        )
        
        enhanced_chapter['enhanced_description'] = chapter.get('description', f"Detailed content for {chapter['title']}")
        enhanced_chapter['key_points'] = [f"Key concepts for {chapter['title']}"]
        enhanced_chapter['structure_template'] = "Introduction -> Development -> Conclusion"
        enhanced_chapter['dependencies'] = []
        enhanced_chapter['complexity_score'] = 0.5
        enhanced_chapter['estimated_writing_time'] = enhanced_chapter['target_word_count'] // 300
        
        return enhanced_chapter
    
    def _create_fallback_writing_plan(self, state: BookWritingState) -> Dict[str, Any]:
        """Create basic fallback writing plan when main planning fails."""
        logger.warning(f"Creating fallback writing plan for book {state['book_id']}")
        
        enhanced_chapters = self._create_fallback_chapter_plans(state['chapters'])
        basic_strategy = self._create_basic_writing_strategy(state)
        basic_style_guide = self._create_basic_style_guide(state)
        
        total_hours = sum(ch.get('estimated_writing_time', 6) for ch in enhanced_chapters)
        estimated_days = total_hours / 4
        
        basic_timeline = {
            'total_estimated_hours': total_hours,
            'estimated_writing_days': round(estimated_days),
            'estimated_completion_date': (datetime.now(timezone.utc) + timedelta(days=estimated_days)).isoformat(),
            'milestones': [],
            'fallback_used': True
        }
        
        return {
            'strategy': basic_strategy,
            'updated_chapters': enhanced_chapters,
            'style_guide': basic_style_guide,
            'timeline': basic_timeline,
            'started_at': datetime.now(timezone.utc).isoformat(),
            'metadata': {
                'fallback_used': True,
                'planning_depth': 'basic',
                'research_chunks_analyzed': 0
            }
        }


def main():
    """Test function for WritingPlannerAgent."""
    from musequill.agents.agent_state import BookWritingState, ProcessingStage, Chapter, ResearchQuery
    from datetime import datetime, timezone
    
    print("Testing WritingPlannerAgent...")
    
    # Create test chapters
    test_chapters = [
        Chapter(
            chapter_number=1,
            title="Introduction to AI",
            description="Overview of artificial intelligence and its current state",
            target_word_count=5000,
            status="planned",
            content=None,
            research_chunks_used=None,
            word_count=None,
            created_at=None,
            completed_at=None
        ),
        Chapter(
            chapter_number=2,
            title="Machine Learning Fundamentals",
            description="Core concepts and algorithms in machine learning",
            target_word_count=7000,
            status="planned",
            content=None,
            research_chunks_used=None,
            word_count=None,
            created_at=None,
            completed_at=None
        )
    ]
    
    # Create test research queries
    test_queries = [
        ResearchQuery(
            query="artificial intelligence history and development",
            priority=5,
            query_type="background_information",
            status="completed",
            results_count=12,
            created_at=datetime.now(timezone.utc).isoformat()
        )
    ]
    
    # Create test state
    test_state = BookWritingState(
        book_id="test_book_writing_plan_123",
        orchestration_id="test_orch_456",
        thread_id="test_thread_789",
        title="The Complete Guide to Artificial Intelligence",
        description="A comprehensive guide covering AI fundamentals and applications",
        genre="Technology/Non-fiction",
        target_word_count=50000,
        target_audience="Technology professionals",
        author_preferences={},
        outline={},
        chapters=test_chapters,
        current_stage=ProcessingStage.RESEARCH_COMPLETE,
        processing_started_at=datetime.now(timezone.utc).isoformat(),
        processing_updated_at=datetime.now(timezone.utc).isoformat(),
        research_queries=test_queries,
        research_strategy="Comprehensive research strategy",
        total_research_chunks=25,
        research_completed_at=datetime.now(timezone.utc).isoformat(),
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
        progress_percentage=40.0,
        estimated_completion_time=None,
        final_book_content=None,
        metadata={}
    )
    
    try:
        planner = WritingPlannerAgent()
        
        print("Creating comprehensive writing plan...")
        writing_plan = planner.create_writing_plan(test_state)
        
        print(f"\nWriting Plan Created!")
        print(f"Strategy length: {len(writing_plan['strategy'])} characters")
        print(f"Enhanced chapters: {len(writing_plan['updated_chapters'])}")
        print(f"Style guide included: {writing_plan['style_guide'] is not None}")
        
        print("\nWritingPlannerAgent test completed successfully!")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()