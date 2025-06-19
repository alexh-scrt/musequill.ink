"""
Research Validator Agent

Validates the completeness and quality of research conducted for book writing.
Analyzes research coverage, identifies gaps, and generates additional queries if needed.

Key Features:
- Comprehensive research quality assessment
- Coverage analysis across query categories and book chapters
- LLM-powered gap analysis and content evaluation
- Automatic generation of additional research queries
- Statistical validation with configurable thresholds
- Content distribution analysis across book structure
- Source diversity and quality evaluation
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import statistics

import chromadb
from chromadb.config import Settings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel

from musequill.config.logging import get_logger
from musequill.agents.research_validator.research_validator_config import ResearchValidatorConfig
from musequill.agents.agent_state import BookWritingState, ResearchQuery

logger = get_logger(__name__)


@dataclass
class ValidationMetrics:
    """Research validation metrics."""
    total_chunks: int
    unique_sources: int
    unique_domains: int
    source_diversity_ratio: float
    avg_quality_score: float
    high_quality_percentage: float
    category_coverage: Dict[str, int]
    query_success_rate: float
    content_distribution_score: float


@dataclass
class GapAnalysis:
    """Research gap analysis results."""
    identified_gaps: List[str]
    missing_categories: List[str]
    weak_areas: List[str]
    recommended_additional_queries: List[str]
    confidence_score: float


@dataclass
class ValidationResult:
    """Complete validation result."""
    is_sufficient: bool
    confidence_score: float
    validation_metrics: ValidationMetrics
    gap_analysis: Optional[GapAnalysis]
    additional_queries: List[ResearchQuery]
    validation_summary: str
    recommendations: List[str]
    completed_at: str


class ResearchGap(BaseModel):
    """Pydantic model for LLM-identified research gaps."""
    gap_description: str
    importance: int  # 1-5
    suggested_query: str
    category: str
    rationale: str


class GapAnalysisResponse(BaseModel):
    """Pydantic model for LLM gap analysis response."""
    gaps: List[ResearchGap]
    overall_assessment: str
    research_quality: int  # 1-10
    missing_critical_information: List[str]
    recommendations: List[str]


class ResearchValidatorAgent:
    """
    Research Validator Agent that assesses research completeness and quality.
    """
    
    def __init__(self, config: Optional[ResearchValidatorConfig] = None):
        if not config:
            config = ResearchValidatorConfig()
        
        self.config = config
        
        # Initialize clients
        self.llm: Optional[ChatOpenAI] = None
        self.chroma_client: Optional[chromadb.HttpClient] = None
        self.chroma_collection = None
        
        # Validation thresholds based on strictness
        self.thresholds = self._get_validation_thresholds()
        
        self._initialize_components()
        
        logger.info("Research Validator Agent initialized")
    
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
            
            logger.info("Research validator components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize research validator components: {e}")
            raise
    
    def _get_validation_thresholds(self) -> Dict[str, Any]:
        """Get validation thresholds based on strictness level."""
        base_thresholds = {
            'min_chunks_per_query': self.config.min_chunks_per_query,
            'min_total_chunks': self.config.min_total_chunks,
            'min_unique_sources': self.config.min_unique_sources,
            'min_source_diversity': self.config.min_source_diversity,
            'min_avg_quality_score': self.config.min_avg_quality_score,
            'min_high_quality_percentage': self.config.min_high_quality_percentage
        }
        
        # Adjust based on strictness
        if self.config.validation_strictness == "high":
            multipliers = {
                'min_chunks_per_query': 1.5,
                'min_total_chunks': 1.3,
                'min_unique_sources': 1.4,
                'min_source_diversity': 1.1,
                'min_avg_quality_score': 1.2,
                'min_high_quality_percentage': 1.3
            }
        elif self.config.validation_strictness == "low":
            multipliers = {
                'min_chunks_per_query': 0.7,
                'min_total_chunks': 0.8,
                'min_unique_sources': 0.8,
                'min_source_diversity': 0.9,
                'min_avg_quality_score': 0.9,
                'min_high_quality_percentage': 0.8
            }
        else:  # medium
            multipliers = {key: 1.0 for key in base_thresholds}
        
        # Apply multipliers
        adjusted_thresholds = {}
        for key, base_value in base_thresholds.items():
            if isinstance(base_value, (int, float)):
                adjusted_value = base_value * multipliers[key]
                # Keep integers as integers
                if isinstance(base_value, int):
                    adjusted_thresholds[key] = max(1, round(adjusted_value))
                else:
                    adjusted_thresholds[key] = min(1.0, adjusted_value)
            else:
                adjusted_thresholds[key] = base_value
        
        logger.info(f"Using {self.config.validation_strictness} strictness validation thresholds")
        return adjusted_thresholds
    
    def validate_research(self, state: BookWritingState) -> Dict[str, Any]:
        """
        Validate research completeness and quality for a book.
        
        Args:
            state: BookWritingState containing research information
            
        Returns:
            Dictionary with validation results
        """
        try:
            logger.info(f"Starting research validation for book {state['book_id']}")
            start_time = time.time()
            
            # Retrieve research data from Chroma
            research_data = self._retrieve_research_data(state['book_id'])
            
            if not research_data['chunks']:
                logger.warning(f"No research data found for book {state['book_id']}")
                return self._create_insufficient_research_result(
                    "No research data found in vector database",
                    state
                )
            
            # Calculate validation metrics
            metrics = self._calculate_validation_metrics(research_data, state)
            
            # Perform statistical validation
            statistical_validation = self._perform_statistical_validation(metrics, state)
            
            # Perform content analysis and gap detection
            gap_analysis = None
            if self.config.enable_gap_analysis:
                gap_analysis = asyncio.run(
                    self._perform_gap_analysis(research_data, state, metrics)
                )
            
            # Generate additional queries if needed
            additional_queries = []
            if not statistical_validation['is_sufficient'] or (gap_analysis and gap_analysis.identified_gaps):
                additional_queries = self._generate_additional_queries(
                    state, metrics, gap_analysis, research_data
                )
            
            # Determine overall sufficiency
            is_sufficient = self._determine_overall_sufficiency(
                statistical_validation, gap_analysis, metrics
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                metrics, statistical_validation, gap_analysis
            )
            
            # Create validation summary
            validation_summary = self._create_validation_summary(
                metrics, statistical_validation, gap_analysis, is_sufficient
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                metrics, gap_analysis, statistical_validation
            )
            
            # Create validation result
            validation_result = ValidationResult(
                is_sufficient=is_sufficient,
                confidence_score=confidence_score,
                validation_metrics=metrics,
                gap_analysis=gap_analysis,
                additional_queries=additional_queries,
                validation_summary=validation_summary,
                recommendations=recommendations,
                completed_at=datetime.now(timezone.utc).isoformat()
            )
            
            execution_time = time.time() - start_time
            
            logger.info(
                f"Research validation completed for book {state['book_id']}: "
                f"sufficient={is_sufficient}, confidence={confidence_score:.2f}, "
                f"additional_queries={len(additional_queries)}, time={execution_time:.2f}s"
            )
            
            return {
                'is_sufficient': validation_result.is_sufficient,
                'confidence_score': validation_result.confidence_score,
                'additional_queries': validation_result.additional_queries,
                'validation_summary': validation_result.validation_summary,
                'recommendations': validation_result.recommendations,
                'metrics': self._metrics_to_dict(validation_result.validation_metrics),
                'gap_analysis': self._gap_analysis_to_dict(validation_result.gap_analysis) if validation_result.gap_analysis else None,
                'completed_at': validation_result.completed_at,
                'execution_time': execution_time
            }
            
        except Exception as e:
            logger.error(f"Error validating research for book {state['book_id']}: {e}")
            if self.config.fallback_validation_mode:
                return self._perform_fallback_validation(state)
            else:
                raise
    
    def _retrieve_research_data(self, book_id: str) -> Dict[str, Any]:
        """Retrieve and organize research data from Chroma."""
        try:
            # Get all research chunks for this book
            results = self.chroma_collection.get(
                where={"book_id": book_id},
                include=["documents", "metadatas", "embeddings"]
            )
            
            chunks = []
            if results['documents']:
                for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
                    chunk_data = {
                        'content': doc,
                        'metadata': metadata,
                        'embedding': results['embeddings'][i] if results['embeddings'] else None
                    }
                    chunks.append(chunk_data)
            
            # Organize by query and category
            query_groups = defaultdict(list)
            category_groups = defaultdict(list)
            
            for chunk in chunks:
                query = chunk['metadata'].get('query', 'unknown')
                category = chunk['metadata'].get('query_type', 'unknown')
                
                query_groups[query].append(chunk)
                category_groups[category].append(chunk)
            
            # Extract source information
            sources = set()
            domains = set()
            
            for chunk in chunks:
                source_url = chunk['metadata'].get('source_url')
                source_domain = chunk['metadata'].get('source_domain')
                
                if source_url:
                    sources.add(source_url)
                if source_domain:
                    domains.add(source_domain)
            
            logger.info(
                f"Retrieved research data for book {book_id}: "
                f"{len(chunks)} chunks, {len(sources)} sources, {len(domains)} domains"
            )
            
            return {
                'chunks': chunks,
                'query_groups': dict(query_groups),
                'category_groups': dict(category_groups),
                'unique_sources': sources,
                'unique_domains': domains,
                'total_chunks': len(chunks)
            }
            
        except Exception as e:
            logger.error(f"Error retrieving research data for book {book_id}: {e}")
            return {
                'chunks': [],
                'query_groups': {},
                'category_groups': {},
                'unique_sources': set(),
                'unique_domains': set(),
                'total_chunks': 0
            }
    
    def _calculate_validation_metrics(
        self, 
        research_data: Dict[str, Any], 
        state: BookWritingState
    ) -> ValidationMetrics:
        """Calculate comprehensive validation metrics."""
        chunks = research_data['chunks']
        
        if not chunks:
            return ValidationMetrics(
                total_chunks=0,
                unique_sources=0,
                unique_domains=0,
                source_diversity_ratio=0.0,
                avg_quality_score=0.0,
                high_quality_percentage=0.0,
                category_coverage={},
                query_success_rate=0.0,
                content_distribution_score=0.0
            )
        
        # Basic counts
        total_chunks = len(chunks)
        unique_sources = len(research_data['unique_sources'])
        unique_domains = len(research_data['unique_domains'])
        
        # Source diversity
        source_diversity_ratio = unique_domains / max(unique_sources, 1)
        
        # Quality scores (if available in metadata)
        quality_scores = []
        for chunk in chunks:
            # Try to get quality score from metadata or calculate a proxy
            quality_score = chunk['metadata'].get('quality_score')
            if quality_score is None:
                # Calculate proxy quality score based on available metadata
                source_score = chunk['metadata'].get('source_score', 0.5)
                chunk_size = chunk['metadata'].get('chunk_size', 500)
                
                # Simple proxy: combine source score with size factor
                size_factor = min(1.0, chunk_size / 800)  # Optimal around 800 chars
                quality_score = (source_score * 0.7) + (size_factor * 0.3)
            
            quality_scores.append(quality_score)
        
        avg_quality_score = statistics.mean(quality_scores) if quality_scores else 0.0
        high_quality_count = sum(1 for score in quality_scores if score >= self.config.high_quality_threshold)
        high_quality_percentage = high_quality_count / total_chunks if total_chunks > 0 else 0.0
        
        # Category coverage
        category_coverage = {}
        for category, chunks_in_category in research_data['category_groups'].items():
            category_coverage[category] = len(chunks_in_category)
        
        # Query success rate
        completed_queries = [q for q in state['research_queries'] if q['status'] == 'completed']
        total_queries = len(state['research_queries'])
        query_success_rate = len(completed_queries) / max(total_queries, 1)
        
        # Content distribution score
        content_distribution_score = self._calculate_content_distribution_score(
            research_data, state
        )
        
        return ValidationMetrics(
            total_chunks=total_chunks,
            unique_sources=unique_sources,
            unique_domains=unique_domains,
            source_diversity_ratio=source_diversity_ratio,
            avg_quality_score=avg_quality_score,
            high_quality_percentage=high_quality_percentage,
            category_coverage=category_coverage,
            query_success_rate=query_success_rate,
            content_distribution_score=content_distribution_score
        )
    
    def _calculate_content_distribution_score(
        self, 
        research_data: Dict[str, Any], 
        state: BookWritingState
    ) -> float:
        """Calculate how well research content is distributed across book structure."""
        if not self.config.analyze_content_distribution:
            return 1.0
        
        # Analyze distribution across required categories
        required_categories = set(self.config.required_query_categories)
        covered_categories = set(research_data['category_groups'].keys())
        
        category_coverage_ratio = len(covered_categories.intersection(required_categories)) / len(required_categories)
        
        # Analyze distribution across chapters (if chapter info available)
        chapter_distribution_score = 1.0
        if state['chapters'] and len(state['chapters']) > 1:
            total_chapters = len(state['chapters'])
            recommended_chunks_per_chapter = max(1, research_data['total_chunks'] // total_chapters)
            
            if recommended_chunks_per_chapter >= self.config.min_content_per_chapter:
                chapter_distribution_score = 1.0
            else:
                chapter_distribution_score = recommended_chunks_per_chapter / self.config.min_content_per_chapter
        
        # Combined distribution score
        distribution_score = (category_coverage_ratio * 0.7) + (chapter_distribution_score * 0.3)
        
        return min(1.0, distribution_score)
    
    def _perform_statistical_validation(
        self, 
        metrics: ValidationMetrics, 
        state: BookWritingState
    ) -> Dict[str, Any]:
        """Perform statistical validation against thresholds."""
        validation_results = {}
        passed_checks = 0
        total_checks = 0
        
        # Check each threshold
        checks = [
            ('total_chunks', metrics.total_chunks >= self.thresholds['min_total_chunks']),
            ('unique_sources', metrics.unique_sources >= self.thresholds['min_unique_sources']),
            ('source_diversity', metrics.source_diversity_ratio >= self.thresholds['min_source_diversity']),
            ('avg_quality', metrics.avg_quality_score >= self.thresholds['min_avg_quality_score']),
            ('high_quality_percentage', metrics.high_quality_percentage >= self.thresholds['min_high_quality_percentage']),
            ('query_success_rate', metrics.query_success_rate >= 0.7),  # 70% of queries should succeed
            ('content_distribution', metrics.content_distribution_score >= 0.6)
        ]
        
        for check_name, passed in checks:
            validation_results[check_name] = passed
            if passed:
                passed_checks += 1
            total_checks += 1
        
        # Check category coverage
        category_coverage_passed = True
        for required_category in self.config.required_query_categories:
            chunks_in_category = metrics.category_coverage.get(required_category, 0)
            if chunks_in_category < self.config.min_coverage_per_category:
                category_coverage_passed = False
                break
        
        validation_results['category_coverage'] = category_coverage_passed
        if category_coverage_passed:
            passed_checks += 1
        total_checks += 1
        
        # Check chunks per query
        chunks_per_query_passed = True
        for query in state['research_queries']:
            if query['status'] == 'completed':
                results_count = query.get('results_count', 0)
                if results_count < self.thresholds['min_chunks_per_query']:
                    chunks_per_query_passed = False
                    break
        
        validation_results['chunks_per_query'] = chunks_per_query_passed
        if chunks_per_query_passed:
            passed_checks += 1
        total_checks += 1
        
        # Overall statistical validation
        pass_rate = passed_checks / total_checks
        is_sufficient = pass_rate >= 0.7  # 70% of checks must pass
        
        validation_results.update({
            'pass_rate': pass_rate,
            'passed_checks': passed_checks,
            'total_checks': total_checks,
            'is_sufficient': is_sufficient
        })
        
        logger.info(f"Statistical validation: {passed_checks}/{total_checks} checks passed ({pass_rate:.1%})")
        
        return validation_results
    
    async def _perform_gap_analysis(
        self, 
        research_data: Dict[str, Any], 
        state: BookWritingState, 
        metrics: ValidationMetrics
    ) -> GapAnalysis:
        """Perform LLM-powered gap analysis."""
        try:
            # Sample content for analysis
            sample_chunks = self._sample_content_for_analysis(
                research_data['chunks'], 
                self.config.gap_analysis_sample_size
            )
            
            # Create analysis prompt
            system_prompt = self._create_gap_analysis_prompt()
            human_prompt = self._create_gap_analysis_context(state, metrics, sample_chunks)
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            # Use structured output for reliable parsing
            structured_llm = self.llm.with_structured_output(GapAnalysisResponse)
            
            response = await structured_llm.ainvoke(messages)
            
            # Process LLM response
            identified_gaps = [gap.gap_description for gap in response.gaps]
            recommended_queries = [gap.suggested_query for gap in response.gaps[:self.config.max_additional_queries]]
            
            # Identify missing categories
            required_categories = set(self.config.required_query_categories)
            covered_categories = set(metrics.category_coverage.keys())
            missing_categories = list(required_categories - covered_categories)
            
            # Identify weak areas (categories with insufficient coverage)
            weak_areas = []
            for category, count in metrics.category_coverage.items():
                if count < self.config.min_coverage_per_category:
                    weak_areas.append(f"{category} (only {count} chunks)")
            
            # Calculate confidence based on research quality
            confidence_score = min(1.0, response.research_quality / 10.0)
            
            gap_analysis = GapAnalysis(
                identified_gaps=identified_gaps,
                missing_categories=missing_categories,
                weak_areas=weak_areas,
                recommended_additional_queries=recommended_queries,
                confidence_score=confidence_score
            )
            
            logger.info(
                f"Gap analysis completed: {len(identified_gaps)} gaps identified, "
                f"confidence={confidence_score:.2f}"
            )
            
            return gap_analysis
            
        except Exception as e:
            logger.error(f"Error performing gap analysis: {e}")
            # Return basic gap analysis based on metrics
            return self._create_fallback_gap_analysis(metrics, state)
    
    def _sample_content_for_analysis(self, chunks: List[Dict], sample_size: int) -> List[Dict]:
        """Sample content chunks for LLM analysis."""
        if len(chunks) <= sample_size:
            return chunks
        
        # Stratified sampling: get chunks from different categories and quality levels
        sampled_chunks = []
        
        # Group by category
        category_groups = defaultdict(list)
        for chunk in chunks:
            category = chunk['metadata'].get('query_type', 'unknown')
            category_groups[category].append(chunk)
        
        # Sample from each category
        chunks_per_category = max(1, sample_size // len(category_groups))
        
        for category, chunks_in_category in category_groups.items():
            # Sort by quality score (if available) and take best chunks
            sorted_chunks = sorted(
                chunks_in_category,
                key=lambda x: x['metadata'].get('source_score', 0.5),
                reverse=True
            )
            
            sampled_from_category = sorted_chunks[:chunks_per_category]
            sampled_chunks.extend(sampled_from_category)
        
        # If we need more chunks, add remaining best ones
        remaining_needed = sample_size - len(sampled_chunks)
        if remaining_needed > 0:
            all_remaining = [chunk for chunk in chunks if chunk not in sampled_chunks]
            sorted_remaining = sorted(
                all_remaining,
                key=lambda x: x['metadata'].get('source_score', 0.5),
                reverse=True
            )
            sampled_chunks.extend(sorted_remaining[:remaining_needed])
        
        return sampled_chunks[:sample_size]
    
    def _create_gap_analysis_prompt(self) -> str:
        """Create system prompt for gap analysis."""
        return """You are an expert research analyst specializing in evaluating research completeness for book writing projects.

Your task is to analyze the research content provided and identify gaps, weaknesses, and areas that need additional investigation. Focus on:

1. **Content Gaps**: Missing information that would be essential for the book's credibility and completeness
2. **Quality Issues**: Areas where the research seems shallow or lacks authoritative sources
3. **Balance Problems**: Topics that are over-researched vs. under-researched
4. **Critical Missing Elements**: Key perspectives, data, or examples that are absent

For each gap you identify, provide:
- Clear description of what's missing
- Importance level (1-5, where 5 is critical)
- Specific search query that would address the gap
- Appropriate research category
- Rationale for why this research is needed

**Research Categories:**
- background_information
- technical_details
- expert_opinions
- case_studies
- current_trends
- historical_context
- statistical_data
- examples_and_illustrations

**Guidelines:**
- Be specific and actionable in your gap identification
- Consider the book's target audience and purpose
- Prioritize gaps that would significantly impact book quality
- Focus on research that would provide unique value
- Consider both breadth (topic coverage) and depth (detail level)

Provide an overall research quality assessment (1-10) and specific recommendations for improvement."""
    
    def _create_gap_analysis_context(
        self, 
        state: BookWritingState, 
        metrics: ValidationMetrics, 
        sample_chunks: List[Dict]
    ) -> str:
        """Create context for gap analysis."""
        # Prepare book information
        book_info = f"""BOOK INFORMATION:
Title: {state['title']}
Genre: {state['genre']}
Target Word Count: {state['target_word_count']:,} words
Target Audience: {state.get('target_audience', 'General audience')}
Description: {state['description']}

CHAPTER STRUCTURE:"""
        
        if state['chapters']:
            for chapter in state['chapters'][:10]:  # Limit to first 10 chapters
                book_info += f"\nChapter {chapter['chapter_number']}: {chapter['title']}"
                if chapter.get('description'):
                    book_info += f" - {chapter['description'][:100]}..."
        else:
            book_info += "\nNo detailed chapter information available"
        
        # Research metrics summary
        metrics_summary = f"""
CURRENT RESEARCH METRICS:
- Total Research Chunks: {metrics.total_chunks}
- Unique Sources: {metrics.unique_sources}
- Source Diversity: {metrics.source_diversity_ratio:.1%}
- Average Quality Score: {metrics.avg_quality_score:.2f}
- High Quality Content: {metrics.high_quality_percentage:.1%}
- Query Success Rate: {metrics.query_success_rate:.1%}

CATEGORY COVERAGE:"""
        
        for category, count in metrics.category_coverage.items():
            metrics_summary += f"\n- {category.replace('_', ' ').title()}: {count} chunks"
        
        # Sample content analysis
        content_summary = "\nSAMPLE RESEARCH CONTENT:\n"
        
        for i, chunk in enumerate(sample_chunks[:10]):  # Limit to 10 samples
            metadata = chunk['metadata']
            content_preview = chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content']
            
            content_summary += f"""
Sample {i+1}:
Query: {metadata.get('query', 'Unknown')}
Category: {metadata.get('query_type', 'Unknown')}
Source: {metadata.get('source_domain', 'Unknown')}
Content Preview: {content_preview}
---"""
        
        # Research queries analysis
        queries_info = "\nCOMPLETED RESEARCH QUERIES:\n"
        completed_queries = [q for q in state['research_queries'] if q['status'] == 'completed']
        
        for query in completed_queries[:15]:  # Limit to 15 queries
            queries_info += f"- {query['query']} (Category: {query['query_type']}, Results: {query.get('results_count', 0)})\n"
        
        full_context = f"""{book_info}

{metrics_summary}

{queries_info}

{content_summary}

ANALYSIS REQUEST:
Based on the book requirements and current research status, identify gaps and recommend additional research. Consider:

1. Are there critical aspects of the book topic that lack sufficient research?
2. Are any required categories missing or under-represented?
3. Does the research provide enough depth for the target audience?
4. Are there missing perspectives or viewpoints?
5. Would additional research significantly improve the book's quality and authority?

Focus on high-impact gaps that would meaningfully enhance the book's content and credibility."""
        
        return full_context
    
    def _create_fallback_gap_analysis(self, metrics: ValidationMetrics, state: BookWritingState) -> GapAnalysis:
        """Create basic gap analysis when LLM analysis fails."""
        logger.warning("Creating fallback gap analysis")
        
        identified_gaps = []
        missing_categories = []
        weak_areas = []
        recommended_queries = []
        
        # Check for missing required categories
        required_categories = set(self.config.required_query_categories)
        covered_categories = set(metrics.category_coverage.keys())
        missing_categories = list(required_categories - covered_categories)
        
        # Identify weak areas
        for category, count in metrics.category_coverage.items():
            if count < self.config.min_coverage_per_category:
                weak_areas.append(f"{category} needs more research")
        
        # Generate basic gaps and queries
        if missing_categories:
            for category in missing_categories:
                gap = f"Missing research in {category.replace('_', ' ')}"
                identified_gaps.append(gap)
                
                query = f"{state['title']} {category.replace('_', ' ')}"
                recommended_queries.append(query)
        
        if weak_areas:
            for weak_area in weak_areas[:3]:  # Limit to 3
                category = weak_area.split()[0]
                query = f"{state['genre']} {category} comprehensive research"
                recommended_queries.append(query)
        
        # Basic confidence based on metrics
        confidence_score = min(1.0, (metrics.total_chunks / self.thresholds['min_total_chunks']) * 0.8)
        
        return GapAnalysis(
            identified_gaps=identified_gaps,
            missing_categories=missing_categories,
            weak_areas=weak_areas,
            recommended_additional_queries=recommended_queries[:self.config.max_additional_queries],
            confidence_score=confidence_score
        )
    
    def _generate_additional_queries(
        self,
        state: BookWritingState,
        metrics: ValidationMetrics,
        gap_analysis: Optional[GapAnalysis],
        research_data: Dict[str, Any]
    ) -> List[ResearchQuery]:
        """Generate additional research queries based on gaps and validation results."""
        if not self.config.enable_automatic_query_generation:
            return []
        
        additional_queries = []
        current_time = datetime.now(timezone.utc).isoformat()
        
        # Generate queries from gap analysis
        if gap_analysis and gap_analysis.recommended_additional_queries:
            for i, query_text in enumerate(gap_analysis.recommended_additional_queries):
                # Determine category and priority
                category = self._determine_query_category(query_text, gap_analysis.missing_categories)
                priority = 5 if category in self.config.required_query_categories else 4
                
                query = ResearchQuery(
                    query=query_text,
                    priority=priority,
                    query_type=category,
                    status="pending",
                    results_count=None,
                    created_at=current_time
                )
                
                # Add metadata
                query['generated_by'] = 'gap_analysis'
                query['rationale'] = f"Generated to address identified research gap"
                additional_queries.append(query)
        
        # Generate queries for missing categories
        missing_categories = set(self.config.required_query_categories) - set(metrics.category_coverage.keys())
        for category in missing_categories:
            if len(additional_queries) >= self.config.max_additional_queries:
                break
            
            query_text = f"{state['title']} {category.replace('_', ' ')} research"
            
            query = ResearchQuery(
                query=query_text,
                priority=5,
                query_type=category,
                status="pending",
                results_count=None,
                created_at=current_time
            )
            
            query['generated_by'] = 'missing_category'
            query['rationale'] = f"Generated to cover missing required category: {category}"
            additional_queries.append(query)
        
        # Generate queries for weak areas
        for category, count in metrics.category_coverage.items():
            if count < self.config.min_coverage_per_category and len(additional_queries) < self.config.max_additional_queries:
                query_text = f"{state['genre']} {category.replace('_', ' ')} detailed information"
                
                query = ResearchQuery(
                    query=query_text,
                    priority=4,
                    query_type=category,
                    status="pending",
                    results_count=None,
                    created_at=current_time
                )
                
                query['generated_by'] = 'insufficient_coverage'
                query['rationale'] = f"Generated to strengthen weak area: {category} (only {count} chunks)"
                additional_queries.append(query)
        
        logger.info(f"Generated {len(additional_queries)} additional research queries")
        return additional_queries[:self.config.max_additional_queries]
    
    def _determine_query_category(self, query_text: str, missing_categories: List[str]) -> str:
        """Determine appropriate category for a query."""
        query_lower = query_text.lower()
        
        # Check for category keywords
        category_keywords = {
            'technical_details': ['technical', 'detailed', 'how', 'mechanism', 'process'],
            'expert_opinions': ['expert', 'opinion', 'perspective', 'view', 'analysis'],
            'background_information': ['background', 'introduction', 'overview', 'basics'],
            'case_studies': ['case', 'example', 'study', 'implementation'],
            'current_trends': ['trend', 'current', 'latest', 'recent', 'modern'],
            'statistical_data': ['data', 'statistics', 'numbers', 'metrics', 'research'],
            'historical_context': ['history', 'historical', 'evolution', 'development'],
            'examples_and_illustrations': ['example', 'illustration', 'demonstration']
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return category
        
        # Default to first missing category or background information
        if missing_categories:
            return missing_categories[0]
        
        return 'background_information'
    
    def _determine_overall_sufficiency(
        self,
        statistical_validation: Dict[str, Any],
        gap_analysis: Optional[GapAnalysis],
        metrics: ValidationMetrics
    ) -> bool:
        """Determine if research is overall sufficient."""
        # Statistical validation must pass
        if not statistical_validation['is_sufficient']:
            return False
        
        # If gap analysis identifies critical gaps, research is insufficient
        if gap_analysis:
            critical_gaps = [gap for gap in gap_analysis.identified_gaps if 'critical' in gap.lower()]
            if critical_gaps:
                return False
            
            # Too many missing categories indicates insufficient research
            if len(gap_analysis.missing_categories) > 1:
                return False
            
            # Confidence score too low
            if gap_analysis.confidence_score < 0.6:
                return False
        
        # Check minimum absolute thresholds regardless of strictness
        absolute_minimums = {
            'total_chunks': 10,
            'unique_sources': 3,
            'avg_quality_score': 0.3
        }
        
        if (metrics.total_chunks < absolute_minimums['total_chunks'] or
            metrics.unique_sources < absolute_minimums['unique_sources'] or
            metrics.avg_quality_score < absolute_minimums['avg_quality_score']):
            return False
        
        return True
    
    def _calculate_confidence_score(
        self,
        metrics: ValidationMetrics,
        statistical_validation: Dict[str, Any],
        gap_analysis: Optional[GapAnalysis]
    ) -> float:
        """Calculate overall confidence score for the validation."""
        confidence_factors = []
        
        # Statistical validation confidence
        pass_rate = statistical_validation.get('pass_rate', 0.0)
        confidence_factors.append(pass_rate)
        
        # Metrics-based confidence
        metrics_confidence = min(1.0, (
            (metrics.total_chunks / max(self.thresholds['min_total_chunks'], 1)) * 0.3 +
            (metrics.avg_quality_score / 1.0) * 0.3 +
            (metrics.source_diversity_ratio / 1.0) * 0.2 +
            (metrics.query_success_rate / 1.0) * 0.2
        ))
        confidence_factors.append(metrics_confidence)
        
        # Gap analysis confidence
        if gap_analysis:
            confidence_factors.append(gap_analysis.confidence_score)
        else:
            confidence_factors.append(0.7)  # Default if no gap analysis
        
        # Overall confidence is weighted average
        weights = [0.4, 0.4, 0.2]  # Statistical, metrics, gap analysis
        weighted_confidence = sum(conf * weight for conf, weight in zip(confidence_factors, weights))
        
        return min(1.0, weighted_confidence)
    
    def _create_validation_summary(
        self,
        metrics: ValidationMetrics,
        statistical_validation: Dict[str, Any],
        gap_analysis: Optional[GapAnalysis],
        is_sufficient: bool
    ) -> str:
        """Create human-readable validation summary."""
        summary_parts = []
        
        # Overall result
        if is_sufficient:
            summary_parts.append("✅ RESEARCH VALIDATION PASSED")
            summary_parts.append("The research is sufficient to proceed with writing.")
        else:
            summary_parts.append("❌ RESEARCH VALIDATION FAILED")
            summary_parts.append("Additional research is required before proceeding with writing.")
        
        # Key metrics
        summary_parts.append(f"\n📊 KEY METRICS:")
        summary_parts.append(f"• Total Research Chunks: {metrics.total_chunks}")
        summary_parts.append(f"• Unique Sources: {metrics.unique_sources}")
        summary_parts.append(f"• Source Diversity: {metrics.source_diversity_ratio:.1%}")
        summary_parts.append(f"• Average Quality Score: {metrics.avg_quality_score:.2f}/1.0")
        summary_parts.append(f"• High Quality Content: {metrics.high_quality_percentage:.1%}")
        summary_parts.append(f"• Query Success Rate: {metrics.query_success_rate:.1%}")
        
        # Statistical validation results
        pass_rate = statistical_validation.get('pass_rate', 0.0)
        passed_checks = statistical_validation.get('passed_checks', 0)
        total_checks = statistical_validation.get('total_checks', 0)
        
        summary_parts.append(f"\n🔍 VALIDATION CHECKS:")
        summary_parts.append(f"• Passed {passed_checks}/{total_checks} validation checks ({pass_rate:.1%})")
        
        # Category coverage
        summary_parts.append(f"\n📚 CATEGORY COVERAGE:")
        for category, count in metrics.category_coverage.items():
            category_name = category.replace('_', ' ').title()
            summary_parts.append(f"• {category_name}: {count} chunks")
        
        # Gap analysis summary
        if gap_analysis:
            summary_parts.append(f"\n🔍 GAP ANALYSIS:")
            if gap_analysis.identified_gaps:
                summary_parts.append(f"• {len(gap_analysis.identified_gaps)} research gaps identified")
            if gap_analysis.missing_categories:
                summary_parts.append(f"• Missing categories: {', '.join(gap_analysis.missing_categories)}")
            if gap_analysis.weak_areas:
                summary_parts.append(f"• Weak areas: {len(gap_analysis.weak_areas)} identified")
            summary_parts.append(f"• Analysis confidence: {gap_analysis.confidence_score:.1%}")
        
        return "\n".join(summary_parts)
    
    def _generate_recommendations(
        self,
        metrics: ValidationMetrics,
        gap_analysis: Optional[GapAnalysis],
        statistical_validation: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []
        
        # Statistical validation recommendations
        if not statistical_validation.get('total_chunks', True):
            recommendations.append(f"Increase total research chunks to at least {self.thresholds['min_total_chunks']}")
        
        if not statistical_validation.get('unique_sources', True):
            recommendations.append(f"Find additional sources to reach minimum of {self.thresholds['min_unique_sources']}")
        
        if not statistical_validation.get('source_diversity', True):
            recommendations.append("Improve source diversity by researching from different types of websites and domains")
        
        if not statistical_validation.get('avg_quality', True):
            recommendations.append("Focus on higher-quality sources to improve average content quality")
        
        if not statistical_validation.get('category_coverage', True):
            missing_cats = [cat for cat in self.config.required_query_categories if cat not in metrics.category_coverage]
            if missing_cats:
                recommendations.append(f"Research missing categories: {', '.join(missing_cats)}")
        
        # Gap analysis recommendations
        if gap_analysis:
            if gap_analysis.missing_categories:
                recommendations.append(f"Priority research needed for: {', '.join(gap_analysis.missing_categories)}")
            
            if gap_analysis.weak_areas:
                recommendations.append("Strengthen weak research areas with additional targeted queries")
            
            if gap_analysis.identified_gaps:
                recommendations.append("Address critical knowledge gaps identified in content analysis")
        
        # Quality improvement recommendations
        if metrics.high_quality_percentage < self.config.min_high_quality_percentage:
            recommendations.append("Seek more authoritative sources to increase high-quality content percentage")
        
        if metrics.query_success_rate < 0.8:
            recommendations.append("Review and refine research queries to improve success rate")
        
        # Specific improvement suggestions
        if metrics.total_chunks < self.thresholds['min_total_chunks'] * 1.5:
            recommendations.append("Consider expanding research depth with more specific and detailed queries")
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def _create_insufficient_research_result(self, reason: str, state: BookWritingState) -> Dict[str, Any]:
        """Create result for insufficient research scenarios."""
        return {
            'is_sufficient': False,
            'confidence_score': 0.0,
            'additional_queries': self._generate_basic_additional_queries(state),
            'validation_summary': f"❌ RESEARCH VALIDATION FAILED\n\nReason: {reason}",
            'recommendations': [
                "Conduct initial research using generated queries",
                "Ensure research queries are properly executed",
                "Verify vector database connectivity and data storage"
            ],
            'metrics': {},
            'gap_analysis': None,
            'completed_at': datetime.now(timezone.utc).isoformat(),
            'error': reason
        }
    
    def _generate_basic_additional_queries(self, state: BookWritingState) -> List[ResearchQuery]:
        """Generate basic queries when no research data exists."""
        current_time = datetime.now(timezone.utc).isoformat()
        basic_queries = []
        
        # Generate basic queries for required categories
        for category in self.config.required_query_categories:
            query_text = f"{state['title']} {category.replace('_', ' ')}"
            
            query = ResearchQuery(
                query=query_text,
                priority=5,
                query_type=category,
                status="pending",
                results_count=None,
                created_at=current_time
            )
            
            query['generated_by'] = 'basic_research'
            query['rationale'] = f"Basic research query for {category}"
            basic_queries.append(query)
        
        return basic_queries[:self.config.max_additional_queries]
    
    def _perform_fallback_validation(self, state: BookWritingState) -> Dict[str, Any]:
        """Perform basic fallback validation when main validation fails."""
        logger.warning(f"Performing fallback validation for book {state['book_id']}")
        
        # Count completed queries
        completed_queries = [q for q in state['research_queries'] if q['status'] == 'completed']
        total_results = sum(q.get('results_count', 0) for q in completed_queries)
        
        # Basic sufficiency check
        is_sufficient = (
            len(completed_queries) >= 3 and
            total_results >= self.config.min_total_chunks // 2
        )
        
        return {
            'is_sufficient': is_sufficient,
            'confidence_score': 0.5 if is_sufficient else 0.2,
            'additional_queries': [] if is_sufficient else self._generate_basic_additional_queries(state),
            'validation_summary': f"Basic validation: {'Passed' if is_sufficient else 'Failed'} ({len(completed_queries)} queries, {total_results} results)",
            'recommendations': ["Consider running full validation when systems are available"],
            'metrics': {'fallback_mode': True},
            'gap_analysis': None,
            'completed_at': datetime.now(timezone.utc).isoformat(),
            'fallback_used': True
        }
    
    def _metrics_to_dict(self, metrics: ValidationMetrics) -> Dict[str, Any]:
        """Convert ValidationMetrics to dictionary."""
        return {
            'total_chunks': metrics.total_chunks,
            'unique_sources': metrics.unique_sources,
            'unique_domains': metrics.unique_domains,
            'source_diversity_ratio': metrics.source_diversity_ratio,
            'avg_quality_score': metrics.avg_quality_score,
            'high_quality_percentage': metrics.high_quality_percentage,
            'category_coverage': metrics.category_coverage,
            'query_success_rate': metrics.query_success_rate,
            'content_distribution_score': metrics.content_distribution_score
        }
    
    def _gap_analysis_to_dict(self, gap_analysis: GapAnalysis) -> Dict[str, Any]:
        """Convert GapAnalysis to dictionary."""
        return {
            'identified_gaps': gap_analysis.identified_gaps,
            'missing_categories': gap_analysis.missing_categories,
            'weak_areas': gap_analysis.weak_areas,
            'recommended_additional_queries': gap_analysis.recommended_additional_queries,
            'confidence_score': gap_analysis.confidence_score
        }


def main():
    """Test function for ResearchValidatorAgent."""
    from musequill.agents.agent_state import BookWritingState, ProcessingStage, ResearchQuery, Chapter
    from datetime import datetime, timezone
    
    print("Testing ResearchValidatorAgent...")
    
    # Create test research queries (simulating completed research)
    test_queries = [
        ResearchQuery(
            query="artificial intelligence fundamentals",
            priority=5,
            query_type="background_information",
            status="completed",
            results_count=8,
            created_at=datetime.now(timezone.utc).isoformat()
        ),
        ResearchQuery(
            query="machine learning algorithms technical details",
            priority=5,
            query_type="technical_details",
            status="completed",
            results_count=12,
            created_at=datetime.now(timezone.utc).isoformat()
        ),
        ResearchQuery(
            query="AI ethics expert opinions",
            priority=4,
            query_type="expert_opinions",
            status="completed",
            results_count=6,
            created_at=datetime.now(timezone.utc).isoformat()
        ),
        ResearchQuery(
            query="AI industry current trends",
            priority=3,
            query_type="current_trends",
            status="failed",
            results_count=0,
            created_at=datetime.now(timezone.utc).isoformat()
        )
    ]
    
    # Create test chapters
    test_chapters = [
        Chapter(
            chapter_number=1,
            title="Introduction to AI",
            description="Overview of artificial intelligence",
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
            title="Machine Learning Basics",
            description="Fundamentals of machine learning",
            target_word_count=7000,
            status="planned",
            content=None,
            research_chunks_used=None,
            word_count=None,
            created_at=None,
            completed_at=None
        )
    ]
    
    # Create test state
    test_state = BookWritingState(
        book_id="test_book_validation_123",
        orchestration_id="test_orch_456",
        thread_id="test_thread_789",
        title="The Future of Artificial Intelligence",
        description="Comprehensive guide to AI technologies and their impact",
        genre="Technology/Non-fiction",
        target_word_count=75000,
        target_audience="Technology professionals and students",
        author_preferences={},
        outline={
            "summary": "Comprehensive exploration of AI",
            "themes": ["machine learning", "neural networks", "ethics"]
        },
        chapters=test_chapters,
        current_stage=ProcessingStage.RESEARCH_COMPLETE,
        processing_started_at=datetime.now(timezone.utc).isoformat(),
        processing_updated_at=datetime.now(timezone.utc).isoformat(),
        research_queries=test_queries,
        research_strategy="Comprehensive research strategy for AI book",
        total_research_chunks=26,
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
        progress_percentage=40.0,
        estimated_completion_time=None,
        final_book_content=None,
        metadata={}
    )
    
    try:
        # Create research validator
        validator = ResearchValidatorAgent()
        
        print("Validating research...")
        validation_results = validator.validate_research(test_state)
        
        print(f"\nValidation Results:")
        print(f"Research Sufficient: {validation_results['is_sufficient']}")
        print(f"Confidence Score: {validation_results['confidence_score']:.2f}")
        print(f"Additional Queries Needed: {len(validation_results['additional_queries'])}")
        
        # Show validation summary
        print(f"\nValidation Summary:")
        print(validation_results['validation_summary'])
        
        # Show recommendations
        if validation_results['recommendations']:
            print(f"\nRecommendations:")
            for i, rec in enumerate(validation_results['recommendations'], 1):
                print(f"{i}. {rec}")
        
        # Show additional queries if any
        if validation_results['additional_queries']:
            print(f"\nAdditional Queries Suggested:")
            for query in validation_results['additional_queries']:
                print(f"- {query['query']} (Category: {query['query_type']}, Priority: {query['priority']})")
        
        # Show metrics if available
        if validation_results.get('metrics'):
            metrics = validation_results['metrics']
            print(f"\nDetailed Metrics:")
            print(f"Total Chunks: {metrics.get('total_chunks', 'N/A')}")
            print(f"Unique Sources: {metrics.get('unique_sources', 'N/A')}")
            print(f"Quality Score: {metrics.get('avg_quality_score', 'N/A')}")
        
        print("\nResearchValidatorAgent test completed successfully!")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()