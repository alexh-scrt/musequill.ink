"""
Research Planner Agent

Analyzes book outlines and generates comprehensive research strategies and specific queries.
Creates targeted research plans that guide the research execution phase.

Key Features:
- Analyzes book content and structure to understand research needs
- Generates categorized and prioritized research queries
- Creates comprehensive research strategies
- Validates query quality and uniqueness
- Provides rationale and estimated source counts
- Handles different research depth levels
"""

import json
import re
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from difflib import SequenceMatcher

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel

from musequill.config.logging import get_logger
from musequill.agents.research.research_planner_config import ResearchPlannerConfig
from musequill.agents.agent_state import BookWritingState, ResearchQuery

logger = get_logger(__name__)


@dataclass
class ResearchPlan:
    """Structured research plan output."""
    strategy: str
    queries: List[ResearchQuery]
    total_queries: int
    estimated_total_sources: int
    research_depth: str
    categories_covered: List[str]
    created_at: str
    metadata: Dict[str, Any]


class ResearchQueryModel(BaseModel):
    """Pydantic model for LLM-generated research queries."""
    query: str
    category: str
    priority: int
    rationale: str
    estimated_sources: int
    specific_focus: str


class ResearchQueriesResponse(BaseModel):
    """Pydantic model for LLM response containing multiple queries."""
    queries: List[ResearchQueryModel]
    strategy_summary: str
    research_approach: str


class ResearchPlannerAgent:
    """
    Research Planner Agent that creates comprehensive research plans for books.
    """
    
    def __init__(self, config: Optional[ResearchPlannerConfig] = None):
        if not config:
            config = ResearchPlannerConfig()
        
        self.config = config
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            api_key=self.config.openai_api_key,
            model=self.config.llm_model,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens
        )
        
        logger.info("Research Planner Agent initialized")
    
    def create_research_plan(self, state: BookWritingState) -> Dict[str, Any]:
        """
        Create a comprehensive research plan for the book.
        
        Args:
            state: Current BookWritingState containing book information
            
        Returns:
            Dictionary containing research strategy and queries
        """
        try:
            logger.info(f"Creating research plan for book {state['book_id']}")
            
            # Analyze book content to determine research needs
            analysis = self._analyze_book_content(state)
            
            # Generate research queries using LLM
            research_queries = self._generate_research_queries(state, analysis)
            
            # Validate and process queries
            processed_queries = self._process_and_validate_queries(research_queries)
            
            # Create research strategy
            strategy = self._create_research_strategy(state, analysis, processed_queries)
            
            # Build final research plan
            plan = ResearchPlan(
                strategy=strategy,
                queries=processed_queries,
                total_queries=len(processed_queries),
                estimated_total_sources=sum(q['estimated_sources'] for q in processed_queries),
                research_depth=analysis['recommended_depth'],
                categories_covered=list(set(q['query_type'] for q in processed_queries)),
                created_at=datetime.now(timezone.utc).isoformat(),
                metadata={
                    'analysis': analysis,
                    'generation_method': 'llm_guided',
                    'config_used': {
                        'max_queries': self.config.max_research_queries,
                        'min_queries': self.config.min_research_queries,
                        'model': self.config.llm_model
                    }
                }
            )
            
            logger.info(f"Research plan created for book {state['book_id']}: {plan.total_queries} queries, {plan.estimated_total_sources} estimated sources")
            
            return {
                'strategy': plan.strategy,
                'queries': plan.queries,
                'metadata': plan.metadata
            }
            
        except Exception as e:
            logger.error(f"Error creating research plan for book {state['book_id']}: {e}")
            # Return fallback plan
            return self._create_fallback_plan(state)
    
    def _analyze_book_content(self, state: BookWritingState) -> Dict[str, Any]:
        """
        Analyze book content to understand research requirements.
        
        Args:
            state: BookWritingState containing book information
            
        Returns:
            Analysis results dictionary
        """
        try:
            title = state['title']
            description = state['description']
            genre = state['genre']
            target_word_count = state['target_word_count']
            chapters = state['chapters']
            outline = state['outline']
            
            # Determine research complexity based on content
            complexity_score = self._calculate_complexity_score(state)
            
            # Determine recommended research depth
            if complexity_score >= 7:
                recommended_depth = "deep"
            elif complexity_score >= 4:
                recommended_depth = "intermediate"
            else:
                recommended_depth = "surface"
            
            # Analyze chapter content for specific research needs
            chapter_analysis = []
            for chapter in chapters:
                if chapter.get('description') and len(chapter['description']) >= self.config.min_chapter_analysis_length:
                    chapter_analysis.append({
                        'chapter_number': chapter['chapter_number'],
                        'title': chapter['title'],
                        'description': chapter['description'],
                        'research_needs': self._extract_research_needs(chapter['description'])
                    })
            
            # Identify key research areas
            research_areas = self._identify_research_areas(title, description, genre, outline)
            
            analysis = {
                'complexity_score': complexity_score,
                'recommended_depth': recommended_depth,
                'research_areas': research_areas,
                'chapter_analysis': chapter_analysis,
                'genre_specific_needs': self._get_genre_specific_needs(genre),
                'estimated_query_count': min(
                    max(complexity_score * 2, self.config.min_research_queries),
                    self.config.max_research_queries
                ),
                'priority_areas': research_areas[:3]  # Top 3 priority areas
            }
            
            logger.info(f"Book analysis completed: complexity={complexity_score}, depth={recommended_depth}, areas={len(research_areas)}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing book content: {e}")
            return self._create_fallback_analysis(state)
    
    def _calculate_complexity_score(self, state: BookWritingState) -> int:
        """Calculate research complexity score (1-10) based on book characteristics."""
        score = 1
        
        # Genre complexity
        complex_genres = ['science fiction', 'fantasy', 'historical', 'technical', 'biography', 'non-fiction']
        if any(genre.lower() in state['genre'].lower() for genre in complex_genres):
            score += 2
        
        # Length complexity
        if state['target_word_count'] > 100000:
            score += 2
        elif state['target_word_count'] > 50000:
            score += 1
        
        # Chapter count
        if len(state['chapters']) > 15:
            score += 1
        
        # Description complexity (technical terms, specific topics)
        description = state['description'].lower()
        technical_indicators = ['research', 'analysis', 'scientific', 'technical', 'historical', 'factual', 'data']
        score += sum(1 for term in technical_indicators if term in description)
        
        # Outline complexity
        if isinstance(state['outline'], dict):
            if 'research_requirements' in state['outline']:
                score += 2
            if 'bibliography' in state['outline'] or 'sources' in state['outline']:
                score += 1
        
        return min(score, 10)  # Cap at 10
    
    def _extract_research_needs(self, text: str) -> List[str]:
        """Extract potential research needs from text content."""
        needs = []
        text_lower = text.lower()
        
        # Look for research indicators
        research_patterns = [
            r'research\s+(\w+(?:\s+\w+){0,2})',
            r'study\s+(\w+(?:\s+\w+){0,2})',
            r'analyze\s+(\w+(?:\s+\w+){0,2})',
            r'investigate\s+(\w+(?:\s+\w+){0,2})',
            r'examine\s+(\w+(?:\s+\w+){0,2})',
            r'explore\s+(\w+(?:\s+\w+){0,2})'
        ]
        
        for pattern in research_patterns:
            matches = re.findall(pattern, text_lower)
            needs.extend(matches)
        
        return list(set(needs))  # Remove duplicates
    
    def _identify_research_areas(self, title: str, description: str, genre: str, outline: Dict[str, Any]) -> List[str]:
        """Identify key research areas based on book content."""
        areas = []
        
        # Genre-based areas
        genre_areas = {
            'science fiction': ['scientific concepts', 'future technologies', 'space exploration'],
            'fantasy': ['mythology', 'world building', 'magical systems'],
            'historical': ['historical events', 'period details', 'historical figures'],
            'mystery': ['investigative procedures', 'forensics', 'crime patterns'],
            'romance': ['relationship dynamics', 'social contexts', 'emotional psychology'],
            'thriller': ['action sequences', 'suspense techniques', 'modern threats'],
            'biography': ['historical context', 'personal relationships', 'achievements'],
            'non-fiction': ['expert opinions', 'current research', 'statistical data']
        }
        
        for genre_key, genre_specific_areas in genre_areas.items():
            if genre_key.lower() in genre.lower():
                areas.extend(genre_specific_areas)
        
        # Extract from title and description
        combined_text = f"{title} {description}".lower()
        
        # Look for specific topics mentioned
        topic_indicators = [
            'technology', 'science', 'history', 'politics', 'economics', 
            'psychology', 'sociology', 'culture', 'religion', 'philosophy',
            'business', 'finance', 'medicine', 'law', 'education', 'art',
            'environment', 'climate', 'energy', 'space', 'military'
        ]
        
        for topic in topic_indicators:
            if topic in combined_text:
                areas.append(topic)
        
        # Add outline-specific areas
        if isinstance(outline, dict):
            for key, value in outline.items():
                if isinstance(value, str) and len(value) > 20:
                    # Extract potential research topics from outline content
                    words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', value)
                    areas.extend(words[:3])  # Add up to 3 capitalized terms
        
        # Remove duplicates and return sorted
        unique_areas = list(set(areas))
        return sorted(unique_areas)[:10]  # Limit to top 10 areas
    
    def _get_genre_specific_needs(self, genre: str) -> List[str]:
        """Get specific research needs based on genre."""
        genre_needs = {
            'science fiction': [
                'scientific accuracy for fictional technologies',
                'current scientific research and trends',
                'expert opinions on future possibilities'
            ],
            'fantasy': [
                'mythological and folklore references',
                'world-building consistency',
                'fantasy literature conventions'
            ],
            'historical': [
                'historical accuracy and context',
                'period-specific details',
                'primary and secondary historical sources'
            ],
            'mystery': [
                'police procedures and forensics',
                'crime statistics and patterns',
                'investigative techniques'
            ],
            'romance': [
                'relationship psychology',
                'social and cultural contexts',
                'emotional authenticity'
            ],
            'non-fiction': [
                'expert interviews and opinions',
                'statistical data and research studies',
                'current trends and developments'
            ]
        }
        
        for genre_key, needs in genre_needs.items():
            if genre_key.lower() in genre.lower():
                return needs
        
        return ['general background information', 'expert perspectives', 'supporting examples']
    
    def _generate_research_queries(self, state: BookWritingState, analysis: Dict[str, Any]) -> List[ResearchQuery]:
        """
        Generate research queries using LLM.
        
        Args:
            state: BookWritingState
            analysis: Book content analysis
            
        Returns:
            List of ResearchQuery objects
        """
        try:
            system_prompt = self._create_research_generation_prompt()
            human_prompt = self._create_book_context_prompt(state, analysis)
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            # Use structured output with Pydantic
            structured_llm = self.llm.with_structured_output(ResearchQueriesResponse)
            
            response = structured_llm.invoke(messages)
            
            # Convert to ResearchQuery format
            research_queries = []
            for i, query_model in enumerate(response.queries):
                priority = self._map_priority_to_number(query_model.priority, query_model.category)
                
                research_query = ResearchQuery(
                    query=query_model.query,
                    priority=priority,
                    query_type=query_model.category,
                    status="pending",
                    results_count=None,
                    created_at=datetime.now(timezone.utc).isoformat()
                )
                
                # Add additional metadata
                research_query['rationale'] = query_model.rationale if self.config.include_research_rationale else ""
                research_query['estimated_sources'] = query_model.estimated_sources if self.config.include_estimated_sources else 5
                research_query['specific_focus'] = query_model.specific_focus
                
                research_queries.append(research_query)
            
            logger.info(f"Generated {len(research_queries)} research queries using LLM")
            
            return research_queries
            
        except Exception as e:
            logger.error(f"Error generating research queries with LLM: {e}")
            # Fallback to rule-based generation
            return self._generate_fallback_queries(state, analysis)
    
    def _create_research_generation_prompt(self) -> str:
        """Create the system prompt for research query generation."""
        categories_text = ", ".join(self.config.query_categories)
        
        return f"""You are an expert research planner helping to create comprehensive research strategies for book writing.

Your task is to generate specific, targeted research queries that will help gather authoritative information for writing a book. Each query should be designed to find factual information, expert insights, examples, or data that will enhance the book's content.

QUERY CATEGORIES (use these exactly):
{categories_text}

PRIORITY LEVELS:
- 5 (Critical): Essential for book credibility and accuracy
- 4 (High): Important for depth and quality  
- 3 (Medium): Useful for enrichment and examples
- 2 (Low): Nice to have for additional context
- 1 (Optional): Background or supplementary information

GUIDELINES:
1. Create queries that are specific and focused (not too broad)
2. Ensure queries will return authoritative, reliable information
3. Focus on queries that will provide unique value to the book
4. Include a mix of factual data, expert opinions, and practical examples
5. Consider the target audience and book's purpose
6. Make queries searchable (good for web search engines)
7. Avoid duplicate or overly similar queries

For each query, provide:
- The specific search query (optimized for web search)
- Category from the approved list
- Priority level (1-5)
- Brief rationale for why this research is valuable
- Estimated number of quality sources this query might return (1-20)
- Specific focus area within the broader topic

Generate between {self.config.min_research_queries} and {self.config.max_research_queries} queries."""
    
    def _create_book_context_prompt(self, state: BookWritingState, analysis: Dict[str, Any]) -> str:
        """Create the human prompt with book context."""
        chapters_text = ""
        if state['chapters']:
            chapters_text = "\n".join([
                f"Chapter {ch['chapter_number']}: {ch['title']} - {ch.get('description', 'No description')}"
                for ch in state['chapters'][:10]  # Limit to first 10 chapters
            ])
        
        research_areas_text = ", ".join(analysis['research_areas'])
        
        return f"""BOOK INFORMATION:
Title: {state['title']}
Genre: {state['genre']}
Target Word Count: {state['target_word_count']:,} words
Target Audience: {state.get('target_audience', 'General audience')}

Description:
{state['description']}

Chapter Overview:
{chapters_text if chapters_text else "No detailed chapter information available"}

RESEARCH ANALYSIS:
Complexity Score: {analysis['complexity_score']}/10
Recommended Research Depth: {analysis['recommended_depth']}
Key Research Areas: {research_areas_text}
Priority Research Areas: {', '.join(analysis['priority_areas'])}

REQUIREMENTS:
- Generate {analysis['estimated_query_count']} high-quality research queries
- Focus on the priority research areas identified above
- Ensure queries will help create authoritative, well-researched content
- Include both broad foundational research and specific detailed queries
- Consider the genre-specific research needs for {state['genre']}

Please generate a comprehensive research plan with specific, actionable research queries."""
    
    def _map_priority_to_number(self, llm_priority: int, category: str) -> int:
        """Map LLM priority and category to final priority number."""
        # Use LLM priority as base
        base_priority = max(1, min(5, llm_priority))
        
        # Adjust based on category configuration
        if category in self.config.high_priority_categories:
            return min(5, base_priority + 1)
        elif category in self.config.low_priority_categories:
            return max(1, base_priority - 1)
        
        return base_priority
    
    def _process_and_validate_queries(self, queries: List[ResearchQuery]) -> List[ResearchQuery]:
        """
        Process and validate research queries for quality and uniqueness.
        
        Args:
            queries: List of raw research queries
            
        Returns:
            List of processed and validated queries
        """
        if not self.config.enable_query_validation:
            return queries
        
        try:
            processed_queries = []
            seen_queries = []
            
            for query in queries:
                # Clean and validate query text
                cleaned_query = self._clean_query_text(query['query'])
                if not self._is_valid_query(cleaned_query):
                    logger.warning(f"Skipping invalid query: {cleaned_query}")
                    continue
                
                # Check for duplicates/similarity
                if self._is_query_unique(cleaned_query, seen_queries):
                    query_copy = query.copy()
                    query_copy['query'] = cleaned_query
                    processed_queries.append(query_copy)
                    seen_queries.append(cleaned_query)
                else:
                    logger.info(f"Skipping duplicate/similar query: {cleaned_query}")
            
            # Ensure we have minimum number of queries
            if len(processed_queries) < self.config.min_research_queries:
                logger.warning(f"Only {len(processed_queries)} valid queries generated, minimum is {self.config.min_research_queries}")
                # Could add fallback logic here
            
            # Sort by priority (highest first)
            processed_queries.sort(key=lambda x: x['priority'], reverse=True)
            
            logger.info(f"Processed {len(processed_queries)} valid queries from {len(queries)} generated")
            
            return processed_queries
            
        except Exception as e:
            logger.error(f"Error processing queries: {e}")
            return queries  # Return original queries if processing fails
    
    def _clean_query_text(self, query: str) -> str:
        """Clean and normalize query text."""
        # Remove extra whitespace
        cleaned = " ".join(query.split())
        
        # Remove quotes if they wrap the entire query
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1]
        
        # Ensure it doesn't end with punctuation that might interfere with search
        if cleaned.endswith(('?', '!', '.')):
            cleaned = cleaned[:-1]
        
        return cleaned.strip()
    
    def _is_valid_query(self, query: str) -> bool:
        """Validate that a query is suitable for research."""
        if not query or len(query.strip()) < 3:
            return False
        
        if len(query) > 200:  # Too long for effective search
            return False
        
        # Check for obvious invalid patterns
        invalid_patterns = [
            r'^(what|how|why|when|where)\s*$',  # Single question words
            r'^\d+$',  # Just numbers
            r'^[^a-zA-Z]*$'  # No letters
        ]
        
        for pattern in invalid_patterns:
            if re.match(pattern, query.lower()):
                return False
        
        return True
    
    def _is_query_unique(self, query: str, existing_queries: List[str]) -> bool:
        """Check if query is sufficiently unique compared to existing queries."""
        for existing in existing_queries:
            similarity = SequenceMatcher(None, query.lower(), existing.lower()).ratio()
            if similarity > self.config.query_uniqueness_threshold:
                return False
        return True
    
    def _create_research_strategy(self, state: BookWritingState, analysis: Dict[str, Any], queries: List[ResearchQuery]) -> str:
        """
        Create a comprehensive research strategy description.
        
        Args:
            state: BookWritingState
            analysis: Book content analysis  
            queries: Processed research queries
            
        Returns:
            Research strategy text
        """
        try:
            # Categorize queries by type
            query_categories = {}
            for query in queries:
                category = query['query_type']
                if category not in query_categories:
                    query_categories[category] = []
                query_categories[category].append(query)
            
            # Count priorities
            priority_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            for query in queries:
                priority_counts[query['priority']] += 1
            
            strategy = f"""RESEARCH STRATEGY FOR "{state['title']}"

OVERVIEW:
This research plan is designed for a {state['genre']} book targeting {state.get('target_audience', 'general audience')} with approximately {state['target_word_count']:,} words. The research complexity has been assessed as {analysis['complexity_score']}/10, requiring {analysis['recommended_depth']} research depth.

RESEARCH APPROACH:
The research will focus on {len(analysis['research_areas'])} key areas: {', '.join(analysis['priority_areas'])}. A total of {len(queries)} research queries have been generated across {len(query_categories)} categories to ensure comprehensive coverage.

QUERY BREAKDOWN:
"""
            
            # Add category breakdown
            for category, cat_queries in query_categories.items():
                strategy += f"- {category.replace('_', ' ').title()}: {len(cat_queries)} queries\n"
            
            strategy += f"""
PRIORITY DISTRIBUTION:
- Critical (Priority 5): {priority_counts[5]} queries
- High (Priority 4): {priority_counts[4]} queries  
- Medium (Priority 3): {priority_counts[3]} queries
- Low (Priority 2): {priority_counts[2]} queries
- Optional (Priority 1): {priority_counts[1]} queries

EXECUTION PLAN:
1. Begin with Priority 5 (Critical) queries to establish foundational knowledge
2. Progress through Priority 4 (High) queries for depth and authority
3. Complete Priority 3 (Medium) queries for enrichment and examples
4. Address lower priority queries as time and resources permit

EXPECTED OUTCOMES:
This research strategy is designed to gather approximately {sum(q.get('estimated_sources', 5) for q in queries)} sources across all queries, providing comprehensive coverage for authoritative book content.

QUALITY STANDARDS:
All research will prioritize authoritative sources, expert opinions, and verifiable information to ensure the book meets high standards of accuracy and credibility."""
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error creating research strategy: {e}")
            return f"Research strategy for {state['title']} with {len(queries)} queries across multiple categories."
    
    def _generate_fallback_queries(self, state: BookWritingState, analysis: Dict[str, Any]) -> List[ResearchQuery]:
        """Generate basic fallback queries when LLM generation fails."""
        fallback_queries = []
        current_time = datetime.now(timezone.utc).isoformat()
        
        # Basic queries based on book information
        basic_patterns = [
            f"{state['title']} background information",
            f"{state['genre']} writing techniques",
            f"expert opinions on {state['title']}",
            f"current trends in {state['genre']}",
            f"research methods for {state['genre']} writing"
        ]
        
        # Add research area specific queries
        for area in analysis['research_areas'][:5]:
            basic_patterns.append(f"{area} information for {state['genre']} writing")
        
        # Create ResearchQuery objects
        for i, pattern in enumerate(basic_patterns[:self.config.max_research_queries]):
            query = ResearchQuery(
                query=pattern,
                priority=5 if i < 3 else 3,  # First 3 are high priority
                query_type="background_information",
                status="pending",
                results_count=None,
                created_at=current_time
            )
            query['estimated_sources'] = 5
            query['rationale'] = f"Fallback query for basic research on {pattern}"
            fallback_queries.append(query)
        
        logger.warning(f"Generated {len(fallback_queries)} fallback queries")
        return fallback_queries
    
    def _create_fallback_analysis(self, state: BookWritingState) -> Dict[str, Any]:
        """Create basic fallback analysis when detailed analysis fails."""
        return {
            'complexity_score': 5,  # Medium complexity
            'recommended_depth': self.config.default_research_depth,
            'research_areas': [state['genre'], 'writing techniques', 'background information'],
            'chapter_analysis': [],
            'genre_specific_needs': ['general background', 'expert opinions', 'examples'],
            'estimated_query_count': self.config.min_research_queries,
            'priority_areas': [state['genre'], 'writing techniques', 'background information']
        }
    
    def _create_fallback_plan(self, state: BookWritingState) -> Dict[str, Any]:
        """Create a basic fallback research plan when main planning fails."""
        logger.warning(f"Creating fallback research plan for book {state['book_id']}")
        
        fallback_analysis = self._create_fallback_analysis(state)
        fallback_queries = self._generate_fallback_queries(state, fallback_analysis)
        
        return {
            'strategy': f"Basic research strategy for {state['title']} focusing on {state['genre']} background information and writing techniques.",
            'queries': fallback_queries,
            'metadata': {
                'fallback_used': True,
                'created_at': datetime.now(timezone.utc).isoformat()
            }
        }
    
    def validate_research_plan(self, plan: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a research plan for completeness and quality.
        
        Args:
            plan: Research plan dictionary
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required fields
        required_fields = ['strategy', 'queries', 'metadata']
        for field in required_fields:
            if field not in plan:
                issues.append(f"Missing required field: {field}")
        
        # Validate queries
        if 'queries' in plan:
            queries = plan['queries']
            
            if not isinstance(queries, list):
                issues.append("Queries must be a list")
            elif len(queries) < self.config.min_research_queries:
                issues.append(f"Too few queries: {len(queries)} < {self.config.min_research_queries}")
            elif len(queries) > self.config.max_research_queries:
                issues.append(f"Too many queries: {len(queries)} > {self.config.max_research_queries}")
            else:
                # Validate individual queries
                for i, query in enumerate(queries):
                    if not isinstance(query, dict):
                        issues.append(f"Query {i} is not a dictionary")
                        continue
                    
                    # Check required query fields
                    required_query_fields = ['query', 'priority', 'query_type', 'status']
                    for field in required_query_fields:
                        if field not in query:
                            issues.append(f"Query {i} missing field: {field}")
                    
                    # Validate priority
                    if 'priority' in query and not (1 <= query['priority'] <= 5):
                        issues.append(f"Query {i} has invalid priority: {query['priority']}")
                    
                    # Validate query text
                    if 'query' in query and not self._is_valid_query(query['query']):
                        issues.append(f"Query {i} has invalid query text: {query['query']}")
        
        # Validate strategy
        if 'strategy' in plan:
            strategy = plan['strategy']
            if not isinstance(strategy, str) or len(strategy.strip()) < 50:
                issues.append("Strategy must be a meaningful text description (min 50 characters)")
        
        return len(issues) == 0, issues
    
    def get_research_statistics(self, queries: List[ResearchQuery]) -> Dict[str, Any]:
        """
        Generate statistics about the research plan.
        
        Args:
            queries: List of research queries
            
        Returns:
            Statistics dictionary
        """
        if not queries:
            return {}
        
        # Category distribution
        category_counts = {}
        for query in queries:
            category = query['query_type']
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Priority distribution
        priority_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for query in queries:
            priority_counts[query['priority']] += 1
        
        # Estimated sources
        total_estimated_sources = sum(query.get('estimated_sources', 5) for query in queries)
        avg_sources_per_query = total_estimated_sources / len(queries) if queries else 0
        
        # Query length statistics
        query_lengths = [len(query['query']) for query in queries]
        avg_query_length = sum(query_lengths) / len(query_lengths) if query_lengths else 0
        
        return {
            'total_queries': len(queries),
            'category_distribution': category_counts,
            'priority_distribution': priority_counts,
            'estimated_total_sources': total_estimated_sources,
            'avg_sources_per_query': round(avg_sources_per_query, 1),
            'avg_query_length': round(avg_query_length, 1),
            'categories_covered': len(category_counts),
            'high_priority_queries': priority_counts[4] + priority_counts[5],
            'low_priority_queries': priority_counts[1] + priority_counts[2]
        }


def main():
    """Test function for ResearchPlannerAgent."""
    from musequill.agents.agent_state import BookWritingState, ProcessingStage, Chapter
    from datetime import datetime, timezone
    
    print("Testing ResearchPlannerAgent...")
    
    # Create test state
    test_chapters = [
        Chapter(
            chapter_number=1,
            title="Introduction to AI",
            description="Overview of artificial intelligence history and current state",
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
            description="Basic concepts and algorithms in machine learning",
            target_word_count=7000,
            status="planned",
            content=None,
            research_chunks_used=None,
            word_count=None,
            created_at=None,
            completed_at=None
        )
    ]
    
    test_state = BookWritingState(
        book_id="test_book_ai_123",
        orchestration_id="test_orch_456",
        thread_id="test_thread_789",
        title="The Future of Artificial Intelligence",
        description="A comprehensive guide to understanding AI technologies, their current applications, and future potential in various industries.",
        genre="Technology/Non-fiction",
        target_word_count=75000,
        target_audience="Technology professionals and enthusiasts",
        author_preferences={},
        outline={
            "summary": "Comprehensive exploration of AI technologies",
            "themes": ["machine learning", "neural networks", "ethics", "future applications"]
        },
        chapters=test_chapters,
        current_stage=ProcessingStage.INITIALIZED,
        processing_started_at=datetime.now(timezone.utc).isoformat(),
        processing_updated_at=datetime.now(timezone.utc).isoformat(),
        research_queries=[],
        research_strategy=None,
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
        # Create research planner
        planner = ResearchPlannerAgent()
        
        print("Creating research plan...")
        research_plan = planner.create_research_plan(test_state)
        
        print(f"\nResearch Plan Created!")
        print(f"Strategy length: {len(research_plan['strategy'])} characters")
        print(f"Number of queries: {len(research_plan['queries'])}")
        
        # Show first few queries
        print("\nFirst 3 Research Queries:")
        for i, query in enumerate(research_plan['queries'][:3]):
            print(f"{i+1}. {query['query']}")
            print(f"   Category: {query['query_type']}, Priority: {query['priority']}")
            print()
        
        # Validate the plan
        is_valid, issues = planner.validate_research_plan(research_plan)
        print(f"Plan validation: {'PASSED' if is_valid else 'FAILED'}")
        if issues:
            print(f"Issues: {issues}")
        
        # Get statistics
        stats = planner.get_research_statistics(research_plan['queries'])
        print(f"\nResearch Statistics:")
        print(f"Total queries: {stats['total_queries']}")
        print(f"Categories covered: {stats['categories_covered']}")
        print(f"High priority queries: {stats['high_priority_queries']}")
        print(f"Estimated total sources: {stats['estimated_total_sources']}")
        
        print("\nResearchPlannerAgent test completed successfully!")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()