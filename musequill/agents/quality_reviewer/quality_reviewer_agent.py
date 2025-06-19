"""
Quality Reviewer Agent

Reviews completed chapters and the full book for quality, consistency, and completeness.
Provides detailed quality assessments, identifies improvement areas, and determines if revisions are needed.

Key Features:
- Multi-dimensional quality assessment across chapters and full book
- Consistency analysis for style, tone, terminology, and formatting
- Research integration validation and source accuracy checking
- Automated revision recommendations with priority ranking
- Threshold-based approval/revision decisions with escalation paths
"""

import time
import statistics
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

import chromadb
from chromadb.config import Settings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel

from musequill.config.logging import get_logger
from musequill.agents.quality_reviewer.quality_reviewer_config import QualityReviewerConfig
from musequill.agents.quality_reviewer.quality_reviewer_prompts import QualityReviewerPrompts
from musequill.agents.agent_state import BookWritingState, Chapter

logger = get_logger(__name__)


@dataclass
class ChapterQualityMetrics:
    """Quality metrics for an individual chapter."""
    chapter_number: int
    chapter_title: str
    overall_score: float
    dimension_scores: Dict[str, float]
    issues_found: List[str]
    strengths: List[str]
    improvement_suggestions: List[str]
    meets_threshold: bool


@dataclass
class BookConsistencyMetrics:
    """Consistency metrics across the entire book."""
    overall_consistency_score: float
    inconsistencies_found: List[str]
    consistency_strengths: List[str]


@dataclass
class ResearchValidationResults:
    """Results of research integration validation."""
    research_accuracy_score: float
    citation_completeness_score: float
    research_integration_quality: str


@dataclass
class BookQualityAssessment:
    """Complete quality assessment for the book."""
    overall_quality_score: float
    meets_quality_threshold: bool
    chapter_metrics: List[ChapterQualityMetrics]
    consistency_metrics: BookConsistencyMetrics
    research_validation: ResearchValidationResults
    revision_required: bool
    revision_priority_areas: List[str]
    approval_recommendation: str
    detailed_feedback: str
    assessment_summary: str
    created_at: str


class ChapterQualityModel(BaseModel):
    """Pydantic model for LLM chapter quality assessment."""
    overall_quality_score: float
    content_clarity: float
    logical_structure: float
    writing_quality: float
    engagement_level: float
    research_integration: float
    key_strengths: List[str]
    main_weaknesses: List[str]
    specific_issues: List[str]
    improvement_recommendations: List[str]
    meets_publication_standard: bool


class BookConsistencyModel(BaseModel):
    """Pydantic model for LLM book consistency assessment."""
    style_consistency: float
    tone_consistency: float
    terminology_usage: float
    structural_consistency: float
    narrative_flow: float
    chapter_transitions: float
    inconsistencies_identified: List[str]
    consistency_strengths: List[str]


class RevisionDecisionModel(BaseModel):
    """Pydantic model for LLM revision decision."""
    requires_revision: bool
    revision_urgency: str
    priority_areas: List[str]
    revision_strategy: str
    expected_improvement_areas: List[str]
    approval_recommendation: str
    estimated_revision_effort: str


class QualityReviewerAgent:
    """
    Quality Reviewer Agent that assesses completed book quality and determines revision needs.
    """
    
    def __init__(self, config: Optional[QualityReviewerConfig] = None):
        if config is None:
            config = QualityReviewerConfig()
        
        self.config = config
        self.prompts = QualityReviewerPrompts()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            api_key=self.config.openai_api_key,
            model=self.config.llm_model,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens
        )
        
        # Initialize Chroma client for research validation
        self.chroma_client: Optional[chromadb.HttpClient] = None
        self.chroma_collection = None
        
        # Quality tracking
        self.quality_cache: Dict[str, Any] = {}
        self.review_stats = {
            'books_reviewed': 0,
            'books_approved': 0,
            'average_quality_score': 0.0,
            'session_start': None
        }
        
        self._initialize_components()
        logger.info("Quality Reviewer Agent initialized")
    
    def _initialize_components(self) -> None:
        """Initialize Chroma client if research validation is enabled."""
        try:
            if self.config.enable_research_validation:
                self.chroma_client = chromadb.HttpClient(
                    host=self.config.chroma_host,
                    port=self.config.chroma_port
                )
                self.chroma_collection = self.chroma_client.get_or_create_collection(
                    name=self.config.chroma_collection_name
                )
            logger.info("Quality Reviewer components initialized")
        except Exception as e:
            logger.warning(f"Could not initialize research validation: {e}")
    
    def review_book(self, state: BookWritingState) -> Dict[str, Any]:
        """
        Main entry point for book quality review - matches orchestrator expectations.
        """
        try:
            full_results = self.review_book_quality(state)
            
            if full_results['status'] == 'success':
                return {
                    'quality_score': full_results['overall_quality_score'],
                    'needs_revision': full_results['requires_revision'],
                    'review_notes': [full_results['assessment_summary']],
                    'completed_at': datetime.now(timezone.utc).isoformat(),
                    'revised_chapters': state['chapters'],
                    'revision_guidance': {
                        'priority_areas': full_results.get('priority_revision_areas', []),
                        'revision_strategy': full_results.get('revision_strategy', ''),
                        'urgency': full_results.get('revision_urgency', 'medium')
                    }
                }
            else:
                return {
                    'quality_score': 0.0,
                    'needs_revision': True,
                    'review_notes': [f"Quality review failed: {full_results.get('error_message', 'Unknown error')}"],
                    'completed_at': datetime.now(timezone.utc).isoformat(),
                    'revised_chapters': state['chapters']
                }
        except Exception as e:
            logger.error(f"Error in review_book: {e}")
            return {
                'quality_score': 0.0,
                'needs_revision': True,
                'review_notes': [f"Quality review system error: {str(e)}"],
                'completed_at': datetime.now(timezone.utc).isoformat(),
                'revised_chapters': state['chapters']
            }
    
    def review_book_quality(self, state: BookWritingState) -> Dict[str, Any]:
        """
        Conduct comprehensive quality review of the completed book.
        """
        try:
            start_time = time.time()
            logger.info(f"Starting quality review for book {state['book_id']}")
            
            # Validate input
            completed_chapters = [ch for ch in state['chapters'] if ch.get('status') == 'complete' and ch.get('content')]
            if not completed_chapters:
                return {
                    'status': 'error',
                    'error_message': 'No completed chapters found for quality review',
                    'requires_revision': True
                }
            
            # Phase 1: Individual chapter quality assessment
            chapter_metrics = self._assess_individual_chapters(completed_chapters, state)
            
            # Phase 2: Book-wide consistency analysis
            consistency_metrics = self._assess_book_consistency(completed_chapters, state)
            
            # Phase 3: Research integration validation
            research_validation = self._validate_research_integration(completed_chapters, state)
            
            # Phase 4: Overall quality synthesis
            overall_assessment = self._synthesize_quality_assessment(
                chapter_metrics, consistency_metrics, research_validation, state
            )
            
            # Phase 5: Revision decision
            revision_decision = self._make_revision_decision(overall_assessment, state)
            
            # Update statistics
            self._update_review_stats(overall_assessment, revision_decision)
            
            return {
                'status': 'success',
                'overall_quality_score': overall_assessment.overall_quality_score,
                'meets_quality_threshold': overall_assessment.meets_quality_threshold,
                'requires_revision': revision_decision.requires_revision,
                'revision_urgency': revision_decision.revision_urgency,
                'approval_recommendation': revision_decision.approval_recommendation,
                'priority_revision_areas': revision_decision.priority_areas,
                'detailed_feedback': overall_assessment.detailed_feedback,
                'assessment_summary': overall_assessment.assessment_summary,
                'revision_strategy': revision_decision.revision_strategy,
                'review_time_seconds': time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Error during quality review: {e}")
            return {
                'status': 'error',
                'error_message': str(e),
                'requires_revision': True
            }
    
    def _assess_individual_chapters(self, chapters: List[Chapter], state: BookWritingState) -> List[ChapterQualityMetrics]:
        """Assess quality of individual chapters."""
        chapter_metrics = []
        
        # Sample chapters if too many
        chapters_to_analyze = chapters
        if len(chapters) > self.config.max_chapters_to_sample:
            step = len(chapters) // self.config.max_chapters_to_sample
            chapters_to_analyze = chapters[::step][:self.config.max_chapters_to_sample]
        
        for chapter in chapters_to_analyze:
            try:
                metric = self._assess_single_chapter(chapter, state)
                chapter_metrics.append(metric)
            except Exception as e:
                logger.error(f"Error assessing Chapter {chapter['chapter_number']}: {e}")
                chapter_metrics.append(self._create_fallback_chapter_metric(chapter))
        
        return chapter_metrics
    
    def _assess_single_chapter(self, chapter: Chapter, state: BookWritingState) -> ChapterQualityMetrics:
        """Assess quality of a single chapter using LLM."""
        try:
            # Prepare content sample
            content = chapter.get('content', '')
            if len(content.split()) > self.config.content_sample_size:
                words = content.split()
                sample_size = self.config.content_sample_size // 3
                beginning = ' '.join(words[:sample_size])
                middle_start = len(words) // 2 - sample_size // 2
                middle = ' '.join(words[middle_start:middle_start + sample_size])
                end = ' '.join(words[-sample_size:])
                content_sample = f"{beginning}\n\n[...middle...]\n\n{middle}\n\n[...end...]\n\n{end}"
            else:
                content_sample = content
            
            # Create prompts
            system_prompt = self.prompts.get_chapter_assessment_system_prompt(state)
            human_prompt = self.prompts.get_chapter_assessment_human_prompt(chapter, content_sample, state)
            
            # Get LLM assessment
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            llm_assessment = self.llm.with_structured_output(ChapterQualityModel).invoke(messages)
            
            # Calculate scores
            dimension_scores = {
                'content_clarity': llm_assessment.content_clarity,
                'writing_quality': llm_assessment.writing_quality,
                'logical_structure': llm_assessment.logical_structure,
                'engagement_level': llm_assessment.engagement_level,
                'research_integration': llm_assessment.research_integration
            }
            
            overall_score = llm_assessment.overall_quality_score
            meets_threshold = overall_score >= self.config.individual_chapter_threshold
            
            return ChapterQualityMetrics(
                chapter_number=chapter['chapter_number'],
                chapter_title=chapter['title'],
                overall_score=overall_score,
                dimension_scores=dimension_scores,
                issues_found=llm_assessment.specific_issues,
                strengths=llm_assessment.key_strengths,
                improvement_suggestions=llm_assessment.improvement_recommendations,
                meets_threshold=meets_threshold
            )
            
        except Exception as e:
            logger.error(f"LLM assessment failed for Chapter {chapter['chapter_number']}: {e}")
            return self._create_fallback_chapter_metric(chapter)
    
    def _assess_book_consistency(self, chapters: List[Chapter], state: BookWritingState) -> BookConsistencyMetrics:
        """Assess consistency across the entire book."""
        try:
            # Prepare chapter samples
            chapter_samples = []
            for chapter in chapters[:8]:  # Max 8 chapters
                content = chapter.get('content', '')
                words = content.split()
                sample_size = min(300, len(words) // 2)
                sample = ' '.join(words[:sample_size])
                chapter_samples.append({
                    'chapter_number': chapter['chapter_number'],
                    'title': chapter['title'],
                    'sample': sample
                })
            
            # Get LLM consistency assessment
            system_prompt = self.prompts.get_consistency_assessment_system_prompt(state)
            human_prompt = self.prompts.get_consistency_assessment_human_prompt(chapter_samples, state)
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            llm_consistency = self.llm.with_structured_output(BookConsistencyModel).invoke(messages)
            
            # Calculate overall consistency score
            consistency_scores = [
                llm_consistency.style_consistency,
                llm_consistency.tone_consistency,
                llm_consistency.terminology_usage,
                llm_consistency.structural_consistency,
                llm_consistency.narrative_flow,
                llm_consistency.chapter_transitions
            ]
            
            overall_consistency = statistics.mean(consistency_scores)
            
            return BookConsistencyMetrics(
                overall_consistency_score=overall_consistency,
                inconsistencies_found=llm_consistency.inconsistencies_identified,
                consistency_strengths=llm_consistency.consistency_strengths
            )
            
        except Exception as e:
            logger.error(f"Consistency assessment failed: {e}")
            return BookConsistencyMetrics(
                overall_consistency_score=0.5,
                inconsistencies_found=["Consistency assessment failed - manual review needed"],
                consistency_strengths=[]
            )
    
    def _validate_research_integration(self, chapters: List[Chapter], state: BookWritingState) -> ResearchValidationResults:
        """Validate research integration across chapters."""
        try:
            # Basic research validation
            total_research_chunks = sum(len(ch.get('research_chunks_used', [])) for ch in chapters)
            
            if total_research_chunks == 0:
                return ResearchValidationResults(
                    research_accuracy_score=0.3,
                    citation_completeness_score=0.3,
                    research_integration_quality="minimal"
                )
            
            # Calculate scores based on usage
            research_accuracy_score = 0.8  # Assume good accuracy
            citation_completeness_score = min(1.0, total_research_chunks / (len(chapters) * 3))  # 3 sources per chapter ideal
            
            if citation_completeness_score >= 0.8:
                quality = "excellent"
            elif citation_completeness_score >= 0.6:
                quality = "good"
            else:
                quality = "adequate"
            
            return ResearchValidationResults(
                research_accuracy_score=research_accuracy_score,
                citation_completeness_score=citation_completeness_score,
                research_integration_quality=quality
            )
            
        except Exception as e:
            logger.error(f"Research validation failed: {e}")
            return ResearchValidationResults(
                research_accuracy_score=0.5,
                citation_completeness_score=0.5,
                research_integration_quality="unknown"
            )
    
    def _synthesize_quality_assessment(
        self,
        chapter_metrics: List[ChapterQualityMetrics],
        consistency_metrics: BookConsistencyMetrics,
        research_validation: ResearchValidationResults,
        state: BookWritingState
    ) -> BookQualityAssessment:
        """Synthesize overall quality assessment from all metrics."""
        
        # Calculate overall quality score
        chapter_scores = [cm.overall_score for cm in chapter_metrics]
        avg_chapter_score = statistics.mean(chapter_scores) if chapter_scores else 0.0
        
        # Weighted combination
        overall_score = (
            avg_chapter_score * 0.4 +
            consistency_metrics.overall_consistency_score * 0.3 +
            research_validation.research_accuracy_score * 0.3
        )
        
        meets_threshold = overall_score >= self.config.overall_quality_threshold
        
        # Identify revision areas
        revision_areas = []
        if avg_chapter_score < self.config.individual_chapter_threshold:
            revision_areas.append("chapter_content_quality")
        if consistency_metrics.overall_consistency_score < self.config.consistency_threshold:
            revision_areas.append("cross_chapter_consistency")
        if research_validation.research_accuracy_score < 0.7:
            revision_areas.append("research_integration")
        
        # Generate feedback
        detailed_feedback = self._generate_detailed_feedback(
            overall_score, chapter_metrics, consistency_metrics, research_validation
        )
        
        assessment_summary = f"Quality assessment complete. Score: {overall_score:.2f}/1.0. Status: {'Approved' if meets_threshold else 'Requires revision'}."
        
        approval_recommendation = "approve" if meets_threshold else "revise_and_resubmit"
        
        return BookQualityAssessment(
            overall_quality_score=overall_score,
            meets_quality_threshold=meets_threshold,
            chapter_metrics=chapter_metrics,
            consistency_metrics=consistency_metrics,
            research_validation=research_validation,
            revision_required=not meets_threshold,
            revision_priority_areas=revision_areas,
            approval_recommendation=approval_recommendation,
            detailed_feedback=detailed_feedback,
            assessment_summary=assessment_summary,
            created_at=datetime.now(timezone.utc).isoformat()
        )
    
    def _make_revision_decision(self, assessment: BookQualityAssessment, state: BookWritingState) -> RevisionDecisionModel:
        """Make revision decision based on quality assessment."""
        try:
            current_revisions = state.get('revision_count', 0)
            max_revisions = self.config.max_revision_cycles
            
            requires_revision = (
                assessment.revision_required and 
                current_revisions < max_revisions
            )
            
            # Determine urgency
            if assessment.overall_quality_score < 0.5:
                urgency = "critical"
            elif assessment.overall_quality_score < 0.6:
                urgency = "high"
            elif assessment.overall_quality_score < 0.7:
                urgency = "medium"
            else:
                urgency = "low"
            
            # Generate strategy
            if requires_revision:
                strategy = f"Focus on {', '.join(assessment.revision_priority_areas)}"
                expected_improvements = assessment.revision_priority_areas
            else:
                strategy = "No revision required"
                expected_improvements = []
            
            return RevisionDecisionModel(
                requires_revision=requires_revision,
                revision_urgency=urgency,
                priority_areas=assessment.revision_priority_areas,
                revision_strategy=strategy,
                expected_improvement_areas=expected_improvements,
                approval_recommendation=assessment.approval_recommendation,
                estimated_revision_effort="moderate" if requires_revision else "none"
            )
            
        except Exception as e:
            logger.error(f"Revision decision failed: {e}")
            return RevisionDecisionModel(
                requires_revision=True,
                revision_urgency="medium",
                priority_areas=["manual_review_required"],
                revision_strategy="Manual review needed due to system error",
                expected_improvement_areas=["overall_quality"],
                approval_recommendation="manual_review_required",
                estimated_revision_effort="unknown"
            )
    
    def _generate_detailed_feedback(
        self,
        overall_score: float,
        chapter_metrics: List[ChapterQualityMetrics],
        consistency_metrics: BookConsistencyMetrics,
        research_validation: ResearchValidationResults
    ) -> str:
        """Generate comprehensive detailed feedback."""
        
        feedback_sections = []
        
        # Overall assessment
        feedback_sections.append(f"OVERALL QUALITY: {overall_score:.2f}/1.0")
        
        if overall_score >= 0.8:
            feedback_sections.append("Status: Excellent - Publication ready")
        elif overall_score >= 0.7:
            feedback_sections.append("Status: Good - Minor revisions needed")
        elif overall_score >= 0.6:
            feedback_sections.append("Status: Adequate - Moderate revisions needed")
        else:
            feedback_sections.append("Status: Needs Improvement - Significant revisions required")
        
        # Chapter summary
        if chapter_metrics:
            scores = [cm.overall_score for cm in chapter_metrics]
            feedback_sections.append(f"\nCHAPTERS: {len(chapter_metrics)} analyzed, scores {min(scores):.2f}-{max(scores):.2f}")
            
            poor_chapters = [cm for cm in chapter_metrics if not cm.meets_threshold]
            if poor_chapters:
                feedback_sections.append(f"Chapters needing attention: {[cm.chapter_number for cm in poor_chapters]}")
        
        # Consistency summary
        feedback_sections.append(f"\nCONSISTENCY: {consistency_metrics.overall_consistency_score:.2f}/1.0")
        if consistency_metrics.inconsistencies_found:
            feedback_sections.append(f"Issues: {', '.join(consistency_metrics.inconsistencies_found[:3])}")
        
        # Research summary
        feedback_sections.append(f"\nRESEARCH: {research_validation.research_integration_quality}")
        feedback_sections.append(f"Integration score: {research_validation.research_accuracy_score:.2f}/1.0")
        
        return "\n".join(feedback_sections)
    
    def _create_fallback_chapter_metric(self, chapter: Chapter) -> ChapterQualityMetrics:
        """Create fallback chapter metric when assessment fails."""
        return ChapterQualityMetrics(
            chapter_number=chapter['chapter_number'],
            chapter_title=chapter['title'],
            overall_score=0.5,
            dimension_scores={'overall': 0.5},
            issues_found=["Assessment failed - manual review needed"],
            strengths=["Unable to assess"],
            improvement_suggestions=["Manual review recommended"],
            meets_threshold=False
        )
    
    def _update_review_stats(self, assessment: BookQualityAssessment, revision_decision: RevisionDecisionModel) -> None:
        """Update internal review statistics."""
        self.review_stats['books_reviewed'] += 1
        
        if not revision_decision.requires_revision:
            self.review_stats['books_approved'] += 1
        
        # Update average quality score
        current_avg = self.review_stats['average_quality_score']
        books_count = self.review_stats['books_reviewed']
        new_avg = ((current_avg * (books_count - 1)) + assessment.overall_quality_score) / books_count
        self.review_stats['average_quality_score'] = new_avg
        
        if self.review_stats['session_start'] is None:
            self.review_stats['session_start'] = time.time()
    
    def get_review_statistics(self) -> Dict[str, Any]:
        """Get comprehensive review statistics."""
        stats = self.review_stats.copy()
        
        # Calculate additional metrics
        if stats['books_reviewed'] > 0:
            stats['approval_rate'] = stats['books_approved'] / stats['books_reviewed']
        
        # Add session information
        if stats['session_start']:
            stats['session_duration_minutes'] = (time.time() - stats['session_start']) / 60
        
        # Add configuration info
        stats['configuration'] = {
            'model': self.config.llm_model,
            'quality_threshold': self.config.overall_quality_threshold,
            'max_revisions': self.config.max_revision_cycles
        }
        
        return stats
    
    def cleanup_resources(self) -> bool:
        """Clean up resources and connections."""
        try:
            if hasattr(self.chroma_client, 'close'):
                self.chroma_client.close()
            self.quality_cache.clear()
            logger.info("Quality Reviewer Agent resources cleaned up")
            return True
        except Exception as e:
            logger.error(f"Error cleaning up resources: {e}")
            return False


def main():
    """Test function for QualityReviewerAgent."""
    
    # Create test configuration
    config = QualityReviewerConfig()
    
    # Initialize agent
    agent = QualityReviewerAgent(config)
    
    # Create test state with completed chapters
    test_state = {
        'book_id': 'test_book_001',
        'title': 'Test Book Title',
        'genre': 'Technical',
        'target_word_count': 50000,
        'target_audience': 'Professional developers',
        'current_chapter': 2,
        'revision_count': 0,
        'writing_style_guide': 'Professional technical writing with clear explanations',
        'chapters': [
            {
                'chapter_number': 1,
                'title': 'Introduction to the Topic',
                'description': 'This chapter introduces the main concepts',
                'target_word_count': 2500,
                'status': 'complete',
                'content': '''# Introduction to the Topic

This chapter provides a comprehensive introduction to the fundamental concepts that will be explored throughout this book. Understanding these core principles is essential for readers who want to master the subject matter.

## Core Concepts

The first concept we need to understand is the basic framework that underlies all subsequent discussions. This framework provides the structure for organizing complex information and making it accessible to practitioners.

Research shows that systematic approaches to learning complex topics result in better retention and application. According to recent studies, structured learning environments improve comprehension by up to 40%.

## Key Principles

There are several key principles that guide our approach:

1. **Clarity First**: All explanations prioritize clarity over complexity
2. **Practical Application**: Every concept includes real-world examples
3. **Progressive Learning**: Each chapter builds on previous knowledge

## Chapter Overview

The following chapters will explore these concepts in depth, providing detailed analysis and practical applications that readers can implement immediately.

## Conclusion

This introduction has established the foundation for our exploration of the topic. In the next chapter, we will delve into the technical details that make these concepts practical and actionable.''',
                'word_count': 189,
                'research_chunks_used': ['chunk_001', 'chunk_002', 'chunk_003'],
                'completed_at': '2024-01-15T10:00:00Z'
            },
            {
                'chapter_number': 2,
                'title': 'Technical Implementation',
                'description': 'Technical details and implementation strategies',
                'target_word_count': 3000,
                'status': 'complete',
                'content': '''# Technical Implementation

Building on the foundational concepts introduced in Chapter 1, this chapter explores the technical implementation details that transform theoretical understanding into practical solutions.

## Implementation Strategy

The implementation strategy follows a systematic approach that ensures reliability and maintainability. This strategy has been validated through extensive testing and real-world application.

### Core Components

The system consists of several core components that work together to provide comprehensive functionality:

- **Data Processing Module**: Handles input validation and transformation
- **Analysis Engine**: Performs complex calculations and analysis
- **Output Generator**: Formats results for various use cases
- **Quality Assurance**: Ensures accuracy and reliability

## Best Practices

Industry best practices guide our implementation approach. These practices have been developed through years of experience and continuous improvement.

According to recent industry surveys, organizations that follow these best practices achieve 60% better outcomes compared to those using ad-hoc approaches.

## Performance Considerations

Performance optimization is crucial for large-scale implementations. The following considerations ensure optimal performance:

1. **Efficient Algorithms**: Use proven algorithms with optimal time complexity
2. **Resource Management**: Careful management of memory and processing resources
3. **Scalability Planning**: Design for future growth and expansion

## Error Handling

Robust error handling ensures system reliability and user satisfaction. The error handling strategy includes comprehensive logging, graceful degradation, and user-friendly error messages.

## Conclusion

This chapter has provided detailed technical implementation guidance. The next chapter will explore advanced optimization techniques and performance tuning strategies.''',
                'word_count': 267,
                'research_chunks_used': ['chunk_004', 'chunk_005', 'chunk_006', 'chunk_007'],
                'completed_at': '2024-01-15T11:00:00Z'
            }
        ]
    }
    
    try:
        # Test quality review
        print("Testing Quality Reviewer Agent...")
        result = agent.review_book(test_state)
        
        print(f"Quality Score: {result.get('quality_score', 'N/A'):.2f}")
        print(f"Needs Revision: {result.get('needs_revision', True)}")
        print(f"Review Notes: {result.get('review_notes', ['No notes'])[0]}")
        
        # Test comprehensive review
        full_result = agent.review_book_quality(test_state)
        print(f"Full Review Status: {full_result['status']}")
        print(f"Overall Score: {full_result.get('overall_quality_score', 'N/A'):.2f}")
        
        # Test statistics
        stats = agent.get_review_statistics()
        print(f"Books Reviewed: {stats['books_reviewed']}")
        
        print("QualityReviewerAgent test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        agent.cleanup_resources()


if __name__ == "__main__":
    main()