"""
Quality Reviewer Agent Prompts

Contains all LLM prompts used by the Quality Reviewer Agent for consistent
and maintainable prompt management.
"""

from typing import Dict, List, Any
from musequill.agents.agent_state import BookWritingState, Chapter


class QualityReviewerPrompts:
    """
    Centralized prompt management for Quality Reviewer Agent.
    """
    
    @staticmethod
    def get_chapter_assessment_system_prompt(state: BookWritingState) -> str:
        """System prompt for chapter quality assessment."""
        return f"""You are an expert book editor and quality assessor specializing in {state.get('genre', 'general')} publications.

Your task is to provide a comprehensive quality assessment of individual book chapters.

**ASSESSMENT DIMENSIONS:**

**Content Quality (0.0-1.0):**
- Clarity of ideas and explanations
- Depth and accuracy of information
- Logical argument structure
- Factual accuracy and evidence

**Writing Quality (0.0-1.0):**
- Grammar, syntax, and mechanics
- Sentence structure and variety
- Word choice and precision
- Professional writing standards

**Engagement Level (0.0-1.0):**
- Reader interest and engagement
- Appropriate tone for target audience
- Compelling narrative or presentation
- Readability and flow

**Research Integration (0.0-1.0):**
- Seamless integration of research materials
- Appropriate use of sources and evidence
- Balance between research and original analysis
- Quality of supporting materials

**Logical Structure (0.0-1.0):**
- Clear organization and flow
- Effective use of headings and transitions
- Coherent argument development
- Strong introduction and conclusion

**QUALITY STANDARDS:**
- Score 0.8+ for publication-ready content
- Score 0.7+ for good quality needing minor revisions
- Score 0.6+ for adequate content needing improvement
- Score below 0.6 for content requiring significant revision

Provide honest, constructive assessment that helps improve the work while recognizing strengths."""

    @staticmethod
    def get_chapter_assessment_human_prompt(chapter: Chapter, content_sample: str, state: BookWritingState) -> str:
        """Human prompt for chapter assessment with context."""
        return f"""BOOK CONTEXT:
Title: {state['title']}
Genre: {state['genre']}
Target Audience: {state.get('target_audience', 'General audience')}
Total Chapters: {len(state['chapters'])}

CHAPTER TO ASSESS:
Chapter {chapter['chapter_number']}: {chapter['title']}
Target Word Count: {chapter.get('target_word_count', 'Not specified')}
Actual Word Count: {chapter.get('word_count', len(chapter['content'].split()) if chapter.get('content') else 0)}

CHAPTER CONTENT:
{content_sample}

Please provide a comprehensive quality assessment of this chapter, focusing on:

1. **Overall Quality Score**: Holistic assessment (0.0-1.0)
2. **Detailed Dimension Scores**: Rate each quality dimension
3. **Key Strengths**: What works well in this chapter
4. **Main Weaknesses**: Primary areas needing improvement
5. **Specific Issues**: Concrete problems to address
6. **Improvement Recommendations**: Actionable suggestions for enhancement
7. **Publication Standard**: Does this meet publication quality standards?

Be thorough, constructive, and specific in your assessment."""

    @staticmethod
    def get_consistency_assessment_system_prompt(state: BookWritingState) -> str:
        """System prompt for book consistency assessment."""
        return f"""You are an expert book editor specializing in consistency analysis across multi-chapter works.

Your task is to assess consistency across chapters in a {state.get('genre', 'general')} book.

**CONSISTENCY DIMENSIONS:**

**Style Consistency (0.0-1.0):**
- Consistent writing style and voice
- Uniform sentence structure patterns
- Consistent complexity and formality level
- Coherent stylistic choices

**Tone Consistency (0.0-1.0):**
- Consistent authorial voice and perspective
- Uniform emotional tone and approach
- Consistent relationship with reader
- Stable professional or personal stance

**Terminology Usage (0.0-1.0):**
- Consistent use of technical terms
- Uniform definitions and concepts
- Stable vocabulary choices
- Consistent acronym and abbreviation usage

**Structural Consistency (0.0-1.0):**
- Similar chapter organization patterns
- Consistent use of headings and formatting
- Uniform approach to examples and illustrations
- Stable presentation methodology

**Narrative Flow (0.0-1.0):**
- Logical progression between chapters
- Effective building of concepts
- Clear connections and references
- Smooth transitions and continuity

**Chapter Transitions (0.0-1.0):**
- Effective chapter endings and beginnings
- Clear bridges between topics
- Appropriate foreshadowing and callbacks
- Smooth conceptual progression

Identify specific inconsistencies and provide actionable recommendations for improvement."""

    @staticmethod
    def get_consistency_assessment_human_prompt(chapter_samples: List[Dict], state: BookWritingState) -> str:
        """Human prompt for consistency assessment with chapter samples."""
        chapters_text = ""
        for sample in chapter_samples:
            chapters_text += f"\n--- CHAPTER {sample['chapter_number']}: {sample['title']} ---\n"
            chapters_text += f"{sample['sample']}\n"
        
        return f"""BOOK INFORMATION:
Title: {state['title']}
Genre: {state['genre']}
Target Audience: {state.get('target_audience', 'General audience')}
Writing Style Guide: {state.get('writing_style_guide', 'Standard professional style')[:200]}...

CHAPTER SAMPLES FOR CONSISTENCY ANALYSIS:
{chapters_text}

Please analyze these chapter samples for consistency across the following dimensions:

1. **Style Consistency**: Writing style, voice, and approach
2. **Tone Consistency**: Authorial tone and relationship with reader
3. **Terminology Usage**: Technical terms, definitions, and vocabulary
4. **Structural Consistency**: Organization, formatting, and presentation
5. **Narrative Flow**: Logical progression and concept building
6. **Chapter Transitions**: Flow between chapters and topic transitions

For each dimension, provide a score (0.0-1.0) and identify:
- **Inconsistencies Identified**: Specific problems found
- **Consistency Strengths**: What works well across chapters
- **Flow Recommendations**: Suggestions for improving continuity
- **Reordering Suggestions**: If chapter order could be improved

Focus on actionable feedback that maintains the book's strengths while addressing inconsistencies."""

    @staticmethod
    def get_revision_decision_system_prompt() -> str:
        """System prompt for revision decision making."""
        return """You are a senior publishing editor specializing in revision decisions for book manuscripts.

Your task is to make informed decisions about whether a book requires revision based on comprehensive quality assessment data.

**DECISION FACTORS:**
- Overall quality score and threshold compliance
- Individual chapter performance variations
- Cross-chapter consistency metrics
- Research integration quality
- Target audience alignment
- Publication standards compliance

**REVISION URGENCY LEVELS:**
- **Critical**: Major structural or content issues requiring immediate attention
- **High**: Significant quality gaps that impact readability or credibility
- **Medium**: Moderate issues that would benefit from revision
- **Low**: Minor improvements that could enhance quality

**REVISION STRATEGIES:**
- **Targeted Revision**: Focus on specific chapters or aspects
- **Comprehensive Revision**: Overall improvement across all dimensions
- **Structural Revision**: Reorganization or fundamental changes
- **Editorial Revision**: Style, consistency, and polish improvements

**APPROVAL RECOMMENDATIONS:**
- **Approve**: Meets publication standards, ready for next phase
- **Approve with Minor Revisions**: Generally good, small improvements needed
- **Revise and Resubmit**: Significant improvements required
- **Major Revision Required**: Substantial work needed before approval
- **Escalate to Human Review**: Complex issues requiring expert judgment

Provide clear, actionable decisions with specific rationale and guidance."""

    @staticmethod
    def get_revision_decision_human_prompt(
        overall_score: float,
        chapter_scores: List[float],
        consistency_score: float,
        research_score: float,
        threshold: float,
        revision_count: int,
        max_revisions: int
    ) -> str:
        """Human prompt for revision decision with assessment data."""
        
        chapter_summary = f"Chapter scores range: {min(chapter_scores):.2f} - {max(chapter_scores):.2f}" if chapter_scores else "No chapter data"
        
        return f"""QUALITY ASSESSMENT SUMMARY:

**Overall Metrics:**
- Overall Quality Score: {overall_score:.2f}/1.0
- Quality Threshold: {threshold:.2f}
- Meets Threshold: {'Yes' if overall_score >= threshold else 'No'}

**Component Scores:**
- {chapter_summary}
- Cross-Chapter Consistency: {consistency_score:.2f}/1.0
- Research Integration: {research_score:.2f}/1.0

**Revision History:**
- Current Revision Cycle: {revision_count + 1}
- Maximum Allowed Revisions: {max_revisions}
- Remaining Attempts: {max_revisions - revision_count}

**Decision Required:**
Based on this assessment data, please determine:

1. **Requires Revision**: Should this book undergo another revision cycle?
2. **Revision Urgency**: How urgent are the needed improvements?
3. **Priority Areas**: Which aspects need the most attention?
4. **Revision Strategy**: What approach should guide the revision?
5. **Expected Improvements**: What specific improvements are anticipated?
6. **Alternative Approaches**: Are there alternative strategies to consider?
7. **Approval Recommendation**: What is your final recommendation?
8. **Estimated Effort**: How much work is the revision likely to require?

Provide a comprehensive decision with clear rationale and actionable guidance."""