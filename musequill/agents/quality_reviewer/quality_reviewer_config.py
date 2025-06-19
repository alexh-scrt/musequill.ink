"""
Configuration management for the Quality Reviewer Agent
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class QualityReviewerConfig(BaseSettings):
    """Configuration settings for the quality reviewer agent."""
    
    # LLM settings for quality assessment
    openai_api_key: str = Field(
        default="",
        validation_alias="OPENAI_API_KEY",
        description="OpenAI API key for LLM operations"
    )
    llm_model: str = Field(
        default="gpt-4o",
        validation_alias="QUALITY_REVIEWER_LLM_MODEL",
        description="LLM model to use for quality review"
    )
    llm_temperature: float = Field(
        default=0.2,
        validation_alias="QUALITY_REVIEWER_LLM_TEMPERATURE",
        description="Temperature for LLM responses (lower for analytical consistency)",
        ge=0.0,
        le=2.0
    )
    llm_max_tokens: int = Field(
        default=6000,
        validation_alias="QUALITY_REVIEWER_MAX_TOKENS",
        description="Maximum tokens for LLM responses",
        ge=2000,
        le=16000
    )
    
    # Chroma Vector Store settings (for research validation)
    chroma_host: str = Field(
        default="localhost",
        validation_alias="CHROMA_HOST",
        description="Chroma database host"
    )
    chroma_port: int = Field(
        default=8000,
        validation_alias="CHROMA_PORT",
        description="Chroma database port"
    )
    chroma_collection_name: str = Field(
        default="book_research",
        validation_alias="CHROMA_COLLECTION_NAME",
        description="Chroma collection name for research materials"
    )
    chroma_tenant: str = Field(
        default="default_tenant",
        validation_alias="CHROMA_TENANT",
        description="Chroma tenant name"
    )
    chroma_database: str = Field(
        default="default_database",
        validation_alias="CHROMA_DATABASE",
        description="Chroma database name"
    )
    
    # OpenAI Embeddings for content analysis
    embedding_model: str = Field(
        default="text-embedding-3-small",
        validation_alias="EMBEDDING_MODEL",
        description="OpenAI embedding model for content analysis"
    )
    embedding_dimensions: int = Field(
        default=1536,
        validation_alias="EMBEDDING_DIMENSIONS",
        description="Embedding vector dimensions",
        ge=256,
        le=3072
    )
    
    # Quality assessment criteria
    overall_quality_threshold: float = Field(
        default=0.8,
        validation_alias="OVERALL_QUALITY_THRESHOLD",
        description="Minimum overall quality score to pass review",
        ge=0.0,
        le=1.0
    )
    individual_chapter_threshold: float = Field(
        default=0.7,
        validation_alias="INDIVIDUAL_CHAPTER_THRESHOLD",
        description="Minimum quality score for individual chapters",
        ge=0.0,
        le=1.0
    )
    consistency_threshold: float = Field(
        default=0.75,
        validation_alias="CONSISTENCY_THRESHOLD",
        description="Minimum consistency score across chapters",
        ge=0.0,
        le=1.0
    )
    
    # Review scope and depth
    enable_comprehensive_review: bool = Field(
        default=True,
        validation_alias="ENABLE_COMPREHENSIVE_REVIEW",
        description="Enable comprehensive multi-dimensional quality review"
    )
    review_dimensions: list = Field(
        default=[
            "content_quality",
            "style_consistency",
            "logical_flow",
            "research_integration",
            "readability",
            "engagement",
            "technical_accuracy",
            "completeness"
        ],
        description="Dimensions to evaluate during quality review"
    )
    enable_chapter_comparison: bool = Field(
        default=True,
        validation_alias="ENABLE_CHAPTER_COMPARISON",
        description="Enable cross-chapter consistency analysis"
    )
    enable_research_validation: bool = Field(
        default=True,
        validation_alias="ENABLE_RESEARCH_VALIDATION",
        description="Validate research integration and accuracy"
    )
    
    # Content analysis settings
    max_chapters_to_sample: int = Field(
        default=10,
        validation_alias="MAX_CHAPTERS_TO_SAMPLE",
        description="Maximum chapters to analyze in detail (for large books)",
        ge=3,
        le=50
    )
    content_sample_size: int = Field(
        default=2000,
        validation_alias="CONTENT_SAMPLE_SIZE",
        description="Number of words to sample from each chapter for analysis",
        ge=500,
        le=5000
    )
    enable_full_book_analysis: bool = Field(
        default=True,
        validation_alias="ENABLE_FULL_BOOK_ANALYSIS",
        description="Analyze complete book structure and flow"
    )
    
    # Style and consistency analysis
    analyze_writing_style: bool = Field(
        default=True,
        validation_alias="ANALYZE_WRITING_STYLE",
        description="Analyze writing style consistency"
    )
    check_terminology_consistency: bool = Field(
        default=True,
        validation_alias="CHECK_TERMINOLOGY_CONSISTENCY",
        description="Check consistency of terminology usage"
    )
    validate_citation_style: bool = Field(
        default=True,
        validation_alias="VALIDATE_CITATION_STYLE",
        description="Validate citation style consistency"
    )
    assess_tone_consistency: bool = Field(
        default=True,
        validation_alias="ASSESS_TONE_CONSISTENCY",
        description="Assess tone and voice consistency"
    )
    
    # Readability and engagement metrics
    calculate_readability_scores: bool = Field(
        default=True,
        validation_alias="CALCULATE_READABILITY_SCORES",
        description="Calculate readability metrics (Flesch, etc.)"
    )
    assess_engagement_level: bool = Field(
        default=True,
        validation_alias="ASSESS_ENGAGEMENT_LEVEL",
        description="Assess content engagement and interest level"
    )
    target_reading_level: str = Field(
        default="college",
        validation_alias="TARGET_READING_LEVEL",
        description="Target reading level (elementary/high_school/college/graduate)"
    )
    
    # Research integration validation
    verify_source_accuracy: bool = Field(
        default=True,
        validation_alias="VERIFY_SOURCE_ACCURACY",
        description="Verify accuracy of research integration"
    )
    check_citation_completeness: bool = Field(
        default=True,
        validation_alias="CHECK_CITATION_COMPLETENESS",
        description="Check completeness of citations and references"
    )
    validate_research_relevance: bool = Field(
        default=True,
        validation_alias="VALIDATE_RESEARCH_RELEVANCE",
        description="Validate relevance of integrated research"
    )
    
    # Revision and improvement recommendations
    generate_improvement_suggestions: bool = Field(
        default=True,
        validation_alias="GENERATE_IMPROVEMENT_SUGGESTIONS",
        description="Generate specific improvement suggestions"
    )
    prioritize_revision_areas: bool = Field(
        default=True,
        validation_alias="PRIORITIZE_REVISION_AREAS",
        description="Prioritize areas needing revision"
    )
    include_positive_feedback: bool = Field(
        default=True,
        validation_alias="INCLUDE_POSITIVE_FEEDBACK",
        description="Include positive feedback on strong areas"
    )
    suggest_chapter_reordering: bool = Field(
        default=False,
        validation_alias="SUGGEST_CHAPTER_REORDERING",
        description="Suggest reordering chapters if flow issues detected"
    )
    
    # Revision decision criteria
    max_revision_cycles: int = Field(
        default=3,
        validation_alias="MAX_REVISION_CYCLES",
        description="Maximum number of revision cycles allowed",
        ge=1,
        le=10
    )
    require_revision_if_below_threshold: bool = Field(
        default=True,
        validation_alias="REQUIRE_REVISION_IF_BELOW_THRESHOLD",
        description="Automatically require revision if below quality threshold"
    )
    allow_partial_approval: bool = Field(
        default=False,
        validation_alias="ALLOW_PARTIAL_APPROVAL",
        description="Allow approval of book with some chapters below threshold"
    )
    escalate_persistent_issues: bool = Field(
        default=True,
        validation_alias="ESCALATE_PERSISTENT_ISSUES",
        description="Escalate if issues persist after maximum revisions"
    )
    
    # Performance and optimization settings
    max_review_time_per_book: int = Field(
        default=600,
        validation_alias="MAX_REVIEW_TIME_PER_BOOK",
        description="Maximum time for reviewing entire book in seconds",
        ge=300,
        le=3600
    )
    enable_parallel_chapter_analysis: bool = Field(
        default=True,
        validation_alias="ENABLE_PARALLEL_CHAPTER_ANALYSIS",
        description="Enable parallel analysis of multiple chapters"
    )
    cache_quality_assessments: bool = Field(
        default=True,
        validation_alias="CACHE_QUALITY_ASSESSMENTS",
        description="Cache quality assessments for efficiency"
    )
    
    # Detailed assessment settings
    word_count_variance_tolerance: float = Field(
        default=0.15,
        validation_alias="WORD_COUNT_VARIANCE_TOLERANCE",
        description="Acceptable word count variance from targets",
        ge=0.0,
        le=0.5
    )
    structure_score_weight: float = Field(
        default=0.2,
        validation_alias="STRUCTURE_SCORE_WEIGHT",
        description="Weight for structure in overall quality score",
        ge=0.0,
        le=1.0
    )
    content_score_weight: float = Field(
        default=0.3,
        validation_alias="CONTENT_SCORE_WEIGHT",
        description="Weight for content quality in overall score",
        ge=0.0,
        le=1.0
    )
    consistency_score_weight: float = Field(
        default=0.25,
        validation_alias="CONSISTENCY_SCORE_WEIGHT",
        description="Weight for consistency in overall score",
        ge=0.0,
        le=1.0
    )
    research_integration_weight: float = Field(
        default=0.25,
        validation_alias="RESEARCH_INTEGRATION_WEIGHT",
        description="Weight for research integration in overall score",
        ge=0.0,
        le=1.0
    )
    
    # Output and reporting settings
    generate_detailed_report: bool = Field(
        default=True,
        validation_alias="GENERATE_DETAILED_REPORT",
        description="Generate detailed quality assessment report"
    )
    include_chapter_by_chapter_analysis: bool = Field(
        default=True,
        validation_alias="INCLUDE_CHAPTER_BY_CHAPTER_ANALYSIS",
        description="Include individual chapter analysis in report"
    )
    include_statistical_summary: bool = Field(
        default=True,
        validation_alias="INCLUDE_STATISTICAL_SUMMARY",
        description="Include statistical summary of quality metrics"
    )
    export_quality_metrics: bool = Field(
        default=True,
        validation_alias="EXPORT_QUALITY_METRICS",
        description="Export quality metrics for external analysis"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )