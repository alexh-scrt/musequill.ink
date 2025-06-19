"""
Configuration management for the Writing Planner Agent
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class WritingPlannerConfig(BaseSettings):
    """Configuration settings for the writing planner agent."""
    
    # LLM settings for writing strategy and planning
    openai_api_key: str = Field(
        default="",
        validation_alias="OPENAI_API_KEY",
        description="OpenAI API key for LLM operations"
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        validation_alias="WRITING_PLANNER_LLM_MODEL",
        description="LLM model to use for writing planning"
    )
    llm_temperature: float = Field(
        default=0.4,
        validation_alias="WRITING_PLANNER_LLM_TEMPERATURE",
        description="Temperature for LLM responses (balanced creativity/consistency)",
        ge=0.0,
        le=2.0
    )
    llm_max_tokens: int = Field(
        default=4000,
        validation_alias="WRITING_PLANNER_MAX_TOKENS",
        description="Maximum tokens for LLM responses",
        ge=1000,
        le=16000
    )
    
    # Chroma Vector Store settings (for research analysis)
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
    
    # OpenAI Embeddings for research analysis
    embedding_model: str = Field(
        default="text-embedding-3-small",
        validation_alias="EMBEDDING_MODEL",
        description="OpenAI embedding model for research analysis"
    )
    
    # Writing strategy settings
    enable_advanced_planning: bool = Field(
        default=True,
        validation_alias="ENABLE_ADVANCED_PLANNING",
        description="Enable advanced LLM-based writing planning"
    )
    planning_depth: str = Field(
        default="comprehensive",
        validation_alias="PLANNING_DEPTH",
        description="Planning depth level (basic/detailed/comprehensive)"
    )
    include_research_analysis: bool = Field(
        default=True,
        validation_alias="INCLUDE_RESEARCH_ANALYSIS",
        description="Include detailed research analysis in planning"
    )
    
    # Chapter planning settings
    auto_adjust_chapter_lengths: bool = Field(
        default=True,
        validation_alias="AUTO_ADJUST_CHAPTER_LENGTHS",
        description="Automatically adjust chapter target word counts"
    )
    min_chapter_word_count: int = Field(
        default=2000,
        validation_alias="MIN_CHAPTER_WORD_COUNT",
        description="Minimum word count per chapter",
        ge=500,
        le=10000
    )
    max_chapter_word_count: int = Field(
        default=8000,
        validation_alias="MAX_CHAPTER_WORD_COUNT",
        description="Maximum word count per chapter",
        ge=2000,
        le=20000
    )
    target_chapter_balance: float = Field(
        default=0.2,
        validation_alias="TARGET_CHAPTER_BALANCE",
        description="Acceptable variance in chapter lengths (0.2 = 20%)",
        ge=0.1,
        le=0.5
    )
    
    # Research integration settings
    research_chunks_per_chapter: int = Field(
        default=5,
        validation_alias="RESEARCH_CHUNKS_PER_CHAPTER",
        description="Target research chunks to assign per chapter",
        ge=2,
        le=20
    )
    research_similarity_threshold: float = Field(
        default=0.7,
        validation_alias="RESEARCH_SIMILARITY_THRESHOLD",
        description="Similarity threshold for matching research to chapters",
        ge=0.5,
        le=0.95
    )
    enable_research_prioritization: bool = Field(
        default=True,
        validation_alias="ENABLE_RESEARCH_PRIORITIZATION",
        description="Prioritize higher-quality research chunks"
    )
    
    # Writing style and structure settings
    generate_style_guide: bool = Field(
        default=True,
        validation_alias="GENERATE_STYLE_GUIDE",
        description="Generate detailed writing style guide"
    )
    style_consistency_level: str = Field(
        default="high",
        validation_alias="STYLE_CONSISTENCY_LEVEL",
        description="Style consistency enforcement level (low/medium/high)"
    )
    include_tone_guidelines: bool = Field(
        default=True,
        validation_alias="INCLUDE_TONE_GUIDELINES",
        description="Include tone and voice guidelines"
    )
    include_structure_templates: bool = Field(
        default=True,
        validation_alias="INCLUDE_STRUCTURE_TEMPLATES",
        description="Include chapter structure templates"
    )
    
    # Content organization settings
    enable_narrative_flow_analysis: bool = Field(
        default=True,
        validation_alias="ENABLE_NARRATIVE_FLOW_ANALYSIS",
        description="Analyze and optimize narrative flow between chapters"
    )
    create_transition_guidelines: bool = Field(
        default=True,
        validation_alias="CREATE_TRANSITION_GUIDELINES",
        description="Create guidelines for chapter transitions"
    )
    generate_chapter_dependencies: bool = Field(
        default=True,
        validation_alias="GENERATE_CHAPTER_DEPENDENCIES",
        description="Identify dependencies between chapters"
    )
    
    # Quality and consistency settings
    target_quality_level: str = Field(
        default="professional",
        validation_alias="TARGET_QUALITY_LEVEL",
        description="Target writing quality level (academic/professional/popular)"
    )
    consistency_check_points: list = Field(
        default=[
            "terminology",
            "tone",
            "formatting",
            "citation_style",
            "argument_structure"
        ],
        description="Aspects to maintain consistency across"
    )
    
    # Research analysis settings
    max_research_sample_size: int = Field(
        default=50,
        validation_alias="MAX_RESEARCH_SAMPLE_SIZE",
        description="Maximum research chunks to analyze for planning",
        ge=10,
        le=200
    )
    research_analysis_method: str = Field(
        default="comprehensive",
        validation_alias="RESEARCH_ANALYSIS_METHOD",
        description="Method for analyzing research (basic/comprehensive/deep)"
    )
    enable_research_clustering: bool = Field(
        default=True,
        validation_alias="ENABLE_RESEARCH_CLUSTERING",
        description="Cluster related research for better organization"
    )
    
    # Outline enhancement settings
    enhance_existing_outline: bool = Field(
        default=True,
        validation_alias="ENHANCE_EXISTING_OUTLINE",
        description="Enhance existing book outline with research insights"
    )
    outline_detail_level: str = Field(
        default="detailed",
        validation_alias="OUTLINE_DETAIL_LEVEL",
        description="Level of detail for outline enhancement (basic/detailed/extensive)"
    )
    include_key_points: bool = Field(
        default=True,
        validation_alias="INCLUDE_KEY_POINTS",
        description="Include key points and arguments for each chapter"
    )
    include_supporting_evidence: bool = Field(
        default=True,
        validation_alias="INCLUDE_SUPPORTING_EVIDENCE",
        description="Include references to supporting research evidence"
    )
    
    # Performance and optimization settings
    max_planning_time: int = Field(
        default=300,
        validation_alias="MAX_PLANNING_TIME",
        description="Maximum time for planning process in seconds",
        ge=60,
        le=1800
    )
    enable_parallel_processing: bool = Field(
        default=True,
        validation_alias="ENABLE_PARALLEL_PROCESSING",
        description="Enable parallel processing for planning tasks"
    )
    cache_research_analysis: bool = Field(
        default=True,
        validation_alias="CACHE_RESEARCH_ANALYSIS",
        description="Cache research analysis results"
    )
    
    # Output formatting settings
    include_writing_timeline: bool = Field(
        default=True,
        validation_alias="INCLUDE_WRITING_TIMELINE",
        description="Include estimated writing timeline"
    )
    generate_progress_milestones: bool = Field(
        default=True,
        validation_alias="GENERATE_PROGRESS_MILESTONES",
        description="Generate writing progress milestones"
    )
    include_resource_recommendations: bool = Field(
        default=True,
        validation_alias="INCLUDE_RESOURCE_RECOMMENDATIONS",
        description="Include additional resource recommendations"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )