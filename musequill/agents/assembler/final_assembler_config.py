"""
Configuration management for the Final Assembler Agent
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class FinalAssemblerConfig(BaseSettings):
    """Configuration settings for the final assembler agent."""
    
    # LLM settings for content assembly and formatting
    openai_api_key: str = Field(
        default="",
        validation_alias="OPENAI_API_KEY",
        description="OpenAI API key for LLM operations"
    )
    llm_model: str = Field(
        default="gpt-4o",
        validation_alias="FINAL_ASSEMBLER_LLM_MODEL",
        description="LLM model to use for assembly tasks"
    )
    llm_temperature: float = Field(
        default=0.1,
        validation_alias="FINAL_ASSEMBLER_LLM_TEMPERATURE",
        description="Temperature for LLM responses (lower for formatting consistency)",
        ge=0.0,
        le=2.0
    )
    llm_max_tokens: int = Field(
        default=4000,
        validation_alias="FINAL_ASSEMBLER_MAX_TOKENS",
        description="Maximum tokens for LLM responses",
        ge=2000,
        le=8000
    )
    
    # Book assembly settings
    enable_content_validation: bool = Field(
        default=True,
        validation_alias="ENABLE_CONTENT_VALIDATION",
        description="Validate content integrity during assembly"
    )
    generate_table_of_contents: bool = Field(
        default=True,
        validation_alias="GENERATE_TABLE_OF_CONTENTS",
        description="Generate table of contents for the book"
    )
    include_index: bool = Field(
        default=True,
        validation_alias="INCLUDE_INDEX",
        description="Generate index for the book"
    )
    generate_bibliography: bool = Field(
        default=True,
        validation_alias="GENERATE_BIBLIOGRAPHY",
        description="Generate bibliography from research sources"
    )
    add_chapter_numbering: bool = Field(
        default=True,
        validation_alias="ADD_CHAPTER_NUMBERING",
        description="Add consistent chapter numbering"
    )
    
    # Format generation settings
    output_formats: list = Field(
        default=["pdf", "epub", "docx", "html"],
        validation_alias="OUTPUT_FORMATS",
        description="List of output formats to generate"
    )
    pdf_generation_engine: str = Field(
        default="weasyprint",
        validation_alias="PDF_GENERATION_ENGINE",
        description="PDF generation engine (weasyprint/pdfkit/reportlab)"
    )
    epub_metadata_complete: bool = Field(
        default=True,
        validation_alias="EPUB_METADATA_COMPLETE",
        description="Include complete metadata in EPUB files"
    )
    html_include_styling: bool = Field(
        default=True,
        validation_alias="HTML_INCLUDE_STYLING",
        description="Include CSS styling in HTML output"
    )
    
    # Chapter organization settings
    validate_chapter_order: bool = Field(
        default=True,
        validation_alias="VALIDATE_CHAPTER_ORDER",
        description="Validate logical chapter ordering"
    )
    add_page_breaks: bool = Field(
        default=True,
        validation_alias="ADD_PAGE_BREAKS",
        description="Add page breaks between chapters"
    )
    include_chapter_summaries: bool = Field(
        default=False,
        validation_alias="INCLUDE_CHAPTER_SUMMARIES",
        description="Include chapter summaries in table of contents"
    )
    normalize_chapter_formatting: bool = Field(
        default=True,
        validation_alias="NORMALIZE_CHAPTER_FORMATTING",
        description="Normalize formatting across all chapters"
    )
    
    # Quality assurance settings
    check_content_completeness: bool = Field(
        default=True,
        validation_alias="CHECK_CONTENT_COMPLETENESS",
        description="Check that all chapters have content"
    )
    validate_internal_links: bool = Field(
        default=True,
        validation_alias="VALIDATE_INTERNAL_LINKS",
        description="Validate internal cross-references and links"
    )
    check_image_references: bool = Field(
        default=True,
        validation_alias="CHECK_IMAGE_REFERENCES",
        description="Check image and figure references"
    )
    verify_citation_format: bool = Field(
        default=True,
        validation_alias="VERIFY_CITATION_FORMAT",
        description="Verify citation formatting consistency"
    )
    
    # Bibliography and references settings
    bibliography_style: str = Field(
        default="apa",
        validation_alias="BIBLIOGRAPHY_STYLE",
        description="Citation style for bibliography (apa/mla/chicago/ieee)"
    )
    auto_generate_citations: bool = Field(
        default=True,
        validation_alias="AUTO_GENERATE_CITATIONS",
        description="Automatically generate citations from research"
    )
    deduplicate_sources: bool = Field(
        default=True,
        validation_alias="DEDUPLICATE_SOURCES",
        description="Remove duplicate sources from bibliography"
    )
    include_source_urls: bool = Field(
        default=True,
        validation_alias="INCLUDE_SOURCE_URLS",
        description="Include URLs in bibliography entries"
    )
    
    # Index generation settings
    auto_generate_index_terms: bool = Field(
        default=True,
        validation_alias="AUTO_GENERATE_INDEX_TERMS",
        description="Automatically identify index terms"
    )
    index_term_frequency_threshold: int = Field(
        default=3,
        validation_alias="INDEX_TERM_FREQUENCY_THRESHOLD",
        description="Minimum frequency for auto-generated index terms",
        ge=1,
        le=10
    )
    include_subindex_terms: bool = Field(
        default=True,
        validation_alias="INCLUDE_SUBINDEX_TERMS",
        description="Include hierarchical index terms"
    )
    
    # Content processing settings
    max_chapter_processing_time: int = Field(
        default=120,
        validation_alias="MAX_CHAPTER_PROCESSING_TIME",
        description="Maximum time to process each chapter in seconds",
        ge=30,
        le=600
    )
    enable_parallel_formatting: bool = Field(
        default=True,
        validation_alias="ENABLE_PARALLEL_FORMATTING",
        description="Enable parallel processing for multiple formats"
    )
    cache_formatted_content: bool = Field(
        default=True,
        validation_alias="CACHE_FORMATTED_CONTENT",
        description="Cache formatted content to avoid reprocessing"
    )
    
    # Output file settings
    output_directory: str = Field(
        default="./generated_books",
        validation_alias="BOOK_OUTPUT_DIRECTORY",
        description="Directory for generated book files"
    )
    filename_template: str = Field(
        default="{title}_{version}_{timestamp}",
        validation_alias="FILENAME_TEMPLATE",
        description="Template for generated filenames"
    )
    include_metadata_file: bool = Field(
        default=True,
        validation_alias="INCLUDE_METADATA_FILE",
        description="Generate metadata file with book information"
    )
    compress_output: bool = Field(
        default=False,
        validation_alias="COMPRESS_OUTPUT",
        description="Compress output files into archives"
    )
    
    # Validation and integrity settings
    content_integrity_check: bool = Field(
        default=True,
        validation_alias="CONTENT_INTEGRITY_CHECK",
        description="Perform content integrity validation"
    )
    chapter_word_count_validation: bool = Field(
        default=True,
        validation_alias="CHAPTER_WORD_COUNT_VALIDATION",
        description="Validate chapter word counts meet targets"
    )
    structure_completeness_check: bool = Field(
        default=True,
        validation_alias="STRUCTURE_COMPLETENESS_CHECK",
        description="Check structural completeness of the book"
    )
    generate_assembly_report: bool = Field(
        default=True,
        validation_alias="GENERATE_ASSEMBLY_REPORT",
        description="Generate detailed assembly process report"
    )
    
    # Performance settings
    memory_optimization: bool = Field(
        default=True,
        validation_alias="MEMORY_OPTIMIZATION",
        description="Enable memory optimization during processing"
    )
    max_concurrent_formats: int = Field(
        default=3,
        validation_alias="MAX_CONCURRENT_FORMATS",
        description="Maximum concurrent format generations",
        ge=1,
        le=6
    )
    processing_timeout: int = Field(
        default=1800,
        validation_alias="PROCESSING_TIMEOUT",
        description="Maximum processing timeout in seconds",
        ge=300,
        le=3600
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )