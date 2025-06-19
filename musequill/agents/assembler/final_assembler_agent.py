"""
Final Assembler Agent - Clean Implementation

Compiles all approved chapters into final book format with professional formatting.
Generates table of contents, index, bibliography, and multiple output formats.

This is a streamlined version focusing on core functionality with external
data structures and utilities for better maintainability.
"""

import os
import time
import tempfile
import shutil
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from pathlib import Path
from uuid import uuid4

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from musequill.config.logging import get_logger
from musequill.agents.assembler.final_assembler_config import FinalAssemblerConfig
from musequill.agents.assembler.final_assembler_structures import (
    BookMetadata, TableOfContentsEntry, IndexEntry, BibliographyEntry,
    FormattedDocument, AssemblyResults, AssemblyPhase, AssemblyPhaseResult,
    DocumentFormat, ValidationStatus,
    BookStructureValidationModel, TableOfContentsEnhancementModel,
    IndexGenerationModel, ContentFormattingModel,
    DEFAULT_FORMATTING_OPTIONS, QUALITY_THRESHOLDS, VALIDATION_RULES
)
from musequill.agents.assembler.final_assembler_generators import (
    PDFGenerator, EPUBGenerator, DOCXGenerator, HTMLGenerator, MarkdownGenerator
)
from musequill.agents.assembler.final_assembler_utils import (
    validate_book_structure, extract_search_keywords, create_content_hash,
    format_bibliography_entries, optimize_table_of_contents
)
from musequill.agents.agent_state import BookWritingState, Chapter

logger = get_logger(__name__)


class FinalAssemblerAgent:
    """
    Final Assembler Agent - Clean, focused implementation for professional book assembly.
    """
    
    def __init__(self, config: Optional[FinalAssemblerConfig] = None):
        self.config = config or FinalAssemblerConfig()
        
        # Initialize LLM for content enhancement
        self.llm = ChatOpenAI(
            api_key=self.config.openai_api_key,
            model=self.config.llm_model,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens
        )
        
        # Initialize format generators
        self.generators = self._initialize_generators()
        
        # Working directory
        self.temp_dir = Path(self.config.temp_directory)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Assembly statistics
        self.assembly_stats = {
            'books_assembled': 0,
            'total_formats_generated': 0,
            'assembly_failures': 0,
            'average_assembly_time': 0.0,
            'session_start': time.time()
        }
        
        logger.info("Final Assembler Agent initialized")
    
    def _initialize_generators(self) -> Dict[str, Any]:
        """Initialize format-specific generators."""
        generators = {}
        
        try:
            generators[DocumentFormat.PDF] = PDFGenerator(self.config)
            generators[DocumentFormat.EPUB] = EPUBGenerator(self.config)
            generators[DocumentFormat.DOCX] = DOCXGenerator(self.config)
            generators[DocumentFormat.HTML] = HTMLGenerator(self.config)
            generators[DocumentFormat.MARKDOWN] = MarkdownGenerator(self.config)
            
            logger.info(f"Initialized {len(generators)} format generators")
        except Exception as e:
            logger.warning(f"Error initializing generators: {e}")
        
        return generators
    
    def assemble_final_book(self, state: BookWritingState) -> Dict[str, Any]:
        """
        Main entry point for final book assembly.
        """
        assembly_start = datetime.now()
        phase_results = []
        
        try:
            logger.info(f"Starting final assembly for book {state['book_id']}")
            
            # Phase 1: Structure Validation
            validation_result = self._execute_phase(
                AssemblyPhase.VALIDATION,
                self._validate_book_structure,
                state
            )
            phase_results.append(validation_result)
            
            if not validation_result.success:
                return self._create_failure_result(state, phase_results, "Structure validation failed")
            
            # Phase 2: Metadata Generation
            metadata_result = self._execute_phase(
                AssemblyPhase.METADATA_GENERATION,
                self._generate_book_metadata,
                state
            )
            phase_results.append(metadata_result)
            book_metadata = metadata_result.output_data.get('metadata')
            
            # Phase 3: Content Assembly
            content_result = self._execute_phase(
                AssemblyPhase.CONTENT_ASSEMBLY,
                self._assemble_book_content,
                state, book_metadata
            )
            phase_results.append(content_result)
            
            if not content_result.success:
                return self._create_failure_result(state, phase_results, "Content assembly failed")
            
            # Phase 4: Format Generation
            format_result = self._execute_phase(
                AssemblyPhase.FORMAT_GENERATION,
                self._generate_all_formats,
                state, book_metadata, content_result.output_data
            )
            phase_results.append(format_result)
            
            # Phase 5: Quality Validation
            quality_result = self._execute_phase(
                AssemblyPhase.QUALITY_VALIDATION,
                self._validate_generated_content,
                format_result.output_data.get('generated_formats', [])
            )
            phase_results.append(quality_result)
            
            # Phase 6: Finalization
            final_result = self._execute_phase(
                AssemblyPhase.FINALIZATION,
                self._finalize_assembly,
                state, book_metadata, format_result.output_data, phase_results
            )
            phase_results.append(final_result)
            
            # Create final assembly results
            assembly_end = datetime.now()
            total_duration = (assembly_end - assembly_start).total_seconds()
            
            assembly_results = AssemblyResults(
                book_id=state['book_id'],
                success=final_result.success,
                overall_start_time=assembly_start,
                overall_end_time=assembly_end,
                total_duration=total_duration,
                total_word_count=book_metadata.word_count,
                total_character_count=content_result.output_data.get('character_count', 0),
                generated_formats=format_result.output_data.get('generated_formats', []),
                failed_formats=format_result.output_data.get('failed_formats', []),
                metadata=book_metadata,
                table_of_contents=content_result.output_data.get('table_of_contents', []),
                index_entries=content_result.output_data.get('index_entries', []),
                bibliography=content_result.output_data.get('bibliography', []),
                phase_results=phase_results,
                validation_results=validation_result.output_data,
                quality_metrics=quality_result.output_data.get('quality_metrics', {}),
                assembly_statistics=self._calculate_assembly_statistics(phase_results)
            )
            
            # Update agent statistics
            self._update_assembly_stats(total_duration, len(assembly_results.generated_formats))
            
            return {
                'status': 'success',
                'assembly_results': assembly_results,
                'success': assembly_results.success,
                'generated_formats': [f.format_type for f in assembly_results.generated_formats],
                'failed_formats': assembly_results.failed_formats,
                'total_word_count': assembly_results.total_word_count,
                'assembly_time': total_duration,
                'quality_score': assembly_results.quality_metrics.get('overall_quality', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Fatal error during assembly: {e}")
            self.assembly_stats['assembly_failures'] += 1
            
            return {
                'status': 'error',
                'error_message': str(e),
                'success': False,
                'assembly_time': (datetime.now() - assembly_start).total_seconds()
            }
    
    def _execute_phase(self, phase: AssemblyPhase, phase_func, *args) -> AssemblyPhaseResult:
        """Execute an assembly phase with tracking."""
        start_time = datetime.now()
        
        try:
            logger.info(f"Executing phase: {phase.value}")
            
            # Execute the phase function
            result_data = phase_func(*args)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return AssemblyPhaseResult(
                phase=phase,
                success=True,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                output_data=result_data if isinstance(result_data, dict) else {'result': result_data}
            )
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.error(f"Phase {phase.value} failed: {e}")
            
            return AssemblyPhaseResult(
                phase=phase,
                success=False,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                errors=[str(e)]
            )
    
    def _validate_book_structure(self, state: BookWritingState) -> Dict[str, Any]:
        """Validate the overall book structure and content."""
        validation_results = validate_book_structure(state, self.config)
        
        # Enhanced LLM validation if enabled
        if self.config.validate_structure and validation_results['basic_validation_passed']:
            try:
                llm_validation = self._llm_validate_structure(state)
                validation_results.update({
                    'llm_validation': llm_validation,
                    'enhanced_validation_performed': True
                })
            except Exception as e:
                logger.warning(f"LLM validation failed: {e}")
                validation_results['llm_validation_error'] = str(e)
        
        return validation_results
    
    def _llm_validate_structure(self, state: BookWritingState) -> Dict[str, Any]:
        """Use LLM to validate book structure and coherence."""
        # Prepare chapter summaries
        chapter_summaries = []
        for chapter in state['chapters']:
            if chapter.get('status') == 'complete' and chapter.get('content'):
                content = chapter['content']
                # Extract first and last paragraphs for flow analysis
                paragraphs = content.split('\n\n')
                summary = {
                    'number': chapter['chapter_number'],
                    'title': chapter['title'],
                    'word_count': len(content.split()),
                    'opening': paragraphs[0][:200] if paragraphs else '',
                    'closing': paragraphs[-1][:200] if len(paragraphs) > 1 else ''
                }
                chapter_summaries.append(summary)
        
        system_prompt = f"""You are an expert book editor analyzing the structural integrity of a {state['genre']} book.

Assess the book structure for publication readiness focusing on:
1. Logical flow and chapter progression
2. Content coherence and consistency  
3. Structural completeness
4. Chapter transitions and continuity
5. Overall narrative or argumentative structure

Provide specific, actionable feedback."""

        human_prompt = f"""BOOK TO ANALYZE:
Title: {state['title']}
Genre: {state['genre']}
Target Audience: {state.get('target_audience', 'General')}
Chapters: {len(chapter_summaries)}

CHAPTER STRUCTURE:
{chr(10).join([f"Chapter {ch['number']}: {ch['title']} ({ch['word_count']} words)" for ch in chapter_summaries])}

SAMPLE CONTENT FLOW:
{chr(10).join([f"Ch{ch['number']} Opening: {ch['opening']}..." for ch in chapter_summaries[:3]])}

Please provide a comprehensive structural assessment."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]

        llm_result = self.llm.with_structured_output(BookStructureValidationModel).invoke(messages)
        
        return {
            'structure_valid': llm_result.structure_valid,
            'coherence_score': llm_result.overall_coherence_score,
            'issues': llm_result.issues_found,
            'critical_issues': llm_result.critical_issues,
            'recommendations': llm_result.improvements_suggested,
            'publication_readiness': llm_result.publication_readiness
        }
    
    def _generate_book_metadata(self, state: BookWritingState) -> Dict[str, Any]:
        """Generate comprehensive book metadata."""
        # Calculate content statistics
        total_words = sum(len(ch.get('content', '').split()) for ch in state['chapters'])
        total_chars = sum(len(ch.get('content', '')) for ch in state['chapters'])
        estimated_pages = max(1, total_words // 250)  # ~250 words per page
        estimated_reading_time = max(1, total_words // 200)  # ~200 words per minute
        
        # Extract keywords and subjects
        keywords = extract_search_keywords(state)
        
        metadata = BookMetadata(
            title=state['title'],
            author=state.get('author', 'MuseQuill AI'),
            genre=state['genre'],
            description=state.get('description', f"A comprehensive {state['genre']} book on {state['title']}"),
            language="en",
            word_count=total_words,
            chapter_count=len(state['chapters']),
            page_count=estimated_pages,
            estimated_reading_time=estimated_reading_time,
            quality_score=state.get('quality_score'),
            target_audience=state.get('target_audience'),
            keywords=keywords[:20],  # Top 20 keywords
            subjects=[state['genre']],
            generation_metadata={
                'pipeline_version': '1.0',
                'generation_timestamp': datetime.now(timezone.utc).isoformat(),
                'agents_used': state.get('agents_used', []),
                'processing_time': state.get('total_processing_time', 0),
                'revision_count': state.get('revision_count', 0)
            }
        )
        
        return {'metadata': metadata}
    
    def _assemble_book_content(self, state: BookWritingState, metadata: BookMetadata) -> Dict[str, Any]:
        """Assemble all book content components."""
        # Generate table of contents
        toc = self._generate_table_of_contents(state)
        
        # Generate index if enabled
        index_entries = []
        if self.config.enable_index:
            index_entries = self._generate_index(state)
        
        # Generate bibliography if enabled
        bibliography = []
        if self.config.enable_bibliography:
            bibliography = self._generate_bibliography(state)
        
        # Assemble master content
        master_content = self._create_master_content(state, metadata, toc, index_entries, bibliography)
        
        return {
            'master_content': master_content,
            'table_of_contents': toc,
            'index_entries': index_entries,
            'bibliography': bibliography,
            'character_count': len(master_content)
        }
    
    def _generate_table_of_contents(self, state: BookWritingState) -> List[TableOfContentsEntry]:
        """Generate enhanced table of contents."""
        if not self.config.enable_table_of_contents:
            return []
        
        toc_entries = []
        
        # Generate basic TOC from chapters
        for chapter in sorted(state['chapters'], key=lambda x: x.get('chapter_number', 0)):
            if chapter.get('status') == 'complete':
                entry = TableOfContentsEntry(
                    level=1,
                    number=str(chapter['chapter_number']),
                    title=chapter['title'],
                    word_count=len(chapter.get('content', '').split()),
                    estimated_reading_time=max(1, len(chapter.get('content', '').split()) // 200),
                    content_type="chapter"
                )
                toc_entries.append(entry)
        
        # Enhance with LLM if available
        try:
            enhanced_toc = self._llm_enhance_toc(toc_entries, state)
            return enhanced_toc
        except Exception as e:
            logger.warning(f"TOC enhancement failed: {e}")
            return toc_entries
    
    def _llm_enhance_toc(self, basic_toc: List[TableOfContentsEntry], state: BookWritingState) -> List[TableOfContentsEntry]:
        """Use LLM to enhance table of contents structure."""
        toc_data = []
        for entry in basic_toc:
            toc_data.append({
                'level': entry.level,
                'number': entry.number,
                'title': entry.title,
                'word_count': entry.word_count,
                'reading_time': entry.estimated_reading_time
            })
        
        system_prompt = """You are a professional book formatter creating an enhanced table of contents.

Analyze the chapter structure and suggest improvements such as:
- Better organization and grouping
- Improved chapter titles for clarity
- Logical flow assessment
- Section groupings where appropriate

Maintain the original structure while enhancing readability and navigation."""

        human_prompt = f"""BOOK: {state['title']} ({state['genre']})
TARGET AUDIENCE: {state.get('target_audience', 'General')}

CURRENT TABLE OF CONTENTS:
{chr(10).join([f"{entry['number']}. {entry['title']} ({entry['word_count']} words)" for entry in toc_data])}

Please enhance this table of contents for better organization and reader navigation."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]

        try:
            enhanced = self.llm.with_structured_output(TableOfContentsEnhancementModel).invoke(messages)
            
            # Convert enhanced entries back to TableOfContentsEntry objects
            enhanced_entries = []
            for entry_data in enhanced.enhanced_entries:
                entry = TableOfContentsEntry(
                    level=entry_data.get('level', 1),
                    number=entry_data.get('number', ''),
                    title=entry_data.get('title', ''),
                    word_count=entry_data.get('word_count'),
                    estimated_reading_time=entry_data.get('reading_time'),
                    content_type=entry_data.get('content_type', 'chapter')
                )
                enhanced_entries.append(entry)
            
            return enhanced_entries if enhanced_entries else basic_toc
            
        except Exception as e:
            logger.warning(f"TOC enhancement failed: {e}")
            return basic_toc
    
    def _generate_index(self, state: BookWritingState) -> List[IndexEntry]:
        """Generate comprehensive book index."""
        if not self.config.enable_index:
            return []
        
        try:
            # Extract content for index generation
            all_content = []
            for chapter in state['chapters']:
                if chapter.get('content'):
                    all_content.append(chapter['content'])
            
            combined_content = ' '.join(all_content)
            
            # Limit content size for LLM processing
            if len(combined_content.split()) > 3000:
                words = combined_content.split()
                sample_content = ' '.join(words[:1500]) + ' [...] ' + ' '.join(words[-1500:])
            else:
                sample_content = combined_content
            
            return self._llm_generate_index(sample_content, state)
            
        except Exception as e:
            logger.error(f"Index generation failed: {e}")
            return []
    
    def _llm_generate_index(self, content: str, state: BookWritingState) -> List[IndexEntry]:
        """Use LLM to generate comprehensive index."""
        system_prompt = f"""You are a professional indexer creating a comprehensive index for a {state['genre']} book.

Identify key terms, concepts, people, places, and important topics that readers would want to reference.

Focus on:
- Technical terms and concepts
- Important names and places  
- Key themes and topics
- Cross-references and related terms
- Frequently referenced items

Categorize terms by importance (1-5, where 5 is most important)."""

        human_prompt = f"""BOOK: {state['title']}
GENRE: {state['genre']}

CONTENT TO INDEX:
{content}

Generate a comprehensive index with key terms, importance rankings, and categories."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]

        try:
            index_data = self.llm.with_structured_output(IndexGenerationModel).invoke(messages)
            
            # Convert to IndexEntry objects
            index_entries = []
            for term_data in index_data.index_terms:
                entry = IndexEntry(
                    term=term_data.get('term', ''),
                    page_references=[1],  # Placeholder - would need actual page mapping
                    importance=term_data.get('importance', 1),
                    category=term_data.get('category'),
                    see_also=term_data.get('see_also', []),
                    subentries=term_data.get('subentries', {})
                )
                index_entries.append(entry)
            
            return index_entries
            
        except Exception as e:
            logger.warning(f"LLM index generation failed: {e}")
            return []
    
    def _generate_bibliography(self, state: BookWritingState) -> List[BibliographyEntry]:
        """Generate bibliography from research sources."""
        if not self.config.enable_bibliography:
            return []
        
        try:
            # Collect research chunks used across chapters
            used_chunks = set()
            for chapter in state['chapters']:
                chunk_ids = chapter.get('research_chunks_used', [])
                used_chunks.update(chunk_ids)
            
            if not used_chunks:
                logger.info("No research chunks found for bibliography")
                return []
            
            # Here you would typically query the research database
            # For now, create placeholder bibliography entries
            bibliography = []
            
            # This would be replaced with actual research source retrieval
            # For demonstration, creating sample entries
            sample_sources = [
                {
                    'title': 'Research Source 1',
                    'authors': ['Author, A.'],
                    'url': 'https://example.com/source1',
                    'access_date': datetime.now().isoformat()[:10]
                },
                {
                    'title': 'Research Source 2', 
                    'authors': ['Author, B.', 'Author, C.'],
                    'url': 'https://example.com/source2',
                    'access_date': datetime.now().isoformat()[:10]
                }
            ]
            
            for i, source in enumerate(sample_sources):
                entry = BibliographyEntry(
                    entry_id=f"ref_{i+1}",
                    title=source['title'],
                    authors=source['authors'],
                    url=source.get('url'),
                    access_date=source.get('access_date'),
                    publication_type='website'
                )
                bibliography.append(entry)
            
            return bibliography
            
        except Exception as e:
            logger.error(f"Bibliography generation failed: {e}")
            return []
    
    def _create_master_content(
        self, 
        state: BookWritingState, 
        metadata: BookMetadata,
        toc: List[TableOfContentsEntry],
        index_entries: List[IndexEntry],
        bibliography: List[BibliographyEntry]
    ) -> str:
        """Create the master content document."""
        content_parts = []
        
        # Title page
        content_parts.append(self._format_title_page(metadata))
        content_parts.append("\n\n\\pagebreak\n\n")
        
        # Copyright page
        if self.config.include_copyright_notice:
            content_parts.append(self._format_copyright_page(metadata))
            content_parts.append("\n\n\\pagebreak\n\n")
        
        # Table of contents
        if self.config.enable_table_of_contents and toc:
            content_parts.append(self._format_table_of_contents(toc))
            content_parts.append("\n\n\\pagebreak\n\n")
        
        # Main content (chapters)
        for chapter in sorted(state['chapters'], key=lambda x: x.get('chapter_number', 0)):
            if chapter.get('status') == 'complete':
                content_parts.append(f"\\chapter{{{chapter['title']}}}")
                content_parts.append("\n\n")
                content_parts.append(chapter.get('content', ''))
                content_parts.append("\n\n\\pagebreak\n\n")
        
        # Bibliography
        if self.config.enable_bibliography and bibliography:
            content_parts.append("\\chapter{Bibliography}")
            content_parts.append("\n\n")
            content_parts.append(self._format_bibliography(bibliography))
            content_parts.append("\n\n\\pagebreak\n\n")
        
        # Index
        if self.config.enable_index and index_entries:
            content_parts.append("\\chapter{Index}")
            content_parts.append("\n\n")
            content_parts.append(self._format_index(index_entries))
        
        return ''.join(content_parts)
    
    def _format_title_page(self, metadata: BookMetadata) -> str:
        """Format the title page."""
        return f"""\\begin{{titlepage}}
\\centering
\\vspace*{{2cm}}

{{\\Huge\\bfseries {metadata.title}}}

\\vspace{{1.5cm}}

{{\\Large by}}

\\vspace{{0.5cm}}

{{\\LARGE {metadata.author}}}

\\vspace{{2cm}}

{{\\large {metadata.genre}}}

\\vspace{{\\fill}}

{{\\large {metadata.publisher}}}

{{\\normalsize {datetime.now().strftime('%B %d, %Y')}}}

\\end{{titlepage}}"""
    
    def _format_copyright_page(self, metadata: BookMetadata) -> str:
        """Format the copyright page."""
        year = datetime.now().year
        return f"""\\thispagestyle{{empty}}

\\vspace*{{\\fill}}

\\noindent
Copyright © {year} {metadata.author}

\\vspace{{0.5cm}}

\\noindent
This book was generated using MuseQuill AI on {metadata.generation_timestamp[:10]}.

\\vspace{{0.5cm}}

\\noindent
All rights reserved. No part of this publication may be reproduced, distributed, or transmitted in any form or by any means, including photocopying, recording, or other electronic or mechanical methods, without the prior written permission of the author.

\\vspace{{0.5cm}}

\\noindent
First Edition: {metadata.version}

\\noindent
Word Count: {metadata.word_count:,} words

\\noindent
Chapter Count: {metadata.chapter_count} chapters

\\vspace*{{\\fill}}"""
    
    def _format_table_of_contents(self, toc: List[TableOfContentsEntry]) -> str:
        """Format table of contents."""
        toc_lines = ["\\tableofcontents\n\n"]
        
        for entry in toc:
            if entry.level == 1:
                toc_lines.append(f"Chapter {entry.number}: {entry.title}")
            else:
                indent = "  " * (entry.level - 1)
                toc_lines.append(f"{indent}{entry.title}")
            
            if entry.page_number:
                toc_lines[-1] += f" \\dotfill {entry.page_number}"
            
            toc_lines.append("\n")
        
        return ''.join(toc_lines)
    
    def _format_bibliography(self, bibliography: List[BibliographyEntry]) -> str:
        """Format bibliography entries."""
        bib_lines = []
        
        for entry in sorted(bibliography, key=lambda x: x.title.lower()):
            citation = f"\\noindent {entry.title}"
            
            if entry.authors:
                authors = ', '.join(entry.authors)
                citation = f"\\noindent {authors}. {entry.title}"
            
            if entry.url:
                citation += f". Retrieved from {entry.url}"
            
            if entry.access_date:
                citation += f" (accessed {entry.access_date})"
            
            bib_lines.append(f"{citation}\n\n")
        
        return ''.join(bib_lines)
    
    def _format_index(self, index_entries: List[IndexEntry]) -> str:
        """Format index entries."""
        index_lines = []
        
        # Sort by term, with importance as secondary sort
        sorted_entries = sorted(index_entries, key=lambda x: (x.term.lower(), -x.importance))
        
        for entry in sorted_entries:
            line = f"\\noindent {entry.term}"
            
            if entry.page_references:
                pages = ", ".join(map(str, entry.page_references))
                line += f" \\dotfill {pages}"
            
            index_lines.append(f"{line}\n")
            
            # Add subentries
            if entry.subentries:
                for subterm, subpages in entry.subentries.items():
                    subpage_str = ", ".join(map(str, subpages))
                    index_lines.append(f"\\quad {subterm} \\dotfill {subpage_str}\n")
            
            # Add see-also references
            if entry.see_also:
                see_also_str = ", ".join(entry.see_also)
                index_lines.append(f"\\quad \\textit{{See also:}} {see_also_str}\n")
        
        return ''.join(index_lines)
    
    def _generate_all_formats(
        self, 
        state: BookWritingState, 
        metadata: BookMetadata, 
        content_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate all requested output formats."""
        generated_formats = []
        failed_formats = []
        
        master_content = content_data['master_content']
        
        for format_type in self.config.output_formats:
            try:
                if format_type in self.generators:
                    generator = self.generators[format_type]
                    
                    # Generate the format
                    formatted_doc = generator.generate(
                        content=master_content,
                        metadata=metadata,
                        output_path=self.temp_dir / f"{state['book_id']}.{format_type}",
                        options=DEFAULT_FORMATTING_OPTIONS.get(format_type, {})
                    )
                    
                    if formatted_doc:
                        generated_formats.append(formatted_doc)
                        logger.info(f"Successfully generated {format_type} format")
                    else:
                        failed_formats.append(format_type)
                        logger.error(f"Failed to generate {format_type} format")
                else:
                    logger.warning(f"No generator available for format: {format_type}")
                    failed_formats.append(format_type)
                    
            except Exception as e:
                logger.error(f"Error generating {format_type} format: {e}")
                failed_formats.append(format_type)
                
                if not self.config.fallback_on_format_failure:
                    break
        
        return {
            'generated_formats': generated_formats,
            'failed_formats': failed_formats
        }
    
    def _validate_generated_content(self, generated_formats: List[FormattedDocument]) -> Dict[str, Any]:
        """Validate generated content for quality and integrity."""
        quality_metrics = {}
        validation_results = {}
        
        try:
            total_files = len(generated_formats)
            valid_files = 0
            total_size = 0
            
            for formatted_doc in generated_formats:
                # Basic file validation
                if formatted_doc.file_path.exists():
                    file_size = formatted_doc.file_path.stat().st_size
                    total_size += file_size
                    
                    if file_size > 0:
                        valid_files += 1
                        formatted_doc.validation_status = ValidationStatus.VALID
                    else:
                        formatted_doc.validation_status = ValidationStatus.INVALID
                        formatted_doc.validation_errors.append("Empty file generated")
                else:
                    formatted_doc.validation_status = ValidationStatus.INVALID
                    formatted_doc.validation_errors.append("File not found")
            
            quality_metrics = {
                'overall_quality': valid_files / total_files if total_files > 0 else 0.0,
                'generation_success_rate': valid_files / total_files if total_files > 0 else 0.0,
                'total_file_size_mb': total_size / (1024 * 1024),
                'valid_formats': valid_files,
                'total_formats': total_files
            }
            
            validation_results = {
                'all_formats_valid': valid_files == total_files,
                'partial_success': valid_files > 0,
                'total_files_validated': total_files,
                'validation_performed': True
            }
            
        except Exception as e:
            logger.error(f"Content validation failed: {e}")
            quality_metrics['validation_error'] = str(e)
            validation_results['validation_performed'] = False
        
        return {
            'quality_metrics': quality_metrics,
            'validation_results': validation_results
        }
    
    def _finalize_assembly(
        self, 
        state: BookWritingState, 
        metadata: BookMetadata,
        format_data: Dict[str, Any],
        phase_results: List[AssemblyPhaseResult]
    ) -> Dict[str, Any]:
        """Finalize the assembly process."""
        try:
            # Calculate final statistics
            total_phases = len(phase_results)
            successful_phases = sum(1 for phase in phase_results if phase.success)
            
            # Determine overall success
            assembly_successful = (
                successful_phases >= total_phases - 1 and  # Allow one non-critical phase to fail
                len(format_data.get('generated_formats', [])) > 0  # At least one format generated
            )
            
            # Create final summary
            summary = {
                'assembly_successful': assembly_successful,
                'phases_completed': successful_phases,
                'total_phases': total_phases,
                'formats_generated': len(format_data.get('generated_formats', [])),
                'formats_failed': len(format_data.get('failed_formats', [])),
                'final_word_count': metadata.word_count,
                'estimated_pages': metadata.page_count,
                'finalization_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Cleanup temporary files if configured
            if self.config.cleanup_temp_files:
                self._cleanup_temporary_files(state['book_id'])
            
            logger.info(f"Assembly finalization complete for book {state['book_id']}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Finalization failed: {e}")
            return {
                'assembly_successful': False,
                'finalization_error': str(e)
            }
    
    def _cleanup_temporary_files(self, book_id: str) -> None:
        """Clean up temporary files created during assembly."""
        try:
            # Remove book-specific temporary files
            for temp_file in self.temp_dir.glob(f"{book_id}*"):
                if temp_file.is_file():
                    temp_file.unlink()
                    
            logger.info(f"Cleaned up temporary files for book {book_id}")
            
        except Exception as e:
            logger.warning(f"Cleanup failed for book {book_id}: {e}")
    
    def _create_failure_result(
        self, 
        state: BookWritingState, 
        phase_results: List[AssemblyPhaseResult], 
        error_message: str
    ) -> Dict[str, Any]:
        """Create a failure result for assembly."""
        return {
            'status': 'error',
            'error_message': error_message,
            'success': False,
            'phase_results': phase_results,
            'book_id': state['book_id'],
            'assembly_time': sum(phase.duration for phase in phase_results)
        }
    
    def _calculate_assembly_statistics(self, phase_results: List[AssemblyPhaseResult]) -> Dict[str, Any]:
        """Calculate detailed assembly statistics."""
        return {
            'total_phases': len(phase_results),
            'successful_phases': sum(1 for phase in phase_results if phase.success),
            'total_duration': sum(phase.duration for phase in phase_results),
            'average_phase_duration': sum(phase.duration for phase in phase_results) / len(phase_results) if phase_results else 0,
            'longest_phase': max((phase.duration for phase in phase_results), default=0),
            'phase_breakdown': {
                phase.phase.value: {
                    'duration': phase.duration,
                    'success': phase.success,
                    'items_processed': phase.items_processed
                } for phase in phase_results
            }
        }
    
    def _update_assembly_stats(self, assembly_time: float, formats_generated: int) -> None:
        """Update internal assembly statistics."""
        self.assembly_stats['books_assembled'] += 1
        self.assembly_stats['total_formats_generated'] += formats_generated
        
        # Update average assembly time
        current_avg = self.assembly_stats['average_assembly_time']
        books_count = self.assembly_stats['books_assembled']
        new_avg = ((current_avg * (books_count - 1)) + assembly_time) / books_count
        self.assembly_stats['average_assembly_time'] = new_avg
    
    def get_assembly_statistics(self) -> Dict[str, Any]:
        """Get comprehensive assembly statistics."""
        stats = self.assembly_stats.copy()
        
        # Add session information
        session_duration = time.time() - stats['session_start']
        stats['session_duration_minutes'] = session_duration / 60
        
        # Calculate additional metrics
        if stats['books_assembled'] > 0:
            stats['success_rate'] = 1 - (stats['assembly_failures'] / stats['books_assembled'])
            stats['avg_formats_per_book'] = stats['total_formats_generated'] / stats['books_assembled']
        
        return stats
    
    def cleanup_resources(self) -> bool:
        """Clean up resources and temporary files."""
        try:
            if self.config.cleanup_temp_files and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
            
            logger.info("Final Assembler Agent resources cleaned up")
            return True
        except Exception as e:
            logger.error(f"Error cleaning up resources: {e}")
            return False


def main():
    """Test function for FinalAssemblerAgent."""
    config = FinalAssemblerConfig()
    agent = FinalAssemblerAgent(config)
    
    # Create test state
    test_state = {
        'book_id': 'test_final_assembly_001',
        'title': 'Complete Guide to Final Assembly',
        'author': 'MuseQuill AI',
        'genre': 'Technical',
        'target_word_count': 15000,
        'target_audience': 'Developers',
        'description': 'A comprehensive guide to final book assembly',
        'chapters': [
            {
                'chapter_number': 1,
                'title': 'Introduction',
                'content': 'This is the introduction to final assembly...',
                'status': 'complete'
            },
            {
                'chapter_number': 2,
                'title': 'Assembly Process',
                'content': 'This chapter covers the assembly process...',
                'status': 'complete'
            }
        ]
    }
    
    try:
        print("Testing Final Assembler Agent...")
        result = agent.assemble_final_book(test_state)
        
        print(f"Assembly Status: {result.get('status')}")
        print(f"Success: {result.get('success')}")
        print(f"Generated Formats: {result.get('generated_formats', [])}")
        print(f"Assembly Time: {result.get('assembly_time', 0):.2f} seconds")
        
        # Test statistics
        stats = agent.get_assembly_statistics()
        print(f"Books Assembled: {stats['books_assembled']}")
        print(f"Average Assembly Time: {stats['average_assembly_time']:.2f} seconds")
        
        print("FinalAssemblerAgent test completed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        agent.cleanup_resources()


if __name__ == "__main__":
    main()