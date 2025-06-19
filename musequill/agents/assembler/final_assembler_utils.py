"""
Final Assembler Agent Utilities

Utility functions for validation, content processing, and optimization
used by the Final Assembler Agent.
"""

import hashlib
import re
from typing import Dict, List, Any, Set, Tuple
from collections import Counter

from musequill.config.logging import get_logger
from musequill.agents.assembler.final_assembler_structures import (
    QUALITY_THRESHOLDS, VALIDATION_RULES
)
from musequill.agents.agent_state import BookWritingState

logger = get_logger(__name__)


def validate_book_structure(state: BookWritingState, config) -> Dict[str, Any]:
    """Comprehensive validation of book structure and content."""
    validation_results = {
        'basic_validation_passed': True,
        'validation_errors': [],
        'validation_warnings': [],
        'content_statistics': {},
        'structure_analysis': {}
    }
    
    try:
        # Basic validation
        basic_results = _validate_basic_requirements(state)
        validation_results.update(basic_results)
        
        # Content validation
        content_results = _validate_content_quality(state)
        validation_results['content_statistics'] = content_results
        
        # Structure validation
        structure_results = _validate_chapter_structure(state)
        validation_results['structure_analysis'] = structure_results
        
        # Check if validation passed
        validation_results['basic_validation_passed'] = len(validation_results['validation_errors']) == 0
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        validation_results['validation_errors'].append(f"Validation system error: {str(e)}")
        validation_results['basic_validation_passed'] = False
    
    return validation_results


def _validate_basic_requirements(state: BookWritingState) -> Dict[str, Any]:
    """Validate basic book requirements."""
    errors = []
    warnings = []
    
    # Required fields validation
    for field in VALIDATION_RULES['required_metadata_fields']:
        if field not in state or not state[field]:
            errors.append(f"Missing required field: {field}")
    
    # Chapter validation
    chapters = state.get('chapters', [])
    if not chapters:
        errors.append("Book must have at least one chapter")
    else:
        completed_chapters = [ch for ch in chapters if ch.get('status') == 'complete' and ch.get('content')]
        if not completed_chapters:
            errors.append("No completed chapters found")
        
        # Validate individual chapters
        for i, chapter in enumerate(completed_chapters):
            for field in VALIDATION_RULES['required_chapter_fields']:
                if field not in chapter or not chapter[field]:
                    errors.append(f"Chapter {i+1} missing required field: {field}")
    
    # Word count validation
    total_words = sum(len(ch.get('content', '').split()) for ch in chapters)
    if total_words < QUALITY_THRESHOLDS['minimum_word_count']:
        errors.append(f"Word count too low: {total_words} < {QUALITY_THRESHOLDS['minimum_word_count']}")
    elif total_words > QUALITY_THRESHOLDS['maximum_word_count']:
        warnings.append(f"Word count very high: {total_words} > {QUALITY_THRESHOLDS['maximum_word_count']}")
    
    # Chapter count validation
    if len(chapters) < QUALITY_THRESHOLDS['minimum_chapters']:
        errors.append(f"Too few chapters: {len(chapters)} < {QUALITY_THRESHOLDS['minimum_chapters']}")
    elif len(chapters) > QUALITY_THRESHOLDS['maximum_chapters']:
        warnings.append(f"Many chapters: {len(chapters)} > {QUALITY_THRESHOLDS['maximum_chapters']}")
    
    return {
        'validation_errors': errors,
        'validation_warnings': warnings
    }


def _validate_content_quality(state: BookWritingState) -> Dict[str, Any]:
    """Validate content quality and completeness."""
    chapters = state.get('chapters', [])
    completed_chapters = [ch for ch in chapters if ch.get('status') == 'complete' and ch.get('content')]
    
    statistics = {
        'total_chapters': len(chapters),
        'completed_chapters': len(completed_chapters),
        'total_word_count': 0,
        'average_chapter_length': 0,
        'chapter_length_variance': 0,
        'content_issues': []
    }
    
    if not completed_chapters:
        return statistics
    
    # Calculate word counts
    word_counts = []
    total_words = 0
    
    for chapter in completed_chapters:
        content = chapter.get('content', '')
        word_count = len(content.split())
        word_counts.append(word_count)
        total_words += word_count
        
        # Check for content issues
        if VALIDATION_RULES['content_validation']['check_for_placeholders']:
            if _has_placeholders(content):
                statistics['content_issues'].append(f"Chapter {chapter.get('chapter_number', '?')} contains placeholders")
    
    statistics['total_word_count'] = total_words
    statistics['average_chapter_length'] = total_words / len(completed_chapters) if completed_chapters else 0
    
    # Calculate variance
    if len(word_counts) > 1:
        mean = statistics['average_chapter_length']
        variance = sum((x - mean) ** 2 for x in word_counts) / len(word_counts)
        statistics['chapter_length_variance'] = variance
    
    return statistics


def _validate_chapter_structure(state: BookWritingState) -> Dict[str, Any]:
    """Validate chapter numbering and structure."""
    chapters = state.get('chapters', [])
    completed_chapters = [ch for ch in chapters if ch.get('status') == 'complete']
    
    structure_analysis = {
        'numbering_valid': True,
        'numbering_issues': [],
        'duplicate_titles': [],
        'missing_descriptions': [],
        'structure_score': 1.0
    }
    
    if not completed_chapters:
        return structure_analysis
    
    # Check chapter numbering
    if VALIDATION_RULES['content_validation']['validate_chapter_numbering']:
        chapter_numbers = [ch.get('chapter_number', 0) for ch in completed_chapters]
        expected_numbers = list(range(1, len(completed_chapters) + 1))
        
        if sorted(chapter_numbers) != expected_numbers:
            structure_analysis['numbering_valid'] = False
            structure_analysis['numbering_issues'].append("Chapter numbering is not sequential")
            structure_analysis['structure_score'] -= 0.2
    
    # Check for duplicate titles
    titles = [ch.get('title', '') for ch in completed_chapters]
    title_counts = Counter(titles)
    duplicates = [title for title, count in title_counts.items() if count > 1 and title]
    
    if duplicates:
        structure_analysis['duplicate_titles'] = duplicates
        structure_analysis['structure_score'] -= 0.1 * len(duplicates)
    
    # Check for missing descriptions
    missing_desc = [
        ch.get('chapter_number', '?') for ch in completed_chapters 
        if not ch.get('description', '').strip()
    ]
    
    if missing_desc:
        structure_analysis['missing_descriptions'] = missing_desc
        structure_analysis['structure_score'] -= 0.05 * len(missing_desc)
    
    # Ensure score doesn't go below 0
    structure_analysis['structure_score'] = max(0.0, structure_analysis['structure_score'])
    
    return structure_analysis


def _has_placeholders(content: str) -> bool:
    """Check if content contains placeholder text."""
    placeholder_patterns = [
        r'\[.*?\]',  # [placeholder]
        r'\{.*?\}',  # {placeholder}
        r'TODO',     # TODO items
        r'FIXME',    # FIXME items
        r'XXX',      # XXX placeholders
        r'Lorem ipsum',  # Lorem ipsum text
        r'placeholder',  # Explicit placeholder text
        r'TBD',      # To be determined
        r'DRAFT',    # Draft indicators
    ]
    
    for pattern in placeholder_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return True
    
    return False


def extract_search_keywords(state: BookWritingState) -> List[str]:
    """Extract relevant keywords for search indexing."""
    keywords = set()
    
    # Add basic keywords
    if state.get('title'):
        keywords.update(state['title'].lower().split())
    
    if state.get('genre'):
        keywords.add(state['genre'].lower())
    
    if state.get('description'):
        # Extract meaningful words from description
        desc_words = state['description'].lower().split()
        keywords.update(word for word in desc_words if len(word) > 3)
    
    # Add chapter titles
    for chapter in state.get('chapters', []):
        if chapter.get('title'):
            title_words = chapter['title'].lower().split()
            keywords.update(word for word in title_words if len(word) > 2)
    
    # Add genre-specific keywords
    genre_keywords = {
        'fiction': ['story', 'novel', 'character', 'plot', 'narrative'],
        'non-fiction': ['guide', 'information', 'facts', 'analysis', 'reference'],
        'technical': ['tutorial', 'guide', 'reference', 'documentation', 'manual'],
        'academic': ['research', 'study', 'analysis', 'theory', 'methodology'],
        'business': ['strategy', 'management', 'leadership', 'growth', 'success'],
        'self-help': ['improvement', 'development', 'success', 'motivation', 'change']
    }
    
    genre = state.get('genre', '').lower()
    if genre in genre_keywords:
        keywords.update(genre_keywords[genre])
    
    # Filter and clean keywords
    cleaned_keywords = set()
    for keyword in keywords:
        # Remove non-alphabetic characters and short words
        clean_word = re.sub(r'[^a-zA-Z]', '', keyword.lower())
        if len(clean_word) > 2 and clean_word.isalpha():
            cleaned_keywords.add(clean_word)
    
    # Return top keywords sorted by length (longer words often more specific)
    return sorted(list(cleaned_keywords), key=len, reverse=True)[:50]


def create_content_hash(state: BookWritingState) -> str:
    """Create a hash of the book content for integrity checking."""
    content_parts = []
    
    # Add title and metadata
    content_parts.append(state.get('title', ''))
    content_parts.append(state.get('genre', ''))
    content_parts.append(state.get('author', ''))
    
    # Add chapter content
    for chapter in sorted(state.get('chapters', []), key=lambda x: x.get('chapter_number', 0)):
        if chapter.get('status') == 'complete' and chapter.get('content'):
            content_parts.append(f"Chapter {chapter.get('chapter_number', 0)}: {chapter.get('title', '')}")
            content_parts.append(chapter['content'])
    
    # Create hash
    combined_content = '\n'.join(content_parts)
    return hashlib.sha256(combined_content.encode('utf-8')).hexdigest()


def optimize_table_of_contents(toc_entries: List[Dict[str, Any]], state: BookWritingState) -> List[Dict[str, Any]]:
    """Optimize table of contents for better navigation."""
    optimized_entries = []
    
    try:
        # Sort entries by chapter number
        sorted_entries = sorted(toc_entries, key=lambda x: x.get('number', 0))
        
        for entry in sorted_entries:
            optimized_entry = entry.copy()
            
            # Improve title formatting
            title = entry.get('title', '')
            if title:
                # Capitalize properly
                optimized_entry['title'] = title.title()
                
                # Add reading time estimate
                word_count = entry.get('word_count', 0)
                if word_count > 0:
                    reading_time = max(1, word_count // 200)  # ~200 words per minute
                    optimized_entry['estimated_reading_time'] = f"{reading_time} min"
            
            # Add content type hints
            if 'introduction' in title.lower():
                optimized_entry['content_type'] = 'introduction'
            elif 'conclusion' in title.lower() or 'summary' in title.lower():
                optimized_entry['content_type'] = 'conclusion'
            elif any(word in title.lower() for word in ['appendix', 'reference', 'bibliography']):
                optimized_entry['content_type'] = 'reference'
            else:
                optimized_entry['content_type'] = 'chapter'
            
            optimized_entries.append(optimized_entry)
        
        # Group related chapters if there are many
        if len(optimized_entries) > 10:
            optimized_entries = _group_related_chapters(optimized_entries, state)
        
    except Exception as e:
        logger.warning(f"TOC optimization failed: {e}")
        return toc_entries
    
    return optimized_entries


def _group_related_chapters(entries: List[Dict[str, Any]], state: BookWritingState) -> List[Dict[str, Any]]:
    """Group related chapters into sections for better organization."""
    # Simple grouping based on chapter titles and content
    grouped_entries = []
    current_group = []
    group_threshold = 5  # Group every 5 chapters
    
    for i, entry in enumerate(entries):
        current_group.append(entry)
        
        # Create group every N chapters or at the end
        if len(current_group) >= group_threshold or i == len(entries) - 1:
            if len(current_group) > 1:
                # Create section header
                start_chapter = current_group[0].get('number', 1)
                end_chapter = current_group[-1].get('number', 1)
                
                section_entry = {
                    'level': 0,  # Section level
                    'title': f"Part {(i // group_threshold) + 1}: Chapters {start_chapter}-{end_chapter}",
                    'content_type': 'section',
                    'subsections': current_group
                }
                grouped_entries.append(section_entry)
            else:
                grouped_entries.extend(current_group)
            
            current_group = []
    
    return grouped_entries


def format_bibliography_entries(bibliography: List[Dict[str, Any]], citation_style: str = "APA") -> List[str]:
    """Format bibliography entries according to citation style."""
    formatted_entries = []
    
    for entry in bibliography:
        try:
            if citation_style.upper() == "APA":
                formatted = _format_apa_citation(entry)
            elif citation_style.upper() == "MLA":
                formatted = _format_mla_citation(entry)
            elif citation_style.upper() == "CHICAGO":
                formatted = _format_chicago_citation(entry)
            else:
                formatted = _format_basic_citation(entry)
            
            formatted_entries.append(formatted)
            
        except Exception as e:
            logger.warning(f"Bibliography formatting failed for entry: {e}")
            # Fallback to basic format
            title = entry.get('title', 'Unknown Title')
            url = entry.get('url', '')
            formatted_entries.append(f"{title}. {url}" if url else title)
    
    return sorted(formatted_entries)


def _format_apa_citation(entry: Dict[str, Any]) -> str:
    """Format citation in APA style."""
    parts = []
    
    # Authors
    authors = entry.get('authors', [])
    if authors:
        if len(authors) == 1:
            parts.append(f"{authors[0]}")
        elif len(authors) == 2:
            parts.append(f"{authors[0]} & {authors[1]}")
        else:
            parts.append(f"{authors[0]} et al.")
    
    # Publication date
    pub_date = entry.get('publication_date', '')
    if pub_date:
        # Extract year
        year_match = re.search(r'\d{4}', pub_date)
        if year_match:
            parts.append(f"({year_match.group()})")
    
    # Title
    title = entry.get('title', '')
    if title:
        parts.append(f"{title}")
    
    # Journal/Publisher
    journal = entry.get('journal', '')
    publisher = entry.get('publisher', '')
    if journal:
        parts.append(f"{journal}")
    elif publisher:
        parts.append(f"{publisher}")
    
    # URL and access date
    url = entry.get('url', '')
    access_date = entry.get('access_date', '')
    if url:
        url_part = f"Retrieved from {url}"
        if access_date:
            url_part = f"Retrieved {access_date}, from {url}"
        parts.append(url_part)
    
    return '. '.join(parts) + '.'


def _format_mla_citation(entry: Dict[str, Any]) -> str:
    """Format citation in MLA style."""
    parts = []
    
    # Authors
    authors = entry.get('authors', [])
    if authors:
        # Last, First format for first author
        if ',' in authors[0]:
            parts.append(authors[0])
        else:
            name_parts = authors[0].split()
            if len(name_parts) >= 2:
                parts.append(f"{name_parts[-1]}, {' '.join(name_parts[:-1])}")
            else:
                parts.append(authors[0])
    
    # Title
    title = entry.get('title', '')
    if title:
        parts.append(f'"{title}"')
    
    # Website/Publisher
    website = entry.get('journal', '') or entry.get('publisher', '')
    if website:
        parts.append(f"{website}")
    
    # Date
    pub_date = entry.get('publication_date', '')
    if pub_date:
        parts.append(pub_date)
    
    # URL
    url = entry.get('url', '')
    if url:
        parts.append(f"Web. {url}")
    
    return ', '.join(parts) + '.'


def _format_chicago_citation(entry: Dict[str, Any]) -> str:
    """Format citation in Chicago style."""
    parts = []
    
    # Authors
    authors = entry.get('authors', [])
    if authors:
        parts.append(authors[0])
    
    # Title
    title = entry.get('title', '')
    if title:
        parts.append(f'"{title}"')
    
    # Publication info
    publisher = entry.get('publisher', '')
    pub_date = entry.get('publication_date', '')
    
    if publisher or pub_date:
        pub_info = []
        if publisher:
            pub_info.append(publisher)
        if pub_date:
            pub_info.append(pub_date)
        parts.append(', '.join(pub_info))
    
    # URL
    url = entry.get('url', '')
    access_date = entry.get('access_date', '')
    if url:
        url_part = url
        if access_date:
            url_part += f" (accessed {access_date})"
        parts.append(url_part)
    
    return '. '.join(parts) + '.'


def _format_basic_citation(entry: Dict[str, Any]) -> str:
    """Format basic citation when specific style formatting fails."""
    parts = []
    
    title = entry.get('title', 'Unknown Title')
    authors = entry.get('authors', [])
    url = entry.get('url', '')
    access_date = entry.get('access_date', '')
    
    if authors:
        parts.append(', '.join(authors))
    
    parts.append(title)
    
    if url:
        if access_date:
            parts.append(f"Retrieved {access_date} from {url}")
        else:
            parts.append(f"Retrieved from {url}")
    
    return '. '.join(parts) + '.'


def analyze_content_readability(content: str) -> Dict[str, Any]:
    """Analyze content readability and provide metrics."""
    analysis = {
        'word_count': 0,
        'sentence_count': 0,
        'paragraph_count': 0,
        'average_words_per_sentence': 0,
        'average_sentences_per_paragraph': 0,
        'readability_score': 0,
        'reading_level': 'Unknown',
        'suggestions': []
    }
    
    try:
        # Basic counts
        words = content.split()
        analysis['word_count'] = len(words)
        
        # Count sentences (rough approximation)
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        analysis['sentence_count'] = len(sentences)
        
        # Count paragraphs
        paragraphs = content.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        analysis['paragraph_count'] = len(paragraphs)
        
        # Calculate averages
        if analysis['sentence_count'] > 0:
            analysis['average_words_per_sentence'] = analysis['word_count'] / analysis['sentence_count']
        
        if analysis['paragraph_count'] > 0:
            analysis['average_sentences_per_paragraph'] = analysis['sentence_count'] / analysis['paragraph_count']
        
        # Simple readability assessment
        avg_words = analysis['average_words_per_sentence']
        if avg_words > 25:
            analysis['reading_level'] = 'Graduate'
            analysis['suggestions'].append('Consider breaking up long sentences for better readability')
        elif avg_words > 20:
            analysis['reading_level'] = 'College'
        elif avg_words > 15:
            analysis['reading_level'] = 'High School'
        else:
            analysis['reading_level'] = 'Elementary'
        
        # Calculate simple readability score (0-100)
        # Simple formula based on sentence length and word complexity
        long_words = len([word for word in words if len(word) > 6])
        complexity_ratio = long_words / len(words) if words else 0
        
        # Simple readability score
        if avg_words < 15 and complexity_ratio < 0.2:
            analysis['readability_score'] = 85  # Easy
        elif avg_words < 20 and complexity_ratio < 0.3:
            analysis['readability_score'] = 70  # Moderate
        elif avg_words < 25 and complexity_ratio < 0.4:
            analysis['readability_score'] = 55  # Difficult
        else:
            analysis['readability_score'] = 30  # Very difficult
        
        # Additional suggestions
        if complexity_ratio > 0.4:
            analysis['suggestions'].append('Consider using simpler vocabulary where possible')
        
        if analysis['paragraph_count'] > 0 and analysis['word_count'] / analysis['paragraph_count'] > 200:
            analysis['suggestions'].append('Consider breaking up long paragraphs for better readability')
        
    except Exception as e:
        logger.warning(f"Readability analysis failed: {e}")
        analysis['suggestions'].append('Could not complete readability analysis')
    
    return analysis


def validate_cross_references(content: str) -> Dict[str, Any]:
    """Validate cross-references within the content."""
    validation = {
        'references_found': [],
        'broken_references': [],
        'internal_links': 0,
        'external_links': 0,
        'citation_references': 0,
        'validation_passed': True
    }
    
    try:
        # Find chapter references
        chapter_refs = re.findall(r'[Cc]hapter\s+(\d+)', content)
        validation['references_found'].extend([f"Chapter {ref}" for ref in chapter_refs])
        
        # Find figure/table references
        figure_refs = re.findall(r'[Ff]igure\s+(\d+)', content)
        validation['references_found'].extend([f"Figure {ref}" for ref in figure_refs])
        
        table_refs = re.findall(r'[Tt]able\s+(\d+)', content)
        validation['references_found'].extend([f"Table {ref}" for ref in table_refs])
        
        # Find citation references
        citation_patterns = [
            r'\[(\d+)\]',  # [1], [2], etc.
            r'\(([^)]+,\s*\d{4})\)',  # (Author, 2023)
        ]
        
        citations = []
        for pattern in citation_patterns:
            citations.extend(re.findall(pattern, content))
        
        validation['citation_references'] = len(citations)
        validation['references_found'].extend([f"Citation: {cite}" for cite in citations[:10]])  # Limit display
        
        # Find URLs
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, content)
        validation['external_links'] = len(urls)
        
        # Internal links (markdown style)
        internal_pattern = r'\[([^\]]+)\]\(#([^)]+)\)'
        internal_links = re.findall(internal_pattern, content)
        validation['internal_links'] = len(internal_links)
        
        # Simple validation - could be enhanced to check if references actually exist
        if validation['citation_references'] > 0 and 'bibliography' not in content.lower():
            validation['broken_references'].append("Citations found but no bibliography section detected")
            validation['validation_passed'] = False
        
    except Exception as e:
        logger.warning(f"Cross-reference validation failed: {e}")
        validation['validation_passed'] = False
    
    return validation


def estimate_reading_time(content: str, words_per_minute: int = 200) -> Dict[str, Any]:
    """Estimate reading time for the content."""
    word_count = len(content.split())
    
    total_minutes = max(1, word_count / words_per_minute)
    hours = int(total_minutes // 60)
    minutes = int(total_minutes % 60)
    
    return {
        'total_words': word_count,
        'words_per_minute': words_per_minute,
        'total_minutes': total_minutes,
        'reading_time_formatted': f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m",
        'reading_sessions': {
            '15_min_sessions': max(1, int(total_minutes / 15)),
            '30_min_sessions': max(1, int(total_minutes / 30)),
            '60_min_sessions': max(1, int(total_minutes / 60))
        }
    }