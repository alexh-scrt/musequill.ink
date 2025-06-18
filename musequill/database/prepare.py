from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4
from datetime import datetime
from musequill.database.book_db import BookStatus

def prepare_book_document(book_data:Dict) -> Dict:
    """
    Prepare a complete book document for MongoDB storage from book_data.
    Handles enum serialization and ensures all fields are captured.
    """
    book_id = book_data.get('_id', str(uuid4()))
    
    # Function to safely extract enum values
    def safe_enum_value(value):
        if hasattr(value, 'value'):
            return value.value
        return value
    
    # Function to safely extract list of enum values
    def safe_enum_list(enum_list):
        if not enum_list:
            return []
        return [safe_enum_value(item) for item in enum_list]
    
    # Prepare complete book document
    book_doc = {
        # Core identification and timestamps
        '_id': book_id,
        'created_at': book_data.get('created_at', datetime.now(timezone.utc)),
        'updated_at': book_data.get('updated_at', datetime.now(timezone.utc)),
        
        # Status and progress
        'status': book_data.get('status', BookStatus.INITIALIZING),
        'planning_status': book_data.get('planning_status', 'pending'),
        'completion_percentage': book_data.get('completion_percentage', 0.0),
        'estimated_word_count': book_data.get('estimated_word_count', 0),
        'estimated_chapters': book_data.get('estimated_chapters', 0),
        'validation_warnings': book_data.get('validation_warnings', []),
        
        # Core book information
        'title': book_data.get('title', 'Untitled'),
        'subtitle': book_data.get('subtitle'),
        'description': book_data.get('description'),
        'additional_notes': book_data.get('additional_notes'),
        
        # Genre and world building
        'genre_info': {
            'genre': book_data.get('genre_info', {}).get('genre', safe_enum_value(book_data.get('parameters', {}).get('genre', ''))),
            'sub_genre': book_data.get('genre_info', {}).get('sub_genre'),
            'is_fiction': book_data.get('genre_info', {}).get('is_fiction', True),
            'available_subgenres': book_data.get('genre_info', {}).get('available_subgenres', []),
            'world_type': book_data.get('genre_info', {}).get('world_type'),
            'magic_system': book_data.get('genre_info', {}).get('magic_system'),
            'technology_level': book_data.get('genre_info', {}).get('technology_level'),
        },
        
        # Story structure and plot
        'story_info': {
            'length': book_data.get('story_info', {}).get('length'),
            'structure': book_data.get('story_info', {}).get('structure'),
            'plot_type': book_data.get('story_info', {}).get('plot_type'),
            'pov': book_data.get('story_info', {}).get('pov'),
            'pacing': book_data.get('story_info', {}).get('pacing'),
            'conflict_types': book_data.get('story_info', {}).get('conflict_types', []),
            'complexity': book_data.get('story_info', {}).get('complexity'),
        },
        
        # Character information
        'character_info': {
            'main_character_role': book_data.get('character_info', {}).get('main_character_role'),
            'character_archetype': book_data.get('character_info', {}).get('character_archetype'),
        },
        
        # Writing style and tone
        'style_info': {
            'writing_style': book_data.get('style_info', {}).get('writing_style'),
            'tone': book_data.get('style_info', {}).get('tone'),
        },
        
        # Target audience
        'audience_info': {
            'age_group': book_data.get('audience_info', {}).get('age_group'),
            'audience_type': book_data.get('audience_info', {}).get('audience_type'),
            'reading_level': book_data.get('audience_info', {}).get('reading_level'),
        },
        
        # Publication information
        'publication_info': {
            'publication_route': book_data.get('publication_info', {}).get('publication_route'),
            'content_warnings': book_data.get('publication_info', {}).get('content_warnings', []),
        },
        
        # AI and writing process
        'process_info': {
            'ai_assistance_level': book_data.get('process_info', {}).get('ai_assistance_level'),
            'research_priority': book_data.get('process_info', {}).get('research_priority'),
            'writing_schedule': book_data.get('process_info', {}).get('writing_schedule'),
        },
        
        # Validation information
        'validation_info': {
            'genre_subgenre_valid': book_data.get('validation_info', {}).get('genre_subgenre_valid', True),
            'warnings': book_data.get('validation_info', {}).get('warnings', []),
        },
        
        # Agent and workflow information
        'agent_id': book_data.get('agent_id'),
        'last_agent_id': book_data.get('last_agent_id'),
        'planning_results': book_data.get('planning_results', {}),
        'agent_workflows': book_data.get('agent_workflows', {}),
        'status_message': book_data.get('status_message'),
        'next_steps': book_data.get('next_steps', []),
        
        # Error tracking
        'error_message': book_data.get('error_message'),
        'retry_count': book_data.get('retry_count', 0),
        'approval_status': book_data.get('approval_status', 'pending'),
        
        # Raw parameters (for backward compatibility and debugging)
        'parameters': book_data.get('parameters', {}),
        
        # Additional metadata
        'tags': book_data.get('tags', []),
        'version': book_data.get('version', 1),
        'metadata': book_data.get('metadata', {}),
    }
    
    # Remove None values to keep the document clean (optional)
    book_doc = {k: v for k, v in book_doc.items() if v is not None}
    
    # Clean nested dictionaries of None values
    for key, value in book_doc.items():
        if isinstance(value, dict):
            book_doc[key] = {k: v for k, v in value.items() if v is not None}
    
    return book_doc


# Alternative approach if you want to maintain your existing structure
# but ensure all fields are captured:

def prepare_book_document_minimal(book_data):
    """
    Minimal approach that captures all fields from book_data without restructuring.
    """
    book_id = book_data.get('_id', str(uuid4()))
    
    # Start with your existing structure
    book_doc = {
        '_id': book_id,
        'created_at': book_data.get('created_at', datetime.now(timezone.utc)),
        'updated_at': book_data.get('updated_at', datetime.now(timezone.utc)),
        'status': book_data.get('status', BookStatus.INITIALIZING),
        'title': book_data.get('title', 'Untitled'),
        'genre': book_data.get('genre_info', {}).get('genre', ''),
        'estimated_word_count': book_data.get('estimated_word_count', 0),
        'completion_percentage': book_data.get('completion_percentage', 0.0),
        
        # Complex nested data (your existing fields)
        'parameters': book_data.get('parameters', {}),
        'planning_results': book_data.get('planning_results', {}),
        'agent_workflows': book_data.get('agent_workflows', {}),
        'validation_info': book_data.get('validation_info', {}),
        
        # Additional fields (your existing)
        'tags': book_data.get('tags', []),
        'last_agent_id': book_data.get('last_agent_id'),
        'approval_status': book_data.get('approval_status', 'pending')
    }
    
    # Add ALL missing fields from book_data
    additional_fields = {
        'planning_status': book_data.get('planning_status'),
        'estimated_chapters': book_data.get('estimated_chapters'),
        'validation_warnings': book_data.get('validation_warnings', []),
        'subtitle': book_data.get('subtitle'),
        'description': book_data.get('description'),
        'additional_notes': book_data.get('additional_notes'),
        'genre_info': book_data.get('genre_info', {}),
        'story_info': book_data.get('story_info', {}),
        'character_info': book_data.get('character_info', {}),
        'style_info': book_data.get('style_info', {}),
        'audience_info': book_data.get('audience_info', {}),
        'publication_info': book_data.get('publication_info', {}),
        'process_info': book_data.get('process_info', {}),
        'agent_id': book_data.get('agent_id'),
        'status_message': book_data.get('status_message'),
        'next_steps': book_data.get('next_steps', []),
        'error_message': book_data.get('error_message'),
        'retry_count': book_data.get('retry_count', 0),
    }
    
    # Add non-None fields
    for key, value in additional_fields.items():
        if value is not None:
            book_doc[key] = value
    
    return book_doc