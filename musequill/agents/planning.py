"""
Enhanced Planning Agent for MuseQuill - Updated for book creation endpoint integration.
"""

import json
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime
import structlog

from musequill.core.base.agent import BaseAgent, AgentConfig, AgentType, AgentState
from musequill.core.openai_client import OpenAIClient
from musequill.models.planning import (
    PlanningRequest, PlanningResult, StoryOutline, ChapterPlan,
    ResearchRequirements, ResearchRequirement, ResearchPriority,
    PlotStructure, StoryStructure, GenreType
)
from musequill.prompts.planning import (
    PLANNING_SYSTEM_PROMPT, STRUCTURE_ANALYSIS_PROMPT,
    OUTLINE_CREATION_PROMPT, CHAPTER_STRUCTURE_PROMPT,
    RESEARCH_ANALYSIS_PROMPT, STORY_VALIDATION_PROMPT
)

logger = structlog.get_logger(__name__)


class PlanningAgent(BaseAgent):
    """Agent responsible for story planning and structure creation."""
    
    def __init__(self, openai_client: OpenAIClient, **kwargs):
        config = AgentConfig(
            agent_type=AgentType.PLANNING,
            name="Planning Agent",
            description="Expert story architect for creating comprehensive story outlines",
            system_prompt=PLANNING_SYSTEM_PROMPT,
            temperature=0.7,  # Balanced creativity and consistency
            max_tokens=4000,
            **kwargs
        )
        super().__init__(config, openai_client, **kwargs)
        self._planning_context = {}
        
    async def process(self, input_data: Any, **kwargs) -> Any:
        """Main processing method for planning requests."""
        if isinstance(input_data, dict):
            request = PlanningRequest(**input_data)
        elif isinstance(input_data, PlanningRequest):
            request = input_data
        else:
            request = PlanningRequest(description=str(input_data))
        
        return await self.create_complete_plan(request)
    
    async def create_book_plan_from_request(self, book_creation_request) -> PlanningResult:
        """Create a book plan from BookCreationRequest (from API endpoint)."""
        try:
            logger.info(
                "Creating book plan from API request",
                title=book_creation_request.title,
                genre=book_creation_request.genre.value,
                length=book_creation_request.length.value
            )
            
            # Convert BookCreationRequest to PlanningRequest
            planning_request = self._convert_book_request_to_planning(book_creation_request)
            
            # Create the complete plan
            result = await self.create_complete_plan(planning_request)
            
            # Update result with additional metadata from the original request
            result.book_metadata = {
                "original_title": book_creation_request.title,
                "subtitle": book_creation_request.subtitle,
                "genre": book_creation_request.genre.value,
                "sub_genre": book_creation_request.sub_genre.value if book_creation_request.sub_genre else None,
                "length": book_creation_request.length.value,
                "structure": book_creation_request.structure.value,
                "pov": book_creation_request.pov.value,
                "age_group": book_creation_request.age_group.value,
                "ai_assistance_level": book_creation_request.ai_assistance_level.value,
                "additional_notes": book_creation_request.additional_notes
            }
            
            logger.info(
                "Book plan created successfully",
                story_id=result.outline.story_id,
                chapters=len(result.chapters),
                research_requirements=len(result.research_requirements.requirements)
            )
            
            return result
            
        except Exception as e:
            logger.error("Failed to create book plan from request", error=str(e))
            raise
    
    def _convert_book_request_to_planning(self, book_request) -> PlanningRequest:
        """Convert BookCreationRequest to PlanningRequest."""
        
        # Build comprehensive description
        description_parts = [
            f"Title: {book_request.title}",
        ]
        
        if book_request.subtitle:
            description_parts.append(f"Subtitle: {book_request.subtitle}")
            
        if book_request.description:
            description_parts.append(f"Description: {book_request.description}")
        
        description_parts.extend([
            f"Genre: {book_request.genre.value}",
            f"Length: {book_request.length.value}",
            f"Structure: {book_request.structure.value}",
            f"Point of View: {book_request.pov.value}",
            f"Target Age Group: {book_request.age_group.value}",
            f"Writing Style: {book_request.writing_style.value}",
            f"Tone: {book_request.tone.value}"
        ])
        
        if book_request.conflict_types:
            conflicts = [ct.value for ct in book_request.conflict_types]
            description_parts.append(f"Conflict Types: {', '.join(conflicts)}")
        
        if book_request.content_warnings:
            warnings = [cw.value for cw in book_request.content_warnings]
            description_parts.append(f"Content Warnings: {', '.join(warnings)}")
        
        if book_request.additional_notes:
            description_parts.append(f"Additional Notes: {book_request.additional_notes}")
        
        description = "\n".join(description_parts)
        
        # Map genre to GenreType
        genre_mapping = {
            "FICTION": GenreType.FICTION,
            "FANTASY": GenreType.FANTASY,
            "SCIENCE_FICTION": GenreType.SCIENCE_FICTION,
            "MYSTERY": GenreType.MYSTERY,
            "ROMANCE": GenreType.ROMANCE,
            "THRILLER": GenreType.THRILLER,
            "HORROR": GenreType.HORROR,
            "NON_FICTION": GenreType.NON_FICTION,
            "BUSINESS": GenreType.BUSINESS,
            "SELF_HELP": GenreType.SELF_HELP,
            "BIOGRAPHY": GenreType.BIOGRAPHY,
            "HISTORY": GenreType.HISTORY,
            "SCIENCE": GenreType.SCIENCE,
            "TECHNOLOGY": GenreType.TECHNOLOGY,
            "HEALTH": GenreType.HEALTH,
            "TRAVEL": GenreType.TRAVEL,
            "COOKING": GenreType.COOKING,
            "CHILDREN": GenreType.CHILDREN,
            "YOUNG_ADULT": GenreType.YOUNG_ADULT,
            "POETRY": GenreType.POETRY,
            "DRAMA": GenreType.DRAMA,
            "ACADEMIC": GenreType.ACADEMIC,
        }
        
        genre = genre_mapping.get(book_request.genre.name, GenreType.OTHER)
        
        # Calculate target length based on BookLength enum
        length_mapping = {
            "FLASH_FICTION": 500,
            "SHORT_STORY": 4000,
            "NOVELETTE": 12500,
            "NOVELLA": 30000,
            "SHORT_NOVEL": 50000,
            "STANDARD_NOVEL": 75000,
            "LONG_NOVEL": 105000,
            "EPIC_NOVEL": 150000,
            "ARTICLE": 1250,
            "ESSAY": 3000,
            "GUIDE": 10000,
            "MANUAL": 32500,
            "COMPREHENSIVE_BOOK": 100000
        }
        
        target_length = length_mapping.get(book_request.length.name, 75000)
        
        # Build special requirements
        special_requirements = []
        
        if hasattr(book_request, 'character_archetype') and book_request.character_archetype:
            special_requirements.append(f"Character archetype: {book_request.character_archetype.value}")
            
        if hasattr(book_request, 'world_type') and book_request.world_type:
            special_requirements.append(f"World type: {book_request.world_type.value}")
            
        if hasattr(book_request, 'magic_system') and book_request.magic_system:
            special_requirements.append(f"Magic system: {book_request.magic_system.value}")
            
        if hasattr(book_request, 'technology_level') and book_request.technology_level:
            special_requirements.append(f"Technology level: {book_request.technology_level.value}")
        
        return PlanningRequest(
            description=description,
            genre=genre,
            target_length=target_length,
            special_requirements=special_requirements
        )
    
    async def create_complete_plan(self, request: PlanningRequest) -> PlanningResult:
        """Create a complete story plan including outline, chapters, and research."""
        try:
            logger.info("Starting complete story planning", 
                       description_length=len(request.description),
                       genre=request.genre)
            
            # Step 1: Create story outline
            outline = await self.create_story_outline(
                request.description, 
                request.genre or GenreType.OTHER,
                request
            )
            
            # Step 2: Plan chapter structure
            chapters = await self.plan_chapter_structure(outline)
            
            # Step 3: Analyze research requirements
            research_requirements = await self.analyze_story_requirements(outline, chapters)
            
            # Step 4: Validate and improve the plan
            result = await self.validate_plan(outline, chapters, research_requirements)
            
            logger.info("Story planning completed successfully",
                       chapters_planned=len(chapters),
                       research_requirements=len(research_requirements.requirements))
            
            return result
            
        except Exception as e:
            logger.error("Story planning failed", error=str(e))
            raise
    
    async def create_story_outline(
        self, 
        description: str, 
        genre: GenreType,
        request: Optional[PlanningRequest] = None
    ) -> StoryOutline:
        """Create a comprehensive story outline from a description."""
        
        logger.info("Creating story outline", genre=genre.value)
        
        # Prepare context for the outline creation
        context = {
            "description": description,
            "genre": genre.value,
            "target_length": getattr(request, 'target_length', 80000),
            "special_requirements": getattr(request, 'special_requirements', [])
        }
        
        # First, analyze optimal story structure
        structure_prompt = STRUCTURE_ANALYSIS_PROMPT.format(**context)
        structure_analysis = await self._execute_llm_request([
            {"role": "user", "content": structure_prompt}
        ])
        
        # Create the main outline
        outline_prompt = OUTLINE_CREATION_PROMPT.format(**context)
        outline_response = await self._execute_llm_request([
            {"role": "user", "content": outline_prompt},
            {"role": "assistant", "content": structure_analysis},
            {"role": "user", "content": "Now create the complete story outline based on this structure analysis."}
        ])
        
        # Parse the response into a structured outline
        outline = await self._parse_outline_response(outline_response, description, genre, request)
        
        # Store context for this outline
        self.set_context(f"outline_{outline.story_id}", outline)
        
        return outline
    
    async def plan_chapter_structure(self, outline: StoryOutline) -> List[ChapterPlan]:
        """Create detailed chapter plans based on the story outline."""
        
        logger.info("Planning chapter structure", 
                   estimated_chapters=outline.estimated_chapters)
        
        # Calculate chapter planning parameters
        target_chapter_length = outline.estimated_word_count // outline.estimated_chapters
        
        context = {
            "outline_summary": await self._create_outline_summary(outline),
            "target_chapter_length": target_chapter_length,
            "total_chapters": outline.estimated_chapters
        }
        
        # Create chapter structure
        chapter_prompt = CHAPTER_STRUCTURE_PROMPT.format(**context)
        chapter_response = await self._execute_llm_request([
            {"role": "user", "content": chapter_prompt}
        ])
        
        # Parse chapters from response
        chapters = await self._parse_chapters_response(chapter_response, outline)
        
        # Store context
        self.set_context(f"chapters_{outline.story_id}", chapters)
        
        return chapters
    
    async def analyze_story_requirements(
        self, 
        outline: StoryOutline, 
        chapters: Optional[List[ChapterPlan]] = None
    ) -> ResearchRequirements:
        """Analyze what research is needed for the story."""
        
        logger.info("Analyzing research requirements", story_id=outline.story_id)
        
        story_summary = await self._create_comprehensive_summary(outline, chapters)
        
        context = {
            "story_summary": story_summary
        }
        
        research_prompt = RESEARCH_ANALYSIS_PROMPT.format(**context)
        research_response = await self._execute_llm_request([
            {"role": "user", "content": research_prompt}
        ])
        
        # Parse research requirements
        research_requirements = await self._parse_research_response(
            research_response, outline.story_id
        )
        
        # Store context
        self.set_context(f"research_{outline.story_id}", research_requirements)
        
        return research_requirements
    
    async def validate_plan(
        self,
        outline: StoryOutline,
        chapters: List[ChapterPlan],
        research_requirements: ResearchRequirements
    ) -> PlanningResult:
        """Validate the complete plan and suggest improvements."""
        
        logger.info("Validating story plan", story_id=outline.story_id)
        
        complete_plan = {
            "outline": outline.dict(),
            "chapters": [chapter.dict() for chapter in chapters],
            "research": research_requirements.dict()
        }
        
        context = {
            "complete_plan": json.dumps(complete_plan, indent=2, default=str)
        }
        
        validation_prompt = STORY_VALIDATION_PROMPT.format(**context)
        validation_response = await self._execute_llm_request([
            {"role": "user", "content": validation_prompt}
        ])
        
        # Parse validation results
        result = await self._parse_validation_response(
            validation_response, outline, chapters, research_requirements
        )
        
        return result
    
    async def get_planning_status(self) -> Dict[str, Any]:
        """Get detailed planning status from this agent."""
        try:
            base_status = {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type.value,
                "state": self.state.value,
                "healthy": self.is_healthy(),
                "created_at": getattr(self, 'created_at', None),
            }
            
            # Add planning-specific status
            planning_context = getattr(self, '_planning_context', {})
            
            active_outlines = [
                key for key in self._context.keys() 
                if key.startswith('outline_')
            ]
            
            active_chapters = [
                key for key in self._context.keys() 
                if key.startswith('chapters_')
            ]
            
            active_research = [
                key for key in self._context.keys() 
                if key.startswith('research_')
            ]
            
            planning_status = {
                **base_status,
                "active_outlines": len(active_outlines),
                "active_chapter_plans": len(active_chapters),
                "active_research_plans": len(active_research),
                "total_context_items": len(self._context),
                "planning_context": planning_context
            }
            
            return planning_status
            
        except Exception as e:
            logger.warning("Failed to get planning status", error=str(e))
            return {
                "agent_id": self.agent_id,
                "error": str(e),
                "state": "error"
            }
    
    # ... [Rest of the methods from the original PlanningAgent]
    # Including all the parsing methods: _parse_outline_response, _parse_chapters_response, etc.
    
    async def _parse_outline_response(
        self, 
        response: str, 
        description: str, 
        genre: GenreType,
        request: Optional[PlanningRequest]
    ) -> StoryOutline:
        """Parse LLM response into a StoryOutline object."""
        
        try:
            # Try to extract JSON structure from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                outline_data = json.loads(json_match.group())
            else:
                # If no JSON, create structure from text analysis
                outline_data = await self._extract_outline_from_text(response)
            
            # Create the outline with required fields
            story_id = str(uuid.uuid4())
            
            outline = StoryOutline(
                story_id=story_id,
                genre=genre,
                premise=description,
                synopsis=outline_data.get('synopsis', response[:500] + '...'),
                structure=PlotStructure(
                    structure_type=StoryStructure.THREE_ACT,
                    major_plot_points=[],
                    themes=outline_data.get('themes', [])
                ),
                setting_overview=outline_data.get('setting_overview', 'To be developed'),
                central_theme=outline_data.get('central_theme', 'To be identified'),
                tone=outline_data.get('tone', 'To be determined'),
                **{k: v for k, v in outline_data.items() if k in StoryOutline.__fields__}
            )
            
            return outline
            
        except Exception as e:
            logger.warning("Failed to parse structured outline, creating basic outline", error=str(e))
            
            # Create a basic outline from the response
            story_id = str(uuid.uuid4())
            return StoryOutline(
                story_id=story_id,
                genre=genre,
                premise=description,
                synopsis=response[:1000] if len(response) > 1000 else response,
                structure=PlotStructure(
                    structure_type=StoryStructure.THREE_ACT,
                    themes=["To be developed"]
                ),
                setting_overview="To be developed based on story requirements",
                central_theme="To be identified through story development",
                tone="To be determined based on genre and content"
            )
    
    async def _parse_chapters_response(
        self, 
        response: str, 
        outline: StoryOutline
    ) -> List[ChapterPlan]:
        """Parse LLM response into ChapterPlan objects."""
        
        try:
            # Try to extract structured data
            import re
            
            # Look for chapter patterns in the response
            chapter_pattern = r'Chapter (\d+):(.*?)(?=Chapter \d+:|$)'
            chapters = []
            
            matches = re.findall(chapter_pattern, response, re.DOTALL | re.IGNORECASE)
            
            if matches:
                for chapter_num, chapter_content in matches:
                    chapter = await self._create_chapter_from_text(
                        int(chapter_num), chapter_content, outline
                    )
                    chapters.append(chapter)
            else:
                # If no clear structure, create basic chapters
                chapters = await self._create_basic_chapters(outline)
            
            return chapters
            
        except Exception as e:
            logger.warning("Failed to parse chapters, creating basic structure", error=str(e))
            return await self._create_basic_chapters(outline)
    
    async def _parse_research_response(
        self, 
        response: str, 
        story_id: str
    ) -> ResearchRequirements:
        """Parse LLM response into ResearchRequirements."""
        
        try:
            # Extract research topics from response
            requirements = []
            
            # Look for research patterns
            import re
            research_pattern = r'(?:Research|Topic|Requirement)[:\s]*([^\n]+)'
            matches = re.findall(research_pattern, response, re.IGNORECASE)
            
            for i, topic in enumerate(matches):
                if topic.strip():
                    requirement = ResearchRequirement(
                        topic=topic.strip(),
                        priority=ResearchPriority.MEDIUM,
                        description=f"Research needed for: {topic.strip()}",
                        reason="Required for story authenticity and accuracy",
                        keywords=[word.strip() for word in topic.split() if len(word) > 3]
                    )
                    requirements.append(requirement)
            
            # If no structured requirements found, create from general analysis
            if not requirements:
                requirements = await self._create_basic_research_requirements(response)
            
            return ResearchRequirements(
                story_id=story_id,
                requirements=requirements,
                estimated_research_time=len(requirements) * 2,  # 2 hours per requirement
                research_categories=self._categorize_research(requirements)
            )
            
        except Exception as e:
            logger.warning("Failed to parse research requirements", error=str(e))
            return ResearchRequirements(
                story_id=story_id,
                requirements=[],
                estimated_research_time=0,
                research_categories={}
            )
    
    async def _parse_validation_response(
        self,
        response: str,
        outline: StoryOutline,
        chapters: List[ChapterPlan],
        research_requirements: ResearchRequirements
    ) -> PlanningResult:
        """Parse validation response into PlanningResult."""
        
        try:
            # Extract validation insights
            import re
            
            # Look for issues and suggestions
            issues_pattern = r'(?:Issue|Problem|Concern)[:\s]*([^\n]+)'
            suggestions_pattern = r'(?:Suggestion|Improvement|Recommendation)[:\s]*([^\n]+)'
            
            potential_issues = re.findall(issues_pattern, response, re.IGNORECASE)
            suggestions = re.findall(suggestions_pattern, response, re.IGNORECASE)
            
            # Create planning result
            result = PlanningResult(
                outline=outline,
                chapters=chapters,
                research_requirements=research_requirements,
                planning_confidence=0.8,  # Default confidence
                potential_plot_holes=[issue.strip() for issue in potential_issues],
                next_steps=[
                    "Review and refine character motivations",
                    "Conduct identified research",
                    "Begin detailed chapter writing",
                    "Develop character backstories"
                ]
            )
            
            # Extract confidence if mentioned
            confidence_pattern = r'confidence[:\s]*(\d+(?:\.\d+)?)'
            confidence_match = re.search(confidence_pattern, response, re.IGNORECASE)
            if confidence_match:
                try:
                    confidence = float(confidence_match.group(1))
                    if confidence <= 1.0:
                        result.planning_confidence = confidence
                    elif confidence <= 100:
                        result.planning_confidence = confidence / 100
                except ValueError:
                    pass
            
            return result
            
        except Exception as e:
            logger.warning("Failed to parse validation, creating basic result", error=str(e))
            return PlanningResult(
                outline=outline,
                chapters=chapters,
                research_requirements=research_requirements
            )
    
    async def _create_outline_summary(self, outline: StoryOutline) -> str:
        """Create a concise summary of the story outline."""
        
        summary = f"""
Story: {outline.title or 'Untitled'}
Genre: {outline.genre.value}
Premise: {outline.premise}
Synopsis: {outline.synopsis}
Setting: {outline.setting_overview}
Theme: {outline.central_theme}
Tone: {outline.tone}
Structure: {outline.structure.structure_type.value}
Estimated Length: {outline.estimated_word_count} words in {outline.estimated_chapters} chapters

Main Characters: {', '.join([char.name for char in outline.main_characters])}
Major Themes: {', '.join(outline.secondary_themes)}
"""
        return summary.strip()
    
    async def _create_comprehensive_summary(
        self, 
        outline: StoryOutline, 
        chapters: Optional[List[ChapterPlan]] = None
    ) -> str:
        """Create a comprehensive summary for research analysis."""
        
        summary = await self._create_outline_summary(outline)
        
        if chapters:
            summary += "\n\nChapter Overview:\n"
            for chapter in chapters[:5]:  # Include first 5 chapters for context
                summary += f"Chapter {chapter.chapter_number}: {chapter.summary}\n"
                summary += f"Settings: {', '.join(chapter.settings)}\n"
                summary += f"Characters: {', '.join(chapter.characters_present)}\n\n"
        
        return summary
    
    async def _create_chapter_from_text(
        self, 
        chapter_num: int, 
        content: str, 
        outline: StoryOutline
    ) -> ChapterPlan:
        """Create a ChapterPlan from text content."""
        
        # Extract key information from the content
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        # Try to find summary
        summary = "Chapter summary to be developed"
        for line in lines:
            if len(line) > 50 and not line.startswith(('-', '*', '•')):
                summary = line
                break
        
        return ChapterPlan(
            chapter_number=chapter_num,
            summary=summary,
            purpose=f"Advance story according to chapter {chapter_num} requirements",
            plot_advancement=f"Chapter {chapter_num} plot development",
            word_count_target=outline.estimated_word_count // outline.estimated_chapters,
            story_percentage=chapter_num / outline.estimated_chapters
        )
    
    async def _create_basic_chapters(self, outline: StoryOutline) -> List[ChapterPlan]:
        """Create basic chapter structure when parsing fails."""
        
        chapters = []
        chapters_per_act = outline.estimated_chapters // 3
        
        for i in range(1, outline.estimated_chapters + 1):
            # Determine act
            if i <= chapters_per_act:
                act = 1
                purpose = "Establish characters, setting, and initial conflict"
            elif i <= chapters_per_act * 2:
                act = 2
                purpose = "Develop conflict and character relationships"
            else:
                act = 3
                purpose = "Resolve conflicts and conclude character arcs"
            
            chapter = ChapterPlan(
                chapter_number=i,
                summary=f"Chapter {i} - {purpose}",
                purpose=purpose,
                plot_advancement=f"Advance {outline.structure.structure_type.value} structure",
                word_count_target=outline.estimated_word_count // outline.estimated_chapters,
                act=act,
                story_percentage=i / outline.estimated_chapters
            )
            chapters.append(chapter)
        
        return chapters
    
    async def _create_basic_research_requirements(
        self, 
        response: str
    ) -> List[ResearchRequirement]:
        """Create basic research requirements from response text."""
        
        # Extract potential research topics from response
        words = response.lower().split()
        potential_topics = []
        
        # Look for research-indicating keywords
        research_keywords = [
            'historical', 'technical', 'scientific', 'cultural', 'professional',
            'location', 'geography', 'language', 'customs', 'traditions',
            'weapons', 'technology', 'medicine', 'law', 'military'
        ]
        
        for keyword in research_keywords:
            if keyword in words:
                potential_topics.append(keyword.title() + " research")
        
        # Create requirements
        requirements = []
        for i, topic in enumerate(potential_topics[:10]):  # Limit to 10
            requirement = ResearchRequirement(
                topic=topic,
                priority=ResearchPriority.MEDIUM,
                description=f"Research {topic.lower()} for story accuracy",
                reason="Ensure authenticity and reader engagement",
                keywords=[topic.lower().replace(' research', '')]
            )
            requirements.append(requirement)
        
        return requirements
    
    def _categorize_research(self, requirements: List[ResearchRequirement]) -> Dict[str, int]:
        """Categorize research requirements by type."""
        
        categories = {
            'historical': 0,
            'technical': 0,
            'cultural': 0,
            'geographical': 0,
            'professional': 0,
            'other': 0
        }
        
        for req in requirements:
            topic_lower = req.topic.lower()
            
            if any(word in topic_lower for word in ['history', 'historical', 'period', 'era']):
                categories['historical'] += 1
            elif any(word in topic_lower for word in ['technical', 'technology', 'science', 'medical']):
                categories['technical'] += 1
            elif any(word in topic_lower for word in ['culture', 'custom', 'tradition', 'language']):
                categories['cultural'] += 1
            elif any(word in topic_lower for word in ['location', 'geography', 'place', 'city']):
                categories['geographical'] += 1
            elif any(word in topic_lower for word in ['job', 'profession', 'career', 'work']):
                categories['professional'] += 1
            else:
                categories['other'] += 1
        
        return categories
    
    async def _extract_outline_from_text(self, text: str) -> Dict[str, Any]:
        """Extract outline data from unstructured text."""
        
        outline_data = {}
        
        # Extract title
        import re
        title_match = re.search(r'(?:title|story)[:\s]*([^\n]+)', text, re.IGNORECASE)
        if title_match:
            outline_data['title'] = title_match.group(1).strip()
        
        # Extract synopsis
        synopsis_match = re.search(r'(?:synopsis|summary)[:\s]*([^\n]+(?:\n[^\n]+)*)', text, re.IGNORECASE)
        if synopsis_match:
            outline_data['synopsis'] = synopsis_match.group(1).strip()
        
        # Extract themes
        themes_match = re.search(r'(?:themes?)[:\s]*([^\n]+)', text, re.IGNORECASE)
        if themes_match:
            themes_text = themes_match.group(1)
            outline_data['themes'] = [theme.strip() for theme in themes_text.split(',')]
        
        # Extract setting
        setting_match = re.search(r'(?:setting|location)[:\s]*([^\n]+)', text, re.IGNORECASE)
        if setting_match:
            outline_data['setting_overview'] = setting_match.group(1).strip()
        
        # Extract tone
        tone_match = re.search(r'(?:tone|mood)[:\s]*([^\n]+)', text, re.IGNORECASE)
        if tone_match:
            outline_data['tone'] = tone_match.group(1).strip()
        
        return outline_data
    
    async def initialize(self) -> None:
        """Initialize the planning agent."""
        self.set_context("initialized", True)
        self.set_context("agent_type", "planning")
        self._planning_context = {
            "initialized_at": datetime.now().isoformat(),
            "total_plans_created": 0,
            "active_outlines": 0
        }
        logger.info("Planning agent initialized")
    
    async def cleanup(self) -> None:
        """Cleanup planning agent resources."""
        self.clear_context()
        self._planning_context = {}
        logger.info("Planning agent cleaned up")