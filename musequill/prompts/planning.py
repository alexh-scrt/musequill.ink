"""
Prompts for the Planning Agent.
"""

PLANNING_SYSTEM_PROMPT = """You are an expert story architect and planning agent for MuseQuill, specializing in creating comprehensive, publication-ready book outlines and structures.

Your core expertise includes:
- Story structure analysis and optimization
- Character development and arc planning
- Plot coherence and pacing
- Theme identification and integration
- Research requirement analysis
- Genre-specific conventions and expectations

When creating story plans, you should:

1. **Structure Analysis**: Analyze the optimal story structure based on genre, length, and content
2. **Character Development**: Create compelling, three-dimensional characters with clear motivations and arcs
3. **Plot Coherence**: Ensure logical progression, proper setup/payoff, and engaging pacing
4. **Theme Integration**: Weave themes naturally throughout the narrative structure
5. **Research Identification**: Identify areas requiring factual research for authenticity
6. **Quality Assessment**: Evaluate potential weaknesses and suggest improvements

Always provide:
- Clear, actionable chapter summaries
- Character descriptions with roles and motivations
- Research requirements with priorities
- Confidence assessments with reasoning
- Next steps for the writing process

Be thorough, creative, and focused on creating stories that engage readers while maintaining professional publishing standards.
"""

STRUCTURE_ANALYSIS_PROMPT = """Analyze the optimal story structure for this book project:

**Project Details:**
Description: {description}
Genre: {genre}
Target Length: {target_length} words
Special Requirements: {special_requirements}

**Analysis Required:**

1. **Genre Conventions**: What structure best suits this genre and target audience?

2. **Optimal Act Structure**: Based on the content, should this use:
   - Three-act structure (setup, confrontation, resolution)
   - Five-act structure (exposition, rising action, climax, falling action, denouement)
   - Hero's journey (departure, initiation, return)
   - Other specialized structure?

3. **Pacing Considerations**: Given the target length, how should tension and reveals be distributed?

4. **Key Structural Elements**: What are the essential plot points that must be included?

5. **Character Arc Integration**: How many character arcs can this length support effectively?

Provide a detailed structural analysis with reasoning for your recommendations.
"""

OUTLINE_CREATION_PROMPT = """Create a comprehensive story outline based on the following:

**Project Specifications:**
Description: {description}
Genre: {genre}
Target Length: {target_length} words
Special Requirements: {special_requirements}

**Required Outline Elements:**

1. **Core Story Elements:**
   - Working title (suggest 3-5 options)
   - One-sentence premise
   - 2-3 paragraph synopsis
   - Central theme and 2-3 secondary themes
   - Tone and mood

2. **Setting & World:**
   - Primary setting(s)
   - Time period
   - World-building requirements
   - Atmosphere and mood

3. **Characters:**
   - Protagonist with clear goal, motivation, and flaw
   - 2-3 key supporting characters
   - Antagonist/opposing force
   - Character relationships and dynamics

4. **Plot Structure:**
   - Major plot points (inciting incident, midpoint, climax, resolution)
   - Conflict types (internal, interpersonal, societal, environmental)
   - Subplot identification
   - Pacing and tension progression

5. **Technical Specifications:**
   - Estimated chapter count
   - Target words per chapter
   - POV and narrative voice
   - Content warnings if applicable

Create a detailed, professionally structured outline that could guide the entire writing process.
"""

CHAPTER_STRUCTURE_PROMPT = """Create detailed chapter plans based on this story outline:

**Story Summary:**
{outline_summary}

**Chapter Planning Parameters:**
- Target chapters: {total_chapters}
- Words per chapter: {target_chapter_length}

**For Each Chapter, Provide:**

1. **Chapter Basics:**
   - Chapter number and working title
   - 2-3 sentence summary
   - Primary purpose in overall story
   - Act/section within story structure

2. **Content Details:**
   - Key scenes (2-4 major scenes per chapter)
   - Characters present
   - Settings/locations
   - Conflict or tension focus

3. **Story Progression:**
   - Plot advancement
   - Character development
   - Theme exploration
   - Emotional arc progression

4. **Chapter Flow:**
   - Opening hook
   - Pacing notes
   - Cliffhanger/transition to next chapter
   - Connection to previous chapter

5. **Writing Considerations:**
   - Estimated difficulty level
   - Key research needs
   - Potential challenges
   - Success metrics

Create a comprehensive chapter-by-chapter breakdown that provides clear guidance for writing each section while maintaining overall story coherence.
"""

RESEARCH_ANALYSIS_PROMPT = """Analyze research requirements for this story project:

**Story Summary:**
{story_summary}

**Research Analysis Required:**

1. **Factual Accuracy Needs:**
   - Historical details requiring verification
   - Technical/scientific information
   - Professional/occupational details
   - Geographic/cultural elements

2. **Authenticity Requirements:**
   - Character backgrounds and experiences
   - Settings and locations
   - Time period specifics
   - Cultural practices and customs

3. **Specialized Knowledge:**
   - Industry-specific terminology
   - Process and procedure accuracy
   - Legal or regulatory details
   - Technical specifications

4. **Research Prioritization:**
   - Critical (story-breaking if wrong)
   - High (affects credibility)
   - Medium (enhances authenticity)
   - Low (nice-to-have details)

**For Each Research Topic:**
- Specific topic/area
- Why it's needed (story relevance)
- Priority level with justification
- Suggested research approach
- Estimated time requirement
- Key questions to answer

Provide a comprehensive research plan that ensures story authenticity while being realistic about time and effort required.
"""

STORY_VALIDATION_PROMPT = """Perform comprehensive validation of this story plan:

**Complete Story Plan:**
{complete_plan}

**Validation Areas:**

1. **Plot Coherence Analysis:**
   - Logical story progression
   - Setup and payoff consistency
   - Plot hole identification
   - Pacing assessment
   - Conflict escalation

2. **Character Consistency:**
   - Character motivation clarity
   - Arc completion potential
   - Relationship dynamics
   - Growth opportunity assessment
   - Backstory integration

3. **Structure Evaluation:**
   - Act balance and timing
   - Chapter flow and transitions
   - Tension curve analysis
   - Climax positioning
   - Resolution satisfying-ness

4. **Theme Integration:**
   - Theme emergence and development
   - Thematic coherence
   - Message clarity without preaching
   - Symbolic element integration

5. **Genre Expectations:**
   - Genre convention adherence
   - Reader expectation management
   - Market positioning
   - Unique element identification

6. **Practical Considerations:**
   - Achievable scope for target length
   - Research feasibility
   - Writing complexity assessment
   - Timeline realism

**Provide:**
- Overall confidence score (0.0-1.0)
- Critical issues requiring attention
- Suggested improvements
- Story strengths to leverage
- Potential publication challenges
- Recommended next steps

Give honest, constructive feedback that helps create the best possible story while being encouraging about the project's potential.
"""

QUICK_PLANNING_PROMPT = """Create a rapid story outline for immediate development:

**Project:** {description}
**Genre:** {genre}
**Target Length:** {target_length} words

**Quick Planning Requirements:**
- Working title
- Core premise (1 sentence)
- Brief synopsis (1 paragraph)
- Main character basics
- 3-act structure outline
- 8-12 chapter breakdown
- Primary research needs

Focus on speed while maintaining quality fundamentals. This should provide enough structure to begin writing immediately.
"""

REFINEMENT_PROMPT = """Refine and improve this existing story plan:

**Current Plan:** {current_plan}
**Specific Issues:** {issues_to_address}
**Improvement Goals:** {improvement_goals}

**Refinement Focus:**
1. Address identified weaknesses
2. Strengthen story structure
3. Improve character development
4. Enhance plot coherence
5. Optimize pacing

Provide specific, actionable improvements that elevate the story plan without losing its core strengths.
"""

GENRE_SPECIFIC_PROMPTS = {
    "fantasy": """
Fantasy-Specific Considerations:
- Magic system rules and limitations
- World-building depth and consistency
- Mythological elements integration
- Character power progression
- Political/social structures
- Geography and cultures
""",
    
    "science_fiction": """
Science Fiction-Specific Considerations:
- Scientific plausibility
- Technology integration and impact
- Future world implications
- Character adaptation to tech
- Ethical/philosophical questions
- Scientific accuracy requirements
""",
    
    "mystery": """
Mystery-Specific Considerations:
- Clue placement and timing
- Red herring integration
- Detective/protagonist capabilities
- Solution fairness and logic
- Pacing of reveals
- Reader engagement strategy
""",
    
    "romance": """
Romance-Specific Considerations:
- Relationship development arc
- Emotional authenticity
- Conflict that serves love story
- Chemistry establishment
- Satisfying resolution
- Genre expectation fulfillment
""",
    
    "non_fiction": """
Non-Fiction-Specific Considerations:
- Factual accuracy verification
- Source credibility and citation
- Logical argument structure
- Reader value proposition
- Practical application
- Expert authority establishment
"""
}

# Helper function to get genre-specific prompts
def get_genre_prompt(genre: str) -> str:
    """Get genre-specific planning considerations."""
    return GENRE_SPECIFIC_PROMPTS.get(genre.lower(), "")

# Prompt templates for different planning modes
PLANNING_MODE_PROMPTS = {
    "detailed": {
        "system": PLANNING_SYSTEM_PROMPT,
        "structure": STRUCTURE_ANALYSIS_PROMPT,
        "outline": OUTLINE_CREATION_PROMPT,
        "chapters": CHAPTER_STRUCTURE_PROMPT,
        "research": RESEARCH_ANALYSIS_PROMPT,
        "validation": STORY_VALIDATION_PROMPT
    },
    
    "quick": {
        "system": PLANNING_SYSTEM_PROMPT,
        "planning": QUICK_PLANNING_PROMPT
    },
    
    "refinement": {
        "system": PLANNING_SYSTEM_PROMPT,
        "refinement": REFINEMENT_PROMPT
    }
}