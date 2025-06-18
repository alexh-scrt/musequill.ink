"""
Complete Enumeration Presets for MuseQuill.ink
Comprehensive preset values for user interface selections and book generation.
"""

from enum import Enum
from typing import Dict, List, Tuple


# ============================================================================
# Genre and Category Enumerations
# ============================================================================

class GenreType(str, Enum):
    """Supported genre types."""
    # Fiction Genres
    FANTASY = "fantasy"
    SCIENCE_FICTION = "science_fiction"
    MYSTERY = "mystery"
    ROMANCE = "romance"
    THRILLER = "thriller"
    HORROR = "horror"
    HISTORICAL_FICTION = "historical_fiction"
    LITERARY_FICTION = "literary_fiction"
    YOUNG_ADULT = "young_adult"
    ADVENTURE = "adventure"
    DRAMA = "drama"
    COMEDY = "comedy"
    WESTERN = "western"
    CRIME = "crime"
    DYSTOPIAN = "dystopian"
    URBAN_FANTASY = "urban_fantasy"
    PARANORMAL = "paranormal"
    CONTEMPORARY = "contemporary"
    
    # Non-Fiction Genres
    BIOGRAPHY = "biography"
    MEMOIR = "memoir"
    SELF_HELP = "self_help"
    BUSINESS = "business"
    HEALTH = "health"
    TRAVEL = "travel"
    COOKING = "cooking"
    HISTORY = "history"
    SCIENCE = "science"
    TECHNOLOGY = "technology"
    PHILOSOPHY = "philosophy"
    RELIGION = "religion"
    POLITICS = "politics"
    EDUCATION = "education"
    REFERENCE = "reference"
    TRUE_CRIME = "true_crime"
    
    # Specialized
    CHILDREN = "children"
    PICTURE_BOOK = "picture_book"
    POETRY = "poetry"
    SCREENPLAY = "screenplay"
    TEXTBOOK = "textbook"
    MANUAL = "manual"
    OTHER = "other"

    @property
    def display_name(self) -> str:
        """Human-readable display name."""
        return self.value.replace("_", " ").title()

    @property
    def is_fiction(self) -> bool:
        """Check if genre is fiction."""
        fiction_genres = {
            self.FANTASY, self.SCIENCE_FICTION, self.MYSTERY, self.ROMANCE,
            self.THRILLER, self.HORROR, self.HISTORICAL_FICTION, self.LITERARY_FICTION,
            self.YOUNG_ADULT, self.ADVENTURE, self.DRAMA, self.COMEDY, self.WESTERN,
            self.CRIME, self.DYSTOPIAN, self.URBAN_FANTASY, self.PARANORMAL, 
            self.CONTEMPORARY, self.CHILDREN, self.PICTURE_BOOK
        }
        return self in fiction_genres

def get_genre_mapping():
    """
    Complete genre mapping dictionary for all GenreType enum values.
    Includes common variations and aliases for flexibility.
    """
    genre_mapping = {
        # Fiction Genres - Direct mappings
        "FANTASY": GenreType.FANTASY,
        "SCIENCE_FICTION": GenreType.SCIENCE_FICTION,
        "MYSTERY": GenreType.MYSTERY,
        "ROMANCE": GenreType.ROMANCE,
        "THRILLER": GenreType.THRILLER,
        "HORROR": GenreType.HORROR,
        "HISTORICAL_FICTION": GenreType.HISTORICAL_FICTION,
        "LITERARY_FICTION": GenreType.LITERARY_FICTION,
        "YOUNG_ADULT": GenreType.YOUNG_ADULT,
        "ADVENTURE": GenreType.ADVENTURE,
        "DRAMA": GenreType.DRAMA,
        "COMEDY": GenreType.COMEDY,
        "WESTERN": GenreType.WESTERN,
        "CRIME": GenreType.CRIME,
        "DYSTOPIAN": GenreType.DYSTOPIAN,
        "URBAN_FANTASY": GenreType.URBAN_FANTASY,
        "PARANORMAL": GenreType.PARANORMAL,
        "CONTEMPORARY": GenreType.CONTEMPORARY,
        
        # Non-Fiction Genres - Direct mappings
        "BIOGRAPHY": GenreType.BIOGRAPHY,
        "MEMOIR": GenreType.MEMOIR,
        "SELF_HELP": GenreType.SELF_HELP,
        "BUSINESS": GenreType.BUSINESS,
        "HEALTH": GenreType.HEALTH,
        "TRAVEL": GenreType.TRAVEL,
        "COOKING": GenreType.COOKING,
        "HISTORY": GenreType.HISTORY,
        "SCIENCE": GenreType.SCIENCE,
        "TECHNOLOGY": GenreType.TECHNOLOGY,
        "PHILOSOPHY": GenreType.PHILOSOPHY,
        "RELIGION": GenreType.RELIGION,
        "POLITICS": GenreType.POLITICS,
        "EDUCATION": GenreType.EDUCATION,
        "REFERENCE": GenreType.REFERENCE,
#        "TRUE_CRIME": GenreType.TRUE_CRIME,
        
        # Specialized Genres - Direct mappings
        "CHILDREN": GenreType.CHILDREN,
        "PICTURE_BOOK": GenreType.PICTURE_BOOK,
        "POETRY": GenreType.POETRY,
        "SCREENPLAY": GenreType.SCREENPLAY,
        "TEXTBOOK": GenreType.TEXTBOOK,
        "MANUAL": GenreType.MANUAL,
        "OTHER": GenreType.OTHER,
        
        # Common Variations and Aliases
        "SCI_FI": GenreType.SCIENCE_FICTION,
        "SCIFI": GenreType.SCIENCE_FICTION,
        "SF": GenreType.SCIENCE_FICTION,
        "YA": GenreType.YOUNG_ADULT,
        "HISTORICAL": GenreType.HISTORICAL_FICTION,
        "LITERARY": GenreType.LITERARY_FICTION,
        "TRUE_CRIME": GenreType.TRUE_CRIME,
        "TRUECRIME": GenreType.TRUE_CRIME,
        "SELF_IMPROVEMENT": GenreType.SELF_HELP,
        "SELFHELP": GenreType.SELF_HELP,
        "HELP": GenreType.SELF_HELP,
        "COOKBOOK": GenreType.COOKING,
        "FOOD": GenreType.COOKING,
        "TECH": GenreType.TECHNOLOGY,
        "IT": GenreType.TECHNOLOGY,
        "COMPUTING": GenreType.TECHNOLOGY,
        "MEDICAL": GenreType.HEALTH,
        "FITNESS": GenreType.HEALTH,
        "WELLNESS": GenreType.HEALTH,
        "CHILDRENS": GenreType.CHILDREN,
        "KIDS": GenreType.CHILDREN,
        "CHILD": GenreType.CHILDREN,
        "PICTURE": GenreType.PICTURE_BOOK,
        "PICTUREBOOK": GenreType.PICTURE_BOOK,
        "POEMS": GenreType.POETRY,
        "SCRIPT": GenreType.SCREENPLAY,
        "SCREENWRITING": GenreType.SCREENPLAY,
        "PLAY": GenreType.SCREENPLAY,
        "ACADEMIC": GenreType.TEXTBOOK,  # Common mapping for academic content
        "EDUCATIONAL": GenreType.EDUCATION,
        "LEARNING": GenreType.EDUCATION,
        "TEACHING": GenreType.EDUCATION,
        "REFERENCE_BOOK": GenreType.REFERENCE,
        "ENCYCLOPEDIA": GenreType.REFERENCE,
        "DICTIONARY": GenreType.REFERENCE,
        "GUIDE": GenreType.MANUAL,
        "HOW_TO": GenreType.MANUAL,
        "HOWTO": GenreType.MANUAL,
        "INSTRUCTIONS": GenreType.MANUAL,
        "TUTORIAL": GenreType.MANUAL,
        
        # Genre Categories (mapping to most representative genre)
        "FICTION": GenreType.LITERARY_FICTION,  # Default fiction mapping
        "NON_FICTION": GenreType.REFERENCE,     # Default non-fiction mapping
        "NONFICTION": GenreType.REFERENCE,
        
        # Additional common terms
        "SUPERNATURAL": GenreType.PARANORMAL,
        "MAGIC": GenreType.FANTASY,
        "MAGICAL": GenreType.FANTASY,
        "DETECTIVE": GenreType.MYSTERY,
        "SUSPENSE": GenreType.THRILLER,
        "LOVE": GenreType.ROMANCE,
        "SCARY": GenreType.HORROR,
        "FUNNY": GenreType.COMEDY,
        "HUMOROUS": GenreType.COMEDY,
        "HUMOR": GenreType.COMEDY,
        "WAR": GenreType.HISTORICAL_FICTION,
        "MILITARY": GenreType.HISTORICAL_FICTION,
        "SPACE": GenreType.SCIENCE_FICTION,
        "FUTURE": GenreType.SCIENCE_FICTION,
        "FUTURISTIC": GenreType.SCIENCE_FICTION,
        "DYSTOPIA": GenreType.DYSTOPIAN,
        "URBAN": GenreType.URBAN_FANTASY,
        "CITY": GenreType.URBAN_FANTASY,
        "GHOST": GenreType.PARANORMAL,
        "VAMPIRE": GenreType.PARANORMAL,
        "WEREWOLF": GenreType.PARANORMAL,
        "MODERN": GenreType.CONTEMPORARY,
        "CURRENT": GenreType.CONTEMPORARY,
        "PRESENT": GenreType.CONTEMPORARY,
        
        # Business subcategories
        "ENTREPRENEURSHIP": GenreType.BUSINESS,
        "MARKETING": GenreType.BUSINESS,
        "FINANCE": GenreType.BUSINESS,
        "MANAGEMENT": GenreType.BUSINESS,
        "LEADERSHIP": GenreType.BUSINESS,
        "ECONOMICS": GenreType.BUSINESS,
        "INVESTING": GenreType.BUSINESS,
        "STARTUP": GenreType.BUSINESS,
        
        # Health subcategories
        "NUTRITION": GenreType.HEALTH,
        "DIET": GenreType.HEALTH,
        "EXERCISE": GenreType.HEALTH,
        "MENTAL_HEALTH": GenreType.HEALTH,
        "PSYCHOLOGY": GenreType.HEALTH,
        
        # Travel subcategories
        "TOURISM": GenreType.TRAVEL,
        "GUIDEBOOK": GenreType.TRAVEL,
        "ADVENTURE_TRAVEL": GenreType.TRAVEL,
        
        # Religion subcategories
        "SPIRITUAL": GenreType.RELIGION,
        "SPIRITUALITY": GenreType.RELIGION,
        "FAITH": GenreType.RELIGION,
        "CHRISTIAN": GenreType.RELIGION,
        "ISLAMIC": GenreType.RELIGION,
        "BUDDHIST": GenreType.RELIGION,
        "JEWISH": GenreType.RELIGION,
    }
    
    return genre_mapping


def map_genre(genre_string: str) -> GenreType:
    """
    Map a genre string to a GenreType enum value.
    
    Args:
        genre_string: String representation of genre (case-insensitive)
        
    Returns:
        GenreType enum value
        
    Raises:
        KeyError: If genre string is not found in mapping
    """
    mapping = get_genre_mapping()
    normalized_genre = genre_string.upper().strip()
    
    if normalized_genre in mapping:
        return mapping[normalized_genre]
    else:
        # Try to find partial matches
        for key, value in mapping.items():
            if normalized_genre in key or key in normalized_genre:
                return value
        
        # If no match found, raise error with suggestions
        raise KeyError(f"Genre '{genre_string}' not found. Available genres: {list(mapping.keys())}")



class SubGenre(str, Enum):
    """Sub-genre classifications for more specific categorization."""
    # Fantasy Sub-genres
    HIGH_FANTASY = "high_fantasy"
    LOW_FANTASY = "low_fantasy"
    EPIC_FANTASY = "epic_fantasy"
    SWORD_AND_SORCERY = "sword_and_sorcery"
    DARK_FANTASY = "dark_fantasy"
    COZY_FANTASY = "cozy_fantasy"
    
    # Science Fiction Sub-genres
    HARD_SF = "hard_science_fiction"
    SOFT_SF = "soft_science_fiction"
    SPACE_OPERA = "space_opera"
    CYBERPUNK = "cyberpunk"
    STEAMPUNK = "steampunk"
    DYSTOPIAN_SF = "dystopian_science_fiction"
    TIME_TRAVEL = "time_travel"
    ALTERNATE_HISTORY = "alternate_history"
    
    # Mystery Sub-genres
    COZY_MYSTERY = "cozy_mystery"
    POLICE_PROCEDURAL = "police_procedural"
    HARD_BOILED = "hard_boiled"
    LOCKED_ROOM = "locked_room"
    
    # Romance Sub-genres
    CONTEMPORARY_ROMANCE = "contemporary_romance"
    HISTORICAL_ROMANCE = "historical_romance"
    PARANORMAL_ROMANCE = "paranormal_romance"
    ROMANTIC_SUSPENSE = "romantic_suspense"
    EROTIC_ROMANCE = "erotic_romance"
    
    # Thriller Sub-genres
    PSYCHOLOGICAL_THRILLER = "psychological_thriller"
    LEGAL_THRILLER = "legal_thriller"
    MEDICAL_THRILLER = "medical_thriller"
    ESPIONAGE = "espionage"
    
    # Business Sub-genres
    ENTREPRENEURSHIP = "entrepreneurship"
    LEADERSHIP = "leadership"
    MARKETING = "marketing"
    FINANCE = "finance"
    MANAGEMENT = "management"
    ECONOMICS = "economics"
    
    # Self-Help Sub-genres
    PERSONAL_DEVELOPMENT = "personal_development"
    PRODUCTIVITY = "productivity"
    RELATIONSHIPS = "relationships"
    SPIRITUALITY = "spirituality"
    CAREER = "career"
    PARENTING = "parenting"
    
    # Technical Sub-genres
    PROGRAMMING = "programming"
    SOFTWARE_DEVELOPMENT = "software_development"
    DATA_SCIENCE = "data_science"
    ARTIFICIAL_INTELLIGENCE = "artificial_intelligence"
    CYBERSECURITY = "cybersecurity"
    WEB_DEVELOPMENT = "web_development"


# ============================================================================
# Story Structure and Planning Enumerations
# ============================================================================

class StoryStructure(str, Enum):
    """Story structure types."""
    THREE_ACT = "three_act"
    HERO_JOURNEY = "hero_journey"
    FREYTAG_PYRAMID = "freytag_pyramid"
    SEVEN_POINT = "seven_point"
    SAVE_THE_CAT = "save_the_cat"
    SNOWFLAKE = "snowflake"
    STORY_CIRCLE = "story_circle"
    FICHTEAN_CURVE = "fichtean_curve"
    IN_MEDIAS_RES = "in_medias_res"
    CUSTOM = "custom"
    
    @property
    def display_name(self) -> str:
        """Human-readable display name."""
        names = {
            self.THREE_ACT: "Three-Act Structure",
            self.HERO_JOURNEY: "Hero's Journey",
            self.FREYTAG_PYRAMID: "Freytag's Pyramid",
            self.SEVEN_POINT: "Seven-Point Story Structure",
            self.SAVE_THE_CAT: "Save the Cat Beat Sheet",
            self.SNOWFLAKE: "Snowflake Method",
            self.STORY_CIRCLE: "Story Circle",
            self.FICHTEAN_CURVE: "Fichtean Curve",
            self.IN_MEDIAS_RES: "In Medias Res",
            self.CUSTOM: "Custom Structure"
        }
        return names.get(self, self.value.replace("_", " ").title())

    @property
    def description(self) -> str:
        """Brief description of the structure."""
        descriptions = {
            self.THREE_ACT: "Classic beginning, middle, and end structure",
            self.HERO_JOURNEY: "Joseph Campbell's monomyth pattern",
            self.FREYTAG_PYRAMID: "Five-part dramatic structure with rising and falling action",
            self.SEVEN_POINT: "Dan Wells' structure focusing on character development",
            self.SAVE_THE_CAT: "Blake Snyder's 15-beat screenplay structure",
            self.SNOWFLAKE: "Randy Ingermanson's step-by-step development method",
            self.STORY_CIRCLE: "Dan Harmon's simplified hero's journey",
            self.FICHTEAN_CURVE: "Rising action with multiple crisis points",
            self.IN_MEDIAS_RES: "Starting in the middle of action",
            self.CUSTOM: "Create your own unique structure"
        }
        return descriptions.get(self, "")


class PlotType(str, Enum):
    """Basic plot archetypes."""
    OVERCOMING_THE_MONSTER = "overcoming_the_monster"
    RAGS_TO_RICHES = "rags_to_riches"
    THE_QUEST = "the_quest"
    VOYAGE_AND_RETURN = "voyage_and_return"
    COMEDY = "comedy"
    TRAGEDY = "tragedy"
    REBIRTH = "rebirth"
    MYSTERY = "mystery"
    ROMANCE = "romance"
    COMING_OF_AGE = "coming_of_age"
    REVENGE = "revenge"
    REDEMPTION = "redemption"
    SACRIFICE = "sacrifice"
    SURVIVAL = "survival"
    FISH_OUT_OF_WATER = "fish_out_of_water"


class ConflictType(str, Enum):
    """Types of conflict in stories."""
    PERSON_VS_PERSON = "person_vs_person"
    PERSON_VS_SELF = "person_vs_self"
    PERSON_VS_SOCIETY = "person_vs_society"
    PERSON_VS_NATURE = "person_vs_nature"
    PERSON_VS_TECHNOLOGY = "person_vs_technology"
    PERSON_VS_SUPERNATURAL = "person_vs_supernatural"
    PERSON_VS_FATE = "person_vs_fate"
    PERSON_VS_GOD = "person_vs_god"


# ============================================================================
# Character Development Enumerations
# ============================================================================

class CharacterRole(str, Enum):
    """Character role types."""
    PROTAGONIST = "protagonist"
    ANTAGONIST = "antagonist"
    DEUTERAGONIST = "deuteragonist"  # Second main character
    LOVE_INTEREST = "love_interest"
    MENTOR = "mentor"
    ALLY = "ally"
    THRESHOLD_GUARDIAN = "threshold_guardian"
    HERALD = "herald"
    TRICKSTER = "trickster"
    SHAPESHIFTER = "shapeshifter"
    SUPPORTING = "supporting"
    MINOR = "minor"
    NARRATOR = "narrator"
    FOIL = "foil"  # Character who contrasts with protagonist
    
    @property
    def description(self) -> str:
        """Description of the character role."""
        descriptions = {
            self.PROTAGONIST: "Main character driving the story",
            self.ANTAGONIST: "Primary opponent or obstacle",
            self.DEUTERAGONIST: "Second most important character",
            self.LOVE_INTEREST: "Romantic partner or potential partner",
            self.MENTOR: "Wise guide who helps the protagonist",
            self.ALLY: "Friend or supporter of the protagonist",
            self.THRESHOLD_GUARDIAN: "Tests the protagonist's resolve",
            self.HERALD: "Announces the need for change",
            self.TRICKSTER: "Provides comic relief and disrupts status quo",
            self.SHAPESHIFTER: "Loyalty and intentions are unclear",
            self.SUPPORTING: "Important to plot but not central",
            self.MINOR: "Small role in the story",
            self.NARRATOR: "Tells the story (may or may not be a character)",
            self.FOIL: "Highlights protagonist's qualities through contrast"
        }
        return descriptions.get(self, "")


class CharacterArchetype(str, Enum):
    """Character archetypes based on psychology and mythology."""
    THE_HERO = "the_hero"
    THE_INNOCENT = "the_innocent"
    THE_EXPLORER = "the_explorer"
    THE_SAGE = "the_sage"
    THE_OUTLAW = "the_outlaw"
    THE_MAGICIAN = "the_magician"
    THE_REGULAR_PERSON = "the_regular_person"
    THE_LOVER = "the_lover"
    THE_JESTER = "the_jester"
    THE_CAREGIVER = "the_caregiver"
    THE_RULER = "the_ruler"
    THE_CREATOR = "the_creator"


class CharacterArcType(str, Enum):
    """Types of character development arcs."""
    CHANGE_ARC = "change_arc"  # Character transforms significantly
    GROWTH_ARC = "growth_arc"  # Character learns and develops
    FALL_ARC = "fall_arc"  # Character degrades or fails
    FLAT_ARC = "flat_arc"  # Character stays the same but changes others
    CORRUPTION_ARC = "corruption_arc"  # Good character becomes bad
    REDEMPTION_ARC = "redemption_arc"  # Bad character becomes good
    DISILLUSIONMENT_ARC = "disillusionment_arc"  # Character loses naive beliefs


class PersonalityTrait(str, Enum):
    """Common personality traits for character development."""
    # Positive Traits
    BRAVE = "brave"
    LOYAL = "loyal"
    INTELLIGENT = "intelligent"
    COMPASSIONATE = "compassionate"
    DETERMINED = "determined"
    HONEST = "honest"
    CREATIVE = "creative"
    PATIENT = "patient"
    OPTIMISTIC = "optimistic"
    HUMBLE = "humble"
    GENEROUS = "generous"
    WISE = "wise"
    
    # Negative Traits
    ARROGANT = "arrogant"
    SELFISH = "selfish"
    IMPULSIVE = "impulsive"
    STUBBORN = "stubborn"
    JEALOUS = "jealous"
    COWARDLY = "cowardly"
    DISHONEST = "dishonest"
    CRUEL = "cruel"
    LAZY = "lazy"
    PESSIMISTIC = "pessimistic"
    GREEDY = "greedy"
    MANIPULATIVE = "manipulative"
    
    # Neutral/Complex Traits
    AMBITIOUS = "ambitious"
    INDEPENDENT = "independent"
    MYSTERIOUS = "mysterious"
    ECCENTRIC = "eccentric"
    PRAGMATIC = "pragmatic"
    CAUTIOUS = "cautious"


# ============================================================================
# World-Building Enumerations
# ============================================================================

class WorldType(str, Enum):
    """Types of fictional worlds."""
    REALISTIC = "realistic"  # Real world, no fantastical elements
    ALTERNATE_HISTORY = "alternate_history"  # Real world with changes
    LOW_FANTASY = "low_fantasy"  # Real world with subtle magic
    HIGH_FANTASY = "high_fantasy"  # Completely fictional world
    URBAN_FANTASY = "urban_fantasy"  # Modern world with hidden magic
    SCIENCE_FICTION = "science_fiction"  # Future or alternate reality
    STEAMPUNK = "steampunk"  # Victorian-era with advanced steam technology
    CYBERPUNK = "cyberpunk"  # High-tech, low-life future
    POST_APOCALYPTIC = "post_apocalyptic"  # After civilization collapse
    SPACE_OPERA = "space_opera"  # Galaxy-spanning adventure
    PARALLEL_UNIVERSE = "parallel_universe"  # Different version of reality


class MagicSystemType(str, Enum):
    """Types of magic systems."""
    HARD_MAGIC = "hard_magic"  # Clearly defined rules and limitations
    SOFT_MAGIC = "soft_magic"  # Mysterious and undefined
    ELEMENTAL = "elemental"  # Based on classical elements
    DIVINE = "divine"  # Power from gods or deities
    ARCANE = "arcane"  # Academic/scholarly magic
    INNATE = "innate"  # Born with magical ability
    RITUAL = "ritual"  # Requires specific ceremonies
    ARTIFACT = "artifact"  # Magic through objects
    PSIONICS = "psionics"  # Mental/psychic powers
    SYMBIOTIC = "symbiotic"  # Magic through creatures/spirits
    BLOOD_MAGIC = "blood_magic"  # Power through sacrifice
    ALCHEMY = "alchemy"  # Transformation and creation


class TechnologyLevel(str, Enum):
    """Technology advancement levels."""
    STONE_AGE = "stone_age"
    BRONZE_AGE = "bronze_age"
    IRON_AGE = "iron_age"
    CLASSICAL = "classical"  # Ancient Rome/Greece level
    MEDIEVAL = "medieval"
    RENAISSANCE = "renaissance"
    INDUSTRIAL = "industrial"  # 1800s steam power
    EARLY_MODERN = "early_modern"  # Early 1900s
    MODERN = "modern"  # Current day
    NEAR_FUTURE = "near_future"  # 50-100 years ahead
    FAR_FUTURE = "far_future"  # Centuries ahead
    POST_HUMAN = "post_human"  # Beyond current humanity
    MIXED = "mixed"  # Multiple levels coexisting


class GovernmentType(str, Enum):
    """Types of government systems."""
    MONARCHY = "monarchy"
    DEMOCRACY = "democracy"
    REPUBLIC = "republic"
    DICTATORSHIP = "dictatorship"
    OLIGARCHY = "oligarchy"
    THEOCRACY = "theocracy"
    ANARCHY = "anarchy"
    FEUDALISM = "feudalism"
    TRIBALISM = "tribalism"
    EMPIRE = "empire"
    CITY_STATE = "city_state"
    CONFEDERATION = "confederation"
    MAGOCRACY = "magocracy"  # Rule by magic users
    TECHNOCRACY = "technocracy"  # Rule by technical experts
    CORPORATE = "corporate"  # Rule by corporations


# ============================================================================
# Writing Style and Tone Enumerations
# ============================================================================

class WritingStyle(str, Enum):
    """Writing style preferences."""
    ACADEMIC = "academic"
    CONVERSATIONAL = "conversational"
    TECHNICAL = "technical"
    NARRATIVE = "narrative"
    INSTRUCTIONAL = "instructional"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    JOURNALISTIC = "journalistic"
    CASUAL = "casual"
    FORMAL = "formal"
    LITERARY = "literary"
    DESCRIPTIVE = "descriptive"
    DIALOGUE_HEAVY = "dialogue_heavy"
    ACTION_PACKED = "action_packed"
    ATMOSPHERIC = "atmospheric"
    MINIMALIST = "minimalist"
    ORNATE = "ornate"


class NarrativePOV(str, Enum):
    """Point of view options."""
    FIRST_PERSON = "first_person"  # I, me, my
    SECOND_PERSON = "second_person"  # You, your
    THIRD_PERSON_LIMITED = "third_person_limited"  # He/she, one character's perspective
    THIRD_PERSON_OMNISCIENT = "third_person_omniscient"  # He/she, all perspectives
    THIRD_PERSON_OBJECTIVE = "third_person_objective"  # He/she, no internal thoughts
    MULTIPLE_POV = "multiple_pov"  # Alternating perspectives
    EPISTOLARY = "epistolary"  # Letters, documents, emails
    STREAM_OF_CONSCIOUSNESS = "stream_of_consciousness"  # Internal thought flow


class ToneType(str, Enum):
    """Overall tone of the writing."""
    SERIOUS = "serious"
    HUMOROUS = "humorous"
    DARK = "dark"
    LIGHTHEARTED = "lighthearted"
    MYSTERIOUS = "mysterious"
    ROMANTIC = "romantic"
    SUSPENSEFUL = "suspenseful"
    MELANCHOLIC = "melancholic"
    OPTIMISTIC = "optimistic"
    CYNICAL = "cynical"
    NOSTALGIC = "nostalgic"
    IRONIC = "ironic"
    SATIRICAL = "satirical"
    DRAMATIC = "dramatic"
    INSPIRATIONAL = "inspirational"
    PHILOSOPHICAL = "philosophical"
    WHIMSICAL = "whimsical"
    GRITTY = "gritty"


class PacingType(str, Enum):
    """Story pacing preferences."""
    FAST_PACED = "fast_paced"
    MODERATE_PACED = "moderate_paced"
    SLOW_BURN = "slow_burn"
    VARIABLE = "variable"
    BREAKNECK = "breakneck"
    LEISURELY = "leisurely"
    MEASURED = "measured"
    ESCALATING = "escalating"


# ============================================================================
# Target Audience and Marketing Enumerations
# ============================================================================

class AgeGroup(str, Enum):
    """Target age groups."""
    CHILDREN = "children"  # 5-12
    MIDDLE_GRADE = "middle_grade"  # 8-12
    YOUNG_ADULT = "young_adult"  # 13-18
    NEW_ADULT = "new_adult"  # 18-25
    ADULT = "adult"  # 25+
    ALL_AGES = "all_ages"


class ReadingLevel(str, Enum):
    """Reading difficulty levels."""
    ELEMENTARY = "elementary"
    MIDDLE_SCHOOL = "middle_school"
    HIGH_SCHOOL = "high_school"
    COLLEGE = "college"
    GRADUATE = "graduate"
    GENERAL_ADULT = "general_adult"


class AudienceType(str, Enum):
    """Target audience types."""
    GENERAL_READERS = "general_readers"
    GENRE_FANS = "genre_fans"
    PROFESSIONALS = "professionals"
    ACADEMICS = "academics"
    STUDENTS = "students"
    HOBBYISTS = "hobbyists"
    BEGINNERS = "beginners"
    EXPERTS = "experts"
    PARENTS = "parents"
    ENTREPRENEURS = "entrepreneurs"
    CREATIVES = "creatives"
    TECHNICAL_AUDIENCE = "technical_audience"


class ContentWarning(str, Enum):
    """Content warnings for sensitive material."""
    VIOLENCE = "violence"
    SEXUAL_CONTENT = "sexual_content"
    STRONG_LANGUAGE = "strong_language"
    SUBSTANCE_ABUSE = "substance_abuse"
    MENTAL_HEALTH = "mental_health"
    DEATH = "death"
    TRAUMA = "trauma"
    DISCRIMINATION = "discrimination"
    RELIGIOUS_CONTENT = "religious_content"
    POLITICAL_CONTENT = "political_content"
    HORROR_ELEMENTS = "horror_elements"
    GORE = "gore"
    SUICIDE = "suicide"
    SELF_HARM = "self_harm"
    EATING_DISORDERS = "eating_disorders"
    DOMESTIC_VIOLENCE = "domestic_violence"


# ============================================================================
# Research and Development Enumerations
# ============================================================================

class ResearchPriority(str, Enum):
    """Research priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    OPTIONAL = "optional"


class ResearchType(str, Enum):
    """Types of research needed."""
    HISTORICAL = "historical"
    SCIENTIFIC = "scientific"
    TECHNICAL = "technical"
    CULTURAL = "cultural"
    GEOGRAPHICAL = "geographical"
    LINGUISTIC = "linguistic"
    PROFESSIONAL = "professional"
    LEGAL = "legal"
    MEDICAL = "medical"
    PSYCHOLOGICAL = "psychological"
    SOCIOLOGICAL = "sociological"
    ECONOMIC = "economic"
    POLITICAL = "political"
    MILITARY = "military"
    RELIGIOUS = "religious"
    ARTISTIC = "artistic"
    ARCHITECTURAL = "architectural"
    CULINARY = "culinary"
    FASHION = "fashion"
    SPORTS = "sports"


class SourceType(str, Enum):
    """Types of research sources."""
    BOOK = "book"
    ARTICLE = "article"
    JOURNAL_PAPER = "journal_paper"
    WEBSITE = "website"
    INTERVIEW = "interview"
    DOCUMENTARY = "documentary"
    VIDEO = "video"
    PODCAST = "podcast"
    BLOG_POST = "blog_post"
    NEWS_ARTICLE = "news_article"
    ACADEMIC_PAPER = "academic_paper"
    GOVERNMENT_REPORT = "government_report"
    SURVEY = "survey"
    PERSONAL_EXPERIENCE = "personal_experience"
    EXPERT_CONSULTATION = "expert_consultation"


# ============================================================================
# Publication and Production Enumerations
# ============================================================================

class PublicationRoute(str, Enum):
    """Publishing strategy options."""
    TRADITIONAL = "traditional"
    SELF_PUBLISHED = "self_published"
    HYBRID = "hybrid"
    PRINT_ON_DEMAND = "print_on_demand"
    EBOOK_ONLY = "ebook_only"
    BLOG_SERIAL = "blog_serial"
    NEWSLETTER = "newsletter"
    COURSE_MATERIAL = "course_material"
    INTERNAL_DOCUMENT = "internal_document"
    ACADEMIC_PRESS = "academic_press"
    INDIE_PRESS = "indie_press"


class BookFormat(str, Enum):
    """Book format options."""
    PAPERBACK = "paperback"
    HARDCOVER = "hardcover"
    EBOOK = "ebook"
    AUDIOBOOK = "audiobook"
    LARGE_PRINT = "large_print"
    DIGITAL_MAGAZINE = "digital_magazine"
    INTERACTIVE_EBOOK = "interactive_ebook"
    GRAPHIC_NOVEL = "graphic_novel"
    ILLUSTRATED = "illustrated"


class BookLength(str, Enum):
    """Book length categories with word counts."""
    FLASH_FICTION = "flash_fiction"  # Under 1,000 words
    SHORT_STORY = "short_story"  # 1,000-7,500 words
    NOVELETTE = "novelette"  # 7,500-17,500 words
    NOVELLA = "novella"  # 17,500-40,000 words
    SHORT_NOVEL = "short_novel"  # 40,000-60,000 words
    STANDARD_NOVEL = "standard_novel"  # 60,000-90,000 words
    LONG_NOVEL = "long_novel"  # 90,000-120,000 words
    EPIC_NOVEL = "epic_novel"  # 120,000+ words
    
    # Non-fiction
    ARTICLE = "article"  # 500-2,000 words
    ESSAY = "essay"  # 1,000-5,000 words
    GUIDE = "guide"  # 5,000-15,000 words
    MANUAL = "manual"  # 15,000-50,000 words
    COMPREHENSIVE_BOOK = "comprehensive_book"  # 50,000+ words
    
    @property
    def word_count_range(self) -> Tuple[int, int]:
        """Get the word count range for this length."""
        ranges = {
            self.FLASH_FICTION: (100, 1000),
            self.SHORT_STORY: (1000, 7500),
            self.NOVELETTE: (7500, 17500),
            self.NOVELLA: (17500, 40000),
            self.SHORT_NOVEL: (40000, 60000),
            self.STANDARD_NOVEL: (60000, 90000),
            self.LONG_NOVEL: (90000, 120000),
            self.EPIC_NOVEL: (120000, 200000),
            self.ARTICLE: (500, 2000),
            self.ESSAY: (1000, 5000),
            self.GUIDE: (5000, 15000),
            self.MANUAL: (15000, 50000),
            self.COMPREHENSIVE_BOOK: (50000, 150000),
        }
        return ranges.get(self, (50000, 90000))


# ============================================================================
# Status and Progress Enumerations
# ============================================================================

class ProjectStatus(str, Enum):
    """Project status options."""
    BRAINSTORMING = "brainstorming"
    PLANNING = "planning"
    RESEARCHING = "researching"
    OUTLINING = "outlining"
    WRITING = "writing"
    FIRST_DRAFT = "first_draft"
    REVISING = "revising"
    EDITING = "editing"
    BETA_READING = "beta_reading"
    FINAL_DRAFT = "final_draft"
    FORMATTING = "formatting"
    SEEKING_PUBLISHER = "seeking_publisher"
    PUBLISHING = "publishing"
    PUBLISHED = "published"
    MARKETING = "marketing"
    ON_HOLD = "on_hold"
    ABANDONED = "abandoned"
    COMPLETED = "completed"


class PriorityLevel(str, Enum):
    """Priority levels for tasks and elements."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    OPTIONAL = "optional"


class DifficultyLevel(str, Enum):
    """Difficulty levels for writing tasks."""
    VERY_EASY = "very_easy"
    EASY = "easy"
    MODERATE = "moderate"
    CHALLENGING = "challenging"
    VERY_CHALLENGING = "very_challenging"
    EXPERT_LEVEL = "expert_level"


# ============================================================================
# AI and Tool Preferences
# ============================================================================

class AIAssistanceLevel(str, Enum):
    """Level of AI assistance desired."""
    MINIMAL = "minimal"  # Just basic suggestions
    MODERATE = "moderate"  # Regular assistance
    HEAVY = "heavy"  # Extensive AI support
    COLLABORATIVE = "collaborative"  # AI as writing partner
    CUSTOM = "custom"  # User-defined settings


class WritingTool(str, Enum):
    """Preferred writing tools and software."""
    WORD_PROCESSOR = "word_processor"  # MS Word, Google Docs
    MARKDOWN_EDITOR = "markdown_editor"  # Typora, Mark Text
    SPECIALIZED_SOFTWARE = "specialized_software"  # Scrivener, Ulysses
    NOTEBOOK = "notebook"  # Physical notebook
    DICTATION = "dictation"  # Voice recording
    MOBILE_APP = "mobile_app"  # Phone/tablet writing
    TYPEWRITER = "typewriter"  # Classic typewriter
    COLLABORATIVE_PLATFORM = "collaborative_platform"  # Notion, Confluence


class WritingSchedule(str, Enum):
    """Writing schedule preferences."""
    DAILY = "daily"
    WEEKDAYS_ONLY = "weekdays_only"
    WEEKENDS_ONLY = "weekends_only"
    EVERY_OTHER_DAY = "every_other_day"
    WEEKLY = "weekly"
    SPORADIC = "sporadic"
    INTENSIVE_PERIODS = "intensive_periods"  # Writing retreats/sprints
    DEADLINE_DRIVEN = "deadline_driven"


# ============================================================================
# Utility Functions for Enumerations
# ============================================================================

class EnumGroups:
    """Groupings of related enumerations for UI organization."""
    
    GENRE_AND_CATEGORY = [GenreType, SubGenre, PlotType]
    STORY_STRUCTURE = [StoryStructure, ConflictType, NarrativePOV]
    CHARACTER_DEVELOPMENT = [CharacterRole, CharacterArchetype, CharacterArcType, PersonalityTrait]
    WORLD_BUILDING = [WorldType, MagicSystemType, TechnologyLevel, GovernmentType]
    WRITING_STYLE = [WritingStyle, ToneType, PacingType]
    AUDIENCE_AND_MARKET = [AgeGroup, ReadingLevel, AudienceType, ContentWarning]
    RESEARCH = [ResearchType, ResearchPriority, SourceType]
    PUBLICATION = [PublicationRoute, BookFormat, BookLength]
    PROJECT_MANAGEMENT = [ProjectStatus, PriorityLevel, DifficultyLevel]
    AI_AND_TOOLS = [AIAssistanceLevel, WritingTool, WritingSchedule]


def get_enum_choices(enum_class) -> List[Tuple[str, str]]:
    """Convert enum to choices list for form fields."""
    return [(item.value, item.value.replace('_', ' ').title()) for item in enum_class]


def get_enum_grouped_choices() -> Dict[str, List[Tuple[str, str]]]:
    """Get all enum choices grouped by category for complex UI forms."""
    grouped = {}
    
    for group_name, enums in EnumGroups.__dict__.items():
        if not group_name.startswith('_') and isinstance(enums, list):
            grouped[group_name.lower().replace('_', ' ').title()] = {}
            for enum_class in enums:
                enum_name = enum_class.__name__.replace('Type', '').replace('Level', '')
                grouped[group_name.lower().replace('_', ' ').title()][enum_name] = get_enum_choices(enum_class)
    
    return grouped


def get_default_values() -> Dict[str, str]:
    """Get sensible default values for common book creation scenarios."""
    return {
        'genre': GenreType.FANTASY.value,
        'structure': StoryStructure.THREE_ACT.value,
        'pov': NarrativePOV.THIRD_PERSON_LIMITED.value,
        'tone': ToneType.SERIOUS.value,
        'pacing': PacingType.MODERATE_PACED.value,
        'age_group': AgeGroup.ADULT.value,
        'reading_level': ReadingLevel.GENERAL_ADULT.value,
        'book_length': BookLength.STANDARD_NOVEL.value,
        'publication_route': PublicationRoute.SELF_PUBLISHED.value,
        'format': BookFormat.EBOOK.value,
        'ai_assistance': AIAssistanceLevel.MODERATE.value,
        'priority': PriorityLevel.MEDIUM.value,
        'difficulty': DifficultyLevel.MODERATE.value,
        'status': ProjectStatus.PLANNING.value
    }


def get_genre_recommendations(genre: GenreType) -> Dict[str, List[str]]:
    """Get recommendations based on selected genre."""
    recommendations = {
        GenreType.FANTASY: {
            'sub_genres': [SubGenre.HIGH_FANTASY.value, SubGenre.EPIC_FANTASY.value],
            'structures': [StoryStructure.HERO_JOURNEY.value, StoryStructure.THREE_ACT.value],
            'character_roles': [CharacterRole.PROTAGONIST.value, CharacterRole.MENTOR.value, CharacterRole.ANTAGONIST.value],
            'world_type': [WorldType.HIGH_FANTASY.value],
            'magic_system': [MagicSystemType.HARD_MAGIC.value, MagicSystemType.ELEMENTAL.value],
            'conflicts': [ConflictType.PERSON_VS_SUPERNATURAL.value, ConflictType.PERSON_VS_PERSON.value],
            'length': [BookLength.STANDARD_NOVEL.value, BookLength.LONG_NOVEL.value]
        },
        GenreType.SCIENCE_FICTION: {
            'sub_genres': [SubGenre.SPACE_OPERA.value, SubGenre.HARD_SF.value],
            'structures': [StoryStructure.THREE_ACT.value, StoryStructure.SEVEN_POINT.value],
            'world_type': [WorldType.SCIENCE_FICTION.value],
            'tech_level': [TechnologyLevel.FAR_FUTURE.value, TechnologyLevel.NEAR_FUTURE.value],
            'conflicts': [ConflictType.PERSON_VS_TECHNOLOGY.value, ConflictType.PERSON_VS_SOCIETY.value]
        },
        GenreType.MYSTERY: {
            'sub_genres': [SubGenre.COZY_MYSTERY.value, SubGenre.POLICE_PROCEDURAL.value],
            'structures': [StoryStructure.THREE_ACT.value, StoryStructure.FICHTEAN_CURVE.value],
            'plot_types': [PlotType.MYSTERY.value],
            'pacing': [PacingType.MODERATE_PACED.value, PacingType.FAST_PACED.value]
        },
        GenreType.ROMANCE: {
            'sub_genres': [SubGenre.CONTEMPORARY_ROMANCE.value, SubGenre.HISTORICAL_ROMANCE.value],
            'character_roles': [CharacterRole.PROTAGONIST.value, CharacterRole.LOVE_INTEREST.value],
            'plot_types': [PlotType.ROMANCE.value],
            'tone': [ToneType.ROMANTIC.value, ToneType.LIGHTHEARTED.value]
        },
        GenreType.BUSINESS: {
            'sub_genres': [SubGenre.ENTREPRENEURSHIP.value, SubGenre.LEADERSHIP.value],
            'writing_style': [WritingStyle.CONVERSATIONAL.value, WritingStyle.INSTRUCTIONAL.value],
            'audience': [AudienceType.PROFESSIONALS.value, AudienceType.ENTREPRENEURS.value],
            'research_types': [ResearchType.ECONOMIC.value, ResearchType.PROFESSIONAL.value]
        },
        GenreType.SELF_HELP: {
            'sub_genres': [SubGenre.PERSONAL_DEVELOPMENT.value, SubGenre.PRODUCTIVITY.value],
            'writing_style': [WritingStyle.CONVERSATIONAL.value, WritingStyle.INSTRUCTIONAL.value],
            'tone': [ToneType.INSPIRATIONAL.value, ToneType.OPTIMISTIC.value],
            'audience': [AudienceType.GENERAL_READERS.value]
        }
    }
    
    return recommendations.get(genre, {})


def validate_enum_combination(selections: Dict[str, str]) -> List[str]:
    """Validate that selected enum combinations make sense together."""
    warnings = []
    
    genre = selections.get('genre')
    structure = selections.get('structure')
    world_type = selections.get('world_type')
    age_group = selections.get('age_group')
    content_warnings = selections.get('content_warnings', [])
    
    # Genre-specific validations
    if genre == GenreType.CHILDREN.value:
        if age_group not in [AgeGroup.CHILDREN.value, AgeGroup.MIDDLE_GRADE.value]:
            warnings.append("Children's genre typically targets children or middle-grade audiences")
        
        if any(warning in content_warnings for warning in [
            ContentWarning.VIOLENCE.value, ContentWarning.SEXUAL_CONTENT.value, ContentWarning.STRONG_LANGUAGE.value
        ]):
            warnings.append("Consider if content warnings are appropriate for children's books")
    
    # Structure validations
    if structure == StoryStructure.HERO_JOURNEY.value and genre in [GenreType.ROMANCE.value, GenreType.BUSINESS.value]:
        warnings.append("Hero's Journey is uncommon for romance and business books")
    
    # World-building validations
    if world_type == WorldType.REALISTIC.value and genre in [GenreType.FANTASY.value, GenreType.SCIENCE_FICTION.value]:
        warnings.append("Realistic world type conflicts with fantasy/sci-fi genre")
    
    return warnings


# ============================================================================
# Advanced Enum Utilities
# ============================================================================

class EnumMetadata:
    """Metadata and additional information for enums."""
    
    POPULARITY_SCORES = {
        # Genre popularity (1-10 scale)
        GenreType.FANTASY: 9,
        GenreType.ROMANCE: 10,
        GenreType.MYSTERY: 8,
        GenreType.SCIENCE_FICTION: 7,
        GenreType.THRILLER: 8,
        GenreType.LITERARY_FICTION: 6,
        GenreType.SELF_HELP: 9,
        GenreType.BUSINESS: 7,
        
        # Structure popularity
        StoryStructure.THREE_ACT: 10,
        StoryStructure.HERO_JOURNEY: 8,
        StoryStructure.SAVE_THE_CAT: 7,
        StoryStructure.SEVEN_POINT: 6,
    }
    
    DIFFICULTY_SCORES = {
        # Writing difficulty (1-10 scale)
        GenreType.LITERARY_FICTION: 9,
        GenreType.SCIENCE_FICTION: 8,
        GenreType.FANTASY: 7,
        GenreType.MYSTERY: 7,
        GenreType.ROMANCE: 5,
        GenreType.THRILLER: 6,
        
        # Structure difficulty
        StoryStructure.THREE_ACT: 5,
        StoryStructure.HERO_JOURNEY: 7,
        StoryStructure.SEVEN_POINT: 8,
        StoryStructure.SNOWFLAKE: 9,
    }
    
    MARKET_VIABILITY = {
        # Commercial success potential (1-10 scale)
        GenreType.ROMANCE: 10,
        GenreType.FANTASY: 9,
        GenreType.MYSTERY: 8,
        GenreType.THRILLER: 8,
        GenreType.SELF_HELP: 9,
        GenreType.BUSINESS: 7,
        GenreType.LITERARY_FICTION: 4,
        GenreType.POETRY: 2,
    }


def get_enum_with_metadata(enum_class) -> List[Dict[str, any]]:
    """Get enum choices with additional metadata for advanced UI."""
    choices = []
    
    for item in enum_class:
        choice_data = {
            'value': item.value,
            'display_name': getattr(item, 'display_name', item.value.replace('_', ' ').title()),
            'description': getattr(item, 'description', ''),
            'popularity': EnumMetadata.POPULARITY_SCORES.get(item, 5),
            'difficulty': EnumMetadata.DIFFICULTY_SCORES.get(item, 5),
            'market_viability': EnumMetadata.MARKET_VIABILITY.get(item, 5),
        }
        
        # Add genre-specific metadata
        if hasattr(item, 'is_fiction'):
            choice_data['is_fiction'] = item.is_fiction
        
        # Add word count ranges for length
        if hasattr(item, 'word_count_range'):
            choice_data['word_count_range'] = item.word_count_range
        
        choices.append(choice_data)
    
    return choices


def create_ui_form_config() -> Dict[str, any]:
    """Create complete UI form configuration for book creation wizard."""
    return {
        'steps': [
            {
                'step': 1,
                'title': 'Book Basics',
                'description': 'Define the core concept of your book',
                'fields': [
                    {
                        'name': 'title',
                        'type': 'text',
                        'label': 'Book Title',
                        'required': True,
                        'placeholder': 'Enter your book title...'
                    },
                    {
                        'name': 'genre',
                        'type': 'select',
                        'label': 'Primary Genre',
                        'required': True,
                        'choices': get_enum_with_metadata(GenreType),
                        'default': GenreType.FANTASY.value
                    },
                    {
                        'name': 'sub_genre',
                        'type': 'select',
                        'label': 'Sub-Genre (Optional)',
                        'choices': get_enum_with_metadata(SubGenre),
                        'dependent_on': 'genre'
                    },
                    {
                        'name': 'length',
                        'type': 'select',
                        'label': 'Target Length',
                        'required': True,
                        'choices': get_enum_with_metadata(BookLength),
                        'default': BookLength.STANDARD_NOVEL.value
                    }
                ]
            },
            {
                'step': 2,
                'title': 'Story Structure',
                'description': 'Choose how your story will be organized',
                'fields': [
                    {
                        'name': 'structure',
                        'type': 'select',
                        'label': 'Story Structure',
                        'required': True,
                        'choices': get_enum_with_metadata(StoryStructure),
                        'default': StoryStructure.THREE_ACT.value
                    },
                    {
                        'name': 'plot_type',
                        'type': 'select',
                        'label': 'Basic Plot Type',
                        'choices': get_enum_choices(PlotType)
                    },
                    {
                        'name': 'conflict_types',
                        'type': 'multiselect',
                        'label': 'Types of Conflict',
                        'choices': get_enum_choices(ConflictType),
                        'max_selections': 3
                    },
                    {
                        'name': 'pov',
                        'type': 'select',
                        'label': 'Point of View',
                        'required': True,
                        'choices': get_enum_choices(NarrativePOV),
                        'default': NarrativePOV.THIRD_PERSON_LIMITED.value
                    }
                ]
            },
            {
                'step': 3,
                'title': 'Characters & World',
                'description': 'Define your characters and setting',
                'fields': [
                    {
                        'name': 'main_character_role',
                        'type': 'select',
                        'label': 'Main Character Type',
                        'choices': get_enum_choices(CharacterRole),
                        'default': CharacterRole.PROTAGONIST.value
                    },
                    {
                        'name': 'character_archetype',
                        'type': 'select',
                        'label': 'Character Archetype',
                        'choices': get_enum_choices(CharacterArchetype)
                    },
                    {
                        'name': 'world_type',
                        'type': 'select',
                        'label': 'World Setting',
                        'choices': get_enum_choices(WorldType),
                        'default': WorldType.REALISTIC.value
                    },
                    {
                        'name': 'magic_system',
                        'type': 'select',
                        'label': 'Magic System (if applicable)',
                        'choices': get_enum_choices(MagicSystemType),
                        'show_if': {'world_type': [WorldType.HIGH_FANTASY.value, WorldType.LOW_FANTASY.value]}
                    }
                ]
            },
            {
                'step': 4,
                'title': 'Writing Style',
                'description': 'Define your writing approach',
                'fields': [
                    {
                        'name': 'writing_style',
                        'type': 'select',
                        'label': 'Writing Style',
                        'choices': get_enum_choices(WritingStyle),
                        'default': WritingStyle.NARRATIVE.value
                    },
                    {
                        'name': 'tone',
                        'type': 'select',
                        'label': 'Overall Tone',
                        'choices': get_enum_choices(ToneType),
                        'default': ToneType.SERIOUS.value
                    },
                    {
                        'name': 'pacing',
                        'type': 'select',
                        'label': 'Story Pacing',
                        'choices': get_enum_choices(PacingType),
                        'default': PacingType.MODERATE_PACED.value
                    }
                ]
            },
            {
                'step': 5,
                'title': 'Audience & Publishing',
                'description': 'Define your target audience and goals',
                'fields': [
                    {
                        'name': 'age_group',
                        'type': 'select',
                        'label': 'Target Age Group',
                        'choices': get_enum_choices(AgeGroup),
                        'default': AgeGroup.ADULT.value
                    },
                    {
                        'name': 'audience_type',
                        'type': 'select',
                        'label': 'Target Audience',
                        'choices': get_enum_choices(AudienceType),
                        'default': AudienceType.GENERAL_READERS.value
                    },
                    {
                        'name': 'publication_route',
                        'type': 'select',
                        'label': 'Publishing Goal',
                        'choices': get_enum_choices(PublicationRoute),
                        'default': PublicationRoute.SELF_PUBLISHED.value
                    },
                    {
                        'name': 'content_warnings',
                        'type': 'multiselect',
                        'label': 'Content Warnings (if any)',
                        'choices': get_enum_choices(ContentWarning),
                        'optional': True
                    }
                ]
            },
            {
                'step': 6,
                'title': 'AI Assistance',
                'description': 'Configure your AI writing assistance',
                'fields': [
                    {
                        'name': 'ai_assistance_level',
                        'type': 'select',
                        'label': 'AI Assistance Level',
                        'choices': get_enum_choices(AIAssistanceLevel),
                        'default': AIAssistanceLevel.MODERATE.value
                    },
                    {
                        'name': 'research_priority',
                        'type': 'select',
                        'label': 'Research Importance',
                        'choices': get_enum_choices(ResearchPriority),
                        'default': ResearchPriority.MEDIUM.value
                    },
                    {
                        'name': 'writing_schedule',
                        'type': 'select',
                        'label': 'Preferred Writing Schedule',
                        'choices': get_enum_choices(WritingSchedule),
                        'default': WritingSchedule.DAILY.value
                    }
                ]
            }
        ],
        'validation_rules': validate_enum_combination,
        'recommendation_engine': get_genre_recommendations,
        'defaults': get_default_values()
    }


# ============================================================================
# Export Configuration for Frontend
# ============================================================================

def export_enum_config_for_frontend() -> Dict[str, any]:
    """Export all enum configurations for frontend JavaScript/TypeScript."""
    config = {
        'enums': {},
        'metadata': {},
        'recommendations': {},
        'ui_config': create_ui_form_config(),
        'validation_rules': {}
    }
    
    # Export all enums
    all_enums = [
        GenreType, SubGenre, StoryStructure, PlotType, ConflictType,
        CharacterRole, CharacterArchetype, CharacterArcType, PersonalityTrait,
        WorldType, MagicSystemType, TechnologyLevel, GovernmentType,
        WritingStyle, NarrativePOV, ToneType, PacingType,
        AgeGroup, ReadingLevel, AudienceType, ContentWarning,
        ResearchType, ResearchPriority, SourceType,
        PublicationRoute, BookFormat, BookLength,
        ProjectStatus, PriorityLevel, DifficultyLevel,
        AIAssistanceLevel, WritingTool, WritingSchedule
    ]
    
    for enum_class in all_enums:
        enum_name = enum_class.__name__
        config['enums'][enum_name] = {
            'choices': get_enum_choices(enum_class),
            'metadata': get_enum_with_metadata(enum_class)
        }
    
    # Export recommendation engine data
    for genre in GenreType:
        config['recommendations'][genre.value] = get_genre_recommendations(genre)
    
    return config


# Usage example and testing
if __name__ == "__main__":
    # Test enum functionality
    print("=== MuseQuill Enum System Test ===")
    
    # Test basic enum usage
    print(f"Available genres: {len(GenreType)}")
    print(f"Fantasy display name: {GenreType.FANTASY.display_name}")
    print(f"Is Fantasy fiction? {GenreType.FANTASY.is_fiction}")
    
    # Test structure metadata
    print(f"Three-Act description: {StoryStructure.THREE_ACT.description}")
    
    # Test word count ranges
    novel_range = BookLength.STANDARD_NOVEL.word_count_range
    print(f"Standard novel range: {novel_range[0]:,} - {novel_range[1]:,} words")
    
    # Test recommendations
    fantasy_recs = get_genre_recommendations(GenreType.FANTASY)
    print(f"Fantasy recommendations: {fantasy_recs}")
    
    # Test validation
    test_selections = {
        'genre': GenreType.CHILDREN.value,
        'age_group': AgeGroup.ADULT.value,
        'content_warnings': [ContentWarning.VIOLENCE.value]
    }
    warnings = validate_enum_combination(test_selections)
    print(f"Validation warnings: {warnings}")
    
    # Test UI config
    ui_config = create_ui_form_config()
    print(f"UI wizard steps: {len(ui_config['steps'])}")
    
    # Test frontend export
    frontend_config = export_enum_config_for_frontend()
    print(f"Frontend config keys: {list(frontend_config.keys())}")
    print(f"Total enums exported: {len(frontend_config['enums'])}")
    
    print("\n✅ All enum systems working correctly!")
    print("🎉 Ready for integration with MuseQuill.ink web interface!")