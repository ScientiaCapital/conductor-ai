"""
Storyboard Configuration
========================

Centralized configuration for storyboard generation.
Extracted from gemini_client.py for cleaner separation of concerns.
"""

from typing import Literal


# =============================================================================
# VALUE ANGLE INSTRUCTIONS
# =============================================================================
# How to frame value extraction based on audience persona

VALUE_ANGLE_INSTRUCTIONS: dict[str, str] = {
    "business_owner": """VALUE FRAMING: COI (Cost of Inaction)
Frame value as what they LOSE by not having this:
- Lost revenue, wasted time, missed opportunities
- Pain they continue to suffer without it
- "Every day without this costs you..."
""",
    "c_suite": """VALUE FRAMING: ROI (Return on Investment)
Frame value as what they GAIN from this:
- Measurable returns: dollars saved, hours reclaimed, % improvement
- Competitive advantage, scalability, data insights
- "This delivers X return on Y investment"
""",
    "btl_champion": """VALUE FRAMING: COI (Cost of Inaction)
Frame value as risk/pain of NOT having this:
- How it makes them look bad to leadership
- Daily frustrations that continue without it
- "Without this, you'll keep dealing with..."
""",
    "top_tier_vc": """VALUE FRAMING: ROI (Return on Investment)
Frame value as investment opportunity:
- Market size, growth potential, defensible moat
- Traction metrics, expansion potential
- "This represents X opportunity with Y traction"
""",
    "field_crew": """VALUE FRAMING: EASE (Simplicity)
Frame value as making their job EASIER:
- Less paperwork, faster completion, get home on time
- Simple, practical benefits they'll actually use
- "This means less hassle and faster work"
""",
}


# =============================================================================
# SECTION HEADERS
# =============================================================================
# Audience-specific section headers for storyboard layout

SECTION_HEADERS: dict[str, str] = {
    "field_crew": '"What It Does", "Who It\'s For", "Why It\'s Better", "How It Helps"',
    "business_owner": '"The Problem", "What You\'re Losing", "The Solution", "Who Benefits"',
    "c_suite": '"The Numbers", "The Impact", "The ROI", "Key Metric"',
    "btl_champion": '"Before", "After", "The Difference", "Who Uses It"',
    "top_tier_vc": '"The Opportunity", "The Solution", "The Moat", "Traction"',
}

DEFAULT_SECTION_HEADERS = '"Value Proposition", "Key Benefit", "Problem Solved", "For"'


# =============================================================================
# PERSONA EXTRACTION FOCUS
# =============================================================================
# What to look for when extracting content for each persona

PERSONA_EXTRACTION_FOCUS: dict[str, str] = {
    "business_owner": """FOCUS FOR BUSINESS OWNER:
- What PROFIT or REVENUE impact was discussed?
- What TIME savings would they get back (family time, less nights/weekends)?
- What HEADACHES would disappear?
- Did they mention competitors or falling behind?""",

    "c_suite": """FOCUS FOR C-SUITE EXECUTIVE:
- What ROI or METRICS were mentioned?
- What SCALABILITY or GROWTH enablement was discussed?
- What DATA or VISIBILITY improvements?
- What COMPETITIVE advantages?""",

    "btl_champion": """FOCUS FOR OPERATIONS/PROJECT MANAGER:
- What DAILY FRUSTRATIONS would be eliminated?
- What would make them LOOK GOOD to leadership?
- What COORDINATION problems were mentioned?
- What would their TEAM actually use?""",

    "top_tier_vc": """FOCUS FOR VC/INVESTOR:
- What MARKET SIZE indicators were mentioned?
- What TRACTION or GROWTH metrics?
- What MOAT or defensibility?
- What makes this a CATEGORY-DEFINING opportunity?""",

    "field_crew": """FOCUS FOR FIELD CREW:
- What would make their JOB EASIER?
- What PAPERWORK or HASSLE would disappear?
- What TOOLS would they actually use on the job site?
- Keep it SIMPLE - 5th grade vocabulary.""",
}


# =============================================================================
# VISUAL STYLE INSTRUCTIONS
# =============================================================================
# How to style the generated storyboard image

VISUAL_STYLE_INSTRUCTIONS: dict[str, str] = {
    "clean": """VISUAL STYLE: CLEAN
- Simple flat icons and shapes
- Minimal decoration, maximum clarity
- Bold typography, lots of whitespace
- No gradients or shadows
- Think: Apple keynote slides""",

    "polished": """VISUAL STYLE: POLISHED PROFESSIONAL
- Refined, corporate-quality graphics
- Subtle gradients and modern touches
- Professional iconography
- Balanced composition with visual hierarchy
- Think: McKinsey or BCG presentation""",

    "photo_realistic": """VISUAL STYLE: PHOTO-REALISTIC
- Include realistic imagery and photos
- High-quality stock photo aesthetic
- Blend photos with text overlays
- Modern editorial feel
- Think: LinkedIn featured image or magazine layout""",

    "minimalist": """VISUAL STYLE: MINIMALIST
- Extreme simplicity, sparse elements
- Maximum whitespace
- Only essential text and icons
- Single accent color usage
- Think: Japanese design or Dieter Rams""",
}


# =============================================================================
# ARTIST STYLE INSTRUCTIONS
# =============================================================================
# Fun artistic variations for storyboard generation

ARTIST_STYLE_INSTRUCTIONS: dict[str, str] = {
    "salvador_dali": """ARTIST STYLE: SALVADOR DALI
- Surrealist elements and dreamlike quality
- Melting or distorted shapes (but keep text readable!)
- Unexpected juxtapositions
- Rich, warm colors with dramatic lighting
- Imaginative, thought-provoking visuals
- Think: The Persistence of Memory meets corporate presentation""",

    "monet": """ARTIST STYLE: CLAUDE MONET
- Impressionist brushstroke texture
- Soft, diffused lighting
- Pastel and natural color palette
- Dreamy, atmospheric quality
- Nature-inspired elements (water lilies, gardens)
- Think: Water Lilies meets executive summary""",

    "diego_rivera": """ARTIST STYLE: DIEGO RIVERA
- Bold muralist style
- Strong, blocky shapes and forms
- Workers and industry themes
- Rich earth tones and vibrant accents
- Social realism aesthetic
- Think: Detroit Industry Murals meets tech infographic""",

    "warhol": """ARTIST STYLE: ANDY WARHOL
- Pop art boldness
- High contrast, vibrant colors
- Repetition and pattern elements
- Commercial art aesthetic
- Bold outlines and flat colors
- Think: Campbell's Soup meets business presentation""",

    "van_gogh": """ARTIST STYLE: VAN GOGH
- Expressive brushstroke texture
- Swirling, dynamic movement
- Bold, emotional color choices
- Starry Night energy
- Intense yellows, blues, and greens
- Think: Starry Night meets executive dashboard""",

    "picasso": """ARTIST STYLE: PICASSO (CUBIST)
- Geometric, fragmented forms
- Multiple perspectives simultaneously
- Bold, angular shapes
- Strong black outlines
- Analytical cubism meets business graphics
- Think: Three Musicians meets corporate storyboard""",

    "kurt_geiger": """ARTIST STYLE: KURT GEIGER
- Bold, glamorous maximalist aesthetic
- Rich jewel tones: deep purples, emerald greens, hot pinks, gold
- Crystal/gem embellishments and sparkle effects
- Luxurious textures and metallic accents
- Fashion-forward, statement-making visuals
- High contrast with dramatic lighting
- Think: London luxury meets bold self-expression""",
}


# =============================================================================
# FORMAT INSTRUCTIONS
# =============================================================================
# Layout and output instructions based on format

FORMAT_LAYOUT_INSTRUCTIONS: dict[str, str] = {
    "storyboard": """LAYOUT (VERTICAL STORYBOARD):
- PORTRAIT orientation - tall, scrollable format
- Visual flow from TOP TO BOTTOM (vertical reading)
- Multiple sections stacked vertically
- Each section tells part of the story
- Good for detailed explanations and step-by-step narratives
- Think: LinkedIn article header or presentation slide deck feel""",

    "infographic": """LAYOUT (HORIZONTAL INFOGRAPHIC):
- LANDSCAPE orientation - wide, single-view format
- Visual flow from LEFT TO RIGHT (horizontal reading)
- Clean, scannable, executive-friendly
- Key points visible at a glance
- Good for quick value communication
- Think: LinkedIn post image or email header""",
}

FORMAT_OUTPUT_INSTRUCTIONS: dict[str, str] = {
    "storyboard": """OUTPUT:
- Single image, PORTRAIT 9:16 aspect ratio (vertical)
- 1080x1920 resolution (mobile/story format)
- PNG format""",

    "infographic": """OUTPUT:
- Single image, LANDSCAPE 16:9 aspect ratio (widescreen horizontal)
- 1920x1080 resolution (HD widescreen)
- PNG format""",
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_value_angle_instruction(audience: str) -> str:
    """Get value angle framing instruction for extraction based on audience."""
    return VALUE_ANGLE_INSTRUCTIONS.get(audience, VALUE_ANGLE_INSTRUCTIONS["c_suite"])


def get_section_headers(audience: str) -> str:
    """Get audience-specific section headers for storyboard layout."""
    return SECTION_HEADERS.get(audience, DEFAULT_SECTION_HEADERS)


def get_persona_extraction_focus(audience: str) -> str:
    """Get persona-specific extraction instructions."""
    return PERSONA_EXTRACTION_FOCUS.get(audience, PERSONA_EXTRACTION_FOCUS["c_suite"])


def get_visual_style_instructions(visual_style: str) -> str:
    """Get visual style instructions based on style preference."""
    return VISUAL_STYLE_INSTRUCTIONS.get(visual_style, VISUAL_STYLE_INSTRUCTIONS["polished"])


def get_artist_style_instructions(artist_style: str | None) -> str:
    """Get artist style instructions for fun variations."""
    if not artist_style:
        return ""
    return ARTIST_STYLE_INSTRUCTIONS.get(artist_style, "")


def get_format_layout_instructions(output_format: str) -> str:
    """Get layout instructions based on output format."""
    return FORMAT_LAYOUT_INSTRUCTIONS.get(output_format, FORMAT_LAYOUT_INSTRUCTIONS["infographic"])


def get_format_output_instructions(output_format: str) -> str:
    """Get output specifications based on format."""
    return FORMAT_OUTPUT_INSTRUCTIONS.get(output_format, FORMAT_OUTPUT_INSTRUCTIONS["infographic"])
