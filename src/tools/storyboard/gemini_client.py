"""
Gemini Storyboard Client
=========================

Shared client for Gemini Vision (understanding) and Image Generation (creating).
Implements the two-stage pipeline:
1. UNDERSTAND - Analyze code/images, extract business value
2. GENERATE - Create beautiful PNG storyboards

Uses Gemini 2.5 Flash for both stages.
NO OpenAI - Gemini only.
"""

import os
import json
import base64
import logging
from typing import Any
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class StoryboardUnderstanding(BaseModel):
    """Extracted understanding from code/roadmap analysis."""

    headline: str = Field(..., description="Catchy, benefit-focused headline (8 words max)")
    what_it_does: str = Field(..., description="Plain English description (2 sentences max)")
    business_value: str = Field(..., description="Quantified benefit (hours saved, % improvement)")
    who_benefits: str = Field(..., description="Target persona description")
    differentiator: str = Field(..., description="What makes this special (1 sentence)")
    pain_point_addressed: str = Field(..., description="The problem this solves")
    suggested_icon: str = Field(default="clipboard-check", description="Icon suggestion for visual")


@dataclass
class GeminiConfig:
    """Configuration for Gemini client."""

    api_key: str | None = None
    vision_model: str = "models/gemini-2.0-flash"  # For understanding
    image_model: str = "models/gemini-3-pro-image-preview"  # For generating storyboard images (Nano Banana)
    timeout: int = 60
    max_retries: int = 3

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv("GOOGLE_API_KEY")


class GeminiStoryboardClient:
    """
    Client for Gemini Vision + Image Generation.

    Two-stage pipeline:
    1. understand_code() / understand_image() - Extract business value
    2. generate_storyboard() - Create beautiful PNG

    Example:
        client = GeminiStoryboardClient()

        # Stage 1: Understand
        understanding = await client.understand_code(
            code_content="def calculate_roi(): ...",
            icp_preset=COPERNIQ_ICP,
            audience="c_suite",
        )

        # Stage 2: Generate
        png_bytes = await client.generate_storyboard(
            understanding=understanding,
            stage="preview",
        )
    """

    def __init__(self, config: GeminiConfig | None = None):
        """
        Initialize Gemini client.

        Args:
            config: Optional GeminiConfig (uses env vars if not provided)
        """
        self.config = config or GeminiConfig()
        self._client = None
        self._initialized = False

    def _ensure_client(self):
        """Lazy initialization of Gemini client."""
        if self._initialized:
            return

        if not self.config.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        try:
            from google import genai

            self._client = genai.Client(api_key=self.config.api_key)
            self._initialized = True
            logger.info("[GEMINI] Client initialized successfully")
        except ImportError:
            raise ImportError(
                "google-genai package not installed. "
                "Install with: pip install google-genai"
            )

    async def understand_code(
        self,
        code_content: str,
        icp_preset: dict[str, Any],
        audience: str = "c_suite",
        file_name: str | None = None,
    ) -> StoryboardUnderstanding:
        """
        Stage 1: Analyze code and extract business value.

        Uses Gemini Vision to "read" the code and extract:
        - What it does (plain English)
        - Business value proposition
        - Who benefits (ICP alignment)
        - Key differentiator

        Args:
            code_content: Source code as string
            icp_preset: ICP configuration dictionary
            audience: Target audience persona
            file_name: Optional file name for context

        Returns:
            StoryboardUnderstanding with extracted insights
        """
        self._ensure_client()

        # Build the analysis prompt
        language_guidelines = self._build_language_guidelines(icp_preset)
        audience_info = icp_preset.get("audience_personas", {}).get(audience, {})

        prompt = f"""Analyze this code file and extract business value for a {icp_preset.get('target', 'business')} audience.

{f"File: {file_name}" if file_name else ""}

CODE:
```
{code_content[:8000]}  # Truncate very long files
```

TARGET AUDIENCE: {audience_info.get('title', audience)}
They care about: {', '.join(audience_info.get('cares_about', ['efficiency', 'results']))}

{language_guidelines}

EXTRACTION REQUIREMENTS:
1. headline: Catchy, benefit-focused headline (8 words max). NO technical terms.
2. what_it_does: Plain English description (2 sentences max). Like explaining to a friend.
3. business_value: Quantified benefit. Use real numbers if possible (hours saved, % improvement).
4. who_benefits: Who in the organization benefits most from this.
5. differentiator: What makes this special compared to doing it manually.
6. pain_point_addressed: The specific problem/frustration this eliminates.
7. suggested_icon: A simple icon name that represents this (e.g., "clock", "dollar", "team").

CRITICAL RULES:
- NO technical jargon (no API, database, async, etc.)
- NO code details (no class names, function names, etc.)
- NO proprietary information (no internal URLs, pricing, etc.)
- Write like you're explaining to a smart 5th grader
- Focus on BENEFITS not FEATURES
- If a competitor could use it to copy us, DON'T include it

Return ONLY valid JSON matching this exact structure:
{{
    "headline": "...",
    "what_it_does": "...",
    "business_value": "...",
    "who_benefits": "...",
    "differentiator": "...",
    "pain_point_addressed": "...",
    "suggested_icon": "..."
}}"""

        try:
            response = self._client.models.generate_content(
                model=self.config.vision_model,
                contents=prompt,
            )

            # Parse JSON response
            json_str = response.text.strip()
            # Handle markdown code blocks
            if json_str.startswith("```"):
                json_str = json_str.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
                json_str = json_str.strip()

            data = json.loads(json_str)
            return StoryboardUnderstanding(**data)

        except json.JSONDecodeError as e:
            logger.error(f"[GEMINI] Failed to parse response: {e}")
            # Return a safe default
            return StoryboardUnderstanding(
                headline="New Feature Coming Soon",
                what_it_does="A powerful new capability that makes your work easier.",
                business_value="Save time and reduce errors.",
                who_benefits="Your entire team",
                differentiator="Built specifically for contractors like you.",
                pain_point_addressed="Manual processes that waste your time.",
                suggested_icon="star",
            )
        except Exception as e:
            logger.error(f"[GEMINI] Understanding failed: {e}")
            raise

    async def understand_image(
        self,
        image_data: bytes | str,
        icp_preset: dict[str, Any],
        audience: str = "c_suite",
        sanitize_ip: bool = True,
    ) -> StoryboardUnderstanding:
        """
        Stage 1: Analyze image (Miro screenshot, roadmap) and extract business value.

        Uses Gemini Vision to analyze visual content and extract sanitized insights.
        Extra sanitization for IP protection when analyzing roadmaps.

        Args:
            image_data: Image bytes or base64 string
            icp_preset: ICP configuration dictionary
            audience: Target audience persona
            sanitize_ip: Whether to apply extra IP sanitization

        Returns:
            StoryboardUnderstanding with extracted insights
        """
        self._ensure_client()

        # Handle base64 string input
        if isinstance(image_data, str):
            if image_data.startswith("data:"):
                # Remove data URL prefix
                image_data = image_data.split(",")[1]
            image_bytes = base64.b64decode(image_data)
        else:
            image_bytes = image_data

        # Build the analysis prompt
        language_guidelines = self._build_language_guidelines(icp_preset)
        audience_info = icp_preset.get("audience_personas", {}).get(audience, {})

        sanitization_rules = ""
        if sanitize_ip:
            sanitization_rules = """
CRITICAL IP SANITIZATION:
- DO NOT include any text visible in the image verbatim
- DO NOT mention specific feature names, product names, or project codes
- DO NOT reference dates, timelines, or milestones specifically
- Transform all specifics into general themes
- If you see "Q1 2025: Launch Feature X" → say "Exciting capabilities coming soon"
"""

        prompt = f"""Analyze this roadmap/planning image and extract a "Coming Soon" teaser for a {icp_preset.get('target', 'business')} audience.

TARGET AUDIENCE: {audience_info.get('title', audience)}
They care about: {', '.join(audience_info.get('cares_about', ['efficiency', 'results']))}

{language_guidelines}

{sanitization_rules}

EXTRACTION REQUIREMENTS (create an EXCITING TEASER, not a summary):
1. headline: Teaser headline that builds excitement (8 words max). NO specifics.
2. what_it_does: Vague but exciting description of what's coming. NO details.
3. business_value: Promise of future value. Be aspirational.
4. who_benefits: Who will love this when it arrives.
5. differentiator: Why they should be excited (without specifics).
6. pain_point_addressed: The problem that will be solved.
7. suggested_icon: Icon representing innovation/future (e.g., "rocket", "lightbulb").

Return ONLY valid JSON matching this exact structure:
{{
    "headline": "...",
    "what_it_does": "...",
    "business_value": "...",
    "who_benefits": "...",
    "differentiator": "...",
    "pain_point_addressed": "...",
    "suggested_icon": "..."
}}"""

        try:
            # Create image part for multimodal request
            from google.genai import types

            image_part = types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/png",  # Assume PNG, could detect
            )

            response = self._client.models.generate_content(
                model=self.config.vision_model,
                contents=[image_part, prompt],
            )

            # Parse JSON response
            json_str = response.text.strip()
            if json_str.startswith("```"):
                json_str = json_str.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
                json_str = json_str.strip()

            data = json.loads(json_str)
            return StoryboardUnderstanding(**data)

        except Exception as e:
            logger.error(f"[GEMINI] Image understanding failed: {e}")
            raise

    async def understand_multiple_images(
        self,
        images_data: list[bytes],
        icp_preset: dict[str, Any],
        audience: str = "c_suite",
        sanitize_ip: bool = True,
    ) -> StoryboardUnderstanding:
        """
        Stage 1: Analyze multiple images together and extract combined business value.

        Useful for combining CTO roadmap + Miro screenshots + marketing materials
        into a single cohesive storyboard.

        Args:
            images_data: List of image bytes (up to 3 images)
            icp_preset: ICP configuration dictionary
            audience: Target audience persona
            sanitize_ip: Whether to apply extra IP sanitization

        Returns:
            StoryboardUnderstanding with combined insights from all images
        """
        self._ensure_client()

        if len(images_data) > 3:
            logger.warning(f"Received {len(images_data)} images, using first 3 only")
            images_data = images_data[:3]

        # Build the analysis prompt for multiple images
        language_guidelines = self._build_language_guidelines(icp_preset)
        audience_info = icp_preset.get("audience_personas", {}).get(audience, {})

        sanitization_rules = ""
        if sanitize_ip:
            sanitization_rules = """
CRITICAL IP SANITIZATION:
- DO NOT include any text visible in the images verbatim
- DO NOT mention specific feature names, product names, or project codes
- DO NOT reference dates, timelines, or milestones specifically
- Transform all specifics into general themes
"""

        prompt = f"""Analyze these {len(images_data)} images TOGETHER and synthesize a UNIFIED business value message.

The images may include:
- CTO roadmap or planning documents
- Miro boards or whiteboard screenshots
- Marketing materials or campaign visuals
- Product screenshots or demos

Combine insights from ALL images into ONE cohesive storyboard message.

TARGET AUDIENCE: {audience_info.get('title', audience)}
They care about: {', '.join(audience_info.get('cares_about', ['efficiency', 'results']))}

{language_guidelines}

{sanitization_rules}

EXTRACTION REQUIREMENTS (synthesize from ALL images):
1. headline: Catchy, benefit-focused headline (8 words max). Capture the overall theme.
2. what_it_does: Plain English description (2 sentences max). Combine key concepts from all images.
3. business_value: Quantified benefit. Use real numbers if visible in any image.
4. who_benefits: Who in the organization benefits most.
5. differentiator: What makes this special - look across all images for unique value.
6. pain_point_addressed: The specific problem/frustration this eliminates.
7. suggested_icon: A simple icon name that represents the overall theme.

CRITICAL: Create ONE unified message, not separate descriptions for each image.

Return ONLY valid JSON matching this exact structure:
{{
    "headline": "...",
    "what_it_does": "...",
    "business_value": "...",
    "who_benefits": "...",
    "differentiator": "...",
    "pain_point_addressed": "...",
    "suggested_icon": "..."
}}"""

        try:
            from google.genai import types

            # Create image parts for all images
            content_parts = []
            for i, img_bytes in enumerate(images_data):
                image_part = types.Part.from_bytes(
                    data=img_bytes,
                    mime_type="image/png",
                )
                content_parts.append(image_part)
                logger.info(f"[GEMINI] Added image {i+1}/{len(images_data)} to multi-image request")

            # Add the prompt at the end
            content_parts.append(prompt)

            response = self._client.models.generate_content(
                model=self.config.vision_model,
                contents=content_parts,
            )

            # Parse JSON response
            json_str = response.text.strip()
            if json_str.startswith("```"):
                json_str = json_str.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
                json_str = json_str.strip()

            data = json.loads(json_str)
            logger.info(f"[GEMINI] Successfully understood {len(images_data)} images together")
            return StoryboardUnderstanding(**data)

        except Exception as e:
            logger.error(f"[GEMINI] Multi-image understanding failed: {e}")
            raise

    async def generate_storyboard(
        self,
        understanding: StoryboardUnderstanding,
        stage: str = "preview",
        audience: str = "c_suite",
        output_format: str = "infographic",
        visual_style: str = "polished",
        artist_style: str | None = None,
        icp_preset: dict[str, Any] | None = None,
        custom_style: dict[str, Any] | None = None,
    ) -> bytes:
        """
        Stage 2: Generate beautiful PNG storyboard.

        Uses Gemini Image Generation to create a professional one-page
        executive storyboard ready for email attachment.

        Args:
            understanding: StoryboardUnderstanding from Stage 1
            stage: "preview", "demo", or "shipped" (affects visual style)
            audience: Target audience (top_tier_vc uses different structure)
            output_format: "infographic" (horizontal 16:9) or "storyboard" (vertical, detailed)
            visual_style: "clean", "polished", "photo_realistic", or "minimalist"
            icp_preset: Optional ICP preset for visual style
            custom_style: Optional custom style overrides

        Returns:
            PNG image bytes
        """
        self._ensure_client()

        from src.tools.storyboard.coperniq_presets import (
            COPERNIQ_ICP,
            COPERNIQ_BRAND,
            get_stage_template,
            get_audience_persona,
        )

        if icp_preset is None:
            icp_preset = COPERNIQ_ICP

        import uuid
        from datetime import datetime

        stage_template = get_stage_template(stage)
        visual_style_config = icp_preset.get("visual_style", {})
        brand = COPERNIQ_BRAND
        persona = get_audience_persona(audience, icp_preset)

        # Add uniqueness to avoid cached/repetitive outputs
        unique_seed = f"{datetime.now().isoformat()}-{uuid.uuid4().hex[:8]}"

        # Build audience-specific content structure
        if audience == "top_tier_vc":
            # VC/Investor storyboard - use investment thesis structure
            vc_structure = persona.get("storyboard_structure", {})
            proof = brand.get("proof_points", {})

            content_section = f"""CONTENT TO DISPLAY (VC/INVESTOR PITCH - NOT A CUSTOMER DEMO):
- HEADLINE: "{understanding.headline}"

SECTIONS (LEFT TO RIGHT FLOW):
1. THE PROBLEM: "{understanding.pain_point_addressed}"
2. THE SOLUTION: "{understanding.what_it_does}"
3. TRACTION: Use these proof points: {proof.get('completion_rate', '99% completion')} | {proof.get('payment_speed', '65% faster payments')} | {proof.get('scale_story', 'scaled 5x without adding staff')}
4. MARKET: $200B TAM → $40B SAM (MEP+Energy contractors) → $2B SOM
5. WHY NOW: AI inflection + workforce shortage + regulatory pressure
6. MOAT: "{understanding.differentiator}"

CRITICAL: NO "Book a demo" or customer CTAs. This is for INVESTORS.
- Use metrics and numbers prominently
- Show market opportunity, not product features
- Confidence and data, not sales pitch"""
        else:
            # Customer-focused storyboard (sales, internal, field crew)
            content_section = f"""CONTENT TO DISPLAY:
- Badge: "{stage_template['badge']}"
- Headline: "{understanding.headline}"
- Description: "{understanding.what_it_does}"
- Value Proposition: "{understanding.business_value}"
- For: "{understanding.who_benefits}"
- Key Benefit: "{understanding.differentiator}"
- Problem Solved: "{understanding.pain_point_addressed}"
- Call to Action: "{stage_template['cta']}"

TARGET AUDIENCE: {persona.get('title', 'Business Professional')}
TONE: {persona.get('tone', 'Professional and friendly')}"""

        # Build the image generation prompt
        prompt = f"""Create a UNIQUE professional one-page executive storyboard infographic.

GENERATION SEED: {unique_seed} (use this to create variation in layout and icons)

BRAND: {brand['company']} - "{brand['tagline']}"

{content_section}

VISUAL REQUIREMENTS:
- Style: {stage_template.get('visual_style', 'Modern professional')}
- Color scheme: {brand['company']} brand colors (MUST USE THESE EXACT COLORS):
  - Primary (CTAs/headers): {visual_style_config.get('primary_color', '#23433E')} (dark teal/forest green)
  - Accent (highlights/emphasis): {visual_style_config.get('accent_color', '#2D9688')} (teal)
  - Text: {visual_style_config.get('text_color', '#333333')} (dark gray)
  - Background: {visual_style_config.get('hero_bg', '#DDEDEB')} (light mint/sage green)
- NO badges, ribbons, or "demo/preview/coming soon" labels - keep it clean and professional
- Include simple icons representing the content (construction/business metaphors)
- Large, readable text (executive-friendly)

TEXT ACCURACY REQUIREMENTS (CRITICAL - DO NOT IGNORE):
- ONLY use the EXACT text provided in the content section above - DO NOT invent or modify words
- Every single word must be spelled correctly - double-check spelling
- Use LARGE fonts (minimum 18pt equivalent) - small text gets garbled
- If you cannot render text clearly, use fewer words or icons instead
- NEVER include random letters or gibberish text
- Section headers: Use ONLY these exact words: "Value Proposition", "Key Benefit", "Problem Solved", "For"
- Keep descriptions SHORT (under 15 words per section) to ensure clarity

{self._get_format_layout_instructions(output_format)}

{self._get_visual_style_instructions(visual_style)}

{self._get_artist_style_instructions(artist_style) if artist_style else ""}

DESIGN PRINCIPLES:
- {visual_style_config.get('aesthetic', 'Modern, professional, teal/green palette. Corporate but approachable.')}
- Light mint/sage backgrounds with clean white sections
- Icons should be simple and metaphorical (tools, buildings, charts)
- Ready to share in presentations, emails, LinkedIn, or Slack
- CRITICAL: Use teal/green color palette, NOT orange or blue
- NO promotional badges or ribbons - this is executive content, not a sales flyer

{self._get_format_output_instructions(output_format)}"""

        try:
            from google.genai import types

            response = self._client.models.generate_content(
                model=self.config.image_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                ),
            )

            # Extract image from response
            for part in response.candidates[0].content.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    return part.inline_data.data

            raise ValueError("No image generated in response")

        except Exception as e:
            logger.error(f"[GEMINI] Image generation failed: {e}")
            raise

    def _build_language_guidelines(self, icp_preset: dict[str, Any]) -> str:
        """Build language guidelines string for prompts."""
        avoid = icp_preset.get("language_style", {}).get("avoid", [])
        use = icp_preset.get("language_style", {}).get("use", [])
        tone = icp_preset.get("tone", "Friendly and professional")

        return f"""LANGUAGE GUIDELINES:
- Tone: {tone}
- AVOID these words/phrases: {', '.join(avoid[:10])}
- USE these words/phrases: {', '.join(use[:10])}
- Write for someone with no technical background
- Focus on benefits, not features"""

    def _get_format_layout_instructions(self, output_format: str) -> str:
        """Get layout instructions based on output format."""
        if output_format == "storyboard":
            return """LAYOUT (VERTICAL STORYBOARD):
- PORTRAIT orientation - tall, scrollable format
- Visual flow from TOP TO BOTTOM (vertical reading)
- Multiple sections stacked vertically
- Each section tells part of the story
- Good for detailed explanations and step-by-step narratives
- Think: LinkedIn article header or presentation slide deck feel"""
        else:  # infographic (default)
            return """LAYOUT (HORIZONTAL INFOGRAPHIC):
- LANDSCAPE orientation - wide, single-view format
- Visual flow from LEFT TO RIGHT (horizontal reading)
- Clean, scannable, executive-friendly
- Key points visible at a glance
- Good for quick value communication
- Think: LinkedIn post image or email header"""

    def _get_format_output_instructions(self, output_format: str) -> str:
        """Get output specifications based on format."""
        if output_format == "storyboard":
            return """OUTPUT:
- Single image, PORTRAIT 9:16 aspect ratio (vertical)
- 1080x1920 resolution (mobile/story format)
- PNG format"""
        else:  # infographic (default)
            return """OUTPUT:
- Single image, LANDSCAPE 16:9 aspect ratio (widescreen horizontal)
- 1920x1080 resolution (HD widescreen)
- PNG format"""

    def _get_visual_style_instructions(self, visual_style: str) -> str:
        """Get visual style instructions based on style preference."""
        styles = {
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
        return styles.get(visual_style, styles["polished"])

    def _get_artist_style_instructions(self, artist_style: str | None) -> str:
        """Get artist style instructions for fun variations."""
        if not artist_style:
            return ""

        artists = {
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
        }
        return artists.get(artist_style, "")

    async def health_check(self) -> dict[str, Any]:
        """
        Check if Gemini client is properly configured.

        Returns:
            Health status dictionary
        """
        try:
            self._ensure_client()
            return {
                "status": "healthy",
                "vision_model": self.config.vision_model,
                "image_model": self.config.image_model,
                "api_key_configured": bool(self.config.api_key),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "api_key_configured": bool(self.config.api_key),
            }
