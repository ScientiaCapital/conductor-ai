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
    vision_model: str = "gemini-2.0-flash"  # For understanding
    image_model: str = "gemini-3-pro-image-preview"  # For generating storyboard images
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

    async def generate_storyboard(
        self,
        understanding: StoryboardUnderstanding,
        stage: str = "preview",
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
            icp_preset: Optional ICP preset for visual style
            custom_style: Optional custom style overrides

        Returns:
            PNG image bytes
        """
        self._ensure_client()

        from src.tools.storyboard.coperniq_presets import COPERNIQ_ICP, get_stage_template

        if icp_preset is None:
            icp_preset = COPERNIQ_ICP

        stage_template = get_stage_template(stage)
        visual_style = icp_preset.get("visual_style", {})

        # Build the image generation prompt
        prompt = f"""Create a professional one-page executive storyboard infographic.

CONTENT TO DISPLAY:
- Badge: "{stage_template['badge']}"
- Headline: "{understanding.headline}"
- Description: "{understanding.what_it_does}"
- Value Proposition: "{understanding.business_value}"
- For: "{understanding.who_benefits}"
- Key Benefit: "{understanding.differentiator}"
- Problem Solved: "{understanding.pain_point_addressed}"
- Call to Action: "{stage_template['cta']}"

VISUAL REQUIREMENTS:
- Style: {stage_template['visual_style']}
- Color scheme: Professional blue and white ({', '.join(visual_style.get('colors', ['#1E40AF', '#FFFFFF']))})
- Layout: Clean infographic with clear visual hierarchy
- Include a simple icon representing: {understanding.suggested_icon}
- Large, readable text (executive-friendly)
- Visual flow from top to bottom
- Include subtle badge/ribbon showing "{stage_template['badge']}"

DESIGN PRINCIPLES:
- Professional but approachable (not corporate-stuffy)
- Clean white space
- Icons should be simple and metaphorical
- No stock photo feel
- Ready to attach to a professional email
- Should look great printed or on screen

OUTPUT:
- Single image, portrait orientation (like a one-pager)
- Resolution suitable for email attachment
- PNG format"""

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
