"""
Storyboard Client
==================

Multi-model client for understanding (vision) and image generation.
Implements the two-stage pipeline:
1. UNDERSTAND - Analyze code/images, extract business value (Gemini or Qwen via OpenRouter)
2. GENERATE - Create beautiful PNG storyboards (Gemini only)

Vision model options:
- gemini: Gemini 2.0 Flash (default)
- qwen: Qwen 2.5 VL 72B via OpenRouter (better for complex documents)

NO OpenAI - Gemini + Chinese VLMs only.
"""

import os
import json
import base64
import logging
import httpx
from typing import Any, Literal
from dataclasses import dataclass

from pydantic import BaseModel, Field

from src.tools.storyboard.storyboard_config import (
    get_value_angle_instruction,
    get_section_headers,
    get_persona_extraction_focus,
    get_visual_style_instructions,
    get_artist_style_instructions,
    get_format_layout_instructions,
    get_format_output_instructions,
)

logger = logging.getLogger(__name__)


# =============================================================================
# JSON UTILITIES
# =============================================================================

def _repair_json(json_str: str) -> str:
    """
    Attempt to repair truncated or malformed JSON.

    Common issues from LLM responses:
    - Unterminated strings
    - Missing closing braces
    - Trailing commas
    """
    import re

    # Remove markdown code blocks
    if json_str.startswith("```"):
        parts = json_str.split("```")
        if len(parts) >= 2:
            json_str = parts[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
        json_str = json_str.strip()

    # Try parsing as-is first
    try:
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError:
        pass

    # Fix unterminated strings by closing them
    quote_count = json_str.count('"') - json_str.count('\\"')
    if quote_count % 2 == 1:
        json_str = json_str + '"'

    # Add missing closing braces
    open_braces = json_str.count('{') - json_str.count('}')
    if open_braces > 0:
        json_str = json_str + '}' * open_braces

    # Remove trailing commas before closing braces
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)

    return json_str


def _safe_parse_understanding(
    response_text: str,
    source: str = "unknown",
) -> "StoryboardUnderstanding":
    """
    Safely parse LLM response to StoryboardUnderstanding.

    Returns error state instead of raising on parse failure.
    """
    try:
        json_str = _repair_json(response_text.strip())
        data = json.loads(json_str)
        return StoryboardUnderstanding(**data)
    except json.JSONDecodeError as e:
        logger.error(f"[UNDERSTAND] Failed to parse {source} response: {e}")
        logger.error(f"[UNDERSTAND] Raw response was: {response_text[:500] if response_text else 'None'}")
        return StoryboardUnderstanding(
            headline="EXTRACTION FAILED - Check Input",
            tagline="Could not extract content",
            what_it_does="The AI returned malformed data. Try again or use a different input.",
            business_value="Unable to determine - extraction failed",
            who_benefits="Unable to determine - extraction failed",
            differentiator="Unable to determine - extraction failed",
            pain_point_addressed="Unable to determine - extraction failed",
            suggested_icon="alert-triangle",
            raw_extracted_text=f"PARSE ERROR ({source}): {str(e)[:200]}",
            extraction_confidence=0.0,
        )
    except Exception as e:
        logger.error(f"[UNDERSTAND] Unexpected error parsing {source}: {e}")
        return StoryboardUnderstanding(
            headline="EXTRACTION FAILED - Unexpected Error",
            tagline="Something went wrong",
            what_it_does=f"Error: {str(e)[:100]}",
            business_value="Unable to determine",
            who_benefits="Unable to determine",
            differentiator="Unable to determine",
            pain_point_addressed="Unable to determine",
            suggested_icon="alert-triangle",
            raw_extracted_text=f"ERROR ({source}): {str(e)[:300]}",
            extraction_confidence=0.0,
        )


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

VisionModel = Literal["gemini", "qwen"]
TextModel = Literal["gemini", "deepseek"]


class StoryboardUnderstanding(BaseModel):
    """Extracted understanding from code/roadmap analysis."""

    headline: str = Field(..., description="Catchy, benefit-focused headline (8 words max)")
    tagline: str = Field(
        default="",
        description="Dynamic tagline specific to content and persona (10 words max) - MUST be unique to this content"
    )
    what_it_does: str = Field(..., description="Plain English description (2 sentences max)")
    business_value: str = Field(..., description="Quantified benefit (hours saved, % improvement)")
    who_benefits: str = Field(..., description="Target persona description")
    differentiator: str = Field(..., description="What makes this special (1 sentence)")
    pain_point_addressed: str = Field(..., description="The problem this solves")
    suggested_icon: str = Field(default="clipboard-check", description="Icon suggestion for visual")
    raw_extracted_text: str = Field(
        default="",
        description="Verbatim text/features extracted from input (for debugging/verification)"
    )
    extraction_confidence: float = Field(
        default=1.0,
        description="Confidence score 0-1. Below 0.7 = flag for review"
    )


@dataclass
class GeminiConfig:
    """Configuration for storyboard client."""

    api_key: str | None = None
    openrouter_api_key: str | None = None

    # Model routing
    vision_provider: VisionModel = "qwen"
    text_provider: TextModel = "deepseek"

    # Refinement settings
    enable_refinement: bool = True
    refinement_threshold: float = 0.75

    # Model identifiers
    gemini_vision_model: str = "models/gemini-2.0-flash"
    qwen_model: str = "qwen/qwen2.5-vl-72b-instruct"
    deepseek_model: str = "deepseek/deepseek-chat"
    image_model: str = "models/gemini-3-pro-image-preview"

    timeout: int = 90
    max_retries: int = 3

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv("GOOGLE_API_KEY")
        if self.openrouter_api_key is None:
            self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")


# =============================================================================
# MAIN CLIENT
# =============================================================================

class GeminiStoryboardClient:
    """
    Client for Gemini Vision + Image Generation.

    Three-stage intelligent pipeline:
    1. EXTRACT - Primary model extracts (DeepSeek for text, Qwen for images)
    2. REFINE - If confidence < threshold, alternate model validates/improves
    3. GENERATE - Gemini creates the image (only model that can generate)
    """

    def __init__(self, config: GeminiConfig | None = None):
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

    # =========================================================================
    # PROVIDER CALLS
    # =========================================================================

    async def _call_openrouter_with_retry(
        self,
        payload: dict,
        max_retries: int = 3,
    ) -> str:
        """Call OpenRouter API with retry logic for rate limits."""
        import asyncio

        if not self.config.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")

        headers = {
            "Authorization": f"Bearer {self.config.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://coperniq.io",
            "X-Title": "Coperniq Storyboard Generator",
        }

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                    response = await client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        json=payload,
                        headers=headers,
                    )

                    if response.status_code == 429:
                        wait_time = (attempt + 1) * 5
                        logger.warning(f"[OPENROUTER] Rate limited, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue

                    response.raise_for_status()
                    data = response.json()
                    return data["choices"][0]["message"]["content"]

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    logger.warning(f"[OPENROUTER] Rate limited, waiting {wait_time}s")
                    await asyncio.sleep(wait_time)
                    continue
                raise

        raise Exception("Max retries exceeded for OpenRouter API")

    async def _call_qwen_vision(
        self,
        prompt: str,
        image_data: bytes | None = None,
        images_data: list[bytes] | None = None,
    ) -> str:
        """Call Qwen VL via OpenRouter for vision understanding."""
        content = []

        if images_data:
            for img_bytes in images_data[:3]:
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                })
        elif image_data:
            img_b64 = base64.b64encode(image_data).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"}
            })

        content.append({"type": "text", "text": prompt})

        payload = {
            "model": self.config.qwen_model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 4096,
            "temperature": 0.5,
        }

        logger.info(f"[QWEN] Calling {self.config.qwen_model} via OpenRouter")
        result = await self._call_openrouter_with_retry(payload)
        logger.info(f"[QWEN] Response received ({len(result)} chars)")
        return result

    async def _call_deepseek(self, prompt: str) -> str:
        """Call DeepSeek via OpenRouter for text understanding."""
        payload = {
            "model": self.config.deepseek_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096,
            "temperature": 0.5,
        }

        logger.info(f"[DEEPSEEK] Calling {self.config.deepseek_model} via OpenRouter")
        result = await self._call_openrouter_with_retry(payload)
        logger.info(f"[DEEPSEEK] Response received ({len(result)} chars)")
        return result

    async def _call_gemini_text(self, prompt: str) -> str:
        """Call Gemini for text understanding."""
        self._ensure_client()
        logger.info(f"[GEMINI] Calling {self.config.gemini_vision_model}")
        response = self._client.models.generate_content(
            model=self.config.gemini_vision_model,
            contents=prompt,
        )
        return response.text

    async def _call_gemini_vision(
        self,
        prompt: str,
        image_data: bytes | None = None,
        images_data: list[bytes] | None = None,
    ) -> str:
        """Call Gemini for vision understanding."""
        self._ensure_client()
        from google.genai import types

        content_parts = []
        if images_data:
            for img_bytes in images_data:
                content_parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))
        elif image_data:
            content_parts.append(types.Part.from_bytes(data=image_data, mime_type="image/png"))

        content_parts.append(prompt)

        logger.info(f"[GEMINI] Calling {self.config.gemini_vision_model} for vision")
        response = self._client.models.generate_content(
            model=self.config.gemini_vision_model,
            contents=content_parts,
        )
        return response.text

    # =========================================================================
    # REFINEMENT
    # =========================================================================

    async def _refine_extraction(
        self,
        initial: StoryboardUnderstanding,
        original_content: str,
        content_type: str = "text",
        audience: str = "c_suite",
    ) -> StoryboardUnderstanding:
        """
        Stage 2 (REFINE): Use alternate model to validate/improve low-confidence extraction.
        """
        if not self.config.enable_refinement:
            return initial
        if initial.extraction_confidence >= self.config.refinement_threshold:
            logger.info(f"[REFINE] Skipping - confidence {initial.extraction_confidence:.2f} >= threshold")
            return initial

        logger.info(f"[REFINE] Low confidence {initial.extraction_confidence:.2f} - triggering refinement")

        refinement_prompt = f"""You are refining an initial extraction that had low confidence ({initial.extraction_confidence:.2f}).

INITIAL EXTRACTION (may be incomplete or inaccurate):
- Headline: "{initial.headline}"
- Tagline: "{initial.tagline}"
- What it does: "{initial.what_it_does}"
- Business value: "{initial.business_value}"
- Pain point: "{initial.pain_point_addressed}"
- Raw extracted: "{initial.raw_extracted_text[:500] if initial.raw_extracted_text else 'None'}"

ORIGINAL CONTENT TO RE-ANALYZE:
{original_content[:6000]}

YOUR TASK: Improve and validate this extraction.
- If the initial extraction missed key details, add them
- If the initial extraction was wrong, correct it
- If the initial extraction was too generic, make it specific
- Increase confidence score ONLY if you found concrete details

TARGET AUDIENCE: {audience}

Return ONLY valid JSON matching this exact structure:
{{
    "raw_extracted_text": "...",
    "extraction_confidence": 0.9,
    "headline": "...",
    "tagline": "...",
    "what_it_does": "...",
    "business_value": "...",
    "who_benefits": "...",
    "differentiator": "...",
    "pain_point_addressed": "...",
    "suggested_icon": "..."
}}"""

        try:
            if content_type == "text":
                response_text = await self._call_gemini_text(refinement_prompt)
            else:
                response_text = await self._call_deepseek(refinement_prompt)

            refined = _safe_parse_understanding(response_text, source="refinement")

            if refined.extraction_confidence > initial.extraction_confidence:
                logger.info(f"[REFINE] Improved: {initial.extraction_confidence:.2f} → {refined.extraction_confidence:.2f}")
                return refined
            else:
                logger.info(f"[REFINE] No improvement, keeping initial")
                return initial

        except Exception as e:
            logger.warning(f"[REFINE] Refinement failed ({e}), keeping initial extraction")
            return initial

    # =========================================================================
    # UNDERSTANDING METHODS
    # =========================================================================

    async def _understand_content(
        self,
        prompt: str,
        content_type: Literal["text", "image"],
        audience: str,
        original_content: str,
        source_name: str,
        image_data: bytes | None = None,
        images_data: list[bytes] | None = None,
    ) -> StoryboardUnderstanding:
        """
        Common understanding logic for all content types.

        This is the base method that all understand_* methods delegate to.
        """
        try:
            # Route to appropriate provider
            if content_type == "text":
                if self.config.text_provider == "deepseek":
                    response_text = await self._call_deepseek(prompt)
                else:
                    response_text = await self._call_gemini_text(prompt)
            else:  # image
                if self.config.vision_provider == "qwen":
                    response_text = await self._call_qwen_vision(prompt, image_data, images_data)
                else:
                    response_text = await self._call_gemini_vision(prompt, image_data, images_data)

            # Parse response
            initial_result = _safe_parse_understanding(response_text, source=source_name)
            if initial_result.extraction_confidence > 0:
                logger.info(f"[UNDERSTAND] Successfully extracted insights from {source_name}")

            # Refine if needed
            refine_content = original_content
            if content_type == "image" and initial_result.raw_extracted_text:
                refine_content = f"[IMAGE DESCRIPTION]\n{initial_result.raw_extracted_text}"

            return await self._refine_extraction(
                initial=initial_result,
                original_content=refine_content,
                content_type=content_type,
                audience=audience,
            )

        except json.JSONDecodeError as e:
            logger.error(f"[UNDERSTAND] Failed to parse {source_name} response: {e}")
            return StoryboardUnderstanding(
                headline="EXTRACTION FAILED - Check Input",
                tagline="Could not extract content",
                what_it_does="The AI could not parse this input. Try a different input or check formatting.",
                business_value="Unable to determine - extraction failed",
                who_benefits="Unable to determine - extraction failed",
                differentiator="Unable to determine - extraction failed",
                pain_point_addressed="Unable to determine - extraction failed",
                suggested_icon="alert-triangle",
                raw_extracted_text=f"PARSE ERROR: {str(e)[:200]}",
                extraction_confidence=0.0,
            )
        except Exception as e:
            logger.error(f"[UNDERSTAND] {source_name} understanding failed: {e}")
            return _safe_parse_understanding("", source=f"{source_name}-error: {str(e)[:100]}")

    def _build_base_prompt_context(self, audience: str) -> tuple[str, str, str]:
        """Build common prompt context elements."""
        knowledge_context = self._build_knowledge_context(audience)
        language_guidelines = self._build_language_guidelines_minimal(audience)
        value_angle = get_value_angle_instruction(audience)
        return knowledge_context, language_guidelines, value_angle

    async def understand_code(
        self,
        code_content: str,
        icp_preset: dict[str, Any] | None = None,
        audience: str = "c_suite",
        file_name: str | None = None,
    ) -> StoryboardUnderstanding:
        """Stage 1: Analyze code and extract business value."""
        knowledge_context, language_guidelines, value_angle = self._build_base_prompt_context(audience)

        prompt = f"""Analyze this code and extract business value.

{f"File: {file_name}" if file_name else ""}

CODE:
```
{code_content[:8000]}
```

TARGET AUDIENCE: {audience}

{knowledge_context if knowledge_context else ""}
{language_guidelines if language_guidelines else ""}

EXTRACT:
- What does this code do (plain English)?
- Who benefits from this?
- What problem does it solve?
- What makes it special?

{value_angle}

CRITICAL RULES:
- NEVER include personal names - use roles/personas
- ALWAYS derive business value - infer from the problem being solved
- If value isn't explicit, INFER it

Return JSON:
{{
    "raw_extracted_text": "Key technical elements: classes, functions, logic",
    "extraction_confidence": 0.0-1.0,
    "headline": "Benefit-focused headline (8 words max)",
    "tagline": "Unique to THIS code (10 words max)",
    "what_it_does": "Plain English (2 sentences max)",
    "business_value": "ALWAYS provide value - quantified if possible, inferred if not",
    "who_benefits": "Role/persona titles ONLY - NO personal names",
    "differentiator": "What makes it special",
    "pain_point_addressed": "Problem solved",
    "suggested_icon": "Simple icon name"
}}"""

        return await self._understand_content(
            prompt=prompt,
            content_type="text",
            audience=audience,
            original_content=code_content,
            source_name="code",
        )

    async def understand_transcript(
        self,
        transcript: str,
        icp_preset: dict[str, Any] | None = None,
        audience: str = "c_suite",
        context: str | None = None,
    ) -> StoryboardUnderstanding:
        """Stage 1: Extract insights from transcript with minimal constraints."""
        knowledge_context, language_guidelines, value_angle = self._build_base_prompt_context(audience)

        # Add randomness to prompt style
        import random
        prompt_styles = [
            "What's the ONE thing that would make {audience} stop scrolling?",
            "If you had 3 seconds to grab {audience}'s attention, what would you say?",
            "What's the most surprising or compelling insight here for {audience}?",
            "What would make {audience} forward this to a colleague?",
        ]
        creative_hook = random.choice(prompt_styles).format(audience=audience)

        prompt = f"""Read this content and find what matters most.

{f"CONTEXT: {context}" if context else ""}

CONTENT:
{transcript[:32000]}

YOUR CHALLENGE: {creative_hook}

TARGET: {audience}

{knowledge_context if knowledge_context else ""}
{language_guidelines if language_guidelines else ""}

{value_angle}

BE CREATIVE. BE SPECIFIC. BE FRESH.
- Pull exact quotes and real numbers
- Use job titles, not personal names
- Infer value if not explicit - what problem is being solved?
- Make it feel DIFFERENT from the last thing you wrote

Return JSON (but make each field UNIQUE to this content):
{{
    "raw_extracted_text": "verbatim quotes and specifics you found",
    "extraction_confidence": 0.0-1.0,
    "headline": "something that would stop them mid-scroll",
    "tagline": "fresh angle on this specific content",
    "what_it_does": "plain english, no jargon",
    "business_value": "real impact - infer if needed",
    "who_benefits": "job titles only",
    "differentiator": "what makes THIS different",
    "pain_point_addressed": "the real problem",
    "suggested_icon": "visual metaphor"
}}"""

        return await self._understand_content(
            prompt=prompt,
            content_type="text",
            audience=audience,
            original_content=transcript,
            source_name="transcript",
        )

    def _get_persona_extraction_focus(self, audience: str, audience_info: dict) -> str:
        """Get persona-specific extraction instructions."""
        return get_persona_extraction_focus(audience)

    async def understand_image(
        self,
        image_data: bytes | str,
        icp_preset: dict[str, Any] | None = None,
        audience: str = "c_suite",
        sanitize_ip: bool = True,
    ) -> StoryboardUnderstanding:
        """Stage 1: Analyze image and extract business value."""
        # Handle base64 string input
        if isinstance(image_data, str):
            if image_data.startswith("data:"):
                image_data = image_data.split(",")[1]
            image_bytes = base64.b64decode(image_data)
        else:
            image_bytes = image_data

        knowledge_context, language_guidelines, value_angle = self._build_base_prompt_context(audience)

        prompt = f"""Analyze this image and extract ALL content.

CRITICAL: Extract the ACTUAL content from this image.
Do NOT generate generic copy. Do NOT make things up.

TARGET AUDIENCE: {audience}

{knowledge_context if knowledge_context else ""}
{language_guidelines if language_guidelines else ""}

EXTRACT:
- Every label, feature name, number visible
- Hierarchy/structure if present
- Timing, versions, phases if shown
- Workflow steps, connections, relationships

{value_angle}

CRITICAL RULES:
- NEVER output "Not mentioned in transcript/image" - always INFER from context
- NEVER include personal names - use roles/personas
- ALWAYS derive business value and problem solved - infer from what you see
- If something isn't explicit, INFER it from the context

Return JSON:
{{
    "raw_extracted_text": "Everything visible: labels, names, numbers",
    "extraction_confidence": 0.0-1.0,
    "headline": "Main theme from image (8 words max)",
    "tagline": "Unique to THIS content (10 words max)",
    "what_it_does": "Specific features/areas shown",
    "business_value": "INFER value from what this enables - never say 'not mentioned'",
    "who_benefits": "Role/persona titles ONLY - NO personal names",
    "differentiator": "What makes this special",
    "pain_point_addressed": "INFER the problem this solves - never say 'not mentioned'",
    "suggested_icon": "Icon representing content"
}}"""

        return await self._understand_content(
            prompt=prompt,
            content_type="image",
            audience=audience,
            original_content="[IMAGE]",
            source_name="image",
            image_data=image_bytes,
        )

    async def understand_multiple_images(
        self,
        images_data: list[bytes],
        icp_preset: dict[str, Any] | None = None,
        audience: str = "c_suite",
        sanitize_ip: bool = True,
    ) -> StoryboardUnderstanding:
        """Stage 1: Analyze multiple images and extract combined business value."""
        if len(images_data) > 3:
            logger.warning(f"Received {len(images_data)} images, using first 3 only")
            images_data = images_data[:3]

        knowledge_context, language_guidelines, value_angle = self._build_base_prompt_context(audience)

        prompt = f"""Analyze these {len(images_data)} images and extract ALL content.

CRITICAL: Extract ACTUAL content from each image.
Do NOT generate generic copy. Do NOT make things up.

TARGET AUDIENCE: {audience}

{knowledge_context if knowledge_context else ""}
{language_guidelines if language_guidelines else ""}

EXTRACT FROM EACH IMAGE:
- Every label, feature name, number visible
- Hierarchy/structure if present
- Timing, versions, phases if shown
- Workflow steps, connections, relationships

{value_angle}

Then SYNTHESIZE into unified message.

CRITICAL RULES:
- NEVER output "Not mentioned" - always INFER from context
- NEVER include personal names - use roles/personas
- ALWAYS derive business value and problem solved

Return JSON:
{{
    "raw_extracted_text": "IMAGE 1: [content]... IMAGE 2: [content]...",
    "extraction_confidence": 0.0-1.0,
    "headline": "Synthesized theme (8 words max)",
    "tagline": "Unique to THIS content (10 words max)",
    "what_it_does": "Specific features across images",
    "business_value": "Numbers from images if present",
    "who_benefits": "Who would use this",
    "differentiator": "What makes this special",
    "pain_point_addressed": "Problem solved",
    "suggested_icon": "Icon representing theme"
}}"""

        return await self._understand_content(
            prompt=prompt,
            content_type="image",
            audience=audience,
            original_content="[MULTI-IMAGE]",
            source_name="multi-image",
            images_data=images_data,
        )

    # =========================================================================
    # STORYBOARD GENERATION
    # =========================================================================

    def _build_storyboard_prompt(
        self,
        understanding: StoryboardUnderstanding,
        stage: str,
        audience: str,
        output_format: str,
        visual_style: str,
        artist_style: str | None,
        persona: dict,
        brand: dict,
        visual_style_config: dict,
        unique_seed: str,
    ) -> str:
        """Build the image generation prompt."""
        knowledge_context = self._build_knowledge_context(audience)
        persona_context = self._build_persona_generation_context(audience, persona)
        section_headers = get_section_headers(audience)

        # Build content section based on audience
        if audience == "top_tier_vc":
            raw_context = ""
            if understanding.raw_extracted_text:
                raw_context = f"\nSOURCE MATERIAL:\n{understanding.raw_extracted_text[:600]}"

            content_section = f"""CONTENT FOR INVESTOR AUDIENCE:

{persona_context}

WHAT WE EXTRACTED:
- Core Insight: "{understanding.headline}"
- Problem Space: "{understanding.pain_point_addressed}"
- Solution: "{understanding.what_it_does}"
- Differentiator: "{understanding.differentiator}"
- Business Value: "{understanding.business_value}"
{raw_context}

{knowledge_context if knowledge_context else ""}

CREATIVE FREEDOM: Design the visual however best tells this story.
You choose the layout, sections, and flow. No rigid template required.
Make it visually compelling - this could end up in a pitch deck."""
        else:
            raw_context = ""
            if understanding.raw_extracted_text:
                raw_context = f"\nRAW EXTRACTION:\n{understanding.raw_extracted_text[:500]}"

            content_section = f"""CONTENT TO DISPLAY:

{persona_context}

WHAT WE EXTRACTED:
- Headline: "{understanding.headline}"
- Description: "{understanding.what_it_does}"
- Value Proposition: "{understanding.business_value}"
- For: "{understanding.who_benefits}"
- Key Benefit: "{understanding.differentiator}"
- Problem Solved: "{understanding.pain_point_addressed}"
{raw_context}

{knowledge_context if knowledge_context else ""}

GUIDELINES:
- Write clear, direct business language
- Transform technical details → business outcomes
- Keep it SPECIFIC to what was extracted
- Derive messaging from the extracted content

NEVER output generic copy. ALWAYS use specifics from the extraction."""

        dynamic_tagline = understanding.tagline if understanding.tagline else brand.get('tagline', '')

        return f"""Create a UNIQUE professional one-page executive storyboard infographic.

GENERATION SEED: {unique_seed} (use this to create variation in layout and icons)

BRAND: {brand.get('company', 'Product')} - "{dynamic_tagline}"

{content_section}

VISUAL REQUIREMENTS:
- Style: Modern professional
- Color scheme: {brand.get('company', 'Product')} brand colors (MUST USE THESE EXACT COLORS):
  - Primary (CTAs/headers): {visual_style_config.get('primary_color', '#23433E')} (dark teal/forest green)
  - Accent (highlights/emphasis): {visual_style_config.get('accent_color', '#2D9688')} (teal)
  - Text: {visual_style_config.get('text_color', '#333333')} (dark gray)
  - Background: {visual_style_config.get('hero_bg', '#DDEDEB')} (light mint/sage green)
- NO badges, ribbons, or "demo/preview/coming soon" labels - keep it clean and professional
- Include simple icons representing the content (construction/business metaphors)
- Large, readable text (executive-friendly)

TEXT REQUIREMENTS:
- Spell everything correctly - double-check spelling
- Use LARGE fonts - small text gets garbled
- Keep text SHORT and IMPACTFUL
- DO NOT add generic labels like "Value Proposition" or "Key Benefit" - let the content speak for itself
- NO section headers unless they add real value to the story
- MAKE IT UNIQUE: Vary layout, icon placement, and flow based on the GENERATION SEED above
- Focus on the VISUAL STORY, not labeling everything

{get_format_layout_instructions(output_format)}

{get_visual_style_instructions(visual_style)}

{get_artist_style_instructions(artist_style) if artist_style else ""}

DESIGN PRINCIPLES:
- {visual_style_config.get('aesthetic', 'Modern, professional, teal/green palette. Corporate but approachable.')}
- Light mint/sage backgrounds with clean white sections
- Icons should be simple and metaphorical (tools, buildings, charts)
- Ready to share in presentations, emails, LinkedIn, or Slack
- CRITICAL: Use teal/green color palette, NOT orange or blue
- NO promotional badges or ribbons - this is executive content, not a sales flyer

{get_format_output_instructions(output_format)}"""

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

        Returns PNG image bytes.
        """
        self._ensure_client()

        from src.tools.storyboard.coperniq_presets import (
            COPERNIQ_ICP,
            COPERNIQ_BRAND,
            get_audience_persona,
        )

        import uuid
        from datetime import datetime

        if icp_preset is None:
            icp_preset = COPERNIQ_ICP

        visual_style_config = icp_preset.get("visual_style", {})
        brand = COPERNIQ_BRAND
        persona = get_audience_persona(audience, icp_preset)
        unique_seed = f"{datetime.now().isoformat()}-{uuid.uuid4().hex[:8]}"

        prompt = self._build_storyboard_prompt(
            understanding=understanding,
            stage=stage,
            audience=audience,
            output_format=output_format,
            visual_style=visual_style,
            artist_style=artist_style,
            persona=persona,
            brand=brand,
            visual_style_config=visual_style_config,
            unique_seed=unique_seed,
        )

        try:
            import asyncio
            from google.genai import types

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self._client.models.generate_content(
                        model=self.config.image_model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_modalities=["IMAGE", "TEXT"],
                        ),
                    )
                    break
                except Exception as e:
                    error_str = str(e)
                    if "503" in error_str or "UNAVAILABLE" in error_str or "overloaded" in error_str.lower():
                        if attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 5
                            logger.warning(f"[GEMINI] Model overloaded (503), waiting {wait_time}s")
                            await asyncio.sleep(wait_time)
                            continue
                    raise

            for part in response.candidates[0].content.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    return part.inline_data.data

            raise ValueError("No image generated in response")

        except Exception as e:
            logger.error(f"[GEMINI] Image generation failed: {e}")
            raise

    # =========================================================================
    # KNOWLEDGE & LANGUAGE HELPERS
    # =========================================================================

    def _build_language_guidelines(self, icp_preset: dict[str, Any], audience: str = "c_suite") -> str:
        """Build language guidelines string for prompts, enriched with knowledge."""
        avoid = icp_preset.get("language_style", {}).get("avoid", [])
        use = icp_preset.get("language_style", {}).get("use", [])
        tone = icp_preset.get("tone", "Friendly and professional")

        try:
            from src.knowledge.cache import KnowledgeCache
            cache = KnowledgeCache.get()
            if cache.is_loaded():
                knowledge = cache.get_language_guidelines(audience)
                avoid = list(set(knowledge["avoid"] + avoid))[:15]
                use = list(set(knowledge["use"] + use))[:15]
        except Exception:
            pass

        return f"""LANGUAGE GUIDELINES:
- Tone: {tone}
- AVOID these words/phrases: {', '.join(avoid[:15])}
- USE these words/phrases: {', '.join(use[:15])}
- Write for someone with no technical background
- Focus on benefits, not features"""

    def _build_knowledge_context(self, audience: str) -> str:
        """Build knowledge context section for prompts."""
        try:
            from src.knowledge.cache import KnowledgeCache
            cache = KnowledgeCache.get()
            if not cache.is_loaded():
                return ""

            ctx = cache.get_context(audience)

            if not any([ctx["pain_points"], ctx["features"], ctx["metrics"]]):
                return ""

            sections = []
            if ctx["pain_points"]:
                sections.append(f"CUSTOMER PAIN POINTS (from real calls): {'; '.join(ctx['pain_points'])}")
            if ctx["features"]:
                sections.append(f"PRODUCT FEATURES TO REFERENCE: {', '.join(ctx['features'])}")
            if ctx["metrics"]:
                sections.append(f"PROOF POINTS TO USE: {'; '.join(ctx['metrics'])}")
            if ctx.get("quotes"):
                sections.append(f"CUSTOMER QUOTES: {'; '.join(ctx['quotes'])}")

            return "\n".join(sections)
        except Exception:
            return ""

    def _build_language_guidelines_minimal(self, audience: str) -> str:
        """Build minimal language guidelines from knowledge cache only."""
        try:
            from src.knowledge.cache import KnowledgeCache
            cache = KnowledgeCache.get()
            if not cache.is_loaded():
                return ""

            knowledge = cache.get_language_guidelines(audience)
            avoid = knowledge.get("avoid", [])[:10]
            use = knowledge.get("use", [])[:10]

            if not avoid and not use:
                return ""

            parts = []
            if avoid:
                parts.append(f"AVOID: {', '.join(avoid)}")
            if use:
                parts.append(f"USE: {', '.join(use)}")

            return "\n".join(parts)
        except Exception:
            return ""

    # =========================================================================
    # PERSONA HELPERS (delegate to config)
    # =========================================================================

    def _get_value_angle_instruction(self, audience: str) -> str:
        """Get value angle framing instruction for extraction based on audience."""
        return get_value_angle_instruction(audience)

    def _get_audience_section_headers(self, audience: str) -> str:
        """Get audience-specific section headers for the storyboard."""
        return get_section_headers(audience)

    def _build_persona_generation_context(self, audience: str, persona: dict) -> str:
        """Build persona-specific context for image generation."""
        value_angle = persona.get("value_angle", "ROI")
        value_framing = persona.get("value_framing", "")
        cares_about = persona.get("cares_about", [])
        hooks = persona.get("hooks", [])
        title = persona.get("title", audience)
        tone = persona.get("tone", "Professional")

        # Simplified persona context - less prescriptive, more creative freedom
        if audience == "field_crew":
            return f"""AUDIENCE: {title}
THEY CARE ABOUT: {', '.join(cares_about[:3])}
MAKE IT: Simple, practical, 5th grade reading level
TONE: {tone}"""

        if audience == "c_suite":
            return f"""AUDIENCE: {title}
THEY CARE ABOUT: {', '.join(cares_about[:3])}
MAKE IT: Data-driven, numbers prominent, executive-friendly
TONE: {tone}"""

        if audience == "business_owner":
            return f"""AUDIENCE: {title}
THEY CARE ABOUT: {', '.join(cares_about[:3])}
MAKE IT: Emotional, relatable, show the pain and solution
TONE: {tone}"""

        if audience == "btl_champion":
            return f"""AUDIENCE: {title}
THEY CARE ABOUT: {', '.join(cares_about[:3])}
MAKE IT: Practical, day-in-life scenarios, career benefits
TONE: {tone}"""

        if audience == "top_tier_vc":
            return f"""AUDIENCE: {title}
THEY CARE ABOUT: {', '.join(cares_about[:3])}
MAKE IT: Investment-focused, market opportunity, no sales pitch
TONE: {tone}"""

        return f"""AUDIENCE: {title}
THEY CARE ABOUT: {', '.join(cares_about[:3]) if cares_about else 'results'}
TONE: {tone}"""

    # =========================================================================
    # FORMAT HELPERS (delegate to config)
    # =========================================================================

    def _get_format_layout_instructions(self, output_format: str) -> str:
        """Get layout instructions based on output format."""
        return get_format_layout_instructions(output_format)

    def _get_format_output_instructions(self, output_format: str) -> str:
        """Get output specifications based on format."""
        return get_format_output_instructions(output_format)

    def _get_visual_style_instructions(self, visual_style: str) -> str:
        """Get visual style instructions based on style preference."""
        return get_visual_style_instructions(visual_style)

    def _get_artist_style_instructions(self, artist_style: str | None) -> str:
        """Get artist style instructions for fun variations."""
        return get_artist_style_instructions(artist_style)

    # =========================================================================
    # HEALTH CHECK
    # =========================================================================

    async def health_check(self) -> dict[str, Any]:
        """Check if Gemini client is properly configured."""
        try:
            self._ensure_client()
            return {
                "status": "healthy",
                "vision_provider": self.config.vision_provider,
                "text_provider": self.config.text_provider,
                "image_model": self.config.image_model,
                "api_key_configured": bool(self.config.api_key),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "api_key_configured": bool(self.config.api_key),
            }
