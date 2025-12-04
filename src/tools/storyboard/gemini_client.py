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
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


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
    # Count quotes to see if we have an odd number (unterminated)
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


# Vision model options for understanding (images)
VisionModel = Literal["gemini", "qwen"]

# Text model options for understanding (code/transcripts)
TextModel = Literal["gemini", "deepseek"]


class StoryboardUnderstanding(BaseModel):
    """Extracted understanding from code/roadmap analysis."""

    headline: str = Field(..., description="Catchy, benefit-focused headline (8 words max)")
    tagline: str = Field(
        default="One platform for contractors who do it all",
        description="Dynamic tagline specific to content and persona (10 words max)"
    )
    what_it_does: str = Field(..., description="Plain English description (2 sentences max)")
    business_value: str = Field(..., description="Quantified benefit (hours saved, % improvement)")
    who_benefits: str = Field(..., description="Target persona description")
    differentiator: str = Field(..., description="What makes this special (1 sentence)")
    pain_point_addressed: str = Field(..., description="The problem this solves")
    suggested_icon: str = Field(default="clipboard-check", description="Icon suggestion for visual")
    # DEBUG/VERIFICATION fields - for CEO/CTO to verify extraction is correct
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

    api_key: str | None = None  # Google API key for Gemini
    openrouter_api_key: str | None = None  # OpenRouter API key for Qwen/DeepSeek

    # ==========================================================================
    # INTELLIGENT MODEL ROUTING
    # ==========================================================================
    # Stage 1 (EXTRACT): Primary models for initial extraction
    vision_provider: VisionModel = "qwen"  # For images (default: qwen for better doc understanding)
    text_provider: TextModel = "deepseek"  # For text/transcripts (default: deepseek for best reasoning)

    # Stage 2 (REFINE): Enable multi-model refinement for low-confidence extractions
    enable_refinement: bool = True  # If True, low-confidence extractions get refined by alternate model
    refinement_threshold: float = 0.75  # Confidence below this triggers refinement pass

    # Model identifiers
    gemini_vision_model: str = "models/gemini-2.0-flash"  # Gemini vision model (fallback)
    qwen_model: str = "qwen/qwen2.5-vl-72b-instruct"  # Qwen 2.5 VL 72B - vision + doc understanding
    deepseek_model: str = "deepseek/deepseek-r1-0528"  # DeepSeek R1-0528 - reasoning + structured extraction

    # Stage 3 (GENERATE): Image generation (Gemini only - no alternatives)
    image_model: str = "models/gemini-3-pro-image-preview"  # Nano Banana - FREE during preview

    timeout: int = 90
    max_retries: int = 3

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv("GOOGLE_API_KEY")
        if self.openrouter_api_key is None:
            self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")


class GeminiStoryboardClient:
    """
    Client for Gemini Vision + Image Generation.

    Three-stage intelligent pipeline:
    1. EXTRACT - Primary model extracts (DeepSeek for text, Qwen for images)
    2. REFINE - If confidence < threshold, alternate model validates/improves
    3. GENERATE - Gemini creates the image (only model that can generate)

    Model Routing Intelligence:
    - DeepSeek R1-0528: Reasoning model, excels at structured extraction from text
    - Qwen 2.5 VL 72B: Vision model, excels at OCR and visual understanding
    - Gemini 3 Pro: Image generation (no alternatives available)

    Example:
        client = GeminiStoryboardClient()

        # Stage 1 + 2: Extract → Refine (automatic model routing)
        understanding = await client.understand_code(
            code_content="def calculate_roi(): ...",
            icp_preset=COPERNIQ_ICP,
            audience="c_suite",
        )

        # Stage 3: Generate
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

    async def _call_openrouter_with_retry(
        self,
        payload: dict,
        max_retries: int = 3,
    ) -> str:
        """
        Call OpenRouter API with retry logic for rate limits.

        Args:
            payload: Request payload
            max_retries: Number of retries on rate limit

        Returns:
            Model response text
        """
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
                        # Rate limited - wait and retry
                        wait_time = (attempt + 1) * 5  # 5s, 10s, 15s
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
        """
        Call Qwen VL via OpenRouter for vision understanding.

        Args:
            prompt: Text prompt for the model
            image_data: Single image bytes (optional)
            images_data: Multiple image bytes (optional)

        Returns:
            Model response text
        """
        # Build message content
        content = []

        # Add images if provided
        if images_data:
            for img_bytes in images_data[:3]:  # Max 3 images
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_b64}"
                    }
                })
        elif image_data:
            img_b64 = base64.b64encode(image_data).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_b64}"
                }
            })

        # Add text prompt
        content.append({
            "type": "text",
            "text": prompt
        })

        payload = {
            "model": self.config.qwen_model,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": 4096,
            "temperature": 0.3,
        }

        logger.info(f"[QWEN] Calling {self.config.qwen_model} via OpenRouter")
        result = await self._call_openrouter_with_retry(payload)
        logger.info(f"[QWEN] Response received ({len(result)} chars)")
        return result

    async def _call_deepseek(
        self,
        prompt: str,
    ) -> str:
        """
        Call DeepSeek R1 via OpenRouter for text understanding (code/transcripts).

        Args:
            prompt: Text prompt for the model

        Returns:
            Model response text
        """
        payload = {
            "model": self.config.deepseek_model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 4096,
            "temperature": 0.3,
        }

        logger.info(f"[DEEPSEEK] Calling {self.config.deepseek_model} via OpenRouter")
        result = await self._call_openrouter_with_retry(payload)
        logger.info(f"[DEEPSEEK] Response received ({len(result)} chars)")
        return result

    async def _refine_extraction(
        self,
        initial: StoryboardUnderstanding,
        original_content: str,
        content_type: str = "text",
        audience: str = "c_suite",
    ) -> StoryboardUnderstanding:
        """
        Stage 2 (REFINE): Use alternate model to validate/improve low-confidence extraction.

        Intelligent routing:
        - If initial extraction used DeepSeek → refine with Qwen (adds vision/structure insight)
        - If initial extraction used Qwen → refine with DeepSeek (adds reasoning depth)

        Args:
            initial: Initial extraction result
            original_content: Original input (code or image description)
            content_type: "text" or "image" to determine which alternate model to use
            audience: Target audience for refinement context

        Returns:
            Refined StoryboardUnderstanding (or original if refinement disabled/unnecessary)
        """
        # Skip refinement if disabled or confidence is high enough
        if not self.config.enable_refinement:
            return initial
        if initial.extraction_confidence >= self.config.refinement_threshold:
            logger.info(f"[REFINE] Skipping - confidence {initial.extraction_confidence:.2f} >= threshold {self.config.refinement_threshold}")
            return initial

        logger.info(f"[REFINE] Low confidence {initial.extraction_confidence:.2f} - triggering refinement pass")

        # Build refinement prompt with initial extraction context
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
            # Route to alternate model based on content type
            if content_type == "text":
                # Text was initially processed by DeepSeek, refine with... DeepSeek again (no vision alt for text)
                # Actually for text, we can try Gemini as alternate
                self._ensure_client()
                logger.info(f"[REFINE] Using Gemini as alternate for text refinement")
                response = self._client.models.generate_content(
                    model=self.config.gemini_vision_model,
                    contents=refinement_prompt,
                )
                response_text = response.text
            else:
                # Image was initially processed by Qwen, refine with DeepSeek for reasoning
                logger.info(f"[REFINE] Using DeepSeek as alternate for image refinement (reasoning pass)")
                response_text = await self._call_deepseek(refinement_prompt)

            # Parse refined result
            refined = _safe_parse_understanding(response_text, source="refinement")

            # Only use refinement if it actually improved confidence
            if refined.extraction_confidence > initial.extraction_confidence:
                logger.info(f"[REFINE] Improved: {initial.extraction_confidence:.2f} → {refined.extraction_confidence:.2f}")
                return refined
            else:
                logger.info(f"[REFINE] No improvement ({refined.extraction_confidence:.2f}), keeping initial")
                return initial

        except Exception as e:
            logger.warning(f"[REFINE] Refinement failed ({e}), keeping initial extraction")
            return initial

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
        # Build the analysis prompt with knowledge enrichment
        language_guidelines = self._build_language_guidelines(icp_preset, audience)
        knowledge_context = self._build_knowledge_context(audience)
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

{knowledge_context}

EXTRACTION REQUIREMENTS:
1. raw_extracted_text: EXTRACT the key technical elements from this code:
   - Main class/function names and their purpose
   - Core business logic (e.g., "calculates optimal route", "validates permits")
   - Key integrations or data sources
   Example: "VideoScriptGeneratorTool: generates 60-sec Loom scripts using DeepSeek V3. Inputs: prospect_company, industry, pain_point. Outputs: script with hook/demo/cta sections."

2. extraction_confidence: How well did you understand this code? (0.0-1.0)
   - 1.0 = Completely understood the business purpose
   - 0.8 = Understood most of it
   - 0.5 = Only partially understood
   - Below 0.5 = Could not determine purpose

3. headline: Catchy, benefit-focused headline (8 words max). Derived from what the code ACTUALLY does.
   NOT ALLOWED: "Transform How You Work" or generic phrases

4. tagline: Create a UNIQUE tagline for THIS specific code (10 words max). Examples:
   - For scheduling features: "Never miss a job. Never double-book again."
   - For payment tools: "Get paid faster. Spend less time chasing checks."
   - For field tools: "Your crew's best friend, rain or shine."
   The tagline should be specific to what THIS code does - NOT generic contractor messaging.

5. what_it_does: Plain English description (2 sentences max). Based on ACTUAL code logic.
   NOT ALLOWED: "A powerful capability that makes your work easier"

6. business_value: Quantified benefit. Use real numbers if possible (hours saved, % improvement).
7. who_benefits: Who in the organization benefits most from this.
8. differentiator: What makes this special compared to doing it manually.
9. pain_point_addressed: The specific problem/frustration this eliminates.
10. suggested_icon: A simple icon name that represents this (e.g., "clock", "dollar", "team").

CRITICAL RULES:
- EXTRACT from the actual code - do NOT make generic statements
- NO technical jargon in the OUTPUT (but DO understand the code technically)
- Write like you're explaining to a smart 5th grader
- Focus on BENEFITS not FEATURES
- The tagline MUST be unique to the content - never use generic phrases

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
            # Route to DeepSeek or Gemini based on config
            if self.config.text_provider == "deepseek":
                logger.info(f"[UNDERSTAND] Using DeepSeek ({self.config.deepseek_model}) for code understanding")
                response_text = await self._call_deepseek(prompt)
            else:
                # Use Gemini
                self._ensure_client()
                logger.info(f"[UNDERSTAND] Using Gemini ({self.config.gemini_vision_model}) for code understanding")
                response = self._client.models.generate_content(
                    model=self.config.gemini_vision_model,
                    contents=prompt,
                )
                response_text = response.text

            # Parse JSON response
            json_str = response_text.strip()
            # Handle markdown code blocks
            if json_str.startswith("```"):
                json_str = json_str.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
                json_str = json_str.strip()

            data = json.loads(json_str)
            initial_result = StoryboardUnderstanding(**data)

            # Stage 2 (REFINE): If low confidence, run through alternate model
            refined_result = await self._refine_extraction(
                initial=initial_result,
                original_content=code_content,
                content_type="text",
                audience=audience,
            )
            return refined_result

        except json.JSONDecodeError as e:
            logger.error(f"[UNDERSTAND] Failed to parse response: {e}")
            logger.error(f"[UNDERSTAND] Raw response was: {response_text[:500] if response_text else 'None'}")
            # DO NOT return generic fallback - that hides extraction failures
            # Instead, return with low confidence so user knows to check
            return StoryboardUnderstanding(
                headline="EXTRACTION FAILED - Check Input",
                tagline="Could not extract content from this code",
                what_it_does="The AI could not parse this input. Try a different code file or check formatting.",
                business_value="Unable to determine - extraction failed",
                who_benefits="Unable to determine - extraction failed",
                differentiator="Unable to determine - extraction failed",
                pain_point_addressed="Unable to determine - extraction failed",
                suggested_icon="alert-triangle",
                raw_extracted_text=f"PARSE ERROR: {str(e)[:200]}",
                extraction_confidence=0.0,  # Zero confidence = failed
            )
        except Exception as e:
            logger.error(f"[GEMINI] Understanding failed: {e}")
            raise

    async def understand_transcript(
        self,
        transcript: str,
        icp_preset: dict[str, Any],
        audience: str = "c_suite",
        context: str | None = None,
    ) -> StoryboardUnderstanding:
        """
        Stage 1: Analyze call transcript/notes and extract persona-specific insights.

        Designed for Loom transcripts, call summaries, meeting notes.
        Uses FULL transcript (up to 32K chars) and extracts SPECIFIC details.

        Args:
            transcript: Full transcript text (longer than code - up to 32K)
            icp_preset: ICP configuration dictionary
            audience: Target audience persona - affects extraction focus
            context: Optional context (e.g., "Sales demo call", "Discovery call")

        Returns:
            StoryboardUnderstanding with insights extracted FROM the transcript
        """
        # Build persona-specific extraction requirements
        audience_info = icp_preset.get("audience_personas", {}).get(audience, {})
        persona_hooks = audience_info.get("hooks", [])
        persona_cares = audience_info.get("cares_about", [])

        # Get Coperniq value props to map against
        value_props = icp_preset.get("value_props", {})
        proof_points = icp_preset.get("brand", {}).get("proof_points", {})

        # Persona-specific extraction focus
        persona_extraction = self._get_persona_extraction_focus(audience, audience_info)

        prompt = f"""You are analyzing a CALL TRANSCRIPT or MEETING NOTES for Coperniq, a software platform for MEP & Energy contractors.

{f"CONTEXT: {context}" if context else "CONTEXT: Call transcript / meeting notes"}

=== TRANSCRIPT (EXTRACT SPECIFIC DETAILS FROM THIS) ===
{transcript[:32000]}
=== END TRANSCRIPT ===

YOUR TASK: Extract SPECIFIC insights from this transcript that will resonate with a {audience_info.get('title', audience)} audience.

TARGET PERSONA: {audience_info.get('title', audience)}
- They care about: {', '.join(persona_cares)}
- Tone: {audience_info.get('tone', 'Professional')}
- Hooks that work: {', '.join(persona_hooks[:2])}

{persona_extraction}

COPERNIQ VALUE PROPS TO MAP AGAINST (use these if relevant to transcript):
- Core: {', '.join(value_props.get('core', ['Projects', 'Dispatch', 'PPM']))}
- AI Features: {', '.join(value_props.get('ai', ['Receptionist AI', 'Project Copilot']))}
- Proof Points: {proof_points.get('completion_rate', '99% completion')} | {proof_points.get('payment_speed', '65% faster')}

EXTRACTION REQUIREMENTS (pull SPECIFIC details from transcript - THIS IS CRITICAL):
1. raw_extracted_text: VERBATIM extraction from the transcript:
   - Key quotes and phrases the prospect used
   - All numbers mentioned (dollars, hours, percentages)
   - Company names, tools mentioned, pain points described
   - Names and roles of people mentioned
   Example: "Prospect: 'We lose about $3K per job on change orders.' PM: 'My guys are still using Excel, I can't get them off it.' Numbers: $3K/job, 8 field crews, 50 projects/month."

2. extraction_confidence: How much of the transcript did you understand? (0.0-1.0)
   - 1.0 = Clear transcript, extracted key points
   - 0.8 = Mostly clear
   - 0.5 = Partial audio issues or unclear

3. headline: Use EXACT words/phrases from the call. (8 words max)
   BAD: "Streamline Operations" (generic)
   GOOD: "Stop Losing $3K Per Job to Spreadsheets" (their words + their numbers)
   NOT ALLOWED: "Transform How You Work" or generic phrases

4. tagline: Build from THEIR pain point, not generic contractor pain. (10 words max)
   BAD: "One platform for contractors" (generic)
   GOOD: "Finally get your PM off Excel and home by 5pm" (their situation)
   NOT ALLOWED: "Built for contractors" or other canned phrases

5. what_it_does: Summarize using THEIR words and situation.
   NOT ALLOWED: "A powerful capability that makes your work easier"

6. business_value: Use SPECIFIC numbers from transcript.
   - If they said "we lose 2 hours a day" → use "2 hours/day"
   - If they said "$50K in change orders" → use "$50K"
   - DO NOT generalize to "save time and money"

7. who_benefits: Who specifically in the organization was mentioned?
8. differentiator: What made this stand out vs their current way (from transcript)?
9. pain_point_addressed: Use THEIR EXACT WORDS for the pain point.
10. suggested_icon: Icon representing the main theme.

CRITICAL RULES (BULLETPROOF EXTRACTION):
- If they said it, USE IT. Do not paraphrase away specifics.
- Numbers are GOLD - preserve every number exactly as stated.
- Names and roles are GOLD - preserve who said what.
- The CEO/CTO will review this - it MUST match what was said.
- If you cannot find specific content, say "Not mentioned in transcript"
- NEVER output generic phrases like "save time and money" if they gave you specifics.

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
            # Route to DeepSeek or Gemini based on config
            if self.config.text_provider == "deepseek":
                logger.info(f"[UNDERSTAND] Using DeepSeek ({self.config.deepseek_model}) for transcript understanding")
                response_text = await self._call_deepseek(prompt)
            else:
                # Use Gemini
                self._ensure_client()
                logger.info(f"[UNDERSTAND] Using Gemini ({self.config.gemini_vision_model}) for transcript understanding")
                response = self._client.models.generate_content(
                    model=self.config.gemini_vision_model,
                    contents=prompt,
                )
                response_text = response.text

            # Parse JSON response with safe fallback
            initial_result = _safe_parse_understanding(response_text, source="transcript")
            if initial_result.extraction_confidence > 0:
                logger.info(f"[UNDERSTAND] Successfully extracted insights from transcript for {audience}")

            # Stage 2 (REFINE): If low confidence, run through alternate model
            refined_result = await self._refine_extraction(
                initial=initial_result,
                original_content=transcript,
                content_type="text",
                audience=audience,
            )
            return refined_result

        except Exception as e:
            logger.error(f"[GEMINI] Transcript understanding failed: {e}")
            return _safe_parse_understanding("", source=f"transcript-error: {str(e)[:100]}")

    def _get_persona_extraction_focus(self, audience: str, audience_info: dict) -> str:
        """Get persona-specific extraction instructions."""
        extractions = {
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
        return extractions.get(audience, extractions["c_suite"])

    async def understand_image(
        self,
        image_data: bytes | str,
        icp_preset: dict[str, Any],
        audience: str = "c_suite",
        sanitize_ip: bool = True,
    ) -> StoryboardUnderstanding:
        """
        Stage 1: Analyze image (Miro screenshot, roadmap) and extract business value.

        Uses Qwen VL (via OpenRouter) or Gemini Vision based on config.
        Extra sanitization for IP protection when analyzing roadmaps.

        Args:
            image_data: Image bytes or base64 string
            icp_preset: ICP configuration dictionary
            audience: Target audience persona
            sanitize_ip: Whether to apply extra IP sanitization

        Returns:
            StoryboardUnderstanding with extracted insights
        """
        # Handle base64 string input
        if isinstance(image_data, str):
            if image_data.startswith("data:"):
                # Remove data URL prefix
                image_data = image_data.split(",")[1]
            image_bytes = base64.b64decode(image_data)
        else:
            image_bytes = image_data

        # Build the analysis prompt with knowledge enrichment
        language_guidelines = self._build_language_guidelines(icp_preset, audience)
        knowledge_context = self._build_knowledge_context(audience)
        audience_info = icp_preset.get("audience_personas", {}).get(audience, {})

        # FULL EXTRACTION - Pull EVERYTHING from the image
        # Professional language transformation happens in generate_storyboard(), not here
        extraction_rules = """
EXTRACTION RULES - Pull EVERYTHING from the image (THIS IS CRITICAL):
- EXTRACT every feature name, product area, and label visible
- INCLUDE all metrics, dates, percentages, numbers EXACTLY as shown
- PRESERVE hierarchy/structure (e.g., "5 product clouds", "3 phases")
- USE exact terminology as written (e.g., "Receptionist AI BETA", "Document Engine V1")
- For roadmaps: capture timing (Q1/Q2/H1), version labels (BETA, V1, V2)
- For Miro boards: extract workflow steps, connections, labels
- For diagrams: capture all boxes, arrows, connections

The GENERATION phase will transform this into professional, external-ready output.
EXTRACT FULLY NOW - sanitization happens later.
"""

        prompt = f"""Analyze this image and EXTRACT ALL CONTENT for business messaging.

CRITICAL: You MUST extract the ACTUAL content from this image.
Do NOT generate generic copy. Do NOT make things up.
If you cannot read something clearly, say so in raw_extracted_text.

TARGET AUDIENCE: {audience_info.get('title', audience)}
They care about: {', '.join(audience_info.get('cares_about', ['efficiency', 'results']))}

{knowledge_context}

{language_guidelines}

{extraction_rules}

EXTRACTION REQUIREMENTS (extract ACTUAL content from the image):
1. raw_extracted_text: LIST EVERYTHING visible in the image. Every label, every feature name, every number.
   Example for a roadmap: "Coperniq Intelligence: Receptionist AI BETA Q1, AI Agent Builder BETA Q1, Design AI BETA Q2. Sales Cloud: Catalog 2.0 V2, Proposals V1..."
   Example for Miro: "Workflow: Lead capture → Qualification → Proposal → Close. Labels: Hot Lead, Warm Lead, Cold Lead..."

2. extraction_confidence: How confident are you that you read the image correctly? (0.0-1.0)
   - 1.0 = Crystal clear, read everything
   - 0.8 = Most content clear, some fuzzy
   - 0.5 = Significant portions unclear
   - Below 0.5 = Low quality, may need re-upload

3. headline: Extract the MAIN theme from the image using ACTUAL words visible. (8 words max)
   Example from roadmap: "5 Product Clouds Launching H1 2026"
   Example from Miro: "Lead-to-Close Workflow in 4 Steps"
   NOT ALLOWED: "Transform How You Work" or other generic phrases

4. tagline: Create tagline FROM what you extracted, not generic messaging. (10 words max)
   Example: "From AI Receptionist to Three-Phase Inverters"
   NOT ALLOWED: "Built for contractors" or other canned phrases

5. what_it_does: Describe the SPECIFIC features/areas shown. Name them explicitly.
   Example: "Coperniq Intelligence brings Receptionist AI and AI Agent Builder in Q1, followed by Design AI in Q2."
   NOT ALLOWED: "A powerful capability that makes your work easier"

6. business_value: Use SPECIFIC numbers from the image if present.
7. who_benefits: Based on what you see, who would use this?
8. differentiator: What makes THIS specific content special?
9. pain_point_addressed: What problem does THIS specific content solve?
10. suggested_icon: Icon representing what you see (e.g., "calendar", "robot", "truck").

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
            # Route to Qwen VL or Gemini based on config
            if self.config.vision_provider == "qwen":
                logger.info(f"[UNDERSTAND] Using Qwen VL ({self.config.qwen_model}) for image understanding")
                response_text = await self._call_qwen_vision(prompt, image_data=image_bytes)
            else:
                # Use Gemini
                self._ensure_client()
                logger.info(f"[UNDERSTAND] Using Gemini ({self.config.gemini_vision_model}) for image understanding")
                from google.genai import types

                image_part = types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/png",
                )

                response = self._client.models.generate_content(
                    model=self.config.gemini_vision_model,
                    contents=[image_part, prompt],
                )
                response_text = response.text

            # Parse JSON response with safe fallback
            initial_result = _safe_parse_understanding(response_text, source="image")
            if initial_result.extraction_confidence > 0:
                logger.info(f"[UNDERSTAND] Successfully extracted insights from image")

            # Stage 2 (REFINE): If low confidence, run through DeepSeek for reasoning pass
            # For images, we pass the raw_extracted_text as context since we can't re-send the image
            refined_result = await self._refine_extraction(
                initial=initial_result,
                original_content=f"[IMAGE DESCRIPTION FROM QWEN VL]\n{initial_result.raw_extracted_text}",
                content_type="image",
                audience=audience,
            )
            return refined_result

        except Exception as e:
            logger.error(f"[GEMINI] Image understanding failed: {e}")
            return _safe_parse_understanding("", source=f"image-error: {str(e)[:100]}")

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
        if len(images_data) > 3:
            logger.warning(f"Received {len(images_data)} images, using first 3 only")
            images_data = images_data[:3]

        # Build the analysis prompt for multiple images with knowledge enrichment
        language_guidelines = self._build_language_guidelines(icp_preset, audience)
        knowledge_context = self._build_knowledge_context(audience)
        audience_info = icp_preset.get("audience_personas", {}).get(audience, {})

        # FULL EXTRACTION - Pull EVERYTHING from ALL images
        # Professional language transformation happens in generate_storyboard(), not here
        extraction_rules = """
EXTRACTION RULES - Pull EVERYTHING from ALL images (THIS IS CRITICAL):
- EXTRACT every feature name, product area, and label visible in EACH image
- INCLUDE all metrics, dates, percentages, numbers EXACTLY as shown
- PRESERVE hierarchy/structure (e.g., "5 product clouds", "3 phases")
- USE exact terminology as written (e.g., "Receptionist AI BETA", "Document Engine V1")
- For roadmaps: capture timing (Q1/Q2/H1), version labels (BETA, V1, V2)
- For Miro boards: extract workflow steps, connections, labels
- For diagrams: capture all boxes, arrows, connections

EXTRACT FROM EACH IMAGE SEPARATELY FIRST, then synthesize.
The GENERATION phase will transform this into professional, external-ready output.
"""

        prompt = f"""Analyze these {len(images_data)} images and EXTRACT ALL CONTENT from each one.

CRITICAL: You MUST extract the ACTUAL content from each image.
Do NOT generate generic copy. Do NOT make things up.
If you cannot read something clearly, say so in raw_extracted_text.

The images may include:
- CTO roadmap or planning documents
- Miro boards or whiteboard screenshots
- Marketing materials or campaign visuals
- Product screenshots or demos

TARGET AUDIENCE: {audience_info.get('title', audience)}
They care about: {', '.join(audience_info.get('cares_about', ['efficiency', 'results']))}

{knowledge_context}

{language_guidelines}

{extraction_rules}

EXTRACTION REQUIREMENTS (extract from EACH image, then synthesize):
1. raw_extracted_text: LIST EVERYTHING visible across ALL images. Organize by image:
   "IMAGE 1 (Roadmap): Coperniq Intelligence: Receptionist AI BETA Q1, AI Agent Builder BETA Q1..."
   "IMAGE 2 (Miro): Workflow steps: Lead → Qualify → Propose → Close..."
   "IMAGE 3 (Campaign): Headline: Get Paid Faster, Subhead: 65% faster payment collection..."

2. extraction_confidence: How confident are you that you read ALL images correctly? (0.0-1.0)
   - 1.0 = All images crystal clear
   - 0.8 = Most content clear
   - 0.5 = Significant portions unclear

3. headline: Extract/synthesize MAIN theme using ACTUAL words from the images. (8 words max)
   Example: "5 Product Clouds Plus Lead-to-Close Workflow"
   NOT ALLOWED: "Transform How You Work" or other generic phrases

4. tagline: Create tagline FROM what you extracted across images. (10 words max)
   Example: "From AI Receptionist to Closed Deal in 4 Steps"
   NOT ALLOWED: "Built for contractors" or other canned phrases

5. what_it_does: Describe SPECIFIC features/areas shown across images. Name them explicitly.
   Example: "Combines Coperniq Intelligence (Receptionist AI, Agent Builder) with streamlined lead workflow..."
   NOT ALLOWED: "A powerful capability that makes your work easier"

6. business_value: Use SPECIFIC numbers from any image if present.
7. who_benefits: Based on what you see across images, who would use this?
8. differentiator: What makes THIS combined content special?
9. pain_point_addressed: What problem does THIS content solve?
10. suggested_icon: Icon representing the combined theme.

CRITICAL: Synthesize into ONE unified message, but base it on ACTUAL extracted content.

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
            # Route to Qwen VL or Gemini based on config
            if self.config.vision_provider == "qwen":
                logger.info(f"[UNDERSTAND] Using Qwen VL ({self.config.qwen_model}) for {len(images_data)} images")
                response_text = await self._call_qwen_vision(prompt, images_data=images_data)
            else:
                # Use Gemini
                self._ensure_client()
                logger.info(f"[UNDERSTAND] Using Gemini ({self.config.gemini_vision_model}) for {len(images_data)} images")
                from google.genai import types

                # Create image parts for all images
                content_parts = []
                for i, img_bytes in enumerate(images_data):
                    image_part = types.Part.from_bytes(
                        data=img_bytes,
                        mime_type="image/png",
                    )
                    content_parts.append(image_part)

                # Add the prompt at the end
                content_parts.append(prompt)

                response = self._client.models.generate_content(
                    model=self.config.gemini_vision_model,
                    contents=content_parts,
                )
                response_text = response.text

            # Parse JSON response with safe fallback
            initial_result = _safe_parse_understanding(response_text, source="multi-image")
            if initial_result.extraction_confidence > 0:
                logger.info(f"[GEMINI] Successfully understood {len(images_data)} images together")

            # Stage 2 (REFINE): If low confidence, run through DeepSeek for reasoning pass
            refined_result = await self._refine_extraction(
                initial=initial_result,
                original_content=f"[MULTI-IMAGE DESCRIPTION FROM QWEN VL]\n{initial_result.raw_extracted_text}",
                content_type="image",
                audience=audience,
            )
            return refined_result

        except Exception as e:
            logger.error(f"[GEMINI] Multi-image understanding failed: {e}")
            return _safe_parse_understanding("", source=f"multi-image-error: {str(e)[:100]}")

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
            # VC/Investor storyboard - flexible investment thesis
            proof = brand.get("proof_points", {})

            # Include raw extraction for richer context
            raw_context = ""
            if understanding.raw_extracted_text:
                raw_context = f"""
SOURCE MATERIAL (use to derive specific insights):
{understanding.raw_extracted_text[:600]}
"""

            content_section = f"""CONTENT FOR INVESTOR AUDIENCE:

WHAT WE EXTRACTED:
- Core Insight: "{understanding.headline}"
- Problem Space: "{understanding.pain_point_addressed}"
- Solution: "{understanding.what_it_does}"
- Differentiator: "{understanding.differentiator}"
- Business Value: "{understanding.business_value}"
{raw_context}

PROOF POINTS (weave in naturally, don't force all):
- {proof.get('completion_rate', 'High completion rates')}
- {proof.get('payment_speed', 'Faster payment cycles')}
- {proof.get('scale_story', 'Scales without adding headcount')}

INVESTOR MINDSET (what they care about):
- Is this a big market? Why now?
- What's defensible? What compounds over time?
- Show traction/momentum, not features
- Data and metrics speak louder than adjectives

TONE: Confident, data-driven, thesis-focused. Like a founder who knows their numbers cold.

CREATIVE FREEDOM: Design the visual however best tells this story.
You choose the layout, sections, and flow. No rigid template required.
Make it visually compelling - this could end up in a pitch deck.

FORBIDDEN (never include):
- "Book a demo", "Get started", "Contact sales", "Free trial"
- Customer testimonials or case study quotes
- Feature walkthroughs or how-to content
- Marketing buzzwords (revolutionary, game-changing, best-in-class)"""
        else:
            # Customer-focused storyboard (sales, internal, field crew)
            # NO badges - these are for LinkedIn/email graphics, not live demos

            # Include raw extraction for context (if available)
            raw_context = ""
            if understanding.raw_extracted_text:
                raw_context = f"""
RAW EXTRACTION (for context - transform into marketing language):
{understanding.raw_extracted_text[:500]}
Extraction Confidence: {understanding.extraction_confidence}
"""

            content_section = f"""CONTENT TO DISPLAY:
- Headline: "{understanding.headline}"
- Description: "{understanding.what_it_does}"
- Value Proposition: "{understanding.business_value}"
- For: "{understanding.who_benefits}"
- Key Benefit: "{understanding.differentiator}"
- Problem Solved: "{understanding.pain_point_addressed}"

{raw_context}

PROFESSIONAL LANGUAGE (CRITICAL):
Write in clear, direct business language - NOT marketing-speak:
- Feature names are OK to use (e.g., "Receptionist AI", "Document Engine", "Partner Portal")
- Transform technical details → business outcomes
- Remove internal codes, version numbers (BETA, V1 → just the feature name)
- Keep it SPECIFIC to what was extracted, NOT generic messaging

FORBIDDEN LANGUAGE (NEVER USE):
- "marketing campaign", "marketing strategy", "brand awareness"
- "promotional", "advertising", "drive engagement"
- "target audience", "buyer persona", "customer journey"
- "content marketing", "lead generation campaigns"
- Any internal marketing/sales team language

USE INSTEAD:
- Direct product benefits ("AI answers calls 24/7")
- Specific outcomes ("Save 5 hours/week", "Get paid 65% faster")
- Feature names ("Receptionist AI", "Document Engine")
- Customer pain points solved ("No more missed calls")

NEVER output generic copy like "Transform How You Work" or "Save time and money"
ALWAYS derive messaging from the specific content that was extracted.
If the headline says "EXTRACTION FAILED", display an error state instead.

TARGET AUDIENCE: {persona.get('title', 'Business Professional')}
TONE: {persona.get('tone', 'Professional and friendly')}"""

        # Use dynamic tagline from understanding (falls back to brand tagline if not set)
        dynamic_tagline = understanding.tagline if understanding.tagline else brand['tagline']

        # Build the image generation prompt
        prompt = f"""Create a UNIQUE professional one-page executive storyboard infographic.

GENERATION SEED: {unique_seed} (use this to create variation in layout and icons)

BRAND: {brand['company']} - "{dynamic_tagline}"

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

    def _build_language_guidelines(self, icp_preset: dict[str, Any], audience: str = "c_suite") -> str:
        """Build language guidelines string for prompts, enriched with knowledge."""
        # Get static defaults from preset
        avoid = icp_preset.get("language_style", {}).get("avoid", [])
        use = icp_preset.get("language_style", {}).get("use", [])
        tone = icp_preset.get("tone", "Friendly and professional")

        # Merge with dynamic knowledge from cache
        try:
            from src.knowledge.cache import KnowledgeCache
            cache = KnowledgeCache.get()
            if cache.is_loaded():
                knowledge = cache.get_language_guidelines(audience)
                # Knowledge terms take priority (fresher data from real conversations)
                avoid = list(set(knowledge["avoid"] + avoid))[:15]
                use = list(set(knowledge["use"] + use))[:15]
        except Exception:
            pass  # Graceful degradation - use static presets only

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
            return ""  # Graceful degradation

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