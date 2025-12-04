"""
Knowledge Extraction using LLMs.

Extracts structured knowledge (pain points, metrics, features, etc.)
from raw content (transcripts, code, notes).
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Optional

import httpx

from src.knowledge.base import (
    ExtractionResult,
    KnowledgeEntry,
    KnowledgeSource,
    KnowledgeType,
)

logger = logging.getLogger(__name__)


@dataclass
class ExtractorConfig:
    """Configuration for knowledge extraction."""

    # Model selection (DeepSeek V3.2 for text extraction)
    openrouter_api_key: str = ""
    model: str = "deepseek/deepseek-chat-v3"

    # Extraction settings
    temperature: float = 0.3  # Low for consistent extraction
    max_tokens: int = 4096

    def __post_init__(self):
        if not self.openrouter_api_key:
            self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")


EXTRACTION_PROMPT = """You are a knowledge extraction specialist for Coperniq, a software platform for MEP contractors (plumbers, electricians, HVAC, solar installers).

Your job is to extract specific, reusable knowledge from the content provided. This knowledge will be used to generate marketing materials and storyboards.

CONTENT TO ANALYZE:
---
{content}
---

CONTEXT (if available): {context}

EXTRACTION INSTRUCTIONS:

1. PAIN POINTS - Extract specific customer pain points mentioned:
   - Use EXACT words when possible ("we lose $3K per job")
   - Include specific numbers, timeframes, frustrations
   - Focus on problems Coperniq solves

2. METRICS - Extract any numbers, statistics, or quantifiable data:
   - Dollar amounts ("$3K/job", "$50K in change orders")
   - Time ("5 hours/week", "2 days per permit")
   - Percentages ("65% faster", "30% more profitable")
   - Counts ("200 jobs per year", "15 technicians")

3. QUOTES - Extract verbatim quotes that are powerful/reusable:
   - Must be exact words (or very close)
   - Should express pain, desire, or satisfaction clearly
   - Note the speaker's role if known

4. FEATURES - Extract Coperniq features or product areas mentioned:
   - Feature names (Receptionist AI, Document Engine)
   - Product areas (Intelligence, Sales Cloud, PM Cloud)
   - Capabilities described

5. APPROVED TERMS - Extract language that resonated well:
   - Phrases the customer agreed with or repeated
   - Simple, benefit-focused language
   - Avoid jargon

6. OBJECTIONS - Extract sales objections or concerns:
   - Price concerns
   - Integration worries
   - Change resistance
   - Competitor comparisons

7. COMPETITOR mentions - Note any competitors mentioned:
   - Name and context
   - What they do well/poorly (from customer view)

8. USE CASES - Extract specific use cases discussed:
   - Industry + problem + solution pattern
   - Example: "solar permit tracking"

RESPONSE FORMAT (JSON):
{{
    "extractions": [
        {{
            "knowledge_type": "pain_point|metric|quote|feature|approved_term|objection|competitor|use_case",
            "content": "The extracted knowledge (verbatim when applicable)",
            "context": "Surrounding context or explanation",
            "verbatim": true/false,
            "confidence_score": 0.0-1.0,
            "speaker_name": "Name if known",
            "speaker_role": "Role if known (CEO, PM, Owner, etc.)",
            "company_name": "Company if known",
            "audience": ["c_suite", "business_owner", "btl_champion", "field_crew"],
            "industries": ["solar", "hvac", "electrical", "plumbing", "roofing", "mep"],
            "product_areas": ["Intelligence", "Sales Cloud", "PM Cloud", "Financial Cloud", "Asset Cloud"]
        }}
    ],
    "summary": "Brief summary of what was extracted"
}}

QUALITY RULES:
- Only extract knowledge that is SPECIFIC and REUSABLE
- Prefer EXACT QUOTES over paraphrasing
- Include confidence_score (0.7+ for clear extractions, lower for inferred)
- Skip generic or vague content
- Focus on contractor-relevant insights

Return valid JSON only, no markdown."""


class KnowledgeExtractor:
    """
    Extracts knowledge from raw content using LLMs.
    """

    def __init__(self, config: Optional[ExtractorConfig] = None):
        self.config = config or ExtractorConfig()

    async def extract(
        self,
        source: KnowledgeSource,
        content: Optional[str] = None,
        additional_context: str = "",
    ) -> ExtractionResult:
        """
        Extract knowledge from a source.

        Args:
            source: The knowledge source metadata
            content: Content to extract from (uses source.raw_content if not provided)
            additional_context: Additional context to help extraction

        Returns:
            ExtractionResult with extracted entries
        """
        start_time = perf_counter()

        content = content or source.raw_content
        if not content:
            return ExtractionResult(
                source_id=source.id,
                error="No content to extract from",
                execution_time_ms=0,
            )

        # Build context string
        context_parts = []
        if source.source_title:
            context_parts.append(f"Source: {source.source_title}")
        if source.source_type:
            context_parts.append(f"Type: {source.source_type.value}")
        if source.participant_names:
            context_parts.append(f"Participants: {', '.join(source.participant_names)}")
        if additional_context:
            context_parts.append(additional_context)
        context_str = " | ".join(context_parts) if context_parts else "No additional context"

        # Build prompt
        prompt = EXTRACTION_PROMPT.format(
            content=content[:15000],  # Limit content length
            context=context_str,
        )

        try:
            # Call LLM via OpenRouter
            response = await self._call_llm(prompt)

            # Parse response
            entries = self._parse_extraction_response(response, source)

            execution_time_ms = int((perf_counter() - start_time) * 1000)

            return ExtractionResult(
                source_id=source.id,
                items_extracted=len(entries),
                entries=entries,
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            logger.exception(f"Extraction failed: {e}")
            return ExtractionResult(
                source_id=source.id,
                error=str(e),
                execution_time_ms=int((perf_counter() - start_time) * 1000),
            )

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM via OpenRouter."""
        if not self.config.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY not configured")

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.config.openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.config.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    def _parse_extraction_response(
        self,
        response: str,
        source: KnowledgeSource,
    ) -> list[KnowledgeEntry]:
        """Parse LLM response into KnowledgeEntry objects."""
        entries = []

        try:
            # Clean response (remove markdown if present)
            json_str = response.strip()
            if json_str.startswith("```"):
                parts = json_str.split("```")
                if len(parts) >= 2:
                    json_str = parts[1]
                    if json_str.startswith("json"):
                        json_str = json_str[4:]
                json_str = json_str.strip()

            # Repair truncated JSON
            json_str = self._repair_json(json_str)

            data = json.loads(json_str)
            extractions = data.get("extractions", [])

            for item in extractions:
                try:
                    knowledge_type_str = item.get("knowledge_type", "").lower()

                    # Map to enum
                    type_map = {
                        "pain_point": KnowledgeType.PAIN_POINT,
                        "metric": KnowledgeType.METRIC,
                        "quote": KnowledgeType.QUOTE,
                        "feature": KnowledgeType.FEATURE,
                        "approved_term": KnowledgeType.APPROVED_TERM,
                        "objection": KnowledgeType.OBJECTION,
                        "competitor": KnowledgeType.COMPETITOR,
                        "use_case": KnowledgeType.USE_CASE,
                        "success_story": KnowledgeType.SUCCESS_STORY,
                    }

                    knowledge_type = type_map.get(knowledge_type_str)
                    if not knowledge_type:
                        logger.warning(f"Unknown knowledge type: {knowledge_type_str}")
                        continue

                    content = item.get("content", "").strip()
                    if not content:
                        continue

                    entry = KnowledgeEntry(
                        knowledge_type=knowledge_type,
                        content=content,
                        context=item.get("context"),
                        verbatim=item.get("verbatim", False),
                        confidence_score=float(item.get("confidence_score", 0.8)),
                        speaker_name=item.get("speaker_name"),
                        speaker_role=item.get("speaker_role"),
                        company_name=item.get("company_name"),
                        audience=item.get("audience", []),
                        industries=item.get("industries", []),
                        product_areas=item.get("product_areas", []),
                        source_id=source.id,
                    )
                    entries.append(entry)

                except Exception as e:
                    logger.warning(f"Failed to parse extraction item: {e}")
                    continue

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse extraction JSON: {e}")
            logger.debug(f"Raw response: {response[:500]}")

        return entries

    def _repair_json(self, json_str: str) -> str:
        """Attempt to repair truncated or malformed JSON."""
        # Fix unterminated strings
        quote_count = json_str.count('"') - json_str.count('\\"')
        if quote_count % 2 == 1:
            json_str = json_str + '"'

        # Add missing closing brackets
        open_brackets = json_str.count('[') - json_str.count(']')
        if open_brackets > 0:
            json_str = json_str + ']' * open_brackets

        # Add missing closing braces
        open_braces = json_str.count('{') - json_str.count('}')
        if open_braces > 0:
            json_str = json_str + '}' * open_braces

        # Remove trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)

        return json_str
