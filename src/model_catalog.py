"""
Model Catalog Module

Unified catalog of all available models across providers.
Provides OpenAI-compatible /v1/models endpoint.
"""

import os
import time
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO)


@dataclass
class ModelInfo:
    """Information about a model"""
    id: str
    provider: str
    owned_by: str

    # Capabilities
    context_window: int = 4096
    max_output_tokens: int = 4096
    supports_vision: bool = False
    supports_functions: bool = False
    supports_streaming: bool = True
    supports_json_mode: bool = False

    # Pricing (per 1M tokens)
    input_price_per_million: float = 0.0
    output_price_per_million: float = 0.0

    # Quality indicators
    quality_tier: str = "standard"  # budget, standard, premium, flagship
    best_for: List[str] = field(default_factory=list)

    # Availability
    available: bool = True
    deprecated: bool = False
    deprecation_date: Optional[str] = None

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI /v1/models format"""
        return {
            "id": self.id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": self.owned_by,
            "permission": [],
            "root": self.id,
            "parent": None,
            # Extended fields
            "provider": self.provider,
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "capabilities": {
                "vision": self.supports_vision,
                "function_calling": self.supports_functions,
                "streaming": self.supports_streaming,
                "json_mode": self.supports_json_mode,
            },
            "pricing": {
                "input_per_million": self.input_price_per_million,
                "output_per_million": self.output_price_per_million,
            },
            "quality_tier": self.quality_tier,
            "best_for": self.best_for,
        }


class ModelCatalog:
    """
    Unified catalog of all available models.

    Configure via environment variables:
    - CATALOG_INCLUDE_PRICING: Include pricing info (default: true)
    - CATALOG_INCLUDE_LOCAL: Include local vLLM models (default: true)
    """

    def __init__(self):
        self.include_pricing = os.getenv("CATALOG_INCLUDE_PRICING", "true").lower() == "true"
        self.include_local = os.getenv("CATALOG_INCLUDE_LOCAL", "true").lower() == "true"

        self._models: Dict[str, ModelInfo] = {}
        self._load_models()

        logging.info(f"[CATALOG] Loaded {len(self._models)} models")

    def _load_models(self):
        """Load all model definitions"""

        # OpenAI Models
        self._add_model(ModelInfo(
            id="gpt-4o",
            provider="openai",
            owned_by="openai",
            context_window=128000,
            max_output_tokens=16384,
            supports_vision=True,
            supports_functions=True,
            supports_json_mode=True,
            input_price_per_million=2.50,
            output_price_per_million=10.00,
            quality_tier="flagship",
            best_for=["general", "vision", "coding", "analysis"],
        ))

        self._add_model(ModelInfo(
            id="gpt-4o-mini",
            provider="openai",
            owned_by="openai",
            context_window=128000,
            max_output_tokens=16384,
            supports_vision=True,
            supports_functions=True,
            supports_json_mode=True,
            input_price_per_million=0.15,
            output_price_per_million=0.60,
            quality_tier="standard",
            best_for=["general", "fast", "cost-effective"],
        ))

        self._add_model(ModelInfo(
            id="gpt-4-turbo",
            provider="openai",
            owned_by="openai",
            context_window=128000,
            max_output_tokens=4096,
            supports_vision=True,
            supports_functions=True,
            supports_json_mode=True,
            input_price_per_million=10.00,
            output_price_per_million=30.00,
            quality_tier="premium",
            best_for=["complex-tasks", "long-context"],
        ))

        self._add_model(ModelInfo(
            id="gpt-3.5-turbo",
            provider="openai",
            owned_by="openai",
            context_window=16384,
            max_output_tokens=4096,
            supports_functions=True,
            supports_json_mode=True,
            input_price_per_million=0.50,
            output_price_per_million=1.50,
            quality_tier="budget",
            best_for=["simple-tasks", "high-volume"],
        ))

        self._add_model(ModelInfo(
            id="o1-preview",
            provider="openai",
            owned_by="openai",
            context_window=128000,
            max_output_tokens=32768,
            input_price_per_million=15.00,
            output_price_per_million=60.00,
            quality_tier="flagship",
            best_for=["reasoning", "math", "science", "coding"],
        ))

        self._add_model(ModelInfo(
            id="o1-mini",
            provider="openai",
            owned_by="openai",
            context_window=128000,
            max_output_tokens=65536,
            input_price_per_million=3.00,
            output_price_per_million=12.00,
            quality_tier="premium",
            best_for=["reasoning", "coding", "math"],
        ))

        # Anthropic Models
        self._add_model(ModelInfo(
            id="claude-3-5-sonnet-20241022",
            provider="anthropic",
            owned_by="anthropic",
            context_window=200000,
            max_output_tokens=8192,
            supports_vision=True,
            input_price_per_million=3.00,
            output_price_per_million=15.00,
            quality_tier="flagship",
            best_for=["coding", "analysis", "writing", "vision"],
        ))

        self._add_model(ModelInfo(
            id="claude-3-5-haiku-20241022",
            provider="anthropic",
            owned_by="anthropic",
            context_window=200000,
            max_output_tokens=8192,
            supports_vision=True,
            input_price_per_million=0.25,
            output_price_per_million=1.25,
            quality_tier="standard",
            best_for=["fast", "cost-effective", "simple-tasks"],
        ))

        self._add_model(ModelInfo(
            id="claude-3-opus-20240229",
            provider="anthropic",
            owned_by="anthropic",
            context_window=200000,
            max_output_tokens=4096,
            supports_vision=True,
            input_price_per_million=15.00,
            output_price_per_million=75.00,
            quality_tier="flagship",
            best_for=["complex-analysis", "research", "nuanced-writing"],
        ))

        # Groq Models (Fast inference)
        self._add_model(ModelInfo(
            id="llama-3.3-70b-versatile",
            provider="groq",
            owned_by="meta",
            context_window=32768,
            max_output_tokens=8192,
            input_price_per_million=0.59,
            output_price_per_million=0.79,
            quality_tier="standard",
            best_for=["fast", "general", "open-source"],
        ))

        self._add_model(ModelInfo(
            id="llama-3.1-8b-instant",
            provider="groq",
            owned_by="meta",
            context_window=8192,
            max_output_tokens=8192,
            input_price_per_million=0.05,
            output_price_per_million=0.08,
            quality_tier="budget",
            best_for=["ultra-fast", "simple-tasks", "high-volume"],
        ))

        self._add_model(ModelInfo(
            id="mixtral-8x7b-32768",
            provider="groq",
            owned_by="mistral",
            context_window=32768,
            max_output_tokens=8192,
            input_price_per_million=0.24,
            output_price_per_million=0.24,
            quality_tier="standard",
            best_for=["fast", "multilingual", "coding"],
        ))

        self._add_model(ModelInfo(
            id="gemma2-9b-it",
            provider="groq",
            owned_by="google",
            context_window=8192,
            max_output_tokens=8192,
            input_price_per_million=0.20,
            output_price_per_million=0.20,
            quality_tier="budget",
            best_for=["fast", "instruction-following"],
        ))

    def _add_model(self, model: ModelInfo):
        """Add a model to the catalog"""
        self._models[model.id] = model

    def add_local_model(self, model_id: str, context_window: int = 4096):
        """Add a local vLLM model to the catalog"""
        if not self.include_local:
            return

        self._models[model_id] = ModelInfo(
            id=model_id,
            provider="local",
            owned_by="local",
            context_window=context_window,
            max_output_tokens=context_window,
            supports_streaming=True,
            input_price_per_million=0.0,  # Self-hosted
            output_price_per_million=0.0,
            quality_tier="standard",
            best_for=["self-hosted", "privacy", "custom"],
        )

    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get a specific model"""
        return self._models.get(model_id)

    def list_models(
        self,
        provider: Optional[str] = None,
        quality_tier: Optional[str] = None,
        supports_vision: Optional[bool] = None,
        max_price: Optional[float] = None
    ) -> List[ModelInfo]:
        """List models with optional filtering"""
        models = list(self._models.values())

        if provider:
            models = [m for m in models if m.provider == provider]

        if quality_tier:
            models = [m for m in models if m.quality_tier == quality_tier]

        if supports_vision is not None:
            models = [m for m in models if m.supports_vision == supports_vision]

        if max_price is not None:
            models = [m for m in models if m.input_price_per_million <= max_price]

        return models

    def get_openai_models_response(self) -> Dict[str, Any]:
        """Get response in OpenAI /v1/models format"""
        models = [m.to_openai_format() for m in self._models.values()]

        return {
            "object": "list",
            "data": models,
        }

    def compare_models(self, model_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple models"""
        comparison = []

        for model_id in model_ids:
            model = self._models.get(model_id)
            if model:
                comparison.append({
                    "id": model.id,
                    "provider": model.provider,
                    "context_window": model.context_window,
                    "input_price": model.input_price_per_million,
                    "output_price": model.output_price_per_million,
                    "quality_tier": model.quality_tier,
                    "supports_vision": model.supports_vision,
                    "supports_functions": model.supports_functions,
                })

        return {
            "models": comparison,
            "count": len(comparison),
        }

    def recommend_for_task(self, task_type: str) -> List[ModelInfo]:
        """Recommend models for a specific task type"""
        recommendations = []

        for model in self._models.values():
            if task_type.lower() in [t.lower() for t in model.best_for]:
                recommendations.append(model)

        # Sort by quality tier
        tier_order = {"flagship": 0, "premium": 1, "standard": 2, "budget": 3}
        recommendations.sort(key=lambda m: tier_order.get(m.quality_tier, 99))

        return recommendations


# Global catalog
_catalog: Optional[ModelCatalog] = None


def get_model_catalog() -> ModelCatalog:
    """Get or create the global model catalog"""
    global _catalog
    if _catalog is None:
        _catalog = ModelCatalog()
    return _catalog


def handle_models_list_request(
    provider: Optional[str] = None,
    quality_tier: Optional[str] = None
) -> Dict[str, Any]:
    """Handle /v1/models request"""
    catalog = get_model_catalog()

    if provider or quality_tier:
        models = catalog.list_models(provider=provider, quality_tier=quality_tier)
        return {
            "object": "list",
            "data": [m.to_openai_format() for m in models],
        }

    return catalog.get_openai_models_response()


def handle_models_retrieve_request(model_id: str) -> Dict[str, Any]:
    """Handle /v1/models/{model} request"""
    catalog = get_model_catalog()
    model = catalog.get_model(model_id)

    if model:
        return model.to_openai_format()
    else:
        return {"error": "model_not_found", "model": model_id}


def handle_models_compare_request(model_ids: List[str]) -> Dict[str, Any]:
    """Handle /models/compare request"""
    catalog = get_model_catalog()
    return catalog.compare_models(model_ids)


def handle_models_recommend_request(task_type: str) -> Dict[str, Any]:
    """Handle /models/recommend request"""
    catalog = get_model_catalog()
    models = catalog.recommend_for_task(task_type)
    return {
        "task_type": task_type,
        "recommendations": [m.to_openai_format() for m in models[:5]],
        "count": len(models),
    }
