"""
Provider Adapter Framework

Abstract interface for multiple LLM providers.
Enables unified API across OpenAI, Anthropic, Google, Mistral, and local vLLM.
"""

import os
import time
import asyncio
import logging
import aiohttp
import json
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logging.basicConfig(level=logging.INFO)


class ProviderType(Enum):
    """Supported provider types"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    MISTRAL = "mistral"
    GROQ = "groq"
    LOCAL = "local"  # Local vLLM


@dataclass
class ProviderConfig:
    """Configuration for a provider"""
    name: str
    provider_type: ProviderType
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    enabled: bool = True

    # Rate limiting
    max_rpm: int = 0  # 0 = unlimited
    max_tpm: int = 0

    # Timeouts
    timeout_seconds: int = 120

    # Cost tracking
    input_cost_per_1k: float = 0.0
    output_cost_per_1k: float = 0.0


@dataclass
class ProviderResponse:
    """Unified response from any provider"""
    content: str
    model: str
    provider: str

    # Usage
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Cost
    cost_cents: float = 0.0

    # Metadata
    finish_reason: Optional[str] = None
    latency_ms: float = 0.0

    # Raw response for debugging
    raw_response: Optional[Dict] = None

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI-compatible response format"""
        return {
            "id": f"chatcmpl-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": self.content,
                },
                "finish_reason": self.finish_reason or "stop",
            }],
            "usage": {
                "prompt_tokens": self.input_tokens,
                "completion_tokens": self.output_tokens,
                "total_tokens": self.total_tokens,
            },
            "provider": self.provider,
            "cost_cents": self.cost_cents,
        }


class ProviderAdapter(ABC):
    """Abstract base class for provider adapters"""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.name = config.name

        # Statistics
        self.total_requests = 0
        self.total_tokens = 0
        self.total_cost_cents = 0.0
        self.total_errors = 0
        self.total_latency_ms = 0.0

    @abstractmethod
    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> ProviderResponse:
        """Execute a completion request"""
        pass

    @abstractmethod
    async def stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream a completion request"""
        pass

    @abstractmethod
    def list_models(self) -> List[str]:
        """List available models for this provider"""
        pass

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in cents"""
        input_cost = (input_tokens / 1000) * self.config.input_cost_per_1k
        output_cost = (output_tokens / 1000) * self.config.output_cost_per_1k
        return input_cost + output_cost

    def record_request(self, response: ProviderResponse):
        """Record request statistics"""
        self.total_requests += 1
        self.total_tokens += response.total_tokens
        self.total_cost_cents += response.cost_cents
        self.total_latency_ms += response.latency_ms

    def get_stats(self) -> Dict[str, Any]:
        """Get provider statistics"""
        return {
            "name": self.name,
            "type": self.config.provider_type.value,
            "enabled": self.config.enabled,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_cost_cents": round(self.total_cost_cents, 2),
            "total_errors": self.total_errors,
            "avg_latency_ms": round(self.total_latency_ms / self.total_requests, 2) if self.total_requests else 0,
        }


class OpenAIAdapter(ProviderAdapter):
    """Adapter for OpenAI API"""

    MODELS = [
        "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4",
        "gpt-3.5-turbo", "gpt-3.5-turbo-16k",
        "o1-preview", "o1-mini",
    ]

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.base_url = config.base_url or "https://api.openai.com/v1"
        self.api_key = config.api_key or os.getenv("OPENAI_API_KEY")

    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> ProviderResponse:
        start_time = time.time()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            ) as resp:
                data = await resp.json()

                if resp.status >= 400:
                    self.total_errors += 1
                    raise ProviderError(f"OpenAI error: {data.get('error', {}).get('message', 'Unknown')}")

        latency_ms = (time.time() - start_time) * 1000

        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        response = ProviderResponse(
            content=data["choices"][0]["message"]["content"],
            model=model,
            provider="openai",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_cents=self.calculate_cost(input_tokens, output_tokens),
            finish_reason=data["choices"][0].get("finish_reason"),
            latency_ms=latency_ms,
            raw_response=data,
        )

        self.record_request(response)
        return response

    async def stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            ) as resp:
                async for line in resp.content:
                    line = line.decode().strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        data = json.loads(line[6:])
                        content = data["choices"][0].get("delta", {}).get("content", "")
                        if content:
                            yield content

    def list_models(self) -> List[str]:
        return self.MODELS


class AnthropicAdapter(ProviderAdapter):
    """Adapter for Anthropic API"""

    MODELS = [
        "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307",
    ]

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.base_url = config.base_url or "https://api.anthropic.com/v1"
        self.api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")

    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> ProviderResponse:
        start_time = time.time()

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        # Convert OpenAI format to Anthropic format
        system_msg = ""
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                anthropic_messages.append(msg)

        payload = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens or 4096,
        }
        if system_msg:
            payload["system"] = system_msg
        if temperature != 1.0:
            payload["temperature"] = temperature

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/messages",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            ) as resp:
                data = await resp.json()

                if resp.status >= 400:
                    self.total_errors += 1
                    raise ProviderError(f"Anthropic error: {data.get('error', {}).get('message', 'Unknown')}")

        latency_ms = (time.time() - start_time) * 1000

        usage = data.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        content = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")

        response = ProviderResponse(
            content=content,
            model=model,
            provider="anthropic",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_cents=self.calculate_cost(input_tokens, output_tokens),
            finish_reason=data.get("stop_reason"),
            latency_ms=latency_ms,
            raw_response=data,
        )

        self.record_request(response)
        return response

    async def stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        # Similar to complete but with streaming
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        system_msg = ""
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                anthropic_messages.append(msg)

        payload = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens or 4096,
            "stream": True,
        }
        if system_msg:
            payload["system"] = system_msg

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/messages",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            ) as resp:
                async for line in resp.content:
                    line = line.decode().strip()
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if data.get("type") == "content_block_delta":
                            content = data.get("delta", {}).get("text", "")
                            if content:
                                yield content

    def list_models(self) -> List[str]:
        return self.MODELS


class GroqAdapter(ProviderAdapter):
    """Adapter for Groq API (fast inference)"""

    MODELS = [
        "llama-3.3-70b-versatile", "llama-3.1-8b-instant",
        "mixtral-8x7b-32768", "gemma2-9b-it",
    ]

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.base_url = config.base_url or "https://api.groq.com/openai/v1"
        self.api_key = config.api_key or os.getenv("GROQ_API_KEY")

    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> ProviderResponse:
        # Groq uses OpenAI-compatible API
        start_time = time.time()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            ) as resp:
                data = await resp.json()

                if resp.status >= 400:
                    self.total_errors += 1
                    raise ProviderError(f"Groq error: {data.get('error', {}).get('message', 'Unknown')}")

        latency_ms = (time.time() - start_time) * 1000

        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        response = ProviderResponse(
            content=data["choices"][0]["message"]["content"],
            model=model,
            provider="groq",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_cents=self.calculate_cost(input_tokens, output_tokens),
            finish_reason=data["choices"][0].get("finish_reason"),
            latency_ms=latency_ms,
            raw_response=data,
        )

        self.record_request(response)
        return response

    async def stream(self, messages, model, **kwargs) -> AsyncGenerator[str, None]:
        # Similar to OpenAI streaming
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            ) as resp:
                async for line in resp.content:
                    line = line.decode().strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        data = json.loads(line[6:])
                        content = data["choices"][0].get("delta", {}).get("content", "")
                        if content:
                            yield content

    def list_models(self) -> List[str]:
        return self.MODELS


class ProviderError(Exception):
    """Error from a provider"""
    pass


class ProviderManager:
    """
    Manages multiple LLM providers.

    Configure via environment variables:
    - PROVIDERS_CONFIG: JSON config for providers (default: auto-detect from API keys)
    - OPENAI_API_KEY: OpenAI API key
    - ANTHROPIC_API_KEY: Anthropic API key
    - GROQ_API_KEY: Groq API key
    """

    def __init__(self):
        self._providers: Dict[str, ProviderAdapter] = {}
        self._model_to_provider: Dict[str, str] = {}
        self._lock = asyncio.Lock()

        # Auto-configure from environment
        self._auto_configure()

        logging.info(f"[PROVIDERS] Initialized {len(self._providers)} providers")

    def _auto_configure(self):
        """Auto-configure providers from environment variables"""

        # OpenAI
        if os.getenv("OPENAI_API_KEY"):
            config = ProviderConfig(
                name="openai",
                provider_type=ProviderType.OPENAI,
                api_key=os.getenv("OPENAI_API_KEY"),
                input_cost_per_1k=0.5,  # $0.50/1M = $0.0005/1K
                output_cost_per_1k=1.5,
            )
            self.register_provider(OpenAIAdapter(config))

        # Anthropic
        if os.getenv("ANTHROPIC_API_KEY"):
            config = ProviderConfig(
                name="anthropic",
                provider_type=ProviderType.ANTHROPIC,
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                input_cost_per_1k=0.3,
                output_cost_per_1k=1.5,
            )
            self.register_provider(AnthropicAdapter(config))

        # Groq
        if os.getenv("GROQ_API_KEY"):
            config = ProviderConfig(
                name="groq",
                provider_type=ProviderType.GROQ,
                api_key=os.getenv("GROQ_API_KEY"),
                input_cost_per_1k=0.05,
                output_cost_per_1k=0.08,
            )
            self.register_provider(GroqAdapter(config))

    def register_provider(self, adapter: ProviderAdapter):
        """Register a provider adapter"""
        self._providers[adapter.name] = adapter

        # Map models to provider
        for model in adapter.list_models():
            self._model_to_provider[model] = adapter.name

        logging.info(f"[PROVIDERS] Registered {adapter.name} with {len(adapter.list_models())} models")

    def get_provider_for_model(self, model: str) -> Optional[ProviderAdapter]:
        """Get the provider adapter for a model"""
        provider_name = self._model_to_provider.get(model)
        if provider_name:
            return self._providers.get(provider_name)
        return None

    def get_provider(self, name: str) -> Optional[ProviderAdapter]:
        """Get a provider by name"""
        return self._providers.get(name)

    def list_all_models(self) -> List[Dict[str, Any]]:
        """List all available models across providers"""
        models = []
        for provider_name, adapter in self._providers.items():
            for model in adapter.list_models():
                models.append({
                    "id": model,
                    "provider": provider_name,
                    "input_cost_per_1k": adapter.config.input_cost_per_1k,
                    "output_cost_per_1k": adapter.config.output_cost_per_1k,
                })
        return models

    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all providers"""
        return {
            name: adapter.get_stats()
            for name, adapter in self._providers.items()
        }

    async def complete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> ProviderResponse:
        """Route a completion to the appropriate provider"""
        adapter = self.get_provider_for_model(model)
        if not adapter:
            raise ProviderError(f"No provider found for model: {model}")

        return await adapter.complete(messages, model, **kwargs)


# Global provider manager
_provider_manager: Optional[ProviderManager] = None


def get_provider_manager() -> ProviderManager:
    """Get or create the global provider manager"""
    global _provider_manager
    if _provider_manager is None:
        _provider_manager = ProviderManager()
    return _provider_manager


def handle_providers_stats_request() -> Dict[str, Any]:
    """Handle /providers/stats request"""
    manager = get_provider_manager()
    return {
        "providers": manager.get_all_stats(),
        "total_providers": len(manager._providers),
    }


def handle_providers_models_request() -> Dict[str, Any]:
    """Handle /providers/models request"""
    manager = get_provider_manager()
    models = manager.list_all_models()
    return {
        "models": models,
        "total_models": len(models),
    }
