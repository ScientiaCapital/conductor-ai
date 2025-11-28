"""
Cost Optimizer Tool
==================

Routes LLM calls through ai-cost-optimizer for 40-70% savings
via intelligent model selection and caching.

Models Supported:
- Anthropic (Claude 4.5 Opus/Sonnet/Haiku)
- Google (Gemini 2.0)
- OpenRouter (40+ models including Chinese LLMs)
- Cerebras (70x inference speed)
- DeepSeek (200x cheaper than GPT-4)

Routing Strategies:
- cost: Minimize cost
- speed: Minimize latency
- quality: Maximize quality
- balanced: Smart tradeoffs
"""

import httpx
from typing import Any, Optional, Literal
from pydantic import BaseModel, Field
from ..base import BaseTool, ToolResult

# AI Cost Optimizer service URL
COST_OPTIMIZER_URL = "http://localhost:8004"


class LLMRequestInput(BaseModel):
    """Input schema for LLM request routing."""
    
    prompt: str = Field(..., min_length=1, description="Prompt to send to LLM")
    system_prompt: Optional[str] = Field(None, description="System prompt")
    strategy: Literal["cost", "speed", "quality", "balanced"] = Field(
        default="balanced",
        description="Routing strategy"
    )
    max_tokens: int = Field(default=1000, ge=1, le=32000, description="Maximum tokens")
    temperature: float = Field(default=0.7, ge=0, le=2, description="Temperature")
    preferred_provider: Optional[str] = Field(
        None,
        description="Preferred provider: anthropic, google, openrouter, cerebras, deepseek"
    )
    task_type: Optional[str] = Field(
        None,
        description="Task type for smart routing: coding, analysis, creative, extraction, chat"
    )
    budget_limit_usd: Optional[float] = Field(
        None,
        description="Maximum cost for this request in USD"
    )


class LLMResponseResult(BaseModel):
    """Output schema for LLM response."""
    
    content: str = Field(..., description="LLM response content")
    model_used: str = Field(..., description="Model that was selected")
    provider: str = Field(..., description="Provider used")
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    savings_usd: float = Field(..., description="Savings vs default model")
    savings_percent: float = Field(..., description="Percentage savings")
    cache_hit: bool = False
    routing_reason: str = Field(..., description="Why this model was selected")


class CostOptimizerTool(BaseTool):
    """
    LLM cost optimization tool via ai-cost-optimizer.
    
    Routes requests across 40+ models from 8 providers
    to achieve 40-70% cost savings while maintaining quality.
    
    Routing strategies:
    - cost: Use cheapest model that can handle task
    - speed: Use fastest model (Cerebras)
    - quality: Use highest quality model
    - balanced: Smart tradeoffs based on task type
    
    Special optimizations:
    - Chinese LLMs for simple tasks (200x cheaper)
    - Cerebras for real-time requirements
    - Caching for repeated queries
    """
    
    name: str = "cost_optimizer"
    description: str = """Route LLM requests through cost optimizer for 40-70% savings.
    
    This tool:
    1. Analyzes your request requirements
    2. Selects optimal model based on strategy
    3. Routes through caching layer
    4. Returns response with cost tracking
    
    Strategies:
    - cost: Cheapest model (DeepSeek, Qwen)
    - speed: Fastest model (Cerebras 70x)
    - quality: Best model (Claude 4.5 Opus)
    - balanced: Smart selection based on task
    """
    
    input_schema: type[BaseModel] = LLMRequestInput
    requires_approval: bool = False
    
    async def execute(self, input_data: LLMRequestInput) -> ToolResult:
        """Route LLM request through cost optimizer."""
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{COST_OPTIMIZER_URL}/api/v1/chat/completions",
                    json={
                        "messages": [
                            {"role": "system", "content": input_data.system_prompt or "You are a helpful assistant."},
                            {"role": "user", "content": input_data.prompt},
                        ],
                        "strategy": input_data.strategy,
                        "max_tokens": input_data.max_tokens,
                        "temperature": input_data.temperature,
                        "preferred_provider": input_data.preferred_provider,
                        "task_type": input_data.task_type,
                        "budget_limit": input_data.budget_limit_usd,
                    }
                )
                
                if response.status_code != 200:
                    return ToolResult(
                        success=False,
                        error=f"Cost optimizer returned status {response.status_code}"
                    )
                
                result = response.json()
                
                llm_response = LLMResponseResult(
                    content=result.get("choices", [{}])[0].get("message", {}).get("content", ""),
                    model_used=result.get("model", "unknown"),
                    provider=result.get("provider", "unknown"),
                    input_tokens=result.get("usage", {}).get("prompt_tokens", 0),
                    output_tokens=result.get("usage", {}).get("completion_tokens", 0),
                    cost_usd=result.get("cost", 0),
                    latency_ms=result.get("latency_ms", 0),
                    savings_usd=result.get("savings_usd", 0),
                    savings_percent=result.get("savings_percent", 0),
                    cache_hit=result.get("cache_hit", False),
                    routing_reason=result.get("routing_reason", ""),
                )
                
                return ToolResult(
                    success=True,
                    data=llm_response.model_dump(),
                    metadata={
                        "strategy": input_data.strategy,
                        "cache_hit": llm_response.cache_hit,
                    }
                )
                
        except httpx.TimeoutException:
            return ToolResult(
                success=False,
                error="LLM request timed out after 120 seconds"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Cost optimizer failed: {str(e)}"
            )
    
    async def validate_input(self, input_data: LLMRequestInput) -> tuple[bool, Optional[str]]:
        """Validate LLM request input."""
        
        if input_data.budget_limit_usd is not None and input_data.budget_limit_usd <= 0:
            return False, "Budget limit must be positive"
        
        valid_providers = ["anthropic", "google", "openrouter", "cerebras", "deepseek", "ollama"]
        if input_data.preferred_provider and input_data.preferred_provider not in valid_providers:
            return False, f"Invalid provider. Valid: {', '.join(valid_providers)}"
        
        return True, None
