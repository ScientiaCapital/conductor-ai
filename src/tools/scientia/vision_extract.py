"""
Vision Extract Tool
==================

Connects conductor-ai to quantify-mvp for blueprint and image analysis
using Chinese Vision Language Models (Qwen VL, Kimi VL).

Performance:
- Accuracy: 98.8% on equipment photos
- Cost: $0.001 per analysis (vs $100-300 manual)
- Speed: 22 seconds per plan

Models:
- Qwen VL 30B: Equipment photos, field images
- Kimi VL A3B: PDFs and blueprints (128K context)
- DeepSeek V3.1: Normalization and parsing
"""

import httpx
import base64
from typing import Any, Optional, Literal
from pydantic import BaseModel, Field
from pathlib import Path
from ..base import BaseTool, ToolResult

# Quantify MVP service URL
QUANTIFY_MVP_URL = "http://localhost:3000"


class VisionInput(BaseModel):
    """Input schema for vision extraction."""
    
    source_type: Literal["url", "base64", "file_path"] = Field(
        ..., 
        description="Type of image source"
    )
    source: str = Field(
        ..., 
        description="URL, base64 string, or file path to image/PDF"
    )
    extraction_type: Literal["blueprint", "equipment", "field_photo", "homeowner"] = Field(
        ...,
        description="Type of extraction to perform"
    )
    trade: Optional[str] = Field(
        None,
        description="Trade category: HVAC, Electrical, Plumbing, Solar, Roofing"
    )
    additional_context: Optional[str] = Field(
        None,
        description="Additional context for extraction"
    )


class ExtractionResult(BaseModel):
    """Output schema for vision extraction."""
    
    extraction_type: str
    trade: Optional[str]
    confidence: float = Field(..., ge=0, le=1, description="Confidence score 0-1")
    raw_extraction: dict[str, Any] = Field(..., description="Raw extraction data")
    normalized_data: dict[str, Any] = Field(..., description="Normalized/structured data")
    line_items: list[dict[str, Any]] = Field(default_factory=list, description="Extracted line items")
    equipment_detected: list[str] = Field(default_factory=list, description="Detected equipment")
    model_used: str = Field(..., description="VLM model used for extraction")
    inference_cost_usd: float = Field(..., description="Cost of inference")
    cache_hit: bool = Field(False, description="Whether result was from cache")


class VisionExtractTool(BaseTool):
    """
    Vision extraction tool that connects to quantify-mvp.
    
    Uses Chinese VLM stack via OpenRouter:
    - Qwen VL 30B: 98.8% accuracy on equipment photos
    - Kimi VL A3B: PDF specialist with 128K context
    - DeepSeek V3.1: Text normalization (not vision)
    
    Features:
    - Self-learning with SHA-256 image hashing
    - Database caching for 60-70% cost reduction
    - RAG similarity search for embeddings
    - Thumbs up/down feedback integration
    """
    
    name: str = "vision_extract"
    description: str = """Extract structured data from blueprints, equipment photos, and field images.
    
    This tool:
    1. Analyzes blueprints for material takeoffs (Kimi VL)
    2. Identifies equipment from photos (Qwen VL 30B)
    3. Processes field photos for proposals (Qwen VL)
    4. Returns structured, trade-specific data
    
    Supported trades: HVAC, Electrical, Plumbing, Solar, Roofing.
    Accuracy: 98.8% on equipment identification.
    Cost: $0.001 per analysis.
    """
    
    input_schema: type[BaseModel] = VisionInput
    requires_approval: bool = False
    
    async def execute(self, input_data: VisionInput) -> ToolResult:
        """Execute vision extraction via quantify-mvp."""
        
        try:
            # Prepare image data based on source type
            image_data = await self._prepare_image(input_data)
            if image_data is None:
                return ToolResult(
                    success=False,
                    error="Failed to load image from source"
                )
            
            # Determine endpoint based on extraction type
            endpoint = self._get_endpoint(input_data.extraction_type)
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{QUANTIFY_MVP_URL}{endpoint}",
                    json={
                        "image": image_data,
                        "trade": input_data.trade,
                        "context": input_data.additional_context,
                    }
                )
                
                if response.status_code != 200:
                    return ToolResult(
                        success=False,
                        error=f"Quantify MVP returned status {response.status_code}: {response.text}"
                    )
                
                result = response.json()
                
                extraction = ExtractionResult(
                    extraction_type=input_data.extraction_type,
                    trade=input_data.trade,
                    confidence=result.get("confidence", 0.95),
                    raw_extraction=result.get("raw", {}),
                    normalized_data=result.get("normalized", {}),
                    line_items=result.get("line_items", []),
                    equipment_detected=result.get("equipment", []),
                    model_used=result.get("model", "qwen/qwen3-vl-30b-a3b-instruct"),
                    inference_cost_usd=result.get("cost", 0.001),
                    cache_hit=result.get("cache_hit", False),
                )
                
                return ToolResult(
                    success=True,
                    data=extraction.model_dump(),
                    metadata={
                        "endpoint": endpoint,
                        "cache_hit": extraction.cache_hit,
                    }
                )
                
        except httpx.TimeoutException:
            return ToolResult(
                success=False,
                error="Vision extraction timed out after 60 seconds"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Vision extraction failed: {str(e)}"
            )
    
    async def _prepare_image(self, input_data: VisionInput) -> Optional[str]:
        """Prepare image data as base64 string."""
        
        if input_data.source_type == "base64":
            return input_data.source
        
        elif input_data.source_type == "url":
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(input_data.source)
                    if response.status_code == 200:
                        return base64.b64encode(response.content).decode("utf-8")
            except Exception:
                return None
        
        elif input_data.source_type == "file_path":
            try:
                path = Path(input_data.source)
                if path.exists():
                    with open(path, "rb") as f:
                        return base64.b64encode(f.read()).decode("utf-8")
            except Exception:
                return None
        
        return None
    
    def _get_endpoint(self, extraction_type: str) -> str:
        """Get quantify-mvp endpoint for extraction type."""
        
        endpoints = {
            "blueprint": "/api/blueprint-analyze",
            "equipment": "/api/homeowner-analyze",
            "field_photo": "/api/field-process",
            "homeowner": "/api/homeowner-analyze",
        }
        return endpoints.get(extraction_type, "/api/expert/extract")
    
    async def validate_input(self, input_data: VisionInput) -> tuple[bool, Optional[str]]:
        """Validate vision input before execution."""
        
        if not input_data.source:
            return False, "Source cannot be empty"
        
        if input_data.source_type == "url":
            if not input_data.source.startswith(("http://", "https://")):
                return False, "URL must start with http:// or https://"
        
        if input_data.source_type == "file_path":
            path = Path(input_data.source)
            if not path.exists():
                return False, f"File not found: {input_data.source}"
        
        valid_trades = ["HVAC", "Electrical", "Plumbing", "Solar", "Roofing"]
        if input_data.trade and input_data.trade not in valid_trades:
            return False, f"Trade must be one of: {', '.join(valid_trades)}"
        
        return True, None
