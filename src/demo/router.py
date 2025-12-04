"""Demo router for interactive storyboard generation.

Endpoints:
- GET /demo/examples - List available example code files
- GET /demo/examples/{name} - Get code content for an example
- POST /demo/generate - Generate storyboard from image or code
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.tools.storyboard.unified_storyboard import UnifiedStoryboardTool

logger = logging.getLogger(__name__)

# ============================================================================
# Router Setup
# ============================================================================

router = APIRouter(prefix="/demo", tags=["demo"])

# ============================================================================
# Example Code Files
# ============================================================================

EXAMPLES = [
    {
        "name": "video_script_generator",
        "path": "src/tools/video/video_script_generator.py",
        "description": "AI-powered video script generation using DeepSeek V3",
    },
    {
        "name": "unified_storyboard",
        "path": "src/tools/storyboard/unified_storyboard.py",
        "description": "Convert any input to PNG storyboard via Gemini",
    },
    {
        "name": "video_scheduler",
        "path": "src/tools/video/video_scheduler.py",
        "description": "Optimal send time prediction for video prospecting",
    },
    {
        "name": "video_analytics",
        "path": "src/tools/video/video_analytics.py",
        "description": "Loom view tracking and engagement scoring",
    },
    {
        "name": "gemini_client",
        "path": "src/tools/storyboard/gemini_client.py",
        "description": "Gemini Vision + Image Generation client",
    },
    {
        "name": "video_generator",
        "path": "src/tools/video/video_generator.py",
        "description": "Multi-provider video generation (Kling/HaiLuo/Runway)",
    },
    {
        "name": "video_template_manager",
        "path": "src/tools/video/video_template_manager.py",
        "description": "Industry-specific video templates (solar, hvac, electrical)",
    },
]

# ============================================================================
# Request/Response Models
# ============================================================================


class ExampleInfo(BaseModel):
    """Example code file information."""

    name: str = Field(..., description="Example name")
    path: str = Field(..., description="Relative path to file")
    description: str = Field(..., description="Brief description")


class ExamplesResponse(BaseModel):
    """Response from GET /demo/examples."""

    examples: list[ExampleInfo]


class ExampleCodeResponse(BaseModel):
    """Response from GET /demo/examples/{name}."""

    name: str
    path: str
    description: str
    code: str = Field(..., description="File contents")
    line_count: int = Field(..., description="Number of lines")


class GenerateRequest(BaseModel):
    """Request for POST /demo/generate."""

    input_type: Literal["image", "code"] = Field(
        ..., description="Type of input: 'image' or 'code'"
    )
    image_base64: str | None = Field(
        None,
        description="Base64-encoded image (with or without data URL prefix). Required if input_type='image'",
    )
    code: str | None = Field(
        None,
        description="Raw code string. Required if input_type='code'",
    )
    icp_preset: str = Field(
        "coperniq_mep",
        description="ICP preset to use",
    )
    stage: Literal["preview", "demo", "shipped"] = Field(
        "preview",
        description="Storyboard stage for BDR cadence",
    )
    audience: Literal["business_owner", "c_suite", "btl_champion", "top_tier_vc"] = Field(
        "c_suite",
        description="Target audience persona",
    )


class GenerateResponse(BaseModel):
    """Response from POST /demo/generate."""

    success: bool
    storyboard_png: str | None = Field(
        None, description="Base64-encoded PNG storyboard"
    )
    understanding: dict[str, Any] | None = Field(
        None, description="Extracted business insights"
    )
    input_type: str
    stage: str
    audience: str
    icp_preset: str
    execution_time_ms: int
    error: str | None = None


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str


# ============================================================================
# Helper Functions
# ============================================================================


def get_project_root() -> Path:
    """Get the project root directory."""
    # Assumes this file is in src/demo/router.py
    return Path(__file__).parent.parent.parent


def get_example_by_name(name: str) -> dict[str, str] | None:
    """Get example metadata by name."""
    for example in EXAMPLES:
        if example["name"] == name:
            return example
    return None


def read_file_content(relative_path: str) -> str:
    """Read file content from project root."""
    project_root = get_project_root()
    file_path = project_root / relative_path

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {relative_path}")

    if not file_path.is_file():
        raise ValueError(f"Not a file: {relative_path}")

    # Security check: ensure file is within project
    try:
        file_path.resolve().relative_to(project_root.resolve())
    except ValueError as e:
        raise ValueError(f"File outside project root: {relative_path}") from e

    with open(file_path) as f:
        return f.read()


# ============================================================================
# Endpoints
# ============================================================================


@router.get(
    "/examples",
    response_model=ExamplesResponse,
    summary="List available example code files",
)
async def list_examples() -> ExamplesResponse:
    """
    List all available example code files.

    Returns metadata for each example including name, path, and description.
    """
    examples = [
        ExampleInfo(
            name=ex["name"],
            path=ex["path"],
            description=ex["description"],
        )
        for ex in EXAMPLES
    ]

    return ExamplesResponse(examples=examples)


@router.get(
    "/examples/{name}",
    response_model=ExampleCodeResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get code content for an example",
)
async def get_example_code(name: str) -> ExampleCodeResponse:
    """
    Get the code content for a specific example.

    Args:
        name: Example name (e.g., 'unified_storyboard')

    Returns:
        Example metadata plus full file contents.

    Raises:
        404: Example not found or file doesn't exist.
    """
    example = get_example_by_name(name)
    if example is None:
        raise HTTPException(status_code=404, detail=f"Example '{name}' not found")

    try:
        code = read_file_content(example["path"])
        line_count = len(code.splitlines())

        return ExampleCodeResponse(
            name=example["name"],
            path=example["path"],
            description=example["description"],
            code=code,
            line_count=line_count,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post(
    "/generate",
    response_model=GenerateResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
    summary="Generate storyboard from image or code",
)
async def generate_storyboard(request: GenerateRequest) -> GenerateResponse:
    """
    Generate executive storyboard PNG from image or code.

    Accepts either:
    - Base64-encoded image (with or without data URL prefix)
    - Raw code string

    Uses UnifiedStoryboardTool with open_browser=False for server-side generation.

    Args:
        request: Generation request with input_type and corresponding data.

    Returns:
        Generated storyboard as base64 PNG with extracted business insights.

    Raises:
        400: Invalid input (missing required fields).
        422: Validation error.
    """
    # Validate input
    if request.input_type == "image":
        if request.image_base64 is None:
            raise HTTPException(
                status_code=400,
                detail="image_base64 is required when input_type='image'",
            )
        input_value = request.image_base64
        # Validate not empty (check for empty string)
        if not input_value or not input_value.strip():
            raise HTTPException(
                status_code=400,
                detail="image input cannot be empty",
            )
    elif request.input_type == "code":
        if request.code is None:
            raise HTTPException(
                status_code=400,
                detail="code is required when input_type='code'",
            )
        input_value = request.code
        # Validate not empty (check for empty string)
        if not input_value or not input_value.strip():
            raise HTTPException(
                status_code=400,
                detail="code input cannot be empty",
            )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input_type: {request.input_type}",
        )

    # Run UnifiedStoryboardTool
    tool = UnifiedStoryboardTool()
    result = await tool.run({
        "input": input_value,
        "icp_preset": request.icp_preset,
        "stage": request.stage,
        "audience": request.audience,
        "open_browser": False,  # Server-side - don't open browser
    })

    if result.success:
        return GenerateResponse(
            success=True,
            storyboard_png=result.result.get("storyboard_png"),
            understanding=result.result.get("understanding"),
            input_type=result.result.get("input_type", request.input_type),
            stage=request.stage,
            audience=request.audience,
            icp_preset=request.icp_preset,
            execution_time_ms=result.execution_time_ms,
        )
    else:
        return GenerateResponse(
            success=False,
            input_type=request.input_type,
            stage=request.stage,
            audience=request.audience,
            icp_preset=request.icp_preset,
            execution_time_ms=result.execution_time_ms,
            error=result.error,
        )
