"""
UnifiedStoryboardTool - The All-in-One Storyboard Generator
============================================================

Generates executive storyboard PNGs from ANY input source:
- Miro board URLs (prompts for screenshot if auth needed)
- Image URLs (.png, .jpg, .jpeg, .webp)
- Base64 image data (data:image/... or raw base64)
- File paths (code files or image files)
- Raw code strings

Auto-detects input type and routes to appropriate handler.
Opens result in browser by default.

NO OpenAI - Gemini only.
"""

import base64
import logging
import os
import tempfile
import webbrowser
from datetime import datetime
from time import perf_counter
from typing import Literal

import httpx

from src.tools.base import BaseTool, ToolCategory, ToolDefinition, ToolResult
from src.tools.storyboard.coperniq_presets import (
    get_audience_persona,
    get_icp_preset,
    sanitize_content,
)
from src.tools.storyboard.gemini_client import (
    GeminiStoryboardClient,
)

logger = logging.getLogger(__name__)

# Image file extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


InputType = Literal["miro_url", "image_url", "image_data", "file_path", "code"]


class UnifiedStoryboardTool(BaseTool):
    """
    Unified storyboard generator that accepts ANY input.

    Automatically detects input type and generates executive-ready PNG storyboards.
    Opens result in default browser for immediate viewing.

    Supported inputs:
    - Miro board URLs: https://miro.com/app/board/...
    - Image URLs: https://example.com/image.png
    - Base64 images: data:image/png;base64,... or raw base64
    - File paths: /path/to/file.py or /path/to/screenshot.png
    - Raw code: def calculate_roi(): return revenue - costs

    Example:
        tool = UnifiedStoryboardTool()

        # From code
        result = await tool.run({
            "input": "def calculate_roi(): return revenue - costs",
            "audience": "c_suite",
        })

        # From screenshot
        result = await tool.run({
            "input": "data:image/png;base64,iVBORw0KGgo...",
            "stage": "demo",
        })

        # Result opens in browser automatically
    """

    DEFAULT_TIMEOUT = 90  # seconds

    def __init__(self, gemini_client: GeminiStoryboardClient | None = None):
        """
        Initialize UnifiedStoryboardTool.

        Args:
            gemini_client: Optional pre-configured Gemini client
        """
        self._gemini_client = gemini_client

    @property
    def gemini_client(self) -> GeminiStoryboardClient:
        """Lazy initialization of Gemini client."""
        if self._gemini_client is None:
            self._gemini_client = GeminiStoryboardClient()
        return self._gemini_client

    @property
    def definition(self) -> ToolDefinition:
        """Tool definition for LLM function calling."""
        return ToolDefinition(
            name="unified_storyboard",
            description=(
                "Generate executive storyboard from ANY input source and open in browser. "
                "Accepts Miro URLs, image URLs, base64 images, file paths, or raw code. "
                "Auto-detects input type. Creates beautiful one-page PNG storyboards "
                "showing business value, benefits, and differentiators. "
                "Perfect for sales demos, cold outreach, and stakeholder presentations."
            ),
            category=ToolCategory.DATA,
            parameters={
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": (
                            "Any input: Miro URL, image URL, base64 image, "
                            "file path, or raw code string"
                        ),
                    },
                    "icp_preset": {
                        "type": "string",
                        "description": "ICP preset to use (default: coperniq_mep)",
                        "default": "coperniq_mep",
                    },
                    "stage": {
                        "type": "string",
                        "enum": ["preview", "demo", "shipped"],
                        "description": "Storyboard stage for BDR cadence",
                        "default": "preview",
                    },
                    "audience": {
                        "type": "string",
                        "enum": ["business_owner", "c_suite", "btl_champion"],
                        "description": "Target audience persona",
                        "default": "c_suite",
                    },
                    "open_browser": {
                        "type": "boolean",
                        "description": "Auto-open result in browser (default: true)",
                        "default": True,
                    },
                },
                "required": ["input"],
            },
            requires_approval=False,
        )

    def detect_input_type(self, input_value: str) -> InputType:
        """
        Detect the type of input provided.

        Args:
            input_value: The raw input string

        Returns:
            One of: "miro_url", "image_url", "image_data", "file_path", "code"
        """
        input_value = input_value.strip()

        # Miro board URL
        if input_value.startswith("https://miro.com"):
            return "miro_url"

        # Base64 image data (data URL or raw base64)
        if input_value.startswith("data:image"):
            return "image_data"

        # Image URL (check before generic URL check)
        if input_value.startswith("http"):
            lower = input_value.lower()
            if any(lower.endswith(ext) for ext in IMAGE_EXTENSIONS):
                return "image_url"

        # File path (only if file actually exists)
        if os.path.isfile(input_value):
            return "file_path"

        # Default: treat as code
        return "code"

    def is_image_file(self, file_path: str) -> bool:
        """Check if a file path points to an image file."""
        ext = os.path.splitext(file_path)[1].lower()
        return ext in IMAGE_EXTENSIONS

    def save_and_open_browser(
        self,
        png_bytes: bytes,
        filename: str | None = None,
        open_browser: bool = True,
    ) -> str:
        """
        Save PNG to temp file and optionally open in browser.

        Args:
            png_bytes: Raw PNG image bytes
            filename: Optional filename (auto-generated if None)
            open_browser: Whether to open in browser

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"storyboard_{timestamp}.png"

        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, filename)

        with open(file_path, "wb") as f:
            f.write(png_bytes)

        logger.info(f"Saved storyboard to: {file_path}")

        if open_browser:
            url = f"file://{file_path}"
            webbrowser.open(url)
            logger.info(f"Opened in browser: {url}")

        return file_path

    async def fetch_image_url(self, url: str) -> bytes:
        """
        Fetch image from URL.

        Args:
            url: Image URL

        Returns:
            Image bytes

        Raises:
            ValueError: If fetch fails
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.content

    async def run(self, arguments: dict) -> ToolResult:
        """
        Execute the unified storyboard pipeline.

        Args:
            arguments: Tool arguments containing:
                - input: Required. Any input type.
                - icp_preset: Optional. ICP preset (default: coperniq_mep)
                - stage: Optional. Storyboard stage (default: preview)
                - audience: Optional. Target audience (default: c_suite)
                - open_browser: Optional. Open in browser (default: true)

        Returns:
            ToolResult with:
            - storyboard_png: Base64-encoded PNG image
            - understanding: Extracted business insights
            - file_path: Path to saved PNG file
            - input_type: Detected input type
        """
        start_time = perf_counter()

        # Extract arguments
        input_value = arguments.get("input")
        if not input_value:
            return ToolResult(
                tool_name=self.definition.name,
                success=False,
                result={},
                error="Missing required 'input' parameter",
                execution_time_ms=int((perf_counter() - start_time) * 1000),
            )

        icp_preset = arguments.get("icp_preset", "coperniq_mep")
        stage = arguments.get("stage", "preview")
        audience = arguments.get("audience", "c_suite")
        open_browser = arguments.get("open_browser", True)

        try:
            # Detect input type
            input_type = self.detect_input_type(input_value)
            logger.info(f"Detected input type: {input_type}")

            # Get content based on input type
            is_image = False
            is_image_file_flag = False
            content: bytes | str

            if input_type == "miro_url":
                # Miro requires authentication - prompt user for screenshot
                return ToolResult(
                    tool_name=self.definition.name,
                    success=False,
                    result={"input_type": input_type},
                    error=(
                        "Miro boards require authentication. Please: "
                        "1. Open the Miro board in your browser "
                        "2. Take a screenshot (Cmd+Shift+4 on Mac) "
                        "3. Copy the image to clipboard "
                        "4. Paste here as base64 data URL"
                    ),
                    execution_time_ms=int((perf_counter() - start_time) * 1000),
                )

            elif input_type == "image_url":
                # Fetch image from URL
                logger.info(f"Fetching image from URL: {input_value}")
                content = await self.fetch_image_url(input_value)
                is_image = True

            elif input_type == "image_data":
                # Decode base64 image
                if "," in input_value:
                    # Data URL format: data:image/png;base64,XXXX
                    content = base64.b64decode(input_value.split(",")[1])
                else:
                    # Raw base64
                    content = base64.b64decode(input_value)
                is_image = True

            elif input_type == "file_path":
                # Read file
                if self.is_image_file(input_value):
                    with open(input_value, "rb") as f:
                        content = f.read()
                    is_image = True
                    is_image_file_flag = True
                else:
                    with open(input_value) as f:
                        content = f.read()

            else:  # code
                content = input_value

            # Get ICP preset and audience persona
            icp = get_icp_preset(icp_preset)
            persona = get_audience_persona(audience)

            # Stage 1: Understand the content
            logger.info("Stage 1: Understanding content...")
            if is_image:
                assert isinstance(content, bytes)
                understanding = await self.gemini_client.understand_image(
                    image_data=content,
                    icp_preset=icp,
                    audience=persona,
                )
            else:
                assert isinstance(content, str)
                # Sanitize code content
                sanitized = sanitize_content(content, icp)
                understanding = await self.gemini_client.understand_code(
                    code_content=sanitized,
                    icp_preset=icp,
                    audience=persona,
                )

            # Stage 2: Generate storyboard
            logger.info("Stage 2: Generating storyboard...")
            png_bytes = await self.gemini_client.generate_storyboard(
                understanding=understanding,
                icp=icp,
                persona=persona,
                stage=stage,
            )

            # Save and optionally open in browser
            file_path = self.save_and_open_browser(
                png_bytes=png_bytes,
                open_browser=open_browser,
            )

            # Encode result as base64
            storyboard_b64 = base64.b64encode(png_bytes).decode("utf-8")

            execution_time_ms = int((perf_counter() - start_time) * 1000)

            result = {
                "storyboard_png": storyboard_b64,
                "understanding": {
                    "headline": understanding.headline,
                    "what_it_does": understanding.what_it_does,
                    "business_value": understanding.business_value,
                    "who_benefits": understanding.who_benefits,
                    "differentiator": understanding.differentiator,
                    "pain_point_addressed": understanding.pain_point_addressed,
                    "suggested_icon": understanding.suggested_icon,
                },
                "file_path": file_path,
                "input_type": input_type,
                "stage": stage,
                "audience": audience,
                "icp_preset": icp_preset,
            }

            if is_image_file_flag:
                result["is_image_file"] = True

            logger.info(f"Storyboard generated in {execution_time_ms}ms")

            return ToolResult(
                tool_name=self.definition.name,
                success=True,
                result=result,
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            logger.exception(f"Failed to generate storyboard: {e}")
            return ToolResult(
                tool_name=self.definition.name,
                success=False,
                result={
                    "input_type": input_type if "input_type" in dir() else "unknown"
                },
                error=str(e),
                execution_time_ms=int((perf_counter() - start_time) * 1000),
            )
