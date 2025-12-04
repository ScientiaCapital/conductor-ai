"""Tests for demo router endpoints."""

import base64
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.tools.base import ToolResult


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


# ============================================================================
# GET /demo/examples Tests
# ============================================================================


def test_list_examples_success(client):
    """Test GET /demo/examples returns list of examples."""
    response = client.get("/demo/examples")

    assert response.status_code == 200
    data = response.json()

    assert "examples" in data
    assert isinstance(data["examples"], list)
    assert len(data["examples"]) > 0

    # Check first example has required fields
    example = data["examples"][0]
    assert "name" in example
    assert "path" in example
    assert "description" in example


def test_list_examples_structure(client):
    """Test example objects have correct structure."""
    response = client.get("/demo/examples")
    data = response.json()

    for example in data["examples"]:
        assert isinstance(example["name"], str)
        assert isinstance(example["path"], str)
        assert isinstance(example["description"], str)
        assert example["name"]  # Not empty
        assert example["path"]  # Not empty
        assert example["description"]  # Not empty


def test_list_examples_contains_expected(client):
    """Test list contains expected examples."""
    response = client.get("/demo/examples")
    data = response.json()

    example_names = {ex["name"] for ex in data["examples"]}

    # Check for key examples
    expected = {
        "unified_storyboard",
        "video_script_generator",
        "gemini_client",
    }

    assert expected.issubset(example_names), f"Missing examples: {expected - example_names}"


# ============================================================================
# GET /demo/examples/{name} Tests
# ============================================================================


def test_get_example_code_success(client):
    """Test GET /demo/examples/{name} returns code."""
    response = client.get("/demo/examples/unified_storyboard")

    assert response.status_code == 200
    data = response.json()

    # Check response structure
    assert data["name"] == "unified_storyboard"
    assert "path" in data
    assert "description" in data
    assert "code" in data
    assert "line_count" in data

    # Check code content
    assert len(data["code"]) > 0
    assert data["line_count"] > 0
    assert isinstance(data["line_count"], int)


def test_get_example_code_content_valid(client):
    """Test returned code is valid Python."""
    response = client.get("/demo/examples/unified_storyboard")
    data = response.json()

    # Should be valid Python file (contains imports, classes, etc.)
    code = data["code"]
    assert "import" in code or "from" in code
    assert "class" in code or "def" in code


def test_get_example_code_line_count_accurate(client):
    """Test line_count matches actual code lines."""
    response = client.get("/demo/examples/unified_storyboard")
    data = response.json()

    actual_lines = len(data["code"].splitlines())
    assert data["line_count"] == actual_lines


def test_get_example_code_not_found(client):
    """Test GET /demo/examples/nonexistent returns 404."""
    response = client.get("/demo/examples/nonexistent_example")

    assert response.status_code == 404
    assert "detail" in response.json()


def test_get_example_code_all_examples_readable(client):
    """Test all listed examples are readable."""
    # Get list of examples
    list_response = client.get("/demo/examples")
    examples = list_response.json()["examples"]

    # Try to read each one
    for example in examples:
        response = client.get(f"/demo/examples/{example['name']}")
        assert response.status_code == 200, f"Failed to read {example['name']}"


# ============================================================================
# POST /demo/generate Tests
# ============================================================================


def test_generate_rejects_empty_code(client):
    """Test POST /demo/generate rejects empty code input."""
    response = client.post(
        "/demo/generate",
        json={
            "input_type": "code",
            "code": "",
            "stage": "preview",
            "audience": "c_suite",
        },
    )

    assert response.status_code == 400
    assert "detail" in response.json()
    assert "empty" in response.json()["detail"].lower()


def test_generate_rejects_empty_image(client):
    """Test POST /demo/generate rejects empty image input."""
    response = client.post(
        "/demo/generate",
        json={
            "input_type": "image",
            "image_base64": "",
            "stage": "preview",
            "audience": "c_suite",
        },
    )

    assert response.status_code == 400
    assert "detail" in response.json()


def test_generate_rejects_whitespace_only(client):
    """Test POST /demo/generate rejects whitespace-only input."""
    response = client.post(
        "/demo/generate",
        json={
            "input_type": "code",
            "code": "   \n\t  ",
            "stage": "preview",
            "audience": "c_suite",
        },
    )

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_generate_requires_image_when_type_image(client):
    """Test POST /demo/generate requires image_base64 when input_type='image'."""
    response = client.post(
        "/demo/generate",
        json={
            "input_type": "image",
            "code": "def foo(): pass",  # Wrong field
            "stage": "preview",
            "audience": "c_suite",
        },
    )

    assert response.status_code == 400
    assert "image" in response.json()["detail"].lower()


def test_generate_requires_code_when_type_code(client):
    """Test POST /demo/generate requires code when input_type='code'."""
    response = client.post(
        "/demo/generate",
        json={
            "input_type": "code",
            "image_base64": "fake_base64",  # Wrong field
            "stage": "preview",
            "audience": "c_suite",
        },
    )

    assert response.status_code == 400
    assert "code" in response.json()["detail"]


def test_generate_validates_input_type(client):
    """Test POST /demo/generate validates input_type enum."""
    response = client.post(
        "/demo/generate",
        json={
            "input_type": "invalid_type",
            "code": "def foo(): pass",
            "stage": "preview",
            "audience": "c_suite",
        },
    )

    # Should fail validation (422)
    assert response.status_code == 422


def test_generate_with_code_mocked(client):
    """Test POST /demo/generate with code input (mocked tool)."""
    mock_result = ToolResult(
        tool_name="unified_storyboard",
        success=True,
        result={
            "storyboard_png": "fake_base64_png",
            "understanding": {
                "headline": "Test Feature",
                "what_it_does": "Does testing",
                "business_value": "Saves time",
                "who_benefits": "Developers",
                "differentiator": "Fast",
                "pain_point_addressed": "Slow tests",
                "suggested_icon": "test",
            },
            "input_type": "code",
        },
        execution_time_ms=1000,
    )

    with patch("src.demo.router.UnifiedStoryboardTool") as MockTool:
        mock_instance = AsyncMock()
        mock_instance.run.return_value = mock_result
        MockTool.return_value = mock_instance

        response = client.post(
            "/demo/generate",
            json={
                "input_type": "code",
                "code": "def calculate_roi(): return 100",
                "stage": "preview",
                "audience": "c_suite",
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["storyboard_png"] == "fake_base64_png"
        assert "understanding" in data
        assert data["input_type"] == "code"
        assert data["stage"] == "preview"
        assert data["audience"] == "c_suite"


def test_generate_handles_tool_failure(client):
    """Test POST /demo/generate handles tool failure gracefully."""
    mock_result = ToolResult(
        tool_name="unified_storyboard",
        success=False,
        result={},
        error="API key missing",
        execution_time_ms=100,
    )

    with patch("src.demo.router.UnifiedStoryboardTool") as MockTool:
        mock_instance = AsyncMock()
        mock_instance.run.return_value = mock_result
        MockTool.return_value = mock_instance

        response = client.post(
            "/demo/generate",
            json={
                "input_type": "code",
                "code": "def foo(): pass",
                "stage": "preview",
                "audience": "c_suite",
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is False
        assert data["error"] == "API key missing"
        assert data["storyboard_png"] is None


def test_generate_passes_open_browser_false(client):
    """Test POST /demo/generate sets open_browser=False."""
    with patch("src.demo.router.UnifiedStoryboardTool") as MockTool:
        mock_instance = AsyncMock()
        mock_instance.run.return_value = ToolResult(
            tool_name="unified_storyboard",
            success=True,
            result={
                "storyboard_png": "fake",
                "understanding": {},
                "input_type": "code",
            },
            execution_time_ms=100,
        )
        MockTool.return_value = mock_instance

        client.post(
            "/demo/generate",
            json={
                "input_type": "code",
                "code": "def foo(): pass",
            },
        )

        # Verify open_browser=False was passed
        call_args = mock_instance.run.call_args
        assert call_args is not None
        assert call_args[0][0]["open_browser"] is False


# ============================================================================
# Integration Tests (require files to exist)
# ============================================================================


def test_integration_read_real_file(client):
    """Integration test: Read a real example file."""
    response = client.get("/demo/examples/gemini_client")

    # Should succeed if file exists
    if response.status_code == 200:
        data = response.json()
        assert len(data["code"]) > 100  # Real file should be substantial
        assert "Gemini" in data["code"] or "gemini" in data["code"]


# ============================================================================
# Video Generation Tests
# ============================================================================


class TestBuildVideoPromptFromUnderstanding:
    """Tests for build_video_prompt_from_understanding helper."""

    def test_builds_prompt_from_understanding(self):
        """Test building video prompt from understanding dict."""
        from src.demo.router import build_video_prompt_from_understanding

        understanding = {
            "headline": "Automated Dispatch System",
            "description": "Real-time crew tracking",
            "value_proposition": "Save 2 hours per day",
            "problem_solved": "Manual paperwork",
        }

        prompt = build_video_prompt_from_understanding(understanding, "field_crew")

        # Check key elements are present
        assert "Automated Dispatch System" in prompt
        assert "Real-time crew tracking" in prompt
        assert "Save 2 hours per day" in prompt
        assert "outdoor work site" in prompt  # field_crew style

    def test_handles_empty_understanding(self):
        """Test handling empty understanding dict."""
        from src.demo.router import build_video_prompt_from_understanding

        prompt = build_video_prompt_from_understanding({}, "c_suite")

        # Should still produce valid prompt
        assert "Create a 5-second" in prompt
        assert "Product Feature" in prompt  # Default headline
        assert "boardroom" in prompt  # c_suite style

    def test_audience_specific_styles(self):
        """Test different audiences get different visual styles."""
        from src.demo.router import build_video_prompt_from_understanding

        understanding = {"headline": "Test Feature"}

        prompt_field = build_video_prompt_from_understanding(understanding, "field_crew")
        prompt_csuite = build_video_prompt_from_understanding(understanding, "c_suite")
        prompt_vc = build_video_prompt_from_understanding(understanding, "top_tier_vc")

        # Each should have different style
        assert "outdoor work site" in prompt_field
        assert "boardroom" in prompt_csuite
        assert "tech startup" in prompt_vc

    def test_truncates_long_content(self):
        """Test long content is truncated."""
        from src.demo.router import build_video_prompt_from_understanding

        understanding = {
            "headline": "Test",
            "description": "x" * 500,  # Very long
            "value_proposition": "y" * 300,  # Very long
        }

        prompt = build_video_prompt_from_understanding(understanding, "c_suite")

        # Prompt should be reasonable length (content truncated)
        assert len(prompt) < 1000


class TestVideoOutputFormat:
    """Tests for video output format handling."""

    def test_video_horizontal_format_accepted(self, client):
        """Test video_horizontal is a valid output_format."""
        with patch("src.demo.router.UnifiedStoryboardTool") as MockTool:
            mock_instance = AsyncMock()
            mock_instance.run.return_value = ToolResult(
                tool_name="unified_storyboard",
                success=True,
                result={
                    "storyboard_png": "fake",
                    "understanding": {"headline": "Test"},
                    "input_type": "code",
                },
                execution_time_ms=100,
            )
            MockTool.return_value = mock_instance

            with patch("src.demo.router.VideoGeneratorTool") as MockVideoTool:
                mock_video_instance = AsyncMock()
                mock_video_instance.run.return_value = ToolResult(
                    tool_name="video_generator",
                    success=True,
                    result={"video_url": "https://example.com/video.mp4"},
                    execution_time_ms=5000,
                )
                MockVideoTool.return_value = mock_video_instance

                response = client.post(
                    "/demo/generate",
                    json={
                        "code": "def foo(): pass",
                        "output_format": "video_horizontal",
                        "audience": "c_suite",
                    },
                )

                assert response.status_code == 200
                data = response.json()
                assert data["output_format"] == "video_horizontal"

    def test_video_vertical_format_accepted(self, client):
        """Test video_vertical is a valid output_format."""
        with patch("src.demo.router.UnifiedStoryboardTool") as MockTool:
            mock_instance = AsyncMock()
            mock_instance.run.return_value = ToolResult(
                tool_name="unified_storyboard",
                success=True,
                result={
                    "storyboard_png": "fake",
                    "understanding": {"headline": "Test"},
                    "input_type": "code",
                },
                execution_time_ms=100,
            )
            MockTool.return_value = mock_instance

            with patch("src.demo.router.VideoGeneratorTool") as MockVideoTool:
                mock_video_instance = AsyncMock()
                mock_video_instance.run.return_value = ToolResult(
                    tool_name="video_generator",
                    success=True,
                    result={"video_url": "https://example.com/video.mp4"},
                    execution_time_ms=5000,
                )
                MockVideoTool.return_value = mock_video_instance

                response = client.post(
                    "/demo/generate",
                    json={
                        "code": "def foo(): pass",
                        "output_format": "video_vertical",
                        "audience": "field_crew",
                    },
                )

                assert response.status_code == 200
                data = response.json()
                assert data["output_format"] == "video_vertical"

    def test_video_response_has_video_mp4_field(self, client):
        """Test video output includes video_mp4 in response."""
        with patch("src.demo.router.UnifiedStoryboardTool") as MockTool:
            mock_instance = AsyncMock()
            mock_instance.run.return_value = ToolResult(
                tool_name="unified_storyboard",
                success=True,
                result={
                    "storyboard_png": "fake",
                    "understanding": {"headline": "Test"},
                    "input_type": "code",
                },
                execution_time_ms=100,
            )
            MockTool.return_value = mock_instance

            with patch("src.demo.router.VideoGeneratorTool") as MockVideoTool:
                mock_video_instance = AsyncMock()
                mock_video_instance.run.return_value = ToolResult(
                    tool_name="video_generator",
                    success=True,
                    result={"video_url": "https://cdn.example.com/video.mp4"},
                    execution_time_ms=5000,
                )
                MockVideoTool.return_value = mock_video_instance

                response = client.post(
                    "/demo/generate",
                    json={
                        "code": "def foo(): pass",
                        "output_format": "video_horizontal",
                    },
                )

                data = response.json()
                assert data["success"] is True
                assert data["video_mp4"] == "https://cdn.example.com/video.mp4"
                assert data["storyboard_png"] is None  # No PNG for video

    def test_video_generation_uses_correct_aspect_ratio(self, client):
        """Test video generation passes correct aspect ratio."""
        with patch("src.demo.router.UnifiedStoryboardTool") as MockTool:
            mock_instance = AsyncMock()
            mock_instance.run.return_value = ToolResult(
                tool_name="unified_storyboard",
                success=True,
                result={
                    "storyboard_png": "fake",
                    "understanding": {"headline": "Test"},
                    "input_type": "code",
                },
                execution_time_ms=100,
            )
            MockTool.return_value = mock_instance

            with patch("src.demo.router.VideoGeneratorTool") as MockVideoTool:
                mock_video_instance = AsyncMock()
                mock_video_instance.run.return_value = ToolResult(
                    tool_name="video_generator",
                    success=True,
                    result={"video_url": "https://example.com/video.mp4"},
                    execution_time_ms=5000,
                )
                MockVideoTool.return_value = mock_video_instance

                # Test horizontal (16:9)
                client.post(
                    "/demo/generate",
                    json={
                        "code": "def foo(): pass",
                        "output_format": "video_horizontal",
                    },
                )

                call_args = mock_video_instance.run.call_args
                assert call_args[0][0]["aspect_ratio"] == "16:9"

                # Test vertical (9:16)
                mock_video_instance.run.reset_mock()
                client.post(
                    "/demo/generate",
                    json={
                        "code": "def foo(): pass",
                        "output_format": "video_vertical",
                    },
                )

                call_args = mock_video_instance.run.call_args
                assert call_args[0][0]["aspect_ratio"] == "9:16"

    def test_video_generation_handles_failure(self, client):
        """Test video generation failure is handled gracefully."""
        with patch("src.demo.router.UnifiedStoryboardTool") as MockTool:
            mock_instance = AsyncMock()
            mock_instance.run.return_value = ToolResult(
                tool_name="unified_storyboard",
                success=True,
                result={
                    "storyboard_png": "fake",
                    "understanding": {"headline": "Test"},
                    "input_type": "code",
                },
                execution_time_ms=100,
            )
            MockTool.return_value = mock_instance

            with patch("src.demo.router.VideoGeneratorTool") as MockVideoTool:
                mock_video_instance = AsyncMock()
                mock_video_instance.run.return_value = ToolResult(
                    tool_name="video_generator",
                    success=False,
                    result={},
                    error="No API key for Kling",
                    execution_time_ms=100,
                )
                MockVideoTool.return_value = mock_video_instance

                response = client.post(
                    "/demo/generate",
                    json={
                        "code": "def foo(): pass",
                        "output_format": "video_horizontal",
                    },
                )

                data = response.json()
                assert data["success"] is False
                assert "No API key for Kling" in data["error"]

    def test_image_format_still_works(self, client):
        """Test infographic/storyboard formats still return PNG."""
        with patch("src.demo.router.UnifiedStoryboardTool") as MockTool:
            mock_instance = AsyncMock()
            mock_instance.run.return_value = ToolResult(
                tool_name="unified_storyboard",
                success=True,
                result={
                    "storyboard_png": "base64_png_data",
                    "understanding": {"headline": "Test"},
                    "input_type": "code",
                },
                execution_time_ms=100,
            )
            MockTool.return_value = mock_instance

            response = client.post(
                "/demo/generate",
                json={
                    "code": "def foo(): pass",
                    "output_format": "infographic",
                },
            )

            data = response.json()
            assert data["success"] is True
            assert data["storyboard_png"] == "base64_png_data"
            assert data["video_mp4"] is None
