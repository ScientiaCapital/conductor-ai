"""
Video Tools Module for Conductor-AI
====================================

AI-powered video prospecting orchestration suite:
- Script generation with Chinese LLM cost optimization
- Loom analytics monitoring + viewer enrichment
- Optimal send time prediction
- Industry-specific demo template management

Cost efficiency: Uses DeepSeek V3 ($0.20/$0.80 per 1M tokens) and Qwen3-8B
for 10-50x cost savings vs Claude/GPT-4 while maintaining quality.
"""

from src.tools.video.video_script_generator import VideoScriptGeneratorTool
from src.tools.video.video_analytics import LoomViewTrackerTool, ViewerEnrichmentTool
from src.tools.video.video_scheduler import VideoSchedulerTool
from src.tools.video.video_template_manager import VideoTemplateManagerTool

__all__ = [
    # Core script generation
    "VideoScriptGeneratorTool",
    # Loom analytics + enrichment
    "LoomViewTrackerTool",
    "ViewerEnrichmentTool",
    # Send time optimization
    "VideoSchedulerTool",
    # Demo template management
    "VideoTemplateManagerTool",
]
