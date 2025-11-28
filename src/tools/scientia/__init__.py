"""
Scientia Stack Tools Integration for Conductor-AI
================================================

This module integrates all Scientia Capital production systems into
the conductor-ai orchestration layer, enabling autonomous multi-agent
workflows across the entire GTM stack.

Tools:
- LeadQualifyTool: Connect to sales-agent for 633ms qualification
- VisionExtractTool: Connect to quantify-mvp for blueprint analysis  
- AcademicSearchTool: Connect to perplexity-agent for research
- VoiceCallTool: Connect to vozlux for bilingual voice AI
- CostOptimizerTool: Route through ai-cost-optimizer for savings
- ContractorSearchTool: Query dealer-scraper-mvp database
"""

from .lead_qualify import LeadQualifyTool
from .vision_extract import VisionExtractTool
from .academic_search import AcademicSearchTool
from .voice_call import VoiceCallTool
from .cost_optimizer import CostOptimizerTool
from .contractor_search import ContractorSearchTool

__all__ = [
    "LeadQualifyTool",
    "VisionExtractTool", 
    "AcademicSearchTool",
    "VoiceCallTool",
    "CostOptimizerTool",
    "ContractorSearchTool",
]

# Tool registry configuration for conductor-ai
SCIENTIA_TOOLS = {
    "lead_qualify": {
        "tool_class": LeadQualifyTool,
        "description": "Qualify leads using 6 LangGraph agents (633ms, $0.000006/lead)",
        "requires_approval": False,
        "category": "gtm",
    },
    "vision_extract": {
        "tool_class": VisionExtractTool,
        "description": "Extract data from blueprints/images using Qwen VL (98.8% accuracy)",
        "requires_approval": False,
        "category": "vision",
    },
    "academic_search": {
        "tool_class": AcademicSearchTool,
        "description": "Search academic databases (ArXiv, PubMed, IEEE) with Cerebras speed",
        "requires_approval": False,
        "category": "research",
    },
    "voice_call": {
        "tool_class": VoiceCallTool,
        "description": "Schedule/make bilingual voice calls (40ms Cartesia latency)",
        "requires_approval": True,  # Requires approval for outbound calls
        "category": "communication",
    },
    "cost_optimizer": {
        "tool_class": CostOptimizerTool,
        "description": "Route LLM calls through cost optimizer (40-70% savings)",
        "requires_approval": False,
        "category": "infrastructure",
    },
    "contractor_search": {
        "tool_class": ContractorSearchTool,
        "description": "Search 8,277 qualified MEP contractors with ICP scoring",
        "requires_approval": False,
        "category": "gtm",
    },
}
