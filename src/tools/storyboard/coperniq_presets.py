"""
Coperniq ICP Presets and Sanitization Rules
============================================

Defines Ideal Customer Profile (ICP) configurations and content
sanitization rules for executive storyboard generation.

Target: Multi-trade contractors (MEP+energy) with $5M+ revenue
Golden Rule: If a competitor could use the info to copy us, strip it.
             If a 5th grader couldn't understand it, simplify it.
"""

from typing import Any
from enum import Enum


# =============================================
# Coperniq Brand Identity (Scraped 2025-12-04)
# =============================================

COPERNIQ_BRAND = {
    "company": "Coperniq",
    "tagline": "One platform to run every trade: build, dispatch, service",
    "website": "https://coperniq.io",

    # Visual Identity (extracted from coperniq.io 2025-12-04)
    "colors": {
        "primary": "#23433E",        # Dark teal/forest green (CTA buttons, primary actions)
        "accent": "#2D9688",         # Teal (accent text like "$5M+ Contractors")
        "text": "#333333",           # Dark gray (body text, logo)
        "background": "#FDFDFC",     # Off-white (body background)
        "hero_bg": "#DDEDEB",        # Light mint/sage (hero section background)
        "light_gray": "#F7F6F3",     # Warm gray (alternate sections)
    },
    "typography": {
        "primary": "Albert Sans",
        "weights": [400, 500, 600],
        "style": "Clean, geometric sans-serif",
    },
    "visual_aesthetic": "Modern, professional, enterprise. Minimal grid patterns, photography-heavy with field workers. Corporate but approachable.",

    # Key Headlines from Landing Pages
    "headlines": [
        "Built for Complex Operations. Designed for $5M+ Contractors",
        "Quote, Book, Pay, Dispatch, and Grow",
        "One platform for your entire operation",
        "AI for the Trades—Not just call answering",
        "Construction CRM that runs the work — not just the contacts",
        "Plan the job, schedule the crews, collect the cash",
    ],

    # Proven Results (for storyboards)
    "proof_points": {
        "completion_rate": "99% first-time completion rate",
        "time_to_completion": "45 days faster to project completion",
        "payment_speed": "65% faster payment collection",
        "cost_savings": "$3,000 soft-cost savings per install",
        "dso_improvement": "24/7 faster DSO (cash collection)",
        "scale_story": "Scaled from 20 to 100+ installs/month without adding staff",
    },

    # AI Capabilities (for VC/innovation focus)
    "ai_features": [
        "AI Receptionist - Answers calls, books work orders 24/7",
        "Smart Forms - Converts paper inspections to digital via AI",
        "Nameplate Scanning - Capture make, model, serial in one shot",
        "Project Copilot - Ask questions within job records",
        "Ask AI Views - Describe data slice you want to see",
        "Quote Generation - Good/better/best options auto-generated",
    ],

    # Product Categories
    "feature_categories": {
        "sales": ["CRM", "Quotes/Proposals", "E-signatures", "Lead conversion"],
        "operations": ["Field service", "Dispatch", "Scheduling", "Mobile app (offline)"],
        "admin": ["Document management", "Payment processing", "Accounting sync"],
        "integrations": ["QuickBooks", "Xero", "NetSuite", "Design tools", "Hardware APIs"],
    },
}


class AudiencePersona(str, Enum):
    """Target audience personas for storyboard content."""

    BUSINESS_OWNER = "business_owner"
    C_SUITE = "c_suite"
    BTL_CHAMPION = "btl_champion"
    TOP_TIER_VC = "top_tier_vc"
    FIELD_CREW = "field_crew"


class StoryboardStage(str, Enum):
    """Storyboard stage for 3-wave BDR cadence."""

    PREVIEW = "preview"  # Wave 1: "Here's what we're building"
    DEMO = "demo"  # Wave 2: "Here it is working"
    SHIPPED = "shipped"  # Wave 3: "It's live + what's next"


# =============================================
# Coperniq Ideal Customer Profile (ICP)
# =============================================

COPERNIQ_ICP = {
    "name": "coperniq_mep",
    "target": "Multi-trade contractors (MEP+energy)",
    "characteristics": {
        "revenue": "$5M+",
        "style": "Asset-centric, self-perform",
        "trades": ["mechanical", "electrical", "plumbing", "energy", "solar", "hvac"],
        "pain_points": [
            "Spreadsheet chaos across multiple jobs",
            "Missed deadlines and change orders",
            "Crew coordination nightmares",
            "Getting paid takes forever",
            "No visibility into job profitability",
        ],
    },
    "audience_personas": {
        AudiencePersona.BUSINESS_OWNER: {
            "title": "Business Owner / Founder",
            "cares_about": ["profit", "growth", "less headaches", "family time"],
            "tone": "Direct, bottom-line focused, respect their time",
            "hooks": [
                "Stop losing money on jobs you thought were profitable",
                "Your competition is already using this",
                "What if you could leave the office at 5pm?",
            ],
        },
        AudiencePersona.C_SUITE: {
            "title": "CEO / CFO / COO",
            "cares_about": ["ROI", "competitive edge", "scalability", "data-driven decisions"],
            "tone": "Strategic, numbers-focused, executive-level",
            "hooks": [
                "See your entire operation at a glance",
                "Make decisions based on real data, not gut feelings",
                "Scale without adding overhead",
            ],
        },
        AudiencePersona.BTL_CHAMPION: {
            "title": "Project Manager / Operations Manager",
            "cares_about": ["easier day-to-day", "looking good to boss", "less fire-fighting"],
            "tone": "Empathetic, practical, day-in-the-life focused",
            "hooks": [
                "Your crews will actually use this",
                "No more chasing down updates",
                "Be the hero who fixed the coordination problem",
            ],
        },
        AudiencePersona.TOP_TIER_VC: {
            "title": "Top Tier VC / Angel / PE Investor",
            "cares_about": ["TAM/SAM/SOM", "traction metrics", "defensible moat", "unit economics", "team"],
            "tone": "Data-driven, confident, investment-thesis focused. NO customer CTAs.",
            "hooks": [
                "$200B+ market, fragmented incumbents, perfect timing",
                "Category-defining platform for contractor operations",
                "Vertical AI that compounds with data network effects",
            ],
            # CRITICAL: VC storyboard structure (NOT customer demo)
            "storyboard_structure": {
                "section_1_problem": {
                    "header": "THE PROBLEM",
                    "format": "$X billion lost annually to [specific pain]. X% of contractors still use [outdated method].",
                    "example": "$47B lost annually to operational inefficiency. 73% of contractors still run on spreadsheets.",
                },
                "section_2_solution": {
                    "header": "THE SOLUTION",
                    "format": "One-sentence UVP. What we do differently.",
                    "example": "One platform that runs the entire contracting operation: quote → dispatch → pay.",
                },
                "section_3_traction": {
                    "header": "TRACTION",
                    "format": "ARR, growth rate, customer count, key metric",
                    "example": "$X ARR | X% MoM growth | X customers | 99% retention",
                    "note": "Use Coperniq proof points if real metrics unavailable",
                },
                "section_4_market": {
                    "header": "MARKET",
                    "format": "TAM → SAM → SOM with clear logic",
                    "example": "$200B TAM (all contractor software) → $40B SAM (MEP+Energy) → $2B SOM (mid-market)",
                },
                "section_5_why_now": {
                    "header": "WHY NOW",
                    "format": "Market shift enabling this opportunity",
                    "example": "AI inflection + workforce shortage + regulatory pressure = perfect storm",
                },
                "section_6_moat": {
                    "header": "DEFENSIBILITY",
                    "format": "What compounds over time",
                    "example": "Data network effects: every job makes the AI smarter for every customer",
                },
            },
            "avoid_in_vc_storyboard": [
                "Book a demo",
                "Get started",
                "Contact sales",
                "Free trial",
                "See pricing",
                "Customer testimonials",  # Use metrics instead
            ],
            "metrics_that_matter": [
                "ARR / MRR",
                "Growth rate (MoM/YoY)",
                "Net Revenue Retention (NRR)",
                "CAC payback period",
                "Gross margin",
                "Logo retention",
            ],
        },
        AudiencePersona.FIELD_CREW: {
            "title": "Field Crew / Technicians / Blue Collar Workers",
            "cares_about": ["making my job easier", "not looking stupid", "getting home on time", "less paperwork"],
            "tone": "Super simple, friendly, visual-first - explain like I'm 10",
            "hooks": [
                "This makes your job way easier",
                "No more paperwork headaches",
                "Works even when you don't have signal",
                "Your boss will think you're a genius",
            ],
            "infographic_style": {
                "design": "Simple icons, big text, minimal words",
                "colors": "Bold primary colors, high contrast",
                "format": "Step-by-step visual flow, numbered steps",
                "language_rules": [
                    "Use 5th grade vocabulary ONLY",
                    "Replace technical words with everyday analogies",
                    "Use pictures/icons instead of text when possible",
                    "Maximum 6 words per bullet point",
                    "Compare to things they already know (phone, truck, tools)",
                ],
                "analogies": {
                    "API": "like a waiter taking your order",
                    "database": "like a filing cabinet",
                    "sync": "like copying to your other phone",
                    "cloud": "like saving to the internet",
                    "automation": "like setting a coffee maker timer",
                    "workflow": "like following a recipe",
                    "integration": "like plugging in an extension cord",
                    "real-time": "instant, like a text message",
                },
            },
        },
    },
    "language_style": {
        "avoid": [
            # Technical jargon (NO IP EXPOSURE)
            "AI",
            "machine learning",
            "algorithm",
            "neural network",
            "deep learning",
            "model",
            "API",
            "microservices",
            "async",
            "database schema",
            "backend",
            "frontend",
            "deployment",
            "infrastructure",
            # Proprietary terms (IP PROTECTION)
            "proprietary",
            "patent-pending",
            "trade secret",
            "competitive advantage",
            "secret sauce",
            # Boring corporate speak
            "leverage",
            "synergy",
            "paradigm",
            "ecosystem",
            "holistic",
            "robust",
            "scalable solution",
            "best-in-class",
            "cutting-edge",
            "revolutionary",
            "disruptive",
            "game-changing",
        ],
        "use": [
            # Simple, benefit-focused language
            "saves you time",
            "gets you paid faster",
            "one place for everything",
            "no more spreadsheets",
            "your crews will love this",
            "works like magic",
            "see everything at a glance",
            "never miss a beat",
            "stop the chaos",
            "know where every dollar goes",
            "finally get home for dinner",
            "your guys can actually use it",
            "works even in the field",
            "no training required",
        ],
    },
    "tone": "Friendly expert friend, not salesy vendor. Like a contractor buddy who found something awesome.",
    "proof_points": {
        "metrics": [
            "hours saved per week",
            "% fewer errors",
            "days faster to payment",
            "% increase in job profitability",
        ],
        "social": [
            "contractors like you",
            "trusted by MEP firms",
            "built for self-performers",
            "designed by people who get it",
        ],
    },
    "visual_style": {
        "colors": ["#23433E", "#2D9688", "#DDEDEB", "#333333", "#FDFDFC"],  # Coperniq: dark teal, teal accent, mint bg, text, off-white
        "primary_color": "#23433E",  # Dark teal/forest green for CTAs and headers
        "accent_color": "#2D9688",   # Teal for accent text and highlights
        "hero_bg": "#DDEDEB",        # Light mint/sage for hero backgrounds
        "text_color": "#333333",     # Dark gray for body text
        "icons": "Simple, construction-related metaphors (tools, buildings, workers)",
        "layout": "Clean, scannable, executive-friendly. Subtle mint backgrounds.",
        "font_style": "Albert Sans or similar. Large, readable, no fine print feel.",
        "aesthetic": "Modern, professional, teal/green palette. Corporate but approachable.",
    },
}


# =============================================
# Content Sanitization Rules
# =============================================

SANITIZE_RULES = {
    "remove": [
        # Code internals (IP PROTECTION)
        "class names",
        "function names",
        "variable names",
        "method signatures",
        "import statements",
        "package names",
        # System architecture (IP PROTECTION)
        "API endpoints",
        "database tables",
        "database columns",
        "internal URLs",
        "service names",
        "queue names",
        "cache keys",
        # Business secrets (IP PROTECTION)
        "employee names",
        "customer names",
        "pricing details",
        "margin information",
        "vendor names",
        "partnership details",
        # Security (CRITICAL)
        "API keys",
        "tokens",
        "passwords",
        "secrets",
        "credentials",
        "authentication details",
    ],
    "keep": [
        # Business value (SAFE TO SHARE)
        "business outcome",
        "user benefit",
        "time saved",
        "problem solved",
        "workflow improvement",
        "pain point addressed",
        # General concepts (SAFE TO SHARE)
        "general workflow description",
        "high-level process",
        "user experience improvement",
        "efficiency gain",
    ],
    "transform": {
        # Technical → Simple mappings
        "technical_process": "simple analogy or metaphor",
        "code_logic": "plain english benefit",
        "system_architecture": "visual workflow icon",
        "database operation": "data management",
        "API call": "automatic sync",
        "async processing": "works in the background",
        "machine learning": "smart automation",
        "algorithm": "smart system",
        "real-time sync": "instant updates",
        "webhook": "automatic notification",
        "microservice": "specialized helper",
        "containerization": "runs anywhere",
    },
}


# =============================================
# Stage-Specific Templates
# =============================================

STAGE_TEMPLATES = {
    StoryboardStage.PREVIEW: {
        "header_prefix": "Coming Soon",
        "tone_modifier": "exciting, forward-looking, exclusive preview",
        "cta": "Want early access? Let's talk.",
        "visual_style": "Blueprint/wireframe aesthetic, future-focused",
        "badge": "SNEAK PEEK",
    },
    StoryboardStage.DEMO: {
        "header_prefix": "Now Working",
        "tone_modifier": "confident, proven, see-it-in-action",
        "cta": "Ready to see it live? Book a demo.",
        "visual_style": "Screenshot-based, real interface glimpses",
        "badge": "LIVE DEMO",
    },
    StoryboardStage.SHIPPED: {
        "header_prefix": "Now Available",
        "tone_modifier": "ready-to-use, immediate value, start today",
        "cta": "Start your free trial today.",
        "visual_style": "Polished, professional, ready-to-use",
        "badge": "AVAILABLE NOW",
    },
}


# =============================================
# Helper Functions
# =============================================


def get_icp_preset(preset_name: str = "coperniq_mep") -> dict[str, Any]:
    """
    Get ICP preset configuration by name.

    Args:
        preset_name: Name of the ICP preset (default: coperniq_mep)

    Returns:
        ICP configuration dictionary

    Raises:
        ValueError: If preset_name is not found
    """
    presets = {
        "coperniq_mep": COPERNIQ_ICP,
        # Future: Add more ICP presets here
        # "solar_residential": SOLAR_ICP,
        # "general_contractor": GC_ICP,
    }

    if preset_name not in presets:
        available = ", ".join(presets.keys())
        raise ValueError(f"Unknown ICP preset: {preset_name}. Available: {available}")

    return presets[preset_name]


def get_audience_persona(
    persona: AudiencePersona | str, icp_preset: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Get audience persona configuration.

    Args:
        persona: AudiencePersona enum or string value
        icp_preset: Optional ICP preset (defaults to COPERNIQ_ICP)

    Returns:
        Audience persona configuration dictionary
    """
    if icp_preset is None:
        icp_preset = COPERNIQ_ICP

    # Convert string to enum if needed
    if isinstance(persona, str):
        persona = AudiencePersona(persona)

    return icp_preset["audience_personas"].get(persona, icp_preset["audience_personas"][AudiencePersona.C_SUITE])


def get_stage_template(stage: StoryboardStage | str) -> dict[str, Any]:
    """
    Get stage-specific template configuration.

    Args:
        stage: StoryboardStage enum or string value

    Returns:
        Stage template configuration dictionary
    """
    # Convert string to enum if needed
    if isinstance(stage, str):
        stage = StoryboardStage(stage)

    return STAGE_TEMPLATES.get(stage, STAGE_TEMPLATES[StoryboardStage.PREVIEW])


def sanitize_content(content: str, rules: dict[str, Any] | None = None) -> str:
    """
    Sanitize content according to IP protection rules.

    This is a lightweight sanitizer - the heavy lifting is done by Gemini
    during the understanding phase. This catches obvious patterns.

    Args:
        content: Raw content to sanitize
        rules: Optional custom rules (defaults to SANITIZE_RULES)

    Returns:
        Sanitized content string
    """
    if rules is None:
        rules = SANITIZE_RULES

    sanitized = content

    # Remove obvious code patterns
    import re

    # Remove import statements
    sanitized = re.sub(r"^import\s+.*$", "", sanitized, flags=re.MULTILINE)
    sanitized = re.sub(r"^from\s+.*import.*$", "", sanitized, flags=re.MULTILINE)

    # Remove class/function definitions (keep generic description)
    sanitized = re.sub(r"^class\s+\w+.*:$", "[Feature Component]", sanitized, flags=re.MULTILINE)
    sanitized = re.sub(r"^def\s+\w+\(.*\):$", "[Process Step]", sanitized, flags=re.MULTILINE)
    sanitized = re.sub(r"^async\s+def\s+\w+\(.*\):$", "[Automated Process]", sanitized, flags=re.MULTILINE)

    # Remove API keys and secrets
    sanitized = re.sub(r'["\']?[A-Za-z_]*(?:KEY|SECRET|TOKEN|PASSWORD)["\']?\s*[=:]\s*["\'][^"\']+["\']', "[REDACTED]", sanitized, flags=re.IGNORECASE)

    # Remove URLs with internal paths
    sanitized = re.sub(r"https?://[^\s]+(?:internal|staging|dev|api\.)[^\s]*", "[Internal URL]", sanitized)

    # Remove email addresses
    sanitized = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[email]", sanitized)

    return sanitized


def build_language_guidelines(icp_preset: dict[str, Any] | None = None) -> str:
    """
    Build language guidelines string for Gemini prompts.

    Args:
        icp_preset: Optional ICP preset (defaults to COPERNIQ_ICP)

    Returns:
        Formatted language guidelines string
    """
    if icp_preset is None:
        icp_preset = COPERNIQ_ICP

    avoid = icp_preset["language_style"]["avoid"]
    use = icp_preset["language_style"]["use"]
    tone = icp_preset["tone"]

    return f"""LANGUAGE GUIDELINES:
- Tone: {tone}
- AVOID these words/phrases: {', '.join(avoid[:10])}...
- USE these words/phrases: {', '.join(use[:10])}...
- Write for a 5th grader - if they can't understand it, simplify it
- NO technical jargon - translate everything to business benefits
- NO proprietary details - if a competitor could copy it, remove it"""
