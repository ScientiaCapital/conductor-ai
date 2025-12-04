# Conductor-AI - Project Instructions

## Critical Rules
- **NO OPENAI models** - Use Anthropic Claude, Google Gemini, or Chinese LLMs via OpenRouter
- API keys ONLY in .env files, never hardcoded
- All code changes require tests

---

## Current Status (2025-12-04)

### Phase 7.8: Storyboard Refactoring + Video Integration - COMPLETE
**Branch**: `main`
**Tests**: 730 total (230 storyboard+demo tests)

Major refactoring and video output format support.

✅ **Config Extraction** (`src/tools/storyboard/storyboard_config.py` - NEW)
- VALUE_ANGLE_INSTRUCTIONS (COI/ROI/EASE per persona)
- SECTION_HEADERS (audience-specific)
- VISUAL_STYLE_INSTRUCTIONS (clean, polished, photo_realistic, minimalist)
- ARTIST_STYLE_INSTRUCTIONS (salvador_dali, monet, van_gogh, warhol, picasso, **kurt_geiger**)
- FORMAT_LAYOUT_INSTRUCTIONS (storyboard 9:16, infographic 16:9)
- Helper functions: `get_value_angle_instruction()`, `get_section_headers()`, etc.

✅ **gemini_client.py Refactored** (1561 → 1145 lines, -27%)
- Removed hardcoded tagline default (was causing "canned responses")
- Simplified persona context injection
- Random creative hooks added to transcript extraction
- Cleaner separation of concerns

✅ **Video Generation** (`src/tools/video/video_generator.py`)
- Added Replicate provider (Luma Ray Flash 2 model)
- VideoProvider.REPLICATE enum value
- `_generate_with_replicate()`, `_poll_replicate_completion()` methods
- Demo router routes `video_horizontal` / `video_vertical` formats

✅ **Demo UI** (`static/demo.html`, `src/demo/router.py`)
- Video output format options in dropdown
- Video player element for MP4 playback
- Format detection and routing

**Needs**: `REPLICATE_API_TOKEN` in .env for video generation

---

### Phase 7.7: Persona Differentiation with COI/ROI/EASE - COMPLETE
**Branch**: `main`

Each persona now gets distinctly different output through value angle framing.

✅ **Value Angle System** (`src/tools/storyboard/gemini_client.py`)
| Audience | Value Angle | Framing |
|----------|-------------|---------|
| business_owner | COI (Cost of Inaction) | What they LOSE by not acting |
| c_suite | ROI (Return on Investment) | What they GAIN from this |
| btl_champion | COI | Risk/pain of NOT having this |
| top_tier_vc | ROI | Investment opportunity framing |
| field_crew | EASE | Making their job simpler |

✅ **Persona Generation Context** (`_build_persona_generation_context()`)
- Rich persona data (cares_about, hooks, value_framing) now injected into generation
- Field Crew gets special handling: simple icons, 5th grade vocabulary
- C-Suite gets data visualization, numbers prominent

✅ **Extraction Rules** (all understand_* methods)
- NEVER output "Not mentioned in transcript" - must INFER
- NEVER use personal names - generalize to roles/personas
- Value angle framing injected at extraction phase

---

### Phase 7.5: Knowledge → Storyboard Integration - COMPLETE
**Branch**: `main`
**Tests**: 58 knowledge tests

Knowledge Brain now feeds directly into storyboard generation for dynamic, learning-based content.

✅ **Knowledge Base Schema** (`sql/004_coperniq_knowledge.sql`) - MIGRATED
- `knowledge_sources` - Track ingestion sources (Close CRM, Loom, Miro, Code)
- `coperniq_knowledge` - Core knowledge entries (features, pain points, metrics, quotes)
- `knowledge_tags` - Flexible tagging system
- `knowledge_extraction_logs` - Audit trail
- Full-text search with `search_knowledge()` function
- Pre-seeded with 22 banned terms + 14 approved terms

✅ **Knowledge Module** (`src/knowledge/`)
| Component | Purpose |
|-----------|---------|
| `base.py` | Core types: `KnowledgeSource`, `KnowledgeEntry`, `KnowledgeType` |
| `cache.py` | **NEW** Singleton cache with startup preload |
| `extraction.py` | LLM extraction using DeepSeek V3.2 |
| `close_crm.py` | Close CRM ingestion (calls + notes) |
| `service.py` | Main orchestrator with query methods |

✅ **Storyboard Integration** (`src/tools/storyboard/gemini_client.py`)
- `_build_language_guidelines(audience)` - Merges preset + knowledge banned/approved terms
- `_build_knowledge_context(audience)` - Injects pain points, features, metrics, quotes
- `understand_code/image/multiple_images` - Now audience-aware with knowledge enrichment

✅ **Startup Preload** (`src/api.py`)
- FastAPI lifespan loads KnowledgeCache at startup
- Graceful degradation: storyboards work even if knowledge fails

✅ **Knowledge CLI** (`knowledge_cli.py`)
```bash
# Ingest from Close CRM (last 7 days)
python knowledge_cli.py close --days 7

# Ingest a Loom transcript
python knowledge_cli.py loom --url "https://loom.com/share/abc" --transcript-file transcript.txt

# Ingest a Miro board screenshot
python knowledge_cli.py miro --image /path/to/screenshot.png

# Ingest code file
python knowledge_cli.py code --file /path/to/feature.py

# Search knowledge base
python knowledge_cli.py search "scheduling pain points"

# Show stats
python knowledge_cli.py stats
```

**Architecture:**
```
INGEST → EXTRACT (LLM) → STORE (Supabase)
- Close CRM calls/notes → Pain points, metrics, quotes
- Loom transcripts → Feature mentions, use cases
- Miro screenshots → Roadmap features, product areas
- Engineer code → Feature names, capabilities
```

**Knowledge Types Extracted:**
- `pain_point` - Customer frustrations ("PM lives in Excel")
- `metric` - Specific numbers ("$3K/job", "5 hours/week")
- `quote` - Verbatim customer quotes
- `feature` - Product features (Receptionist AI, Document Engine)
- `approved_term` - Language that resonates
- `banned_term` - Language to avoid
- `objection` - Sales objections
- `competitor` - Competitor mentions
- `use_case` - Specific use cases
- `success_story` - Customer wins

### Today's Work Summary (2025-12-04)
- **Phase 7.7 Persona Differentiation**: COI/ROI/EASE value angles per persona
- **Value Angle Extraction**: All understand_* methods now include value framing instructions
- **Extraction Rules**: Never output "Not mentioned", never use personal names
- **Generation Context**: Rich persona data (cares_about, hooks) injected into Nano Banana prompts
- **Cleanup**: Deleted unused src/tools/scientia/ module (1264 lines)
- **Tests**: 260 storyboard+knowledge tests passing

### Key Architectural Patterns

**Singleton Cache Pattern** (KnowledgeCache)
```python
cache = KnowledgeCache.get()  # Always same instance
await cache.load()            # Idempotent, only loads once
cache.get_language_guidelines("c_suite")  # Fast in-memory access
```

**Graceful Degradation Pattern**
```python
try:
    from src.knowledge.cache import KnowledgeCache
    cache = KnowledgeCache.get()
    if cache.is_loaded():
        # Use knowledge
except Exception:
    pass  # Fall back to static presets
```

**Two-Phase Architecture** (EXTRACT → GENERATE)
```
Input → [FULL EXTRACTION] → Specific understanding → [MARKETING TRANSFORM] → External-ready
```

**Guidance Over Templates** (Storyboard Prompts)
```
❌ RIGID: "1. THE PROBLEM: {x} 2. THE SOLUTION: {y} 3. MARKET: $200B TAM..."
✅ FLEXIBLE: "INVESTOR MINDSET: What's defensible? Show traction. CREATIVE FREEDOM: Design however best tells this story."
```
Give the model guardrails (forbidden words, tone) but let it choose structure and layout.

**Value Angle Framing** (COI/ROI/EASE)
```python
def _get_value_angle_instruction(self, audience: str) -> str:
    """
    COI (Cost of Inaction): What they LOSE by not acting
    ROI (Return on Investment): What they GAIN from this
    EASE: How much simpler their life becomes
    """
    # business_owner, btl_champion → COI
    # c_suite, top_tier_vc → ROI
    # field_crew → EASE
```
Use at both EXTRACTION and GENERATION phases for consistent persona differentiation.

**Extraction Critical Rules** (Never Violate)
```
1. NEVER output "Not mentioned in transcript" - must INFER from context
2. NEVER use personal names - generalize to roles/personas
3. ALWAYS derive business value - even if not explicitly stated
```

## Completed Phases

### Phase 1: SDK Foundation - COMPLETE
**Branch**: `main`
**Tests**: 59 SDK tests

### Phase 2: 3-Way Plugin Integration - COMPLETE
**Branch**: `main`
**Tests**: 28 new tests (observability + integration)

✅ **Audit/Observability Layer** (`src/observability/`)
- `AuditRecord` - Full audit trail for tool executions
- `AuditLogger` - In-memory logger with Supabase persistence ready
- `@audit_logged` decorator - Auto-log any async function

✅ **dealer-scraper-mvp Plugin** (`plugins/scraper_tools/`)
- `DealerLocatorTool` - Scrape 25+ OEM dealer locators
- `ContractorEnrichTool` - Apollo/Clay/Hunter enrichment
- `LicenseValidateTool` - State license validation

✅ **sales-agent Plugin** (`plugins/sales_tools/`)
- `OutreachTool` - Email/SMS/LinkedIn outreach
- `QualifyTool` - Lead scoring (0-100)
- `CRMSyncTool` - Close/HubSpot/Apollo sync

### Phase 3: Video Tools Module - COMPLETE (2025-12-02)
**Branch**: `main`
**Tests**: 186 new tests (video module)

✅ **Video Prospecting Tools** (`src/tools/video/`)
| Tool | Purpose |
|------|---------|
| `VideoScriptGeneratorTool` | Generate 60-sec Loom scripts via DeepSeek V3 |
| `VideoGeneratorTool` | Multi-provider video gen (Kling/HaiLuo/Runway/Pika/Luma) |
| `BatchVideoGeneratorTool` | Batch video processing |
| `LoomViewTrackerTool` | View analytics + engagement scoring |
| `ViewerEnrichmentTool` | Apollo/Hunter enrichment |
| `VideoSchedulerTool` | Optimal send time prediction |
| `VideoTemplateManagerTool` | Industry-specific templates (solar, hvac, electrical, roofing, mep) |

**Usage:**
```python
from src.tools.video import (
    VideoScriptGeneratorTool,
    VideoGeneratorTool,
    BatchVideoGeneratorTool,
    LoomViewTrackerTool,
    ViewerEnrichmentTool,
    VideoSchedulerTool,
    VideoTemplateManagerTool,
)

# Generate video script for solar prospect
script_tool = VideoScriptGeneratorTool()
result = await script_tool.run({
    "product_name": "SolarMax CRM",
    "prospect_company": "Acme Solar",
    "prospect_name": "John Smith",
    "industry": "solar",
    "key_pain_point": "permit tracking",
})

# Schedule optimal send time
scheduler = VideoSchedulerTool()
result = await scheduler.run({
    "prospect_timezone": "America/New_York",
    "industry": "solar",
    "role_level": "director",
    "use_llm": False,  # Fast mode without LLM
})
```

### Phase 4: Storyboard Tools Module - COMPLETE (2025-12-02)
**Branch**: `main`
**Tests**: 202 tests (storyboard module)

✅ **Storyboard Tools** (`src/tools/storyboard/`)
| Tool | Purpose |
|------|---------|
| `CodeToStoryboardTool` | Transform code files into executive PNG storyboards |
| `RoadmapToStoryboardTool` | Transform Miro/roadmaps into sanitized "Coming Soon" teasers |
| `UnifiedStoryboardTool` | **NEW** - All-in-one tool accepting ANY input (Miro URL, image, code) |
| `GeminiStoryboardClient` | Two-stage pipeline: Understand (Vision) → Generate (Image Gen) |
| `coperniq_presets` | ICP-optimized presets for MEP+energy contractors |

**Features:**
- Gemini 2.0 Flash Vision + Gemini 3 Pro Image Preview (NO OpenAI)
- Automatic IP sanitization (strips code internals, API keys, secrets)
- Three stages: preview, demo, shipped
- **Five audiences**: business_owner, c_suite, btl_champion, **top_tier_vc**, **field_crew**
- Target: MEP+energy contractors ($5M+ ICP)

### Phase 6: Storyboard Demo App - COMPLETE (2025-12-04)
**Branch**: `main`
**Tests**: 18 new tests (demo router)

✅ **Demo Application** (`src/demo/`, `static/`, `demo_cli.py`)
| Component | Description |
|-----------|-------------|
| `src/demo/router.py` | FastAPI endpoints: `/demo/examples`, `/demo/generate` |
| `demo_cli.py` | CLI script with 10 example commands |
| `static/demo.html` | Web UI with drag/drop, paste (Cmd+V), file browse |
| `vercel.json` | Deployment config for Vercel |

**New Audiences Added:**
- `top_tier_vc` - VC pitch focus (UVP, moat, 10x better, TAM/SAM/SOM)
- `field_crew` - Simple infographics for blue collar workers (5th grade vocabulary)

**Usage:**
```bash
# CLI - Generate storyboard from project code
python demo_cli.py --example video_script_generator --audience field_crew --stage demo

# Web UI - Local
uvicorn src.api:app --reload
# Open http://localhost:8000

# Web UI - Production
# https://conductor-ai.vercel.app (requires GOOGLE_API_KEY in Vercel env)
```

**Deployment:** Vercel project `conductor-ai` under `scientia-capital`
- **Auto-opens result in default browser**

**Usage:**
```python
from src.tools.storyboard import (
    CodeToStoryboardTool,
    RoadmapToStoryboardTool,
    UnifiedStoryboardTool,
)

# RECOMMENDED: Use UnifiedStoryboardTool for any input type
tool = UnifiedStoryboardTool()

# From code string - opens in browser automatically
result = await tool.run({
    "input": "def calculate_roi(): return revenue - costs",
    "audience": "c_suite",
})

# From Miro screenshot (paste base64)
result = await tool.run({
    "input": "data:image/png;base64,iVBORw0KGgo...",
    "stage": "demo",
})

# From code file
result = await tool.run({
    "input": "/path/to/calculator.py",
})
# result.result["storyboard_png"] = base64 PNG
# result.result["file_path"] = path to saved PNG
```

### Phase 5: Storyboard Pipeline API - COMPLETE (2025-12-02)
**Branch**: `main`
**Tests**: 61 new API tests (229 total storyboard tests)

✅ **Async FastAPI Endpoints** (`src/storyboard/`)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/storyboard/code` | POST | Code → PNG storyboard (202 + job_id) |
| `/storyboard/roadmap` | POST | Roadmap screenshot → PNG teaser (202 + job_id) |
| `/storyboard/jobs/{job_id}` | GET | Poll job status and results |

**Architecture:**
- Redis for hot state (1hr TTL), Supabase for cold persistence
- Multi-tenant isolation with X-Org-ID header validation
- 5-minute timeout protection on background tasks
- Singleton job manager to prevent connection leaks

**Usage:**
```bash
# Generate storyboard from code
curl -X POST http://localhost:8000/storyboard/code \
  -H "Content-Type: application/json" \
  -H "X-Org-ID: my-org" \
  -d '{"file_content": "def calculate_roi(): pass", "stage": "preview"}'

# Poll for result
curl http://localhost:8000/storyboard/jobs/{job_id} \
  -H "X-Org-ID: my-org"
```

### SQL Migrations - PENDING USER ACTION
**File**: `sql/001_audit_and_leads.sql`
- Run in Supabase SQL Editor to create:
  - `audit_logs` - Tool execution audit trail
  - `tool_executions` - Detailed I/O records
  - `leads` - Shared lead storage

**File**: `sql/002_storyboard_jobs.sql`
- Run in Supabase SQL Editor to create:
  - `storyboard_jobs` - Job state persistence with RLS

**File**: `sql/004_coperniq_knowledge.sql` (NEW)
- Run in Supabase SQL Editor to create:
  - `knowledge_sources` - Ingestion source tracking
  - `coperniq_knowledge` - Knowledge entries with full-text search
  - `knowledge_tags` - Flexible tagging
  - `knowledge_extraction_logs` - Extraction audit trail
  - Views: `active_knowledge`, `pain_points_for_storyboards`, `approved_terms`, `banned_terms`
  - Functions: `search_knowledge()`, `get_knowledge_for_storyboard()`

---

## SDK Quick Start

### For Plugin Developers (5 imports only)
```python
from conductor_ai.sdk import (
    BaseTool,
    ToolCategory,
    ToolDefinition,
    ToolResult,
    PluginRegistry,
)

registry = PluginRegistry()

@registry.tool
class MyTool(BaseTool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="my_tool",
            description="Does something useful",
            category=ToolCategory.DATA,
            parameters={"type": "object", "properties": {...}},
        )

    async def run(self, arguments: dict) -> ToolResult:
        # Your implementation
        return ToolResult(
            tool_name="my_tool",
            success=True,
            result={"data": "..."},
            execution_time_ms=100,
        )

def register(global_registry):
    """Called by PluginLoader to register tools."""
    for tool in registry.tools:
        global_registry.register(tool)
```

### Using the HTTP Client
```python
from conductor_ai.sdk import ConductorClient

async with ConductorClient("http://localhost:8000", org_id="my-org") as client:
    # Start agent
    session = await client.run_agent(
        messages=[{"role": "user", "content": "Scrape Tesla dealer locator"}],
        tools=["web_fetch", "my_tool"],
    )

    # Wait for completion
    result = await client.wait_for_completion(session.session_id)
    print(result.steps[-1]["final_answer"])
```

---

## Model Catalog (2025 Best-in-Class)

### Flagship Models
| Model | Provider | Best For | Input/Output per 1M |
|-------|----------|----------|---------------------|
| DeepSeek V3.1 | deepseek | Agents, coding, reasoning | $0.20 / $0.80 |
| DeepSeek R1 | deepseek | Deep reasoning, math | $3.00 / $7.00 |
| Qwen3-235B | alibaba | Hybrid thinking, multilingual | $0.30 / $1.20 |
| Kimi K2 | moonshot | Coding, tool-use | $0.15 / $0.60 |
| Claude Sonnet 4.5 | anthropic | Vision, analysis | $3.00 / $15.00 |

### Smart Model Selection (Black Box)
```python
from src.model_catalog import select_model

# Automatic best-model selection
model = select_model(task="coding")           # -> qwen/qwen-2.5-coder-32b-instruct
model = select_model(task="reasoning")        # -> deepseek/deepseek-r1
model = select_model(task="agents")           # -> deepseek/deepseek-chat-v3
model = select_model(budget="budget")         # -> deepseek/deepseek-r1-distill-qwen-8b
model = select_model(require_vision=True)     # -> claude-sonnet-4-5-20250929
model = select_model(prefer_chinese=True)     # -> deepseek/deepseek-chat-v3
```

---

## Directory Structure

```
conductor-ai/
├── src/
│   ├── sdk/                    # Plugin SDK (public API)
│   │   ├── __init__.py         # Exports: BaseTool, ToolCategory, etc.
│   │   ├── registry.py         # PluginRegistry, PluginLoader
│   │   ├── client.py           # ConductorClient HTTP client
│   │   ├── security/           # SSRF, timeout, approval utilities
│   │   └── testing/            # MockRegistry, ToolTestBase
│   ├── tools/                  # Core tools
│   │   ├── base.py             # BaseTool, ToolDefinition, ToolResult
│   │   ├── registry.py         # Global ToolRegistry singleton
│   │   ├── web_fetch.py        # HTTP requests
│   │   ├── code_run.py         # Docker sandboxed code
│   │   ├── sql_query.py        # Supabase queries
│   │   ├── video/              # Video prospecting tools (7 tools, 3.5k LOC)
│   │   └── storyboard/         # Storyboard tools (3 tools, 202 tests)
│   ├── knowledge/              # Knowledge Brain
│   │   ├── base.py             # Core types: KnowledgeSource, KnowledgeEntry
│   │   ├── cache.py            # Singleton cache with startup preload
│   │   ├── extraction.py       # LLM extraction via DeepSeek V3.2
│   │   ├── close_crm.py        # Close CRM ingestion
│   │   └── service.py          # Main orchestrator + queries
│   ├── agents/                 # Agent system
│   │   ├── schemas.py          # AgentSession, AgentStep, etc.
│   │   ├── state.py            # Redis + Supabase state manager
│   │   └── runner.py           # ReAct loop executor
│   ├── model_catalog.py        # Model catalog + smart selector
│   └── api.py                  # FastAPI endpoints
├── plugins/                    # Plugin directory (auto-discovered)
│   └── example_plugin/         # Example plugin template
├── sql/
│   ├── 001_audit_and_leads.sql
│   ├── 002_storyboard_jobs.sql
│   └── 004_coperniq_knowledge.sql  # Knowledge Brain schema (NEW)
├── tests/
│   ├── sdk/                    # SDK tests (59 tests)
│   ├── tools/                  # Tool tests
│   └── knowledge/              # Knowledge tests (58 tests)
├── knowledge_cli.py            # Knowledge ingestion CLI (NEW)
├── demo_cli.py                 # Storyboard demo CLI
├── pyproject.toml              # Package config with SDK extras
├── .claude/
│   └── commands/               # Pipeline commands
│       ├── pipeline-feature.md # 6-phase feature development workflow
│       └── pipeline-eod.md     # End of day audit/sync workflow
└── docker-compose.yml          # Redis + API services
```

---

## Pipeline Commands (NEW 2025-12-03)

### /pipeline:feature [description]
6-phase feature development workflow with gates:
1. **Planning** - Brainstorm + write plan
2. **Database** - Schema design + migrations
3. **Parallel Implementation** - Backend + Frontend + Tests agents
4. **Integration** - Wire components + security scan
5. **Testing** - Full test suite + coverage
6. **Code Review** - BLOCKING gate, must pass

### /pipeline:eod
End of day audit/sync workflow:
1. **Audit** - Review git activity, summarize work
2. **Security Sweep** - Secrets scan, OpenAI check
3. **Docs Update** - TASK.md, CLAUDE.md
4. **Code Quality** - Lint, type check
5. **Git Sync** - Commit, push, worktree cleanup
6. **Tomorrow Context** - Generate "Start Here" context

---

## Development Commands

```bash
# Start services
docker-compose up -d

# Run all tests
python3 -m pytest tests/ -v

# SDK tests only
python3 -m pytest tests/sdk/ -v

# Install with SDK extras
pip install -e ".[sdk]"

# Install with all features
pip install -e ".[all]"
```

---

## Environment Variables

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...
OPENROUTER_API_KEY=sk-or-...
REDIS_URL=redis://localhost:6379
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_SERVICE_KEY=eyJ...

# Optional
GOOGLE_API_KEY=...              # For Gemini storyboard generation
CLOSE_API_KEY=...               # For Close CRM knowledge ingestion (NEW)
APP_ENV=development
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/agents/run` | Start agent execution |
| GET | `/agents/{id}` | Poll session status |
| POST | `/agents/{id}/cancel` | Cancel running session |
| GET | `/tools` | List available tools |
| GET | `/health` | Health check |
| POST | `/storyboard/code` | Code → PNG storyboard (202 async) |
| POST | `/storyboard/roadmap` | Roadmap → PNG teaser (202 async) |
| GET | `/storyboard/jobs/{job_id}` | Poll storyboard job status |

---

## Plugin Integration Plan

### Tier 1: HIGH Priority
| Project | Tool | Value |
|---------|------|-------|
| dealer-scraper-mvp | web_fetch | Fix 16 broken scrapers |
| sales-agent | web_fetch | LinkedIn scraping, lead enrichment |

### Tier 2: MEDIUM Priority
| Project | Tool | Value |
|---------|------|-------|
| vozlux | sql_query | Real-time booking availability |
| netzeroexpert-os | code_run | Energy calculations |
| quantify-mvp | code_run | PDF batch processing |

---

## Architecture

### 3-Way Plugin Integration
```
┌─────────────────────────────────────────────────────────────────┐
│                        CONDUCTOR-AI                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ AgentRunner │  │ ToolRegistry │  │ AuditLogger             │  │
│  │ (ReAct)     │──│ (Global)    │──│ @audit_logged decorator │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│         │                │                       │               │
│         │        ┌───────┴───────┐               │               │
│         │        ▼               ▼               ▼               │
│  ┌──────┴────────────┐   ┌────────────────┐   ┌───────┐         │
│  │ dealer-scraper-mvp│   │ sales-agent    │   │Supabase│        │
│  │ Plugin            │   │ Plugin         │   │        │        │
│  │ ┌───────────────┐ │   │ ┌────────────┐ │   │audit_  │        │
│  │ │DealerLocator  │ │   │ │OutreachTool│ │   │logs    │        │
│  │ │ContractorEnr. │ │   │ │QualifyTool │ │   │leads   │        │
│  │ │LicenseValidate│ │   │ │CRMSyncTool │ │   │tool_   │        │
│  │ └───────────────┘ │   │ └────────────┘ │   │exec    │        │
│  └───────────────────┘   └────────────────┘   └───────┘         │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow
```
1. Scrape    → dealer-scraper-mvp/DealerLocatorTool
                     ↓
2. Enrich    → dealer-scraper-mvp/ContractorEnrichTool
                     ↓
3. Validate  → dealer-scraper-mvp/LicenseValidateTool
                     ↓
4. Qualify   → sales-agent/QualifyTool (score 0-100)
                     ↓
5. Outreach  → sales-agent/OutreachTool (email/sms/linkedin)
                     ↓
6. Sync      → sales-agent/CRMSyncTool (close/hubspot/apollo)
                     ↓
All steps   → AuditLogger → Supabase:audit_logs
```

### ReAct Loop
```
Client Request
     │
     ▼
POST /agents/run
     │
     ▼
AgentRunner (ReAct Loop)
     │
     ├─► LLM Provider (Claude / DeepSeek / Qwen / Gemini)
     │
     ├─► Tool Execution
     │   ├─► web_fetch (HTTP)
     │   ├─► code_run (Docker)
     │   ├─► sql_query (Supabase)
     │   └─► [Custom Plugin Tools]
     │
     └─► StateManager
         ├─► Redis (hot state, 1hr TTL)
         └─► Supabase (cold persistence)
```

---

## Contact
- **Repo**: https://github.com/ScientiaCapital/conductor-ai
- **Branch**: main
