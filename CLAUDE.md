# Conductor-AI - Project Instructions

## Critical Rules
- **NO OPENAI models** - Use Anthropic Claude, Google Gemini, or Chinese LLMs via OpenRouter
- API keys ONLY in .env files, never hardcoded
- All code changes require tests

---

## Current Status (2025-12-02)

### Phase 1: SDK Foundation - COMPLETE
**Branch**: `main`
**Tests**: 59 SDK tests + 204 core tests + 186 video tests + 229 storyboard tests = 678 total

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
**Tests**: 168 new tests (storyboard module)

✅ **Storyboard Tools** (`src/tools/storyboard/`)
| Tool | Purpose |
|------|---------|
| `CodeToStoryboardTool` | Transform code files into executive PNG storyboards |
| `RoadmapToStoryboardTool` | Transform Miro/roadmaps into sanitized "Coming Soon" teasers |
| `GeminiStoryboardClient` | Two-stage pipeline: Understand (Vision) → Generate (Image Gen) |
| `coperniq_presets` | ICP-optimized presets for MEP+energy contractors |

**Features:**
- Gemini 2.0 Flash Vision + Image Generation (NO OpenAI)
- Automatic IP sanitization (strips code internals, API keys, secrets)
- Three stages: preview, demo, shipped
- Three audiences: business_owner, c_suite, btl_champion
- Target: MEP+energy contractors ($5M+ ICP)

**Usage:**
```python
from src.tools.storyboard import (
    CodeToStoryboardTool,
    RoadmapToStoryboardTool,
)

# Transform code to executive storyboard
tool = CodeToStoryboardTool()
result = await tool.run({
    "file_path": "src/calculator.py",
    "icp_preset": "coperniq_mep",
    "stage": "preview",
    "audience": "c_suite",
})
# result.result["storyboard_png"] = base64 PNG image

# Transform roadmap screenshot to teaser
roadmap_tool = RoadmapToStoryboardTool()
result = await roadmap_tool.run({
    "image_path": "roadmap_screenshot.png",
    "sanitize_ip": True,
})
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
│   │   └── storyboard/         # Storyboard tools (2 tools, 168 tests)
│   ├── agents/                 # Agent system
│   │   ├── schemas.py          # AgentSession, AgentStep, etc.
│   │   ├── state.py            # Redis + Supabase state manager
│   │   └── runner.py           # ReAct loop executor
│   ├── model_catalog.py        # Model catalog + smart selector
│   └── api.py                  # FastAPI endpoints
├── plugins/                    # Plugin directory (auto-discovered)
│   └── example_plugin/         # Example plugin template
├── tests/
│   ├── sdk/                    # SDK tests (59 tests)
│   └── tools/                  # Tool tests
├── pyproject.toml              # Package config with SDK extras
└── docker-compose.yml          # Redis + API services
```

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
GOOGLE_API_KEY=...              # For Gemini
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
