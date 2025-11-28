# Conductor-AI - Project Instructions

## Critical Rules
- **NO OPENAI models** - Use Anthropic Claude, Google Gemini, or Chinese LLMs via OpenRouter
- API keys ONLY in .env files, never hardcoded
- All code changes require tests

---

## Current Status (2025-11-27)

### Phase 1: SDK Foundation - COMPLETE
**Branch**: `main`
**Tests**: 59 SDK tests + 204 core tests = 263 total

The plugin SDK is production-ready. External projects can now create custom tools and integrate with conductor-ai agents.

### Phase 2: Project Integrations - IN PROGRESS
**Next**: dealer-scraper-mvp + sales-agent

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
│   │   └── sql_query.py        # Supabase queries
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
