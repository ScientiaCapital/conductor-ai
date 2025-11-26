# Conductor-AI - Project Instructions

## Critical Rules
- **NO OPENAI models** - Use Anthropic Claude 4.5 or Google Gemini
- API keys ONLY in .env files, never hardcoded
- All tools require security review before deployment

## Project Status (2025-11-25)

### Current Phase: Agent Orchestration System
**Branch**: `feature/agent-system`
**Worktree**: `.worktrees/agent-system`

### Completed Tasks (5/9)
1. **Task 1: Agent Schemas** - 44 tests passing
   - `src/agents/schemas.py`: SessionStatus, ToolCall, AgentStep, AgentSession
   - Pydantic v2 models with validation

2. **Task 2: Tool Base Classes** - 32 tests passing
   - `src/tools/base.py`: BaseTool abstract class, ToolDefinition, ToolResult
   - `src/tools/registry.py`: ToolRegistry singleton with decorator

3. **Task 3: web_fetch Tool** - 38 tests passing
   - SSRF protection (DNS resolution + redirect blocking)
   - URL validation, 50KB response limit, 30s timeout

4. **Task 4: code_run Tool** - 37 tests passing
   - Docker sandbox: python:3.11-slim, node:20-slim
   - Resource limits: 128MB RAM, 50% CPU, no network
   - Subprocess fallback for development

5. **Task 5: sql_query Tool** - 53 tests passing
   - Dangerous pattern detection (DROP, TRUNCATE, etc.)
   - DELETE/UPDATE require WHERE clause
   - Parameterized queries, requires approval

### Remaining Tasks (4/9)
6. **Task 6: State Manager** - Redis (hot) + Supabase (cold)
7. **Task 7: AgentRunner** - ReAct loop implementation
8. **Task 8: API Endpoints** - /v1/agents/* routes
9. **Task 9: Integration Test** - End-to-end testing

## Architecture

### Tech Stack
- **Backend**: FastAPI, Pydantic v2, asyncio
- **AI Models**: Claude 4.5 Sonnet/Opus (primary), Qwen/DeepSeek via OpenRouter (budget)
- **State**: Redis (hot, 1hr TTL) + Supabase PostgreSQL (cold)
- **Sandbox**: Docker containers with resource limits

### ReAct Pattern
```
1. Build system prompt with tool definitions
2. Call LLM with conversation history
3. Parse JSON response (thought/action/is_final)
4. If is_final: return final_answer
5. If action: run tool, add observation
6. Repeat until max_steps
```

### API Endpoints (Planned)
- `POST /v1/agents/run` - Start session, return session_id
- `GET /v1/agents/{id}` - Poll status and steps
- `POST /v1/agents/{id}/cancel` - Cancel running session
- `GET /v1/tools` - List available tools

## Project Structure
```
conductor-ai/
├── .worktrees/agent-system/     # Active worktree
│   ├── src/
│   │   ├── agents/
│   │   │   ├── schemas.py       # Pydantic schemas
│   │   │   └── state.py         # State manager (WIP)
│   │   └── tools/
│   │       ├── base.py          # BaseTool, ToolRegistry
│   │       ├── web_fetch.py     # HTTP fetching tool
│   │       ├── code_run.py      # Docker code sandbox
│   │       └── sql_query.py     # Supabase SQL tool
│   ├── tests/
│   │   ├── agents/              # Schema tests
│   │   ├── tools/               # Tool tests
│   │   └── integration/         # E2E tests (pending)
│   └── docs/plans/
│       ├── 2025-01-25-agent-orchestration-design.md
│       └── 2025-01-25-agent-implementation.md
├── builder/                     # vLLM builder (existing)
├── src/                         # Main source
│   ├── model_catalog.py         # Claude 4.5 models added
│   └── cost_optimizer.py        # Claude 4.5 costs
└── docker-compose.yml           # Redis + services
```

## Environment Variables
```bash
# Redis
REDIS_URL=redis://localhost:6379

# Supabase
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_SERVICE_KEY=xxx
SUPABASE_DB_URL=postgresql://...

# AI Providers
ANTHROPIC_API_KEY=xxx
```

## Development Commands
```bash
# Work in agent-system worktree
cd .worktrees/agent-system

# Run all tests
python -m pytest tests/ -v

# Run specific tool tests
python -m pytest tests/tools/test_web_fetch.py -v

# Start Redis (from main repo)
docker-compose up -d redis
```

## Security Checklist
- [x] SSRF protection in web_fetch (DNS + redirects)
- [x] SQL injection prevention in sql_query (parameterized)
- [x] Docker sandbox isolation in code_run
- [x] No shell=True anywhere
- [x] Approval required for sql_query
- [ ] Rate limiting (Task 8)
- [ ] Authentication (Task 8)

## GitHub Repository
- **URL**: https://github.com/ScientiaCapital/conductor-ai
- **Main Branch**: main
- **Feature Branch**: feature/agent-system
- **PR Ready**: https://github.com/ScientiaCapital/conductor-ai/pull/new/feature/agent-system
