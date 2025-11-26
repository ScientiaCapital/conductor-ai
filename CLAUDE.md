# Conductor-AI - Project Instructions

## Critical Rules
- **NO OPENAI models** - Use Anthropic Claude 4.5 or Google Gemini only
- API keys ONLY in .env files, never hardcoded
- All code changes require tests

---

## Team Handoff (2025-11-25)

### What's Done (5/9 Tasks Complete)
**Branch**: `feature/agent-system` (pushed to origin)
**Worktree**: `.worktrees/agent-system`
**Tests**: 204 passing

| Task | File | Tests |
|------|------|-------|
| Agent Schemas | `src/agents/schemas.py` | 44 |
| Tool Base Classes | `src/tools/base.py`, `registry.py` | 32 |
| web_fetch Tool | `src/tools/web_fetch.py` | 38 |
| code_run Tool | `src/tools/code_run.py` | 37 |
| sql_query Tool | `src/tools/sql_query.py` | 53 |

### What's Left (4 Tasks Remaining)

#### Task 6: State Manager (NEXT)
**File**: `src/agents/state.py`
**Plan**: `docs/plans/2025-01-25-agent-implementation.md` (lines 100-114)

Build Redis + Supabase hybrid state manager:
```python
class StateManager:
    create_session(org_id, model) -> AgentSession
    get_session(session_id) -> AgentSession | None  # Redis first, Supabase fallback
    update_session(session) -> None
    add_step(session_id, step) -> None
    get_steps(session_id) -> list[AgentStep]
    persist_to_supabase(session) -> None  # On completion
    update_token_usage(session_id, input_tokens, output_tokens, cost) -> None
```

Redis keys: `session:{id}`, `steps:{id}` (1 hour TTL)

#### Task 7: AgentRunner (ReAct Loop)
**File**: `src/agents/runner.py`
**Plan**: `docs/plans/2025-01-25-agent-implementation.md` (lines 116-129)

Build ReAct execution loop:
1. Build system prompt with tool definitions
2. Call Claude 4.5 with conversation history
3. Parse JSON response (thought/action/is_final)
4. If is_final: return final_answer
5. If action: run tool, add observation
6. Repeat until max_steps

#### Task 8: API Endpoints
**File**: `src/api.py`
**Plan**: `docs/plans/2025-01-25-agent-implementation.md` (lines 131-143)

FastAPI routes:
- `POST /v1/agents/run` - Start session
- `GET /v1/agents/{id}` - Poll status
- `POST /v1/agents/{id}/cancel` - Cancel session
- `GET /v1/tools` - List tools

#### Task 9: Integration Test
**File**: `tests/integration/test_full.py`

End-to-end test with real Claude API call.

---

## Where to Find Everything

### Working Directory
```bash
cd /Users/tmkipper/Desktop/tk_projects/conductor-ai/.worktrees/agent-system
```

### Key Files
- **Implementation Plan**: `docs/plans/2025-01-25-agent-implementation.md`
- **Architecture Design**: `docs/plans/2025-01-25-agent-orchestration-design.md`
- **Schemas (reference)**: `src/agents/schemas.py`
- **Tool patterns**: `src/tools/web_fetch.py` (good example)

### Run Tests
```bash
# All tests
python -m pytest tests/ -v

# Specific tests
python -m pytest tests/tools/test_web_fetch.py -v
```

---

## Supabase Details

### Project
- **URL**: Set in `SUPABASE_URL` env var
- **Dashboard**: https://supabase.com/dashboard

### Required Tables
Create these tables for agent state persistence:

```sql
-- Agent sessions table
CREATE TABLE agent_sessions (
    id UUID PRIMARY KEY,
    org_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    model TEXT NOT NULL,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    total_cost_usd DECIMAL(10,6) DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Agent steps table
CREATE TABLE agent_steps (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES agent_sessions(id),
    step_number INTEGER NOT NULL,
    thought TEXT,
    action JSONB,
    observation TEXT,
    is_final BOOLEAN DEFAULT FALSE,
    final_answer TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_sessions_org ON agent_sessions(org_id);
CREATE INDEX idx_sessions_status ON agent_sessions(status);
CREATE INDEX idx_steps_session ON agent_steps(session_id);
```

### Environment Variables
```bash
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_SERVICE_KEY=eyJ...  # Service role key (from Settings > API)
SUPABASE_DB_URL=postgresql://postgres:xxx@db.xxx.supabase.co:5432/postgres
```

---

## Redis Setup

### Local Development
```bash
# Start with Docker Compose
docker-compose up -d redis

# Or standalone
docker run -d -p 6379:6379 redis:7-alpine
```

### Environment
```bash
REDIS_URL=redis://localhost:6379
```

---

## Vercel (Future Deployment)

Not deployed yet - API server will need:
- Python runtime (or containerized)
- Environment variables from above
- Redis addon or external Redis
- Supabase connection

Consider Railway or Render for Python API hosting.

---

## Git Workflow

```bash
# You're on feature/agent-system branch
git status
git add .
git commit -m "feat(agents): Description"
git push origin feature/agent-system

# When ready, create PR to main
gh pr create --title "Agent System MVP" --body "..."
```

---

## Quick Commands

```bash
# Navigate to worktree
cd .worktrees/agent-system

# Run all tests
python -m pytest tests/ -v

# Run tests with coverage
python -m pytest tests/ -v --cov=src

# Check what's changed
git status
git diff

# Start Redis
docker-compose up -d redis
```

---

## Architecture Summary

```
Client Request
     │
     ▼
POST /v1/agents/run
     │
     ▼
AgentRunner (ReAct Loop)
     │
     ├─► Claude 4.5 API (thought/action)
     │
     ├─► Tool Execution
     │   ├─► web_fetch (HTTP)
     │   ├─► code_run (Docker)
     │   └─► sql_query (Supabase)
     │
     └─► StateManager
         ├─► Redis (hot state)
         └─► Supabase (persistence)
```

---

## Contact

- **Repo**: https://github.com/ScientiaCapital/conductor-ai
- **Branch**: feature/agent-system
- **PR**: https://github.com/ScientiaCapital/conductor-ai/pull/new/feature/agent-system
