# Conductor-AI Agent Orchestration System Design

**Date:** 2025-01-25
**Status:** Approved
**Author:** Claude + Human Partner

---

## Overview

Conductor-AI is a commercial SaaS agent orchestration platform that enables developers to build AI agents with tool-calling capabilities. The system uses the ReAct (Reasoning + Acting) pattern with structured JSON outputs.

**Key Constraints:**
- NO OpenAI or Groq - Uses Anthropic Claude, Google Gemini, OpenRouter only
- Docker containers for sandboxed code execution
- Async polling model (not SSE for main loop)
- Redis for hot state, Supabase for persistence

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FastAPI Application                         │
├─────────────────────────────────────────────────────────────────────┤
│  POST /v1/agents/run     → Creates session, spawns async executor   │
│  GET  /v1/agents/{id}    → Returns session status + steps           │
│  POST /v1/agents/{id}/cancel → Cancels running session              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
            │Claude 4.5   │  │Claude 4.5   │  │   Qwen /    │
            │  Opus       │  │  Sonnet     │  │  DeepSeek   │
            │ (Flagship)  │  │ (Default)   │  │  (Budget)   │
            └─────────────┘  └─────────────┘  └─────────────┘
                    │               │               │
                    └───────────────┼───────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │     Pydantic JSON Enforcer    │
                    │   (validates all tool calls)  │
                    └───────────────────────────────┘
                                    │
            ┌───────────────┬───────┼───────┬───────────────┐
            ▼               ▼       ▼       ▼               ▼
      ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
      │web_fetch │  │sql_query │  │code_exec │  │file_ops  │
      └──────────┘  └──────────┘  └──────────┘  └──────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼                               ▼
            ┌─────────────┐                  ┌─────────────┐
            │    Redis    │                  │  Supabase   │
            │ (hot state) │                  │ (persist)   │
            └─────────────┘                  └─────────────┘
```

---

## Model Selection Strategy

| Tier | Model | Use Case | Cost (per 1M tokens) |
|------|-------|----------|---------------------|
| Flagship | `claude-opus-4-5-20251101` | Complex reasoning, research | $15 / $75 |
| Default | `claude-sonnet-4-5-20250929` | Most agent tasks | $3 / $15 |
| Budget | `qwen/qwen-2.5-72b-instruct` | High volume, cost-sensitive | $0.35 / $0.40 |
| Fast | `gemini-1.5-flash` | Simple tasks, low latency | $0.075 / $0.30 |

**JSON Enforcement:** All LLM outputs pass through Pydantic v2 validation before tool execution.

---

## ReAct Execution Loop

Each agent step follows the Thought → Action → Observation pattern:

```python
class AgentStep(BaseModel):
    step_number: int
    thought: str                    # LLM's reasoning
    action: Optional[ToolCall]      # Tool to execute (null if final)
    observation: Optional[str]      # Tool result (filled after execution)
    is_final: bool = False          # True when agent is done
    final_answer: Optional[str]     # Final response to user
    tokens_used: int = 0
    latency_ms: int = 0

class ToolCall(BaseModel):
    tool: str                       # Tool name
    args: dict                      # Tool arguments
```

**Loop Flow:**
1. Send messages + tool definitions to LLM
2. Parse structured JSON response (Pydantic validates)
3. If `is_final=True` → Return final_answer, end session
4. Execute tool, capture observation
5. Append step to history, goto 1 (max_steps limit)

---

## Tool Registry

### Available Tools

| Tool | Category | Docker | Approval | Description |
|------|----------|--------|----------|-------------|
| `web_fetch` | web | No | No | HTTP GET/POST requests |
| `sql_query` | data | No | Yes | Execute SQL queries |
| `code_execute` | code | Yes | No | Run Python/JS in sandbox |
| `file_read` | file | Yes | No | Read files in sandbox |
| `file_write` | file | Yes | Yes | Write files (needs approval) |
| `shell_exec` | system | Yes | Yes | Shell commands (restricted) |

### Tool Schema

```python
class ToolDefinition(BaseModel):
    name: str
    description: str               # For LLM context
    parameters: dict               # JSON Schema for args
    category: Literal["web", "data", "code", "file", "system"]
    requires_approval: bool
    timeout_seconds: int = 30

class ToolResult(BaseModel):
    tool_name: str
    success: bool
    result: Any
    error: Optional[str]
    execution_time_ms: int
```

---

## State Management

### Hot State (Redis) - TTL 1 hour

```
session:{id}:status     → "pending" | "running" | "completed" | "failed"
session:{id}:steps      → List of AgentStep JSONs
session:{id}:current    → Current step number
session:{id}:lock       → Distributed lock for execution
```

### Cold State (Supabase) - Permanent

- `agent_sessions` → Session metadata, config, totals
- `agent_steps` → Full step history with tokens/cost
- `tool_executions` → Tool call logs for debugging
- `usage_records` → Billing/analytics data

### Write Pattern

1. Every step → Write to Redis immediately (client polls here)
2. On completion/failure → Batch write to Supabase
3. Redis TTL expires → Data only in Supabase

---

## API Endpoints

### POST /v1/agents/run

Start a new agent session.

**Request:**
```json
{
  "messages": [{"role": "user", "content": "..."}],
  "model": "claude-sonnet-4-5-20250929",
  "tools": ["web_fetch", "code_execute"],
  "max_steps": 10,
  "system_prompt": "You are a helpful assistant..."
}
```

**Response:**
```json
{
  "session_id": "sess_abc123",
  "status": "running",
  "poll_url": "/v1/agents/sess_abc123"
}
```

### GET /v1/agents/{session_id}

Poll session status and steps.

**Response:**
```json
{
  "session_id": "sess_abc123",
  "status": "running",
  "steps": [
    {
      "step_number": 1,
      "thought": "I need to fetch...",
      "action": {"tool": "web_fetch", "args": {...}},
      "observation": "...",
      "is_final": false,
      "tokens_used": 342,
      "latency_ms": 890
    }
  ],
  "current_step": 2,
  "total_tokens": 1205,
  "total_cost_cents": 0.45,
  "final_answer": null,
  "error": null
}
```

### POST /v1/agents/{session_id}/cancel

Cancel a running session.

### POST /v1/agents/{session_id}/approve/{step_id}

Approve a tool execution that requires human approval.

**Request:**
```json
{
  "approved": true,
  "modified_args": {}
}
```

---

## Directory Structure

```
src/
├── agents/
│   ├── __init__.py
│   ├── executor.py          # AgentExecutor class (ReAct loop)
│   ├── schemas.py           # Pydantic models (AgentStep, ToolCall, etc.)
│   └── state.py             # Redis/Supabase state management
├── tools/
│   ├── __init__.py
│   ├── registry.py          # ToolRegistry class
│   ├── base.py              # BaseTool abstract class
│   ├── web_fetch.py         # HTTP requests
│   ├── sql_query.py         # Database queries
│   ├── code_execute.py      # Docker sandbox execution
│   ├── file_ops.py          # file_read, file_write
│   └── shell_exec.py        # Shell commands
├── api.py                   # FastAPI endpoints (add /v1/agents/*)
├── providers.py             # LLM adapters (already done)
├── model_catalog.py         # Model registry (already done)
└── cost_optimizer.py        # Cost tracking (already done)
```

---

## Security Considerations

1. **Docker Isolation** - All code/file/shell tools run in ephemeral containers
2. **SQL Injection** - sql_query requires human approval by default
3. **Rate Limiting** - Per-API-key limits enforced at API layer
4. **Tool Permissions** - Organizations can whitelist/blacklist tools
5. **Timeout Enforcement** - All tool executions have hard timeouts

---

## Implementation Order

1. **Phase 2.1**: Create `src/agents/` directory structure with schemas
2. **Phase 2.2**: Build `AgentExecutor` with ReAct loop (no tools yet)
3. **Phase 2.3**: Implement `ToolRegistry` and all 6 tools
4. **Phase 2.4**: Add tool calling to Anthropic adapter
5. **Phase 3.0**: Add `/v1/agents/*` API endpoints
6. **Phase 3.1**: Integration testing with real LLM calls

---

## Success Criteria

- [ ] Agent can complete multi-step tasks with tool calls
- [ ] Polling API returns real-time step updates
- [ ] Docker sandbox isolates code execution
- [ ] Human approval flow works for sensitive tools
- [ ] Cost tracking accurate to within 1%
- [ ] p95 latency < 500ms for API responses (excluding LLM time)
