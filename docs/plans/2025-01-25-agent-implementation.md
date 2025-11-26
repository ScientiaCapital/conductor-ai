# Agent Orchestration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a ReAct-pattern agent orchestration system with tool calling, async polling, and Docker sandboxing.

**Architecture:** FastAPI async executor with Pydantic-validated tool calls. Redis for hot state, Supabase for persistence.

**Tech Stack:** FastAPI, Pydantic v2, Redis, Supabase, Docker SDK, Claude 4.5 Sonnet/Opus

---

## Task Overview

| Task | Component | Files | Test |
|------|-----------|-------|------|
| 1 | Agent Schemas | `src/agents/schemas.py` | `tests/agents/test_schemas.py` |
| 2 | Tool Base Classes | `src/tools/base.py`, `registry.py` | `tests/tools/test_registry.py` |
| 3 | web_fetch Tool | `src/tools/web_fetch.py` | `tests/tools/test_web_fetch.py` |
| 4 | code_run Tool | `src/tools/code_run.py` | `tests/tools/test_code_run.py` |
| 5 | sql_query Tool | `src/tools/sql_query.py` | `tests/tools/test_sql_query.py` |
| 6 | State Manager | `src/agents/state.py` | `tests/agents/test_state.py` |
| 7 | AgentRunner | `src/agents/runner.py` | `tests/agents/test_runner.py` |
| 8 | API Endpoints | `src/api.py` | `tests/test_api.py` |
| 9 | Integration | - | `tests/integration/test_full.py` |

---

## Task 1: Agent Schemas

**Create:** `src/agents/schemas.py`

Pydantic models for:
- `SessionStatus` (enum: pending, running, completed, failed, cancelled)
- `ToolCall` (tool name + args dict)
- `AgentStep` (thought, action, observation, is_final, final_answer)
- `AgentSession` (session_id, org_id, status, steps, totals)
- `AgentRunRequest` (messages, model, tools, max_steps)
- `AgentRunResponse` (session_id, status, poll_url)

**Test:** Validate schema creation and serialization

---

## Task 2: Tool Base Classes

**Create:** `src/tools/base.py`
- `ToolDefinition` - name, description, parameters (JSON Schema), category, requires_approval
- `ToolResult` - tool_name, success, result, error, execution_time_ms
- `BaseTool` - abstract class with `definition` property and `run()` method

**Create:** `src/tools/registry.py`
- `ToolRegistry` - register, get, list_tools, get_tools_for_llm

---

## Task 3: web_fetch Tool

**Create:** `src/tools/web_fetch.py`

HTTP GET/POST requests using aiohttp:
- URL validation
- Timeout handling (30s default)
- Response truncation (50KB max)
- Error handling

**No approval required.**

---

## Task 4: code_run Tool

**Create:** `src/tools/code_run.py`

Docker-sandboxed code runner:
- Python 3.11 and Node.js 20 support
- Memory limit: 128MB
- CPU limit: 50%
- Network disabled
- 30s timeout
- Fallback to subprocess if Docker unavailable

**No approval required** (sandboxed).

---

## Task 5: sql_query Tool

**Create:** `src/tools/sql_query.py`

Supabase SQL queries:
- Dangerous pattern detection (DROP, TRUNCATE, DELETE without WHERE)
- Parameterized queries
- Result size limits

**Requires approval** for all queries.

---

## Task 6: State Manager

**Create:** `src/agents/state.py`

Redis (hot) + Supabase (cold):
- `create_session()` - new session in Redis
- `get_session()` - Redis first, fallback Supabase
- `update_session()` - update Redis
- `add_step()` - append to Redis list
- `get_steps()` - get all steps
- `persist_to_supabase()` - batch write on completion

Redis TTL: 1 hour

---

## Task 7: AgentRunner

**Create:** `src/agents/runner.py`

ReAct loop implementation:
1. Build system prompt with tool definitions
2. Call LLM with conversation history
3. Parse JSON response (thought/action/is_final)
4. If is_final: return final_answer
5. If action: run tool, add observation
6. Repeat until max_steps

JSON enforcement via Pydantic validation.

---

## Task 8: API Endpoints

**Modify:** `src/api.py`

Endpoints:
- `POST /v1/agents/run` - Start session, return session_id
- `GET /v1/agents/{id}` - Poll status and steps
- `POST /v1/agents/{id}/cancel` - Cancel running session
- `GET /v1/tools` - List available tools

Background task execution for async polling model.

---

## Task 9: Integration Test

End-to-end test:
1. POST /v1/agents/run with simple question
2. Poll GET /v1/agents/{id} until completed
3. Verify final_answer contains expected content

Requires: ANTHROPIC_API_KEY, Redis running

---

## Execution Order

```
Task 1 (schemas) ──┐
                   ├──► Task 6 (state) ──┐
Task 2 (base) ─────┤                     │
                   │                     ├──► Task 7 (runner) ──► Task 8 (api) ──► Task 9 (test)
Task 3 (web) ──────┤                     │
Task 4 (code) ─────┼─────────────────────┘
Task 5 (sql) ──────┘
```

Tasks 1-5 can run in parallel. Task 6 depends on 1. Task 7 depends on 2,6. Task 8 depends on all.

---

## Success Criteria

- [ ] All unit tests pass
- [ ] Agent completes multi-step task with tool calls
- [ ] Polling API returns real-time updates
- [ ] Docker sandbox isolates code
- [ ] Cost tracking accurate
- [ ] p95 API latency < 500ms (excluding LLM)
