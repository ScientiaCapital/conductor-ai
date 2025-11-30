# conductor-ai

**Branch**: main | **Updated**: 2025-11-30

## Status
Phase 2 complete: 3-way plugin integration with observability layer. 59 SDK tests + 204 core tests = 263 total. Audit/observability layer with Supabase persistence ready. dealer-scraper-mvp and sales-agent plugins implemented.

## Today's Focus
1. [ ] Run SQL migration (sql/001_audit_and_leads.sql) in Supabase
2. [ ] (Add additional tasks here)

## Done (This Session)
- (none yet)

## Critical Rules
- **NO OpenAI models** - Use Anthropic Claude, Google Gemini, or Chinese LLMs via OpenRouter
- API keys in `.env` only, never hardcoded
- All code changes require tests

## Blockers
- SQL migration pending user action (sql/001_audit_and_leads.sql)

## Quick Commands
```bash
docker-compose up -d                      # Start Redis + API services
python3 -m pytest tests/ -v               # All tests (263 total)
python3 -m pytest tests/sdk/ -v           # SDK tests only (59 tests)
pip install -e ".[sdk]"                   # Install with SDK extras
pip install -e ".[all]"                   # Install all features
```

## Tech Stack
- **Core**: FastAPI, Redis, Supabase PostgreSQL
- **AI Models**: DeepSeek V3.1, Qwen3-235B, Kimi K2, Claude Sonnet 4.5 (via OpenRouter)
- **Agent System**: ReAct loop with tool execution
- **SDK**: Plugin system for dealer-scraper-mvp and sales-agent integration
- **Observability**: AuditLogger with @audit_logged decorator

## Recent Commits
- 15d4732 feat: Add 3-way plugin integration with observability layer
- 753302b feat(sdk): Complete Plugin SDK + Chinese LLMs + Model Selector
- 8ae78f4 Merge feature/agent-system: Complete Agent System implementation (Tasks 1-9)
