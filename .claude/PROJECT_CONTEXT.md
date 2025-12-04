# conductor-ai

**Branch**: main | **Updated**: 2025-12-04

## Status
Phase 7 complete: Coperniq Knowledge Brain - intelligent learning pipeline. 712 total tests (59 SDK + 204 core + 186 video + 202 storyboard + 43 knowledge + 18 demo). Knowledge ingestion from Close CRM, Loom, Miro, and code files.

## Today's Focus
1. [ ] Run SQL migration (sql/004_coperniq_knowledge.sql) in Supabase
2. [ ] Decide next phase direction

## Done (This Session)
- Pushed Knowledge Brain commit to origin
- Updated project documentation

## Critical Rules
- **NO OpenAI models** - Use Anthropic Claude, Google Gemini, or Chinese LLMs via OpenRouter
- API keys in `.env` only, never hardcoded
- All code changes require tests

## Blockers
- SQL migrations pending (run in order: 001, 002, 003, 004)

## Quick Commands
```bash
docker-compose up -d                      # Start Redis + API services
python3 -m pytest tests/ -v               # All tests (712 total)
python knowledge_cli.py close --days 7    # Ingest from Close CRM
python knowledge_cli.py search "pain"     # Search knowledge base
python knowledge_cli.py stats             # Show knowledge stats
```

## Tech Stack
- **Core**: FastAPI, Redis, Supabase PostgreSQL
- **AI Models**: DeepSeek V3.1/R1, Qwen 2.5 VL, Gemini 2.0 Flash (via OpenRouter)
- **Agent System**: ReAct loop with tool execution
- **SDK**: Plugin system for dealer-scraper-mvp and sales-agent
- **Storyboard**: Code/Roadmap to PNG with IP sanitization
- **Knowledge Brain**: Multi-source ingestion (Close CRM, Loom, Miro, Code)

## Recent Commits
- 0b728ab feat(knowledge): Add Coperniq Knowledge Brain - intelligent learning pipeline
- 3e14a7b fix(storyboard): Add JSON repair for truncated LLM responses
- 636a19a fix(storyboard): Remove marketing language, use professional business tone
