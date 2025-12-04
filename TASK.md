# TASK.md - Conductor-AI

**Project**: conductor-ai
**Last Updated**: 2025-12-04

---

## Current Status

**Phase 7: Coperniq Knowledge Brain - COMPLETE**

- ✅ Knowledge Base Schema (`sql/004_coperniq_knowledge.sql`)
- ✅ Knowledge Module (`src/knowledge/`) - base, extraction, close_crm, service
- ✅ Knowledge CLI (`knowledge_cli.py`) - close, loom, miro, code, search, stats
- ✅ 43 new tests for knowledge module
- ✅ Pre-seeded 22 banned terms + 6 approved terms

**Total Tests**: 712 (59 SDK + 204 core + 186 video + 202 storyboard + 43 knowledge + 18 demo)

---

## Active Tasks

### 🔴 HIGH Priority

**SQL Migration Required**
- Run `sql/004_coperniq_knowledge.sql` in Supabase SQL Editor
- Also verify 001, 002, 003 migrations have been run

---

### 🟡 MEDIUM Priority

**None currently** - Awaiting user direction for Phase 8

---

### 🟢 LOW Priority / Nice-to-Have

**None currently**

---

## Pending User Action

### SQL Migrations Required
**Files** (run in order):
1. `sql/001_audit_and_leads.sql` - audit_logs, tool_executions, leads
2. `sql/002_storyboard_jobs.sql` - storyboard job state persistence
3. `sql/003_storage_buckets.sql` - storage buckets for images
4. `sql/004_coperniq_knowledge.sql` - knowledge brain schema

**Action**: Run in Supabase SQL Editor

---

## Next Phase Options (Awaiting User Decision)

### Option A: Knowledge Integration into Storyboards
Connect the Knowledge Brain to storyboard generation:
- Retrieve approved/banned terms during generation
- Use extracted pain points and quotes for content
- Dynamic storyboard content based on knowledge

### Option B: Real Data Ingestion Pipeline
Populate the knowledge base with actual data:
- Close CRM calls/notes from last 30 days
- Loom transcripts from product demos
- Miro roadmap screenshots
- Engineer code for feature names

### Option C: Phase 8 - Advanced Features
- Streaming responses (Server-Sent Events)
- Multi-agent collaboration
- Memory system (conversation history)
- Cost tracking per session

### Option D: Plugin Integrations (Tier 1)
1. **dealer-scraper-mvp** - Fix 16 broken scrapers with web_fetch tool
2. **sales-agent** - LinkedIn scraping + lead enrichment

---

## Completed Phases

### Phase 7: Knowledge Brain (2025-12-04)
- ✅ Knowledge schema with full-text search
- ✅ Multi-source ingestion (Close CRM, Loom, Miro, Code)
- ✅ LLM extraction via DeepSeek V3.2
- ✅ CLI for ingestion and search
- ✅ 43 tests

### Phase 6: Demo App (2025-12-04)
- ✅ FastAPI demo router
- ✅ CLI with 10 example commands
- ✅ Web UI with drag/drop, paste
- ✅ Vercel deployment

### Phase 5: Storyboard Pipeline API (2025-12-02)
- ✅ Async FastAPI endpoints
- ✅ Redis hot state + Supabase cold persistence
- ✅ Job polling and status

### Phase 4: Storyboard Tools (2025-12-02)
- ✅ CodeToStoryboardTool
- ✅ RoadmapToStoryboardTool
- ✅ UnifiedStoryboardTool
- ✅ 5 audiences (business_owner, c_suite, btl_champion, top_tier_vc, field_crew)

### Phase 3: Video Tools (2025-12-02)
- ✅ VideoScriptGeneratorTool
- ✅ VideoGeneratorTool (multi-provider)
- ✅ LoomViewTrackerTool
- ✅ VideoSchedulerTool

### Phase 2: Plugin Integration (2025-11-28)
- ✅ SDK Foundation (5-import API)
- ✅ Audit/Observability Layer
- ✅ dealer-scraper-mvp plugin
- ✅ sales-agent plugin

### Phase 1: SDK Foundation (2025-11-27)
- ✅ BaseTool, ToolDefinition, ToolResult
- ✅ PluginRegistry, PluginLoader
- ✅ HTTP Client (ConductorClient)
- ✅ Security utilities (SSRF, timeout, approval)

---

## Technical Debt

**None currently** - All code follows modern Python best practices

---

## Questions for User

1. **Which Phase 8 direction?**
   - A) Integrate knowledge into storyboards
   - B) Ingest real data from Close CRM
   - C) Advanced features (streaming, multi-agent, memory)
   - D) Plugin integrations

---

## Notes

- All code follows project standards (see PLANNING.md)
- All tests passing (712 total)
- No OpenAI models used (DeepSeek/Qwen/Gemini only)
- API keys in .env only (never hardcoded)

---

**Phase 7 complete - awaiting user direction for Phase 8**
