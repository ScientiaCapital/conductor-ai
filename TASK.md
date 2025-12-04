# TASK.md - Conductor-AI

**Project**: conductor-ai
**Last Updated**: 2025-12-04 (Late EOD)

---

## Current Status

**Phase 7.8: Storyboard Refactoring + Video Integration - COMPLETE**

- ✅ Extracted `storyboard_config.py` (271 lines) from gemini_client.py
- ✅ Refactored gemini_client.py: 1561 → ~1145 lines (-27%)
- ✅ Added Kurt Geiger artist style (for user's wife)
- ✅ Fixed "canned response" issue - removed hardcoded taglines
- ✅ Added Replicate video provider (Luma Ray Flash 2)
- ✅ Video format routing in demo UI (video_horizontal, video_vertical)
- ✅ Random creative hooks in transcript extraction for uniqueness

**Total Tests**: 730 (59 SDK + 204 core + 186 video + 202 storyboard + 58 knowledge + 21 demo)

**SQL Migrations**: All 4 migrations completed in Supabase

---

## Active Tasks

### 🔴 HIGH Priority

**Video API Keys Needed** - Video generation wired up but needs:
- `REPLICATE_API_TOKEN` in .env for video generation

---

### 🟡 MEDIUM Priority

**Ingest Real Data** - Knowledge Brain is connected but needs actual content:
- **Loom transcripts** (~20-25 videos, manual export → batch ingest)
- Close CRM calls/notes (pain points, metrics, quotes)
- Miro screenshots (roadmap features)

**Series A Prep** (Target: May/June 2025)
- Test storyboard generation with real knowledge data
- CEO/CTO review of investor-focused outputs

---

### 🟢 LOW Priority / Nice-to-Have

**None currently**

---

## Next Phase Options (Awaiting User Decision)

### Option A: Real Data Ingestion Pipeline
Populate the knowledge base with actual data:
- Close CRM calls/notes from last 30 days
- Loom transcripts from product demos
- Miro roadmap screenshots
- Engineer code for feature names

### Option B: Phase 8 - Advanced Features
- Streaming responses (Server-Sent Events)
- Multi-agent collaboration
- Memory system (conversation history)
- Cost tracking per session

### Option C: Plugin Integrations (Tier 1)
1. **dealer-scraper-mvp** - Fix 16 broken scrapers with web_fetch tool
2. **sales-agent** - LinkedIn scraping + lead enrichment

---

## Completed Phases

### Phase 7.8: Storyboard Refactoring + Video (2025-12-04 Late)
- ✅ Extracted `storyboard_config.py` from gemini_client.py
- ✅ Refactored gemini_client.py (1561 → 1145 lines, -27%)
- ✅ Added Kurt Geiger artist style
- ✅ Fixed hardcoded tagline issue ("canned responses")
- ✅ Added Replicate video provider (Luma Ray Flash 2)
- ✅ Video format routing in demo router
- ✅ Random creative hooks for transcript uniqueness

### Phase 7.7: Persona Differentiation (2025-12-04)
- ✅ COI (Cost of Inaction) framing for: business_owner, btl_champion
- ✅ ROI (Return on Investment) framing for: c_suite, top_tier_vc
- ✅ EASE framing for: field_crew
- ✅ Value angle injection in all understand_* methods

### Phase 7.6: Intelligent Model Routing (2025-12-04)
- ✅ 3-Stage Pipeline architecture (EXTRACT → REFINE → GENERATE)
- ✅ DeepSeek V3 (`deepseek/deepseek-chat`) for text/code extraction
- ✅ Qwen 2.5 VL 72B (`qwen/qwen2.5-vl-72b-instruct`) for vision/OCR
- ✅ Gemini 3 Pro Image Preview for final PNG generation (FREE)
- ✅ Confidence-based refinement (< 0.75 threshold triggers alternate model)
- ✅ Fixed R1 hang issue (60-120s → 3-5s with V3)
- ✅ 202 storyboard tests passing

### Phase 7.5: Knowledge → Storyboard Integration (2025-12-04)
- ✅ KnowledgeCache singleton (`src/knowledge/cache.py`)
- ✅ Startup preload in FastAPI lifespan
- ✅ Language guidelines enriched with banned/approved terms
- ✅ Knowledge context injection into prompts
- ✅ 15 new cache tests (58 total knowledge tests)
- ✅ Graceful degradation (storyboards work even if knowledge fails)

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
   - A) Ingest real data from Close CRM / Loom / Miro
   - B) Advanced features (streaming, multi-agent, memory)
   - C) Plugin integrations

---

## Notes

- All code follows project standards (see PLANNING.md)
- All tests passing (727 total)
- No OpenAI models used (DeepSeek/Qwen/Gemini only)
- API keys in .env only (never hardcoded)
- Knowledge Brain connected to storyboard generation

---

**Phase 7.5 complete - Knowledge → Storyboard integration live. Awaiting user direction for Phase 8.**
