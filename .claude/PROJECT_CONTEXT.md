# conductor-ai

**Branch**: main | **Updated**: 2025-12-04 (EOD)

## Status
Phase 7.6 complete: Intelligent 3-stage model routing (EXTRACT → REFINE → GENERATE). DeepSeek V3 for text, Qwen VL 72B for vision, Gemini 3 Pro for image gen. 727 total tests passing. Knowledge Brain integrated with storyboard generation.

## Today's Focus
1. [x] Model research and upgrades (DeepSeek V3, Qwen VL 72B)
2. [x] Implement 3-stage intelligent routing pipeline
3. [x] Fix R1 hang issue (reverted to V3)
4. [x] VC storyboard polish (creative freedom)

## Done (This Session)
- 10 commits pushed to origin/main
- Phase 7.6: Intelligent Model Routing complete
- 3-Stage Pipeline: EXTRACT → REFINE → GENERATE
- DeepSeek V3 for text extraction (fast, 3-5s)
- Qwen 2.5 VL 72B for vision/OCR
- Confidence-based refinement (<0.75 triggers alternate model)
- Fixed R1 hang (was 60-120s, now 3-5s with V3)
- VC storyboard polish - removed rigid template

## Next Session
1. [ ] Decide Phase 8 direction (A: Real Data, B: Advanced Features, C: Plugin Integrations)
2. [ ] Series A prep (May/June 2025) - test with real knowledge data
3. [ ] Loom transcript ingestion (~20-25 videos manual export)

## Critical Rules
- **NO OpenAI models** - Use Anthropic Claude, Google Gemini, or Chinese LLMs via OpenRouter
- API keys in `.env` only, never hardcoded
- All code changes require tests

## Blockers
None - all SQL migrations complete, git clean

## Quick Commands
```bash
docker-compose up -d                      # Start Redis + API services
python3 -m pytest tests/ -v               # All tests (727 total)
python knowledge_cli.py close --days 7    # Ingest from Close CRM
python knowledge_cli.py search "pain"     # Search knowledge base
python demo_cli.py --example video_script_generator --audience field_crew
```

## Tech Stack
- **Core**: FastAPI, Redis, Supabase PostgreSQL
- **AI Models**: DeepSeek V3 (text), Qwen 2.5 VL 72B (vision), Gemini 3 Pro (image gen)
- **Pipeline**: 3-stage EXTRACT → REFINE → GENERATE with confidence-based routing
- **SDK**: Plugin system for dealer-scraper-mvp and sales-agent
- **Storyboard**: Code/Roadmap to PNG with IP sanitization
- **Knowledge Brain**: Multi-source ingestion (Close CRM, Loom, Miro, Code)

## Recent Commits (Today)
- 0e58f8e fix(storyboard): Use DeepSeek V3 (fast) instead of R1 (slow reasoning)
- dcd1ac4 feat(storyboard): Intelligent 3-stage model routing pipeline
- 224e688 feat(storyboard): Add ROI vs COI value angle per persona
- 5f8add8 docs: EOD update - VC polish complete, Series A prep noted
- a1da14a refactor(storyboard): Remove rigid VC template, enable creative freedom
