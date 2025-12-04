# conductor-ai

**Branch**: main | **Updated**: 2025-12-04 (EOD)

## Status
Phase 7.7 complete: Persona Differentiation with COI/ROI/EASE value angles. Each audience (Business Owner, C-Suite, BTL Champion, VC, Field Crew) now produces distinctly different output through value angle framing at both extraction and generation phases.

## Today's Focus (Next Session)
1. [ ] Test persona differentiation with real transcripts/images
2. [ ] Demo to CEO/CTO for feedback on persona outputs
3. [ ] Consider Phase 8: Output format variations (infographic vs storyboard aspect ratios)
4. [ ] Loom transcript ingestion (~20-25 videos manual export)

## Done (This Session - Late 2025-12-04)
- Phase 7.7: Persona Differentiation COMPLETE
- Added `_build_persona_generation_context()` - rich persona data injection
- Added `_get_value_angle_instruction()` - COI/ROI/EASE framing per audience
- Updated all `understand_*` methods with value angle awareness
- Fixed "Not mentioned in transcript" - must now INFER
- Fixed personal names in output - generalize to roles/personas
- Deleted unused `src/tools/scientia/` module (1264 lines cleanup)
- 260 storyboard+knowledge tests passing
- 2 commits pushed: feat(storyboard) + docs(EOD)

## Value Angle System
| Audience | Value Angle | Framing |
|----------|-------------|---------|
| business_owner | COI | What they LOSE by not acting |
| c_suite | ROI | What they GAIN from this |
| btl_champion | COI | Risk of NOT having this |
| top_tier_vc | ROI | Investment opportunity |
| field_crew | EASE | Making job simpler |

## Critical Rules
- **NO OpenAI models** - Use Anthropic Claude, Google Gemini, or Chinese LLMs via OpenRouter
- API keys in `.env` only, never hardcoded
- All code changes require tests
- **NEVER** output "Not mentioned in transcript" - must INFER
- **NEVER** use personal names - generalize to roles/personas

## Blockers
None - git clean, all tests passing

## Quick Commands
```bash
python3 -m pytest tests/tools/storyboard/ tests/knowledge/ -v  # Core tests (260)
python demo_cli.py --example video_script_generator --audience field_crew
uvicorn src.api:app --reload               # Start local server
open http://localhost:8000                 # Demo UI
```

## Tech Stack
Python 3.13 | FastAPI | Supabase | Redis | Gemini 3 Pro (Nano Banana) | DeepSeek V3 | Qwen 2.5 VL 72B

## Key Files for Tomorrow
- `src/tools/storyboard/gemini_client.py` - Line 1303: `_build_persona_generation_context()`, Line 1225: `_get_value_angle_instruction()`
- Test with different audience values to verify distinct outputs

## Recent Commits
- c5dbc4b docs: EOD update - Phase 7.7 Persona Differentiation complete
- 2479049 feat(storyboard): Phase 2 - Persona differentiation with COI/ROI/EASE framing
- 9892b1f docs: EOD update - Phase 7.6 Intelligent Model Routing complete
- 0e58f8e fix(storyboard): Use DeepSeek V3 (fast) instead of R1 (slow reasoning)
