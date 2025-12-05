# conductor-ai

**Branch**: main | **Updated**: 2025-12-04 (Late EOD)

## Status
Phase 7.9 complete: Persona Resonance Polish + critical bug fixes. Storyboard generation now properly uses text input (Gong transcripts) and busts API caching. All 8 commits today focused on improving storyboard reliability and persona differentiation.

## Today's Focus (Next Session)
1. [ ] Test with Solar Hut Gong transcript - verify text is properly extracted
2. [ ] Verify Vercel deployment is using fresh context (no caching)
3. [ ] Demo to CEO/CTO for feedback on persona outputs
4. [ ] Consider Phase 8: Output format variations

## Done (This Session - 2025-12-04)
- **Phase 7.9: Persona Resonance Polish** - Visual styles (isometric, sketch, data_viz, bold), artist style (giger)
- **fix(demo): Prioritize text input over images** - Text/code now wins when both provided (was ignoring Gong transcripts!)
- **fix(storyboard): Add cache-busting to extraction phase** - REQUEST_ID + no-cache headers prevent API caching
- **fix(storyboard): Remove canned speech fallback** - Never fall back to brand tagline
- **fix(storyboard): Add industry guardrails** - NEVER mention beef/agriculture, stick to MEP contractors
- **fix(storyboard): Visual-first output** - Removed rigid header constraints, trust Nano Banana
- **Rolled back Phase 7.8** - Config extraction approach was overcomplicating things
- 8 commits pushed to main, all deployed to Vercel

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
- **Text input takes priority over images** (fixed today)

## Blockers
None - git clean, all tests passing, Vercel deployed

## Quick Commands
```bash
python3 -m pytest tests/tools/storyboard/ tests/knowledge/ -v  # Core tests
python demo_cli.py --example video_script_generator --audience field_crew
uvicorn src.api:app --reload --port 8001   # Start local server
open http://localhost:8001                  # Demo UI (local)
open https://conductor-ai.vercel.app        # Demo UI (production)
```

## Tech Stack
Python 3.13 | FastAPI | Supabase | Redis | Gemini 3 Pro (Nano Banana) | DeepSeek V3 | Qwen 2.5 VL 72B

## Key Files for Tomorrow
- `src/demo/router.py` - Line 343: Text now prioritized over images
- `src/tools/storyboard/gemini_client.py` - Line 525+: Cache-busting REQUEST_ID in all understand_* methods
- Test with Gong transcripts pasted into text box

## Recent Commits (Today)
- 3c64919 fix(demo): Prioritize text input over images
- 80c5dad fix(storyboard): Add cache-busting to extraction phase
- a45ce9a fix(storyboard): Remove canned speech fallback
- cc36b38 fix(storyboard): Add industry guardrails and cache-busting
- 7f1cc32 feat(storyboard): Persona Resonance Polish - Phase 7.9
- 4695e28 feat(storyboard): Visual-first output
- eba1cf9 docs: EOD update - Phase 7.8 rollback documented
- fe605e2 revert: Roll back to Phase 7.7 working state
