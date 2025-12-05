# conductor-ai

**Branch**: main | **Updated**: 2025-12-04 (EOD Complete)

## Status
Phase 7.9 complete + critical bug fixes. Storyboard generation now truly visual-first:
- Text input prioritized over images (Gong transcripts work!)
- Cache-busting prevents stale API responses
- Canned marketing copy (value_framing) REMOVED - model uses extracted content only

## Tomorrow Start Here
1. **Test Solar Hut Gong transcript** - Verify text is properly extracted and NO canned copy appears
2. **Hard refresh Vercel** (Cmd+Shift+R) - Clear browser cache after deployment
3. Demo to CEO/CTO for feedback
4. Consider Phase 8: Output format variations

## Done (This Session - 2025-12-04)
- **fix: Remove canned value_framing** - THE ROOT CAUSE of repetitive output
- **fix: Text input priority over images** - Gong transcripts were being ignored
- **fix: Cache-busting to extraction phase** - REQUEST_ID prevents API caching
- **fix: Remove canned speech fallback** - Never fall back to brand tagline
- **fix: Industry guardrails** - NEVER mention beef/agriculture
- **Phase 7.9: Persona Resonance Polish** - Visual styles + artist styles
- **Rolled back Phase 7.8** - Config extraction was overcomplicating things
- 9 commits pushed to main

## Key Insight
The `value_framing` field in persona presets contained hardcoded phrases like:
- "Every day without this = money walking out the door"
- "Consolidate tools, automate payments..."

These were injected via `FRAMING: {value_framing}` into EVERY prompt, overriding whatever the user pasted. Now removed - Nano Banana will use only extracted content.

## Quick Commands
```bash
python3 -m pytest tests/tools/storyboard/ tests/knowledge/ -v  # Core tests
python demo_cli.py --example video_script_generator --audience field_crew
uvicorn src.api:app --reload --port 8001   # Local server
open https://conductor-ai.vercel.app        # Production (wait for deploy)
```

## Tech Stack
Python 3.13 | FastAPI | Supabase | Redis | Gemini 3 Pro (Nano Banana) | DeepSeek V3 | Qwen 2.5 VL 72B

## Key Files Modified Today
- `src/demo/router.py:343` - Text now prioritized over images
- `src/tools/storyboard/gemini_client.py:525+` - Cache-busting REQUEST_ID
- `src/tools/storyboard/gemini_client.py:1430+` - REMOVED value_framing injection

## Recent Commits (Today - 9 total)
- bdfce51 fix(storyboard): Remove canned value_framing from all persona contexts
- 3c64919 fix(demo): Prioritize text input over images
- 80c5dad fix(storyboard): Add cache-busting to extraction phase
- a45ce9a fix(storyboard): Remove canned speech fallback
- cc36b38 fix(storyboard): Add industry guardrails and cache-busting
- 7f1cc32 feat(storyboard): Persona Resonance Polish - Phase 7.9
- 4695e28 feat(storyboard): Visual-first output
- eba1cf9 docs: EOD update - Phase 7.8 rollback documented
- fe605e2 revert: Roll back to Phase 7.7 working state

## Blockers
None - git clean, all pushed, Vercel deploying
