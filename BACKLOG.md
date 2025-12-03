# BACKLOG.md - Project Task Board

**Project**: Conductor-AI - Multi-Model AI Agent Platform with SDK
**Last Updated**: 2025-12-02
**Sprint**: Current

---

## Quick Stats

| Status | Count |
|--------|-------|
| 🔴 Blocked | 2 |
| 🟡 In Progress | 0 |
| 🟢 Ready | 1 |
| ✅ Done (this sprint) | 9 |

---

## 📋 Board View

### 🔴 Blocked
<!-- Tasks waiting on external dependencies or decisions -->

#### 1. [MEDIUM] SQL Migration for Observability Layer
- **ID**: TASK-001
- **Assignee**: User
- **Labels**: `database`, `migration`, `user-action`
- **Est. Time**: 15 minutes
- **Dependencies**: None

**Description**: Run SQL migration file `sql/001_audit_and_leads.sql` in Supabase SQL Editor to create required tables for observability layer.

**Blocker**: Requires user to manually run migration in Supabase dashboard.

**Tables to Create**:
- `audit_logs` - Tool execution audit trail
- `tool_executions` - Detailed I/O records
- `leads` - Shared lead storage

**Acceptance Criteria**:
- [ ] SQL migration executed in Supabase
- [ ] Tables created and accessible
- [ ] RLS policies applied

---

#### 2. [MEDIUM] SQL Migration for Storyboard Jobs
- **ID**: TASK-020
- **Assignee**: User
- **Labels**: `database`, `migration`, `user-action`
- **Est. Time**: 10 minutes
- **Dependencies**: None

**Description**: Run SQL migration file `sql/002_storyboard_jobs.sql` in Supabase SQL Editor to enable storyboard job persistence.

**Blocker**: Requires user to manually run migration in Supabase dashboard.

**Tables to Create**:
- `storyboard_jobs` - Job state with RLS policies

**Acceptance Criteria**:
- [ ] SQL migration executed in Supabase
- [ ] Table created and accessible
- [ ] RLS policies applied

---

### 🟡 In Progress
<!-- Tasks actively being worked on - LIMIT: 2 tasks max -->

*None currently - Awaiting user decision on Phase 3 direction*

---

### 🟢 Ready (Prioritized)
<!-- Tasks ready to start, ordered by priority -->

#### 1. [HIGH] Add GOOGLE_API_KEY and Test Storyboard API
- **ID**: TASK-021
- **Assignee**: User
- **Labels**: `setup`, `testing`, `api`
- **Est. Time**: 15 minutes
- **Dependencies**: Redis running (✅ done), SQL migrations

**Description**: Add Google API key to `.env` and test the storyboard API with real images.

**Steps**:
1. Get Gemini API key from https://ai.google.dev/
2. Add to `.env`: `GOOGLE_API_KEY=AIza...`
3. Start API: `python -m uvicorn src.api:app --reload`
4. Test with curl or image upload

**Acceptance Criteria**:
- [ ] GOOGLE_API_KEY set in .env
- [ ] API server starts without errors
- [ ] POST /storyboard/code returns 202
- [ ] GET /storyboard/jobs/{id} returns job status

---

### ⏸️ Backlog (Future)
<!-- Tasks not yet prioritized for this sprint -->

#### Phase 3 Option 1: Advanced Features
| ID | Title | Priority | Labels | Est. |
|----|-------|----------|--------|------|
| TASK-002 | Streaming Responses (SSE) | High | `feature`, `streaming` | 3 days |
| TASK-003 | Multi-Agent Collaboration | High | `feature`, `agents` | 4 days |
| TASK-004 | Memory System (Conversation History) | Medium | `feature`, `memory` | 3 days |
| TASK-005 | Cost Tracking Per Session | Medium | `feature`, `analytics` | 2 days |

#### Phase 3 Option 2: Plugin Integrations (Tier 1)
| ID | Title | Priority | Labels | Est. |
|----|-------|----------|--------|------|
| TASK-006 | dealer-scraper-mvp Integration (Fix 16 scrapers) | High | `plugin`, `integration`, `high-value` | 1 week |
| TASK-007 | sales-agent LinkedIn Scraping Integration | High | `plugin`, `integration`, `high-value` | 1 week |

#### Phase 3 Option 3: Dashboard & Monitoring
| ID | Title | Priority | Labels | Est. |
|----|-------|----------|--------|------|
| TASK-008 | Streamlit Dashboard for Session Monitoring | Medium | `dashboard`, `monitoring` | 4 days |
| TASK-009 | Cost Tracking Per Org | Medium | `analytics`, `dashboard` | 2 days |
| TASK-010 | Tool Usage Analytics | Low | `analytics`, `dashboard` | 2 days |
| TASK-011 | Error Rate Monitoring | Low | `monitoring`, `dashboard` | 2 days |

---

### ✅ Done (This Sprint)
<!-- Completed tasks - move here when done -->

| ID | Title | Completed | By |
|----|-------|-----------|-----|
| TASK-000 | Context Engineering Setup | 2025-11-30 | Claude |
| TASK-012 | SDK Foundation (59 tests, 5-import API) | 2025-11-27 | Team |
| TASK-013 | Audit/Observability Layer | 2025-11-28 | Team |
| TASK-014 | dealer-scraper-mvp Plugin (3 tools) | 2025-11-28 | Team |
| TASK-015 | sales-agent Plugin (3 tools) | 2025-11-28 | Team |
| TASK-016 | Model Catalog (DeepSeek, Qwen, Kimi, Claude) | 2025-11-29 | Team |
| TASK-017 | Video Tools Module (7 tools, 186 tests) | 2025-12-02 | Claude |
| TASK-018 | Storyboard Tools (2 tools, 168 tests) - Code/Roadmap to PNG | 2025-12-02 | Claude |
| TASK-019 | Storyboard Pipeline API (3 endpoints, 61 tests) - Async job processing | 2025-12-02 | Claude |

---

## 📊 Sprint Metrics

### Velocity
- **Last Sprint**: 6 tasks completed
- **This Sprint Target**: TBD (awaiting direction)
- **Avg Task Time**: 2-4 days

### Quality
- **Tests Passing**: ✅ (678 total: 59 SDK + 204 core + 186 video + 229 storyboard)
- **Type Errors**: 0
- **Lint Issues**: 0
- **NO OpenAI**: ✅ (Using DeepSeek, Qwen, Claude, Gemini)

---

## 🔄 Workflow

### Task Lifecycle
```
Ready → In Progress → Review → Done
         ↓
       Blocked (if dependencies)
```

### How to Update This File

**Starting a task**:
1. Move task from "Ready" to "In Progress"
2. Add your name as Assignee
3. Update the date in header

**Completing a task**:
1. ✅ Check all acceptance criteria boxes
2. Move entry to "Done" section
3. Add completion date and your name

**Adding a new task**:
1. Add to "Backlog" table with ID, title, priority
2. When prioritized, create full entry in "Ready"
3. Must include: ID, description, acceptance criteria

**Task ID Format**: `TASK-XXX` (increment from last ID)

---

## 🚨 Blockers & Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| SQL migration not run | High | User must execute migration in Supabase |
| Unclear Phase 3 priorities | Medium | User decision needed on next direction |
| Plugin integration scope creep | Medium | Define clear boundaries for Tier 1 integrations |

---

## 📝 Sprint Notes

### Decisions Made
- 2025-12-02: Storyboard Pipeline API complete - 3 endpoints, 61 tests, async job processing
- 2025-12-02: Storyboard tools module complete - 2 tools, 168 tests, Gemini Vision + Image Gen
- 2025-12-02: Video tools module merged - 7 tools, 186 tests, SSRF protected
- 2025-11-30: Context engineering deployed
- 2025-11-29: Phase 2 complete - 3-way plugin integration successful
- 2025-11-27: SDK foundation established with 5-import public API

### Questions to Resolve
1. **Which Phase 3 feature is highest priority?**
   - Streaming responses?
   - Multi-agent collaboration?
   - Memory system?
   - Cost tracking?

2. **Should we prioritize plugin integrations over Phase 3 features?**
   - dealer-scraper-mvp integration would fix 16 broken scrapers immediately
   - sales-agent integration would unlock LinkedIn scraping

3. **Do you want a dashboard/monitoring UI?**
   - Streamlit app for session monitoring
   - Cost tracking per org
   - Tool usage analytics

### Learnings
- Phase 2 completed successfully with 263 tests passing
- SDK design pattern working well (5-import simplicity)
- Audit/observability layer provides good foundation for monitoring

---

## 🔗 Related Files

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Project overview and rules |
| `PLANNING.md` | Architecture decisions |
| `TASK.md` | Quick task reference |
| `PRPs/` | Implementation plans |
| `/validate` | Run before completing tasks |
| `sql/001_audit_and_leads.sql` | Database migration file |
| `sql/002_storyboard_jobs.sql` | Storyboard jobs migration |
| `.env.example` | Environment template |

---

## Critical Rules Reminder

- **NO OpenAI** - Use Anthropic Claude, Google Gemini, or Chinese LLMs via OpenRouter
- **API keys in .env only** - Never hardcode
- **All code changes require tests** - Maintain >90% coverage
- **Run `/validate` before marking tasks done**
- **Update this file as work progresses**

---

*This file is the source of truth for sprint tasks. Keep it updated!*
