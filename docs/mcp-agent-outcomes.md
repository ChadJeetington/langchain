# Outcomes: LangChain + MCP Agent Architecture Design

Summary of architecture decisions and deliverables from this design session.

## Problems Addressed

| Problem | Status | Solution |
|---------|--------|----------|
| Tool failure retry loops | Solved | Structured errors with actionable guidance |
| Context truncation in long sessions | Solved | Persist to MCP resources, reload at session start |
| State location confusion | Clarified | Agent = ephemeral, MCP = persistent, AI = stateless |
| AI server integration | Architected | MCP proxies AI server calls |

## Architecture Decisions

### Decision 1: Three-Tier Architecture
```
LangChain Agent → MCP Server → AI Infrastructure Server
```
- Agent orchestrates and reasons
- MCP manages context, tools, and proxies backends
- AI server runs pure trading logic

### Decision 2: MCP as Facade
MCP server proxies all backend calls, including AI server. Agent only speaks MCP protocol.

**Rationale:**
- Single error handling layer
- Caching and rate limiting in one place
- Agent stays simple
- AI server stays reusable

### Decision 3: State Ownership
| Layer | State Type | Examples |
|-------|-----------|----------|
| Agent | Ephemeral execution state | tool_history, current_goal, working_memory |
| MCP | Persistent domain state | session context, user preferences, entities |
| AI | None (stateless) | Just inference, no conversation awareness |

### Decision 4: Structured Error Responses
All MCP tool errors return:
```json
{
  "error": true,
  "error_type": "not_found|invalid_input|rate_limited|...",
  "message": "Human-readable description",
  "suggestion": "What to do instead",
  "retryable": false
}
```

Agent adapter transforms to guidance the model can act on.

### Decision 5: Session Lifecycle
```
START:  load_session_context → get prior facts from MCP
DURING: track tool_history locally, call MCP tools
END:    save_session_context → persist summary to MCP
```

## Deliverables Created

### 1. Architecture Document
`docs/mcp-agent-architecture.md`

Contents:
- Component responsibilities
- State ownership matrix
- Connection patterns
- Error handling architecture
- Code templates

### 2. Lessons Learned Document
`docs/mcp-agent-lessons-learned.md`

Contents:
- Root cause analysis of retry loop issue
- Root cause analysis of context truncation
- State location guidance
- Anti-patterns to avoid
- Implementation checklist

### 3. Outcomes Document
`docs/mcp-agent-outcomes.md` (this file)

Contents:
- Problems addressed
- Architecture decisions
- Deliverables summary
- Next steps

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────┐
│                    ARCHITECTURE SUMMARY                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Agent ──MCP──▶ MCP Server ──HTTP──▶ AI Server                  │
│                                                                 │
│  AGENT STATE         MCP STATE            AI SERVER             │
│  (ephemeral)         (persistent)         (stateless)           │
│  ────────────        ───────────          ───────────           │
│  tool_history        session://ctx        /analyze              │
│  current_goal        portfolio://pos      /signal               │
│  working_memory      user prefs           /execute              │
│                                                                 │
│  ERROR HANDLING                                                 │
│  ───────────────                                                │
│  MCP returns: {error_type, message, suggestion, retryable}      │
│  Agent sees:  "Error (type): msg. Suggestion. [NOT] RETRYABLE"  │
│                                                                 │
│  SESSION FLOW                                                   │
│  ────────────                                                   │
│  START:  load_session_context → prior facts                     │
│  END:    save_session_context → persist summary                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Next Steps

1. **Implement error adapter** - Wrap MCP tools with structured error handling
2. **Add session tools** - `load_session_context` and `save_session_context`
3. **Configure middleware** - Set up `ToolRetryMiddleware` with correct `retry_on` exceptions
4. **Proxy AI server** - Add MCP tools that call AI infrastructure endpoints
5. **Add logging** - Capture tool inputs/outputs for debugging
6. **Test failure modes** - Verify model moves on when errors are non-retryable

## Validation Criteria

The architecture is working correctly when:

- [ ] Tool failures don't cause infinite retry loops
- [ ] Non-retryable errors result in model moving on or asking user
- [ ] Long sessions maintain context through MCP resources
- [ ] AI server calls go through MCP proxy
- [ ] Session context persists across conversations
- [ ] Agent state dies cleanly after each session
