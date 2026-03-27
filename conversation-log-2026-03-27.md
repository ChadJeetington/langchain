# Conversation Log: LangChain + MCP Agent Architecture Design

**Date:** 2026-03-27
**Topic:** Production architecture for LangChain ReAct agents with MCP tool servers
**Outcome:** Architecture design, documentation, and production guidance

---

## Initial Problem Statement

**User:**
> I've got a LangChain ReAct agent in production using an MCP tool server over streamable HTTP. When a tool call fails or returns something the model can't parse, it just retries in a loop. I've bumped max_iterations and tweaked the system prompt but it keeps happening. This is a serious issue for my team.
>
> The other thing is bad for our performance on longer sessions is that early tool outputs get truncated and then the model just forgets context it still needs.
>
> I'm trying to figure out what the right architecture actually looks like here. How do you handle tool failures that aren't just a simple retry? Also, where should persistent state live. Should it live with the langchain agent or should it live on the MCP server where the tool calls and resources are?

---

## Diagnostic Questions

**Claude asked for clarification:**

1. How are you exposing MCP tools to LangChain? (custom adapter, BaseTool, LanGraph ToolNode)
2. What do your tool failures look like? (HTTP errors, schema validation, semantic failures)
3. Current memory setup? (trim_intermediate_steps, memory class, checkpointing)
4. Session length? (how many tool calls before truncation)

**User response:**
> I am using the most recent langchain. Right now it is custom tools on an Anthropic MCP server. They are structured json. I am not sure on how to check the errors specifics yet.

---

## Key Findings from Codebase Exploration

LangChain has two distinct retry mechanisms:

1. **ToolRetryMiddleware** - automatic retry with backoff for transient failures (network, rate limits)
2. **Agent-level "retry"** - model sees confusing response and decides to call tool again

The retry loop issue was likely the model choosing to re-invoke tools because error responses didn't give actionable guidance.

Context truncation happens via `trim_intermediate_steps` (by count, not content length) and XML formatting truncation (500 char limit).

---

## Solution: Structured Error Responses

### Pattern A: Tool returns JSON the model can't act on

**Problem:** MCP returns `{"status": "error", "code": "RATE_LIMITED"}` - model doesn't know what to do.

**Solution:** Transform MCP responses into actionable observations:

```python
class MCPToolAdapter(BaseTool):
    def _run(self, **kwargs) -> str:
        result = self.mcp_client.call_tool(self.name, kwargs)

        if result.get("error"):
            error_type = result.get("error_type", "unknown")

            if error_type == "not_found":
                return f"Resource not found. Available options: {result.get('suggestions')}. Try a different query or ask the user for clarification."

            if error_type == "rate_limited":
                return "This service is temporarily unavailable. Move on to other tasks or inform the user."

            # Non-retryable: give explicit stop signal
            return f"Tool failed permanently: {result.get('message')}. Do not retry. Inform the user of the failure."

        return self._format_success(result)
```

---

## State Ownership Question

**User asked:**
> Should some state be with the langchain agent directly? Yes or no with examples.

**Answer: Yes** — some state belongs with the agent.

### Agent-side state (ephemeral):
```python
agent_state = {
    "tool_history": [...],      # What tools were called this session
    "current_goal": "...",      # Active objective
    "attempted_approaches": [], # What's been tried
    "working_memory": {...},    # Scratchpad
}
```

### MCP-side state (persistent):
```python
# Resource: session://abc123
{
    "user_id": "user_456",
    "preferences": {"timezone": "PST"},
    "conversation_summary": "User asked about billing...",
    "facts_learned": [...]
}
```

**Key insight:** Agent state dies with session. MCP state persists across sessions.

---

## Architecture Diagram

**User requested:** "Give me a simple diagram in this terminal to help me visualize."

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         THREE-TIER ARCHITECTURE                              │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────┐
                    │        LANGCHAIN AGENT          │
                    │         (Orchestrator)          │
                    ├─────────────────────────────────┤
                    │  • Interprets user intent       │
                    │  • Selects tools/actions        │
                    │  • Manages conversation flow    │
                    │  • Error recovery & retry       │
                    └───────────────┬─────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
   ┌────────────────────────────┐    ┌────────────────────────────┐
   │        MCP SERVER          │    │    AI INFRASTRUCTURE       │
   │    (Context & Tools)       │    │       SERVER               │
   ├────────────────────────────┤    │    (Trading Logic)         │
   │  Resources:                │    ├────────────────────────────┤
   │  • session://context       │    │  Endpoints:                │
   │  • portfolio://positions   │    │  • /analyze                │
   │                            │    │  • /signal                 │
   │  Tools:                    │    │  • /execute                │
   │  • get_portfolio           │    │                            │
   │  • save_session            │    │  Models:                   │
   │                            │    │  • Price prediction        │
   └────────────────────────────┘    └────────────────────────────┘
```

---

## Connection Pattern Decision

**User asked:** "Help me architect the connection between the agent, mcp server, and then a related AI server."

**Clarification:** AI server = AI infrastructure server where agentic trading logic lives, separate from MCP.

### Recommended: Option A - MCP Proxies AI Server

```
Agent ──MCP──▶ MCP Server ──HTTP──▶ AI Server
```

**Benefits:**
- Agent only speaks MCP protocol
- Unified error handling in one place
- MCP can cache, rate-limit, audit
- AI server stays pure and reusable

### MCP Proxy Pattern:

```python
@server.tool()
def analyze_position(symbol: str, timeframe: str) -> dict:
    """Analyze a position using AI models."""
    try:
        response = ai_client.post("/analyze", {
            "symbol": symbol,
            "timeframe": timeframe
        })
        return response.data
    except AIServerError as e:
        return {
            "error": True,
            "error_type": "ai_server_error",
            "message": str(e),
            "retryable": e.is_transient,
            "suggestion": "Try with different parameters or wait"
        }
```

---

## Quick Reference Summary

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

---

## Documents Created

| Document | Purpose |
|----------|---------|
| `docs/mcp-agent-architecture.md` | Full architecture reference with code templates |
| `docs/mcp-agent-lessons-learned.md` | Root causes and solutions for the original problems |
| `docs/mcp-agent-outcomes.md` | Decisions made and next steps |
| `docs/mcp-agent-production-pitfalls.md` | 10 edge cases to watch for in production |

---

## Production Pitfalls Identified

| Pitfall | Risk | Solution |
|---------|------|----------|
| Idempotency failures | Double trade execution | Idempotency keys on mutating ops |
| Latency cascades | System-wide slowdown | Circuit breakers on AI server |
| Token explosion | Context window exceeded | Pagination, summary-first responses |
| Prompt injection | Malicious tool responses | Sanitize at MCP layer |
| Race conditions | Lost updates | Optimistic locking with versions |
| Observability gaps | Can't debug failures | Trace IDs across all tiers |
| No degradation | AI down = system down | Fallback modes in MCP |
| Cost runaway | Budget blown by loops | Session cost limits |
| Schema drift | Parser breaks silently | Versioned APIs, Pydantic validation |
| Session cleanup | Crash loses work | Periodic auto-save |

---

## Key Takeaways

1. **Retry loops** are caused by unhelpful error messages. Return structured errors with `suggestion` and `retryable` fields.

2. **Context truncation** is solved by persisting important facts to MCP resources and reloading at session start.

3. **State ownership:** Agent = ephemeral execution state. MCP = persistent domain state. AI Server = stateless.

4. **Architecture:** MCP server should proxy AI server calls. Agent only speaks MCP protocol.

5. **Trading systems:** Idempotency keys are critical. A retry that executes a second trade is expensive.

---

## Files in Repository

```
langchain/
├── docs/
│   ├── mcp-agent-architecture.md
│   ├── mcp-agent-lessons-learned.md
│   ├── mcp-agent-outcomes.md
│   └── mcp-agent-production-pitfalls.md
└── conversation-log-2026-03-27.md   # This file
```

---

*End of conversation log*
