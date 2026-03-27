# Lessons Learned: LangChain + MCP Agent Production Issues

Documented solutions to production issues with LangChain ReAct agents using MCP tool servers.

## Problem 1: Tool Failure Retry Loops

### Symptom
Agent retries the same tool call indefinitely when a tool fails or returns something the model can't parse. Bumping `max_iterations` and tweaking system prompts doesn't fix it.

### Root Cause
Two distinct retry mechanisms were conflated:

1. **ToolRetryMiddleware** - automatic retry for transient failures (network, rate limits)
2. **Agent-level "retry"** - model chooses to call tool again because it doesn't know what else to do

The model wasn't retrying due to transient errors. It was retrying because the error response didn't give it a clear path forward.

### Solution
Return structured errors with explicit guidance:

```python
# BAD: Model doesn't know what to do
return {"error": "not found"}

# GOOD: Model knows to move on
return {
    "error": True,
    "error_type": "not_found",
    "message": "No customer with ID 'xyz'",
    "suggestion": "Use search_customers tool to find by email or name",
    "retryable": False
}
```

Adapter transforms to natural language:
```
"Error (not_found): No customer with ID 'xyz'.
Suggested action: Use search_customers to find by email or name.
This error is not retryable. Move on or ask the user."
```

### Key Insight
The model needs **actionable guidance**, not just error codes. "Do not retry" and "try X instead" are crucial signals.

---

## Problem 2: Context Truncation in Long Sessions

### Symptom
Early tool outputs get truncated. Model forgets context it still needs. Causes repeated tool calls and confusion.

### Root Cause
`intermediate_steps` grows unbounded. When it gets large, either:
- Context window fills up
- `trim_intermediate_steps` drops early steps
- Critical context is lost

### Solution
Three-part fix:

**1. Summarize tool history explicitly**
```python
def summarize_tool_history(intermediate_steps: list) -> str:
    summary = []
    for action, observation in intermediate_steps:
        status = "failed" if "error" in observation.lower() else "succeeded"
        summary.append(f"- {action.tool}: {status}")
    return "\n".join(summary[-10:])
```

**2. Persist important facts to MCP resource**
Don't rely on message history for critical context. Write it to a persistent resource:
```python
@server.tool()
def save_session_context(session_id: str, summary: str, facts: list) -> dict:
    # Persists to session://session_id resource
    ...
```

**3. Load context at session start**
```python
@server.tool()
def load_session_context(session_id: str) -> str:
    resource = read_resource(f"session://{session_id}")
    return f"Prior context: {resource['summary']}\nKnown facts: {resource['facts']}"
```

### Key Insight
Message history is ephemeral. Important context should be persisted to MCP resources and reloaded each session.

---

## Problem 3: State Location Confusion

### Question
Should persistent state live with the LangChain agent or on the MCP server?

### Answer
**Both, but different state.**

| State | Location | Reason |
|-------|----------|--------|
| Tool call history | Agent | Only needed during execution |
| Current goal | Agent | Session-specific |
| Working memory | Agent | Scratchpad, transient |
| User preferences | MCP | Persists across sessions |
| Conversation summary | MCP | Long-term memory |
| Domain entities | MCP | Source of truth |

### Key Insight
- **Agent state** = cheap to rebuild, model needs direct access, dies with session
- **MCP state** = persists across sessions, source of truth, other systems may need it

---

## Problem 4: AI Server Integration

### Question
How should the LangChain agent connect to a separate AI infrastructure server?

### Solution
MCP server proxies AI server calls.

```
Agent ──MCP──▶ MCP Server ──HTTP──▶ AI Server
```

**Not:**
```
Agent ──MCP──▶ MCP Server
Agent ──HTTP─▶ AI Server  (separate connection)
```

### Benefits
- Agent only speaks one protocol (MCP)
- Error handling normalized in one place
- MCP can cache expensive AI calls
- MCP can rate-limit, circuit-break
- AI server stays pure (no MCP awareness)

### Key Insight
The MCP server is a **facade** over all backend services. Agent complexity stays low.

---

## Anti-Patterns to Avoid

### 1. Bare error strings
```python
# BAD
return "Error: something went wrong"

# GOOD
return {"error": True, "error_type": "...", "suggestion": "...", "retryable": False}
```

### 2. Relying on message history for critical context
```python
# BAD: Context lost when history truncated
# Hope the model remembers user_id from 50 messages ago

# GOOD: Persist and reload
save_session_context(session_id, {"user_id": "123", "key_facts": [...]})
load_session_context(session_id)  # At session start
```

### 3. Agent connecting to multiple backends directly
```python
# BAD: Agent manages two protocols
tools = mcp_tools + http_ai_tools

# GOOD: MCP proxies everything
tools = mcp_tools  # AI tools are MCP tools that proxy internally
```

### 4. Stateful AI server
```python
# BAD: AI server tracks conversation
ai_server.analyze(symbol, session_id=session_id)

# GOOD: AI server is stateless, MCP handles context
mcp_server.analyze_position(symbol)  # MCP adds context internally
```

---

## Implementation Checklist

- [ ] Structured error responses from MCP server
- [ ] Error adapter in LangChain tool wrapper
- [ ] `retryable` flag on all errors
- [ ] `suggestion` field for non-retryable errors
- [ ] `load_session_context` tool for session start
- [ ] `save_session_context` tool for session end
- [ ] MCP server proxies AI server calls
- [ ] AI server stays stateless
- [ ] Tool history summarization for long sessions
