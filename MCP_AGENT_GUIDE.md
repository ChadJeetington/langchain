# MCP Agent: Troubleshooting Guide

Production findings, fixes, and prevention guidelines for the LangChain ReAct agent running over a streamable HTTP MCP tool server.

---

## Architecture overview

```
User Input
    │
    ▼
AgentState.messages  ◄── Postgres checkpointer
    │
    ▼
SummarizationMiddleware   tokens > 70% → summarize, keep 25%
    │
    ▼
LLM  ◄── reads args_schema + Field descriptions
    │
    ▼
MCPDiagnosticMiddleware   logs raw args in/out
    │
    ▼
ToolRetryMiddleware        5xx/timeout → retry ×2 | 4xx → "do not retry"
    │
    ▼
ToolCallLimitMiddleware    calls > 25 → stop
    │
    ▼
MCPQueryTool._run()
  1. Pydantic validates args
  2. Request adapter (snake_case → camelCase, field renames)
  3. HTTP POST to MCP server
    │
    ├── 4xx → ToolException "do not retry"
    ├── 5xx → ToolException (retry middleware handles)
    └── 200 → _normalize_mcp_response() → ToolMessage
```

---

## Problems found and fixes applied

### Problem 1: Agent stuck in a retry loop

**Symptom:** Agent calls the same tool repeatedly without making progress.

**Root cause chain:**
1. Model sends bad args (wrong format, invented UUID, nested JSON string)
2. Pydantic `ValidationError` is raised inside the tool
3. Without `handle_validation_error`, the error propagates to `ToolRetryMiddleware`
4. `ToolRetryMiddleware` exhausts retries and returns the default message: *"Tool failed after 2 attempts. Please try again."*
5. Model reads "please try again" and calls the same tool with the same bad args
6. Loop repeats indefinitely

**Fix — three layers working together:**

Layer 1: `handle_validation_error` on every tool so `ValidationError` becomes a
"fix your args" `ToolMessage` instead of propagating:

```python
@tool(
    args_schema=YourInputSchema,
    handle_validation_error=lambda e: (
        f"Invalid arguments: {e}. "
        "Check the field descriptions and retry with corrected values. "
        "Do not repeat the same arguments."
    ),
)
def your_tool(...) -> str:
    ...
```

Layer 2: `on_failure` callable on `ToolRetryMiddleware` so exhausted retries
never say "please try again":

```python
def _classify_mcp_failure(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        if exc.response.status_code < 500:
            return (
                f"Permanent error ({exc.response.status_code}): {exc.response.text}. "
                "Do not retry with the same arguments."
            )
    return f"Tool failed: {exc}. Consider an alternative approach."

ToolRetryMiddleware(
    max_retries=2,
    retry_on=_should_retry_mcp,
    on_failure=_classify_mcp_failure,
)
```

Layer 3: `ToolCallLimitMiddleware` as a hard safety net:

```python
ToolCallLimitMiddleware(max_tool_calls=25)
```

---

### Problem 2: MCP server rejecting input (400/422)

**Symptom:** `TOOL_ERROR status=400` or `status=422` in diagnostic logs immediately
after the tool is called.

**Root cause:** The LangChain tool wrapper was passing valid-to-Pydantic args that
the MCP server did not accept. Field names and casing did not match the server's
wire format.

**Fix — request adapter inside `_run`:**

```python
def _run(self, resource_id: str, query: str, limit: int = 10) -> str:
    # Translate validated LangChain args to exact MCP server wire format
    payload = {
        "resourceId": resource_id,   # server uses camelCase
        "searchQuery": query,         # server field name differs
        "maxResults": limit,
    }
    response = self._client.post("/query", json=payload)

    if response.status_code == 400:
        raise ToolException(
            f"Bad request (400): {response.text}. "
            "Do not retry with the same arguments."
        )
    if response.status_code == 422:
        raise ToolException(
            f"Unprocessable (422): {response.text}. Permanent error, do not retry."
        )
    if response.status_code >= 500:
        raise ToolException(f"Server error ({response.status_code}). Try again.")

    return _normalize_mcp_response(response.text)
```

---

### Problem 3: UUID validation error and model inventing IDs

**Symptom:** `ValidationError: value is not a valid uuid (type=value_error.uuid)`

**Root cause:** The model was calling a downstream tool with an invented UUID because:
1. The upstream list/search tool did not expose `resource_id` clearly in its output
2. The UUID was buried in a nested structure the model read past
3. The tool description did not tell the model where to get a valid UUID from

**Fix — output schema of the list/search tool:**

```python
# Bad — UUID buried or absent
{"results": [{"title": "Sales Report", "meta": {"id": "3f2a..."}}]}

# Good — UUID at top level, field name matches downstream schema exactly
[
    {"resource_id": "3f2a1b4c-d5e6-7890-abcd-ef1234567890", "name": "Sales Report"},
    {"resource_id": "9c8b7a6d-e5f4-3210-ghij-kl0987654321", "name": "Q4 Summary"},
]
```

**Fix — Field description on the downstream tool:**

```python
resource_id: str = Field(
    description=(
        "UUID of the resource. Must be obtained from a prior list_resources or "
        "search_resources call. Example: '3f2a1b4c-d5e6-7890-abcd-ef1234567890'. "
        "Do not invent or guess a UUID."
    )
)
```

**Fix — tool description reinforcing call order:**

```python
@tool(args_schema=YourInputSchema)
def get_resource(resource_id: str) -> str:
    """Fetch a resource by UUID. Always call list_resources or search_resources
    first to obtain a valid resource_id. Never invent a UUID."""
    ...
```

**Fix — system prompt enforcing call order:**

```python
system_prompt = """
When fetching a resource, you must first call list_resources or search_resources
to obtain a valid resource_id. Never invent or guess a resource_id.
"""
```

---

### Problem 4: Context truncation on long sessions

**Symptom:** Model loses context from earlier in the session. Early tool outputs
are referenced later but the model behaves as if they never happened.

**Root cause:** Tool outputs accumulate in the message list. When the context window
fills, the LLM provider silently truncates from the beginning. No graceful
degradation occurs.

**Fix — `SummarizationMiddleware`:**

```python
SummarizationMiddleware(
    model=model,
    trigger=[
        ("fraction", 0.70),   # fire at 70% of context window
        ("messages", 80),     # or at 80 messages, whichever comes first
    ],
    keep=("fraction", 0.25),  # keep most recent 25% after summarization
)
```

The middleware runs `before_model`, detects the threshold, summarizes old messages
into a structured `HumanMessage` (session intent, key decisions, artifacts, next
steps), and replaces them. It preserves `AIMessage`/`ToolMessage` pairs atomically
so tool call context is never split.

---

### Problem 5: No persistent state across sessions

**Symptom:** Agent starts fresh on every invocation with no memory of prior sessions.

**Fix — Postgres checkpointer:**

```python
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection

conn = Connection.connect("postgresql://user:pass@host:5432/db", autocommit=True)
checkpointer = PostgresSaver(conn)
checkpointer.setup()  # run once on startup

agent = create_agent(model, tools=mcp_tools, middleware=[...], checkpointer=checkpointer)

# Every invocation must pass a thread_id
config = {"configurable": {"thread_id": "user-session-abc123"}}
result = agent.invoke({"messages": [HumanMessage("...")]}, config=config)
```

---

## Debugging checklist

When the agent loops or a tool fails, go through this in order:

```
1. Check diagnostic logs
   TOOL_INPUT  → what did the model actually send?
   TOOL_OUTPUT → what came back, and what was the status?

2. Identify the failure layer
   ValidationError before HTTP call  →  Problem 1 or 3 (args or UUID)
   400/422 from server               →  Problem 2 (request adapter)
   500 from server                   →  Transient, retry middleware handles it
   200 but model retries anyway      →  Output schema unclear, model misread result

3. For UUID errors specifically
   - Does the upstream tool return resource_id at the top level?
   - Does the field name match exactly what the downstream tool expects?
   - Does the tool description say where to get the UUID from?

4. For any retry loop
   - Is handle_validation_error set on every tool?
   - Does on_failure in ToolRetryMiddleware say "do not retry"?
   - Is ToolCallLimitMiddleware in the middleware stack?
```

---

## Things to watch for going forward

**When adding a new MCP tool:**
- [ ] Define an explicit `args_schema` with `Field(description=...)` on every field
- [ ] Include a concrete example value in every `Field` description
- [ ] Set `handle_tool_error=True` and `handle_validation_error` with a callable
- [ ] Add a request adapter mapping LangChain field names to MCP server wire format
- [ ] Wrap the response in `_normalize_mcp_response()`
- [ ] If the tool returns IDs used by other tools, expose them at the top level with the exact field name the downstream tool expects
- [ ] If call order matters, document it in the tool description and system prompt

**When changing an existing MCP tool:**
- [ ] Any output schema change may break downstream tools that depend on it — check all tools that consume the output
- [ ] Any field rename in `args_schema` must be reflected in the request adapter
- [ ] Any field rename in output must be reflected in the downstream tool's `Field` description

**Ongoing:**
- Monitor `TOOL_ERROR` log lines — a spike means a schema drift between LangChain and the MCP server
- Monitor tool call counts per session — if average approaches the 25-call limit, the agent is looping somewhere
- Run `MCPDiagnosticMiddleware` in staging before deploying any tool changes
