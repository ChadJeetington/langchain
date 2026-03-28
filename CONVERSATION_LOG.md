# Conversation Log (Agent A — summary)

**Date:** 2026-03-27

---

## Turn 1
**1:24:42 PM** *(initial prompt)*

**User:** I've got a LangChain ReAct agent in production using an MCP tool server over streamable HTTP. When a tool call fails or returns something the model can't parse, it just retries in a loop. I've bumped max_iterations and tweaked the system prompt but it keeps happening. This is a serious issue for my team.

The other thing is bad for our performance on longer sessions is that early tool outputs get truncated and then the model just forgets context it still needs.

I'm trying to figure out what the right architecture actually looks like here. How do you handle tool failures that aren't just a simple retry? Also, where should persistent state live. Should it live with the langchain agent or should it live on the MCP server where the tool calls and resources are?

We will need a production-grade architecture and fix. If you need more context, feel free to ask me as we solve this.

---

**Assistant:** Explored the codebase and identified two root problems.

Root cause diagnosis:

Problem 1 — Retry loops: There are two distinct retry layers being conflated. `ToolRetryMiddleware` retries the underlying execution (transient network/MCP failures). But the loop is happening at the ReAct graph level: the tool returns `ToolMessage(status="error")`, the model sees it, and makes another identical tool call because the error message says "please try again." The model is doing exactly what it was trained to do.

Problem 2 — Context truncation: The agent accumulates `ToolMessage` payloads in the message list. When the context window fills, the LLM provider truncates from the beginning, silently dropping tool results the model still needs.

Fix 1 — Error classification and circuit breaker:

Classify errors at the `ToolRetryMiddleware` boundary so permanent failures never tell the model "please try again":

```python
def _classify_mcp_failure(exc: Exception) -> str:
 if isinstance(exc, httpx.HTTPStatusError):
 if exc.response.status_code == 400:
 return (
 f"Tool input was rejected (400 Bad Request): {exc.response.text}. "
 "Do not retry this call — fix the arguments or use a different tool."
 )
 if exc.response.status_code == 422:
 return (
 f"Tool output could not be parsed (422 Unprocessable): {exc.response.text}. "
 "This is a permanent server-side error. Do not retry."
 )
 if 500 <= exc.response.status_code < 600:
 return f"MCP server temporarily unavailable ({exc.response.status_code}). Skipping this step."
 if isinstance(exc, ValueError):
 return (
 f"Tool returned unparseable output: {exc}. "
 "Do not call this tool again in this session — report the issue to the user."
 )
 return f"Tool failed: {exc}. Consider an alternative approach."

def _should_retry_mcp(exc: Exception) -> bool:
 if isinstance(exc, httpx.HTTPStatusError):
 return exc.response.status_code >= 500
 if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException)):
 return True
 return False

mcp_retry_middleware = ToolRetryMiddleware(
 max_retries=2,
 retry_on=_should_retry_mcp,
 on_failure=_classify_mcp_failure,
 backoff_factor=2.0,
 initial_delay=1.0,
 max_delay=10.0,
)
```

Add `ToolCallLimitMiddleware` as a hard safety net:

```python
tool_cap = ToolCallLimitMiddleware(max_tool_calls=25)
```

Fix 2 — Context management with `SummarizationMiddleware`:

```python
summarizer = SummarizationMiddleware(
 model="claude-sonnet-4-6",
 trigger=[
 ("fraction", 0.70),
 ("messages", 80),
 ],
 keep=("fraction", 0.25),
)

agent = create_agent(
 model,
 tools=mcp_tools,
 middleware=[summarizer, mcp_retry_middleware, tool_cap],
)
```

State architecture:

- Conversation messages → LangGraph checkpointer (Postgres/Redis)
- Domain/application state → MCP server resources
- Working context (current turn) → managed by trim_messages + summarization

Questions to sharpen the implementation:
1. Are the MCP tool outputs themselves large?
2. What is your checkpointer setup?
3. Is the unparseable output coming from the MCP server's response format, or from the model's tool call arguments being malformed?

---

## Turn 2
**1:34:50 PM** *("1. No, they are quite small…")*

**User:** 1. No, they are quite small and reasonable. 2. I am not using any of these. Should I be? 3. I can't tell. That is why I need the guidance on how to implement this.

---

**Assistant:** On checkpointer: Yes, for production. Without a checkpointer, every call to `agent.invoke()` is completely isolated. Use `langgraph-checkpoint-postgres`:

```python
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection

conn = Connection.connect("postgresql://user:pass@host:5432/db", autocommit=True)
checkpointer = PostgresSaver(conn)
checkpointer.setup()

agent = create_agent(
 model=model,
 tools=mcp_tools,
 middleware=[...],
 checkpointer=checkpointer,
)

config = {"configurable": {"thread_id": "user-session-abc123"}}
result = agent.invoke({"messages": [HumanMessage("do X")]}, config=config)
```

On diagnosing the parse failure — add a diagnostic middleware:

```python
class MCPDiagnosticMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
 def wrap_tool_call(self, request, handler):
 tool_name = request.tool.name if request.tool else request.tool_call["name"]
 tool_call_id = request.tool_call.get("id", "unknown")
 raw_args = request.tool_call.get("args", {})

 logger.debug("TOOL_INPUT tool=%s id=%s args=%s", tool_name, tool_call_id,
 json.dumps(raw_args, default=str))

 result = handler(request)

 if isinstance(result, ToolMessage):
 logger.debug("TOOL_OUTPUT tool=%s id=%s status=%s content=%r",
 tool_name, tool_call_id, result.status, str(result.content)[:500])
 if result.status == "error":
 logger.warning("TOOL_ERROR tool=%s id=%s content=%r",
 tool_name, tool_call_id, result.content)
 return result
```

Diagnostic patterns to look for:

- `TOOL_INPUT args={}` or args missing required fields → Model sending malformed tool calls
- `TOOL_ERROR content="422 Unprocessable"` → MCP server rejected the input
- `TOOL_ERROR content="could not parse"` → MCP server returned something undeserializable
- `TOOL_OUTPUT status=success` followed by another identical `TOOL_INPUT` → Tool succeeded but model interpreted it as failure

---

## Turn 3
**1:38:38 PM** *("Getting these 3 errors…")*

**User:** Getting all 3 errors. Tell me how to fix them.

---

**Assistant:** Fix 1 — Model sends bad args: make the schema and Field descriptions explicit:

```python
class QueryResourceInput(BaseModel):
 resource_id: str = Field(
 description="The unique resource identifier. Must be a UUID string, e.g. '3f2a1b4c-...'"
 )
 query: str = Field(
 description="The search query. Use plain natural language, not a JSON object."
 )
 limit: int = Field(default=10, ge=1, le=100,
 description="Max number of results to return. Defaults to 10.")

@tool(
 args_schema=QueryResourceInput,
 handle_validation_error=lambda e: (
 f"Invalid arguments: {e}. "
 "Check the field descriptions and retry with corrected values."
 ),
)
def query_mcp_resource(resource_id: str, query: str, limit: int = 10) -> str:
 """Search a specific MCP resource by ID. Pass each field separately, not as a JSON string."""
 ...
```

Fix 2 — MCP server rejects input (4xx): add a request adapter:

```python
def _run(self, resource_id: str, query: str, limit: int = 10) -> str:
 payload = {
 "resourceId": resource_id,
 "searchQuery": query,
 "maxResults": limit,
 }
 response = self._client.post("/query", json=payload)

 if response.status_code == 400:
 raise ToolException(
 f"MCP server rejected the request (400): {response.text}. "
 "Do not retry with the same arguments."
 )
 if response.status_code == 422:
 raise ToolException(
 f"MCP server could not process the input (422): {response.text}. "
 "This is a permanent error. Do not retry."
 )
 if response.status_code >= 500:
 raise ToolException(f"MCP server error ({response.status_code}). Skipping this step.")

 response.raise_for_status()
 return response.text
```

Fix 3 — MCP server response unparseable: add a response normalizer:

```python
def _normalize_mcp_response(raw: str | bytes | Any) -> str:
 if isinstance(raw, bytes):
 raw = raw.decode("utf-8", errors="replace")
 if isinstance(raw, str):
 try:
 parsed = json.loads(raw)
 return json.dumps(parsed, indent=None, ensure_ascii=False)
 except json.JSONDecodeError:
 return raw.strip()
 if isinstance(raw, dict):
 try:
 if "result" in raw:
 return json.dumps(raw["result"], ensure_ascii=False)
 if "data" in raw:
 return json.dumps(raw["data"], ensure_ascii=False)
 return json.dumps(raw, ensure_ascii=False)
 except (TypeError, ValueError) as e:
 return f"[unserializable response: {e}]"
 if isinstance(raw, list):
 try:
 return json.dumps(raw, ensure_ascii=False)
 except (TypeError, ValueError) as e:
 return f"[unserializable list: {e}]"
 return str(raw)
```

---

## Turn 4
**1:41:22 PM** *("I was able to fix error number two…")*

**User:** I was able to fix error number two with this guidance, but I am still confused on the first. Give me some examples for a frame of reference. I will then go check my code repo.

---

**Assistant:** Three common patterns showing bad vs fixed schemas:

Example 1 — Wrong type because field name is ambiguous:

```python
# Bad — model sends "the latest sales report" instead of a UUID
@tool
def get_record(id: str) -> str:
 """Get a record."""
 ...

# Fixed
class GetRecordInput(BaseModel):
 id: str = Field(description="UUID of the record, e.g. '3f2a1b4c-...'. Do not use names.")

@tool(args_schema=GetRecordInput)
def get_record(id: str) -> str:
 """Fetch a record by its UUID. The id must be a UUID obtained from a previous call."""
 ...
```

Example 2 — Model wraps everything in a JSON string:

```python
# Bad — model sends {"query": "{\"term\": \"invoices\", \"limit\": 10}"}
@tool
def search(query: str) -> str:
 """Search for items."""
 ...

# Fixed — split into real fields
class SearchInput(BaseModel):
 term: str = Field(description="The search term. Plain text, not JSON.")
 limit: int = Field(default=10, ge=1, le=50, description="Max results to return.")
```

Example 3 — Model sends an object where a scalar is expected:

```python
# Bad — model sends {"date_range": {"start": "2024-01-01", "end": "2024-12-31"}}
@tool
def filter_by_date(date_range: str) -> str:
 ...

# Fixed
class DateRangeInput(BaseModel):
 start_date: str = Field(description="Start date in YYYY-MM-DD format, e.g. '2024-01-01'.")
 end_date: str = Field(description="End date in YYYY-MM-DD format, e.g. '2024-12-31'.")
```

Signals to look for in your codebase:
- `def my_tool(input: str)` with no schema
- Fields named `id`, `data`, `config`, `params`
- Tool description under one sentence
- Multiple logical params merged into one `str`

---

## Turn 5
**1:42:41 PM** *("Create a simple diagram…" — condensed with Turn 6 in the assistant reply below; no separate user block in this summary.)*

## Turn 6
**1:44:48 PM** *("I need to see the flow…")*

**User:** I need to see the flow from agent to MCP server too.

---

**Assistant:** Provided a full flow ASCII diagram showing the complete path from User Input through AgentState, SummarizationMiddleware, LLM, MCPDiagnosticMiddleware, ToolRetryMiddleware, ToolCallLimitMiddleware, MCPQueryTool._run(), and MCP server with status code routing back to ToolMessage and AgentState.

---

## Turn 7
**1:50:34 PM** *("Better but still messy…")*

**User:** The diagrams have messy formatting and they are hard to read. Fix them for me.

---

**Assistant:** Provided cleaner versions of both diagrams with consistent box widths, minimal text inside boxes, and annotations on the side.

---

## Turn 8
**1:52:00 PM** *(“most likely cause” follow-up — canonical Turn 8)*

**User:** Better but still messy. Based on what I have told you and these diagrams, what is the most likely cause of our issue.

---

**Assistant:** Error 1 is the root cause of the retry loop.

The chain:
1. Model sends bad args
2. Pydantic ValidationError is raised
3. Without `handle_validation_error` set, it propagates to `ToolRetryMiddleware`
4. `ToolRetryMiddleware` exhausts retries and returns the default message: "Tool failed after 2 attempts. Please try again."
5. Model reads "please try again" and calls the same tool with the same bad args
6. Go to step 1

Error 3 (unparseable response) is likely a secondary symptom — if the model is sending bad args, the MCP server may be returning malformed error payloads rather than clean 4xx responses.

Fix priority:
1. `handle_validation_error` on tools so ValidationError becomes a "fix your args" ToolMessage
2. `on_failure=_classify_mcp_failure` in ToolRetryMiddleware so exhausted retries never say "please try again"
3. `_normalize_mcp_response` — likely clears up on its own once args are correct

Check tools for `handle_validation_error` first. That single missing flag is almost certainly driving the loop.

---

## Turn 9
**1:53:18 PM** *("Tell me exactly how to debug…")*

**User:** Tell me exactly how to debug handle_validation_error and steps to go through.

---

**Assistant:** Step-by-step debugging guide:

Step 1 — Check if `handle_validation_error` is set:
```bash
grep -rn "handle_validation_error" your_tools/
```
If nothing comes back, it is not set anywhere.

Step 2 — Temporarily set it to `True` with logging to confirm ValidationErrors are being raised:
```python
@tool(
 args_schema=YourInputSchema,
 handle_validation_error=lambda e: (
 logger.warning("VALIDATION_FAILED: %s", e) or
 f"Invalid arguments: {e}. Fix the fields and retry."
 ),
)
def your_tool(...):
 ...
```

Step 3 — Read the ValidationError to find which field is failing. The error message names the exact field and constraint.

Step 4 — Trace the bad arg back to the schema using the error type table:
- `field required` → model didn't know this field was needed
- `value is not a valid uuid` → field named ambiguously, model sent a name/description
- `value is not a valid integer` → model sent a string where int was expected
- `none is not an allowed value` → model sent null

Step 5 — Fix the Field description for that specific field.

Step 6 — Verify the fix stopped the loop by checking diagnostic logs for `TOOL_INPUT` showing correct format and `TOOL_OUTPUT status=success`.

Step 7 — Once confirmed, switch from `True` to a callable that includes "Do not repeat the same arguments."

---

## Turn 10
**1:54:57 PM** *("I followed the steps…")*

**User:** I followed the steps and received this error: `value is not a valid uuid (type=value_error.uuid)`. How do we fix it?

---

**Assistant:** This is the exact case from Step 4. The model sent a name or description instead of a UUID.

Fix Part 1 — Update the Field description:
```python
class YourInputSchema(BaseModel):
 resource_id: str = Field(
 description=(
 "UUID of the resource. Must be a UUID string obtained from a prior "
 "list or search call. Example: '3f2a1b4c-d5e6-7890-abcd-ef1234567890'. "
 "Do not use names, titles, or descriptions as the ID."
 )
 )
```

Fix Part 2 — Update the tool description:
```python
@tool(args_schema=YourInputSchema)
def your_tool(resource_id: str) -> str:
 """Fetch a resource by UUID. The resource_id must come from a prior
 list_resources or search call — do not guess or invent a UUID."""
 ...
```

The model is inventing a UUID because it has no other source for one. Check:
1. Does a list/search tool exist that returns UUIDs?
2. Does the system prompt tell the model the correct call order?

```python
system_prompt = """
When fetching a resource, you must first call list_resources or search_resources
to obtain a valid resource_id before calling get_resource. Never invent a resource_id.
"""
```

---

## Turn 11
**1:56:03 PM** *("Could it be my output schema…")*

**User:** Could it be my output schema needs adjusting?

---

**Assistant:** Yes. If the tool that returns UUIDs does not expose them clearly in its output, the model never has a valid UUID to pass downstream.

```python
# Bad — model can't extract a UUID
def list_resources() -> str:
 return "Found: Sales Report, Q4 Summary, Budget Doc"

# Bad — UUID is buried
def list_resources() -> str:
 return json.dumps({"results": [{"title": "Sales Report", "meta": {"id": "3f2a..."}}]})

# Good — UUID at top level, field name matches downstream schema exactly
def list_resources() -> str:
 return json.dumps([
 {"resource_id": "3f2a1b4c-d5e6-7890-abcd-ef1234567890", "name": "Sales Report"},
 {"resource_id": "9c8b7a6d-e5f4-3210-ghij-kl0987654321", "name": "Q4 Summary"},
 ])
```

Two things to verify:
1. Is the UUID field named `resource_id` to match exactly what the downstream tool's `args_schema` expects?
2. Is the UUID at the top level of each result, not nested?

---

## Turn 12
**2:11:05 PM** *(documentation / `MCP_AGENT_GUIDE.md`; canonical log also notes 1:57:20 PM for “I found the error!”)*

**User:** I found the error! It was a mistake in the output schema and the issue is resolved. Create documentation for our findings, fixes, and things to look out for in the future to keep this from happening again.

---

**Assistant:** Created `MCP_AGENT_GUIDE.md` at the repo root covering:
- Architecture overview with full middleware stack
- All 5 problems with root cause chains and exact fixes
- Debugging checklist for future failures
- New tool checklist to prevent schema mistakes
- Change checklist for modifying existing tools

---

## Turn 13
**2:18 PM** *(new MCP server steps)*

**User:** Now let's say we want to implement a new mcp server. Give me the recommended steps on how to do so.

---

**Assistant:** Recommended steps to implement a new MCP server:

1. **Choose transport type** — `http` for remote/cloud APIs (recommended), `stdio` for local processes, `sse` is deprecated.
2. **Add the server** via CLI or directly in `.mcp.json`:
```bash
claude mcp add --transport http my-server https://api.example.com/mcp
```
3. **Choose a scope** — `--scope project` for team-shared (committed to version control), `--scope user` for personal only.
4. **Handle authentication** — use environment variables for secrets (`"Bearer ${API_KEY}"`). For OAuth add `--callback-port 8080`.
5. **Verify** — `claude mcp list` to confirm registered, then `/mcp` inside Claude Code to confirm connected.

Note: The project already has `.mcp.json` with two LangChain servers — add new entries following the same pattern.

---

## Turn 14
**2:33 PM** *(external vs internal MCP — using `MCP_AGENT_GUIDE.md`)*

**User:** Using the guide, help me set up an external MCP server and how to properly connect to it and architect with my internal mcp server.

---

**Assistant:** Both existing servers (`docs-langchain`, `reference-langchain`) are already external HTTP servers in `.mcp.json`. To add a new external server:

Step 1 — Register in `.mcp.json` alongside the existing entries.

Step 2 — Wrap as a LangChain tool with a request adapter translating your `snake_case` field names to the server's wire format.

Step 3 — Define a strict `args_schema` with `Field(description=...)` that includes an example UUID and says "Do not invent or guess."

The external server slots into the bottom of the middleware stack defined in the guide. Critical integration rules:
- `handle_validation_error` on every tool (stops retry loops)
- `on_failure` must say "do not retry" for 4xx
- Expose IDs at top-level in output so the model doesn't invent them
- Add request adapter inside `_run` to decouple your schema from server wire format
- For cross-server tool chains: upstream output field name must match downstream `args_schema` field name exactly

---

## Turn 15
**2:49 PM** *(ideal architecture diagram — full repo explore)*

**User:** Create an ideal architecture diagram based off of the repo documentation. What it should look like with agent, internal mcp server, and external mcp server.

---

**Assistant:** Produced the full three-tier architecture diagram showing:
- **Agent layer** (`create_agent()` → `CompiledStateGraph`): `AgentState.messages` ← `PostgresSaver`, then `SummarizationMiddleware`, `LLM`, `MCPDiagnosticMiddleware`, `ToolRetryMiddleware`, `ToolCallLimitMiddleware`
- **Internal MCP Tool** and **External MCP Tool** branches side by side, each with `MCPQueryTool._run()` doing Pydantic validation → request adapter → HTTP POST
- Internal server (your own infra, `.mcp.json` stdio or http) and external servers (`docs-langchain`, `reference-langchain`)
- Shared `_normalize_mcp_response()` → `ToolMessage` at the bottom with 200/4xx/5xx routing

Key design rules table: thread_id always passed, summarizer runs before_model, diagnostic middleware monitors TOOL_ERROR spikes, `on_failure` says "do not retry" for 4xx, tool cap is last line of defense, `handle_validation_error` + request adapter mandatory, IDs always at top level.

---

## Turn 16
**3:02:15 PM** *(session end / final turn)*

**User:** What do you think of this? Any room for improvement?

---

**Assistant:** The diagram is accurate and covers the critical pieces. Honest gaps:

1. **Retry loop not shown** — `ToolRetryMiddleware` retries back to the tool, not the LLM. The 5xx loop within the middleware stack is missing.
2. **Return path absent** — the upward path (`ToolMessage → ToolRetryMiddleware → MCPDiagnosticMiddleware → AgentState → LLM re-evaluates`) is where the hard bugs live and isn't shown.
3. **LangGraph node loop hidden** — `create_agent()` compiles a `CompiledStateGraph` with `agent_node` and `tools_node` that loop until the LLM stops calling tools; the diagram abstracts this away entirely.
4. **`_normalize_mcp_response` placement** — it's inside each tool's `_run` individually, not a shared layer beneath both.
5. **Middleware hook types conflated** — `SummarizationMiddleware` fires on `before_model`; tool middlewares fire on `wrap_tool_call`. These are different hook points.

What's solid: middleware ordering is correct, 4xx vs 5xx handling is accurately placed, internal vs external distinction is useful, key design rules table is the most actionable part.

---
