# LangChain + MCP + AI Server Architecture

Production architecture for LangChain ReAct agents with MCP tool servers and separate AI infrastructure.

## Overview

```
                        ┌─────────────────────────────────────────┐
                        │            LOCAL ENVIRONMENT             │
                        │                                          │
  ┌──────────────┐      │  ┌──────────────┐                       │
  │   LANGCHAIN  │ MCP  │  │   INTERNAL   │                       │
  │    AGENT     │─────────│  MCP SERVER  │                       │
  │ (Orchestrate)│◀────────│  (stdio/IPC) │                       │
  └──────┬───────┘      │  └──────────────┘                       │
         │              │   Domain state, session context,         │
         │              │   local tools (no auth required)         │
         │              └─────────────────────────────────────────┘
         │
         │ MCP (HTTPS + Auth)
         │
         ▼
  ┌──────────────┐      ┌──────────────┐
  │   EXTERNAL   │ HTTP │     AI       │
  │  MCP SERVER  │─────▶│   SERVER     │
  │  (Deployed)  │◀─────│(Domain Logic)│
  └──────────────┘      └──────────────┘
  HTTPS, API key auth       Stateless
  Rate limiting             Pure Logic
  Audit logging
  Public/shared tools
```

**Key distinction:**
- **Internal MCP server** — runs locally via `stdio`/IPC, no network exposure, no auth overhead. Owns session state and local domain data.
- **External MCP server** — deployed service reachable over HTTPS, requires authentication, handles shared/public tools and proxies the AI server.

## MCP Server Types

### Internal MCP Server (`stdio`)

Registered in `.mcp.json` as:
```json
"my-internal-server": {
  "type": "stdio",
  "command": "python",
  "args": ["-m", "my_internal_server"]
}
```

- Runs as a local subprocess — no network, no TLS, no auth token needed
- Owns session context, local domain state, user preferences
- Fast (IPC vs. network round-trip)
- Dies when the agent process dies (or can be kept alive as a daemon)

### External MCP Server (`http`)

Registered in `.mcp.json` as:
```json
"my-external-server": {
  "type": "http",
  "url": "https://your-domain.com/mcp",
  "headers": {
    "Authorization": "Bearer ${MY_SERVER_API_KEY}"
  }
}
```

See `.mcp.json` for examples already in use (`docs-langchain`, `reference-langchain`).

- Persistent, deployed service (e.g., FastAPI + streamable HTTP transport)
- Requires HTTPS and per-request auth (`Authorization` header via env var)
- Handles shared/public tools and proxies the AI server
- Responsible for rate limiting, circuit breaking, audit logging, and prompt-injection sanitization

---

## Component Responsibilities

### LangChain Agent (Orchestrator)

- Interprets user intent
- Selects tools and actions
- Manages conversation flow
- Error recovery and retry logic
- Explains decisions to user

**State owned:**
- `tool_history` - what was called this session
- `current_goal` - active objective
- `working_memory` - scratchpad data
- `intermediate_steps` - reasoning trace

**Lifecycle:** Ephemeral. Dies after task completion. Rebuilt each session.

### MCP Server (Context & Tools)

- Exposes tools via MCP protocol (streamable HTTP)
- Manages persistent resources
- Proxies calls to AI server
- Normalizes errors across all backends
- Handles caching, rate limiting, audit logging

**Resources owned:**
- `session://context` - conversation memory, user facts
- `portfolio://positions` - domain entities
- `market://watchlist` - user preferences
- `customer://` - domain data

**Tools exposed:**
- Native: `get_portfolio`, `search_history`, `save_session`, `get_market_data`
- Proxied: `analyze_position`, `get_trade_signal`, `execute_trade`

**Lifecycle:** Persistent. Source of truth for domain state.

### AI Infrastructure Server (Trading Logic)

- Pure ML inference and trading logic
- No MCP protocol awareness
- No conversation context handling
- Stateless request/response

**Endpoints:**
- `/analyze` - position analysis
- `/signal` - trade signal generation
- `/execute` - trade execution
- `/backtest` - strategy testing

**Lifecycle:** Stateless. Can be called by any system.

## Connection Pattern (Recommended)

The agent speaks only MCP protocol to both server types. The external MCP server is the single point of contact for the AI server — the agent never calls the AI server directly.

```
Agent ──stdio──▶ Internal MCP Server  (local state, session context)

Agent ──HTTPS──▶ External MCP Server ──HTTP──▶ AI Server
                 (auth, rate limit,              (stateless,
                  audit, caching)                 pure logic)
```

Benefits:
- Agent speaks one protocol (MCP) regardless of server type
- Internal server: zero-latency for local state access
- External server: unified error handling, caching, circuit-breaking for remote calls
- AI Server stays pure (no MCP awareness, callable by any system)
- Auth and rate limiting enforced at the external MCP boundary, not inside the agent

## State Ownership Matrix

| State Type | Owner | Why |
|------------|-------|-----|
| Tool call history | Agent | Transient, needed for reasoning |
| Current goal | Agent | Session-specific |
| Working memory | Agent | Scratchpad, dies with session |
| User preferences | MCP Server | Persists across sessions |
| Conversation summary | MCP Server | Long-term memory |
| Domain entities | MCP Server | Source of truth |
| Model weights | AI Server | ML infrastructure |
| Strategy configs | AI Server | Trading logic |

## Data Flow

### Session Start
```
Agent calls load_session_context
    → MCP returns prior facts, preferences, last summary
    → Agent has context for new session
```

### During Session
```
Agent tracks tool_history, working_memory locally
Agent calls MCP tools (get_portfolio, search_history)
Agent calls proxied AI tools (analyze_position, get_signal)
MCP server handles all external calls, normalizes responses
```

### Session End
```
Agent calls save_session_context
    → Passes summary + new facts learned
    → MCP persists to session resource
    → Next session can load this context
```

## Error Handling Architecture

### Structured Error Response

MCP Server returns:
```json
{
  "content": [{
    "type": "text",
    "text": "{\"error\": true, \"error_type\": \"not_found\", \"message\": \"...\", \"suggestion\": \"...\", \"retryable\": false}"
  }],
  "isError": true
}
```

Agent adapter transforms to:
```
"Error (not_found): No customer with ID 'xyz'.
Suggested action: search by email or name.
This error is not retryable. Move on or ask the user."
```

### Error Classification

| Error Type | Retryable | Agent Action |
|------------|-----------|--------------|
| `connection_error` | Yes | Middleware retries with backoff |
| `timeout` | Yes | Middleware retries with backoff |
| `rate_limited` | Yes | Middleware waits, retries |
| `not_found` | No | Move on, try alternative |
| `invalid_input` | No | Fix input, try again |
| `permission_denied` | No | Inform user |
| `ai_server_error` | Maybe | Check `is_transient` flag |

### Tool Adapter Pattern

```python
class MCPToolAdapter(BaseTool):
    name: str
    description: str
    mcp_client: Any

    def _run(self, **kwargs) -> str:
        result = self.mcp_client.call_tool(self.name, kwargs)

        if result.get("isError"):
            return self._handle_error(result)

        return self._format_success(result)

    def _handle_error(self, result: dict) -> str:
        error = json.loads(result["content"][0]["text"])

        response = f"Error ({error['error_type']}): {error['message']}"

        if error.get("suggestion"):
            response += f"\nSuggested action: {error['suggestion']}"

        if not error.get("retryable", False):
            response += "\nThis error is not retryable. Move on or ask the user."

        return response
```

## MCP Server Proxy Pattern

```python
@server.tool()
def analyze_position(symbol: str, timeframe: str) -> dict:
    """Analyze a position using AI models."""

    try:
        response = ai_client.post("/analyze", {
            "symbol": symbol,
            "timeframe": timeframe
        })

        if response.status == "success":
            return {
                "signal": response.data["signal"],
                "confidence": response.data["confidence"],
                "reasoning": response.data["reasoning"]
            }

    except AIServerError as e:
        return {
            "error": True,
            "error_type": "ai_server_error",
            "message": str(e),
            "retryable": e.is_transient,
            "suggestion": "Try with different parameters or wait"
        }
```

## Quick Reference

```
AGENT STATE (ephemeral)          MCP STATE (persistent)
────────────────────────         ─────────────────────
• tool_history                   • session://context
• current_goal                   • portfolio://positions
• working_memory                 • user preferences
• intermediate_steps             • conversation summary

ERROR FLOW
──────────
MCP returns structured error → Adapter formats for model → Model moves on

SESSION FLOW
────────────
START:  load_session_context → get prior facts
DURING: track locally, call tools
END:    save_session_context → persist summary
```
