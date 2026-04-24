# Frontend API

The Go control plane exposes a local HTTP API for frontend clients. gRPC and
`engine/proto/engine.proto` remain the canonical internal contract; this API is
the browser/Electron-friendly gateway.

## Runtime

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/runtime/status` | Runtime health, loaded models, resources, and execution mode |
| `GET` | `/api/v1/runtime/models` | Discovered local/provider models |
| `POST` | `/api/v1/runtime/models/load` | Mark/load a model for use |
| `POST` | `/api/v1/runtime/models/unload` | Unload a model |
| `POST` | `/api/v1/runtime/inference/stream` | Stream inference tokens with SSE |

Load a model:

```json
{
  "model_id": "demo.gguf",
  "options": {
    "provider": "local-openai"
  }
}
```

Stream inference:

```json
{
  "model_id": "local-openai/llama",
  "provider": "local-openai",
  "prompt": "Explain this workspace",
  "parameters": {
    "session_id": "sess-1",
    "temperature": "0.2"
  },
  "context_refs": ["viking://resources/doc-1"]
}
```

SSE events:

```text
event: token
data: {"token":"Hello","complete":false,"metrics":{}}

event: complete
data: {"token":"","complete":true,"metrics":{"provider":"local-openai"}}

event: error
data: {"error":{"code":"backend_unavailable","message":"model not loaded"}}
```

SSE is the MVP streaming transport because chat inference is server-to-client
token flow. If the frontend later needs bidirectional realtime controls,
`github.com/coder/websocket` is the preferred websocket package to evaluate.

**Note on reconnect semantics:** Browser and Electron `EventSource` clients
auto-reconnect by default when a connection is lost. The server **does not**
support `Last-Event-ID` replay, so reconnects will **not** resume an in-flight
inference. Clients must re-issue the `POST` request to
`/api/v1/runtime/inference/stream` if a reconnect is required, which may
accidentally resend requests. To reduce spurious reconnects, consider periodic
server-side keep-alive (e.g., comment or ping events) or shorter server
timeouts to cleanly close idle streams. See the `event: token`, `event:
complete`, and `event: error` examples above for the expected event shapes.

## RAG

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/rag/status` | RAG index status |
| `GET` | `/api/v1/rag/documents` | Indexed documents |
| `POST` | `/api/v1/rag/documents` | Upsert a document |
| `DELETE` | `/api/v1/rag/documents/:id` | Delete a document |
| `POST` | `/api/v1/rag/search` | Search indexed content |

Upsert a document:

```json
{
  "document_id": "doc-1",
  "content": "Text to index",
  "metadata": {
    "title": "Doc 1"
  }
}
```

Search:

```json
{
  "query": "install flow",
  "top_k": 5,
  "filters": {
    "title": "Doc 1"
  }
}
```

## Context

Existing context endpoints remain available under `/api/v1/context/*` for
resources, workspace sync, managed files, and session history.

## Capabilities

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/capabilities` | Frontend feature capability map |
| `GET` | `/api/v1/training/status` | Staged training status |
| `GET` | `/api/v1/mcp/status` | Staged MCP status |

Training and MCP are visible for the MVP but remain explicitly staged. MCP only
advertises built-in plumbing tools: `mcp.describe_connection` and `mcp.echo`.

## Errors

New frontend gateway endpoints return errors in this shape:

```json
{
  "error": {
    "code": "invalid_request",
    "message": "prompt is required"
  }
}
```

Known codes are `invalid_request`, `not_found`, `backend_unavailable`, and
`internal_error`.

## CORS

Default frontend origins are local-first:

```yaml
server:
  cors:
    enabled: true
    allowed_origins:
      - "http://localhost:*"
      - "http://127.0.0.1:*"
      - "app://ai-engine"
    allowed_headers:
      - "Content-Type"
      - "Authorization"
```
