# Frontend API

The Go control plane exposes a local HTTP API for frontend clients. gRPC and
`engine/proto/engine.proto` remain the canonical internal contract; this API is
the browser/Electron-friendly gateway.

## Runtime

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/runtime/status` | Runtime health, loaded models, resources, and execution mode |
| `GET` | `/api/v1/runtime/models` | Discovered local/provider models |
| `GET` | `/api/v1/runtime/hub/search` | Search compatible public Hugging Face models |
| `GET` | `/api/v1/runtime/hub/model` | Inspect one Hugging Face model and file choices |
| `POST` | `/api/v1/runtime/hub/downloads` | Start a Hugging Face model-file download |
| `GET` | `/api/v1/runtime/hub/downloads` | List active/recent Hugging Face downloads |
| `GET` | `/api/v1/runtime/hub/downloads/:id/events` | Stream download progress with SSE |
| `DELETE` | `/api/v1/runtime/hub/downloads/:id` | Cancel an active download |
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

### Hugging Face Model Browser

The Hub integration is backend-managed. The frontend can browse compatible
public artifacts, download a selected file into `runtime.models_path`, then load
the downloaded local model through the normal runtime load endpoint.

Search:

```http
GET /api/v1/runtime/hub/search?query=llama&author=TheBloke&sort=downloads&limit=20
```

If the response includes `next_cursor`, request the next page by passing it back
as `cursor=<next_cursor>`. The cursor is opaque; clients should not parse it.

By default, search returns only v1-compatible public models:

- the repository is not private or gated,
- at least one visible file has `.gguf`, `.ggml`, or `.bin`,
- license metadata is visible enough to show in the UI.

Model detail:

```http
GET /api/v1/runtime/hub/model?repo_id=acme%2Ftiny
```

Download:

```json
{
  "repo_id": "acme/tiny",
  "filename": "tiny.Q4_K_M.gguf",
  "revision": "main"
}
```

Download events use the same SSE framing:

```text
event: progress
data: {"id":"...","status":"downloading","downloaded_bytes":1048576,"size_bytes":734003200}

event: complete
data: {"id":"...","status":"completed","target_path":"..."}
```

Downloaded files are written atomically with a `.partial` temporary file and a
`.hf.json` sidecar manifest. `/api/v1/runtime/models` surfaces those models
with `metadata.source=huggingface`, `repo_id`, `license`, `revision`, and
`downloaded=true`.

## RAG

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/rag/status` | RAG index status |
| `GET` | `/api/v1/rag/documents` | Indexed documents |
| `POST` | `/api/v1/rag/documents` | Upsert a document |
| `DELETE` | `/api/v1/rag/documents/:id` | Delete a document |
| `POST` | `/api/v1/rag/search` | Search indexed content |

Status includes embedding metadata so the frontend can surface whether an index
needs a reindex after the embedding backend changes:

```json
{
  "status": {
    "document_count": 1,
    "chunk_count": 3,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "embedding_provider": "fastembed",
    "embedding_dimension": 384,
    "embedding_version": "fastembed-5.13.3",
    "requires_reindex": false,
    "reindex_reasons": []
  },
  "execution_mode": "daemon"
}
```

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

Known codes are `invalid_request`, `not_found`, `backend_unavailable`,
`internal_error`, `unsupported_model_format`, `download_too_large`, and
`hf_token_required`.

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
