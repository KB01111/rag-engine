# WinUI gRPC v1 Contract

## Startup Contract

- The WinUI app launches the Go server with a config file.
- The Go supervisor launches the Rust daemon and waits for it to become reachable on the daemon gRPC address.
- If the daemon is required and missing, the server exits instead of silently falling back to in-memory Go managers.
- The frontend connects to the Go control plane over the configured gRPC address.

## Supported Services

- `Runtime`
- `RAG`

## Deferred Services

- `Training` returns `UNIMPLEMENTED` from the Go control plane unless explicitly enabled.
- `MCP` returns `UNIMPLEMENTED` from the Go control plane unless explicitly enabled.

## Runtime Semantics

- `LoadModel` marks a discovered model as ready for CLI-backed inference.
- `StreamInference` requires a loaded model and a non-empty prompt.
- Runtime streaming forwards backend stdout chunks as `InferenceResponse.token`.
- The final message uses `complete=true` with an empty token.

## RAG Semantics

- `UpsertDocument` requires non-empty content.
- `Search` requires a non-empty query.
- `ListDocuments` and `Search` operate on the Rust-backed persistent store.
- The packaged smoke test verifies persistence by restarting the stack and re-querying the same document.

## Error Mapping

- `NOT_FOUND`
  - requested model does not exist
- `INVALID_ARGUMENT`
  - missing inference request
  - empty `model_id`
  - empty prompt
  - empty document content
  - empty search query
- `FAILED_PRECONDITION`
  - model exists but is not loaded
  - runtime backend command is missing or not resolvable
- `UNIMPLEMENTED`
  - `Training` or `MCP` is disabled for the WinUI v1 surface
- `INTERNAL`
  - storage, embedding, daemon, or backend-process failures not covered above

## Current Limitations

- The runtime path depends on a `llama-cli`-compatible command configured through `daemon.llama_cli`.
- The smoke test uses a fake backend command to validate process orchestration and streaming without requiring a real model runner installation.
- `Training` and `MCP` are intentionally out of the first frontend surface.