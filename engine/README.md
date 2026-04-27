# AI Engine

A high-performance local AI engine with a Go control plane, a Rust execution daemon, and a dedicated Rust context service.

## WinUI v1 Surface

The packaged WinUI integration path is:

1. WinUI launches the Go server with a config file.
2. The Go supervisor launches and health-checks the Rust daemon.
3. The WinUI app talks to the Go control plane over local gRPC.

For this v1 surface:

- `Runtime` and `RAG` are the supported frontend services.
- The HTTP frontend gateway also exposes a Hugging Face model browser/downloader for public GGUF/GGML/BIN artifacts.
- `Training` and `MCP` are disabled by default and return `UNIMPLEMENTED` from the Go control plane.
- The daemon is required by default. If no daemon binary or command is available, startup fails fast instead of silently falling back to in-memory managers.
- RAG uses FastEmbed locally by default and reports embedding metadata plus reindex status through `RagStatus`.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 WinUI / Native Frontend                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              Go Control Plane (gRPC/HTTP API)                │
│  daemon-backed Runtime/RAG/Training/MCP + Context gateway    │
└─────────────────────────────────────────────────────────────┘
                             │ │
                             │ └──────────────┐
                             ↓                ↓
│                  Rust Execution Daemon                       │
│  runtime_engine + rag/storage + training_engine + MCP       │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                   Rust Context Service                       │
│  context_server + context_engine (resources/files/sessions) │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Build Binaries

`build.bat` and `build.sh` now bootstrap a pinned official `protoc` automatically if the machine does not already have one on `PATH`.

```bash
cd engine
./build.sh   # Linux/macOS
build.bat    # Windows
```

### 2. Run Server

```bash
# Development binary
./go/bin/server --config ./config.example.yaml

# Windows bundle output from build.bat
dist/windows-backend/bin/server.exe --config dist/windows-backend/config.example.yaml
```

To build the daemon with embedded `mistral.rs` inference:

```bash
cd engine/rust
cargo build --release -p ai_engine_daemon --features mistralrs-backend
```

Optional hardware feature bundles are exposed through the daemon crate:

```bash
cargo build --release -p ai_engine_daemon --features mistralrs-mkl
cargo build --release -p ai_engine_daemon --features mistralrs-cuda
cargo build --release -p ai_engine_daemon --features mistralrs-cuda-flash-attn
```

### 3. Run Runtime + RAG Demo

```bash
cd engine/go
go run ./cmd/client/main.go
```

## API Services

Frontend clients should use the local HTTP/SSE gateway documented in
[`docs/frontend-api.md`](docs/frontend-api.md). gRPC remains the canonical
internal service contract.

### Runtime Service
- `GetStatus` - Get runtime status and system resources
- `ListModels` - List available models
- `LoadModel` - Load a model into memory
- `UnloadModel` - Unload a model
- `StreamInference` - Stream inference responses through the daemon-backed runtime path
- HTTP Hub endpoints - Search Hugging Face, download compatible public artifacts, then load the downloaded local file

The preferred local runtime backend is embedded `mistral.rs`. Configure it with:

```yaml
runtime:
  models_path: "~/ai-engine/models"
  backend: "mistralrs"
  max_memory_mb: 8192
  mistralrs:
    force_cpu: false
    max_num_seqs: 32
    auto_isq: ""
```

For low-risk sidecar validation, run `mistralrs serve --port 1234 -m <model>` and keep using the existing OpenAI-compatible provider path:

```yaml
runtime:
  providers:
    - name: "mistralrs-sidecar"
      type: "openai-compatible"
      url: "http://127.0.0.1:1234/v1"
      api_key: ""
```

When model loading fails, run `mistralrs doctor` first to check CUDA/Metal/MKL, Hugging Face, and local model environment issues before changing engine code.

Hugging Face browsing/downloading is handled by the Go control plane and remains
local-runtime-first. v1 accepts public `.gguf`, `.ggml`, and `.bin` artifacts
only; gated/private/token-required repositories are rejected until token support
is intentionally added.

```yaml
huggingface:
  enabled: true
  endpoint: "https://huggingface.co"
  max_download_bytes: 0
  compatible_extensions:
    - ".gguf"
    - ".ggml"
    - ".bin"
```

Downloaded models land under `runtime.models_path` with a `.hf.json` manifest
and appear in `ListModels` with Hugging Face metadata.

### RAG Service
- `UpsertDocument` - Add/update documents
- `DeleteDocument` - Remove a document
- `Search` - Semantic search
- `GetStatus` - Get RAG index status
- `ListDocuments` - List all documents

The packaged RAG path uses FastEmbed by default:

```yaml
rag:
  storage_path: "~/ai-engine/lancedb"
  embedding_provider: "fastembed"
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  embedding_cache_dir: "~/ai-engine/embedding-cache"
  embedding_allow_download: true
```

`RagStatus` includes the active embedding provider, model, dimension, provider
version, and whether persisted chunks need a reindex after an embedding change.
Use `embedding_provider: "mock"` only for deterministic tests.

### Context Service
- `GetContextStatus` - Context readiness and managed-root status
- `ListResources` / `SearchContext` - Resource inventory and layered search
- `SyncWorkspace` - Managed workspace indexing
- `ListFiles` / `ReadFile` / `WriteFile` / `DeleteFile` / `MoveFile` - Managed file operations
- `AppendSession` / `GetSession` - Session persistence

### Training Service
- Disabled by default for the WinUI v1 surface.
- `StartRun` - Start a training job
- `CancelRun` - Cancel a running job
- `ListRuns` - List all training runs
- `ListArtifacts` - List training artifacts
- `StreamLogs` - Stream training logs

### MCP Service
- Disabled by default for the WinUI v1 surface.
- `Connect` - Connect to MCP server
- `Disconnect` - Disconnect from server
- `ListTools` - List available tools
- `CallTool` - Execute built-in plumbing tools (`mcp.describe_connection`, `mcp.echo`) or return an explicit staged error for unimplemented transports

## Configuration

See `config.example.yaml` for configuration options:

- Server host/port (HTTP and gRPC)
- Daemon host/port plus binary auto-detection
- Runtime model path and memory limits
- Hugging Face public model browsing and download limits
- Context service URL, binary path, data dir, and managed roots
- RAG settings (chunk size, overlap, embedding provider/model/cache)
- Training working directory and job limits
- MCP timeout and retry settings

## Development

### Packaged Smoke Test

`smoke.ps1` is the release-style verification path for the bundled local backend. It:

- builds the server and daemon,
- provisions a temporary local config with ephemeral ports,
- copies a caller-provided GGUF model into the temporary model directory,
- runs `doctor.ps1` to validate bundled binaries, model readability, loopback ports, runtime backend, and embedding cache settings,
- launches the Go server and Rust daemon together,
- runs the `Runtime + RAG` client flow,
- restarts the stack, and
- verifies real inference output, FastEmbed-backed RAG status, and document persistence after restart.

Run it with a real model path:

```powershell
cd engine
.\smoke.ps1 -ModelPath C:\models\your-model.gguf -Force
```

Use `-ForceCpu` for CPU-only smoke environments. If FastEmbed/model downloads
are not allowed, pre-populate the embedding cache and pass
`-DisableEmbeddingDownload -EmbeddingCacheDir <cache-dir>`.

### Directory Structure

```
engine/
├── proto/              # Protocol Buffer definitions
├── go/                 # Go control plane
│   ├── cmd/
│   │   ├── server/     # Main server entrypoint
│   │   └── client/     # Example client
│   └── internal/
│       ├── api/        # gRPC/HTTP handlers
│       ├── config/     # Configuration
│       ├── mcp/        # Local fallback MCP stubs for dev/test only
│       ├── rag/        # RAG orchestration
│       ├── runtime/    # Local fallback runtime manager for dev/test only
│       ├── supervisor/ # Lifecycle management
│       └── training/   # Local fallback training manager for dev/test only
└── rust/               # Rust execution layer
    └── crates/
        ├── ai_engine_daemon/ # Canonical Runtime/RAG/Training/MCP backend
        ├── context_engine/   # Context indexing, search, files, sessions
        ├── context_server/   # HTTP context service wrapper
        ├── runtime_engine/   # Pluggable local runtime backend + model state
        ├── training_engine/  # Training lifecycle and artifacts
        ├── chunking/         # Text chunking
        ├── embedding/        # Embedding pipeline
        ├── rag_engine/       # Shared RAG logic
        └── storage/          # LanceDB-backed persistence
```

## gRPC API Reference

### Protocol Buffer Definitions

Full API definitions are in `proto/engine.proto`. Key services:

**Runtime**: Model loading, unloading, inference streaming
**RAG**: Document indexing, semantic search
**Context**: Managed roots, resource search, file/session operations
**Training**: Job orchestration, log streaming
**MCP**: Tool execution, connection management

## Environment Variables

- `AI_ENGINE_CONFIG` - Path to config file
- `AI_ENGINE_LOG_LEVEL` - Log level (debug, info, warn, error)
- `AI_ENGINE_RUNTIME_BACKEND` - Runtime backend selected for the Rust daemon (`mistralrs` or `mock`)
- `AI_ENGINE_MISTRALRS_FORCE_CPU` - Force CPU execution for embedded `mistral.rs`
- `AI_ENGINE_MISTRALRS_MAX_NUM_SEQS` - Maximum concurrent `mistral.rs` sequences
- `AI_ENGINE_MISTRALRS_AUTO_ISQ` - Optional `mistral.rs` in-situ quantization mode
- `AI_ENGINE_EMBEDDING_PROVIDER` - RAG embedding provider (`fastembed` or `mock`)
- `AI_ENGINE_EMBEDDING_MODEL` - RAG embedding model, default `sentence-transformers/all-MiniLM-L6-v2`
- `AI_ENGINE_EMBEDDING_CACHE_DIR` - Local FastEmbed model/cache directory
- `AI_ENGINE_EMBEDDING_ALLOW_DOWNLOAD` - Whether FastEmbed may download missing model assets

## Verification Notes

- `build.bat` / `build.sh` set workspace-local `GOCACHE` and `CARGO_HOME` under `engine/.cache` so local builds do not depend on machine-global cache configuration.
- `build.bat` bootstraps `protoc` (version defined in `engine/scripts/ensure_protoc.ps1`) when the extracted binary is missing, then emits a runnable Windows backend bundle under `engine/dist/windows-backend/`.
- If Cargo cannot resolve `crates.io`, that indicates a machine/network problem rather than an engine contract failure.
- The Go tests use IPv4-only local servers to avoid Windows environments where IPv6 loopback is unavailable.
