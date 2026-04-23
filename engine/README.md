# AI Engine

A high-performance local AI engine with a Go control plane and Rust execution layer.

## WinUI v1 Surface

The packaged WinUI integration path is:

1. WinUI launches the Go server with a config file.
2. The Go supervisor launches and health-checks the Rust daemon.
3. The WinUI app talks to the Go control plane over local gRPC.

For this v1 surface:

- `Runtime` and `RAG` are the supported frontend services.
- `Training` and `MCP` are disabled by default and return `UNIMPLEMENTED` from the Go control plane.
- The daemon is required by default. If no daemon binary or command is available, startup fails fast instead of silently falling back to in-memory managers.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 WinUI / Native Frontend                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              Go Control Plane (gRPC/HTTP API)                │
│  ┌─────────────┐ ┌──────────┐ ┌───────────┐ ┌───────────┐  │
│  │   Runtime   │ │   RAG    │ │ Training  │ │    MCP    │  │
│  │  Manager    │ │ Manager  │ │  Manager  │ │  Manager  │  │
│  └─────────────┘ └──────────┘ └───────────┘ └───────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Rust Execution Layer                       │
│  ┌─────────────┐ ┌────────────┐ ┌──────────┐ ┌──────────┐  │
│  │   Daemon    │ │  Runtime   │ │   RAG    │ │ Storage  │  │
│  │ (tonic)     │ │  Engine    │ │ Pipeline │ │ VectorDB │  │
│  └─────────────┘ └────────────┘ └──────────┘ └──────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Build Binaries

`build.bat` and `build.sh` now bootstrap a pinned official `protoc` automatically if the machine does not already have one on `PATH`.

```bash
cd engine
./build.sh
```

On Windows:

```powershell
cd engine
.\build.bat
```

### 2. Run Server

```bash
./bin/server
# Or with custom config
./bin/server --config ../config.example.yaml
```

### 3. Run Runtime + RAG Demo

```bash
cd engine/go
go run ./cmd/client/main.go
```

## API Services

### Runtime Service
- `GetStatus` - Get runtime status and system resources
- `ListModels` - List available models
- `LoadModel` - Load a model into memory
- `UnloadModel` - Unload a model
- `StreamInference` - Stream backend stdout chunks for a loaded model

### RAG Service
- `UpsertDocument` - Add/update documents
- `DeleteDocument` - Remove a document
- `Search` - Semantic search
- `GetStatus` - Get RAG index status
- `ListDocuments` - List all documents

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
- `CallTool` - Execute a tool

## Configuration

See `config.example.yaml` for configuration options:

- Server host/port (HTTP and gRPC)
- Runtime model path and memory limits
- RAG settings (chunk size, overlap, embedding model)
- Training working directory and job limits
- MCP timeout and retry settings

## Development

### Packaged Smoke Test

`smoke.ps1` is the release-style verification path for the bundled local backend. It:

- builds the server and daemon,
- provisions a temporary local config with ephemeral ports,
- launches the Go server and Rust daemon together,
- runs the `Runtime + RAG` client flow,
- restarts the stack, and
- verifies document persistence after restart.

The smoke path uses a deterministic fake `llama-cli` command so it can validate the runtime contract without depending on an external model runner being preinstalled.

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
│       ├── mcp/        # MCP tool system
│       ├── rag/        # RAG orchestration
│       ├── runtime/    # Model runtime management
│       ├── supervisor/ # Lifecycle management
│       └── training/   # Training orchestration
└── rust/               # Rust execution layer
    └── crates/
        ├── ai_engine_daemon/ # tonic gRPC daemon used in packaged mode
        ├── chunking/   # Text chunking
        ├── embedding/  # Embedding pipeline
        ├── rag_engine/ # RAG orchestration
        ├── runtime_engine/ # Model discovery and CLI-backed inference
        └── storage/    # Vector storage
```

## gRPC API Reference

### Protocol Buffer Definitions

Full API definitions are in `proto/engine.proto`. Key services:

**Runtime**: Model loading, unloading, inference streaming
**RAG**: Document indexing, semantic search
**Training**: Job orchestration, log streaming
**MCP**: Tool execution, connection management

## Environment Variables

- `AI_ENGINE_CONFIG` - Path to config file
- `AI_ENGINE_LOG_LEVEL` - Log level (debug, info, warn, error)
