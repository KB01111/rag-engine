# AI Engine

A high-performance local AI engine with a Go control plane and Rust execution layer.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Electron (Renderer)                      │
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
│  │   RAG       │ │ Embedding  │ │ Chunking │ │ Storage  │  │
│  │   Engine    │ │   Pipeline │ │   Text   │ │ VectorDB │  │
│  └─────────────┘ └────────────┘ └──────────┘ └──────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Generate Proto Files

```bash
# Install protoc and Go plugins
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

# Generate Go code
cd engine/proto
protoc --go_out=../go --go_opt=paths=source_relative \
       --go-grpc_out=../go --go-grpc_opt=paths=source_relative \
       engine.proto
```

### 2. Build Go Server

```bash
cd engine/go
go mod tidy
go build -o bin/server ./cmd/server
```

### 3. Run Server

```bash
./bin/server
# Or with custom config
./bin/server --config ../config.example.yaml
```

### 4. Build Rust Crates

```bash
cd engine/rust
cargo build --release
```

### 5. Run Client Demo

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
- `StreamInference` - Stream inference responses

### RAG Service
- `UpsertDocument` - Add/update documents
- `DeleteDocument` - Remove a document
- `Search` - Semantic search
- `GetStatus` - Get RAG index status
- `ListDocuments` - List all documents

### Training Service
- `StartRun` - Start a training job
- `CancelRun` - Cancel a running job
- `ListRuns` - List all training runs
- `ListArtifacts` - List training artifacts
- `StreamLogs` - Stream training logs

### MCP Service
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
        ├── chunking/   # Text chunking
        ├── embedding/  # Embedding pipeline
        ├── rag_engine/ # RAG orchestration
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