# AI Engine — Agent Guide

## Build Commands

- **Windows**: `cd engine && build.bat` (Go deps, proto gen, server + client binaries)
- **Linux/macOS**: `cd engine && ./build.sh`
- **Go server only**: `cd engine/go && go build -o bin/server ./cmd/server`
- **Rust crates**: `cd engine/rust && cargo build --release`
- **Single Rust crate test**: `cd engine/rust && cargo test -p <crate>`
- **Go tests**: `cd engine/go && go test ./... -v`

## Go Dependencies

Key libraries integrated:
- **gin-gonic/gin** — HTTP framework for REST endpoints
- **rs/zerolog** — Structured logging with fluent API
- **golang/x/sync/errgroup** — Concurrent goroutine management with error handling
- **stretchr/testify** — Testing with suite package for setup/teardown
- **golang/mock/mockgen** — Mock generation: `go install github.com/golang/mock/mockgen@latest`

## Proto Codegen

Protoc generates Go stubs. Requires: `go install google.golang.org/protobuf/cmd/protoc-gen-go@latest` and `go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest`

Generated code lives in two places:
- `engine/proto/` (root-level .pb.go files)
- `engine/proto/go/` (sub-module with its own go.mod/go.sum)

## Go Module Quirk

`engine/go/go.mod` has `replace github.com/ai-engine/proto/go => ../proto/go`. Always run `go mod tidy` or `go mod download` from `engine/go/`, not from `engine/proto/`, or the replace directive won't resolve.

## Server Entry Point

`engine/go/cmd/server/main.go` starts the Supervisor, then launches HTTP (`:8080`) and gRPC (`:50051`) servers concurrently. Both ports must be available. Uses zerolog for structured logging and Gin for HTTP routing.

## Config

`--config <path>` flag. Defaults to `config.example.yaml` relative to the working directory.

## Architecture

- **Go layer** (`engine/go/internal/`): API gateway using Gin + zerolog, delegates to sub-managers
- **Rust layer** (`engine/rust/crates/`): `rag_engine` (orchestrates chunking + embedding + storage), `embedding`, `chunking`, `storage`
- **Proto** (`engine/proto/engine.proto`): 4 services — Runtime, RAG, Training, MCP

## gRPC Services

Runtime, RAG, Training, and MCP stub types are in `github.com/ai-engine/proto/go`. The client at `engine/go/cmd/client/main.go` shows all four service calls.