#!/bin/bash
set -euo pipefail

echo "Building AI Engine..."

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
GO_DIR="$ROOT_DIR/go"
RUST_DIR="$ROOT_DIR/rust"
PROTO_DIR="$ROOT_DIR/proto"

GOEXE="${GOEXE:-go}"
CARGOEXE="${CARGOEXE:-cargo}"
PROTOCEXE="${PROTOCEXE:-${PROTOC:-protoc}}"
PROTOC_GEN_GO="${PROTOC_GEN_GO:-protoc-gen-go}"
PROTOC_GEN_GO_GRPC="${PROTOC_GEN_GO_GRPC:-protoc-gen-go-grpc}"

if [ -x "$HOME/go/bin/protoc-gen-go" ]; then
    PROTOC_GEN_GO="$HOME/go/bin/protoc-gen-go"
fi
if [ -x "$HOME/go/bin/protoc-gen-go-grpc" ]; then
    PROTOC_GEN_GO_GRPC="$HOME/go/bin/protoc-gen-go-grpc"
fi

cd "$GO_DIR"
"$GOEXE" mod download
GOPATH_BIN="$("$GOEXE" env GOPATH)/bin"
export PATH="$GOPATH_BIN:$PATH"

echo "Installing Go proto plugins..."
"$GOEXE" install google.golang.org/protobuf/cmd/protoc-gen-go@latest
"$GOEXE" install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

if [ -f "$PROTO_DIR/engine.proto" ]; then
    echo "Generating proto files..."
    cd "$PROTO_DIR"
    "$PROTOCEXE" -I . \
        --plugin=protoc-gen-go="$PROTOC_GEN_GO" \
        --plugin=protoc-gen-go-grpc="$PROTOC_GEN_GO_GRPC" \
        --go_out=. --go_opt=paths=source_relative \
        --go-grpc_out=. --go-grpc_opt=paths=source_relative \
        engine.proto
    "$PROTOCEXE" -I . \
        --plugin=protoc-gen-go="$PROTOC_GEN_GO" \
        --plugin=protoc-gen-go-grpc="$PROTOC_GEN_GO_GRPC" \
        --go_out=go --go_opt=paths=source_relative \
        --go-grpc_out=go --go-grpc_opt=paths=source_relative \
        engine.proto
fi

mkdir -p "$GO_DIR/bin"

echo "Building Rust daemon..."
cd "$RUST_DIR"
export PROTOC="$PROTOCEXE"
"$CARGOEXE" build --release -p ai_engine_daemon
cp "target/release/ai_engine_daemon" "$GO_DIR/bin/ai_engine_daemon"

echo "Building Rust context service..."
"$CARGOEXE" build --release -p context_server

cd "$GO_DIR"

echo "Building server..."
"$GOEXE" build -o bin/server ./cmd/server

echo "Building client..."
"$GOEXE" build -o bin/client ./cmd/client

echo "Build complete!"
echo "Run: ./go/bin/server --config ./config.example.yaml"
