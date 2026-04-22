#!/bin/bash
set -e

echo "Building AI Engine..."

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
GO_DIR="$ROOT_DIR/go"
RUST_DIR="$ROOT_DIR/rust"

if [ -z "${PROTOC:-}" ] && command -v protoc >/dev/null 2>&1; then
    PROTOC_PATH="$(command -v protoc)"
    export PROTOC="$PROTOC_PATH"
fi
PROTOC_BIN="${PROTOC:-protoc}"

# Setup Go modules
cd "$GO_DIR"
go mod download
GOPATH_BIN="$(go env GOPATH)/bin"
export PATH="$GOPATH_BIN:$PATH"

echo "Installing Go proto plugins..."
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

# Generate proto files
if [ -f "../proto/engine.proto" ]; then
    echo "Generating proto files..."
    cd ../proto
    "$PROTOC_BIN" --go_out=. --go_opt=paths=source_relative \
                  --go-grpc_out=. --go-grpc_opt=paths=source_relative \
                  engine.proto
    cd ../go
fi

mkdir -p bin

# Build Rust daemon
echo "Building Rust daemon..."
cd "$RUST_DIR"
cargo build --release -p ai_engine_daemon
cp "target/release/ai_engine_daemon" "$GO_DIR/bin/ai_engine_daemon"

cd "$GO_DIR"

# Build server
echo "Building server..."
go build -o bin/server ./cmd/server

# Build client
echo "Building client..."
go build -o bin/client ./cmd/client

echo "Build complete!"
echo "Run: ./bin/server --config ../config.example.yaml"