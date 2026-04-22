#!/bin/bash
set -e

echo "Building AI Engine..."

GOEXE="${GOEXE:-go}"
CARGOEXE="${CARGOEXE:-cargo}"
PROTOCEXE="${PROTOCEXE:-protoc}"
PROTOC_GEN_GO="${PROTOC_GEN_GO:-protoc-gen-go}"
PROTOC_GEN_GO_GRPC="${PROTOC_GEN_GO_GRPC:-protoc-gen-go-grpc}"
if [ -x "../.tools/protoc-34.1-win64/bin/protoc.exe" ]; then
    PROTOCEXE="../.tools/protoc-34.1-win64/bin/protoc.exe"
fi
if [ -x "$HOME/go/bin/protoc-gen-go" ]; then
    PROTOC_GEN_GO="$HOME/go/bin/protoc-gen-go"
fi
if [ -x "$HOME/go/bin/protoc-gen-go-grpc" ]; then
    PROTOC_GEN_GO_GRPC="$HOME/go/bin/protoc-gen-go-grpc"
fi

# Setup Go modules
cd "$(dirname "$0")/go"
"$GOEXE" mod download

# Generate proto files
if [ -f "../proto/engine.proto" ]; then
    echo "Generating proto files..."
    cd ../proto
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
    cd ../go
fi

# Build server
echo "Building server..."
mkdir -p bin
"$GOEXE" build -o bin/server ./cmd/server

# Build client
echo "Building client..."
"$GOEXE" build -o bin/client ./cmd/client

# Build Rust context service
echo "Building Rust context service..."
cd ../rust
if [ -x "../.tools/protoc-34.1-win64/bin/protoc.exe" ]; then
    export PROTOC="$(pwd)/../.tools/protoc-34.1-win64/bin/protoc.exe"
fi
"$CARGOEXE" build --release -p context_server

echo "Build complete!"
echo "Run: ./go/bin/server"
