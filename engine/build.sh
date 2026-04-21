#!/bin/bash
set -e

echo "Building AI Engine..."

# Setup Go modules
cd "$(dirname "$0")/go"
go mod download

# Generate proto files
if [ -f "../proto/engine.proto" ]; then
    echo "Generating proto files..."
    cd ../proto
    protoc --go_out=. --go_opt=paths=source_relative \
           --go-grpc_out=. --go-grpc_opt=paths=source_relative \
           engine.proto
    cd ../go
fi

# Build server
echo "Building server..."
mkdir -p bin
go build -o bin/server ./cmd/server

# Build client
echo "Building client..."
go build -o bin/client ./cmd/client

echo "Build complete!"
echo "Run: ./bin/server"