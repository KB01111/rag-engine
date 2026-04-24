FROM rust:1-bookworm AS rust-builder
WORKDIR /src

RUN apt-get update \
    && apt-get install -y --no-install-recommends protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

COPY rust ./rust
COPY proto ./proto

WORKDIR /src/rust
RUN cargo build --release -p ai_engine_daemon
RUN cargo build --release -p context_server

FROM golang:1.25-bookworm AS go-builder
WORKDIR /src

COPY go ./go
COPY proto ./proto

WORKDIR /src/go
RUN go mod download && \
    go build -o /out/server ./cmd/server && \
    go build -o /out/client ./cmd/client

FROM debian:bookworm-slim
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r aiengine && useradd -r -g aiengine aiengine

COPY --from=go-builder /out/server /app/bin/server
COPY --from=go-builder /out/client /app/bin/client
COPY --from=rust-builder /src/rust/target/release/ai_engine_daemon /app/bin/ai_engine_daemon
COPY --from=rust-builder /src/rust/target/release/context_server /app/bin/context_server
COPY config.example.yaml /app/config.example.yaml

RUN chown -R aiengine:aiengine /app

USER aiengine

EXPOSE 8080 50051

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

CMD ["/app/bin/server", "--config", "/app/config.example.yaml"]
