FROM rust:1-bookworm AS rust-builder
WORKDIR /src

RUN apt-get update \
    && apt-get install -y --no-install-recommends protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

COPY rust ./rust
COPY proto ./proto

WORKDIR /src/rust
RUN cargo build --release -p ai_engine_daemon

FROM golang:1.25-bookworm AS go-builder
WORKDIR /src

COPY go ./go
COPY proto ./proto

WORKDIR /src/go
RUN go mod download
RUN go build -o /out/server ./cmd/server
RUN go build -o /out/client ./cmd/client

FROM debian:bookworm-slim
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=go-builder /out/server /app/bin/server
COPY --from=go-builder /out/client /app/bin/client
COPY --from=rust-builder /src/rust/target/release/ai_engine_daemon /app/bin/ai_engine_daemon
COPY config.example.yaml /app/config.example.yaml

EXPOSE 8080 50051

CMD ["/app/bin/server"]
