#!/bin/bash
set -euo pipefail

VERSION="34.1"

matches_version() {
    local candidate="$1"
    [ -x "$candidate" ] || return 1
    [ "$("$candidate" --version 2>/dev/null || true)" = "libprotoc $VERSION" ]
}

if [ -n "${PROTOC:-}" ] && matches_version "${PROTOC}"; then
    printf '%s\n' "${PROTOC}"
    exit 0
fi

if command -v protoc >/dev/null 2>&1; then
    EXISTING_PROTOC="$(command -v protoc)"
    if matches_version "$EXISTING_PROTOC"; then
        printf '%s\n' "$EXISTING_PROTOC"
        exit 0
    fi
fi

ENGINE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"

case "$OS_NAME" in
  Linux) OS_PART="linux" ;;
  Darwin) OS_PART="osx" ;;
  *)
    echo "Unsupported OS for protoc bootstrap: $OS_NAME" >&2
    exit 1
    ;;
esac

case "$ARCH_NAME" in
  x86_64|amd64) ARCH_PART="x86_64" ;;
  aarch64|arm64) ARCH_PART="aarch_64" ;;
  *)
    echo "Unsupported architecture for protoc bootstrap: $ARCH_NAME" >&2
    exit 1
    ;;
esac

TOOL_DIR="$ENGINE_ROOT/.tools/protoc/$VERSION/$OS_PART-$ARCH_PART"
BIN_PATH="$TOOL_DIR/bin/protoc"

if [ ! -x "$BIN_PATH" ]; then
  mkdir -p "$TOOL_DIR"
  ARCHIVE_PATH="$TOOL_DIR/protoc-$VERSION-$OS_PART-$ARCH_PART.zip"
  URL="https://github.com/protocolbuffers/protobuf/releases/download/v$VERSION/protoc-$VERSION-$OS_PART-$ARCH_PART.zip"

  echo "Downloading protoc $VERSION for $OS_PART-$ARCH_PART..." >&2
  curl -fsSL "$URL" -o "$ARCHIVE_PATH"
  python3 - <<'PY' "$ARCHIVE_PATH" "$TOOL_DIR"
import sys
from zipfile import ZipFile

archive_path, target_dir = sys.argv[1], sys.argv[2]
with ZipFile(archive_path) as archive:
    archive.extractall(target_dir)
PY
  chmod +x "$BIN_PATH"
fi

printf '%s\n' "$BIN_PATH"
