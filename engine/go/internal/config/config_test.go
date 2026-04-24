package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDefaultConfigSetsLanceDBStorage(t *testing.T) {
	cfg := DefaultConfig()

	if cfg.Storage.LanceDBURI == "" {
		t.Fatal("expected default LanceDB URI to be set")
	}
	if cfg.RAG.StoragePath != cfg.Storage.LanceDBURI {
		t.Fatalf("expected RAG storage path alias to match LanceDB URI, got %q and %q", cfg.RAG.StoragePath, cfg.Storage.LanceDBURI)
	}
	if cfg.Daemon.Addr() == "" {
		t.Fatal("expected daemon address to be derived from config")
	}
	if !cfg.Server.CORS.Enabled {
		t.Fatal("expected frontend CORS defaults to be enabled")
	}
	if len(cfg.Server.CORS.AllowedOrigins) == 0 {
		t.Fatal("expected default frontend CORS origins")
	}
}

func TestLoadParsesServerCORSConfig(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(path, []byte(`
server:
  cors:
    enabled: true
    allowed_origins:
      - "http://localhost:*"
      - "app://ai-engine"
    allowed_headers:
      - "Content-Type"
      - "Authorization"
`), 0644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("load config: %v", err)
	}

	if !cfg.Server.CORS.Enabled {
		t.Fatal("expected CORS to be enabled")
	}
	if got, want := cfg.Server.CORS.AllowedOrigins[1], "app://ai-engine"; got != want {
		t.Fatalf("expected origin %q, got %q", want, got)
	}
	if got, want := cfg.Server.CORS.AllowedHeaders[0], "Content-Type"; got != want {
		t.Fatalf("expected header %q, got %q", want, got)
	}
}

func TestLoadMapsLegacyRagStoragePathToLanceDBURI(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(path, []byte(`
rag:
  storage_path: "C:/tmp/legacy-rag"
`), 0644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("load config: %v", err)
	}

	if cfg.Storage.LanceDBURI != "C:/tmp/legacy-rag" {
		t.Fatalf("expected legacy rag.storage_path to populate storage.lancedb_uri, got %q", cfg.Storage.LanceDBURI)
	}
}

func TestLoadPrefersExplicitLanceDBURIOverLegacyAlias(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(path, []byte(`
storage:
  lancedb_uri: "C:/tmp/lancedb"
rag:
  storage_path: "C:/tmp/legacy-rag"
`), 0644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("load config: %v", err)
	}

	if cfg.Storage.LanceDBURI != "C:/tmp/lancedb" {
		t.Fatalf("expected explicit storage.lancedb_uri to win, got %q", cfg.Storage.LanceDBURI)
	}
}

func TestDetectDaemonCommandFindsLocalBinary(t *testing.T) {
	dir := t.TempDir()
	binDir := filepath.Join(dir, "go", "bin")
	if err := os.MkdirAll(binDir, 0755); err != nil {
		t.Fatalf("mkdir bin: %v", err)
	}

	expected := filepath.Join(binDir, daemonBinaryName())
	if err := os.WriteFile(expected, []byte("stub"), 0644); err != nil {
		t.Fatalf("write daemon binary: %v", err)
	}

	if actual := detectDaemonCommand(dir); actual != expected {
		t.Fatalf("expected daemon command %q, got %q", expected, actual)
	}
}
