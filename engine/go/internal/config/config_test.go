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
	if cfg.Runtime.Backend != "mistralrs" {
		t.Fatalf("expected runtime backend to default to mistralrs, got %q", cfg.Runtime.Backend)
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

func TestLoadRuntimeBackendAndMistralRSOptions(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(path, []byte(`
runtime:
  backend: "mock"
  mistralrs:
    force_cpu: true
    max_num_seqs: 4
    auto_isq: "q4"
`), 0644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("load config: %v", err)
	}

	if cfg.Runtime.Backend != "mock" {
		t.Fatalf("expected runtime backend mock, got %q", cfg.Runtime.Backend)
	}
	if !cfg.Runtime.MistralRS.ForceCPU {
		t.Fatal("expected runtime.mistralrs.force_cpu to load")
	}
	if cfg.Runtime.MistralRS.MaxNumSeqs != 4 {
		t.Fatalf("expected runtime.mistralrs.max_num_seqs 4, got %d", cfg.Runtime.MistralRS.MaxNumSeqs)
	}
	if cfg.Runtime.MistralRS.AutoISQ != "q4" {
		t.Fatalf("expected runtime.mistralrs.auto_isq q4, got %q", cfg.Runtime.MistralRS.AutoISQ)
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
