package config

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Server   ServerConfig   `yaml:"server"`
	Daemon   DaemonConfig   `yaml:"daemon"`
	Storage  StorageConfig  `yaml:"storage"`
	Runtime  RuntimeConfig  `yaml:"runtime"`
	RAG      RAGConfig      `yaml:"rag"`
	Training TrainingConfig `yaml:"training"`
	MCP      MCPConfig      `yaml:"mcp"`
	Logging  LoggingConfig  `yaml:"logging"`
}

type ServerConfig struct {
	Host string `yaml:"host"`
	Port int    `yaml:"port"`
	GRPC struct {
		Host string `yaml:"host"`
		Port int    `yaml:"port"`
	} `yaml:"grpc"`
}

type RuntimeConfig struct {
	ModelsPath string           `yaml:"models_path"`
	MaxMemory  int64            `yaml:"max_memory_mb"`
	Providers  []ProviderConfig `yaml:"providers"`
}

type DaemonConfig struct {
	Host           string        `yaml:"host"`
	Port           int           `yaml:"port"`
	Command        string        `yaml:"command"`
	Args           []string      `yaml:"args"`
	StartupTimeout time.Duration `yaml:"startup_timeout"`
	RestartBackoff time.Duration `yaml:"restart_backoff"`
	ReadyTimeout   time.Duration `yaml:"ready_timeout"`
	LlamaCLI       string        `yaml:"llama_cli"`
	TrainingCLI    string        `yaml:"training_cli"`
}

type StorageConfig struct {
	LanceDBURI         string `yaml:"lancedb_uri"`
	EnableFTS          bool   `yaml:"enable_fts"`
	EnableHybridSearch bool   `yaml:"enable_hybrid_search"`
}

type ProviderConfig struct {
	Name   string `yaml:"name"`
	Type   string `yaml:"type"`
	URL    string `yaml:"url"`
	APIKey string `yaml:"api_key"`
}

type RAGConfig struct {
	StoragePath    string `yaml:"storage_path"`
	EmbeddingModel string `yaml:"embedding_model"`
	ChunkSize      int    `yaml:"chunk_size"`
	ChunkOverlap   int    `yaml:"chunk_overlap"`
	TopK           int    `yaml:"top_k"`
}

type TrainingConfig struct {
	WorkingDir string `yaml:"working_dir"`
	MaxJobs    int    `yaml:"max_concurrent_jobs"`
}

type MCPConfig struct {
	Timeout time.Duration `yaml:"timeout"`
	Retries int           `yaml:"retries"`
}

type LoggingConfig struct {
	Level  string `yaml:"level"`
	Format string `yaml:"format"`
}

func DefaultConfig() *Config {
	homeDir, _ := os.UserHomeDir()
	engineDir := filepath.Join(homeDir, ".ai-engine")

	return &Config{
		Server: ServerConfig{
			Host: "127.0.0.1",
			Port: 8080,
			GRPC: struct {
				Host string `yaml:"host"`
				Port int    `yaml:"port"`
			}{
				Host: "127.0.0.1",
				Port: 50051,
			},
		},
		Daemon: DaemonConfig{
			Host:           "127.0.0.1",
			Port:           50061,
			Command:        defaultDaemonCommand(),
			Args:           []string{},
			StartupTimeout: 15 * time.Second,
			RestartBackoff: 3 * time.Second,
			ReadyTimeout:   10 * time.Second,
			LlamaCLI:       "llama-cli",
			TrainingCLI:    "llama-train",
		},
		Storage: StorageConfig{
			LanceDBURI:         filepath.Join(engineDir, "lancedb"),
			EnableFTS:          true,
			EnableHybridSearch: true,
		},
		Runtime: RuntimeConfig{
			ModelsPath: filepath.Join(engineDir, "models"),
			MaxMemory:  8192,
			Providers:  []ProviderConfig{},
		},
		RAG: RAGConfig{
			StoragePath:    filepath.Join(engineDir, "lancedb"),
			EmbeddingModel: "sentence-transformers/all-MiniLM-L6-v2",
			ChunkSize:      512,
			ChunkOverlap:   50,
			TopK:           10,
		},
		Training: TrainingConfig{
			WorkingDir: filepath.Join(engineDir, "training"),
			MaxJobs:    2,
		},
		MCP: MCPConfig{
			Timeout: 30 * time.Second,
			Retries: 3,
		},
		Logging: LoggingConfig{
			Level:  "info",
			Format: "json",
		},
	}
}

func (c *Config) Addr() string {
	return fmt.Sprintf("%s:%d", c.Server.Host, c.Server.Port)
}

func (c *Config) GRPCAddr() string {
	return fmt.Sprintf("%s:%d", c.Server.GRPC.Host, c.Server.GRPC.Port)
}

func (c *Config) applyCompatAliases() {
	switch {
	case c.Storage.LanceDBURI == "" && c.RAG.StoragePath != "":
		c.Storage.LanceDBURI = c.RAG.StoragePath
	case c.Storage.LanceDBURI != "" && c.RAG.StoragePath == "":
		c.RAG.StoragePath = c.Storage.LanceDBURI
	}
}

func Load(path string) (*Config, error) {
	if path == "" {
		cfg := DefaultConfig()
		cfg.applyCompatAliases()
		return cfg, nil
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config: %w", err)
	}

	var source struct {
		Storage struct {
			LanceDBURI string `yaml:"lancedb_uri"`
		} `yaml:"storage"`
		RAG struct {
			StoragePath string `yaml:"storage_path"`
		} `yaml:"rag"`
	}
	if err := yaml.Unmarshal(data, &source); err != nil {
		return nil, fmt.Errorf("failed to parse config source: %w", err)
	}

	cfg := DefaultConfig()
	if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, fmt.Errorf("failed to parse config: %w", err)
	}
	if cfg.Daemon.Command == "" {
		cfg.Daemon.Command = defaultDaemonCommand()
	}
	if source.Storage.LanceDBURI == "" && source.RAG.StoragePath != "" {
		cfg.Storage.LanceDBURI = source.RAG.StoragePath
	}
	cfg.applyCompatAliases()

	return cfg, nil
}

func (c *Config) EnsureDirs() error {
	dirs := []string{
		c.Runtime.ModelsPath,
		c.Training.WorkingDir,
	}
	if isLocalPath(c.Storage.LanceDBURI) {
		dirs = append(dirs, c.Storage.LanceDBURI)
	}

	for _, dir := range dirs {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return fmt.Errorf("failed to create directory %s: %w", dir, err)
		}
	}

	return nil
}

func (c DaemonConfig) Addr() string {
	return fmt.Sprintf("%s:%d", c.Host, c.Port)
}

func isLocalPath(uri string) bool {
	if uri == "" {
		return false
	}
	return !containsScheme(uri)
}

func containsScheme(value string) bool {
	for i := 0; i < len(value)-2; i++ {
		if value[i] == ':' && value[i+1] == '/' && value[i+2] == '/' {
			return true
		}
	}
	return false
}

func defaultDaemonCommand() string {
	wd, err := os.Getwd()
	if err != nil {
		return ""
	}
	return detectDaemonCommand(wd)
}

func detectDaemonCommand(root string) string {
	candidates := []string{
		filepath.Join(root, "bin", daemonBinaryName()),
		filepath.Join(root, "go", "bin", daemonBinaryName()),
		filepath.Join(root, "rust", "target", "debug", daemonBinaryName()),
		filepath.Join(root, "rust", "target", "release", daemonBinaryName()),
		filepath.Join(root, "..", "rust", "target", "debug", daemonBinaryName()),
		filepath.Join(root, "..", "rust", "target", "release", daemonBinaryName()),
	}

	for _, candidate := range candidates {
		info, err := os.Stat(candidate)
		if err == nil && !info.IsDir() {
			return candidate
		}
	}

	return ""
}

func daemonBinaryName() string {
	if runtime.GOOS == "windows" {
		return "ai_engine_daemon.exe"
	}
	return "ai_engine_daemon"
}
