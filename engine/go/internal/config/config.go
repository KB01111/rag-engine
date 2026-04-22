package config

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Server   ServerConfig   `yaml:"server"`
	Runtime  RuntimeConfig  `yaml:"runtime"`
	Context  ContextConfig  `yaml:"context"`
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

type ProviderConfig struct {
	Name   string `yaml:"name"`
	Type   string `yaml:"type"`
	URL    string `yaml:"url"`
	APIKey string `yaml:"api_key"`
}

type ContextConfig struct {
	Enabled        bool             `yaml:"enabled"`
	ServiceURL     string           `yaml:"service_url"`
	BinaryPath     string           `yaml:"binary_path"`
	DataDir        string           `yaml:"data_dir"`
	AutoStart      bool             `yaml:"auto_start"`
	StartupTimeout time.Duration    `yaml:"startup_timeout"`
	ManagedRoots   []string         `yaml:"managed_roots"`
	OpenViking     OpenVikingConfig `yaml:"openviking"`
}

type OpenVikingConfig struct {
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
		Runtime: RuntimeConfig{
			ModelsPath: filepath.Join(engineDir, "models"),
			MaxMemory:  8192,
			Providers:  []ProviderConfig{},
		},
		Context: ContextConfig{
			Enabled:        false,
			ServiceURL:     "http://127.0.0.1:9191",
			BinaryPath:     "",
			DataDir:        filepath.Join(engineDir, "context"),
			AutoStart:      false,
			StartupTimeout: 20 * time.Second,
			ManagedRoots:   []string{},
			OpenViking:     OpenVikingConfig{},
		},
		RAG: RAGConfig{
			StoragePath:    filepath.Join(engineDir, "rag"),
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

func Load(path string) (*Config, error) {
	if path == "" {
		return DefaultConfig(), nil
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config: %w", err)
	}

	cfg := DefaultConfig()
	if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, fmt.Errorf("failed to parse config: %w", err)
	}

	return cfg, nil
}

func (c *Config) EnsureDirs() error {
	dirs := []string{
		c.Runtime.ModelsPath,
		c.Context.DataDir,
		c.RAG.StoragePath,
		c.Training.WorkingDir,
	}

	for _, dir := range dirs {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return fmt.Errorf("failed to create directory %s: %w", dir, err)
		}
	}

	return nil
}
