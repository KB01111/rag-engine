package config

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Server      ServerConfig      `yaml:"server"`
	Daemon      DaemonConfig      `yaml:"daemon"`
	Services    ServicesConfig    `yaml:"services"`
	Storage     StorageConfig     `yaml:"storage"`
	Runtime     RuntimeConfig     `yaml:"runtime"`
	Context     ContextConfig     `yaml:"context"`
	RAG         RAGConfig         `yaml:"rag"`
	Training    TrainingConfig    `yaml:"training"`
	MCP         MCPConfig         `yaml:"mcp"`
	HuggingFace HuggingFaceConfig `yaml:"huggingface"`
	Logging     LoggingConfig     `yaml:"logging"`
}

type ServerConfig struct {
	Host string `yaml:"host"`
	Port int    `yaml:"port"`
	Mode string `yaml:"mode"`
	GRPC struct {
		Host string `yaml:"host"`
		Port int    `yaml:"port"`
	} `yaml:"grpc"`
	CORS CORSConfig `yaml:"cors"`
}

type CORSConfig struct {
	Enabled        bool     `yaml:"enabled"`
	AllowedOrigins []string `yaml:"allowed_origins"`
	AllowedHeaders []string `yaml:"allowed_headers"`
}

type RuntimeConfig struct {
	ModelsPath string           `yaml:"models_path"`
	Backend    string           `yaml:"backend"`
	Providers  []ProviderConfig `yaml:"providers"`
	MistralRS  MistralRSConfig  `yaml:"mistralrs"`
}

type HuggingFaceConfig struct {
	Enabled              bool     `yaml:"enabled"`
	Endpoint             string   `yaml:"endpoint"`
	MaxDownloadBytes     int64    `yaml:"max_download_bytes"`
	CompatibleExtensions []string `yaml:"compatible_extensions"`
}

type MistralRSConfig struct {
	ForceCPU   bool   `yaml:"force_cpu"`
	MaxNumSeqs int    `yaml:"max_num_seqs"`
	AutoISQ    string `yaml:"auto_isq"`
}

type DaemonConfig struct {
	Host           string        `yaml:"host"`
	Port           int           `yaml:"port"`
	Required       bool          `yaml:"required"`
	Command        string        `yaml:"command"`
	Args           []string      `yaml:"args"`
	StartupTimeout time.Duration `yaml:"startup_timeout"`
	RestartBackoff time.Duration `yaml:"restart_backoff"`
	ReadyTimeout   time.Duration `yaml:"ready_timeout"`
	LlamaCLI       string        `yaml:"llama_cli"`
	TrainingCLI    string        `yaml:"training_cli"`
}

type ServicesConfig struct {
	EnableTraining bool `yaml:"enable_training"`
	EnableMCP      bool `yaml:"enable_mcp"`
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
	StoragePath            string `yaml:"storage_path"`
	EmbeddingProvider      string `yaml:"embedding_provider"`
	EmbeddingModel         string `yaml:"embedding_model"`
	EmbeddingCacheDir      string `yaml:"embedding_cache_dir"`
	EmbeddingAllowDownload bool   `yaml:"embedding_allow_download"`
	ChunkSize              int    `yaml:"chunk_size"`
	ChunkOverlap           int    `yaml:"chunk_overlap"`
	TopK                   int    `yaml:"top_k"`
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
			Mode: "development",
			GRPC: struct {
				Host string `yaml:"host"`
				Port int    `yaml:"port"`
			}{
				Host: "127.0.0.1",
				Port: 50051,
			},
			CORS: CORSConfig{
				Enabled: true,
				AllowedOrigins: []string{
					"http://localhost:*",
					"http://127.0.0.1:*",
					"app://ai-engine",
				},
				AllowedHeaders: []string{
					"Content-Type",
					"Authorization",
				},
			},
		},
		Daemon: DaemonConfig{
			Host:           "127.0.0.1",
			Port:           50061,
			Required:       true,
			Command:        defaultDaemonCommand(),
			Args:           []string{},
			StartupTimeout: 15 * time.Second,
			RestartBackoff: 3 * time.Second,
			ReadyTimeout:   10 * time.Second,
			LlamaCLI:       "llama-cli",
			TrainingCLI:    "llama-train",
		},
		Services: ServicesConfig{
			EnableTraining: false,
			EnableMCP:      false,
		},
		Storage: StorageConfig{
			LanceDBURI:         filepath.Join(engineDir, "lancedb"),
			EnableFTS:          true,
			EnableHybridSearch: true,
		},
		Runtime: RuntimeConfig{
			ModelsPath: filepath.Join(engineDir, "models"),
			Backend:    "mistralrs",
			Providers:  []ProviderConfig{},
			MistralRS: MistralRSConfig{
				ForceCPU:   false,
				MaxNumSeqs: 32,
				AutoISQ:    "",
			},
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
			StoragePath:            filepath.Join(engineDir, "lancedb"),
			EmbeddingProvider:      "fastembed",
			EmbeddingModel:         "sentence-transformers/all-MiniLM-L6-v2",
			EmbeddingCacheDir:      filepath.Join(engineDir, "embedding-cache"),
			EmbeddingAllowDownload: true,
			ChunkSize:              512,
			ChunkOverlap:           50,
			TopK:                   10,
		},
		Training: TrainingConfig{
			WorkingDir: filepath.Join(engineDir, "training"),
			MaxJobs:    2,
		},
		MCP: MCPConfig{
			Timeout: 30 * time.Second,
			Retries: 3,
		},
		HuggingFace: HuggingFaceConfig{
			Enabled:          true,
			Endpoint:         "https://huggingface.co",
			MaxDownloadBytes: 0,
			CompatibleExtensions: []string{
				".gguf",
				".ggml",
				".bin",
			},
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

func (c *Config) IsProduction() bool {
	return normalizeServerMode(c.Server.Mode) == "production"
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

func (c *Config) validate() error {
	// Validate runtime.backend
	validBackends := []string{"mistralrs", "mock"}
	backendValid := false
	normalizedBackend := normalizeBackendName(c.Runtime.Backend)
	for _, valid := range validBackends {
		if normalizedBackend == valid {
			backendValid = true
			break
		}
	}
	if !backendValid {
		return fmt.Errorf("invalid runtime.backend %q: must be one of %v", c.Runtime.Backend, validBackends)
	}
	// Persist the normalized backend name
	c.Runtime.Backend = normalizedBackend

	// Validate MistralRS.MaxNumSeqs
	if c.Runtime.MistralRS.MaxNumSeqs <= 0 {
		return fmt.Errorf("invalid runtime.mistralrs.max_num_seqs %d: must be greater than 0", c.Runtime.MistralRS.MaxNumSeqs)
	}

	validEmbeddingProviders := []string{"fastembed", "mock"}
	embeddingProviderValid := false
	normalizedEmbeddingProvider := normalizeEmbeddingProvider(c.RAG.EmbeddingProvider)
	for _, valid := range validEmbeddingProviders {
		if normalizedEmbeddingProvider == valid {
			embeddingProviderValid = true
			break
		}
	}
	if !embeddingProviderValid {
		return fmt.Errorf("invalid rag.embedding_provider %q: must be one of %v", c.RAG.EmbeddingProvider, validEmbeddingProviders)
	}
	c.RAG.EmbeddingProvider = normalizedEmbeddingProvider
	if strings.TrimSpace(c.RAG.EmbeddingModel) == "" {
		return fmt.Errorf("invalid rag.embedding_model: must not be empty")
	}

	return nil
}

func normalizeBackendName(name string) string {
	normalized := strings.ToLower(strings.TrimSpace(name))
	switch normalized {
	case "mistral.rs", "mistral_rs", "mistral-rs":
		return "mistralrs"
	case "":
		return "mistralrs"
	default:
		return normalized
	}
}

func normalizeEmbeddingProvider(name string) string {
	normalized := strings.ToLower(strings.TrimSpace(name))
	switch normalized {
	case "":
		return "fastembed"
	case "fast-embed", "fast_embed":
		return "fastembed"
	default:
		return normalized
	}
}

func Load(path string) (*Config, error) {
	if path == "" {
		cfg := DefaultConfig()
		cfg.applyCompatAliases()
		if err := cfg.validate(); err != nil {
			return nil, err
		}
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

	// Expand ~ in local paths
	cfg.Storage.LanceDBURI = expandPath(cfg.Storage.LanceDBURI)
	cfg.Runtime.ModelsPath = expandPath(cfg.Runtime.ModelsPath)
	cfg.Context.BinaryPath = expandPath(cfg.Context.BinaryPath)
	cfg.Context.DataDir = expandPath(cfg.Context.DataDir)
	cfg.Training.WorkingDir = expandPath(cfg.Training.WorkingDir)
	cfg.RAG.StoragePath = expandPath(cfg.RAG.StoragePath)
	cfg.RAG.EmbeddingCacheDir = expandPath(cfg.RAG.EmbeddingCacheDir)
	cfg.Daemon.Command = expandPath(cfg.Daemon.Command)

	if cfg.Daemon.Command == "" {
		cfg.Daemon.Command = defaultDaemonCommand()
	}
	cfg.Server.Mode = normalizeServerMode(cfg.Server.Mode)
	if source.Storage.LanceDBURI == "" && source.RAG.StoragePath != "" {
		cfg.Storage.LanceDBURI = source.RAG.StoragePath
	}
	cfg.applyCompatAliases()

	if err := cfg.validate(); err != nil {
		return nil, err
	}

	return cfg, nil
}

func (c *Config) EnsureDirs() error {
	dirs := make([]string, 0, 5)
	seen := make(map[string]struct{}, 5)
	appendDir := func(dir string) {
		if dir == "" {
			return
		}
		if _, ok := seen[dir]; ok {
			return
		}
		seen[dir] = struct{}{}
		dirs = append(dirs, dir)
	}

	appendDir(c.Runtime.ModelsPath)
	appendDir(c.Context.DataDir)
	appendDir(c.Training.WorkingDir)
	appendDir(c.RAG.EmbeddingCacheDir)
	if isLocalPath(c.Storage.LanceDBURI) {
		appendDir(c.Storage.LanceDBURI)
	}
	if isLocalPath(c.RAG.StoragePath) {
		appendDir(c.RAG.StoragePath)
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
	return detectDaemonCommand(searchRoots()...)
}

func detectDaemonCommand(roots ...string) string {
	binary := daemonBinaryName()
	for _, root := range roots {
		if root == "" {
			continue
		}

		for _, candidate := range []string{
			filepath.Join(root, binary),
			filepath.Join(root, "bin", binary),
			filepath.Join(root, "go", "bin", binary),
			filepath.Join(root, "rust", "target", "debug", binary),
			filepath.Join(root, "rust", "target", "release", binary),
			filepath.Join(root, "..", "rust", "target", "debug", binary),
			filepath.Join(root, "..", "rust", "target", "release", binary),
		} {
			info, err := os.Stat(candidate)
			if err == nil && !info.IsDir() {
				return candidate
			}
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

func expandPath(path string) string {
	if path == "" || !containsScheme(path) {
		if len(path) > 0 && path[0] == '~' {
			homeDir, err := os.UserHomeDir()
			if err == nil {
				if len(path) == 1 {
					return homeDir
				}
				if path[1] == '/' || path[1] == '\\' {
					return filepath.Join(homeDir, path[2:])
				}
			}
		}
	}
	return path
}

func normalizeServerMode(mode string) string {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "production":
		return "production"
	default:
		return "development"
	}
}

func searchRoots() []string {
	roots := make([]string, 0, 2)
	if exe, err := os.Executable(); err == nil {
		roots = append(roots, filepath.Dir(exe))
	}
	if wd, err := os.Getwd(); err == nil {
		roots = append(roots, wd)
	}
	return roots
}
