package supervisor

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/ai-engine/go/internal/config"
	contextsvc "github.com/ai-engine/go/internal/contextsvc"
	"github.com/ai-engine/go/internal/daemon"
	"github.com/ai-engine/go/internal/mcp"
	"github.com/ai-engine/go/internal/rag"
	"github.com/ai-engine/go/internal/runtime"
	"github.com/ai-engine/go/internal/training"
	"github.com/rs/zerolog/log"
	stdlog "log"
)

type Supervisor struct {
	mu      sync.RWMutex
	running bool
	ctx     context.Context
	cancel  context.CancelFunc
	config  *config.Config
	mode    string

	Context  *contextsvc.Manager
	Runtime  runtime.Service
	RAG      rag.Service
	Training training.Service
	MCP      mcp.Service

	daemonClient *daemon.Client
	daemonCmd    *exec.Cmd

	wg sync.WaitGroup
}

func NewSupervisor(cfg *config.Config) *Supervisor {
	ctx, cancel := context.WithCancel(context.Background())
	contextManager := contextsvc.NewManager(contextsvc.Config{
		Enabled:          cfg.Context.Enabled,
		BaseURL:          cfg.Context.ServiceURL,
		BinaryPath:       cfg.Context.BinaryPath,
		DataDir:          cfg.Context.DataDir,
		AutoStart:        cfg.Context.AutoStart,
		StartupTimeout:   cfg.Context.StartupTimeout,
		ManagedRoots:     cfg.Context.ManagedRoots,
		OpenVikingURL:    cfg.Context.OpenViking.URL,
		OpenVikingAPIKey: cfg.Context.OpenViking.APIKey,
	})
	sup := &Supervisor{
		ctx:     ctx,
		cancel:  cancel,
		config:  cfg,
		Context: contextManager,
	}
	sup.initLocalServicesLocked()
	return sup
}

func (s *Supervisor) Start() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.running {
		return fmt.Errorf("supervisor already running")
	}
	if s.config.IsProduction() {
		if !s.config.Context.Enabled {
			return fmt.Errorf("context backend is required in production mode")
		}
		if s.config.Daemon.Command == "" {
			return fmt.Errorf("daemon backend is required in production mode")
		}
	}

	if err := s.config.EnsureDirs(); err != nil {
		return fmt.Errorf("failed to ensure directories: %w", err)
	}

	daemonLaunched := false
	if s.config.Daemon.Command != "" {
		if err := s.launchDaemonLocked(); err != nil {
			if s.config.Daemon.Required || s.config.IsProduction() {
				return fmt.Errorf("failed to launch required daemon: %w", err)
			}
			log.Error().Err(err).Msg("failed to launch optional daemon, falling back to local services")
			s.initLocalServicesLocked()
		} else {
			daemonLaunched = true
		}
	} else {
		if s.config.Daemon.Required {
			return fmt.Errorf("daemon is required but no command or binary was found")
		}
		s.initLocalServicesLocked()
	}

	if s.config.Context.Enabled {
		if daemonLaunched {
			log.Info().Msg("context backend startup skipped; daemon is acting as context client")
		} else {
			if err := s.Context.Start(s.ctx); err != nil {
				if stopErr := s.Context.Stop(context.Background()); stopErr != nil {
					log.Error().Err(stopErr).Msg("failed to stop context backend after startup error")
				}
				return fmt.Errorf("failed to start context backend: %w", err)
			}
		}
	}

	s.running = true

	stdlog.Printf("AI Engine supervisor started")

	s.wg.Add(1)
	go s.handleSignals()

	return nil
}

func (s *Supervisor) Stop() error {
	s.mu.Lock()
	if !s.running {
		s.mu.Unlock()
		return fmt.Errorf("supervisor not running")
	}

	s.cancel()
	if s.daemonClient != nil {
		_ = s.daemonClient.Close()
		s.daemonClient = nil
	}
	s.mu.Unlock()
	if err := s.Context.Stop(context.Background()); err != nil {
		stdlog.Printf("failed to stop context backend: %v", err)
	}
	s.wg.Wait()

	s.mu.Lock()
	defer s.mu.Unlock()
	s.running = false

	stdlog.Printf("AI Engine supervisor stopped")
	return nil
}

func (s *Supervisor) IsRunning() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.running
}

func (s *Supervisor) handleSignals() {
	defer s.wg.Done()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	select {
	case sig := <-sigCh:
		stdlog.Printf("Received signal: %v", sig)
		s.cancel()
	case <-s.ctx.Done():
	}
}

func (s *Supervisor) Health() map[string]interface{} {
	s.mu.RLock()
	running := s.running
	contextManager := s.Context
	runtimeSvc := s.Runtime
	ragSvc := s.RAG
	trainingSvc := s.Training
	mcpSvc := s.MCP
	mode := s.mode
	daemonConfigured := s.config.Daemon.Command != ""
	daemonCommand := s.config.Daemon.Command
	daemonAddr := s.config.Daemon.Addr()
	daemonConnected := s.daemonClient != nil
	s.mu.RUnlock()

	contextHealth := map[string]interface{}{
		"enabled": false,
		"ready":   false,
	}
	if contextManager != nil {
		contextHealth["enabled"] = contextManager.Enabled()
	}
	if contextManager != nil && contextManager.Enabled() {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		if health, err := contextManager.Readiness(ctx); err == nil {
			contextHealth["ready"] = health.Ready
			contextHealth["status"] = health.Status
		} else {
			contextHealth["error"] = err.Error()
		}
	}

	degraded := false
	status := "ok"
	if !running {
		status = "stopped"
		degraded = true
	} else if !daemonConnected && daemonConfigured {
		status = "degraded"
		degraded = true
	} else if contextManager != nil && contextManager.Enabled() {
		if ready, ok := contextHealth["ready"].(bool); ok && !ready {
			status = "degraded"
			degraded = true
		}
	}

	return map[string]interface{}{
		"running":        running,
		"ready":          running && !degraded,
		"status":         status,
		"execution_mode": mode,
		"degraded":       degraded,
		"daemon": map[string]interface{}{
			"configured": daemonConfigured,
			"required":   s.config.Daemon.Required || s.config.IsProduction(),
			"connected":  daemonConnected,
			"addr":       daemonAddr,
			"command":    daemonCommand,
		},
		"context": contextHealth,
		"service_modes": map[string]interface{}{
			"runtime":  mode,
			"rag":      mode,
			"training": mode,
			"mcp":      mode,
			"context":  "context-service",
		},
		"runtime": map[string]interface{}{
			"loaded_models": loadedModelCount(runtimeSvc),
		},
		"rag": map[string]interface{}{
			"documents": documentCount(ragSvc),
		},
		"training": map[string]interface{}{
			"active_runs": activeRunCount(trainingSvc),
		},
		"mcp": map[string]interface{}{
			"connections": connectionCount(mcpSvc),
		},
	}
}

func (s *Supervisor) initLocalServicesLocked() {
	s.Runtime = s.wrapRuntimeService(runtime.NewManager(s.config))
	s.RAG = rag.NewManager(s.config, s.Context)
	s.Training = training.NewManager(s.config)
	s.MCP = mcp.NewManager(s.config)
	s.mode = "local-fallback"
}

func (s *Supervisor) launchDaemonLocked() error {
	cmd := exec.CommandContext(s.ctx, s.config.Daemon.Command, s.config.Daemon.Args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = append(os.Environ(), s.daemonEnv()...)
	if err := cmd.Start(); err != nil {
		return err
	}

	// Snapshot daemon address and release lock before waiting
	daemonAddr := s.config.Daemon.Addr()
	s.mu.Unlock()

	client, err := s.waitForDaemonClient(daemonAddr)
	if err != nil {
		_ = cmd.Process.Kill()
		s.mu.Lock()
		return err
	}

	// Re-acquire lock to update supervisor state
	s.mu.Lock()
	if s.daemonClient != nil {
		_ = s.daemonClient.Close()
	}
	s.daemonClient = client
	s.daemonCmd = cmd
	s.Context.SetDaemonContextClient(client)
	s.Runtime = s.wrapRuntimeService(client)
	s.RAG = client
	s.Training = client
	s.MCP = client
	s.mode = "daemon"

	s.wg.Add(1)
	go s.watchDaemon(cmd)
	return nil
}

func (s *Supervisor) waitForDaemonClient(addr string) (*daemon.Client, error) {
	deadline := time.Now().Add(s.config.Daemon.StartupTimeout)
	var lastErr error

	for time.Now().Before(deadline) {
		select {
		case <-s.ctx.Done():
			return nil, s.ctx.Err()
		default:
		}

		client, err := daemon.NewClient(s.ctx, addr)
		if err == nil {
			return client, nil
		}
		lastErr = err
		time.Sleep(250 * time.Millisecond)
	}

	if lastErr == nil {
		lastErr = fmt.Errorf("daemon did not become ready")
	}
	return nil, lastErr
}

func (s *Supervisor) daemonEnv() []string {
	env := []string{
		fmt.Sprintf("AI_ENGINE_DAEMON_ADDR=%s", s.config.Daemon.Addr()),
		fmt.Sprintf("AI_ENGINE_LANCEDB_URI=%s", s.config.Storage.LanceDBURI),
		fmt.Sprintf("AI_ENGINE_MODELS_PATH=%s", s.config.Runtime.ModelsPath),
		fmt.Sprintf("AI_ENGINE_TRAINING_DIR=%s", s.config.Training.WorkingDir),
		fmt.Sprintf("AI_ENGINE_LLAMA_CLI=%s", s.config.Daemon.LlamaCLI),
		fmt.Sprintf("AI_ENGINE_TRAINING_CLI=%s", s.config.Daemon.TrainingCLI),
	}
	if s.config.Context.Enabled {
		env = append(env, fmt.Sprintf("CONTEXT_DATA_DIR=%s", s.config.Context.DataDir))
		if len(s.config.Context.ManagedRoots) > 0 {
			env = append(env, fmt.Sprintf("CONTEXT_ROOTS=%s", strings.Join(s.config.Context.ManagedRoots, string(os.PathListSeparator))))
		}
		if s.config.Context.OpenViking.URL != "" {
			env = append(env, fmt.Sprintf("CONTEXT_OPENVIKING_URL=%s", s.config.Context.OpenViking.URL))
		}
		if s.config.Context.OpenViking.APIKey != "" {
			env = append(env, fmt.Sprintf("CONTEXT_OPENVIKING_API_KEY=%s", s.config.Context.OpenViking.APIKey))
		}
	}
	return env
}

func (s *Supervisor) wrapRuntimeService(service runtime.Service) runtime.Service {
	if service == nil || s.Context == nil || !s.Context.Enabled() {
		return service
	}
	return runtime.NewContextAwareService(service, s.Context)
}

func (s *Supervisor) watchDaemon(cmd *exec.Cmd) {
	defer s.wg.Done()

	if err := cmd.Wait(); err != nil && s.ctx.Err() == nil {
		stdlog.Printf("AI Engine daemon exited: %v", err)
	}

	if s.ctx.Err() != nil {
		return
	}

	const maxRestarts = 5
	restartCount := 0
	backoff := s.config.Daemon.RestartBackoff

	for {
		select {
		case <-s.ctx.Done():
			return
		case <-time.After(backoff):
		}

		s.mu.Lock()
		if !s.running || s.config.Daemon.Command == "" {
			s.mu.Unlock()
			return
		}

		restartCount++
		if restartCount > maxRestarts {
			stdlog.Printf("AI Engine daemon exceeded max restart attempts (%d)", maxRestarts)
			s.mu.Unlock()
			return
		}

		stdlog.Printf("Restarting AI Engine daemon (attempt %d/%d)...", restartCount, maxRestarts)
		if err := s.launchDaemonLocked(); err != nil {
			stdlog.Printf("AI Engine daemon restart failed: %v", err)
			backoff = backoff * 2
			if backoff > 30*time.Second {
				backoff = 30 * time.Second
			}
			s.mu.Unlock()
			continue
		}
		s.mu.Unlock()
		return
	}
}

func loadedModelCount(svc runtime.Service) int {
	if svc == nil {
		return 0
	}
	return svc.LoadedModelCount()
}

func documentCount(svc rag.Service) int64 {
	if svc == nil {
		return 0
	}
	return svc.DocumentCount()
}

func activeRunCount(svc training.Service) int {
	if svc == nil {
		return 0
	}
	return svc.ActiveRunCount()
}

func connectionCount(svc mcp.Service) int {
	if svc == nil {
		return 0
	}
	return svc.ConnectionCount()
}

func (s *Supervisor) ExecutionMode() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.mode
}
