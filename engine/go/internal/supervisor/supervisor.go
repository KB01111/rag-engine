package supervisor

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/ai-engine/go/internal/config"
	"github.com/ai-engine/go/internal/daemon"
	"github.com/ai-engine/go/internal/mcp"
	"github.com/ai-engine/go/internal/rag"
	"github.com/ai-engine/go/internal/runtime"
	"github.com/ai-engine/go/internal/training"
	"log"
)

type Supervisor struct {
	mu      sync.RWMutex
	running bool
	ctx     context.Context
	cancel  context.CancelFunc
	config  *config.Config

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
	sup := &Supervisor{
		ctx:    ctx,
		cancel: cancel,
		config: cfg,
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

	if err := s.config.EnsureDirs(); err != nil {
		return fmt.Errorf("failed to ensure directories: %w", err)
	}

	if s.config.Daemon.Command != "" {
		if err := s.launchDaemonLocked(); err != nil {
			s.initLocalServicesLocked()
			return fmt.Errorf("failed to launch daemon: %w", err)
		}
	} else {
		s.initLocalServicesLocked()
	}

	s.running = true

	log.Printf("AI Engine supervisor started")

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
	s.wg.Wait()

	s.mu.Lock()
	defer s.mu.Unlock()
	s.running = false

	log.Printf("AI Engine supervisor stopped")
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
		log.Printf("Received signal: %v", sig)
		s.cancel()
	case <-s.ctx.Done():
	}
}

func (s *Supervisor) Health() map[string]interface{} {
	s.mu.RLock()
	defer s.mu.RUnlock()

	return map[string]interface{}{
		"running": s.running,
		"runtime": map[string]interface{}{
			"loaded_models": s.Runtime.LoadedModelCount(),
		},
		"rag": map[string]interface{}{
			"documents": s.RAG.DocumentCount(),
		},
		"training": map[string]interface{}{
			"active_runs": s.Training.ActiveRunCount(),
		},
		"mcp": map[string]interface{}{
			"connections": s.MCP.ConnectionCount(),
		},
	}
}

func (s *Supervisor) initLocalServicesLocked() {
	s.Runtime = runtime.NewManager(s.config)
	s.RAG = rag.NewManager(s.config)
	s.Training = training.NewManager(s.config)
	s.MCP = mcp.NewManager(s.config)
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
	s.Runtime = client
	s.RAG = client
	s.Training = client
	s.MCP = client

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
	return []string{
		fmt.Sprintf("AI_ENGINE_DAEMON_ADDR=%s", s.config.Daemon.Addr()),
		fmt.Sprintf("AI_ENGINE_LANCEDB_URI=%s", s.config.Storage.LanceDBURI),
		fmt.Sprintf("AI_ENGINE_MODELS_PATH=%s", s.config.Runtime.ModelsPath),
		fmt.Sprintf("AI_ENGINE_TRAINING_DIR=%s", s.config.Training.WorkingDir),
		fmt.Sprintf("AI_ENGINE_LLAMA_CLI=%s", s.config.Daemon.LlamaCLI),
		fmt.Sprintf("AI_ENGINE_TRAINING_CLI=%s", s.config.Daemon.TrainingCLI),
	}
}

func (s *Supervisor) watchDaemon(cmd *exec.Cmd) {
	defer s.wg.Done()

	if err := cmd.Wait(); err != nil && s.ctx.Err() == nil {
		log.Printf("AI Engine daemon exited: %v", err)
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
			log.Printf("AI Engine daemon exceeded max restart attempts (%d)", maxRestarts)
			s.mu.Unlock()
			return
		}

		log.Printf("Restarting AI Engine daemon (attempt %d/%d)...", restartCount, maxRestarts)
		if err := s.launchDaemonLocked(); err != nil {
			log.Printf("AI Engine daemon restart failed: %v", err)
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
