package supervisor

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/ai-engine/go/internal/config"
	contextsvc "github.com/ai-engine/go/internal/contextsvc"
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

	Context  *contextsvc.Manager
	Runtime  *runtime.Manager
	RAG      *rag.Manager
	Training *training.Manager
	MCP      *mcp.Manager

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
	return &Supervisor{
		ctx:      ctx,
		cancel:   cancel,
		config:   cfg,
		Context:  contextManager,
		Runtime:  runtime.NewManager(cfg),
		RAG:      rag.NewManager(cfg, contextManager),
		Training: training.NewManager(cfg),
		MCP:      mcp.NewManager(cfg),
	}
}

func (s *Supervisor) Start() error {
	s.mu.Lock()
	if s.running {
		s.mu.Unlock()
		return fmt.Errorf("supervisor already running")
	}

	if err := s.config.EnsureDirs(); err != nil {
		s.mu.Unlock()
		return fmt.Errorf("failed to ensure directories: %w", err)
	}
	if err := s.Context.Start(s.ctx); err != nil {
		s.mu.Unlock()
		return fmt.Errorf("failed to start context backend: %w", err)
	}

	s.running = true
	s.mu.Unlock()

	log.Printf("AI Engine supervisor started")

	s.wg.Add(1)
	go s.handleSignals()

	return nil
}

func (s *Supervisor) Stop() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.running {
		return fmt.Errorf("supervisor not running")
	}

	s.cancel()
	if err := s.Context.Stop(context.Background()); err != nil {
		log.Printf("failed to stop context backend: %v", err)
	}
	s.wg.Wait()
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
	running := s.running
	s.mu.RUnlock()

	contextHealth := map[string]interface{}{
		"enabled": s.Context.Enabled(),
		"ready":   false,
	}
	if s.Context.Enabled() {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		if health, err := s.Context.Readiness(ctx); err == nil {
			contextHealth["ready"] = health.Ready
			contextHealth["status"] = health.Status
		} else {
			contextHealth["error"] = err.Error()
		}
	}

	return map[string]interface{}{
		"running": running,
		"context": contextHealth,
		"runtime": map[string]interface{}{
			"loaded_models": len(s.Runtime.ListModelsCached()),
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
