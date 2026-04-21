package supervisor

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"sync"
	"syscall"

	"github.com/ai-engine/go/internal/config"
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

	Runtime  *runtime.Manager
	RAG      *rag.Manager
	Training *training.Manager
	MCP      *mcp.Manager

	wg sync.WaitGroup
}

func NewSupervisor(cfg *config.Config) *Supervisor {
	ctx, cancel := context.WithCancel(context.Background())
	return &Supervisor{
		ctx:      ctx,
		cancel:   cancel,
		config:   cfg,
		Runtime:  runtime.NewManager(cfg),
		RAG:      rag.NewManager(cfg),
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
	defer s.mu.RUnlock()

	return map[string]interface{}{
		"running": s.running,
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
