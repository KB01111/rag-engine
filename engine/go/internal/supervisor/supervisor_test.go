package supervisor

import (
	"strings"
	"testing"

	"github.com/ai-engine/go/internal/config"
)

func TestStartFailsWhenDaemonIsRequiredAndMissing(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.Daemon.Command = ""
	cfg.Daemon.Required = true

	sup := NewSupervisor(cfg)
	err := sup.Start()
	if err == nil {
		t.Fatal("expected supervisor start to fail when daemon is required but missing")
	}
	if !strings.Contains(err.Error(), "daemon is required") {
		t.Fatalf("expected daemon-required error, got %v", err)
	}
}

func TestStartFallsBackToLocalServicesWhenDaemonIsOptional(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.Daemon.Command = ""
	cfg.Daemon.Required = false

	sup := NewSupervisor(cfg)
	if err := sup.Start(); err != nil {
		t.Fatalf("expected optional daemon mode to start with local services, got %v", err)
	}
	defer sup.Stop()

	if !sup.IsRunning() {
		t.Fatal("expected supervisor to be running")
	}
}
