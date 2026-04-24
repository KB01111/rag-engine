package supervisor

import (
	"strings"
	"testing"

	"github.com/ai-engine/go/internal/config"
	"github.com/ai-engine/go/internal/runtime"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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

func TestStartFallsBackToLocalServicesWhenOptionalDaemonLaunchFails(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.Daemon.Command = "__missing_daemon_command__"
	cfg.Daemon.Required = false

	sup := NewSupervisor(cfg)
	if err := sup.Start(); err != nil {
		t.Fatalf("expected optional daemon launch failures to fall back to local services, got %v", err)
	}
	defer sup.Stop()

	if !sup.IsRunning() {
		t.Fatal("expected supervisor to be running")
	}
	assert.Equal(t, "local-fallback", sup.mode)
}

func TestNewSupervisorWrapsRuntimeWithContextAwareService(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.Context.Enabled = true
	cfg.Context.AutoStart = false
	cfg.Daemon.Command = ""

	sup := NewSupervisor(cfg)

	_, ok := sup.Runtime.(*runtime.ContextAwareService)
	require.True(t, ok, "expected runtime service to be wrapped for context assembly")
}

func TestNewSupervisorDoesNotWrapWhenContextDisabled(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.Context.Enabled = false
	cfg.Daemon.Command = ""

	sup := NewSupervisor(cfg)

	_, ok := sup.Runtime.(*runtime.ContextAwareService)
	assert.False(t, ok, "runtime should NOT be wrapped when context is disabled")
}

func TestWrapRuntimeServiceNilServiceReturnsNil(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.Context.Enabled = true
	cfg.Daemon.Command = ""

	sup := NewSupervisor(cfg)
	got := sup.wrapRuntimeService(nil)
	assert.Nil(t, got)
}

func TestWrapRuntimeServiceNilContextReturnsService(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.Daemon.Command = ""

	sup := NewSupervisor(cfg)
	sup.Context = nil

	inner := runtime.NewManager(cfg)
	got := sup.wrapRuntimeService(inner)
	assert.Equal(t, inner, got, "service should be returned as-is when Context is nil")
}

func TestHealthContainsExpectedTopLevelKeys(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.Context.Enabled = false
	cfg.Daemon.Command = ""

	sup := NewSupervisor(cfg)
	health := sup.Health()

	assert.Contains(t, health, "running")
	assert.Contains(t, health, "status")
	assert.Contains(t, health, "execution_mode")
	assert.Contains(t, health, "degraded")
	assert.Contains(t, health, "daemon")
	assert.Contains(t, health, "context")
	assert.Contains(t, health, "service_modes")
	assert.Contains(t, health, "runtime")
	assert.Contains(t, health, "rag")
	assert.Contains(t, health, "training")
	assert.Contains(t, health, "mcp")
}

func TestHealthRunningIsFalseBeforeStart(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.Context.Enabled = false
	cfg.Daemon.Command = ""

	sup := NewSupervisor(cfg)
	health := sup.Health()

	running, ok := health["running"].(bool)
	require.True(t, ok)
	assert.False(t, running)
}

func TestHealthReportsFallbackModeWithoutDaemon(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.Context.Enabled = false
	cfg.Daemon.Command = ""
	cfg.Daemon.Required = false

	sup := NewSupervisor(cfg)
	require.NoError(t, sup.Start())
	t.Cleanup(func() {
		require.NoError(t, sup.Stop())
	})

	require.Equal(t, "local-fallback", sup.ExecutionMode())

	health := sup.Health()
	require.Equal(t, "local-fallback", health["execution_mode"])
	require.Equal(t, false, health["degraded"])
	require.Equal(t, true, health["ready"])
	require.Equal(t, "ok", health["status"])

	daemon, ok := health["daemon"].(map[string]interface{})
	require.True(t, ok)
	require.Equal(t, false, daemon["configured"])
	require.Equal(t, false, daemon["connected"])
}

func TestStartFailsInProductionWithoutDaemon(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.Server.Mode = "production"
	cfg.Context.Enabled = true
	cfg.Context.AutoStart = false
	cfg.Context.ServiceURL = "http://127.0.0.1:9"
	cfg.Daemon.Command = ""

	sup := NewSupervisor(cfg)
	err := sup.Start()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "daemon")
}

func TestStartFailsInProductionWhenContextDisabled(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.Server.Mode = "production"
	cfg.Context.Enabled = false
	cfg.Daemon.Command = "ai_engine_daemon.exe"

	sup := NewSupervisor(cfg)
	err := sup.Start()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "context")
}
