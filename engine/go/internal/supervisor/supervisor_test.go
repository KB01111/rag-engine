package supervisor

import (
	"testing"

	"github.com/ai-engine/go/internal/config"
	"github.com/ai-engine/go/internal/runtime"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

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
	assert.Contains(t, health, "context")
	assert.Contains(t, health, "runtime")
	assert.Contains(t, health, "rag")
	assert.Contains(t, health, "training")
	assert.Contains(t, health, "mcp")
}

func TestHealthDoesNotContainRemovedKeys(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.Context.Enabled = false
	cfg.Daemon.Command = ""

	sup := NewSupervisor(cfg)
	health := sup.Health()

	// Keys that were removed in this PR
	assert.NotContains(t, health, "execution_mode")
	assert.NotContains(t, health, "degraded")
	assert.NotContains(t, health, "daemon")
	assert.NotContains(t, health, "service_modes")
	assert.NotContains(t, health, "status")
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