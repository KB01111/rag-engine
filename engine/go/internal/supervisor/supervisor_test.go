package supervisor

import (
	"testing"

	"github.com/ai-engine/go/internal/config"
	"github.com/ai-engine/go/internal/runtime"
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
