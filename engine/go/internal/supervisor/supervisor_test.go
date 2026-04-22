package supervisor

import (
	"testing"

	"github.com/ai-engine/go/internal/config"
	"github.com/stretchr/testify/require"
)

func TestHealthReportsFallbackModeWithoutDaemon(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.Context.Enabled = false
	cfg.Daemon.Command = ""

	sup := NewSupervisor(cfg)
	require.Equal(t, "local-fallback", sup.ExecutionMode())

	health := sup.Health()
	require.Equal(t, "local-fallback", health["execution_mode"])
	require.Equal(t, true, health["degraded"])

	daemon, ok := health["daemon"].(map[string]interface{})
	require.True(t, ok)
	require.Equal(t, false, daemon["configured"])
	require.Equal(t, false, daemon["connected"])
}
