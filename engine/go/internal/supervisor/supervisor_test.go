package supervisor

import (
	"strings"
	"testing"

	"github.com/ai-engine/go/internal/config"
)

func TestDaemonEnvIncludesRuntimeBackendConfig(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.Runtime.Backend = "mock"
	cfg.Runtime.MaxMemory = 4096
	cfg.Runtime.MistralRS.ForceCPU = true
	cfg.Runtime.MistralRS.MaxNumSeqs = 8
	cfg.Runtime.MistralRS.AutoISQ = "q8"

	env := NewSupervisor(cfg).daemonEnv()
	joined := "\n" + strings.Join(env, "\n") + "\n"

	for _, expected := range []string{
		"\nAI_ENGINE_RUNTIME_BACKEND=mock\n",
		"\nAI_ENGINE_RUNTIME_MAX_MEMORY_MB=4096\n",
		"\nAI_ENGINE_MISTRALRS_FORCE_CPU=true\n",
		"\nAI_ENGINE_MISTRALRS_MAX_NUM_SEQS=8\n",
		"\nAI_ENGINE_MISTRALRS_AUTO_ISQ=q8\n",
	} {
		if !strings.Contains(joined, expected) {
			t.Fatalf("expected daemon env to contain %q, got %v", strings.TrimSpace(expected), env)
		}
	}
}
