package training

import (
	"context"
	"testing"
	"time"

	"github.com/ai-engine/go/internal/config"
	pb "github.com/ai-engine/proto/go"
)

func testConfig(t *testing.T) *config.Config {
	t.Helper()

	cfg := config.DefaultConfig()
	cfg.Training.WorkingDir = t.TempDir()
	cfg.Training.MaxJobs = 1
	return cfg
}

func TestStartRunIgnoresCompletedRunsInAdmission(t *testing.T) {
	manager := NewManager(testConfig(t))
	manager.runs["completed-run"] = &Run{
		ID:     "completed-run",
		Name:   "old",
		Status: "completed",
	}

	run, err := manager.StartRun(context.Background(), &pb.TrainingRunRequest{
		Name:        "fresh-run",
		ModelId:     "llama",
		DatasetPath: "dataset.jsonl",
	})
	if err != nil {
		t.Fatalf("expected completed runs to be ignored during admission, got error: %v", err)
	}
	if run.Id == "" {
		t.Fatal("expected a new run id to be assigned")
	}
}

func TestCancelledRunStaysCancelled(t *testing.T) {
	cfg := testConfig(t)
	cfg.Training.MaxJobs = 2

	manager := NewManager(cfg)
	run, err := manager.StartRun(context.Background(), &pb.TrainingRunRequest{
		Name:        "cancel-me",
		ModelId:     "llama",
		DatasetPath: "dataset.jsonl",
	})
	if err != nil {
		t.Fatalf("start run: %v", err)
	}

	if _, err := manager.CancelRun(context.Background(), &pb.CancelRequest{RunId: run.Id}); err != nil {
		t.Fatalf("cancel run: %v", err)
	}

	// Poll with timeout to verify cancelled status persists
	start := time.Now()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		time.Sleep(100 * time.Millisecond)
		manager.mu.RLock()
		status := manager.runs[run.Id].Status
		manager.mu.RUnlock()

		if status != "cancelled" {
			t.Fatalf("expected cancelled status to remain terminal, got %q after %v", status, time.Since(start))
		}
	}
}