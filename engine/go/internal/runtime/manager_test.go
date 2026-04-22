package runtime

import (
	"context"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/ai-engine/go/internal/config"
	"google.golang.org/protobuf/types/known/emptypb"
)

func TestListModelsConcurrentAccess(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.Runtime.ModelsPath = t.TempDir()

	if err := os.WriteFile(filepath.Join(cfg.Runtime.ModelsPath, "model.gguf"), []byte("weights"), 0644); err != nil {
		t.Fatalf("write model: %v", err)
	}

	manager := NewManager(cfg)

	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := manager.ListModels(context.Background(), &emptypb.Empty{}); err != nil {
				t.Errorf("list models: %v", err)
			}
		}()
	}
	wg.Wait()
}
