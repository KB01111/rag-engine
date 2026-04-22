package runtime

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/ai-engine/go/internal/config"
	"golang.org/x/sync/errgroup"
	"google.golang.org/protobuf/types/known/emptypb"
)

func TestListModelsConcurrentAccess(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.Runtime.ModelsPath = t.TempDir()

	if err := os.WriteFile(filepath.Join(cfg.Runtime.ModelsPath, "model.gguf"), []byte("weights"), 0644); err != nil {
		t.Fatalf("write model: %v", err)
	}

	manager := NewManager(cfg)

	var g errgroup.Group
	for i := 0; i < 8; i++ {
		g.Go(func() error {
			_, err := manager.ListModels(context.Background(), &emptypb.Empty{})
			return err
		})
	}
	if err := g.Wait(); err != nil {
		t.Fatalf("list models: %v", err)
	}
}
