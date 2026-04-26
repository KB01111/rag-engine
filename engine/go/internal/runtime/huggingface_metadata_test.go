package runtime

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ai-engine/go/internal/config"
	"github.com/ai-engine/go/internal/hub"
	"github.com/stretchr/testify/require"
)

func TestDiscoverLocalModelsIncludesHuggingFaceSidecarMetadata(t *testing.T) {
	modelsPath := t.TempDir()
	modelPath := filepath.Join(modelsPath, "acme__tiny__tiny.gguf")
	require.NoError(t, os.WriteFile(modelPath, []byte("weights"), 0o644))

	manifest := hub.Manifest{
		RepoID:       "acme/tiny",
		Filename:     "tiny.gguf",
		Revision:     "main",
		ResolvedURL:  "https://huggingface.co/acme/tiny/resolve/main/tiny.gguf",
		ETag:         `"abc123"`,
		License:      "mit",
		SizeBytes:    7,
		DownloadedAt: time.Date(2026, 4, 26, 10, 0, 0, 0, time.UTC),
	}
	data, err := json.Marshal(manifest)
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(modelPath+".hf.json", data, 0o644))

	cfg := config.DefaultConfig()
	cfg.Runtime.ModelsPath = modelsPath
	manager := NewManager(cfg)

	models, err := manager.discoverLocalModels()
	require.NoError(t, err)
	require.Len(t, models, 1)

	metadata := models[0].Metadata
	require.Equal(t, "huggingface", metadata["source"])
	require.Equal(t, "true", metadata["downloaded"])
	require.Equal(t, "acme/tiny", metadata["repo_id"])
	require.Equal(t, "tiny.gguf", metadata["filename"])
	require.Equal(t, "main", metadata["revision"])
	require.Equal(t, manifest.ResolvedURL, metadata["resolved_url"])
	require.Equal(t, "mit", metadata["license"])
	require.Equal(t, `"abc123"`, metadata["etag"])
	require.Equal(t, manifest.DownloadedAt.Format(time.RFC3339), metadata["downloaded_at"])
}
