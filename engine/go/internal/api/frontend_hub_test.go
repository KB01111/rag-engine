package api_test

import (
	"bytes"
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ai-engine/go/internal/api"
	"github.com/ai-engine/go/internal/config"
	"github.com/ai-engine/go/internal/hub"
	"github.com/ai-engine/go/internal/supervisor"
	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog"
	"github.com/stretchr/testify/require"
)

func TestRuntimeHubFrontendEndpoints(t *testing.T) {
	gin.SetMode(gin.TestMode)

	cfg := config.DefaultConfig()
	cfg.Runtime.ModelsPath = t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(cfg.Runtime.ModelsPath, "demo.gguf"), []byte("weights"), 0o644))

	sup := supervisor.NewSupervisor(cfg)
	t.Cleanup(func() { _ = sup.Stop() })

	fake := &fakeRuntimeHub{
		searchResult: hub.SearchResponse{
			NextCursor: "next-page",
			HasMore:    true,
			Models: []hub.Model{{
				ID:      "acme/tiny",
				Author:  "acme",
				License: "mit",
				CompatibleFiles: []hub.File{{
					Filename:  "tiny.gguf",
					SizeBytes: 12,
					Extension: ".gguf",
				}},
			}},
		},
		modelResult: hub.Model{
			ID:      "acme/tiny",
			License: "mit",
			CompatibleFiles: []hub.File{{
				Filename:  "tiny.gguf",
				SizeBytes: 12,
				Extension: ".gguf",
			}},
		},
		downloadResult: hub.Download{
			ID:        "download-1",
			RepoID:    "acme/tiny",
			Filename:  "tiny.gguf",
			Revision:  "main",
			Status:    hub.DownloadStatusCompleted,
			StartedAt: time.Now(),
		},
	}

	server := api.NewServer(cfg, sup, zerolog.Nop())
	server.SetRuntimeHubService(fake)
	router := gin.New()
	server.RegisterHTTP(router)

	searchReq := httptest.NewRequest(http.MethodGet, "/api/v1/runtime/hub/search?query=llama&author=acme&sort=downloads&limit=5&cursor=page-one", nil)
	searchResp := httptest.NewRecorder()
	router.ServeHTTP(searchResp, searchReq)
	require.Equal(t, http.StatusOK, searchResp.Code)
	require.JSONEq(t, `{"models":[{"id":"acme/tiny","author":"acme","private":false,"gated":false,"requires_hf_token":false,"license":"mit","tags":null,"files":null,"compatible_files":[{"filename":"tiny.gguf","size_bytes":12,"extension":".gguf"}]}],"query":"llama","author":"acme","sort":"downloads","limit":5,"compatible_only":true,"next_cursor":"next-page","has_more":true}`, searchResp.Body.String())
	require.Equal(t, "llama", fake.lastSearch.Query)
	require.Equal(t, "acme", fake.lastSearch.Author)
	require.Equal(t, "downloads", fake.lastSearch.Sort)
	require.Equal(t, 5, fake.lastSearch.Limit)
	require.Equal(t, "page-one", fake.lastSearch.Cursor)

	modelReq := httptest.NewRequest(http.MethodGet, "/api/v1/runtime/hub/model?repo_id=acme%2Ftiny", nil)
	modelResp := httptest.NewRecorder()
	router.ServeHTTP(modelResp, modelReq)
	require.Equal(t, http.StatusOK, modelResp.Code)
	require.Contains(t, modelResp.Body.String(), `"id":"acme/tiny"`)
	require.Equal(t, "acme/tiny", fake.lastModelRepoID)

	createReq := httptest.NewRequest(http.MethodPost, "/api/v1/runtime/hub/downloads", bytes.NewReader([]byte(`{"repo_id":"acme/tiny","filename":"tiny.gguf"}`)))
	createReq.Header.Set("Content-Type", "application/json")
	createResp := httptest.NewRecorder()
	router.ServeHTTP(createResp, createReq)
	require.Equal(t, http.StatusAccepted, createResp.Code)
	require.Contains(t, createResp.Body.String(), `"id":"download-1"`)
	require.Equal(t, "acme/tiny", fake.lastDownload.RepoID)
	require.Equal(t, "tiny.gguf", fake.lastDownload.Filename)

	listReq := httptest.NewRequest(http.MethodGet, "/api/v1/runtime/hub/downloads", nil)
	listResp := httptest.NewRecorder()
	router.ServeHTTP(listResp, listReq)
	require.Equal(t, http.StatusOK, listResp.Code)
	require.Contains(t, listResp.Body.String(), `"downloads"`)

	eventsReq := httptest.NewRequest(http.MethodGet, "/api/v1/runtime/hub/downloads/download-1/events", nil)
	eventsResp := httptest.NewRecorder()
	router.ServeHTTP(eventsResp, eventsReq)
	require.Equal(t, http.StatusOK, eventsResp.Code)
	require.Contains(t, eventsResp.Body.String(), "event: complete")

	deleteReq := httptest.NewRequest(http.MethodDelete, "/api/v1/runtime/hub/downloads/download-1", nil)
	deleteResp := httptest.NewRecorder()
	router.ServeHTTP(deleteResp, deleteReq)
	require.Equal(t, http.StatusOK, deleteResp.Code)
	require.Contains(t, deleteResp.Body.String(), `"canceled":true`)
	require.Equal(t, "download-1", fake.canceledID)
}

func TestRuntimeHubDownloadUnsupportedFormatUsesSpecificErrorCode(t *testing.T) {
	gin.SetMode(gin.TestMode)

	cfg := config.DefaultConfig()
	cfg.Runtime.ModelsPath = t.TempDir()
	sup := supervisor.NewSupervisor(cfg)
	t.Cleanup(func() { _ = sup.Stop() })

	server := api.NewServer(cfg, sup, zerolog.Nop())
	server.SetRuntimeHubService(&fakeRuntimeHub{startErr: hub.NewError(hub.ErrorCodeUnsupportedModelFormat, hub.ErrUnsupportedModelFormat, "This backend currently supports GGUF/GGML/BIN local runtime artifacts.")})
	router := gin.New()
	server.RegisterHTTP(router)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/runtime/hub/downloads", bytes.NewReader([]byte(`{"repo_id":"acme/tiny","filename":"model.safetensors"}`)))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusBadRequest, resp.Code)
	require.JSONEq(t, `{"error":{"code":"unsupported_model_format","message":"This backend currently supports GGUF/GGML/BIN local runtime artifacts."}}`, resp.Body.String())
}

type fakeRuntimeHub struct {
	searchResult   hub.SearchResponse
	modelResult    hub.Model
	downloadResult hub.Download
	startErr       error

	lastSearch      hub.SearchRequest
	lastModelRepoID string
	lastDownload    hub.DownloadRequest
	canceledID      string
}

func (f *fakeRuntimeHub) Search(_ context.Context, req hub.SearchRequest) (hub.SearchResponse, error) {
	f.lastSearch = req
	return f.searchResult, nil
}

func (f *fakeRuntimeHub) Model(_ context.Context, repoID string) (hub.Model, error) {
	f.lastModelRepoID = repoID
	return f.modelResult, nil
}

func (f *fakeRuntimeHub) StartDownload(_ context.Context, req hub.DownloadRequest) (hub.Download, error) {
	f.lastDownload = req
	if f.startErr != nil {
		return hub.Download{}, f.startErr
	}
	return f.downloadResult, nil
}

func (f *fakeRuntimeHub) ListDownloads() []hub.Download {
	return []hub.Download{f.downloadResult}
}

func (f *fakeRuntimeHub) GetDownload(id string) (hub.Download, bool) {
	if id == f.downloadResult.ID {
		return f.downloadResult, true
	}
	return hub.Download{}, false
}

func (f *fakeRuntimeHub) CancelDownload(id string) (hub.Download, error) {
	f.canceledID = id
	if id == f.downloadResult.ID {
		return f.downloadResult, nil
	}
	return hub.Download{}, errors.New("not found")
}
