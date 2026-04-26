package hub

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func TestSearchFiltersPublicCompatibleModels(t *testing.T) {
	client := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		require.Equal(t, "/api/models", req.URL.Path)
		require.Equal(t, "true", req.URL.Query().Get("full"))
		require.Equal(t, "llama", req.URL.Query().Get("search"))

		body := `[
			{"id":"acme/tiny-gguf","author":"acme","private":false,"gated":false,"tags":["license:mit"],"siblings":[{"rfilename":"tiny.Q4_K_M.gguf","size":12}]},
			{"id":"acme/safetensors-only","author":"acme","private":false,"gated":false,"siblings":[{"rfilename":"model.safetensors","size":12}]},
			{"id":"acme/gated-gguf","author":"acme","private":false,"gated":"manual","siblings":[{"rfilename":"gated.gguf","size":12}]},
			{"id":"acme/private-gguf","author":"acme","private":true,"gated":false,"siblings":[{"rfilename":"private.gguf","size":12}]}
		]`
		return jsonResponse(http.StatusOK, body), nil
	})}

	service := NewService(Config{
		Endpoint:   "https://huggingface.test",
		ModelsPath: t.TempDir(),
		HTTPClient: client,
	})

	result, err := service.Search(context.Background(), SearchRequest{Query: "llama", Limit: 10})
	require.NoError(t, err)
	require.Len(t, result.Models, 1)
	require.Equal(t, "acme/tiny-gguf", result.Models[0].ID)
	require.Equal(t, "mit", result.Models[0].License)
	require.Len(t, result.Models[0].CompatibleFiles, 1)
	require.Equal(t, "tiny.Q4_K_M.gguf", result.Models[0].CompatibleFiles[0].Filename)
}

func TestSearchReturnsAndUsesNextCursorFromHuggingFaceLinkHeader(t *testing.T) {
	client := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		switch req.URL.Query().Get("cursor") {
		case "":
			resp := jsonResponse(http.StatusOK, `[
				{"id":"acme/page-one","author":"acme","private":false,"gated":false,"tags":["license:mit"],"siblings":[{"rfilename":"one.gguf","size":1}]}
			]`)
			resp.Header.Set("Link", `<https://huggingface.test/api/models?cursor=abc123&limit=1>; rel="next"`)
			return resp, nil
		case "abc123":
			return jsonResponse(http.StatusOK, `[
				{"id":"acme/page-two","author":"acme","private":false,"gated":false,"tags":["license:mit"],"siblings":[{"rfilename":"two.gguf","size":1}]}
			]`), nil
		default:
			return jsonResponse(http.StatusBadRequest, `{"error":"unexpected cursor"}`), nil
		}
	})}

	service := NewService(Config{
		Endpoint:   "https://huggingface.test",
		ModelsPath: t.TempDir(),
		HTTPClient: client,
	})

	firstPage, err := service.Search(context.Background(), SearchRequest{Query: "llama", Limit: 1})
	require.NoError(t, err)
	require.True(t, firstPage.HasMore)
	require.NotEmpty(t, firstPage.NextCursor)
	require.Equal(t, "acme/page-one", firstPage.Models[0].ID)

	secondPage, err := service.Search(context.Background(), SearchRequest{Cursor: firstPage.NextCursor})
	require.NoError(t, err)
	require.False(t, secondPage.HasMore)
	require.Empty(t, secondPage.NextCursor)
	require.Equal(t, "acme/page-two", secondPage.Models[0].ID)
}

func TestDownloadRejectsUnsafeAndUnsupportedFiles(t *testing.T) {
	service := NewService(Config{
		Endpoint:   "https://huggingface.test",
		ModelsPath: t.TempDir(),
	})

	_, err := service.StartDownload(context.Background(), DownloadRequest{
		RepoID:   "acme/tiny",
		Filename: "../tiny.gguf",
	})
	require.ErrorIs(t, err, ErrInvalidRequest)

	_, err = service.StartDownload(context.Background(), DownloadRequest{
		RepoID:   "acme/tiny",
		Filename: "model.safetensors",
	})
	require.ErrorIs(t, err, ErrUnsupportedModelFormat)
	require.Contains(t, err.Error(), "GGUF/GGML/BIN")
}

func TestDownloadCreatesManifestAndAtomicallyFinalizesFile(t *testing.T) {
	var mu sync.Mutex
	requested := make([]string, 0, 4)
	payload := []byte("gguf-weights")
	client := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		mu.Lock()
		requested = append(requested, req.Method+" "+req.URL.Path)
		mu.Unlock()

		switch {
		case req.Method == http.MethodGet && req.URL.Path == "/api/models/acme/tiny":
			return jsonResponse(http.StatusOK, `{"id":"acme/tiny","author":"acme","private":false,"gated":false,"cardData":{"license":"apache-2.0"},"siblings":[{"rfilename":"tiny.gguf","size":12}]}`), nil
		case req.Method == http.MethodHead && req.URL.Path == "/acme/tiny/resolve/main/tiny.gguf":
			resp := jsonResponse(http.StatusOK, "")
			resp.Header.Set("Content-Length", "12")
			resp.Header.Set("ETag", `"abc123"`)
			resp.Body = http.NoBody
			return resp, nil
		case req.Method == http.MethodGet && req.URL.Path == "/acme/tiny/resolve/main/tiny.gguf":
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Length": []string{"12"}, "ETag": []string{`"abc123"`}},
				Body:       io.NopCloser(bytes.NewReader(payload)),
			}, nil
		default:
			return jsonResponse(http.StatusNotFound, `{"error":"missing"}`), nil
		}
	})}
	modelsPath := t.TempDir()
	service := NewService(Config{
		Endpoint:   "https://huggingface.test",
		ModelsPath: modelsPath,
		HTTPClient: client,
	})

	download, err := service.StartDownload(context.Background(), DownloadRequest{
		RepoID:   "acme/tiny",
		Filename: "tiny.gguf",
	})
	require.NoError(t, err)
	download = waitForDownload(t, service, download.ID)
	require.Equal(t, DownloadStatusCompleted, download.Status)
	require.Equal(t, int64(len(payload)), download.DownloadedBytes)
	require.FileExists(t, download.TargetPath)
	require.NoFileExists(t, download.TargetPath+".partial")
	require.Equal(t, payload, mustReadFile(t, download.TargetPath))

	manifest := Manifest{}
	require.NoError(t, readJSONFile(download.ManifestPath, &manifest))
	require.Equal(t, "acme/tiny", manifest.RepoID)
	require.Equal(t, "tiny.gguf", manifest.Filename)
	require.Equal(t, "main", manifest.Revision)
	require.Equal(t, "apache-2.0", manifest.License)
	require.Equal(t, `"abc123"`, manifest.ETag)
	require.Equal(t, int64(len(payload)), manifest.SizeBytes)
	require.NotZero(t, manifest.DownloadedAt)

	mu.Lock()
	seen := strings.Join(requested, "\n")
	mu.Unlock()
	require.Contains(t, seen, "GET /api/models/acme/tiny")
	require.Contains(t, seen, "HEAD /acme/tiny/resolve/main/tiny.gguf")
	require.Contains(t, seen, "GET /acme/tiny/resolve/main/tiny.gguf")
}

func TestDuplicateDownloadReturnsExistingCompletedRecord(t *testing.T) {
	modelsPath := t.TempDir()
	targetPath := filepath.Join(modelsPath, "acme__tiny__tiny.gguf")
	require.NoError(t, os.WriteFile(targetPath, []byte("existing"), 0o644))

	service := NewService(Config{
		Endpoint:   "https://huggingface.test",
		ModelsPath: modelsPath,
	})

	download, err := service.StartDownload(context.Background(), DownloadRequest{
		RepoID:   "acme/tiny",
		Filename: "tiny.gguf",
	})
	require.NoError(t, err)
	require.Equal(t, DownloadStatusCompleted, download.Status)
	require.Equal(t, targetPath, download.TargetPath)
	require.FileExists(t, download.ManifestPath)
}

func TestCancelDownloadStopsActiveTransfer(t *testing.T) {
	client := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		switch {
		case req.Method == http.MethodGet && req.URL.Path == "/api/models/acme/tiny":
			return jsonResponse(http.StatusOK, `{"id":"acme/tiny","author":"acme","private":false,"gated":false,"cardData":{"license":"mit"},"siblings":[{"rfilename":"tiny.gguf","size":128}]}`), nil
		case req.Method == http.MethodHead && req.URL.Path == "/acme/tiny/resolve/main/tiny.gguf":
			resp := jsonResponse(http.StatusOK, "")
			resp.Header.Set("Content-Length", "128")
			resp.Body = http.NoBody
			return resp, nil
		case req.Method == http.MethodGet && req.URL.Path == "/acme/tiny/resolve/main/tiny.gguf":
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Length": []string{"128"}},
				Body:       cancelBlockingBody{ctx: req.Context()},
			}, nil
		default:
			return jsonResponse(http.StatusNotFound, `{"error":"missing"}`), nil
		}
	})}

	service := NewService(Config{
		Endpoint:   "https://huggingface.test",
		ModelsPath: t.TempDir(),
		HTTPClient: client,
	})

	download, err := service.StartDownload(context.Background(), DownloadRequest{
		RepoID:   "acme/tiny",
		Filename: "tiny.gguf",
	})
	require.NoError(t, err)
	_, err = service.CancelDownload(download.ID)
	require.NoError(t, err)

	download = waitForDownload(t, service, download.ID)
	require.Equal(t, DownloadStatusCanceled, download.Status)
	require.NoFileExists(t, download.TargetPath+".partial")
}

func jsonResponse(status int, body string) *http.Response {
	return &http.Response{
		StatusCode: status,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}

func waitForDownload(t *testing.T, service *Service, id string) Download {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	var download Download
	for time.Now().Before(deadline) {
		var ok bool
		download, ok = service.GetDownload(id)
		require.True(t, ok)
		if download.Terminal() {
			return download
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("download %s did not finish; last status=%s error=%s", id, download.Status, download.Error)
	return Download{}
}

func mustReadFile(t *testing.T, path string) []byte {
	t.Helper()
	data, err := os.ReadFile(path)
	require.NoError(t, err)
	return data
}

type cancelBlockingBody struct {
	ctx context.Context
}

func (b cancelBlockingBody) Read([]byte) (int, error) {
	<-b.ctx.Done()
	return 0, b.ctx.Err()
}

func (b cancelBlockingBody) Close() error {
	return nil
}
