package contextsvc

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestManagerUpsertResource(t *testing.T) {
	t.Helper()

	type capturedRequest struct {
		URI      string            `json:"uri"`
		Content  string            `json:"content"`
		Title    string            `json:"title"`
		Metadata map[string]string `json:"metadata"`
	}

	var captured capturedRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, http.MethodPost, r.Method)
		require.Equal(t, "/v1/resources", r.URL.Path)

		defer r.Body.Close()
		require.NoError(t, json.NewDecoder(r.Body).Decode(&captured))

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"resource":{"uri":"viking://resources/doc-1","title":"Doc 1","layer":"l2","metadata":{"title":"Doc 1"}},"chunks_indexed":3}`))
	}))
	defer server.Close()

	manager := NewManager(Config{
		BaseURL: server.URL,
		Enabled: true,
	})

	resp, err := manager.UpsertResource(context.Background(), UpsertResourceRequest{
		URI:     "viking://resources/doc-1",
		Title:   "Doc 1",
		Content: "hello world",
		Layer:   LayerL2,
		Metadata: map[string]string{
			"title": "Doc 1",
		},
	})

	require.NoError(t, err)
	require.Equal(t, "viking://resources/doc-1", resp.Resource.URI)
	require.Equal(t, int32(3), resp.ChunksIndexed)
	require.Equal(t, "viking://resources/doc-1", captured.URI)
	require.Equal(t, "hello world", captured.Content)
	require.Equal(t, "Doc 1", captured.Title)
}

func TestManagerSearch(t *testing.T) {
	t.Helper()

	var body bytes.Buffer
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, http.MethodPost, r.Method)
		require.Equal(t, "/v1/search", r.URL.Path)

		defer r.Body.Close()
		_, _ = body.ReadFrom(r.Body)

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"results":[{"uri":"viking://resources/doc-1","document_id":"doc-1","chunk_text":"matched text","score":0.91,"metadata":{"kind":"resource"},"layer":"l2"}],"query_time_ms":2.5}`))
	}))
	defer server.Close()

	manager := NewManager(Config{
		BaseURL: server.URL,
		Enabled: true,
	})

	resp, err := manager.Search(context.Background(), SearchRequest{
		Query:    "matched text",
		ScopeURI: "viking://resources/",
		TopK:     5,
		Rerank:   true,
		Layer:    LayerL2,
	})

	require.NoError(t, err)
	require.Len(t, resp.Results, 1)
	require.Equal(t, "viking://resources/doc-1", resp.Results[0].URI)
	require.Contains(t, body.String(), `"scope_uri":"viking://resources/"`)
	require.Contains(t, body.String(), `"rerank":true`)
}

func TestManagerReadiness(t *testing.T) {
	t.Helper()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/health", r.URL.Path)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"ok","ready":true}`))
	}))
	defer server.Close()

	manager := NewManager(Config{
		BaseURL: server.URL,
		Enabled: true,
	})

	status, err := manager.Readiness(context.Background())
	require.NoError(t, err)
	require.True(t, status.Ready)
	require.Equal(t, "ok", status.Status)
}

func TestManagerSyncWorkspace(t *testing.T) {
	t.Helper()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, http.MethodPost, r.Method)
		require.Equal(t, "/v1/workspaces/sync", r.URL.Path)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"root":"workspace","prefix":"docs","indexed_resources":4,"reindexed_resources":2,"deleted_resources":1,"skipped_files":0}`))
	}))
	defer server.Close()

	manager := NewManager(Config{
		BaseURL: server.URL,
		Enabled: true,
	})

	resp, err := manager.SyncWorkspace(context.Background(), WorkspaceSyncRequest{
		Root: "workspace",
		Path: "docs",
	})

	require.NoError(t, err)
	require.Equal(t, "workspace", resp.Root)
	require.NotNil(t, resp.Prefix)
	require.Equal(t, "docs", *resp.Prefix)
	require.Equal(t, int32(4), resp.IndexedResources)
	require.Equal(t, int32(2), resp.ReindexedResources)
}

func TestManagerFileAndSessionContracts(t *testing.T) {
	t.Helper()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/v1/files/read":
			require.Equal(t, http.MethodPost, r.Method)
			_, _ = w.Write([]byte(`{"path":"notes/todo.md","content":"ship it","version":42}`))
		case "/v1/files/write":
			require.Equal(t, http.MethodPost, r.Method)
			_, _ = w.Write([]byte(`{"path":"notes/todo.md","version":43}`))
		case "/v1/files/delete":
			require.Equal(t, http.MethodPost, r.Method)
			_, _ = w.Write([]byte(`{"path":"notes/todo.md","deleted":true}`))
		case "/v1/files/move":
			require.Equal(t, http.MethodPost, r.Method)
			_, _ = w.Write([]byte(`{"from_path":"notes/todo.md","to_path":"archive/todo.md","version":99}`))
		case "/v1/sessions/append":
			require.Equal(t, http.MethodPost, r.Method)
			_, _ = w.Write([]byte(`{"session_id":"sess-1","entries":[{"session_id":"sess-1","role":"user","content":"hello","metadata":{"source":"test"},"created_at":1710000000}]}`))
		case "/v1/sessions/sess-1":
			require.Equal(t, http.MethodGet, r.Method)
			_, _ = w.Write([]byte(`{"session_id":"sess-1","entries":[{"session_id":"sess-1","role":"user","content":"hello","metadata":{"source":"test"},"created_at":1710000000}]}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	manager := NewManager(Config{
		BaseURL: server.URL,
		Enabled: true,
	})

	readResp, err := manager.ReadFile(context.Background(), FileReadRequest{
		Root: "workspace",
		Path: "notes/todo.md",
	})
	require.NoError(t, err)
	require.Equal(t, "notes/todo.md", readResp.Path)
	require.Equal(t, int64(42), readResp.Version)

	version := int64(42)
	writeResp, err := manager.WriteFile(context.Background(), FileWriteRequest{
		Root:    "workspace",
		Path:    "notes/todo.md",
		Content: "ship it",
		Version: &version,
	})
	require.NoError(t, err)
	require.Equal(t, int64(43), writeResp.Version)

	deleteResp, err := manager.DeleteFile(context.Background(), FileDeleteRequest{
		Root:    "workspace",
		Path:    "notes/todo.md",
		Version: &version,
	})
	require.NoError(t, err)
	require.True(t, deleteResp.Deleted)

	moveResp, err := manager.MoveFile(context.Background(), FileMoveRequest{
		Root:     "workspace",
		FromPath: "notes/todo.md",
		ToPath:   "archive/todo.md",
	})
	require.NoError(t, err)
	require.Equal(t, "archive/todo.md", moveResp.ToPath)
	require.Equal(t, int64(99), moveResp.Version)

	appendResp, err := manager.AppendSession(context.Background(), SessionAppendRequest{
		SessionID: "sess-1",
		Role:      "user",
		Content:   "hello",
		Metadata:  map[string]string{"source": "test"},
	})
	require.NoError(t, err)
	require.Len(t, appendResp.Entries, 1)
	require.Equal(t, int64(1710000000), appendResp.Entries[0].CreatedAt)

	sessionResp, err := manager.GetSession(context.Background(), "sess-1")
	require.NoError(t, err)
	require.Len(t, sessionResp.Entries, 1)
	require.Equal(t, "hello", sessionResp.Entries[0].Content)
}
