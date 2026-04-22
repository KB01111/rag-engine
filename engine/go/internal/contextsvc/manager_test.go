package contextsvc

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/suite"
)

type ManagerTestSuite struct {
	suite.Suite
	server  *httptest.Server
	manager *Manager
	// Captured request buffers for validation
	capturedUpsert      UpsertResourceRequest
	capturedSearchBody  bytes.Buffer
	capturedFileRead    FileReadRequest
	capturedFileWrite   FileWriteRequest
	capturedFileDelete  FileDeleteRequest
	capturedFileMove    FileMoveRequest
}

func (s *ManagerTestSuite) SetupTest() {
	s.capturedSearchBody.Reset()
	s.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/v1/resources":
			s.Require().Equal(http.MethodPost, r.Method)
			defer r.Body.Close()
			s.Require().NoError(json.NewDecoder(r.Body).Decode(&s.capturedUpsert))
			_, _ = w.Write([]byte(`{"resource":{"uri":"viking://resources/doc-1","title":"Doc 1","layer":"l2","metadata":{"title":"Doc 1"}},"chunks_indexed":3}`))
		case "/v1/search":
			s.Require().Equal(http.MethodPost, r.Method)
			defer r.Body.Close()
			_, _ = s.capturedSearchBody.ReadFrom(r.Body)
			_, _ = w.Write([]byte(`{"results":[{"uri":"viking://resources/doc-1","document_id":"doc-1","chunk_text":"matched text","score":0.91,"metadata":{"kind":"resource"},"layer":"l2"}],"query_time_ms":2.5}`))
		case "/health":
			_, _ = w.Write([]byte(`{"status":"ok","ready":true}`))
		case "/v1/workspaces/sync":
			s.Require().Equal(http.MethodPost, r.Method)
			_, _ = w.Write([]byte(`{"root":"workspace","prefix":"docs","indexed_resources":4,"reindexed_resources":2,"deleted_resources":1,"skipped_files":0}`))
		case "/v1/files/read":
			s.Require().Equal(http.MethodPost, r.Method)
			defer r.Body.Close()
			s.Require().NoError(json.NewDecoder(r.Body).Decode(&s.capturedFileRead))
			s.Require().Equal("workspace", s.capturedFileRead.Root)
			s.Require().Equal("notes/todo.md", s.capturedFileRead.Path)
			_, _ = w.Write([]byte(`{"path":"notes/todo.md","content":"ship it","version":42}`))
		case "/v1/files/write":
			s.Require().Equal(http.MethodPost, r.Method)
			defer r.Body.Close()
			s.Require().NoError(json.NewDecoder(r.Body).Decode(&s.capturedFileWrite))
			s.Require().Equal("workspace", s.capturedFileWrite.Root)
			s.Require().Equal("notes/todo.md", s.capturedFileWrite.Path)
			s.Require().NotNil(s.capturedFileWrite.Version)
			s.Require().Equal(int64(42), *s.capturedFileWrite.Version)
			_, _ = w.Write([]byte(`{"path":"notes/todo.md","version":43}`))
		case "/v1/files/delete":
			s.Require().Equal(http.MethodPost, r.Method)
			defer r.Body.Close()
			s.Require().NoError(json.NewDecoder(r.Body).Decode(&s.capturedFileDelete))
			s.Require().Equal("workspace", s.capturedFileDelete.Root)
			s.Require().Equal("notes/todo.md", s.capturedFileDelete.Path)
			s.Require().NotNil(s.capturedFileDelete.Version)
			s.Require().Equal(int64(42), *s.capturedFileDelete.Version)
			_, _ = w.Write([]byte(`{"path":"notes/todo.md","deleted":true}`))
		case "/v1/files/move":
			s.Require().Equal(http.MethodPost, r.Method)
			defer r.Body.Close()
			s.Require().NoError(json.NewDecoder(r.Body).Decode(&s.capturedFileMove))
			s.Require().Equal("workspace", s.capturedFileMove.Root)
			s.Require().Equal("notes/todo.md", s.capturedFileMove.FromPath)
			s.Require().Equal("archive/todo.md", s.capturedFileMove.ToPath)
			_, _ = w.Write([]byte(`{"from_path":"notes/todo.md","to_path":"archive/todo.md","version":99}`))
		case "/v1/sessions/append":
			s.Require().Equal(http.MethodPost, r.Method)
			_, _ = w.Write([]byte(`{"session_id":"sess-1","entries":[{"session_id":"sess-1","role":"user","content":"hello","metadata":{"source":"test"},"created_at":1710000000}]}`))
		case "/v1/sessions/sess-1":
			s.Require().Equal(http.MethodGet, r.Method)
			_, _ = w.Write([]byte(`{"session_id":"sess-1","entries":[{"session_id":"sess-1","role":"user","content":"hello","metadata":{"source":"test"},"created_at":1710000000}]}`))
		default:
			http.NotFound(w, r)
		}
	}))

	s.manager = NewManager(Config{
		BaseURL: s.server.URL,
		Enabled: true,
	})
}

func (s *ManagerTestSuite) TearDownTest() {
	if s.server != nil {
		s.server.Close()
	}
}

func (s *ManagerTestSuite) TestManagerUpsertResource() {
	resp, err := s.manager.UpsertResource(context.Background(), UpsertResourceRequest{
		URI:     "viking://resources/doc-1",
		Title:   "Doc 1",
		Content: "hello world",
		Layer:   LayerL2,
		Metadata: map[string]string{
			"title": "Doc 1",
		},
	})

	s.Require().NoError(err)
	s.Equal("viking://resources/doc-1", resp.Resource.URI)
	s.Equal(int32(3), resp.ChunksIndexed)
	s.Equal("viking://resources/doc-1", s.capturedUpsert.URI)
	s.Equal("hello world", s.capturedUpsert.Content)
	s.Equal("Doc 1", s.capturedUpsert.Title)
}

func (s *ManagerTestSuite) TestManagerSearch() {
	rerank := true
	resp, err := s.manager.Search(context.Background(), SearchRequest{
		Query:    "matched text",
		ScopeURI: "viking://resources/",
		TopK:     5,
		Rerank:   &rerank,
		Layer:    LayerL2,
	})

	s.Require().NoError(err)
	s.Require().Len(resp.Results, 1)
	s.Equal("viking://resources/doc-1", resp.Results[0].URI)
	s.Contains(s.capturedSearchBody.String(), `"scope_uri":"viking://resources/"`)
	s.Contains(s.capturedSearchBody.String(), `"rerank":true`)
}

func (s *ManagerTestSuite) TestManagerReadiness() {
	status, err := s.manager.Readiness(context.Background())
	s.Require().NoError(err)
	s.True(status.Ready)
	s.Equal("ok", status.Status)
}

func (s *ManagerTestSuite) TestManagerSyncWorkspace() {
	resp, err := s.manager.SyncWorkspace(context.Background(), WorkspaceSyncRequest{
		Root: "workspace",
		Path: "docs",
	})

	s.Require().NoError(err)
	s.Equal("workspace", resp.Root)
	s.Require().NotNil(resp.Prefix)
	s.Equal("docs", *resp.Prefix)
	s.Equal(int32(4), resp.IndexedResources)
	s.Equal(int32(2), resp.ReindexedResources)
}

func (s *ManagerTestSuite) TestManagerFileAndSessionContracts() {
	readResp, err := s.manager.ReadFile(context.Background(), FileReadRequest{
		Root: "workspace",
		Path: "notes/todo.md",
	})
	s.Require().NoError(err)
	s.Equal("notes/todo.md", readResp.Path)
	s.Equal(int64(42), readResp.Version)

	version := int64(42)
	writeResp, err := s.manager.WriteFile(context.Background(), FileWriteRequest{
		Root:    "workspace",
		Path:    "notes/todo.md",
		Content: "ship it",
		Version: &version,
	})
	s.Require().NoError(err)
	s.Equal(int64(43), writeResp.Version)

	deleteResp, err := s.manager.DeleteFile(context.Background(), FileDeleteRequest{
		Root:    "workspace",
		Path:    "notes/todo.md",
		Version: &version,
	})
	s.Require().NoError(err)
	s.True(deleteResp.Deleted)

	moveResp, err := s.manager.MoveFile(context.Background(), FileMoveRequest{
		Root:     "workspace",
		FromPath: "notes/todo.md",
		ToPath:   "archive/todo.md",
	})
	s.Require().NoError(err)
	s.Equal("archive/todo.md", moveResp.ToPath)
	s.Equal(int64(99), moveResp.Version)

	appendResp, err := s.manager.AppendSession(context.Background(), SessionAppendRequest{
		SessionID: "sess-1",
		Role:      "user",
		Content:   "hello",
		Metadata:  map[string]string{"source": "test"},
	})
	s.Require().NoError(err)
	s.Require().Len(appendResp.Entries, 1)
	s.Equal(int64(1710000000), appendResp.Entries[0].CreatedAt)

	sessionResp, err := s.manager.GetSession(context.Background(), "sess-1")
	s.Require().NoError(err)
	s.Require().Len(sessionResp.Entries, 1)
	s.Equal("hello", sessionResp.Entries[0].Content)
}

func TestManagerTestSuite(t *testing.T) {
	suite.Run(t, &ManagerTestSuite{})
}