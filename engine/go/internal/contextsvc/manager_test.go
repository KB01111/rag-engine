package contextsvc

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	pb "github.com/ai-engine/proto/go"
	"github.com/stretchr/testify/suite"
	"google.golang.org/protobuf/types/known/emptypb"
)

type ManagerTestSuite struct {
	suite.Suite
	server  *httptest.Server
	manager *Manager
	// Captured request buffers for validation
	capturedUpsert     UpsertResourceRequest
	capturedSearchBody bytes.Buffer
	capturedFileRead   FileReadRequest
	capturedFileWrite  FileWriteRequest
	capturedFileDelete FileDeleteRequest
	capturedFileMove   FileMoveRequest
	httpCalls          int
}

type daemonContextStub struct{}

func (s *daemonContextStub) GetContextStatus(context.Context, *emptypb.Empty) (*pb.ContextStatus, error) {
	return &pb.ContextStatus{
		DocumentCount: 11,
		ChunkCount:    22,
		Ready:         true,
	}, nil
}

func (s *daemonContextStub) ListResources(context.Context, *emptypb.Empty) (*pb.ContextResourceList, error) {
	return &pb.ContextResourceList{
		Resources: []*pb.ContextResource{
			{
				Uri:   "viking://resources/graph/project-x",
				Title: "Project X",
				Layer: "l1",
			},
		},
	}, nil
}

func (s *daemonContextStub) UpsertResource(context.Context, *pb.ContextUpsertResourceRequest) (*pb.ContextUpsertResourceResponse, error) {
	return &pb.ContextUpsertResourceResponse{
		Resource: &pb.ContextResource{
			Uri:   "viking://resources/graph/project-x",
			Title: "Project X",
			Layer: "l2",
		},
		ChunksIndexed: 2,
	}, nil
}

func (s *daemonContextStub) DeleteResource(context.Context, *pb.ContextDeleteResourceRequest) (*emptypb.Empty, error) {
	return &emptypb.Empty{}, nil
}

func (s *daemonContextStub) SearchContext(_ context.Context, req *pb.ContextSearchRequest) (*pb.ContextSearchResponse, error) {
	return &pb.ContextSearchResponse{
		Results: []*pb.ContextSearchResult{
			{
				Uri:        "viking://resources/graph/project-x",
				DocumentId: "project-x",
				ChunkText:  "Project X uses Dragonfly and prefers local-first context.",
				Score:      0.99,
				Metadata: map[string]string{
					"query": req.GetQuery(),
				},
				Layer: "l2",
			},
		},
		QueryTimeMs: 0.7,
	}, nil
}

func (s *daemonContextStub) SyncWorkspace(context.Context, *pb.ContextWorkspaceSyncRequest) (*pb.ContextWorkspaceSyncResponse, error) {
	return &pb.ContextWorkspaceSyncResponse{Root: "workspace"}, nil
}

func (s *daemonContextStub) ListFiles(context.Context, *pb.ContextFileListRequest) (*pb.ContextFileListResponse, error) {
	return &pb.ContextFileListResponse{}, nil
}

func (s *daemonContextStub) ReadFile(context.Context, *pb.ContextFileReadRequest) (*pb.ContextFileReadResponse, error) {
	return &pb.ContextFileReadResponse{}, nil
}

func (s *daemonContextStub) WriteFile(context.Context, *pb.ContextFileWriteRequest) (*pb.ContextFileWriteResponse, error) {
	return &pb.ContextFileWriteResponse{}, nil
}

func (s *daemonContextStub) DeleteFile(context.Context, *pb.ContextFileDeleteRequest) (*pb.ContextFileDeleteResponse, error) {
	return &pb.ContextFileDeleteResponse{}, nil
}

func (s *daemonContextStub) MoveFile(context.Context, *pb.ContextFileMoveRequest) (*pb.ContextFileMoveResponse, error) {
	return &pb.ContextFileMoveResponse{}, nil
}

func (s *daemonContextStub) AppendSession(_ context.Context, req *pb.ContextSessionAppendRequest) (*pb.ContextSessionHistory, error) {
	return &pb.ContextSessionHistory{
		SessionId: req.GetSessionId(),
		Entries: []*pb.ContextSessionEntry{
			{
				SessionId: req.GetSessionId(),
				Role:      req.GetRole(),
				Content:   req.GetContent(),
				Metadata:  req.GetMetadata(),
			},
		},
	}, nil
}

func (s *daemonContextStub) GetSession(_ context.Context, req *pb.ContextSessionGetRequest) (*pb.ContextSessionHistory, error) {
	return &pb.ContextSessionHistory{
		SessionId: req.GetSessionId(),
		Entries: []*pb.ContextSessionEntry{
			{
				SessionId: req.GetSessionId(),
				Role:      "assistant",
				Content:   "Project X prefers Dragonfly.",
			},
		},
	}, nil
}

func (s *ManagerTestSuite) SetupTest() {
	s.capturedSearchBody.Reset()
	s.httpCalls = 0
	s.server = nil
	if strings.Contains(s.T().Name(), "UsesDaemonContextClientWhenAttached") {
		s.manager = NewManager(Config{
			Enabled: true,
		})
		return
	}
	s.server = newIPv4Server(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.httpCalls++
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

func (s *ManagerTestSuite) TestManagerUsesDaemonContextClientWhenAttached() {
	s.manager.SetDaemonContextClient(&daemonContextStub{})

	status, err := s.manager.Status(context.Background())
	s.Require().NoError(err)
	s.Equal(int64(11), status.DocumentCount)
	s.Equal(int64(22), status.ChunkCount)

	searchResp, err := s.manager.Search(context.Background(), SearchRequest{
		Query: "Project X",
		TopK:  5,
	})
	s.Require().NoError(err)
	s.Require().Len(searchResp.Results, 1)
	s.Equal("project-x", searchResp.Results[0].DocumentID)
	s.Contains(searchResp.Results[0].ChunkText, "Dragonfly")

	appendResp, err := s.manager.AppendSession(context.Background(), SessionAppendRequest{
		SessionID: "sess-daemon",
		Role:      "user",
		Content:   "Remember Project X prefers Dragonfly.",
		Metadata:  map[string]string{"source": "daemon"},
	})
	s.Require().NoError(err)
	s.Require().Len(appendResp.Entries, 1)
	s.Equal("sess-daemon", appendResp.SessionID)

	sessionResp, err := s.manager.GetSession(context.Background(), "sess-daemon")
	s.Require().NoError(err)
	s.Require().Len(sessionResp.Entries, 1)
	s.Contains(sessionResp.Entries[0].Content, "Dragonfly")

	s.Equal(0, s.httpCalls)
}

func (s *ManagerTestSuite) TestResolveBinaryPathFromRootsFindsBundleBinary() {
	root := s.T().TempDir()
	binDir := filepath.Join(root, "bin")
	s.Require().NoError(os.MkdirAll(binDir, 0755))

	expected := filepath.Join(binDir, "context_server.exe")
	s.Require().NoError(os.WriteFile(expected, []byte("stub"), 0644))

	actual := resolveBinaryPathFromRoots("context_server", root, binDir)
	s.Equal(expected, actual)
}

func TestManagerTestSuite(t *testing.T) {
	suite.Run(t, &ManagerTestSuite{})
}
