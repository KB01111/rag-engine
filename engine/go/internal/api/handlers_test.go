package api_test

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/ai-engine/go/internal/api"
	"github.com/ai-engine/go/internal/config"
	"github.com/ai-engine/go/internal/supervisor"
	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog"
	"github.com/stretchr/testify/suite"
)

// ExtendedServerSuite covers HTTP handlers not exercised by ServerSuite:
// list/upsert/delete context resources, list/write/delete/move context files,
// and error paths for handlers that require request bodies or query params.
type ExtendedServerSuite struct {
	suite.Suite
	server  *api.Server
	router  *gin.Engine
	sup     *supervisor.Supervisor
	ctxHTTP *httptest.Server
}

func (s *ExtendedServerSuite) SetupSuite() {
	gin.SetMode(gin.TestMode)
	logger := zerolog.New(zerolog.NewTestWriter(s.T()))

	cfg, err := config.Load("")
	s.NoError(err)

	s.ctxHTTP = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.URL.Path == "/health":
			_, _ = w.Write([]byte(`{"status":"ok","ready":true}`))
		case r.URL.Path == "/v1/status":
			_, _ = w.Write([]byte(`{"document_count":5,"chunk_count":20,"index_size_bytes":4096,"ready":true}`))
		case r.URL.Path == "/v1/resources" && r.Method == http.MethodGet:
			_, _ = w.Write([]byte(`{"resources":[{"uri":"viking://resources/test-doc","title":"Test Doc","layer":"l2","metadata":{"source":"unit-test"}}]}`))
		case r.URL.Path == "/v1/resources" && r.Method == http.MethodPost:
			_, _ = w.Write([]byte(`{"resource":{"uri":"viking://resources/new-doc","title":"New Doc","layer":"l2","metadata":{}},"chunks_indexed":3}`))
		case r.URL.Path == "/v1/resources" && r.Method == http.MethodDelete:
			w.WriteHeader(http.StatusNoContent)
		case r.URL.Path == "/v1/search":
			_, _ = w.Write([]byte(`{"results":[],"query_time_ms":0.5}`))
		case r.URL.Path == "/v1/workspaces/sync":
			_, _ = w.Write([]byte(`{"root":"workspace","indexed_resources":1,"reindexed_resources":0,"deleted_resources":0,"skipped_files":0}`))
		case r.URL.Path == "/v1/files/list":
			_, _ = w.Write([]byte(`{"entries":[{"name":"readme.md","path":"docs/readme.md","is_dir":false,"size_bytes":512}]}`))
		case r.URL.Path == "/v1/files/read":
			_, _ = w.Write([]byte(`{"path":"docs/readme.md","content":"file content","version":1}`))
		case r.URL.Path == "/v1/files/write":
			_, _ = w.Write([]byte(`{"path":"docs/readme.md","version":2}`))
		case r.URL.Path == "/v1/files/delete":
			_, _ = w.Write([]byte(`{"path":"docs/readme.md","deleted":true}`))
		case r.URL.Path == "/v1/files/move":
			_, _ = w.Write([]byte(`{"from_path":"docs/readme.md","to_path":"archive/readme.md","version":3}`))
		case r.URL.Path == "/v1/sessions/append":
			_, _ = w.Write([]byte(`{"session_id":"sess-x","entries":[{"session_id":"sess-x","role":"user","content":"hi","created_at":1710000000}]}`))
		case r.URL.Path == "/v1/sessions/sess-x":
			_, _ = w.Write([]byte(`{"session_id":"sess-x","entries":[{"session_id":"sess-x","role":"user","content":"hi","created_at":1710000000}]}`))
		default:
			http.NotFound(w, r)
		}
	}))

	cfg.Context.Enabled = true
	cfg.Context.AutoStart = false
	cfg.Context.ServiceURL = s.ctxHTTP.URL

	s.sup = supervisor.NewSupervisor(cfg)
	s.NoError(s.sup.Start())

	s.server = api.NewServer(cfg, s.sup, logger)
	s.router = gin.New()
	s.server.RegisterHTTP(s.router)
}

func (s *ExtendedServerSuite) TearDownSuite() {
	if s.sup != nil {
		s.sup.Stop()
	}
	if s.ctxHTTP != nil {
		s.ctxHTTP.Close()
	}
}

// --- Context resource endpoints ---

func (s *ExtendedServerSuite) TestListContextResources() {
	req := httptest.NewRequest(http.MethodGet, "/api/v1/context/resources", nil)
	w := httptest.NewRecorder()
	s.router.ServeHTTP(w, req)

	s.Equal(http.StatusOK, w.Code)
	s.Contains(w.Body.String(), "viking://resources/test-doc")
	s.Contains(w.Body.String(), "Test Doc")
}

func (s *ExtendedServerSuite) TestUpsertContextResource() {
	body, err := json.Marshal(map[string]any{
		"uri":     "viking://resources/new-doc",
		"title":   "New Doc",
		"content": "some content to index",
		"layer":   "l2",
	})
	s.Require().NoError(err)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/context/resources", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	s.router.ServeHTTP(w, req)

	s.Equal(http.StatusOK, w.Code)
	s.Contains(w.Body.String(), "chunks_indexed")
}

func (s *ExtendedServerSuite) TestUpsertContextResource_BadJSON() {
	req := httptest.NewRequest(http.MethodPost, "/api/v1/context/resources", bytes.NewReader([]byte("not json")))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	s.router.ServeHTTP(w, req)

	s.Equal(http.StatusBadRequest, w.Code)
	s.Contains(w.Body.String(), "error")
}

func (s *ExtendedServerSuite) TestDeleteContextResource() {
	req := httptest.NewRequest(http.MethodDelete, "/api/v1/context/resources?uri=viking%3A%2F%2Fresources%2Ftest-doc", nil)
	w := httptest.NewRecorder()
	s.router.ServeHTTP(w, req)

	s.Equal(http.StatusNoContent, w.Code)
}

func (s *ExtendedServerSuite) TestDeleteContextResource_MissingURI() {
	req := httptest.NewRequest(http.MethodDelete, "/api/v1/context/resources", nil)
	w := httptest.NewRecorder()
	s.router.ServeHTTP(w, req)

	s.Equal(http.StatusBadRequest, w.Code)
	s.Contains(w.Body.String(), "missing uri")
}

// --- Context file endpoints ---

func (s *ExtendedServerSuite) TestListContextFiles() {
	body, err := json.Marshal(map[string]any{
		"root": "workspace",
		"path": "docs",
	})
	s.Require().NoError(err)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/context/files/list", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	s.router.ServeHTTP(w, req)

	s.Equal(http.StatusOK, w.Code)
	s.Contains(w.Body.String(), "readme.md")
}

func (s *ExtendedServerSuite) TestListContextFiles_BadJSON() {
	req := httptest.NewRequest(http.MethodPost, "/api/v1/context/files/list", bytes.NewReader([]byte("{")))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	s.router.ServeHTTP(w, req)

	s.Equal(http.StatusBadRequest, w.Code)
}

func (s *ExtendedServerSuite) TestWriteContextFile() {
	body, err := json.Marshal(map[string]any{
		"root":    "workspace",
		"path":    "docs/readme.md",
		"content": "updated content",
	})
	s.Require().NoError(err)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/context/files/write", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	s.router.ServeHTTP(w, req)

	s.Equal(http.StatusOK, w.Code)
	s.Contains(w.Body.String(), "docs/readme.md")
	s.Contains(w.Body.String(), "version")
}

func (s *ExtendedServerSuite) TestWriteContextFile_BadJSON() {
	req := httptest.NewRequest(http.MethodPost, "/api/v1/context/files/write", bytes.NewReader([]byte("bad")))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	s.router.ServeHTTP(w, req)

	s.Equal(http.StatusBadRequest, w.Code)
}

func (s *ExtendedServerSuite) TestDeleteContextFile() {
	body, err := json.Marshal(map[string]any{
		"root": "workspace",
		"path": "docs/readme.md",
	})
	s.Require().NoError(err)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/context/files/delete", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	s.router.ServeHTTP(w, req)

	s.Equal(http.StatusOK, w.Code)
	s.Contains(w.Body.String(), "deleted")
	s.Contains(w.Body.String(), "true")
}

func (s *ExtendedServerSuite) TestDeleteContextFile_BadJSON() {
	req := httptest.NewRequest(http.MethodPost, "/api/v1/context/files/delete", bytes.NewReader([]byte("x")))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	s.router.ServeHTTP(w, req)

	s.Equal(http.StatusBadRequest, w.Code)
}

func (s *ExtendedServerSuite) TestMoveContextFile() {
	body, err := json.Marshal(map[string]any{
		"root":      "workspace",
		"from_path": "docs/readme.md",
		"to_path":   "archive/readme.md",
	})
	s.Require().NoError(err)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/context/files/move", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	s.router.ServeHTTP(w, req)

	s.Equal(http.StatusOK, w.Code)
	s.Contains(w.Body.String(), "from_path")
	s.Contains(w.Body.String(), "archive/readme.md")
}

func (s *ExtendedServerSuite) TestMoveContextFile_BadJSON() {
	req := httptest.NewRequest(http.MethodPost, "/api/v1/context/files/move", bytes.NewReader([]byte("")))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	s.router.ServeHTTP(w, req)

	s.Equal(http.StatusBadRequest, w.Code)
}

// --- Regression: context search still works with layer param ---

func (s *ExtendedServerSuite) TestContextSearch_L0Layer() {
	body, err := json.Marshal(map[string]any{
		"query":     "test query",
		"scope_uri": "viking://resources/",
		"top_k":     3,
		"layer":     "l0",
	})
	s.Require().NoError(err)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/context/search", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	s.router.ServeHTTP(w, req)

	s.Equal(http.StatusOK, w.Code)
	s.Contains(w.Body.String(), "results")
}

func (s *ExtendedServerSuite) TestContextSearch_L1Layer() {
	body, err := json.Marshal(map[string]any{
		"query": "l1 query",
		"top_k": 1,
		"layer": "l1",
	})
	s.Require().NoError(err)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/context/search", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	s.router.ServeHTTP(w, req)

	s.Equal(http.StatusOK, w.Code)
}

func (s *ExtendedServerSuite) TestContextSearch_BadJSON() {
	req := httptest.NewRequest(http.MethodPost, "/api/v1/context/search", bytes.NewReader([]byte("not-json")))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	s.router.ServeHTTP(w, req)

	s.Equal(http.StatusBadRequest, w.Code)
}

// --- Regression: append session bad JSON ---

func (s *ExtendedServerSuite) TestAppendContextSession_BadJSON() {
	req := httptest.NewRequest(http.MethodPost, "/api/v1/context/sessions/append", bytes.NewReader([]byte("{")))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	s.router.ServeHTTP(w, req)

	s.Equal(http.StatusBadRequest, w.Code)
}

func TestExtendedServerSuite(t *testing.T) {
	suite.Run(t, new(ExtendedServerSuite))
}
