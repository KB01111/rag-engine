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

type ServerSuite struct {
	suite.Suite
	server   *api.Server
	router   *gin.Engine
	cfg      *config.Config
	sup      *supervisor.Supervisor
	httpAddr string
	ctxHTTP  *httptest.Server
}

func (s *ServerSuite) SetupSuite() {
	gin.SetMode(gin.TestMode)
	logger := zerolog.New(zerolog.NewTestWriter(s.T()))

	cfg, err := config.Load("")
	s.NoError(err)
	s.cfg = cfg

	s.ctxHTTP = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.URL.Path == "/health":
			_, _ = w.Write([]byte(`{"status":"ok","ready":true}`))
		case r.URL.Path == "/v1/status":
			_, _ = w.Write([]byte(`{"document_count":1,"chunk_count":2,"index_size_bytes":1024,"ready":true}`))
		case r.URL.Path == "/v1/resources" && r.Method == http.MethodGet:
			_, _ = w.Write([]byte(`{"resources":[{"uri":"viking://resources/doc-1","title":"Doc 1","layer":"l2","metadata":{"title":"Doc 1"}}]}`))
		case r.URL.Path == "/v1/search":
			_, _ = w.Write([]byte(`{"results":[{"uri":"viking://resources/doc-1","document_id":"doc-1","chunk_text":"hello world","score":0.9,"metadata":{"title":"Doc 1"},"layer":"l2"}],"query_time_ms":1.2}`))
		case r.URL.Path == "/v1/workspaces/sync":
			_, _ = w.Write([]byte(`{"root":"workspace","prefix":"docs","indexed_resources":3,"reindexed_resources":1,"deleted_resources":0,"skipped_files":0}`))
		case r.URL.Path == "/v1/files/read":
			_, _ = w.Write([]byte(`{"path":"docs/readme.md","content":"hello world","version":7}`))
		case r.URL.Path == "/v1/files/write":
			_, _ = w.Write([]byte(`{"path":"docs/readme.md","version":8}`))
		case r.URL.Path == "/v1/files/delete":
			_, _ = w.Write([]byte(`{"path":"docs/readme.md","deleted":true}`))
		case r.URL.Path == "/v1/files/move":
			_, _ = w.Write([]byte(`{"from_path":"docs/readme.md","to_path":"archive/readme.md","version":9}`))
		case r.URL.Path == "/v1/sessions/append":
			_, _ = w.Write([]byte(`{"session_id":"sess-1","entries":[{"session_id":"sess-1","role":"user","content":"hello","metadata":{"source":"test"},"created_at":1710000000}]}`))
		case r.URL.Path == "/v1/sessions/sess-1":
			_, _ = w.Write([]byte(`{"session_id":"sess-1","entries":[{"session_id":"sess-1","role":"user","content":"hello","metadata":{"source":"test"},"created_at":1710000000}]}`))
		default:
			http.NotFound(w, r)
		}
	}))
	s.cfg.Context.Enabled = true
	s.cfg.Context.AutoStart = false
	s.cfg.Context.ServiceURL = s.ctxHTTP.URL

	s.sup = supervisor.NewSupervisor(cfg)
	s.NoError(s.sup.Start())

	s.server = api.NewServer(cfg, s.sup, logger)
	s.router = gin.New()
	s.server.RegisterHTTP(s.router)

	s.httpAddr = cfg.Addr()
}

func (s *ServerSuite) TearDownSuite() {
	s.sup.Stop()
	if s.ctxHTTP != nil {
		s.ctxHTTP.Close()
	}
}

func (s *ServerSuite) TestHealthEndpoint() {
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()
	s.router.ServeHTTP(w, req)

	s.Equal(http.StatusOK, w.Code)
	s.Contains(w.Body.String(), "ok")
}

func (s *ServerSuite) TestStatusEndpoint() {
	req := httptest.NewRequest(http.MethodGet, "/api/v1/status", nil)
	w := httptest.NewRecorder()
	s.router.ServeHTTP(w, req)

	s.Equal(http.StatusOK, w.Code)
	s.Contains(w.Body.String(), "running")
}

func (s *ServerSuite) TestModelsEndpoint() {
	req := httptest.NewRequest(http.MethodGet, "/api/v1/runtime/models", nil)
	w := httptest.NewRecorder()
	s.router.ServeHTTP(w, req)

	s.Equal(http.StatusOK, w.Code)
	s.Contains(w.Body.String(), "models")
}

func (s *ServerSuite) TestContextStatusEndpoint() {
	req := httptest.NewRequest(http.MethodGet, "/api/v1/context/status", nil)
	w := httptest.NewRecorder()
	s.router.ServeHTTP(w, req)

	s.Equal(http.StatusOK, w.Code)
	s.Contains(w.Body.String(), "document_count")
	s.Contains(w.Body.String(), "chunk_count")
}

func (s *ServerSuite) TestContextSearchEndpoint() {
	body, err := json.Marshal(map[string]any{
		"query":     "hello world",
		"scope_uri": "viking://resources/",
		"top_k":     5,
		"layer":     "l2",
	})
	s.Require().NoError(err)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/context/search", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	s.router.ServeHTTP(w, req)

	s.Equal(http.StatusOK, w.Code)
	s.Contains(w.Body.String(), "hello world")
	s.Contains(w.Body.String(), "doc-1")
}

func (s *ServerSuite) TestContextWorkspaceSyncEndpoint() {
	body, err := json.Marshal(map[string]any{
		"root": "workspace",
		"path": "docs",
	})
	s.Require().NoError(err)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/context/workspaces/sync", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	s.router.ServeHTTP(w, req)

	s.Equal(http.StatusOK, w.Code)
	s.Contains(w.Body.String(), "indexed_resources")
	s.Contains(w.Body.String(), "workspace")
}

func (s *ServerSuite) TestContextFileReadEndpoint() {
	body, err := json.Marshal(map[string]any{
		"root": "workspace",
		"path": "docs/readme.md",
	})
	s.Require().NoError(err)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/context/files/read", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	s.router.ServeHTTP(w, req)

	s.Equal(http.StatusOK, w.Code)
	s.Contains(w.Body.String(), "docs/readme.md")
	s.Contains(w.Body.String(), "hello world")
}

func (s *ServerSuite) TestContextSessionEndpoint() {
	body, err := json.Marshal(map[string]any{
		"session_id": "sess-1",
		"role":       "user",
		"content":    "hello",
		"metadata": map[string]string{
			"source": "test",
		},
	})
	s.Require().NoError(err)

	appendReq := httptest.NewRequest(http.MethodPost, "/api/v1/context/sessions/append", bytes.NewReader(body))
	appendReq.Header.Set("Content-Type", "application/json")
	appendResp := httptest.NewRecorder()
	s.router.ServeHTTP(appendResp, appendReq)

	s.Equal(http.StatusOK, appendResp.Code)
	s.Contains(appendResp.Body.String(), "created_at")

	getReq := httptest.NewRequest(http.MethodGet, "/api/v1/context/sessions/sess-1", nil)
	getResp := httptest.NewRecorder()
	s.router.ServeHTTP(getResp, getReq)

	s.Equal(http.StatusOK, getResp.Code)
	s.Contains(getResp.Body.String(), "sess-1")
	s.Contains(getResp.Body.String(), "hello")
}

func TestServerSuite(t *testing.T) {
	suite.Run(t, new(ServerSuite))
}
