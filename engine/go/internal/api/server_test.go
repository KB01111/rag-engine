package api_test

import (
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
}

func (s *ServerSuite) SetupSuite() {
	gin.SetMode(gin.TestMode)
	logger := zerolog.New(zerolog.NewTestWriter(s.T()))

	cfg, err := config.Load("")
	s.NoError(err)
	cfg.Daemon.Required = false
	s.cfg = cfg

	s.sup = supervisor.NewSupervisor(cfg)
	s.NoError(s.sup.Start())

	s.server = api.NewServer(cfg, s.sup, logger)
	s.router = gin.New()
	s.server.RegisterHTTP(s.router)

	s.httpAddr = cfg.Addr()
}

func (s *ServerSuite) TearDownSuite() {
	s.sup.Stop()
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

func TestServerSuite(t *testing.T) {
	suite.Run(t, new(ServerSuite))
}
