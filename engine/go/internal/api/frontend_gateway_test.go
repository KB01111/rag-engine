package api_test

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ai-engine/go/internal/api"
	"github.com/ai-engine/go/internal/config"
	"github.com/ai-engine/go/internal/supervisor"
	pb "github.com/ai-engine/proto/go"
	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog"
	"github.com/stretchr/testify/suite"
	"google.golang.org/protobuf/types/known/emptypb"
)

type FrontendGatewaySuite struct {
	suite.Suite
	router *gin.Engine
	sup    *supervisor.Supervisor
}

func (s *FrontendGatewaySuite) SetupTest() {
	gin.SetMode(gin.TestMode)

	cfg, err := config.Load("")
	s.Require().NoError(err)
	cfg.Runtime.ModelsPath = s.T().TempDir()
	s.Require().NoError(os.WriteFile(filepath.Join(cfg.Runtime.ModelsPath, "demo.gguf"), []byte("weights"), 0644))
	cfg.Context.Enabled = false
	cfg.Context.AutoStart = false
	cfg.Server.CORS.Enabled = true

	s.sup = supervisor.NewSupervisor(cfg)
	s.sup.RAG = &fakeRAGService{}

	server := api.NewServer(cfg, s.sup, zerolog.New(zerolog.NewTestWriter(s.T())))
	s.router = gin.New()
	server.RegisterHTTP(s.router)
}

func (s *FrontendGatewaySuite) TearDownTest() {
	if s.sup != nil {
		_ = s.sup.Stop()
	}
}

func (s *FrontendGatewaySuite) TestRuntimeStatusLoadAndUnloadModel() {
	statusReq := httptest.NewRequest(http.MethodGet, "/api/v1/runtime/status", nil)
	statusResp := httptest.NewRecorder()
	s.router.ServeHTTP(statusResp, statusReq)

	s.Equal(http.StatusOK, statusResp.Code)
	s.Contains(statusResp.Body.String(), `"execution_mode"`)
	s.Contains(statusResp.Body.String(), `"status"`)

	loadBody := bytes.NewReader([]byte(`{"model_id":"demo.gguf"}`))
	loadReq := httptest.NewRequest(http.MethodPost, "/api/v1/runtime/models/load", loadBody)
	loadReq.Header.Set("Content-Type", "application/json")
	loadResp := httptest.NewRecorder()
	s.router.ServeHTTP(loadResp, loadReq)

	s.Equal(http.StatusOK, loadResp.Code)
	s.Contains(loadResp.Body.String(), `"id":"demo.gguf"`)
	s.Contains(loadResp.Body.String(), `"loaded":true`)

	unloadBody := bytes.NewReader([]byte(`{"model_id":"demo.gguf"}`))
	unloadReq := httptest.NewRequest(http.MethodPost, "/api/v1/runtime/models/unload", unloadBody)
	unloadReq.Header.Set("Content-Type", "application/json")
	unloadResp := httptest.NewRecorder()
	s.router.ServeHTTP(unloadResp, unloadReq)

	s.Equal(http.StatusOK, unloadResp.Code)
	s.Contains(unloadResp.Body.String(), `"unloaded":true`)
}

func (s *FrontendGatewaySuite) TestRuntimeInferenceStreamsTokenAndCompleteEvents() {
	body := bytes.NewReader([]byte(`{"model_id":"demo.gguf","prompt":"hello frontend"}`))
	req := httptest.NewRequest(http.MethodPost, "/api/v1/runtime/inference/stream", body)
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	s.router.ServeHTTP(resp, req)

	s.Equal(http.StatusOK, resp.Code)
	s.Equal("text/event-stream", strings.Split(resp.Header().Get("Content-Type"), ";")[0])
	s.Contains(resp.Body.String(), "event: token")
	s.Contains(resp.Body.String(), "event: complete")
	s.Contains(resp.Body.String(), `"complete":true`)
}

func (s *FrontendGatewaySuite) TestRuntimeInferenceWritesErrorEventAfterStreamStarts() {
	s.sup.Runtime = &streamThenErrorRuntime{}

	body := bytes.NewReader([]byte(`{"model_id":"demo.gguf","prompt":"hello frontend"}`))
	req := httptest.NewRequest(http.MethodPost, "/api/v1/runtime/inference/stream", body)
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	s.router.ServeHTTP(resp, req)

	s.Equal(http.StatusOK, resp.Code)
	s.Contains(resp.Body.String(), "event: token")
	s.Contains(resp.Body.String(), "event: error")
	s.Contains(resp.Body.String(), `"code":"backend_unavailable"`)
}

func (s *FrontendGatewaySuite) TestRuntimeInferenceValidationUsesErrorEnvelope() {
	req := httptest.NewRequest(http.MethodPost, "/api/v1/runtime/inference/stream", bytes.NewReader([]byte(`{"model_id":"demo.gguf"}`)))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	s.router.ServeHTTP(resp, req)

	s.Equal(http.StatusBadRequest, resp.Code)
	s.JSONEq(`{"error":{"code":"invalid_request","message":"prompt is required"}}`, resp.Body.String())
}

func (s *FrontendGatewaySuite) TestRAGEndpointsDelegateThroughManager() {
	upsertBody := bytes.NewReader([]byte(`{"document_id":"doc-2","content":"hello gateway","metadata":{"title":"Doc 2"}}`))
	upsertReq := httptest.NewRequest(http.MethodPost, "/api/v1/rag/documents", upsertBody)
	upsertReq.Header.Set("Content-Type", "application/json")
	upsertResp := httptest.NewRecorder()
	s.router.ServeHTTP(upsertResp, upsertReq)

	s.Equal(http.StatusOK, upsertResp.Code)
	s.Contains(upsertResp.Body.String(), `"document_id":"doc-2"`)
	s.Contains(upsertResp.Body.String(), `"chunks_indexed":2`)

	searchBody := bytes.NewReader([]byte(`{"query":"hello","top_k":1}`))
	searchReq := httptest.NewRequest(http.MethodPost, "/api/v1/rag/search", searchBody)
	searchReq.Header.Set("Content-Type", "application/json")
	searchResp := httptest.NewRecorder()
	s.router.ServeHTTP(searchResp, searchReq)

	s.Equal(http.StatusOK, searchResp.Code)
	s.Contains(searchResp.Body.String(), `"document_id":"doc-1"`)

	listReq := httptest.NewRequest(http.MethodGet, "/api/v1/rag/documents", nil)
	listResp := httptest.NewRecorder()
	s.router.ServeHTTP(listResp, listReq)

	s.Equal(http.StatusOK, listResp.Code)
	s.Contains(listResp.Body.String(), `"documents"`)

	deleteReq := httptest.NewRequest(http.MethodDelete, "/api/v1/rag/documents/doc-2", nil)
	deleteResp := httptest.NewRecorder()
	s.router.ServeHTTP(deleteResp, deleteReq)

	s.Equal(http.StatusOK, deleteResp.Code)
	s.Contains(deleteResp.Body.String(), `"deleted":true`)
}

func (s *FrontendGatewaySuite) TestCapabilitiesAndStagedStatuses() {
	req := httptest.NewRequest(http.MethodGet, "/api/v1/capabilities", nil)
	resp := httptest.NewRecorder()
	s.router.ServeHTTP(resp, req)

	s.Equal(http.StatusOK, resp.Code)
	s.Contains(resp.Body.String(), `"execution_mode"`)
	s.Contains(resp.Body.String(), `"runtime"`)
	s.Contains(resp.Body.String(), `"staged":true`)
	s.Contains(resp.Body.String(), `mcp.describe_connection`)

	trainingReq := httptest.NewRequest(http.MethodGet, "/api/v1/training/status", nil)
	trainingResp := httptest.NewRecorder()
	s.router.ServeHTTP(trainingResp, trainingReq)

	s.Equal(http.StatusOK, trainingResp.Code)
	s.Contains(trainingResp.Body.String(), `"staged":true`)

	mcpReq := httptest.NewRequest(http.MethodGet, "/api/v1/mcp/status", nil)
	mcpResp := httptest.NewRecorder()
	s.router.ServeHTTP(mcpResp, mcpReq)

	s.Equal(http.StatusOK, mcpResp.Code)
	s.Contains(mcpResp.Body.String(), `"staged":true`)
}

func (s *FrontendGatewaySuite) TestCORSAllowsLoopbackAndRejectsRemoteOrigins() {
	allowedReq := httptest.NewRequest(http.MethodOptions, "/api/v1/runtime/status", nil)
	allowedReq.Header.Set("Origin", "http://localhost:5173")
	allowedReq.Header.Set("Access-Control-Request-Method", http.MethodGet)
	allowedResp := httptest.NewRecorder()
	s.router.ServeHTTP(allowedResp, allowedReq)

	s.Equal(http.StatusNoContent, allowedResp.Code)
	s.Equal("http://localhost:5173", allowedResp.Header().Get("Access-Control-Allow-Origin"))

	deniedReq := httptest.NewRequest(http.MethodOptions, "/api/v1/runtime/status", nil)
	deniedReq.Header.Set("Origin", "https://example.com")
	deniedReq.Header.Set("Access-Control-Request-Method", http.MethodGet)
	deniedResp := httptest.NewRecorder()
	s.router.ServeHTTP(deniedResp, deniedReq)

	s.Equal(http.StatusForbidden, deniedResp.Code)
	s.Empty(deniedResp.Header().Get("Access-Control-Allow-Origin"))
}

func TestFrontendGatewaySuite(t *testing.T) {
	suite.Run(t, new(FrontendGatewaySuite))
}

type streamThenErrorRuntime struct{}

func (s *streamThenErrorRuntime) GetStatus(context.Context, *emptypb.Empty) (*pb.RuntimeStatus, error) {
	return &pb.RuntimeStatus{Healthy: true}, nil
}

func (s *streamThenErrorRuntime) ListModels(context.Context, *emptypb.Empty) (*pb.ModelList, error) {
	return &pb.ModelList{}, nil
}

func (s *streamThenErrorRuntime) LoadModel(context.Context, *pb.LoadModelRequest) (*pb.ModelInfo, error) {
	return &pb.ModelInfo{}, nil
}

func (s *streamThenErrorRuntime) UnloadModel(context.Context, *pb.UnloadModelRequest) (*emptypb.Empty, error) {
	return &emptypb.Empty{}, nil
}

func (s *streamThenErrorRuntime) LoadedModelCount() int {
	return 0
}

func (s *streamThenErrorRuntime) StreamInference(_ context.Context, stream pb.Runtime_StreamInferenceServer) error {
	if _, err := stream.Recv(); err != nil && !errors.Is(err, io.EOF) {
		return err
	}
	if err := stream.Send(&pb.InferenceResponse{
		Token:    "partial",
		Complete: false,
		Metrics:  map[string]string{"mode": "test"},
	}); err != nil {
		return err
	}
	return errors.New("backend boom")
}

type fakeRAGService struct{}

func (s *fakeRAGService) UpsertDocument(_ context.Context, req *pb.UpsertRequest) (*pb.UpsertResponse, error) {
	return &pb.UpsertResponse{
		DocumentId:    req.GetDocumentId(),
		ChunksIndexed: 2,
	}, nil
}

func (s *fakeRAGService) DeleteDocument(context.Context, *pb.DeleteRequest) (*emptypb.Empty, error) {
	return &emptypb.Empty{}, nil
}

func (s *fakeRAGService) Search(context.Context, *pb.SearchRequest) (*pb.SearchResponse, error) {
	return &pb.SearchResponse{
		Results: []*pb.SearchResult{
			{
				DocumentId: "doc-1",
				ChunkText:  "hello gateway",
				Score:      0.91,
				Metadata:   map[string]string{"title": "Doc 1"},
			},
		},
		QueryTimeMs: 1.5,
	}, nil
}

func (s *fakeRAGService) GetRagStatus(context.Context, *emptypb.Empty) (*pb.RagStatus, error) {
	return &pb.RagStatus{
		DocumentCount:  2,
		ChunkCount:     4,
		IndexSizeBytes: 2048,
		EmbeddingModel: "test-embed",
	}, nil
}

func (s *fakeRAGService) ListDocuments(context.Context, *emptypb.Empty) (*pb.DocumentList, error) {
	return &pb.DocumentList{
		Documents: []*pb.DocumentInfo{
			{Id: "doc-1", Title: "Doc 1", ChunkCount: 2},
		},
	}, nil
}

func (s *fakeRAGService) DocumentCount() int64 {
	return 2
}
