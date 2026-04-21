package api

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"strings"
	"sync"

	"github.com/ai-engine/go/internal/config"
	"github.com/ai-engine/go/internal/supervisor"
	pb "github.com/ai-engine/proto/go"
	"google.golang.org/grpc"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/protobuf/types/known/emptypb"
	"log"
	"time"
)

type Server struct {
	pb.UnimplementedRuntimeServer
	pb.UnimplementedRagServer
	pb.UnimplementedTrainingServer
	pb.UnimplementedMCPServer

	mu         sync.RWMutex
	config     *config.Config
	supervisor *supervisor.Supervisor

	httpServer *http.Server
	grpcServer *grpc.Server
}

func NewServer(cfg *config.Config, sup *supervisor.Supervisor) *Server {
	return &Server{
		config:     cfg,
		supervisor: sup,
	}
}

func (s *Server) RegisterGRPC(server *grpc.Server) {
	pb.RegisterRuntimeServer(server, s)
	pb.RegisterRagServer(server, s)
	pb.RegisterTrainingServer(server, s)
	pb.RegisterMCPServer(server, s)
	grpc_health_v1.RegisterHealthServer(server, health.NewServer())
}

// Runtime gRPC handlers
func (s *Server) GetStatus(ctx context.Context, _ *emptypb.Empty) (*pb.RuntimeStatus, error) {
	return s.supervisor.Runtime.GetStatus(ctx, &emptypb.Empty{})
}

func (s *Server) ListModels(ctx context.Context, _ *emptypb.Empty) (*pb.ModelList, error) {
	return s.supervisor.Runtime.ListModels(ctx, &emptypb.Empty{})
}

func (s *Server) LoadModel(ctx context.Context, req *pb.LoadModelRequest) (*pb.ModelInfo, error) {
	return s.supervisor.Runtime.LoadModel(ctx, req)
}

func (s *Server) UnloadModel(ctx context.Context, req *pb.UnloadModelRequest) (*emptypb.Empty, error) {
	return s.supervisor.Runtime.UnloadModel(ctx, req)
}

func (s *Server) StreamInference(stream pb.Runtime_StreamInferenceServer) error {
	return s.supervisor.Runtime.StreamInference(stream.Context(), stream)
}

// RAG gRPC handlers
func (s *Server) UpsertDocument(ctx context.Context, req *pb.UpsertRequest) (*pb.UpsertResponse, error) {
	return s.supervisor.RAG.UpsertDocument(ctx, req)
}

func (s *Server) DeleteDocument(ctx context.Context, req *pb.DeleteRequest) (*emptypb.Empty, error) {
	return s.supervisor.RAG.DeleteDocument(ctx, req)
}

func (s *Server) Search(ctx context.Context, req *pb.SearchRequest) (*pb.SearchResponse, error) {
	return s.supervisor.RAG.Search(ctx, req)
}

func (s *Server) GetRagStatus(ctx context.Context, _ *emptypb.Empty) (*pb.RagStatus, error) {
	return s.supervisor.RAG.GetStatus(ctx, &emptypb.Empty{})
}

func (s *Server) ListDocuments(ctx context.Context, _ *emptypb.Empty) (*pb.DocumentList, error) {
	return s.supervisor.RAG.ListDocuments(ctx, &emptypb.Empty{})
}

// Training gRPC handlers
func (s *Server) StartRun(ctx context.Context, req *pb.TrainingRunRequest) (*pb.TrainingRun, error) {
	return s.supervisor.Training.StartRun(ctx, req)
}

func (s *Server) CancelRun(ctx context.Context, req *pb.CancelRequest) (*emptypb.Empty, error) {
	return s.supervisor.Training.CancelRun(ctx, req)
}

func (s *Server) ListRuns(ctx context.Context, _ *emptypb.Empty) (*pb.TrainingRunList, error) {
	return s.supervisor.Training.ListRuns(ctx, &emptypb.Empty{})
}

func (s *Server) ListArtifacts(ctx context.Context, req *pb.ArtifactsRequest) (*pb.ArtifactList, error) {
	return s.supervisor.Training.ListArtifacts(ctx, req)
}

func (s *Server) StreamLogs(req *pb.LogsRequest, stream pb.Training_StreamLogsServer) error {
	return s.supervisor.Training.StreamLogs(req, stream)
}

// MCP gRPC handlers
func (s *Server) Connect(ctx context.Context, req *pb.MCPConnectionRequest) (*pb.MCPConnection, error) {
	return s.supervisor.MCP.Connect(ctx, req)
}

func (s *Server) Disconnect(ctx context.Context, req *pb.DisconnectRequest) (*emptypb.Empty, error) {
	return s.supervisor.MCP.Disconnect(ctx, req)
}

func (s *Server) ListTools(ctx context.Context, req *pb.MCPConnectionRequest) (*pb.ToolList, error) {
	return s.supervisor.MCP.ListTools(ctx, req)
}

func (s *Server) CallTool(ctx context.Context, req *pb.CallToolRequest) (*pb.CallToolResponse, error) {
	return s.supervisor.MCP.CallTool(ctx, req)
}

// HTTP handlers for REST compatibility
func (s *Server) RegisterHTTP(mux *http.ServeMux) {
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/api/v1/status", s.handleStatus)
	mux.HandleFunc("/api/v1/runtime/models", s.handleListModels)
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write([]byte(`{"status":"ok"}`))
}

func (s *Server) handleStatus(w http.ResponseWriter, r *http.Request) {
	health := s.supervisor.Health()
	w.Header().Set("Content-Type", "application/json")
	fmt.Fprintf(w, `{"running":%v}`, health["running"])
}

func (s *Server) handleListModels(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	models, err := s.supervisor.Runtime.ListModels(ctx, &emptypb.Empty{})
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	fmt.Fprintf(w, `{"models":%d}`, len(models.Models))
}

func (s *Server) StartHTTP(addr string) error {
	mux := http.NewServeMux()
	s.RegisterHTTP(mux)

	s.httpServer = &http.Server{
		Addr:         addr,
		Handler:      mux,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
	}

	return s.httpServer.ListenAndServe()
}

func (s *Server) StartGRPC(addr string) error {
	lis, err := net.Listen("tcp", addr)
	if err != nil {
		return fmt.Errorf("failed to listen: %w", err)
	}

	s.grpcServer = grpc.NewServer(
		grpc.StreamInterceptor(loggingStreamInterceptor),
		grpc.UnaryInterceptor(loggingUnaryInterceptor),
	)
	s.RegisterGRPC(s.grpcServer)

	log.Printf("gRPC server listening on %s", addr)
	return s.grpcServer.Serve(lis)
}

func (s *Server) Stop() error {
	if s.httpServer != nil {
		s.httpServer.Shutdown(context.Background())
	}
	if s.grpcServer != nil {
		s.grpcServer.GracefulStop()
	}
	return nil
}

func loggingStreamInterceptor(srv interface{}, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
	start := time.Now()
	err := handler(srv, ss)
	log.Printf("stream: %s duration: %v error: %v", info.FullMethod, time.Since(start), err)
	return err
}

func loggingUnaryInterceptor(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
	start := time.Now()
	resp, err := handler(ctx, req)
	log.Printf("unary: %s duration: %v error: %v", info.FullMethod, time.Since(start), err)
	return resp, err
}

type responseWriter struct {
	http.ResponseWriter
	status int
}

func (w *responseWriter) WriteHeader(status int) {
	w.status = status
	w.ResponseWriter.WriteHeader(status)
}

func loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rw := &responseWriter{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(rw, r)
		log.Printf("%s %s %d %v", r.Method, r.URL.Path, rw.status, time.Since(start))
	})
}

func normalizeAddress(addr string) string {
	if !strings.Contains(addr, ":") {
		return ":" + addr
	}
	return addr
}
