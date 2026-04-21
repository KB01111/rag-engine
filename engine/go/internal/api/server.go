package api

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"sync"
	"time"

	"github.com/ai-engine/go/internal/config"
	"github.com/ai-engine/go/internal/supervisor"
	pb "github.com/ai-engine/proto/go"
	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog"
	"google.golang.org/grpc"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/protobuf/types/known/emptypb"
)

type Server struct {
	pb.UnimplementedRuntimeServer
	pb.UnimplementedRagServer
	pb.UnimplementedTrainingServer
	pb.UnimplementedMCPServer

	mu         sync.RWMutex
	config     *config.Config
	supervisor *supervisor.Supervisor
	log        zerolog.Logger

	httpServer *http.Server
	grpcServer *grpc.Server
}

func NewServer(cfg *config.Config, sup *supervisor.Supervisor, log zerolog.Logger) *Server {
	return &Server{
		config:     cfg,
		supervisor: sup,
		log:        log,
	}
}

func (s *Server) RegisterGRPC(server *grpc.Server) {
	pb.RegisterRuntimeServer(server, s)
	pb.RegisterRagServer(server, s)
	pb.RegisterTrainingServer(server, s)
	pb.RegisterMCPServer(server, s)
	grpc_health_v1.RegisterHealthServer(server, health.NewServer())
}

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

func (s *Server) RegisterHTTP(router *gin.Engine) {
	router.GET("/health", s.handleHealth)
	router.GET("/api/v1/status", s.handleStatus)
	router.GET("/api/v1/runtime/models", s.handleListModels)
}

func (s *Server) handleHealth(c *gin.Context) {
	c.JSON(200, gin.H{"status": "ok"})
}

func (s *Server) handleStatus(c *gin.Context) {
	health := s.supervisor.Health()
	c.JSON(200, gin.H{"running": health["running"]})
}

func (s *Server) handleListModels(c *gin.Context) {
	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()

	models, err := s.supervisor.Runtime.ListModels(ctx, &emptypb.Empty{})
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	c.JSON(200, gin.H{"models": len(models.Models)})
}

func (s *Server) StartHTTP(addr string, router *gin.Engine) error {
	s.httpServer = &http.Server{
		Addr:         addr,
		Handler:      router,
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
		grpc.StreamInterceptor(s.logStreamInterceptor),
		grpc.UnaryInterceptor(s.logUnaryInterceptor),
	)
	s.RegisterGRPC(s.grpcServer)

	s.log.Info().Str("addr", addr).Msg("gRPC server listening")
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

func (s *Server) logStreamInterceptor(srv interface{}, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
	start := time.Now()
	err := handler(srv, ss)
	s.log.Debug().Str("method", info.FullMethod).Dur("duration", time.Since(start)).Err(err).Msg("stream")
	return err
}

func (s *Server) logUnaryInterceptor(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
	start := time.Now()
	resp, err := handler(ctx, req)
	s.log.Debug().Str("method", info.FullMethod).Dur("duration", time.Since(start)).Err(err).Msg("unary")
	return resp, err
}
