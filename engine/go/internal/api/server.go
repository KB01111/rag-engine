package api

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"sync"
	"time"

	"github.com/ai-engine/go/internal/config"
	contextsvc "github.com/ai-engine/go/internal/contextsvc"
	"github.com/ai-engine/go/internal/supervisor"
	pb "github.com/ai-engine/proto/go"
	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog"
	"google.golang.org/grpc"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/protobuf/types/known/emptypb"
	"google.golang.org/protobuf/types/known/timestamppb"
)

type Server struct {
	pb.UnimplementedRuntimeServer
	pb.UnimplementedRagServer
	pb.UnimplementedContextServer
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
	pb.RegisterContextServer(server, s)
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

func (s *Server) GetContextStatus(ctx context.Context, _ *emptypb.Empty) (*pb.ContextStatus, error) {
	status, err := s.supervisor.Context.Status(ctx)
	if err != nil {
		return nil, err
	}
	return &pb.ContextStatus{
		DocumentCount:  status.DocumentCount,
		ChunkCount:     status.ChunkCount,
		IndexSizeBytes: status.IndexSizeBytes,
		EmbeddingModel: status.EmbeddingModel,
		Ready:          status.Ready,
		ManagedRoots:   status.ManagedRoots,
	}, nil
}

func (s *Server) ListResources(ctx context.Context, _ *emptypb.Empty) (*pb.ContextResourceList, error) {
	resp, err := s.supervisor.Context.ListResources(ctx)
	if err != nil {
		return nil, err
	}

	resources := make([]*pb.ContextResource, 0, len(resp.Resources))
	for _, resource := range resp.Resources {
		resources = append(resources, toProtoContextResource(resource))
	}

	return &pb.ContextResourceList{Resources: resources}, nil
}

func (s *Server) UpsertResource(ctx context.Context, req *pb.ContextUpsertResourceRequest) (*pb.ContextUpsertResourceResponse, error) {
	layer := layerFromProto(req.GetLayer())
	if layer == "" {
		layer = contextsvc.LayerL2
	}

	request := contextsvc.UpsertResourceRequest{
		URI:         req.GetUri(),
		Title:       req.GetTitle(),
		Content:     req.GetContent(),
		Layer:       layer,
		Metadata:    req.GetMetadata(),
		PreviousURI: req.GetPreviousUri(),
	}

	resp, err := s.supervisor.Context.UpsertResource(ctx, request)
	if err != nil {
		return nil, err
	}

	return &pb.ContextUpsertResourceResponse{
		Resource:      toProtoContextResource(resp.Resource),
		ChunksIndexed: resp.ChunksIndexed,
	}, nil
}

func (s *Server) DeleteResource(ctx context.Context, req *pb.ContextDeleteResourceRequest) (*emptypb.Empty, error) {
	if err := s.supervisor.Context.DeleteResource(ctx, req.GetUri()); err != nil {
		return nil, err
	}
	return &emptypb.Empty{}, nil
}

func (s *Server) SearchContext(ctx context.Context, req *pb.ContextSearchRequest) (*pb.ContextSearchResponse, error) {
	searchReq := contextsvc.SearchRequest{
		Query:    req.GetQuery(),
		ScopeURI: req.GetScopeUri(),
		TopK:     int(req.GetTopK()),
		Filters:  req.GetFilters(),
		Layer:    layerFromProto(req.GetLayer()),
	}
	if req.Rerank != nil {
		searchReq.Rerank = req.GetRerank()
	}

	resp, err := s.supervisor.Context.Search(ctx, searchReq)
	if err != nil {
		return nil, err
	}

	results := make([]*pb.ContextSearchResult, 0, len(resp.Results))
	for _, hit := range resp.Results {
		results = append(results, &pb.ContextSearchResult{
			Uri:        hit.URI,
			DocumentId: hit.DocumentID,
			ChunkText:  hit.ChunkText,
			Score:      hit.Score,
			Metadata:   hit.Metadata,
			Layer:      string(hit.Layer),
		})
	}

	return &pb.ContextSearchResponse{
		Results:     results,
		QueryTimeMs: resp.QueryTimeMs,
	}, nil
}

func (s *Server) SyncWorkspace(ctx context.Context, req *pb.ContextWorkspaceSyncRequest) (*pb.ContextWorkspaceSyncResponse, error) {
	resp, err := s.supervisor.Context.SyncWorkspace(ctx, contextsvc.WorkspaceSyncRequest{
		Root: req.GetRoot(),
		Path: req.GetPath(),
	})
	if err != nil {
		return nil, err
	}

	return &pb.ContextWorkspaceSyncResponse{
		Root:               resp.Root,
		Prefix:             resp.Prefix,
		IndexedResources:   resp.IndexedResources,
		ReindexedResources: resp.ReindexedResources,
		DeletedResources:   resp.DeletedResources,
		SkippedFiles:       resp.SkippedFiles,
	}, nil
}

func (s *Server) ListFiles(ctx context.Context, req *pb.ContextFileListRequest) (*pb.ContextFileListResponse, error) {
	resp, err := s.supervisor.Context.ListFiles(ctx, contextsvc.FileListRequest{
		Root: req.GetRoot(),
		Path: req.GetPath(),
	})
	if err != nil {
		return nil, err
	}

	entries := make([]*pb.ContextFileEntry, 0, len(resp.Entries))
	for _, entry := range resp.Entries {
		entries = append(entries, &pb.ContextFileEntry{
			Name:      entry.Name,
			Path:      entry.Path,
			IsDir:     entry.IsDir,
			SizeBytes: entry.SizeBytes,
			Version:   entry.Version,
		})
	}

	return &pb.ContextFileListResponse{Entries: entries}, nil
}

func (s *Server) ReadFile(ctx context.Context, req *pb.ContextFileReadRequest) (*pb.ContextFileReadResponse, error) {
	resp, err := s.supervisor.Context.ReadFile(ctx, contextsvc.FileReadRequest{
		Root: req.GetRoot(),
		Path: req.GetPath(),
	})
	if err != nil {
		return nil, err
	}

	return &pb.ContextFileReadResponse{
		Path:    resp.Path,
		Content: resp.Content,
		Version: resp.Version,
	}, nil
}

func (s *Server) WriteFile(ctx context.Context, req *pb.ContextFileWriteRequest) (*pb.ContextFileWriteResponse, error) {
	var version *int64
	if req.Version != nil {
		version = req.Version
	}

	resp, err := s.supervisor.Context.WriteFile(ctx, contextsvc.FileWriteRequest{
		Root:    req.GetRoot(),
		Path:    req.GetPath(),
		Content: req.GetContent(),
		Version: version,
	})
	if err != nil {
		return nil, err
	}

	return &pb.ContextFileWriteResponse{
		Path:    resp.Path,
		Version: resp.Version,
	}, nil
}

func (s *Server) DeleteFile(ctx context.Context, req *pb.ContextFileDeleteRequest) (*pb.ContextFileDeleteResponse, error) {
	var version *int64
	if req.Version != nil {
		version = req.Version
	}

	resp, err := s.supervisor.Context.DeleteFile(ctx, contextsvc.FileDeleteRequest{
		Root:    req.GetRoot(),
		Path:    req.GetPath(),
		Version: version,
	})
	if err != nil {
		return nil, err
	}

	return &pb.ContextFileDeleteResponse{
		Path:    resp.Path,
		Deleted: resp.Deleted,
	}, nil
}

func (s *Server) MoveFile(ctx context.Context, req *pb.ContextFileMoveRequest) (*pb.ContextFileMoveResponse, error) {
	var version *int64
	if req.Version != nil {
		version = req.Version
	}

	resp, err := s.supervisor.Context.MoveFile(ctx, contextsvc.FileMoveRequest{
		Root:     req.GetRoot(),
		FromPath: req.GetFromPath(),
		ToPath:   req.GetToPath(),
		Version:  version,
	})
	if err != nil {
		return nil, err
	}

	return &pb.ContextFileMoveResponse{
		FromPath: resp.FromPath,
		ToPath:   resp.ToPath,
		Version:  resp.Version,
	}, nil
}

func (s *Server) AppendSession(ctx context.Context, req *pb.ContextSessionAppendRequest) (*pb.ContextSessionHistory, error) {
	resp, err := s.supervisor.Context.AppendSession(ctx, contextsvc.SessionAppendRequest{
		SessionID: req.GetSessionId(),
		Role:      req.GetRole(),
		Content:   req.GetContent(),
		Metadata:  req.GetMetadata(),
	})
	if err != nil {
		return nil, err
	}
	return toProtoContextSessionHistory(resp), nil
}

func (s *Server) GetSession(ctx context.Context, req *pb.ContextSessionGetRequest) (*pb.ContextSessionHistory, error) {
	resp, err := s.supervisor.Context.GetSession(ctx, req.GetSessionId())
	if err != nil {
		return nil, err
	}
	return toProtoContextSessionHistory(resp), nil
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
	router.GET("/api/v1/context/status", s.handleContextStatus)
	router.GET("/api/v1/context/resources", s.handleListContextResources)
	router.POST("/api/v1/context/resources", s.handleUpsertContextResource)
	router.DELETE("/api/v1/context/resources", s.handleDeleteContextResource)
	router.POST("/api/v1/context/search", s.handleContextSearch)
	router.POST("/api/v1/context/workspaces/sync", s.handleSyncContextWorkspace)
	router.POST("/api/v1/context/files/list", s.handleListContextFiles)
	router.POST("/api/v1/context/files/read", s.handleReadContextFile)
	router.POST("/api/v1/context/files/write", s.handleWriteContextFile)
	router.POST("/api/v1/context/files/delete", s.handleDeleteContextFile)
	router.POST("/api/v1/context/files/move", s.handleMoveContextFile)
	router.POST("/api/v1/context/sessions/append", s.handleAppendContextSession)
	router.GET("/api/v1/context/sessions/:id", s.handleGetContextSession)
}

func (s *Server) handleHealth(c *gin.Context) {
	ctx, cancel := context.WithTimeout(c.Request.Context(), 3*time.Second)
	defer cancel()

	ready := true
	status := "ok"
	if s.supervisor.Context.Enabled() {
		health, err := s.supervisor.Context.Readiness(ctx)
		if err != nil {
			ready = false
			status = "degraded"
		} else {
			ready = health.Ready
			status = health.Status
		}
	}

	c.JSON(200, gin.H{"status": status, "ready": ready})
}

func (s *Server) handleStatus(c *gin.Context) {
	health := s.supervisor.Health()
	c.JSON(200, health)
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

func (s *Server) handleContextStatus(c *gin.Context) {
	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()

	status, err := s.supervisor.Context.Status(ctx)
	if err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, status)
}

func (s *Server) handleListContextResources(c *gin.Context) {
	ctx, cancel := context.WithTimeout(c.Request.Context(), 10*time.Second)
	defer cancel()

	resources, err := s.supervisor.Context.ListResources(ctx)
	if err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, resources)
}

func (s *Server) handleUpsertContextResource(c *gin.Context) {
	var req contextsvc.UpsertResourceRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 30*time.Second)
	defer cancel()

	resp, err := s.supervisor.Context.UpsertResource(ctx, req)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, resp)
}

func (s *Server) handleDeleteContextResource(c *gin.Context) {
	uri := c.Query("uri")
	if uri == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "missing uri"})
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 10*time.Second)
	defer cancel()

	if err := s.supervisor.Context.DeleteResource(ctx, uri); err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}
	c.Status(http.StatusNoContent)
}

func (s *Server) handleContextSearch(c *gin.Context) {
	var req contextsvc.SearchRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 20*time.Second)
	defer cancel()

	resp, err := s.supervisor.Context.Search(ctx, req)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, resp)
}

func (s *Server) handleSyncContextWorkspace(c *gin.Context) {
	var req contextsvc.WorkspaceSyncRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 60*time.Second)
	defer cancel()

	resp, err := s.supervisor.Context.SyncWorkspace(ctx, req)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, resp)
}

func (s *Server) handleListContextFiles(c *gin.Context) {
	var req contextsvc.FileListRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 10*time.Second)
	defer cancel()

	resp, err := s.supervisor.Context.ListFiles(ctx, req)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, resp)
}

func (s *Server) handleDeleteContextFile(c *gin.Context) {
	var req contextsvc.FileDeleteRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 10*time.Second)
	defer cancel()

	resp, err := s.supervisor.Context.DeleteFile(ctx, req)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, resp)
}

func (s *Server) handleMoveContextFile(c *gin.Context) {
	var req contextsvc.FileMoveRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 10*time.Second)
	defer cancel()

	resp, err := s.supervisor.Context.MoveFile(ctx, req)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, resp)
}

func (s *Server) handleReadContextFile(c *gin.Context) {
	var req contextsvc.FileReadRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 10*time.Second)
	defer cancel()

	resp, err := s.supervisor.Context.ReadFile(ctx, req)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, resp)
}

func (s *Server) handleWriteContextFile(c *gin.Context) {
	var req contextsvc.FileWriteRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 10*time.Second)
	defer cancel()

	resp, err := s.supervisor.Context.WriteFile(ctx, req)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, resp)
}

func (s *Server) handleAppendContextSession(c *gin.Context) {
	var req contextsvc.SessionAppendRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 10*time.Second)
	defer cancel()

	resp, err := s.supervisor.Context.AppendSession(ctx, req)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, resp)
}

func (s *Server) handleGetContextSession(c *gin.Context) {
	ctx, cancel := context.WithTimeout(c.Request.Context(), 10*time.Second)
	defer cancel()

	resp, err := s.supervisor.Context.GetSession(ctx, c.Param("id"))
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, resp)
}

func toProtoContextResource(resource contextsvc.Resource) *pb.ContextResource {
	return &pb.ContextResource{
		Uri:      resource.URI,
		Title:    resource.Title,
		Layer:    string(resource.Layer),
		Metadata: resource.Metadata,
	}
}

func toProtoContextSessionHistory(resp *contextsvc.SessionResponse) *pb.ContextSessionHistory {
	entries := make([]*pb.ContextSessionEntry, 0, len(resp.Entries))
	for _, entry := range resp.Entries {
		entries = append(entries, &pb.ContextSessionEntry{
			SessionId: entry.SessionID,
			Role:      entry.Role,
			Content:   entry.Content,
			Metadata:  entry.Metadata,
			CreatedAt: timestamppb.New(time.UnixMilli(entry.CreatedAt)),
		})
	}

	return &pb.ContextSessionHistory{
		SessionId: resp.SessionID,
		Entries:   entries,
	}
}

func layerFromProto(layer pb.ContextLayer) contextsvc.Layer {
	switch layer {
	case pb.ContextLayer_CONTEXT_LAYER_L0:
		return contextsvc.LayerL0
	case pb.ContextLayer_CONTEXT_LAYER_L1:
		return contextsvc.LayerL1
	case pb.ContextLayer_CONTEXT_LAYER_L2:
		return contextsvc.LayerL2
	default:
		return ""
	}
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
