package rag

import (
	"context"
	"fmt"
	"strings"

	"github.com/ai-engine/go/internal/config"
	contextsvc "github.com/ai-engine/go/internal/contextsvc"
	pb "github.com/ai-engine/proto/go"
	"google.golang.org/protobuf/types/known/emptypb"
	"google.golang.org/protobuf/types/known/timestamppb"
)

type Service interface {
	UpsertDocument(context.Context, *pb.UpsertRequest) (*pb.UpsertResponse, error)
	DeleteDocument(context.Context, *pb.DeleteRequest) (*emptypb.Empty, error)
	Search(context.Context, *pb.SearchRequest) (*pb.SearchResponse, error)
	GetRagStatus(context.Context, *emptypb.Empty) (*pb.RagStatus, error)
	ListDocuments(context.Context, *emptypb.Empty) (*pb.DocumentList, error)
	DocumentCount() int64
}

type Manager struct {
	backend contextsvc.Backend
	topK    int
}

func NewManager(cfg *config.Config, backend contextsvc.Backend) *Manager {
	return &Manager{
		backend: backend,
		topK:    cfg.RAG.TopK,
	}
}

func (m *Manager) UpsertDocument(ctx context.Context, req *pb.UpsertRequest) (*pb.UpsertResponse, error) {
	if m.backend == nil || !m.backend.Enabled() {
		return nil, fmt.Errorf("context backend is not enabled")
	}

	title := req.DocumentId
	if req.Metadata != nil {
		if maybeTitle, ok := req.Metadata["title"]; ok && strings.TrimSpace(maybeTitle) != "" {
			title = maybeTitle
		}
	}

	resp, err := m.backend.UpsertResource(ctx, contextsvc.UpsertResourceRequest{
		URI:      documentURI(req.DocumentId),
		Title:    title,
		Content:  req.Content,
		Layer:    contextsvc.LayerL2,
		Metadata: req.Metadata,
	})
	if err != nil {
		return nil, err
	}

	return &pb.UpsertResponse{
		DocumentId:    req.DocumentId,
		ChunksIndexed: resp.ChunksIndexed,
	}, nil
}

func (m *Manager) DeleteDocument(ctx context.Context, req *pb.DeleteRequest) (*emptypb.Empty, error) {
	if m.backend == nil || !m.backend.Enabled() {
		return nil, fmt.Errorf("context backend is not enabled")
	}
	if err := m.backend.DeleteResource(ctx, documentURI(req.DocumentId)); err != nil {
		return nil, err
	}
	return &emptypb.Empty{}, nil
}

func (m *Manager) Search(ctx context.Context, req *pb.SearchRequest) (*pb.SearchResponse, error) {
	if m.backend == nil || !m.backend.Enabled() {
		return nil, fmt.Errorf("context backend is not enabled")
	}

	topK := int(req.TopK)
	if topK == 0 {
		topK = m.topK
	}

	resp, err := m.backend.Search(ctx, contextsvc.SearchRequest{
		Query:    req.Query,
		ScopeURI: "viking://resources/",
		TopK:     topK,
		Filters:  req.Filters,
		Layer:    contextsvc.LayerL2,
	})
	if err != nil {
		return nil, err
	}

	results := make([]*pb.SearchResult, 0, len(resp.Results))
	for _, hit := range resp.Results {
		results = append(results, &pb.SearchResult{
			DocumentId: documentIDFromURI(hit.URI, hit.DocumentID),
			ChunkText:  hit.ChunkText,
			Score:      hit.Score,
			Metadata:   hit.Metadata,
		})
	}

	return &pb.SearchResponse{
		Results:     results,
		QueryTimeMs: resp.QueryTimeMs,
	}, nil
}

func (m *Manager) GetRagStatus(ctx context.Context, _ *emptypb.Empty) (*pb.RagStatus, error) {
	if m.backend == nil || !m.backend.Enabled() {
		return &pb.RagStatus{}, nil
	}

	status, err := m.backend.Status(ctx)
	if err != nil {
		return nil, err
	}

	return &pb.RagStatus{
		DocumentCount:  status.DocumentCount,
		ChunkCount:     status.ChunkCount,
		IndexSizeBytes: status.IndexSizeBytes,
		EmbeddingModel: status.EmbeddingModel,
	}, nil
}

func (m *Manager) ListDocuments(ctx context.Context, _ *emptypb.Empty) (*pb.DocumentList, error) {
	if m.backend == nil || !m.backend.Enabled() {
		return &pb.DocumentList{}, nil
	}

	resp, err := m.backend.ListResources(ctx)
	if err != nil {
		return nil, err
	}

	documents := make([]*pb.DocumentInfo, 0, len(resp.Resources))
	for _, resource := range resp.Resources {
		documents = append(documents, &pb.DocumentInfo{
			Id:         documentIDFromURI(resource.URI, ""),
			Title:      resource.Title,
			ChunkCount: 0,
			CreatedAt:  timestamppb.Now(),
			UpdatedAt:  timestamppb.Now(),
		})
	}

	return &pb.DocumentList{Documents: documents}, nil
}

func (m *Manager) DocumentCount() int64 {
	if m.backend == nil || !m.backend.Enabled() {
		return 0
	}
	status, err := m.backend.Status(context.Background())
	if err != nil {
		return 0
	}
	return status.DocumentCount
}

func documentURI(documentID string) string {
	return "viking://resources/" + documentID
}

func documentIDFromURI(uri, fallback string) string {
	if fallback != "" {
		return fallback
	}
	return strings.TrimPrefix(uri, "viking://resources/")
}
