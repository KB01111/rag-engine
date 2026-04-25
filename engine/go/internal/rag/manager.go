package rag

import (
	"context"
	"fmt"
	"net/url"
	"strings"

	"github.com/ai-engine/go/internal/config"
	contextsvc "github.com/ai-engine/go/internal/contextsvc"
	pb "github.com/ai-engine/proto/go"
	"google.golang.org/protobuf/types/known/emptypb"
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

const (
	ragDocumentRoot      = "workspace"
	ragDocumentPathBase  = "rag"
	ragDocumentKind      = "rag-document"
	ragDocumentIDKey     = "rag_document_id"
	ragDocumentURIPrefix = "viking://resources/" + ragDocumentRoot + "/" + ragDocumentPathBase + "/"
)

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

	metadata := ragMetadata(req.Metadata, req.DocumentId)
	title := req.DocumentId
	if maybeTitle, ok := metadata["title"]; ok && strings.TrimSpace(maybeTitle) != "" {
		title = maybeTitle
	}

	resp, err := m.backend.UpsertResource(ctx, contextsvc.UpsertResourceRequest{
		URI:      documentURI(req.DocumentId),
		Title:    title,
		Content:  req.Content,
		Layer:    contextsvc.LayerL2,
		Metadata: metadata,
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
		ScopeURI: ragDocumentScopeURI(),
		TopK:     topK,
		Filters:  searchFilters(req.Filters),
		Layer:    contextsvc.LayerL2,
	})
	if err != nil {
		return nil, err
	}

	results := make([]*pb.SearchResult, 0, len(resp.Results))
	for _, hit := range resp.Results {
		results = append(results, &pb.SearchResult{
			DocumentId: documentIDFromResource(hit.URI, hit.Metadata, hit.DocumentID),
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

	documentCount := status.DocumentCount
	if resources, err := m.backend.ListResources(ctx); err == nil {
		documentCount = int64(len(filterRagResources(resources.Resources)))
	}

	return &pb.RagStatus{
		DocumentCount:  documentCount,
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
	for _, resource := range filterRagResources(resp.Resources) {
		documents = append(documents, &pb.DocumentInfo{
			Id:         documentIDFromResource(resource.URI, resource.Metadata, ""),
			Title:      resource.Title,
			ChunkCount: 0,
		})
	}

	return &pb.DocumentList{Documents: documents}, nil
}

func (m *Manager) DocumentCount() int64 {
	if m.backend == nil || !m.backend.Enabled() {
		return 0
	}
	if resources, err := m.backend.ListResources(context.Background()); err == nil {
		return int64(len(filterRagResources(resources.Resources)))
	}
	status, err := m.backend.Status(context.Background())
	if err != nil {
		return 0
	}
	return status.DocumentCount
}

func documentURI(documentID string) string {
	escapedID := url.PathEscape(documentID)
	return ragDocumentURIPrefix + escapedID + ".md"
}

func ragDocumentScopeURI() string {
	return ragDocumentURIPrefix
}

func documentIDFromResource(uri string, metadata map[string]string, fallback string) string {
	if documentID := strings.TrimSpace(metadata[ragDocumentIDKey]); documentID != "" {
		return documentID
	}
	if strings.HasPrefix(uri, ragDocumentURIPrefix) {
		trimmed := strings.TrimPrefix(uri, ragDocumentURIPrefix)
		trimmed = strings.TrimSuffix(trimmed, ".md")
		if decoded, err := url.PathUnescape(trimmed); err == nil && decoded != "" {
			return decoded
		}
	}
	if fallback != "" {
		return fallback
	}
	return strings.TrimPrefix(uri, "viking://resources/")
}

func ragMetadata(metadata map[string]string, documentID string) map[string]string {
	copied := make(map[string]string, len(metadata)+2)
	for key, value := range metadata {
		copied[key] = value
	}
	copied["kind"] = ragDocumentKind
	copied[ragDocumentIDKey] = documentID
	return copied
}

func searchFilters(filters map[string]string) map[string]string {
	merged := make(map[string]string, len(filters)+1)
	for key, value := range filters {
		merged[key] = value
	}
	merged["kind"] = ragDocumentKind
	return merged
}

func filterRagResources(resources []contextsvc.Resource) []contextsvc.Resource {
	filtered := make([]contextsvc.Resource, 0, len(resources))
	for _, resource := range resources {
		if strings.TrimSpace(resource.Metadata["kind"]) == ragDocumentKind || strings.HasPrefix(resource.URI, ragDocumentURIPrefix) {
			filtered = append(filtered, resource)
		}
	}
	return filtered
}
