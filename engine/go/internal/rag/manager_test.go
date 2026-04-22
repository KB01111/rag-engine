package rag

import (
	"context"
	"testing"

	"github.com/ai-engine/go/internal/config"
	"github.com/ai-engine/go/internal/contextsvc"
	pb "github.com/ai-engine/proto/go"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/types/known/emptypb"
)

type stubBackend struct {
	upsertReq contextsvc.UpsertResourceRequest
	searchReq contextsvc.SearchRequest
	deleteURI string
}

func (s *stubBackend) Start(context.Context) error { return nil }

func (s *stubBackend) Stop(context.Context) error { return nil }

func (s *stubBackend) Enabled() bool { return true }

func (s *stubBackend) Readiness(context.Context) (*contextsvc.HealthStatus, error) {
	return &contextsvc.HealthStatus{Status: "ok", Ready: true}, nil
}

func (s *stubBackend) Status(context.Context) (*contextsvc.StatusResponse, error) {
	return &contextsvc.StatusResponse{
		DocumentCount: 1,
		ChunkCount:    2,
	}, nil
}

func (s *stubBackend) UpsertResource(_ context.Context, req contextsvc.UpsertResourceRequest) (*contextsvc.UpsertResourceResponse, error) {
	s.upsertReq = req
	return &contextsvc.UpsertResourceResponse{
		Resource: contextsvc.Resource{
			URI: req.URI,
		},
		ChunksIndexed: 4,
	}, nil
}

func (s *stubBackend) DeleteResource(_ context.Context, uri string) error {
	s.deleteURI = uri
	return nil
}

func (s *stubBackend) Search(_ context.Context, req contextsvc.SearchRequest) (*contextsvc.SearchResponse, error) {
	s.searchReq = req
	return &contextsvc.SearchResponse{
		Results: []contextsvc.SearchHit{
			{
				URI:        "viking://resources/doc-1",
				DocumentID: "doc-1",
				ChunkText:  "hello world",
				Score:      0.88,
				Metadata: map[string]string{
					"title": "Doc 1",
				},
			},
		},
		QueryTimeMs: 1.25,
	}, nil
}

func (s *stubBackend) ListResources(context.Context) (*contextsvc.ListResourcesResponse, error) {
	return &contextsvc.ListResourcesResponse{
		Resources: []contextsvc.Resource{
			{
				URI:   "viking://resources/doc-1",
				Title: "Doc 1",
			},
		},
	}, nil
}

func TestManagerUpsertDocumentDelegatesToContextBackend(t *testing.T) {
	cfg := config.DefaultConfig()
	backend := &stubBackend{}
	manager := NewManager(cfg, backend)

	resp, err := manager.UpsertDocument(context.Background(), &pb.UpsertRequest{
		DocumentId: "doc-1",
		Content:    "hello world",
		Metadata: map[string]string{
			"title": "Doc 1",
		},
	})

	require.NoError(t, err)
	require.Equal(t, int32(4), resp.ChunksIndexed)
	require.Equal(t, "viking://resources/doc-1", backend.upsertReq.URI)
	require.Equal(t, "hello world", backend.upsertReq.Content)
	require.Equal(t, "Doc 1", backend.upsertReq.Title)
}

func TestManagerSearchDelegatesToContextBackend(t *testing.T) {
	cfg := config.DefaultConfig()
	backend := &stubBackend{}
	manager := NewManager(cfg, backend)

	resp, err := manager.Search(context.Background(), &pb.SearchRequest{
		Query: "hello world",
		TopK:  3,
		Filters: map[string]string{
			"title": "Doc 1",
		},
	})

	require.NoError(t, err)
	require.Len(t, resp.Results, 1)
	require.Equal(t, "doc-1", resp.Results[0].DocumentId)
	require.Equal(t, "hello world", backend.searchReq.Query)
	require.Equal(t, 3, backend.searchReq.TopK)
	require.Equal(t, "viking://resources/", backend.searchReq.ScopeURI)
	require.InDelta(t, 1.25, resp.QueryTimeMs, 0.001)
}

func TestManagerDeleteAndStatusUseContextBackend(t *testing.T) {
	cfg := config.DefaultConfig()
	backend := &stubBackend{}
	manager := NewManager(cfg, backend)

	_, err := manager.DeleteDocument(context.Background(), &pb.DeleteRequest{DocumentId: "doc-1"})
	require.NoError(t, err)
	require.Equal(t, "viking://resources/doc-1", backend.deleteURI)

	status, err := manager.GetRagStatus(context.Background(), &emptypb.Empty{})
	require.NoError(t, err)
	require.Equal(t, int64(1), status.DocumentCount)
	require.Equal(t, int64(2), status.ChunkCount)
}
