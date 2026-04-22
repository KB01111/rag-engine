package rag

import (
	"context"
	"testing"

	"github.com/ai-engine/go/internal/config"
	"github.com/ai-engine/go/internal/contextsvc"
	pb "github.com/ai-engine/proto/go"
	"github.com/stretchr/testify/suite"
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

type ManagerTestSuite struct {
	suite.Suite
	cfg     *config.Config
	backend *stubBackend
	manager *Manager
}

func (s *ManagerTestSuite) SetupTest() {
	s.cfg = config.DefaultConfig()
	s.backend = &stubBackend{}
	s.manager = NewManager(s.cfg, s.backend)
}

func (s *ManagerTestSuite) TestManagerUpsertDocumentDelegatesToContextBackend() {
	resp, err := s.manager.UpsertDocument(context.Background(), &pb.UpsertRequest{
		DocumentId: "doc-1",
		Content:    "hello world",
		Metadata: map[string]string{
			"title": "Doc 1",
		},
	})

	s.Require().NoError(err)
	s.Equal(int32(4), resp.ChunksIndexed)
	s.Equal("viking://resources/doc-1", s.backend.upsertReq.URI)
	s.Equal("hello world", s.backend.upsertReq.Content)
	s.Equal("Doc 1", s.backend.upsertReq.Title)
}

func (s *ManagerTestSuite) TestManagerSearchDelegatesToContextBackend() {
	resp, err := s.manager.Search(context.Background(), &pb.SearchRequest{
		Query: "hello world",
		TopK:  3,
		Filters: map[string]string{
			"title": "Doc 1",
		},
	})

	s.Require().NoError(err)
	s.Require().Len(resp.Results, 1)
	s.Equal("doc-1", resp.Results[0].DocumentId)
	s.Equal("hello world", s.backend.searchReq.Query)
	s.Equal(3, s.backend.searchReq.TopK)
	s.Equal("viking://resources/", s.backend.searchReq.ScopeURI)
	s.InDelta(1.25, resp.QueryTimeMs, 0.001)
}

func (s *ManagerTestSuite) TestManagerDeleteAndStatusUseContextBackend() {
	_, err := s.manager.DeleteDocument(context.Background(), &pb.DeleteRequest{DocumentId: "doc-1"})
	s.Require().NoError(err)
	s.Equal("viking://resources/doc-1", s.backend.deleteURI)

	status, err := s.manager.GetRagStatus(context.Background(), &emptypb.Empty{})
	s.Require().NoError(err)
	s.Equal(int64(1), status.DocumentCount)
	s.Equal(int64(2), status.ChunkCount)
}

func TestManagerTestSuite(t *testing.T) {
	suite.Run(t, &ManagerTestSuite{})
}