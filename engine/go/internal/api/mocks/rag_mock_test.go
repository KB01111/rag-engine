package mocks

import (
	"context"
	"errors"
	"testing"

	pb "github.com/ai-engine/proto/go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"google.golang.org/protobuf/types/known/emptypb"
)

func TestMockRagServer_UpsertDocument(t *testing.T) {
	m := &MockRagServer{}
	ctx := context.Background()
	req := &pb.UpsertRequest{
		DocumentId: "doc-1",
		Content:    "hello world",
		Metadata:   map[string]string{"author": "test"},
	}
	expected := &pb.UpsertResponse{DocumentId: "doc-1", ChunksIndexed: 2}

	m.On("UpsertDocument", ctx, req).Return(expected, nil)

	resp, err := m.UpsertDocument(ctx, req)

	assert.NoError(t, err)
	assert.Equal(t, expected.DocumentId, resp.DocumentId)
	assert.Equal(t, expected.ChunksIndexed, resp.ChunksIndexed)
	m.AssertExpectations(t)
}

func TestMockRagServer_UpsertDocument_Error(t *testing.T) {
	m := &MockRagServer{}
	ctx := context.Background()
	req := &pb.UpsertRequest{DocumentId: "bad-doc"}
	wantErr := errors.New("embedding failed")

	m.On("UpsertDocument", ctx, req).Return((*pb.UpsertResponse)(nil), wantErr)

	resp, err := m.UpsertDocument(ctx, req)

	assert.Nil(t, resp)
	assert.ErrorIs(t, err, wantErr)
	m.AssertExpectations(t)
}

func TestMockRagServer_DeleteDocument(t *testing.T) {
	m := &MockRagServer{}
	ctx := context.Background()
	req := &pb.DeleteRequest{DocumentId: "doc-1"}

	m.On("DeleteDocument", ctx, req).Return(&emptypb.Empty{}, nil)

	resp, err := m.DeleteDocument(ctx, req)

	assert.NoError(t, err)
	assert.NotNil(t, resp)
	m.AssertExpectations(t)
}

func TestMockRagServer_DeleteDocument_Error(t *testing.T) {
	m := &MockRagServer{}
	ctx := context.Background()
	req := &pb.DeleteRequest{DocumentId: "missing"}
	wantErr := errors.New("document not found")

	m.On("DeleteDocument", ctx, req).Return(&emptypb.Empty{}, wantErr)

	_, err := m.DeleteDocument(ctx, req)

	assert.ErrorIs(t, err, wantErr)
	m.AssertExpectations(t)
}

func TestMockRagServer_Search(t *testing.T) {
	m := &MockRagServer{}
	ctx := context.Background()
	req := &pb.SearchRequest{
		Query: "machine learning",
		TopK:  5,
	}
	expected := &pb.SearchResponse{
		Results: []*pb.SearchResult{
			{DocumentId: "doc-1", ChunkText: "ML overview", Score: 0.95},
		},
		QueryTimeMs: 3.14,
	}

	m.On("Search", ctx, req).Return(expected, nil)

	resp, err := m.Search(ctx, req)

	assert.NoError(t, err)
	assert.Len(t, resp.Results, 1)
	assert.Equal(t, "doc-1", resp.Results[0].DocumentId)
	assert.InDelta(t, 0.95, float64(resp.Results[0].Score), 0.001)
	assert.InDelta(t, 3.14, resp.QueryTimeMs, 0.001)
	m.AssertExpectations(t)
}

func TestMockRagServer_Search_Empty(t *testing.T) {
	m := &MockRagServer{}
	ctx := context.Background()
	req := &pb.SearchRequest{Query: "no results here"}
	expected := &pb.SearchResponse{Results: []*pb.SearchResult{}, QueryTimeMs: 0.1}

	m.On("Search", ctx, req).Return(expected, nil)

	resp, err := m.Search(ctx, req)

	assert.NoError(t, err)
	assert.Empty(t, resp.Results)
	m.AssertExpectations(t)
}

func TestMockRagServer_Search_Error(t *testing.T) {
	m := &MockRagServer{}
	ctx := context.Background()
	req := &pb.SearchRequest{Query: "error case"}
	wantErr := errors.New("index unavailable")

	m.On("Search", ctx, req).Return(&pb.SearchResponse{}, wantErr)

	_, err := m.Search(ctx, req)

	assert.ErrorIs(t, err, wantErr)
	m.AssertExpectations(t)
}

func TestMockRagServer_GetRagStatus(t *testing.T) {
	m := &MockRagServer{}
	ctx := context.Background()
	req := &emptypb.Empty{}
	expected := &pb.RagStatus{
		DocumentCount: 42,
		ChunkCount:    200,
	}

	m.On("GetRagStatus", ctx, req).Return(expected, nil)

	resp, err := m.GetRagStatus(ctx, req)

	assert.NoError(t, err)
	assert.Equal(t, int64(42), resp.DocumentCount)
	assert.Equal(t, int64(200), resp.ChunkCount)
	m.AssertExpectations(t)
}

func TestMockRagServer_GetRagStatus_Error(t *testing.T) {
	m := &MockRagServer{}
	ctx := context.Background()
	req := &emptypb.Empty{}
	wantErr := errors.New("service unavailable")

	m.On("GetRagStatus", ctx, req).Return(&pb.RagStatus{}, wantErr)

	_, err := m.GetRagStatus(ctx, req)

	assert.ErrorIs(t, err, wantErr)
	m.AssertExpectations(t)
}

func TestMockRagServer_ListDocuments(t *testing.T) {
	m := &MockRagServer{}
	ctx := context.Background()
	req := &emptypb.Empty{}
	expected := &pb.DocumentList{
		Documents: []*pb.DocumentInfo{
			{Id: "doc-1", ChunkCount: 3},
			{Id: "doc-2", ChunkCount: 7},
		},
	}

	m.On("ListDocuments", ctx, req).Return(expected, nil)

	resp, err := m.ListDocuments(ctx, req)

	assert.NoError(t, err)
	assert.Len(t, resp.Documents, 2)
	assert.Equal(t, "doc-1", resp.Documents[0].Id)
	assert.Equal(t, "doc-2", resp.Documents[1].Id)
	m.AssertExpectations(t)
}

func TestMockRagServer_ListDocuments_Empty(t *testing.T) {
	m := &MockRagServer{}
	ctx := context.Background()
	req := &emptypb.Empty{}
	expected := &pb.DocumentList{Documents: []*pb.DocumentInfo{}}

	m.On("ListDocuments", ctx, req).Return(expected, nil)

	resp, err := m.ListDocuments(ctx, req)

	assert.NoError(t, err)
	assert.Empty(t, resp.Documents)
	m.AssertExpectations(t)
}

func TestMockRagServer_SatisfiesMockInterface(t *testing.T) {
	// Verify that the mock can be configured with Anything matchers and called multiple times.
	m := &MockRagServer{}
	ctx := context.Background()

	m.On("GetRagStatus", mock.Anything, mock.Anything).Return(&pb.RagStatus{DocumentCount: 1}, nil)

	for i := 0; i < 3; i++ {
		resp, err := m.GetRagStatus(ctx, &emptypb.Empty{})
		assert.NoError(t, err)
		assert.Equal(t, int64(1), resp.DocumentCount)
	}
}
