package mocks

import (
	"context"

	pb "github.com/ai-engine/proto/go"
	"github.com/stretchr/testify/mock"
	"google.golang.org/protobuf/types/known/emptypb"
)

type MockRagServer struct {
	mock.Mock
}

func (m *MockRagServer) UpsertDocument(ctx context.Context, req *pb.UpsertRequest) (*pb.UpsertResponse, error) {
	args := m.Called(ctx, req)
	resp, _ := args.Get(0).(*pb.UpsertResponse)
	return resp, args.Error(1)
}

func (m *MockRagServer) DeleteDocument(ctx context.Context, req *pb.DeleteRequest) (*emptypb.Empty, error) {
	args := m.Called(ctx, req)
	v, ok := args.Get(0).(*emptypb.Empty)
	if !ok || v == nil {
		return nil, args.Error(1)
	}
	return v, args.Error(1)
}

func (m *MockRagServer) Search(ctx context.Context, req *pb.SearchRequest) (*pb.SearchResponse, error) {
	args := m.Called(ctx, req)
	v, ok := args.Get(0).(*pb.SearchResponse)
	if !ok || v == nil {
		return nil, args.Error(1)
	}
	return v, args.Error(1)
}

func (m *MockRagServer) GetRagStatus(ctx context.Context, req *emptypb.Empty) (*pb.RagStatus, error) {
	args := m.Called(ctx, req)
	v, ok := args.Get(0).(*pb.RagStatus)
	if !ok || v == nil {
		return nil, args.Error(1)
	}
	return v, args.Error(1)
}

func (m *MockRagServer) ListDocuments(ctx context.Context, req *emptypb.Empty) (*pb.DocumentList, error) {
	args := m.Called(ctx, req)
	v, ok := args.Get(0).(*pb.DocumentList)
	if !ok || v == nil {
		return nil, args.Error(1)
	}
	return v, args.Error(1)
}