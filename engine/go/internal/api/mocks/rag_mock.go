package api_test

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
	return args.Get(0).(*emptypb.Empty), args.Error(1)
}

func (m *MockRagServer) Search(ctx context.Context, req *pb.SearchRequest) (*pb.SearchResponse, error) {
	args := m.Called(ctx, req)
	return args.Get(0).(*pb.SearchResponse), args.Error(1)
}

func (m *MockRagServer) GetRagStatus(ctx context.Context, req *emptypb.Empty) (*pb.RagStatus, error) {
	args := m.Called(ctx, req)
	return args.Get(0).(*pb.RagStatus), args.Error(1)
}

func (m *MockRagServer) ListDocuments(ctx context.Context, _ *emptypb.Empty) (*pb.DocumentList, error) {
	args := m.Called(ctx)
	return args.Get(0).(*pb.DocumentList), args.Error(1)
}
