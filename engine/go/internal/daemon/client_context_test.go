package daemon

import (
	"context"
	"net"
	"testing"

	pb "github.com/ai-engine/proto/go"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"
	"google.golang.org/protobuf/types/known/emptypb"
	"google.golang.org/protobuf/types/known/timestamppb"
)

const bufConnSize = 1024 * 1024

type contextServerStub struct {
	pb.UnimplementedContextServer
}

func (s *contextServerStub) GetContextStatus(context.Context, *emptypb.Empty) (*pb.ContextStatus, error) {
	return &pb.ContextStatus{
		DocumentCount: 3,
		ChunkCount:    7,
		Ready:         true,
	}, nil
}

func (s *contextServerStub) SearchContext(_ context.Context, req *pb.ContextSearchRequest) (*pb.ContextSearchResponse, error) {
	return &pb.ContextSearchResponse{
		Results: []*pb.ContextSearchResult{
			{
				Uri:        "viking://resources/workspace/preferences.md",
				DocumentId: "preferences",
				ChunkText:  "User prefers Dragonfly over Redis.",
				Score:      0.94,
				Metadata: map[string]string{
					"topic": req.GetQuery(),
				},
				Layer: "l2",
			},
		},
		QueryTimeMs: 1.2,
	}, nil
}

func (s *contextServerStub) AppendSession(_ context.Context, req *pb.ContextSessionAppendRequest) (*pb.ContextSessionHistory, error) {
	return &pb.ContextSessionHistory{
		SessionId: req.GetSessionId(),
		Entries: []*pb.ContextSessionEntry{
			{
				SessionId: req.GetSessionId(),
				Role:      req.GetRole(),
				Content:   req.GetContent(),
				Metadata:  req.GetMetadata(),
				CreatedAt: timestamppb.Now(),
			},
		},
	}, nil
}

func (s *contextServerStub) GetSession(_ context.Context, req *pb.ContextSessionGetRequest) (*pb.ContextSessionHistory, error) {
	return &pb.ContextSessionHistory{
		SessionId: req.GetSessionId(),
		Entries: []*pb.ContextSessionEntry{
			{
				SessionId: req.GetSessionId(),
				Role:      "assistant",
				Content:   "Dragonfly has been preferred in this session.",
				CreatedAt: timestamppb.Now(),
			},
		},
	}, nil
}

func dialBufConn(t *testing.T, server *grpc.Server) *grpc.ClientConn {
	t.Helper()

	listener := bufconn.Listen(bufConnSize)
	go func() {
		_ = server.Serve(listener)
	}()

	conn, err := grpc.DialContext(
		context.Background(),
		"passthrough:///bufnet",
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) {
			return listener.Dial()
		}),
	)
	require.NoError(t, err)
	t.Cleanup(func() {
		_ = conn.Close()
		server.Stop()
	})
	return conn
}

type clientContextSuite struct {
	suite.Suite
	server *grpc.Server
	conn   *grpc.ClientConn
	client *Client
}

func (s *clientContextSuite) SetupTest() {
	s.server = grpc.NewServer()
	pb.RegisterContextServer(s.server, &contextServerStub{})

	listener := bufconn.Listen(bufConnSize)
	go func() {
		_ = s.server.Serve(listener)
	}()

	conn, err := grpc.DialContext(
		context.Background(),
		"passthrough:///bufnet",
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) {
			return listener.Dial()
		}),
	)
	s.Require().NoError(err)
	s.conn = conn
	s.client = &Client{
		conn:           conn,
		context:        pb.NewContextClient(conn),
		mcpConnections: make(map[string]struct{}),
	}

	s.T().Cleanup(func() {
		_ = s.conn.Close()
		s.server.Stop()
	})
}

func (s *clientContextSuite) TestClientContextRPCs() {
	ctx := context.Background()

	status, err := s.client.GetContextStatus(ctx, &emptypb.Empty{})
	s.Require().NoError(err)
	s.Require().True(status.Ready)
	s.Require().EqualValues(3, status.DocumentCount)

	search, err := s.client.SearchContext(ctx, &pb.ContextSearchRequest{
		Query: "Dragonfly",
		TopK:  3,
	})
	s.Require().NoError(err)
	s.Require().Len(search.Results, 1)
	s.Require().Contains(search.Results[0].ChunkText, "Dragonfly")

	history, err := s.client.AppendSession(ctx, &pb.ContextSessionAppendRequest{
		SessionId: "sess-42",
		Role:      "user",
		Content:   "I prefer Dragonfly over Redis.",
	})
	s.Require().NoError(err)
	s.Require().Len(history.Entries, 1)
	s.Require().Equal("sess-42", history.SessionId)

	session, err := s.client.GetSession(ctx, &pb.ContextSessionGetRequest{SessionId: "sess-42"})
	s.Require().NoError(err)
	s.Require().Len(session.Entries, 1)
	s.Require().Contains(session.Entries[0].Content, "Dragonfly")
}

func TestClientContextSuite(t *testing.T) {
	suite.Run(t, new(clientContextSuite))
}