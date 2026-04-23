package daemon

import (
	"context"
	"net"
	"testing"

	pb "github.com/ai-engine/proto/go"
	"github.com/stretchr/testify/require"
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

func TestClientContextRPCs(t *testing.T) {
	server := grpc.NewServer()
	pb.RegisterContextServer(server, &contextServerStub{})

	conn := dialBufConn(t, server)
	client := &Client{
		conn:           conn,
		context:        pb.NewContextClient(conn),
		mcpConnections: make(map[string]struct{}),
	}

	ctx := context.Background()

	status, err := client.GetContextStatus(ctx, &emptypb.Empty{})
	require.NoError(t, err)
	require.True(t, status.Ready)
	require.EqualValues(t, 3, status.DocumentCount)

	search, err := client.SearchContext(ctx, &pb.ContextSearchRequest{
		Query: "Dragonfly",
		TopK:  3,
	})
	require.NoError(t, err)
	require.Len(t, search.Results, 1)
	require.Contains(t, search.Results[0].ChunkText, "Dragonfly")

	history, err := client.AppendSession(ctx, &pb.ContextSessionAppendRequest{
		SessionId: "sess-42",
		Role:      "user",
		Content:   "I prefer Dragonfly over Redis.",
	})
	require.NoError(t, err)
	require.Len(t, history.Entries, 1)
	require.Equal(t, "sess-42", history.SessionId)

	session, err := client.GetSession(ctx, &pb.ContextSessionGetRequest{SessionId: "sess-42"})
	require.NoError(t, err)
	require.Len(t, session.Entries, 1)
	require.Contains(t, session.Entries[0].Content, "Dragonfly")
}
