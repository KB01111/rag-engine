package daemon

import (
	"context"
	"errors"
	"io"
	"net"
	"sync/atomic"
	"testing"
	"time"

	pb "github.com/ai-engine/proto/go"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
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

type runtimeCountStub struct {
	pb.UnimplementedRuntimeServer
	calls atomic.Int32
}

func (s *runtimeCountStub) GetStatus(context.Context, *emptypb.Empty) (*pb.RuntimeStatus, error) {
	s.calls.Add(1)
	return &pb.RuntimeStatus{
		LoadedModels: []*pb.ModelInfo{{Id: "loaded.gguf", Loaded: true}},
		Healthy:      true,
	}, nil
}

type ragCountStub struct {
	pb.UnimplementedRagServer
	calls atomic.Int32
}

func (s *ragCountStub) GetRagStatus(context.Context, *emptypb.Empty) (*pb.RagStatus, error) {
	s.calls.Add(1)
	return &pb.RagStatus{DocumentCount: 7}, nil
}

type trainingCountStub struct {
	pb.UnimplementedTrainingServer
	calls atomic.Int32
}

func (s *trainingCountStub) ListRuns(context.Context, *emptypb.Empty) (*pb.TrainingRunList, error) {
	s.calls.Add(1)
	return &pb.TrainingRunList{
		Runs: []*pb.TrainingRun{
			{Id: "1", Status: "queued"},
			{Id: "2", Status: "running"},
			{Id: "3", Status: "completed"},
		},
	}, nil
}

func TestCountHelpersUseTTLCache(t *testing.T) {
	server := grpc.NewServer()
	runtimeStub := &runtimeCountStub{}
	ragStub := &ragCountStub{}
	trainingStub := &trainingCountStub{}
	pb.RegisterRuntimeServer(server, runtimeStub)
	pb.RegisterRagServer(server, ragStub)
	pb.RegisterTrainingServer(server, trainingStub)
	conn := dialBufConn(t, server)

	client := &Client{
		conn:           conn,
		runtime:        pb.NewRuntimeClient(conn),
		rag:            pb.NewRagClient(conn),
		training:       pb.NewTrainingClient(conn),
		mcpConnections: make(map[string]struct{}),
		countCacheTTL:  2 * time.Second,
	}

	require.Equal(t, 1, client.LoadedModelCount())
	require.Equal(t, 1, client.LoadedModelCount())
	require.Equal(t, int32(1), runtimeStub.calls.Load())

	require.EqualValues(t, 7, client.DocumentCount())
	require.EqualValues(t, 7, client.DocumentCount())
	require.Equal(t, int32(1), ragStub.calls.Load())

	require.Equal(t, 2, client.ActiveRunCount())
	require.Equal(t, 2, client.ActiveRunCount())
	require.Equal(t, int32(1), trainingStub.calls.Load())
}

func TestWithOutgoingRequestIDCopiesIncomingMetadata(t *testing.T) {
	ctx := metadata.NewIncomingContext(context.Background(), metadata.Pairs("x-request-id", "rid-123"))

	got := withOutgoingRequestID(ctx)

	md, ok := metadata.FromOutgoingContext(got)
	require.True(t, ok)
	require.Equal(t, []string{"rid-123"}, md.Get("x-request-id"))
}

type runtimeStreamStub struct {
	pb.UnimplementedRuntimeServer
	canceledBeforeSend atomic.Bool
}

func (s *runtimeStreamStub) StreamInference(stream pb.Runtime_StreamInferenceServer) error {
	if _, err := stream.Recv(); err != nil {
		return err
	}
	if _, err := stream.Recv(); !errors.Is(err, io.EOF) {
		return err
	}

	select {
	case <-stream.Context().Done():
		s.canceledBeforeSend.Store(true)
		return stream.Context().Err()
	case <-time.After(10 * time.Millisecond):
	}

	return stream.Send(&pb.InferenceResponse{Token: "ok", Complete: true})
}

type serverInferenceStreamStub struct {
	pb.Runtime_StreamInferenceServer
	ctx      context.Context
	requests []*pb.InferenceRequest
	sent     []*pb.InferenceResponse
}

func (s *serverInferenceStreamStub) Context() context.Context {
	if s.ctx != nil {
		return s.ctx
	}
	return context.Background()
}

func (s *serverInferenceStreamStub) Recv() (*pb.InferenceRequest, error) {
	if len(s.requests) == 0 {
		return nil, io.EOF
	}
	req := s.requests[0]
	s.requests = s.requests[1:]
	return req, nil
}

func (s *serverInferenceStreamStub) Send(resp *pb.InferenceResponse) error {
	s.sent = append(s.sent, resp)
	return nil
}

func TestStreamInferenceHalfCloseDoesNotCancelReceiveSide(t *testing.T) {
	server := grpc.NewServer()
	runtimeStub := &runtimeStreamStub{}
	pb.RegisterRuntimeServer(server, runtimeStub)
	conn := dialBufConn(t, server)

	client := &Client{
		conn:           conn,
		runtime:        pb.NewRuntimeClient(conn),
		mcpConnections: make(map[string]struct{}),
	}
	stream := &serverInferenceStreamStub{
		ctx: context.Background(),
		requests: []*pb.InferenceRequest{
			{ModelId: "local.gguf", Prompt: "hi"},
		},
	}

	require.NoError(t, client.StreamInference(context.Background(), stream))
	require.False(t, runtimeStub.canceledBeforeSend.Load())
	require.Len(t, stream.sent, 1)
	require.Equal(t, "ok", stream.sent[0].Token)
	require.True(t, stream.sent[0].Complete)
}
