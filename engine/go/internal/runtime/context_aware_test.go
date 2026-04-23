package runtime

import (
	"context"
	"io"
	"strings"
	"testing"

	"github.com/ai-engine/go/internal/contextsvc"
	pb "github.com/ai-engine/proto/go"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/metadata"
	"google.golang.org/protobuf/types/known/emptypb"
)

type runtimeRecorder struct {
	seen []*pb.InferenceRequest
}

func (r *runtimeRecorder) GetStatus(context.Context, *emptypb.Empty) (*pb.RuntimeStatus, error) {
	return &pb.RuntimeStatus{}, nil
}

func (r *runtimeRecorder) ListModels(context.Context, *emptypb.Empty) (*pb.ModelList, error) {
	return &pb.ModelList{}, nil
}

func (r *runtimeRecorder) LoadModel(context.Context, *pb.LoadModelRequest) (*pb.ModelInfo, error) {
	return &pb.ModelInfo{}, nil
}

func (r *runtimeRecorder) UnloadModel(context.Context, *pb.UnloadModelRequest) (*emptypb.Empty, error) {
	return &emptypb.Empty{}, nil
}

func (r *runtimeRecorder) LoadedModelCount() int {
	return 0
}

func (r *runtimeRecorder) StreamInference(_ context.Context, stream pb.Runtime_StreamInferenceServer) error {
	for {
		req, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		r.seen = append(r.seen, req)
	}

	return stream.Send(&pb.InferenceResponse{
		Token:    "Use Dragonfly.",
		Complete: true,
	})
}

type contextBackendStub struct {
	enabled         bool
	session         *contextsvc.SessionResponse
	sessionErr      error
	searchResponses []*contextsvc.SearchResponse
	searchErr       error
	searchErrors    map[string]error
	searchRequests  []contextsvc.SearchRequest
	appendRequests  []contextsvc.SessionAppendRequest
	upsertRequests  []contextsvc.UpsertResourceRequest
}

func (s *contextBackendStub) Enabled() bool {
	return s.enabled
}

func (s *contextBackendStub) GetSession(_ context.Context, sessionID string) (*contextsvc.SessionResponse, error) {
	if s.sessionErr != nil {
		return nil, s.sessionErr
	}
	if s.session != nil {
		return s.session, nil
	}
	return &contextsvc.SessionResponse{SessionID: sessionID}, nil
}

func (s *contextBackendStub) Search(_ context.Context, req contextsvc.SearchRequest) (*contextsvc.SearchResponse, error) {
	s.searchRequests = append(s.searchRequests, req)
	if err := s.searchErrors[req.Filters["kind"]]; err != nil {
		return nil, err
	}
	if s.searchErr != nil {
		return nil, s.searchErr
	}
	if len(s.searchResponses) == 0 {
		return &contextsvc.SearchResponse{}, nil
	}
	resp := s.searchResponses[0]
	s.searchResponses = s.searchResponses[1:]
	return resp, nil
}

func (s *contextBackendStub) AppendSession(_ context.Context, req contextsvc.SessionAppendRequest) (*contextsvc.SessionResponse, error) {
	s.appendRequests = append(s.appendRequests, req)
	return &contextsvc.SessionResponse{
		SessionID: req.SessionID,
		Entries: []contextsvc.SessionEntry{
			{
				SessionID: req.SessionID,
				Role:      req.Role,
				Content:   req.Content,
			},
		},
	}, nil
}

func (s *contextBackendStub) UpsertResource(_ context.Context, req contextsvc.UpsertResourceRequest) (*contextsvc.UpsertResourceResponse, error) {
	s.upsertRequests = append(s.upsertRequests, req)
	return &contextsvc.UpsertResourceResponse{
		Resource: contextsvc.Resource{
			URI:      req.URI,
			Title:    req.Title,
			Layer:    req.Layer,
			Metadata: req.Metadata,
		},
		ChunksIndexed: 1,
	}, nil
}

type inferenceStreamHarness struct {
	ctx      context.Context
	requests []*pb.InferenceRequest
	sent     []*pb.InferenceResponse
}

func (s *inferenceStreamHarness) Context() context.Context { return s.ctx }
func (s *inferenceStreamHarness) Send(resp *pb.InferenceResponse) error {
	s.sent = append(s.sent, resp)
	return nil
}
func (s *inferenceStreamHarness) Recv() (*pb.InferenceRequest, error) {
	if len(s.requests) == 0 {
		return nil, io.EOF
	}
	req := s.requests[0]
	s.requests = s.requests[1:]
	return req, nil
}
func (s *inferenceStreamHarness) SetHeader(metadata.MD) error  { return nil }
func (s *inferenceStreamHarness) SendHeader(metadata.MD) error { return nil }
func (s *inferenceStreamHarness) SetTrailer(metadata.MD)       {}
func (s *inferenceStreamHarness) SendMsg(any) error            { return nil }
func (s *inferenceStreamHarness) RecvMsg(any) error            { return nil }

func TestContextAwareServiceAssemblesPromptAndPersistsTurns(t *testing.T) {
	inner := &runtimeRecorder{}
	contextBackend := &contextBackendStub{
		enabled: true,
		session: &contextsvc.SessionResponse{
			SessionID: "sess-1",
			Entries: []contextsvc.SessionEntry{
				{Role: "user", Content: "We moved away from Redis."},
				{Role: "assistant", Content: "Dragonfly is the preferred working memory store."},
			},
		},
		searchResponses: []*contextsvc.SearchResponse{
			{
				Results: []contextsvc.SearchHit{
					{
						URI:       "viking://graph/project-x",
						ChunkText: "Project X uses Dragonfly for hot memory.",
						Metadata:  map[string]string{"kind": "graph"},
					},
				},
			},
			{
				Results: []contextsvc.SearchHit{
					{
						URI:       "viking://resources/workspace/docs/project-x.md",
						ChunkText: "Project X is local-first and stores session context durably.",
						Metadata:  map[string]string{"kind": "resource"},
					},
				},
			},
		},
	}

	service := NewContextAwareService(inner, contextBackend)
	stream := &inferenceStreamHarness{
		ctx: context.Background(),
		requests: []*pb.InferenceRequest{
			{
				ModelId: "cloud/gpt-4o-mini",
				Prompt:  "What should Project X use for working memory?",
				Parameters: map[string]string{
					"session_id": "sess-1",
				},
			},
		},
	}

	err := service.StreamInference(stream.ctx, stream)
	require.NoError(t, err)
	require.Len(t, inner.seen, 1)
	require.Len(t, stream.sent, 1)
	require.True(t, stream.sent[0].Complete)

	assembledPrompt := inner.seen[0].GetPrompt()
	require.Contains(t, assembledPrompt, "Recent working memory")
	require.Contains(t, assembledPrompt, "Graph facts")
	require.Contains(t, assembledPrompt, "Retrieved documents")
	require.Contains(t, assembledPrompt, "Dragonfly")
	require.Contains(t, assembledPrompt, "What should Project X use for working memory?")
	require.Contains(t, inner.seen[0].GetContextRefs(), "viking://resources/workspace/docs/project-x.md")

	require.Len(t, contextBackend.searchRequests, 2)
	require.Len(t, contextBackend.appendRequests, 2)
	require.Len(t, contextBackend.upsertRequests, 1)
	require.Equal(t, "user", contextBackend.appendRequests[0].Role)
	require.Equal(t, "assistant", contextBackend.appendRequests[1].Role)
	require.True(t, strings.Contains(contextBackend.appendRequests[1].Content, "Dragonfly"))
	require.Equal(t, "graph", contextBackend.upsertRequests[0].Metadata["kind"])
	require.Equal(t, "PREFERS", contextBackend.upsertRequests[0].Metadata["relation"])
	require.Equal(t, "dragonfly", contextBackend.upsertRequests[0].Metadata["object_id"])
}

func TestContextAwareServiceFallsBackWhenContextLookupsFail(t *testing.T) {
	inner := &runtimeRecorder{}
	contextBackend := &contextBackendStub{
		enabled:    true,
		sessionErr: io.ErrUnexpectedEOF,
		searchErr:  io.ErrUnexpectedEOF,
	}

	service := NewContextAwareService(inner, contextBackend)
	stream := &inferenceStreamHarness{
		ctx: context.Background(),
		requests: []*pb.InferenceRequest{
			{
				ModelId: "local.gguf",
				Prompt:  "Answer directly.",
				Parameters: map[string]string{
					"session_id": "sess-2",
				},
			},
		},
	}

	err := service.StreamInference(stream.ctx, stream)
	require.NoError(t, err)
	require.Len(t, inner.seen, 1)
	require.Equal(t, "Answer directly.", inner.seen[0].GetPrompt())
	require.Empty(t, inner.seen[0].GetContextRefs())
	require.Empty(t, contextBackend.upsertRequests)
	require.Len(t, contextBackend.appendRequests, 2)
}

func TestContextAwareServiceKeepsSurvivingContextSourcesOnPartialFailures(t *testing.T) {
	inner := &runtimeRecorder{}
	contextBackend := &contextBackendStub{
		enabled: true,
		session: &contextsvc.SessionResponse{
			SessionID: "sess-3",
			Entries: []contextsvc.SessionEntry{
				{Role: "user", Content: "Project X used Redis before."},
				{Role: "assistant", Content: "We should keep local-first context."},
			},
		},
		searchResponses: []*contextsvc.SearchResponse{
			{
				Results: []contextsvc.SearchHit{
					{
						URI:       "viking://resources/workspace/docs/project-x.md",
						ChunkText: "Project X keeps context local and durable.",
						Metadata:  map[string]string{"kind": "resource"},
					},
				},
			},
		},
		searchErrors: map[string]error{
			"graph": io.ErrUnexpectedEOF,
		},
	}

	service := NewContextAwareService(inner, contextBackend)
	stream := &inferenceStreamHarness{
		ctx: context.Background(),
		requests: []*pb.InferenceRequest{
			{
				ModelId: "local.gguf",
				Prompt:  "How should Project X keep context?",
				Parameters: map[string]string{
					"session_id": "sess-3",
				},
			},
		},
	}

	err := service.StreamInference(stream.ctx, stream)
	require.NoError(t, err)
	require.Len(t, inner.seen, 1)
	require.Contains(t, inner.seen[0].GetPrompt(), "Recent working memory")
	require.NotContains(t, inner.seen[0].GetPrompt(), "Graph facts")
	require.Contains(t, inner.seen[0].GetPrompt(), "Retrieved documents")
	require.Contains(t, inner.seen[0].GetContextRefs(), "viking://resources/workspace/docs/project-x.md")
}