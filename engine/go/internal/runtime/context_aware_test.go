package runtime

import (
	"context"
	"io"
	"strings"
	"sync"
	"testing"

	"github.com/ai-engine/go/internal/contextsvc"
	pb "github.com/ai-engine/proto/go"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
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
	mu              sync.Mutex
	enabled         bool
	session         *contextsvc.SessionResponse
	sessionErr      error
	searchResponses map[string]*contextsvc.SearchResponse
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
	s.mu.Lock()
	defer s.mu.Unlock()

	s.searchRequests = append(s.searchRequests, req)

	if err := s.searchErrors[req.Filters["kind"]]; err != nil {
		return nil, err
	}
	if s.searchErr != nil {
		return nil, s.searchErr
	}

	kind := req.Filters["kind"]
	if resp, ok := s.searchResponses[kind]; ok {
		return resp, nil
	}

	return &contextsvc.SearchResponse{}, nil
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

type ContextAwareSuite struct {
	suite.Suite
	inner          *runtimeRecorder
	contextBackend *contextBackendStub
	service        *ContextAwareService
	stream         *inferenceStreamHarness
}

func (s *ContextAwareSuite) SetupTest() {
	s.inner = &runtimeRecorder{}
	s.contextBackend = &contextBackendStub{}
	s.service = NewContextAwareService(s.inner, s.contextBackend)
	s.stream = &inferenceStreamHarness{
		ctx: context.Background(),
	}
}

func TestContextAwareSuite(t *testing.T) {
	suite.Run(t, new(ContextAwareSuite))
}

func (s *ContextAwareSuite) TestAssemblesPromptAndPersistsTurns() {
	s.contextBackend.enabled = true
	s.contextBackend.session = &contextsvc.SessionResponse{
		SessionID: "sess-1",
		Entries: []contextsvc.SessionEntry{
			{Role: "user", Content: "We moved away from Redis."},
			{Role: "assistant", Content: "Dragonfly is the preferred working memory store."},
		},
	}
	s.contextBackend.searchResponses = map[string]*contextsvc.SearchResponse{
		"graph": {
			Results: []contextsvc.SearchHit{
				{
					URI:       "viking://graph/project-x",
					ChunkText: "Project X uses Dragonfly for hot memory.",
					Metadata:  map[string]string{"kind": "graph"},
				},
			},
		},
		"resource": {
			Results: []contextsvc.SearchHit{
				{
					URI:       "viking://resources/workspace/docs/project-x.md",
					ChunkText: "Project X is local-first and stores session context durably.",
					Metadata:  map[string]string{"kind": "resource"},
				},
			},
		},
	}

	s.stream.requests = []*pb.InferenceRequest{
		{
			ModelId: "cloud/gpt-4o-mini",
			Prompt:  "What should Project X use for working memory?",
			Parameters: map[string]string{
				"session_id": "sess-1",
			},
		},
	}

	err := s.service.StreamInference(s.stream.ctx, s.stream)
	s.Require().NoError(err)
	s.Require().Len(s.inner.seen, 1)
	s.Require().Len(s.stream.sent, 1)
	s.Require().True(s.stream.sent[0].Complete)

	assembledPrompt := s.inner.seen[0].GetPrompt()
	s.Require().Contains(assembledPrompt, "Recent working memory")
	s.Require().Contains(assembledPrompt, "Graph facts")
	s.Require().Contains(assembledPrompt, "Retrieved documents")
	s.Require().Contains(assembledPrompt, "Dragonfly")
	s.Require().Contains(assembledPrompt, "What should Project X use for working memory?")
	s.Require().Contains(s.inner.seen[0].GetContextRefs(), "viking://resources/workspace/docs/project-x.md")

	s.Require().Len(s.contextBackend.searchRequests, 2)
	s.Require().Len(s.contextBackend.appendRequests, 2)
	s.Require().Len(s.contextBackend.upsertRequests, 1)
	s.Require().Equal("user", s.contextBackend.appendRequests[0].Role)
	s.Require().Equal("assistant", s.contextBackend.appendRequests[1].Role)
	s.Require().True(strings.Contains(s.contextBackend.appendRequests[1].Content, "Dragonfly"))
	s.Require().Equal("graph", s.contextBackend.upsertRequests[0].Metadata["kind"])
	s.Require().Equal("PREFERS", s.contextBackend.upsertRequests[0].Metadata["relation"])
	s.Require().Equal("dragonfly", s.contextBackend.upsertRequests[0].Metadata["object_id"])
}

func (s *ContextAwareSuite) TestFallsBackWhenContextLookupsFail() {
	s.contextBackend.enabled = true
	s.contextBackend.sessionErr = io.ErrUnexpectedEOF
	s.contextBackend.searchErr = io.ErrUnexpectedEOF

	s.stream.requests = []*pb.InferenceRequest{
		{
			ModelId: "local.gguf",
			Prompt:  "Answer directly.",
			Parameters: map[string]string{
				"session_id": "sess-2",
			},
		},
	}

	err := s.service.StreamInference(s.stream.ctx, s.stream)
	s.Require().NoError(err)
	s.Require().Len(s.inner.seen, 1)
	s.Require().Equal("Answer directly.", s.inner.seen[0].GetPrompt())
	s.Require().Empty(s.inner.seen[0].GetContextRefs())
	s.Require().Empty(s.contextBackend.upsertRequests)
	s.Require().Len(s.contextBackend.appendRequests, 2)
}

func (s *ContextAwareSuite) TestKeepsSurvivingContextSourcesOnPartialFailures() {
	s.contextBackend.enabled = true
	s.contextBackend.session = &contextsvc.SessionResponse{
		SessionID: "sess-3",
		Entries: []contextsvc.SessionEntry{
			{Role: "user", Content: "Project X used Redis before."},
			{Role: "assistant", Content: "We should keep local-first context."},
		},
	}
	s.contextBackend.searchResponses = map[string]*contextsvc.SearchResponse{
		"resource": {
			Results: []contextsvc.SearchHit{
				{
					URI:       "viking://resources/workspace/docs/project-x.md",
					ChunkText: "Project X keeps context local and durable.",
					Metadata:  map[string]string{"kind": "resource"},
				},
			},
		},
	}
	s.contextBackend.searchErrors = map[string]error{
		"graph": io.ErrUnexpectedEOF,
	}

	s.stream.requests = []*pb.InferenceRequest{
		{
			ModelId: "local.gguf",
			Prompt:  "How should Project X keep context?",
			Parameters: map[string]string{
				"session_id": "sess-3",
			},
		},
	}

	err := s.service.StreamInference(s.stream.ctx, s.stream)
	s.Require().NoError(err)
	s.Require().Len(s.inner.seen, 1)
	s.Require().Contains(s.inner.seen[0].GetPrompt(), "Recent working memory")
	s.Require().NotContains(s.inner.seen[0].GetPrompt(), "Graph facts")
	s.Require().Contains(s.inner.seen[0].GetPrompt(), "Retrieved documents")
	s.Require().Contains(s.inner.seen[0].GetContextRefs(), "viking://resources/workspace/docs/project-x.md")
}