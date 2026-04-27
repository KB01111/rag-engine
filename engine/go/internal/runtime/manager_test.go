package runtime

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ai-engine/go/internal/config"
	pb "github.com/ai-engine/proto/go"
	"github.com/stretchr/testify/suite"
	"golang.org/x/sync/errgroup"
	"google.golang.org/grpc/metadata"
	"google.golang.org/protobuf/types/known/emptypb"
)

type ManagerTestSuite struct {
	suite.Suite
	server   *httptest.Server
	cfg      *config.Config
	modelDir string
	manager  *Manager
}

func (s *ManagerTestSuite) SetupTest() {
	s.modelDir = s.T().TempDir()

	s.server = newIPv4Server(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"data":[{"id":"gpt-4o-mini","owned_by":"openai"}]}`))
		case "/v1/chat/completions":
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte(strings.Join([]string{
				`data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}`,
				``,
				`data: {"choices":[{"delta":{"content":" world"},"finish_reason":null}]}`,
				``,
				`data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"total_tokens":2}}`,
				``,
				`data: [DONE]`,
				``,
			}, "\n")))
		default:
			http.NotFound(w, r)
		}
	}))

	s.cfg = config.DefaultConfig()
	s.cfg.Runtime.ModelsPath = s.modelDir
	s.cfg.Runtime.Providers = []config.ProviderConfig{
		{Name: "cloud", Type: "openai-compatible", URL: s.server.URL},
	}

	s.manager = NewManager(s.cfg)
}

func (s *ManagerTestSuite) TearDownTest() {
	if s.server != nil {
		s.server.Close()
	}
}

func (s *ManagerTestSuite) TestListModelsIncludesFilesystemAndProviderModels() {
	s.Require().NoError(os.WriteFile(filepath.Join(s.modelDir, "local.gguf"), []byte("model"), 0o644))

	models, err := s.manager.ListModels(context.Background(), &emptypb.Empty{})
	s.Require().NoError(err)

	var ids []string
	for _, model := range models.Models {
		ids = append(ids, model.Id)
	}

	s.Contains(ids, "local.gguf")
	s.Contains(ids, "gpt-4o-mini")
}

func (s *ManagerTestSuite) TestLoadAndStreamInferenceWithProvider() {
	loaded, err := s.manager.LoadModel(context.Background(), &pb.LoadModelRequest{
		ModelId: "cloud/gpt-4o-mini",
	})
	s.Require().NoError(err)
	s.Equal("cloud/gpt-4o-mini", loaded.Id)
	s.True(loaded.Loaded)

	stream := &inferenceStreamStub{
		ctx: context.Background(),
		requests: []*pb.InferenceRequest{
			{
				ModelId:     "cloud/gpt-4o-mini",
				Provider:    "cloud",
				Prompt:      "Say hello",
				ContextRefs: []string{"viking://resources/workspace/docs/readme.md"},
			},
		},
	}

	err = s.manager.StreamInference(stream.ctx, stream)
	s.Require().NoError(err)
	s.NotEmpty(stream.sent)

	var combined strings.Builder
	for _, resp := range stream.sent {
		combined.WriteString(resp.Token)
	}

	s.Equal("Hello world", combined.String())
	s.True(stream.sent[len(stream.sent)-1].Complete)
	s.Equal("cloud", stream.sent[len(stream.sent)-1].Metrics["provider"])
}

func (s *ManagerTestSuite) TestListModelsConcurrentAccess() {
	s.Require().NoError(os.WriteFile(filepath.Join(s.cfg.Runtime.ModelsPath, "model.gguf"), []byte("weights"), 0o644))

	var g errgroup.Group
	for i := 0; i < 8; i++ {
		g.Go(func() error {
			_, err := s.manager.ListModels(context.Background(), &emptypb.Empty{})
			return err
		})
	}

	s.Require().NoError(g.Wait())
}

func TestManagerTestSuite(t *testing.T) {
	suite.Run(t, &ManagerTestSuite{})
}

func TestProviderMessagesIncludeSystemPromptAndContextRefs(t *testing.T) {
	req := &pb.InferenceRequest{
		Prompt:       "What changed?",
		SystemPrompt: protoString("You are concise."),
		ContextRefs:  []string{"viking://resources/doc-a", "viking://resources/doc-b"},
	}

	messages := buildMessages(req)

	if len(messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(messages))
	}
	if messages[0]["role"] != "system" {
		t.Fatalf("expected first message to be system, got %q", messages[0]["role"])
	}
	if !strings.Contains(messages[0]["content"], "You are concise.") {
		t.Fatalf("system message missing explicit system prompt: %q", messages[0]["content"])
	}
	if !strings.Contains(messages[0]["content"], "viking://resources/doc-a") {
		t.Fatalf("system message missing context ref: %q", messages[0]["content"])
	}
	if messages[1]["role"] != "user" || messages[1]["content"] != "What changed?" {
		t.Fatalf("unexpected user message: %#v", messages[1])
	}
}

func TestProviderPresetAndRequestIDHeaderRetry(t *testing.T) {
	var attempts int
	var requestIDs []string
	client := &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			requestIDs = append(requestIDs, r.Header.Get("X-Request-ID"))
			if r.URL.Path != "/api/v1/chat/completions" {
				return &http.Response{
					StatusCode: http.StatusNotFound,
					Status:     "404 Not Found",
					Header:     make(http.Header),
					Body:       io.NopCloser(strings.NewReader("not found")),
					Request:    r,
				}, nil
			}
			attempts++
			if attempts == 1 {
				return &http.Response{
					StatusCode: http.StatusTooManyRequests,
					Status:     "429 Too Many Requests",
					Header:     make(http.Header),
					Body:       io.NopCloser(strings.NewReader("try again")),
					Request:    r,
				}, nil
			}
			header := make(http.Header)
			header.Set("Content-Type", "application/json")
			return &http.Response{
				StatusCode: http.StatusOK,
				Status:     "200 OK",
				Header:     header,
				Body:       io.NopCloser(strings.NewReader(`{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"total_tokens":1}}`)),
				Request:    r,
			}, nil
		}),
	}

	provider, err := newOpenAICompatibleProvider(config.ProviderConfig{
		Name:   "lemonade",
		Preset: "lemonade",
		URL:    "http://provider.test/api/v1",
	}, client, client)
	if err != nil {
		t.Fatalf("new provider: %v", err)
	}

	ctx := metadata.AppendToOutgoingContext(context.Background(), "x-request-id", "req-123")
	var sent []*pb.InferenceResponse
	err = provider.StreamInference(ctx, "model-a", &pb.InferenceRequest{Prompt: "hi"}, func(resp *pb.InferenceResponse) error {
		sent = append(sent, resp)
		return nil
	})
	if err != nil {
		t.Fatalf("stream inference: %v", err)
	}
	if attempts != 2 {
		t.Fatalf("expected 2 attempts, got %d", attempts)
	}
	if got := strings.Join(requestIDs, ","); got != "req-123,req-123" {
		t.Fatalf("unexpected request IDs: %v", requestIDs)
	}
	if len(sent) == 0 || sent[0].Token != "ok" {
		t.Fatalf("unexpected sent responses: %#v", sent)
	}
}

func protoString(value string) *string {
	return &value
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

type inferenceStreamStub struct {
	ctx      context.Context
	requests []*pb.InferenceRequest
	sent     []*pb.InferenceResponse
}

func (s *inferenceStreamStub) Context() context.Context {
	return s.ctx
}

func (s *inferenceStreamStub) Send(resp *pb.InferenceResponse) error {
	s.sent = append(s.sent, resp)
	return nil
}

func (s *inferenceStreamStub) Recv() (*pb.InferenceRequest, error) {
	if len(s.requests) == 0 {
		return nil, io.EOF
	}
	req := s.requests[0]
	s.requests = s.requests[1:]
	return req, nil
}

func (s *inferenceStreamStub) SetHeader(metadata.MD) error {
	return nil
}

func (s *inferenceStreamStub) SendHeader(metadata.MD) error {
	return nil
}

func (s *inferenceStreamStub) SetTrailer(metadata.MD) {}

func (s *inferenceStreamStub) SendMsg(any) error {
	return nil
}

func (s *inferenceStreamStub) RecvMsg(any) error {
	return nil
}
