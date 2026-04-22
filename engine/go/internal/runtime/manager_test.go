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

	s.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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
	s.Contains(ids, "cloud/gpt-4o-mini")
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
