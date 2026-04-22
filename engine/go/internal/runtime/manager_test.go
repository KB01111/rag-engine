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
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/metadata"
	"google.golang.org/protobuf/types/known/emptypb"
)

func TestListModelsIncludesFilesystemAndProviderModels(t *testing.T) {
	t.Helper()

	modelDir := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(modelDir, "local.gguf"), []byte("model"), 0o644))

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"data":[{"id":"gpt-4o-mini","owned_by":"openai"}]}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	cfg := config.DefaultConfig()
	cfg.Runtime.ModelsPath = modelDir
	cfg.Runtime.Providers = []config.ProviderConfig{
		{Name: "cloud", Type: "openai-compatible", URL: server.URL},
	}

	manager := NewManager(cfg)
	models, err := manager.ListModels(context.Background(), &emptypb.Empty{})
	require.NoError(t, err)

	var ids []string
	for _, model := range models.Models {
		ids = append(ids, model.Id)
	}

	require.Contains(t, ids, "local.gguf")
	require.Contains(t, ids, "cloud/gpt-4o-mini")
}

func TestLoadAndStreamInferenceWithProvider(t *testing.T) {
	t.Helper()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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
	defer server.Close()

	cfg := config.DefaultConfig()
	cfg.Runtime.ModelsPath = t.TempDir()
	cfg.Runtime.Providers = []config.ProviderConfig{
		{Name: "cloud", Type: "openai-compatible", URL: server.URL},
	}

	manager := NewManager(cfg)
	loaded, err := manager.LoadModel(context.Background(), &pb.LoadModelRequest{
		ModelId: "cloud/gpt-4o-mini",
	})
	require.NoError(t, err)
	require.Equal(t, "cloud/gpt-4o-mini", loaded.Id)
	require.True(t, loaded.Loaded)

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

	err = manager.StreamInference(stream.ctx, stream)
	require.NoError(t, err)
	require.NotEmpty(t, stream.sent)

	var combined strings.Builder
	for _, resp := range stream.sent {
		combined.WriteString(resp.Token)
	}

	require.Equal(t, "Hello world", combined.String())
	require.True(t, stream.sent[len(stream.sent)-1].Complete)
	require.Equal(t, "cloud", stream.sent[len(stream.sent)-1].Metrics["provider"])
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
