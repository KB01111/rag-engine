package api

import (
	"context"
	"testing"

	"github.com/ai-engine/go/internal/config"
	"github.com/ai-engine/go/internal/supervisor"
	pb "github.com/ai-engine/proto/go"
	"github.com/rs/zerolog"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func TestTrainingEndpointsReturnUnimplementedWhenDisabled(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.Daemon.Required = false
	cfg.Services.EnableTraining = false

	server := NewServer(cfg, supervisor.NewSupervisor(cfg), zerolog.Nop())
	_, err := server.StartRun(context.Background(), &pb.TrainingRunRequest{Name: "demo"})
	if status.Code(err) != codes.Unimplemented {
		t.Fatalf("expected unimplemented error, got %v", err)
	}
}

func TestMCPEndpointsReturnUnimplementedWhenDisabled(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.Daemon.Required = false
	cfg.Services.EnableMCP = false

	server := NewServer(cfg, supervisor.NewSupervisor(cfg), zerolog.Nop())
	_, err := server.Connect(context.Background(), &pb.MCPConnectionRequest{ServerUrl: "http://localhost:3000"})
	if status.Code(err) != codes.Unimplemented {
		t.Fatalf("expected unimplemented error, got %v", err)
	}
}
