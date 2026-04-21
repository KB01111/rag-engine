package training

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"time"

	"github.com/ai-engine/go/internal/config"
	pb "github.com/ai-engine/proto/go"
	"google.golang.org/protobuf/types/known/emptypb"
	"google.golang.org/protobuf/types/known/timestamppb"
)

type Manager struct {
	mu      sync.RWMutex
	runs    map[string]*Run
	config  *config.Config
	workDir string
}

type Run struct {
	ID          string
	Name        string
	Status      string
	ModelID     string
	DatasetPath string
	Config      map[string]string
	StartedAt   time.Time
	CompletedAt time.Time
	Progress    float32
	Error       string
	Cmd         *exec.Cmd
}

func NewManager(cfg *config.Config) *Manager {
	return &Manager{
		runs:    make(map[string]*Run),
		config:  cfg,
		workDir: cfg.Training.WorkingDir,
	}
}

func (m *Manager) StartRun(ctx context.Context, req *pb.TrainingRunRequest) (*pb.TrainingRun, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if len(m.runs) >= m.config.Training.MaxJobs {
		return nil, fmt.Errorf("max concurrent jobs reached")
	}

	runID := fmt.Sprintf("run-%d", time.Now().Unix())
	run := &Run{
		ID:          runID,
		Name:        req.Name,
		Status:      "starting",
		ModelID:     req.ModelId,
		DatasetPath: req.DatasetPath,
		Config:      req.Config,
		StartedAt:   time.Now(),
		Progress:    0,
	}

	// Create working directory for this run
	runDir := filepath.Join(m.workDir, runID)
	if err := os.MkdirAll(runDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create run directory: %w", err)
	}

	// In production, this would launch actual training (llama.cpp, etc.)
	// For now, simulate training progress
	go m.simulateTraining(run)

	m.runs[runID] = run

	return &pb.TrainingRun{
		Id:          run.ID,
		Name:        run.Name,
		Status:      run.Status,
		StartedAt:   timestamppb.New(run.StartedAt),
		CompletedAt: timestamppb.New(run.CompletedAt),
		Progress:    run.Progress,
		Error:       run.Error,
	}, nil
}

func (m *Manager) simulateTraining(run *Run) {
	m.mu.Lock()
	run.Status = "running"
	m.mu.Unlock()

	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	for i := 0; i < 10; i++ {
		<-ticker.C
		m.mu.Lock()
		run.Progress = float32(i+1) / 10.0
		m.mu.Unlock()
	}

	m.mu.Lock()
	run.Status = "completed"
	run.CompletedAt = time.Now()
	run.Progress = 1.0
	m.mu.Unlock()
}

func (m *Manager) CancelRun(ctx context.Context, req *pb.CancelRequest) (*emptypb.Empty, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	run, exists := m.runs[req.RunId]
	if !exists {
		return nil, fmt.Errorf("run not found: %s", req.RunId)
	}

	if run.Cmd != nil && run.Cmd.Process != nil {
		run.Cmd.Process.Kill()
	}

	run.Status = "cancelled"
	run.CompletedAt = time.Now()

	return &emptypb.Empty{}, nil
}

func (m *Manager) ListRuns(ctx context.Context, _ *emptypb.Empty) (*pb.TrainingRunList, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	runs := make([]*pb.TrainingRun, 0, len(m.runs))
	for _, run := range m.runs {
		runs = append(runs, &pb.TrainingRun{
			Id:          run.ID,
			Name:        run.Name,
			Status:      run.Status,
			StartedAt:   timestamppb.New(run.StartedAt),
			CompletedAt: timestamppb.New(run.CompletedAt),
			Progress:    run.Progress,
			Error:       run.Error,
		})
	}

	return &pb.TrainingRunList{Runs: runs}, nil
}

func (m *Manager) ListArtifacts(ctx context.Context, req *pb.ArtifactsRequest) (*pb.ArtifactList, error) {
	m.mu.RLock()
	run, exists := m.runs[req.RunId]
	m.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("run not found: %s", req.RunId)
	}

	_ = run

	runDir := filepath.Join(m.workDir, req.RunId)
	entries, err := os.ReadDir(runDir)
	if err != nil {
		return nil, fmt.Errorf("failed to read artifacts: %w", err)
	}

	artifacts := make([]*pb.Artifact, 0)
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			continue
		}
		artifacts = append(artifacts, &pb.Artifact{
			Name:      entry.Name(),
			Path:      filepath.Join(runDir, entry.Name()),
			SizeBytes: info.Size(),
			CreatedAt: timestamppb.New(info.ModTime()),
		})
	}

	return &pb.ArtifactList{Artifacts: artifacts}, nil
}

func (m *Manager) StreamLogs(req *pb.LogsRequest, stream pb.Training_StreamLogsServer) error {
	m.mu.RLock()
	run, exists := m.runs[req.RunId]
	m.mu.RUnlock()

	if !exists {
		return fmt.Errorf("run not found: %s", req.RunId)
	}

	_ = run

	// Simulate log streaming
	logs := []string{
		"Initializing training run...",
		"Loading dataset from " + run.DatasetPath,
		"Starting epoch 1/10",
		"Loss: 2.345",
		"Progress: 10%",
	}

	for i, msg := range logs {
		logEntry := &pb.TrainingLog{
			RunId:     run.ID,
			Level:     "info",
			Message:   msg,
			Timestamp: timestamppb.New(time.Now()),
		}
		if err := stream.Send(logEntry); err != nil {
			return err
		}
		time.Sleep(500 * time.Millisecond)
		if i == len(logs)-1 && !req.Follow {
			break
		}
	}

	return nil
}

func (m *Manager) ActiveRunCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	count := 0
	for _, run := range m.runs {
		if run.Status == "running" || run.Status == "starting" {
			count++
		}
	}
	return count
}
