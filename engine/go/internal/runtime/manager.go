package runtime

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
)

type Manager struct {
	mu        sync.RWMutex
	models    map[string]*Model
	processes map[string]*exec.Cmd
	config    *config.Config
	logs      map[string][]string
}

type Service interface {
	GetStatus(context.Context, *emptypb.Empty) (*pb.RuntimeStatus, error)
	ListModels(context.Context, *emptypb.Empty) (*pb.ModelList, error)
	LoadModel(context.Context, *pb.LoadModelRequest) (*pb.ModelInfo, error)
	UnloadModel(context.Context, *pb.UnloadModelRequest) (*emptypb.Empty, error)
	StreamInference(context.Context, pb.Runtime_StreamInferenceServer) error
	LoadedModelCount() int
}

type Model struct {
	ID        string
	Name      string
	Path      string
	SizeBytes int64
	Loaded    bool
	Metadata  map[string]string
}

func NewManager(cfg *config.Config) *Manager {
	return &Manager{
		models:    make(map[string]*Model),
		processes: make(map[string]*exec.Cmd),
		logs:      make(map[string][]string),
		config:    cfg,
	}
}

func (m *Manager) GetStatus(ctx context.Context, _ *emptypb.Empty) (*pb.RuntimeStatus, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	loaded := make([]*pb.ModelInfo, 0, len(m.models))
	for _, model := range m.models {
		if model.Loaded {
			loaded = append(loaded, &pb.ModelInfo{
				Id:        model.ID,
				Name:      model.Name,
				Path:      model.Path,
				SizeBytes: model.SizeBytes,
				Loaded:    model.Loaded,
				Metadata:  model.Metadata,
			})
		}
	}

	return &pb.RuntimeStatus{
		Version:      "1.0.0",
		LoadedModels: loaded,
		Resources: &pb.SystemResources{
			// TODO: Add real metrics
			CpuPercent:       0,
			MemoryUsedBytes:  0,
			MemoryTotalBytes: 0,
		},
		Healthy: true,
	}, nil
}

func (m *Manager) ListModels(ctx context.Context, _ *emptypb.Empty) (*pb.ModelList, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	modelsPath := m.config.Runtime.ModelsPath
	models := make([]*pb.ModelInfo, 0)

	entries, err := os.ReadDir(modelsPath)
	if err != nil && !os.IsNotExist(err) {
		return nil, fmt.Errorf("failed to read models directory: %w", err)
	}

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		ext := filepath.Ext(entry.Name())
		if ext != ".bin" && ext != ".gguf" && ext != ".ggml" {
			continue
		}

		path := filepath.Join(modelsPath, entry.Name())
		info, err := os.Stat(path)
		if err != nil {
			continue
		}

		modelID := entry.Name()
		model, exists := m.models[modelID]
		if !exists {
			model = &Model{
				ID:        modelID,
				Name:      entry.Name(),
				Path:      path,
				SizeBytes: info.Size(),
				Loaded:    false,
				Metadata:  make(map[string]string),
			}
			m.models[modelID] = model
		}

		models = append(models, &pb.ModelInfo{
			Id:        model.ID,
			Name:      model.Name,
			Path:      model.Path,
			SizeBytes: model.SizeBytes,
			Loaded:    model.Loaded,
			Metadata:  model.Metadata,
		})
	}

	return &pb.ModelList{Models: models}, nil
}

func (m *Manager) LoadModel(ctx context.Context, req *pb.LoadModelRequest) (*pb.ModelInfo, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	modelID := req.ModelId
	model, exists := m.models[modelID]
	if !exists {
		return nil, fmt.Errorf("model not found: %s", modelID)
	}

	if model.Loaded {
		return &pb.ModelInfo{
			Id:        model.ID,
			Name:      model.Name,
			Path:      model.Path,
			SizeBytes: model.SizeBytes,
			Loaded:    model.Loaded,
			Metadata:  model.Metadata,
		}, nil
	}

	// In production, this would start llama.cpp or similar
	// For now, we simulate loading
	model.Loaded = true
	model.Metadata["loaded_at"] = time.Now().Format(time.RFC3339)

	return &pb.ModelInfo{
		Id:        model.ID,
		Name:      model.Name,
		Path:      model.Path,
		SizeBytes: model.SizeBytes,
		Loaded:    model.Loaded,
		Metadata:  model.Metadata,
	}, nil
}

func (m *Manager) UnloadModel(ctx context.Context, req *pb.UnloadModelRequest) (*emptypb.Empty, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	modelID := req.ModelId
	model, exists := m.models[modelID]
	if !exists {
		return nil, fmt.Errorf("model not found: %s", modelID)
	}

	model.Loaded = false
	delete(model.Metadata, "loaded_at")

	return &emptypb.Empty{}, nil
}

func (m *Manager) StreamInference(ctx context.Context, stream pb.Runtime_StreamInferenceServer) error {
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			req, err := stream.Recv()
			if err != nil {
				return err
			}
			_ = req

			// Simulate token streaming
			tokens := []string{"This", " is", " a", " sample", " response", "."}
			for i, token := range tokens {
				resp := &pb.InferenceResponse{
					Token:    token,
					Complete: i == len(tokens)-1,
					Metrics:  map[string]string{},
				}
				if err := stream.Send(resp); err != nil {
					return err
				}
				time.Sleep(50 * time.Millisecond)
			}
		}
	}
}

func (m *Manager) ListModelsCached() []*Model {
	m.mu.RLock()
	defer m.mu.RUnlock()
	result := make([]*Model, 0, len(m.models))
	for _, v := range m.models {
		result = append(result, v)
	}
	return result
}

func (m *Manager) LoadedModelCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()

	count := 0
	for _, model := range m.models {
		if model.Loaded {
			count++
		}
	}
	return count
}
