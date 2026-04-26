package runtime

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ai-engine/go/internal/config"
	"github.com/ai-engine/go/internal/hub"
	pb "github.com/ai-engine/proto/go"
	"google.golang.org/protobuf/types/known/emptypb"
)

type Manager struct {
	mu        sync.RWMutex
	models    map[string]*Model
	processes map[string]struct{}
	config    *config.Config
	http      *http.Client
	providers map[string]*openAICompatibleProvider
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
	ID              string
	Name            string
	Path            string
	Provider        string
	ProviderModelID string
	SizeBytes       int64
	Loaded          bool
	Metadata        map[string]string
}

type openAICompatibleProvider struct {
	name         string
	providerType string
	baseURL      string
	apiKey       string
	http         *http.Client
	streamHttp   *http.Client
}

type openAIModelListResponse struct {
	Data []struct {
		ID      string `json:"id"`
		OwnedBy string `json:"owned_by"`
	} `json:"data"`
}

type openAIChatResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
		Text         string  `json:"text"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Usage map[string]any `json:"usage"`
}

type openAIStreamChunk struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
		Text         string  `json:"text"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Usage map[string]any `json:"usage"`
}

func NewManager(cfg *config.Config) *Manager {
	httpClient := &http.Client{Timeout: 45 * time.Second}
	streamHttpClient := &http.Client{}
	providers := make(map[string]*openAICompatibleProvider)
	for _, providerCfg := range cfg.Runtime.Providers {
		provider, err := newOpenAICompatibleProvider(providerCfg, httpClient, streamHttpClient)
		if err != nil {
			continue
		}
		providers[provider.name] = provider
	}

	return &Manager{
		models:    make(map[string]*Model),
		processes: make(map[string]struct{}),
		config:    cfg,
		http:      httpClient,
		providers: providers,
	}
}

func (m *Manager) GetStatus(ctx context.Context, _ *emptypb.Empty) (*pb.RuntimeStatus, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	loaded := make([]*pb.ModelInfo, 0, len(m.models))
	for _, model := range m.models {
		if model.Loaded {
			loaded = append(loaded, model.toProto())
		}
	}

	sort.Slice(loaded, func(i, j int) bool {
		return loaded[i].Id < loaded[j].Id
	})

	return &pb.RuntimeStatus{
		Version:      "1.1.0",
		LoadedModels: loaded,
		Resources: &pb.SystemResources{
			CpuPercent:       0,
			MemoryUsedBytes:  0,
			MemoryTotalBytes: 0,
		},
		Healthy: true,
	}, nil
}

func (m *Manager) ListModels(ctx context.Context, _ *emptypb.Empty) (*pb.ModelList, error) {
	discovered := make(map[string]*pb.ModelInfo)

	localModels, err := m.discoverLocalModels()
	if err != nil {
		return nil, err
	}
	for _, model := range localModels {
		discovered[modelStoreKey(model)] = model
	}

	for _, provider := range m.providers {
		models, err := provider.ListModels(ctx)
		if err != nil {
			continue
		}
		for _, model := range models {
			discovered[modelStoreKey(model)] = model
		}
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	for key, modelInfo := range discovered {
		existing := m.models[key]
		loaded := existing != nil && existing.Loaded
		displayID := modelInfo.Id
		if loaded && existing != nil && existing.ID != "" {
			displayID = existing.ID
		}
		metadata := cloneStringMap(modelInfo.Metadata)
		if metadata == nil {
			metadata = map[string]string{}
		}
		if loaded && existing.Metadata != nil {
			metadata["loaded_at"] = existing.Metadata["loaded_at"]
		}

		m.models[key] = &Model{
			ID:              displayID,
			Name:            modelInfo.Name,
			Path:            modelInfo.Path,
			Provider:        metadata["provider"],
			ProviderModelID: metadata["provider_model_id"],
			SizeBytes:       modelInfo.SizeBytes,
			Loaded:          loaded,
			Metadata:        metadata,
		}
		discovered[key] = m.models[key].toProto()
	}

	for id, model := range m.models {
		if model.Loaded {
			discovered[id] = model.toProto()
		} else if _, inDiscovered := discovered[id]; !inDiscovered {
			delete(m.models, id)
		}
	}

	models := make([]*pb.ModelInfo, 0, len(discovered))
	for _, model := range discovered {
		models = append(models, model)
	}
	sort.Slice(models, func(i, j int) bool {
		return models[i].Id < models[j].Id
	})

	return &pb.ModelList{Models: models}, nil
}

func (m *Manager) LoadModel(ctx context.Context, req *pb.LoadModelRequest) (*pb.ModelInfo, error) {
	if _, err := m.ListModels(ctx, &emptypb.Empty{}); err != nil {
		return nil, err
	}

	modelID, providerName, providerModelID := m.resolveRequestedModel(req.GetModelId(), req.GetOptions()["provider"])

	m.mu.Lock()
	defer m.mu.Unlock()

	model, exists := m.models[modelID]
	if !exists && providerName != "" {
		if _, ok := m.providers[providerName]; !ok {
			return nil, fmt.Errorf("provider not found: %s", providerName)
		}

		metadata := map[string]string{
			"provider":          providerName,
			"provider_model_id": providerModelID,
			"source":            "provider",
		}
		model = &Model{
			ID:              modelID,
			Name:            providerModelID,
			Provider:        providerName,
			ProviderModelID: providerModelID,
			Loaded:          false,
			Metadata:        metadata,
		}
		m.models[modelID] = model
	}
	if !exists && providerName == "" {
		return nil, fmt.Errorf("model not found: %s", req.GetModelId())
	}

	model.Loaded = true
	if providerName != "" {
		model.ID = modelID
	}
	if model.Metadata == nil {
		model.Metadata = map[string]string{}
	}
	model.Metadata["loaded_at"] = time.Now().Format(time.RFC3339)
	return model.toProto(), nil
}

func (m *Manager) UnloadModel(ctx context.Context, req *pb.UnloadModelRequest) (*emptypb.Empty, error) {
	_, _, _ = ctx, req, time.Now()

	modelID, providerName, providerModelID := m.resolveRequestedModel(req.GetModelId(), "")

	m.mu.Lock()
	defer m.mu.Unlock()
	if providerName != "" {
		modelID = providerModelIDToID(providerName, providerModelID)
	}

	model, exists := m.models[modelID]
	if !exists {
		return nil, fmt.Errorf("model not found: %s", req.GetModelId())
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
		}

		req, err := stream.Recv()
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}

		provider, providerModelID, ok := m.resolveProviderForInference(req)
		if ok {
			if err := provider.StreamInference(ctx, providerModelID, req, stream.Send); err != nil {
				return err
			}
			continue
		}

		if req.GetProvider() != "" {
			return fmt.Errorf("unknown provider: %s", req.GetProvider())
		}
		if providerName, _, ok := splitProviderModelID(req.GetModelId()); ok {
			return fmt.Errorf("unknown provider: %s", providerName)
		}

		if err := streamLocalFallback(ctx, req, stream.Send); err != nil {
			return err
		}
	}
}

func (m *Manager) ListModelsCached() []*Model {
	m.mu.RLock()
	defer m.mu.RUnlock()
	result := make([]*Model, 0, len(m.models))
	for _, model := range m.models {
		result = append(result, model)
	}
	return result
}

func (m *Manager) discoverLocalModels() ([]*pb.ModelInfo, error) {
	entries, err := os.ReadDir(m.config.Runtime.ModelsPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("failed to read models directory: %w", err)
	}

	m.mu.RLock()
	loaded := make(map[string]bool, len(m.models))
	loadedAt := make(map[string]string, len(m.models))
	for id, model := range m.models {
		loaded[id] = model.Loaded
		if model.Metadata != nil {
			loadedAt[id] = model.Metadata["loaded_at"]
		}
	}
	m.mu.RUnlock()

	models := make([]*pb.ModelInfo, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		ext := strings.ToLower(filepath.Ext(entry.Name()))
		if ext != ".bin" && ext != ".gguf" && ext != ".ggml" {
			continue
		}

		path := filepath.Join(m.config.Runtime.ModelsPath, entry.Name())
		info, err := os.Stat(path)
		if err != nil {
			continue
		}

		metadata := huggingFaceSidecarMetadata(path)
		if metadata == nil {
			metadata = map[string]string{
				"source": "filesystem",
			}
		}
		if loadedAt[entry.Name()] != "" {
			metadata["loaded_at"] = loadedAt[entry.Name()]
		}

		models = append(models, &pb.ModelInfo{
			Id:        entry.Name(),
			Name:      entry.Name(),
			Path:      path,
			SizeBytes: info.Size(),
			Loaded:    loaded[entry.Name()],
			Metadata:  metadata,
		})
	}

	return models, nil
}

func huggingFaceSidecarMetadata(modelPath string) map[string]string {
	var manifest hub.Manifest
	data, err := os.ReadFile(modelPath + ".hf.json")
	if err != nil {
		return nil
	}
	if err := json.Unmarshal(data, &manifest); err != nil {
		return nil
	}
	if strings.TrimSpace(manifest.RepoID) == "" {
		return nil
	}

	metadata := map[string]string{
		"source":     "huggingface",
		"downloaded": "true",
		"repo_id":    manifest.RepoID,
		"filename":   manifest.Filename,
		"revision":   manifest.Revision,
	}
	if manifest.ResolvedURL != "" {
		metadata["resolved_url"] = manifest.ResolvedURL
	}
	if manifest.ETag != "" {
		metadata["etag"] = manifest.ETag
	}
	if manifest.SHA != "" {
		metadata["sha"] = manifest.SHA
	}
	if manifest.License != "" {
		metadata["license"] = manifest.License
	}
	if !manifest.DownloadedAt.IsZero() {
		metadata["downloaded_at"] = manifest.DownloadedAt.Format(time.RFC3339)
	}
	return metadata
}

func (m *Manager) resolveRequestedModel(modelID, providerName string) (string, string, string) {
	if providerName != "" {
		actualModel := stripProviderPrefix(modelID, providerName)
		return providerModelIDToID(providerName, actualModel), providerName, actualModel
	}

	if inferredProvider, actualModel, ok := splitProviderModelID(modelID); ok {
		return providerModelIDToID(inferredProvider, actualModel), inferredProvider, actualModel
	}

	m.mu.RLock()
	defer m.mu.RUnlock()
	if model, ok := m.models[modelID]; ok && model.Provider != "" {
		return modelID, model.Provider, effectiveProviderModelID(model)
	}
	for key, model := range m.models {
		if model.Provider == "" || model.ID != modelID {
			continue
		}
		return key, model.Provider, effectiveProviderModelID(model)
	}

	return modelID, "", modelID
}

func (m *Manager) resolveProviderForInference(req *pb.InferenceRequest) (*openAICompatibleProvider, string, bool) {
	modelID, providerName, providerModelID := m.resolveRequestedModel(req.GetModelId(), req.GetProvider())
	_ = modelID
	if providerName == "" {
		if req.GetProvider() != "" {
			return nil, "", false
		}
		if _, _, ok := splitProviderModelID(req.GetModelId()); ok {
			return nil, "", false
		}
		return nil, "", false
	}

	provider, ok := m.providers[providerName]
	if !ok {
		return nil, "", false
	}

	return provider, providerModelID, true
}

func (m *Manager) modelByID(modelID string) *Model {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.models[modelID]
}

func (m *Model) toProto() *pb.ModelInfo {
	return &pb.ModelInfo{
		Id:        m.ID,
		Name:      m.Name,
		Path:      m.Path,
		SizeBytes: m.SizeBytes,
		Loaded:    m.Loaded,
		Metadata:  cloneStringMap(m.Metadata),
	}
}

func newOpenAICompatibleProvider(cfg config.ProviderConfig, httpClient *http.Client, streamHttpClient *http.Client) (*openAICompatibleProvider, error) {
	name := strings.TrimSpace(cfg.Name)
	if name == "" {
		return nil, fmt.Errorf("provider name is required")
	}
	baseURL := strings.TrimSpace(cfg.URL)
	if baseURL == "" {
		return nil, fmt.Errorf("provider url is required for %s", name)
	}

	parsed, err := url.Parse(baseURL)
	if err != nil {
		return nil, fmt.Errorf("invalid provider url for %s: %w", name, err)
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return nil, fmt.Errorf("invalid provider url for %s: must be absolute http(s) URL", name)
	}
	if parsed.Host == "" {
		return nil, fmt.Errorf("invalid provider url for %s: must be absolute http(s) URL", name)
	}

	providerType := strings.ToLower(strings.TrimSpace(cfg.Type))
	if providerType == "" {
		providerType = "openai-compatible"
	}

	return &openAICompatibleProvider{
		name:         name,
		providerType: providerType,
		baseURL:      strings.TrimRight(baseURL, "/"),
		apiKey:       cfg.APIKey,
		http:         httpClient,
		streamHttp:   streamHttpClient,
	}, nil
}

func (p *openAICompatibleProvider) ListModels(ctx context.Context) ([]*pb.ModelInfo, error) {
	endpoint, err := p.endpoint("models")
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, err
	}
	p.decorate(req)

	resp, err := p.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("list provider models for %s: %w", p.name, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 64*1024))
		return nil, fmt.Errorf("provider %s returned %s: %s", p.name, resp.Status, strings.TrimSpace(string(body)))
	}

	var payload openAIModelListResponse
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, fmt.Errorf("decode provider models for %s: %w", p.name, err)
	}

	models := make([]*pb.ModelInfo, 0, len(payload.Data))
	for _, model := range payload.Data {
		metadata := map[string]string{
			"provider":          p.name,
			"provider_model_id": model.ID,
			"source":            "provider",
			"provider_type":     p.providerType,
		}
		if model.OwnedBy != "" {
			metadata["owned_by"] = model.OwnedBy
		}

		models = append(models, &pb.ModelInfo{
			Id:        model.ID,
			Name:      model.ID,
			Path:      p.baseURL,
			SizeBytes: 0,
			Loaded:    false,
			Metadata:  metadata,
		})
	}

	return models, nil
}

func (p *openAICompatibleProvider) StreamInference(
	ctx context.Context,
	modelID string,
	req *pb.InferenceRequest,
	send func(*pb.InferenceResponse) error,
) error {
	endpoint, err := p.endpoint("chat/completions")
	if err != nil {
		return err
	}

	payload := map[string]any{
		"model":    modelID,
		"stream":   true,
		"messages": buildMessages(req),
	}
	for key, value := range req.GetParameters() {
		payload[key] = coerceParameterValue(value)
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("marshal inference payload: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	p.decorate(httpReq)

	resp, err := p.streamHttp.Do(httpReq)
	if err != nil {
		return fmt.Errorf("stream inference with provider %s: %w", p.name, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 300 {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 64*1024))
		return fmt.Errorf("provider %s returned %s: %s", p.name, resp.Status, strings.TrimSpace(string(raw)))
	}

	contentType := strings.ToLower(resp.Header.Get("Content-Type"))
	if strings.Contains(contentType, "text/event-stream") {
		return streamSSEResponse(resp.Body, p.name, modelID, send)
	}

	return streamJSONResponse(resp.Body, p.name, modelID, send)
}

func (p *openAICompatibleProvider) endpoint(resource string) (string, error) {
	base, err := url.Parse(p.baseURL)
	if err != nil {
		return "", err
	}

	cleanPath := strings.TrimRight(base.Path, "/")
	if strings.HasSuffix(cleanPath, "/v1") {
		base.Path = cleanPath + "/" + strings.TrimLeft(resource, "/")
	} else if cleanPath == "" {
		base.Path = "/v1/" + strings.TrimLeft(resource, "/")
	} else {
		base.Path = cleanPath + "/v1/" + strings.TrimLeft(resource, "/")
	}

	return base.String(), nil
}

func (p *openAICompatibleProvider) decorate(req *http.Request) {
	if p.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+p.apiKey)
	}
}

func streamLocalFallback(
	ctx context.Context,
	req *pb.InferenceRequest,
	send func(*pb.InferenceResponse) error,
) error {
	tokens := []string{"This", " is", " a", " local", " fallback", " response", "."}
	for i, token := range tokens {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		if err := send(&pb.InferenceResponse{
			Token:    token,
			Complete: i == len(tokens)-1,
			Metrics: map[string]string{
				"mode":  "local-fallback",
				"model": req.GetModelId(),
			},
		}); err != nil {
			return err
		}
		time.Sleep(25 * time.Millisecond)
	}

	return nil
}

func streamSSEResponse(
	body io.Reader,
	providerName, modelID string,
	send func(*pb.InferenceResponse) error,
) error {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	sentComplete := false
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, ":") {
			continue
		}
		if !strings.HasPrefix(line, "data:") {
			continue
		}

		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "[DONE]" {
			break
		}

		var chunk openAIStreamChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			return fmt.Errorf("decode streaming chunk: %w", err)
		}

		token := ""
		complete := false
		if len(chunk.Choices) > 0 {
			token = chunk.Choices[0].Delta.Content
			if token == "" {
				token = chunk.Choices[0].Text
			}
			complete = chunk.Choices[0].FinishReason != nil && *chunk.Choices[0].FinishReason != ""
		}

		metrics := usageMetrics(chunk.Usage, providerName, modelID)
		if token == "" && !complete && len(metrics) == 0 {
			continue
		}

		if err := send(&pb.InferenceResponse{
			Token:    token,
			Complete: complete,
			Metrics:  metrics,
		}); err != nil {
			return err
		}
		sentComplete = sentComplete || complete
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("read streaming response: %w", err)
	}

	if !sentComplete {
		return send(&pb.InferenceResponse{
			Complete: true,
			Metrics:  usageMetrics(nil, providerName, modelID),
		})
	}

	return nil
}

func streamJSONResponse(
	body io.Reader,
	providerName, modelID string,
	send func(*pb.InferenceResponse) error,
) error {
	var payload openAIChatResponse
	if err := json.NewDecoder(body).Decode(&payload); err != nil {
		return fmt.Errorf("decode provider response: %w", err)
	}

	token := ""
	complete := true
	if len(payload.Choices) > 0 {
		token = payload.Choices[0].Message.Content
		if token == "" {
			token = payload.Choices[0].Text
		}
		if payload.Choices[0].FinishReason != nil && *payload.Choices[0].FinishReason == "" {
			complete = false
		}
	}

	return send(&pb.InferenceResponse{
		Token:    token,
		Complete: complete,
		Metrics:  usageMetrics(payload.Usage, providerName, modelID),
	})
}

func buildMessages(req *pb.InferenceRequest) []map[string]string {
	messages := make([]map[string]string, 0, 2)
	if len(req.GetContextRefs()) > 0 {
		messages = append(messages, map[string]string{
			"role":    "system",
			"content": "Context references:\n- " + strings.Join(req.GetContextRefs(), "\n- "),
		})
	}
	messages = append(messages, map[string]string{
		"role":    "user",
		"content": req.GetPrompt(),
	})
	return messages
}

func usageMetrics(usage map[string]any, providerName, modelID string) map[string]string {
	metrics := map[string]string{
		"provider": providerName,
		"model":    modelID,
	}
	for key, value := range usage {
		metrics[key] = fmt.Sprint(value)
	}
	return metrics
}

func coerceParameterValue(value string) any {
	if parsed, err := strconv.ParseBool(value); err == nil {
		return parsed
	}
	if parsed, err := strconv.ParseInt(value, 10, 64); err == nil {
		return parsed
	}
	if parsed, err := strconv.ParseFloat(value, 64); err == nil {
		return parsed
	}
	return value
}

func providerModelIDToID(providerName, modelID string) string {
	return providerName + "/" + modelID
}

func splitProviderModelID(modelID string) (string, string, bool) {
	parts := strings.SplitN(modelID, "/", 2)
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
		return "", "", false
	}
	return parts[0], parts[1], true
}

func stripProviderPrefix(modelID, providerName string) string {
	if providerName == "" {
		return modelID
	}

	prefix := providerName + "/"
	if strings.HasPrefix(modelID, prefix) {
		return strings.TrimPrefix(modelID, prefix)
	}
	return modelID
}

func effectiveProviderModelID(model *Model) string {
	if model.ProviderModelID != "" {
		return model.ProviderModelID
	}
	return stripProviderPrefix(model.ID, model.Provider)
}

func cloneStringMap(input map[string]string) map[string]string {
	if input == nil {
		return nil
	}

	cloned := make(map[string]string, len(input))
	for key, value := range input {
		cloned[key] = value
	}
	return cloned
}

func modelStoreKey(model *pb.ModelInfo) string {
	if model == nil {
		return ""
	}
	if provider := model.GetMetadata()["provider"]; provider != "" {
		providerModelID := model.GetMetadata()["provider_model_id"]
		if providerModelID == "" {
			providerModelID = model.GetId()
		}
		return providerModelIDToID(provider, providerModelID)
	}
	return model.GetId()
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
