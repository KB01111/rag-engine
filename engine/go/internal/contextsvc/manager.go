package contextsvc

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	pb "github.com/ai-engine/proto/go"
	"google.golang.org/protobuf/types/known/emptypb"
)

type Layer string

const (
	LayerL0 Layer = "l0"
	LayerL1 Layer = "l1"
	LayerL2 Layer = "l2"
)

type Config struct {
	Enabled          bool
	BaseURL          string
	BinaryPath       string
	DataDir          string
	AutoStart        bool
	StartupTimeout   time.Duration
	ManagedRoots     []string
	OpenVikingURL    string
	OpenVikingAPIKey string
}

type Backend interface {
	Start(context.Context) error
	Stop(context.Context) error
	Enabled() bool
	Readiness(context.Context) (*HealthStatus, error)
	Status(context.Context) (*StatusResponse, error)
	UpsertResource(context.Context, UpsertResourceRequest) (*UpsertResourceResponse, error)
	DeleteResource(context.Context, string) error
	Search(context.Context, SearchRequest) (*SearchResponse, error)
	ListResources(context.Context) (*ListResourcesResponse, error)
}

type DaemonContextClient interface {
	GetContextStatus(context.Context, *emptypb.Empty) (*pb.ContextStatus, error)
	ListResources(context.Context, *emptypb.Empty) (*pb.ContextResourceList, error)
	UpsertResource(context.Context, *pb.ContextUpsertResourceRequest) (*pb.ContextUpsertResourceResponse, error)
	DeleteResource(context.Context, *pb.ContextDeleteResourceRequest) (*emptypb.Empty, error)
	SearchContext(context.Context, *pb.ContextSearchRequest) (*pb.ContextSearchResponse, error)
	SyncWorkspace(context.Context, *pb.ContextWorkspaceSyncRequest) (*pb.ContextWorkspaceSyncResponse, error)
	ListFiles(context.Context, *pb.ContextFileListRequest) (*pb.ContextFileListResponse, error)
	ReadFile(context.Context, *pb.ContextFileReadRequest) (*pb.ContextFileReadResponse, error)
	WriteFile(context.Context, *pb.ContextFileWriteRequest) (*pb.ContextFileWriteResponse, error)
	DeleteFile(context.Context, *pb.ContextFileDeleteRequest) (*pb.ContextFileDeleteResponse, error)
	MoveFile(context.Context, *pb.ContextFileMoveRequest) (*pb.ContextFileMoveResponse, error)
	AppendSession(context.Context, *pb.ContextSessionAppendRequest) (*pb.ContextSessionHistory, error)
	GetSession(context.Context, *pb.ContextSessionGetRequest) (*pb.ContextSessionHistory, error)
}

type Manager struct {
	mu           sync.RWMutex
	cfg          Config
	client       *http.Client
	cmd          *exec.Cmd
	daemonClient DaemonContextClient
}

type HealthStatus struct {
	Status  string `json:"status"`
	Ready   bool   `json:"ready"`
	Message string `json:"message,omitempty"`
}

type StatusResponse struct {
	DocumentCount  int64    `json:"document_count"`
	ChunkCount     int64    `json:"chunk_count"`
	IndexSizeBytes int64    `json:"index_size_bytes"`
	EmbeddingModel string   `json:"embedding_model,omitempty"`
	Ready          bool     `json:"ready"`
	ManagedRoots   []string `json:"managed_roots,omitempty"`
}

type Resource struct {
	URI      string            `json:"uri"`
	Title    string            `json:"title,omitempty"`
	Layer    Layer             `json:"layer,omitempty"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

type UpsertResourceRequest struct {
	URI         string            `json:"uri"`
	Title       string            `json:"title,omitempty"`
	Content     string            `json:"content"`
	Layer       Layer             `json:"layer,omitempty"`
	Metadata    map[string]string `json:"metadata,omitempty"`
	PreviousURI string            `json:"previous_uri,omitempty"`
}

type UpsertResourceResponse struct {
	Resource      Resource `json:"resource"`
	ChunksIndexed int32    `json:"chunks_indexed"`
}

type SearchRequest struct {
	Query    string            `json:"query"`
	ScopeURI string            `json:"scope_uri,omitempty"`
	TopK     int               `json:"top_k,omitempty"`
	Filters  map[string]string `json:"filters,omitempty"`
	Layer    Layer             `json:"layer,omitempty"`
	Rerank   *bool             `json:"rerank,omitempty"`
}

type SearchHit struct {
	URI        string            `json:"uri"`
	DocumentID string            `json:"document_id,omitempty"`
	ChunkText  string            `json:"chunk_text,omitempty"`
	Score      float32           `json:"score"`
	Metadata   map[string]string `json:"metadata,omitempty"`
	Layer      Layer             `json:"layer,omitempty"`
}

type SearchResponse struct {
	Results     []SearchHit `json:"results"`
	QueryTimeMs float64     `json:"query_time_ms"`
}

type ListResourcesResponse struct {
	Resources []Resource `json:"resources"`
}

type WorkspaceSyncRequest struct {
	Root string `json:"root"`
	Path string `json:"path,omitempty"`
}

type WorkspaceSyncResponse struct {
	Root               string  `json:"root"`
	Prefix             *string `json:"prefix,omitempty"`
	IndexedResources   int32   `json:"indexed_resources"`
	ReindexedResources int32   `json:"reindexed_resources"`
	DeletedResources   int32   `json:"deleted_resources"`
	SkippedFiles       int32   `json:"skipped_files"`
}

type FileListRequest struct {
	Root string `json:"root"`
	Path string `json:"path,omitempty"`
}

type FileListEntry struct {
	Name      string `json:"name"`
	Path      string `json:"path"`
	IsDir     bool   `json:"is_dir"`
	SizeBytes int64  `json:"size_bytes,omitempty"`
	Version   *int64 `json:"version,omitempty"`
}

type FileListResponse struct {
	Entries []FileListEntry `json:"entries"`
}

type FileReadRequest struct {
	Root string `json:"root"`
	Path string `json:"path"`
}

type FileReadResponse struct {
	Path    string `json:"path"`
	Content string `json:"content"`
	Version int64  `json:"version,omitempty"`
}

type FileWriteRequest struct {
	Root    string `json:"root"`
	Path    string `json:"path"`
	Content string `json:"content"`
	Version *int64 `json:"version,omitempty"`
}

type FileWriteResponse struct {
	Path    string `json:"path"`
	Version int64  `json:"version,omitempty"`
}

type FileDeleteRequest struct {
	Root    string `json:"root"`
	Path    string `json:"path"`
	Version *int64 `json:"version,omitempty"`
}

type FileDeleteResponse struct {
	Path    string `json:"path"`
	Deleted bool   `json:"deleted"`
}

type FileMoveRequest struct {
	Root     string `json:"root"`
	FromPath string `json:"from_path"`
	ToPath   string `json:"to_path"`
	Version  *int64 `json:"version,omitempty"`
}

type FileMoveResponse struct {
	FromPath string `json:"from_path"`
	ToPath   string `json:"to_path"`
	Version  int64  `json:"version,omitempty"`
}

type SessionAppendRequest struct {
	SessionID string            `json:"session_id"`
	Role      string            `json:"role"`
	Content   string            `json:"content"`
	Metadata  map[string]string `json:"metadata,omitempty"`
}

type SessionEntry struct {
	SessionID string            `json:"session_id"`
	Role      string            `json:"role"`
	Content   string            `json:"content"`
	Metadata  map[string]string `json:"metadata,omitempty"`
	CreatedAt int64             `json:"created_at,omitempty"`
}

type SessionResponse struct {
	SessionID string         `json:"session_id"`
	Entries   []SessionEntry `json:"entries"`
}

func NewManager(cfg Config) *Manager {
	timeout := cfg.StartupTimeout
	if timeout <= 0 {
		timeout = 20 * time.Second
	}

	baseURL := strings.TrimRight(cfg.BaseURL, "/")
	if baseURL == "" {
		baseURL = "http://127.0.0.1:9191"
	}

	cfg.BaseURL = baseURL
	cfg.StartupTimeout = timeout

	return &Manager{
		cfg:    cfg,
		client: &http.Client{},
	}
}

func (m *Manager) Enabled() bool {
	return m.cfg.Enabled
}

func (m *Manager) SetDaemonContextClient(client DaemonContextClient) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.daemonClient = client
}

func (m *Manager) daemon() DaemonContextClient {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.daemonClient
}

func (m *Manager) Start(ctx context.Context) error {
	if !m.cfg.Enabled {
		return nil
	}
	if m.daemon() != nil {
		return nil
	}

	if m.cfg.AutoStart && m.cfg.BinaryPath != "" {
		if err := m.startProcess(); err != nil {
			return err
		}
	}

	deadline := time.Now().Add(m.cfg.StartupTimeout)
	for {
		health, err := m.Readiness(ctx)
		if err == nil && health.Ready {
			return nil
		}
		if time.Now().After(deadline) {
			if err != nil {
				return fmt.Errorf("context service not ready: %w", err)
			}
			return fmt.Errorf("context service not ready before timeout")
		}
		time.Sleep(250 * time.Millisecond)
	}
}

func (m *Manager) startProcess() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.cmd != nil && m.cmd.Process != nil {
		return nil
	}

	parsedURL, err := url.Parse(m.cfg.BaseURL)
	if err != nil {
		return fmt.Errorf("invalid context service url: %w", err)
	}

	host := parsedURL.Hostname()
	port := parsedURL.Port()
	if host == "" {
		host = "127.0.0.1"
	}
	if port == "" {
		port = "9191"
	}

	args := []string{
		"--host", host,
		"--port", port,
	}
	if m.cfg.DataDir != "" {
		args = append(args, "--data-dir", m.cfg.DataDir)
	}
	for _, root := range m.cfg.ManagedRoots {
		args = append(args, "--managed-root", root)
	}
	if m.cfg.OpenVikingURL != "" {
		args = append(args, "--openviking-url", m.cfg.OpenVikingURL)
	}

	binaryPath := m.resolveBinaryPath()
	cmd := exec.Command(binaryPath, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if m.cfg.OpenVikingAPIKey != "" {
		cmd.Env = append(os.Environ(), "CONTEXT_OPENVIKING_API_KEY="+m.cfg.OpenVikingAPIKey)
	}
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start context service: %w", err)
	}

	m.cmd = cmd
	go func() {
		_ = cmd.Wait()
	}()

	return nil
}

func (m *Manager) resolveBinaryPath() string {
	cleaned := filepath.Clean(m.cfg.BinaryPath)
	if _, err := os.Stat(cleaned); err == nil {
		return cleaned
	}
	if runtime.GOOS == "windows" {
		withExt := cleaned + ".exe"
		if _, err := os.Stat(withExt); err == nil {
			return withExt
		}
	}
	return cleaned
}

func (m *Manager) Stop(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.daemonClient != nil {
		return nil
	}

	if m.cmd == nil || m.cmd.Process == nil {
		return nil
	}

	done := make(chan error, 1)
	go func() {
		done <- m.cmd.Process.Kill()
	}()

	select {
	case err := <-done:
		m.cmd = nil
		return err
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (m *Manager) Readiness(ctx context.Context) (*HealthStatus, error) {
	if daemon := m.daemon(); daemon != nil {
		status, err := daemon.GetContextStatus(ctx, &emptypb.Empty{})
		if err != nil {
			return nil, err
		}
		return &HealthStatus{
			Status:  "ok",
			Ready:   status.GetReady(),
			Message: "",
		}, nil
	}
	var status HealthStatus
	if err := m.doJSON(ctx, http.MethodGet, "/health", nil, &status); err != nil {
		return nil, err
	}
	return &status, nil
}

func (m *Manager) Status(ctx context.Context) (*StatusResponse, error) {
	if daemon := m.daemon(); daemon != nil {
		status, err := daemon.GetContextStatus(ctx, &emptypb.Empty{})
		if err != nil {
			return nil, err
		}
		return statusFromProto(status), nil
	}
	var status StatusResponse
	if err := m.doJSON(ctx, http.MethodGet, "/v1/status", nil, &status); err != nil {
		return nil, err
	}
	return &status, nil
}

func (m *Manager) UpsertResource(ctx context.Context, req UpsertResourceRequest) (*UpsertResourceResponse, error) {
	if req.Layer == "" {
		req.Layer = LayerL2
	}
	if daemon := m.daemon(); daemon != nil {
		resp, err := daemon.UpsertResource(ctx, &pb.ContextUpsertResourceRequest{
			Uri:         req.URI,
			Title:       req.Title,
			Content:     req.Content,
			Layer:       layerToProto(req.Layer),
			Metadata:    req.Metadata,
			PreviousUri: req.PreviousURI,
		})
		if err != nil {
			return nil, err
		}
		return upsertResponseFromProto(resp), nil
	}
	var resp UpsertResourceResponse
	if err := m.doJSON(ctx, http.MethodPost, "/v1/resources", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

func (m *Manager) DeleteResource(ctx context.Context, uri string) error {
	if daemon := m.daemon(); daemon != nil {
		_, err := daemon.DeleteResource(ctx, &pb.ContextDeleteResourceRequest{Uri: uri})
		return err
	}
	path := "/v1/resources?uri=" + url.QueryEscape(uri)
	return m.doJSON(ctx, http.MethodDelete, path, nil, nil)
}

func (m *Manager) Search(ctx context.Context, req SearchRequest) (*SearchResponse, error) {
	if daemon := m.daemon(); daemon != nil {
		protoReq := &pb.ContextSearchRequest{
			Query:    req.Query,
			ScopeUri: req.ScopeURI,
			TopK:     int32(req.TopK),
			Filters:  req.Filters,
			Layer:    layerToProto(req.Layer),
			Rerank:   req.Rerank,
		}
		resp, err := daemon.SearchContext(ctx, protoReq)
		if err != nil {
			return nil, err
		}
		return searchResponseFromProto(resp), nil
	}
	var resp SearchResponse
	if err := m.doJSON(ctx, http.MethodPost, "/v1/search", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

func (m *Manager) ListResources(ctx context.Context) (*ListResourcesResponse, error) {
	if daemon := m.daemon(); daemon != nil {
		resp, err := daemon.ListResources(ctx, &emptypb.Empty{})
		if err != nil {
			return nil, err
		}
		return listResourcesFromProto(resp), nil
	}
	var resp ListResourcesResponse
	if err := m.doJSON(ctx, http.MethodGet, "/v1/resources", nil, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

func (m *Manager) SyncWorkspace(ctx context.Context, req WorkspaceSyncRequest) (*WorkspaceSyncResponse, error) {
	if daemon := m.daemon(); daemon != nil {
		resp, err := daemon.SyncWorkspace(ctx, &pb.ContextWorkspaceSyncRequest{
			Root: req.Root,
			Path: req.Path,
		})
		if err != nil {
			return nil, err
		}
		return workspaceSyncFromProto(resp), nil
	}
	var resp WorkspaceSyncResponse
	if err := m.doJSON(ctx, http.MethodPost, "/v1/workspaces/sync", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

func (m *Manager) ListFiles(ctx context.Context, req FileListRequest) (*FileListResponse, error) {
	if daemon := m.daemon(); daemon != nil {
		resp, err := daemon.ListFiles(ctx, &pb.ContextFileListRequest{
			Root: req.Root,
			Path: req.Path,
		})
		if err != nil {
			return nil, err
		}
		return fileListFromProto(resp), nil
	}
	var resp FileListResponse
	if err := m.doJSON(ctx, http.MethodPost, "/v1/files/list", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

func (m *Manager) ReadFile(ctx context.Context, req FileReadRequest) (*FileReadResponse, error) {
	if daemon := m.daemon(); daemon != nil {
		resp, err := daemon.ReadFile(ctx, &pb.ContextFileReadRequest{
			Root: req.Root,
			Path: req.Path,
		})
		if err != nil {
			return nil, err
		}
		return &FileReadResponse{
			Path:    resp.GetPath(),
			Content: resp.GetContent(),
			Version: resp.GetVersion(),
		}, nil
	}
	var resp FileReadResponse
	if err := m.doJSON(ctx, http.MethodPost, "/v1/files/read", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

func (m *Manager) WriteFile(ctx context.Context, req FileWriteRequest) (*FileWriteResponse, error) {
	if daemon := m.daemon(); daemon != nil {
		resp, err := daemon.WriteFile(ctx, &pb.ContextFileWriteRequest{
			Root:    req.Root,
			Path:    req.Path,
			Content: req.Content,
			Version: req.Version,
		})
		if err != nil {
			return nil, err
		}
		return &FileWriteResponse{
			Path:    resp.GetPath(),
			Version: resp.GetVersion(),
		}, nil
	}
	var resp FileWriteResponse
	if err := m.doJSON(ctx, http.MethodPost, "/v1/files/write", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

func (m *Manager) DeleteFile(ctx context.Context, req FileDeleteRequest) (*FileDeleteResponse, error) {
	if daemon := m.daemon(); daemon != nil {
		resp, err := daemon.DeleteFile(ctx, &pb.ContextFileDeleteRequest{
			Root:    req.Root,
			Path:    req.Path,
			Version: req.Version,
		})
		if err != nil {
			return nil, err
		}
		return &FileDeleteResponse{
			Path:    resp.GetPath(),
			Deleted: resp.GetDeleted(),
		}, nil
	}
	var resp FileDeleteResponse
	if err := m.doJSON(ctx, http.MethodPost, "/v1/files/delete", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

func (m *Manager) MoveFile(ctx context.Context, req FileMoveRequest) (*FileMoveResponse, error) {
	if daemon := m.daemon(); daemon != nil {
		resp, err := daemon.MoveFile(ctx, &pb.ContextFileMoveRequest{
			Root:     req.Root,
			FromPath: req.FromPath,
			ToPath:   req.ToPath,
			Version:  req.Version,
		})
		if err != nil {
			return nil, err
		}
		return &FileMoveResponse{
			FromPath: resp.GetFromPath(),
			ToPath:   resp.GetToPath(),
			Version:  resp.GetVersion(),
		}, nil
	}
	var resp FileMoveResponse
	if err := m.doJSON(ctx, http.MethodPost, "/v1/files/move", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

func (m *Manager) AppendSession(ctx context.Context, req SessionAppendRequest) (*SessionResponse, error) {
	if daemon := m.daemon(); daemon != nil {
		resp, err := daemon.AppendSession(ctx, &pb.ContextSessionAppendRequest{
			SessionId: req.SessionID,
			Role:      req.Role,
			Content:   req.Content,
			Metadata:  req.Metadata,
		})
		if err != nil {
			return nil, err
		}
		return sessionResponseFromProto(resp), nil
	}
	var resp SessionResponse
	if err := m.doJSON(ctx, http.MethodPost, "/v1/sessions/append", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

func (m *Manager) GetSession(ctx context.Context, sessionID string) (*SessionResponse, error) {
	if daemon := m.daemon(); daemon != nil {
		resp, err := daemon.GetSession(ctx, &pb.ContextSessionGetRequest{SessionId: sessionID})
		if err != nil {
			return nil, err
		}
		return sessionResponseFromProto(resp), nil
	}
	var resp SessionResponse
	if err := m.doJSON(ctx, http.MethodGet, "/v1/sessions/"+url.PathEscape(sessionID), nil, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

func (m *Manager) doJSON(ctx context.Context, method, path string, payload any, out any) error {
	var body io.Reader
	if payload != nil {
		raw, err := json.Marshal(payload)
		if err != nil {
			return fmt.Errorf("marshal request: %w", err)
		}
		body = bytes.NewReader(raw)
	}

	req, err := http.NewRequestWithContext(ctx, method, m.cfg.BaseURL+path, body)
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}
	if payload != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	resp, err := m.client.Do(req)
	if err != nil {
		return fmt.Errorf("context service request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 300 {
		raw, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("context service returned %s: %s", resp.Status, strings.TrimSpace(string(raw)))
	}

	if out == nil || resp.StatusCode == http.StatusNoContent {
		return nil
	}

	if err := json.NewDecoder(resp.Body).Decode(out); err != nil {
		return fmt.Errorf("decode response: %w", err)
	}
	return nil
}

// layerToProto converts a contextsvc Layer into the corresponding protobuf ContextLayer.
// It maps LayerL0, LayerL1, and LayerL2 to their proto equivalents and returns CONTEXT_LAYER_UNSPECIFIED for any other value.
func layerToProto(layer Layer) pb.ContextLayer {
	switch layer {
	case LayerL0:
		return pb.ContextLayer_CONTEXT_LAYER_L0
	case LayerL1:
		return pb.ContextLayer_CONTEXT_LAYER_L1
	case LayerL2:
		return pb.ContextLayer_CONTEXT_LAYER_L2
	default:
		return pb.ContextLayer_CONTEXT_LAYER_UNSPECIFIED
	}
}

// statusFromProto converts a protobuf ContextStatus message into a StatusResponse DTO.
// If the input is nil it returns an empty StatusResponse.
func statusFromProto(status *pb.ContextStatus) *StatusResponse {
	if status == nil {
		return &StatusResponse{}
	}
	return &StatusResponse{
		DocumentCount:  status.GetDocumentCount(),
		ChunkCount:     status.GetChunkCount(),
		IndexSizeBytes: status.GetIndexSizeBytes(),
		EmbeddingModel: status.GetEmbeddingModel(),
		Ready:          status.GetReady(),
		ManagedRoots:   status.GetManagedRoots(),
	}
}

// listResourcesFromProto converts a protobuf ContextResourceList into a ListResourcesResponse.
// If resp is nil, it returns an empty ListResourcesResponse. Each proto resource is mapped to a
// Resource with URI, Title, Layer (cast from the proto enum), and Metadata.
func listResourcesFromProto(resp *pb.ContextResourceList) *ListResourcesResponse {
	out := &ListResourcesResponse{}
	if resp == nil {
		return out
	}
	out.Resources = make([]Resource, 0, len(resp.GetResources()))
	for _, resource := range resp.GetResources() {
		out.Resources = append(out.Resources, Resource{
			URI:      resource.GetUri(),
			Title:    resource.GetTitle(),
			Layer:    Layer(resource.GetLayer()),
			Metadata: resource.GetMetadata(),
		})
	}
	return out
}

// upsertResponseFromProto converts a protobuf ContextUpsertResourceResponse into an UpsertResourceResponse.
// If resp is nil it returns an empty UpsertResourceResponse. The resulting value contains the mapped
// Resource (zero-value if the proto resource is absent) and the ChunksIndexed count from the proto.
func upsertResponseFromProto(resp *pb.ContextUpsertResourceResponse) *UpsertResourceResponse {
	if resp == nil {
		return &UpsertResourceResponse{}
	}
	var resource Resource
	if resp.GetResource() != nil {
		resource = Resource{
			URI:      resp.GetResource().GetUri(),
			Title:    resp.GetResource().GetTitle(),
			Layer:    Layer(resp.GetResource().GetLayer()),
			Metadata: resp.GetResource().GetMetadata(),
		}
	}
	return &UpsertResourceResponse{
		Resource:      resource,
		ChunksIndexed: resp.GetChunksIndexed(),
	}
}

// searchResponseFromProto converts a protobuf ContextSearchResponse into a SearchResponse.
// If resp is nil it returns an empty SearchResponse. The returned value contains Results
// populated from the proto results (URI, DocumentID, ChunkText, Score, Metadata, and Layer)
// and QueryTimeMs copied from the proto's QueryTimeMs.
func searchResponseFromProto(resp *pb.ContextSearchResponse) *SearchResponse {
	out := &SearchResponse{}
	if resp == nil {
		return out
	}
	out.Results = make([]SearchHit, 0, len(resp.GetResults()))
	for _, result := range resp.GetResults() {
		out.Results = append(out.Results, SearchHit{
			URI:        result.GetUri(),
			DocumentID: result.GetDocumentId(),
			ChunkText:  result.GetChunkText(),
			Score:      result.GetScore(),
			Metadata:   result.GetMetadata(),
			Layer:      Layer(result.GetLayer()),
		})
	}
	out.QueryTimeMs = resp.GetQueryTimeMs()
	return out
}

// workspaceSyncFromProto converts a protobuf ContextWorkspaceSyncResponse into a WorkspaceSyncResponse.
// If resp is nil, it returns an empty WorkspaceSyncResponse.
func workspaceSyncFromProto(resp *pb.ContextWorkspaceSyncResponse) *WorkspaceSyncResponse {
	if resp == nil {
		return &WorkspaceSyncResponse{}
	}
	return &WorkspaceSyncResponse{
		Root:               resp.GetRoot(),
		Prefix:             resp.Prefix,
		IndexedResources:   resp.GetIndexedResources(),
		ReindexedResources: resp.GetReindexedResources(),
		DeletedResources:   resp.GetDeletedResources(),
		SkippedFiles:       resp.GetSkippedFiles(),
	}
}

// fileListFromProto converts a protobuf ContextFileListResponse into a FileListResponse.
// If resp is nil it returns an empty FileListResponse.
// Each proto entry is mapped to a FileListEntry with Name, Path, IsDir, and SizeBytes.
// The proto entry Version is converted to a *int64 when present; otherwise the resulting Version is nil.
func fileListFromProto(resp *pb.ContextFileListResponse) *FileListResponse {
	out := &FileListResponse{}
	if resp == nil {
		return out
	}
	out.Entries = make([]FileListEntry, 0, len(resp.GetEntries()))
	for _, entry := range resp.GetEntries() {
		var version *int64
		if entry != nil && entry.Version != nil {
			value := entry.GetVersion()
			version = &value
		}
		out.Entries = append(out.Entries, FileListEntry{
			Name:      entry.GetName(),
			Path:      entry.GetPath(),
			IsDir:     entry.GetIsDir(),
			SizeBytes: entry.GetSizeBytes(),
			Version:   version,
		})
	}
	return out
}

// sessionResponseFromProto converts a protobuf ContextSessionHistory into a SessionResponse.
// If resp is nil it returns an empty SessionResponse. Each proto entry is mapped to a
// SessionEntry; when an entry's created_at timestamp is present it is converted to
// milliseconds since the Unix epoch and stored in CreatedAt.
func sessionResponseFromProto(resp *pb.ContextSessionHistory) *SessionResponse {
	out := &SessionResponse{}
	if resp == nil {
		return out
	}
	out.SessionID = resp.GetSessionId()
	out.Entries = make([]SessionEntry, 0, len(resp.GetEntries()))
	for _, entry := range resp.GetEntries() {
		var createdAt int64
		if entry.GetCreatedAt() != nil {
			createdAt = entry.GetCreatedAt().AsTime().UnixMilli()
		}
		out.Entries = append(out.Entries, SessionEntry{
			SessionID: entry.GetSessionId(),
			Role:      entry.GetRole(),
			Content:   entry.GetContent(),
			Metadata:  entry.GetMetadata(),
			CreatedAt: createdAt,
		})
	}
	return out
}
