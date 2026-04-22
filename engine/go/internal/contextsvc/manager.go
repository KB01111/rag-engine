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

type Manager struct {
	mu     sync.RWMutex
	cfg    Config
	client *http.Client
	cmd    *exec.Cmd
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

func (m *Manager) Start(ctx context.Context) error {
	if !m.cfg.Enabled {
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
	var status HealthStatus
	if err := m.doJSON(ctx, http.MethodGet, "/health", nil, &status); err != nil {
		return nil, err
	}
	return &status, nil
}

func (m *Manager) Status(ctx context.Context) (*StatusResponse, error) {
	var status StatusResponse
	if err := m.doJSON(ctx, http.MethodGet, "/v1/status", nil, &status); err != nil {
		return nil, err
	}
	return &status, nil
}

func (m *Manager) UpsertResource(ctx context.Context, req UpsertResourceRequest) (*UpsertResourceResponse, error) {
	var resp UpsertResourceResponse
	if req.Layer == "" {
		req.Layer = LayerL2
	}
	if err := m.doJSON(ctx, http.MethodPost, "/v1/resources", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

func (m *Manager) DeleteResource(ctx context.Context, uri string) error {
	path := "/v1/resources?uri=" + url.QueryEscape(uri)
	return m.doJSON(ctx, http.MethodDelete, path, nil, nil)
}

func (m *Manager) Search(ctx context.Context, req SearchRequest) (*SearchResponse, error) {
	var resp SearchResponse
	if err := m.doJSON(ctx, http.MethodPost, "/v1/search", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

func (m *Manager) ListResources(ctx context.Context) (*ListResourcesResponse, error) {
	var resp ListResourcesResponse
	if err := m.doJSON(ctx, http.MethodGet, "/v1/resources", nil, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

func (m *Manager) SyncWorkspace(ctx context.Context, req WorkspaceSyncRequest) (*WorkspaceSyncResponse, error) {
	var resp WorkspaceSyncResponse
	if err := m.doJSON(ctx, http.MethodPost, "/v1/workspaces/sync", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

func (m *Manager) ListFiles(ctx context.Context, req FileListRequest) (*FileListResponse, error) {
	var resp FileListResponse
	if err := m.doJSON(ctx, http.MethodPost, "/v1/files/list", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

func (m *Manager) ReadFile(ctx context.Context, req FileReadRequest) (*FileReadResponse, error) {
	var resp FileReadResponse
	if err := m.doJSON(ctx, http.MethodPost, "/v1/files/read", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

func (m *Manager) WriteFile(ctx context.Context, req FileWriteRequest) (*FileWriteResponse, error) {
	var resp FileWriteResponse
	if err := m.doJSON(ctx, http.MethodPost, "/v1/files/write", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

func (m *Manager) DeleteFile(ctx context.Context, req FileDeleteRequest) (*FileDeleteResponse, error) {
	var resp FileDeleteResponse
	if err := m.doJSON(ctx, http.MethodPost, "/v1/files/delete", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

func (m *Manager) MoveFile(ctx context.Context, req FileMoveRequest) (*FileMoveResponse, error) {
	var resp FileMoveResponse
	if err := m.doJSON(ctx, http.MethodPost, "/v1/files/move", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

func (m *Manager) AppendSession(ctx context.Context, req SessionAppendRequest) (*SessionResponse, error) {
	var resp SessionResponse
	if err := m.doJSON(ctx, http.MethodPost, "/v1/sessions/append", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

func (m *Manager) GetSession(ctx context.Context, sessionID string) (*SessionResponse, error) {
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