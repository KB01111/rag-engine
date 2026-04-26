package hub

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
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

	"github.com/google/uuid"
)

const (
	ErrorCodeInvalidRequest          = "invalid_request"
	ErrorCodeNotFound                = "not_found"
	ErrorCodeUnsupportedModelFormat  = "unsupported_model_format"
	ErrorCodeBackendUnavailable      = "backend_unavailable"
	ErrorCodeDownloadTooLarge        = "download_too_large"
	ErrorCodeHuggingFaceTokenMissing = "hf_token_required"

	DownloadStatusQueued      = "queued"
	DownloadStatusDownloading = "downloading"
	DownloadStatusCompleted   = "completed"
	DownloadStatusFailed      = "failed"
	DownloadStatusCanceled    = "canceled"
)

var (
	ErrInvalidRequest         = errors.New(ErrorCodeInvalidRequest)
	ErrNotFound               = errors.New(ErrorCodeNotFound)
	ErrUnsupportedModelFormat = errors.New(ErrorCodeUnsupportedModelFormat)
	ErrDownloadTooLarge       = errors.New(ErrorCodeDownloadTooLarge)
	ErrHuggingFaceToken       = errors.New(ErrorCodeHuggingFaceTokenMissing)
)

type Error struct {
	Code    string
	Message string
	Err     error
}

func (e *Error) Error() string {
	if e == nil {
		return ""
	}
	if e.Message != "" {
		return e.Message
	}
	return e.Code
}

func (e *Error) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

func NewError(code string, cause error, message string) error {
	if code == "" {
		code = ErrorCodeBackendUnavailable
	}
	if cause == nil {
		cause = errors.New(code)
	}
	if message == "" {
		message = cause.Error()
	}
	return &Error{Code: code, Message: message, Err: cause}
}

func Code(err error) string {
	if err == nil {
		return ""
	}
	var hubErr *Error
	if errors.As(err, &hubErr) {
		return hubErr.Code
	}
	switch {
	case errors.Is(err, ErrInvalidRequest):
		return ErrorCodeInvalidRequest
	case errors.Is(err, ErrNotFound):
		return ErrorCodeNotFound
	case errors.Is(err, ErrUnsupportedModelFormat):
		return ErrorCodeUnsupportedModelFormat
	case errors.Is(err, ErrDownloadTooLarge):
		return ErrorCodeDownloadTooLarge
	case errors.Is(err, ErrHuggingFaceToken):
		return ErrorCodeHuggingFaceTokenMissing
	default:
		return ErrorCodeBackendUnavailable
	}
}

func Message(err error) string {
	if err == nil {
		return ""
	}
	var hubErr *Error
	if errors.As(err, &hubErr) && hubErr.Message != "" {
		return hubErr.Message
	}
	return err.Error()
}

type Config struct {
	Endpoint             string
	ModelsPath           string
	MaxDownloadBytes     int64
	CompatibleExtensions []string
	HTTPClient           *http.Client
	UserAgent            string
}

type Service struct {
	endpoint             string
	modelsPath           string
	maxDownloadBytes     int64
	compatibleExtensions map[string]struct{}
	metadataClient       *http.Client
	streamingClient      *http.Client
	userAgent            string

	mu        sync.RWMutex
	downloads map[string]*downloadState
}

type SearchRequest struct {
	Query               string `json:"query,omitempty"`
	Author              string `json:"author,omitempty"`
	Sort                string `json:"sort,omitempty"`
	Limit               int    `json:"limit,omitempty"`
	Cursor              string `json:"cursor,omitempty"`
	IncludeIncompatible bool   `json:"include_incompatible,omitempty"`
}

type SearchResponse struct {
	Models         []Model `json:"models"`
	Query          string  `json:"query,omitempty"`
	Author         string  `json:"author,omitempty"`
	Sort           string  `json:"sort,omitempty"`
	Limit          int     `json:"limit"`
	CompatibleOnly bool    `json:"compatible_only"`
	NextCursor     string  `json:"next_cursor,omitempty"`
	HasMore        bool    `json:"has_more"`
}

type Model struct {
	ID              string   `json:"id"`
	Author          string   `json:"author,omitempty"`
	Private         bool     `json:"private"`
	Gated           bool     `json:"gated"`
	RequiresHFToken bool     `json:"requires_hf_token"`
	License         string   `json:"license,omitempty"`
	Downloads       int64    `json:"downloads,omitempty"`
	Likes           int64    `json:"likes,omitempty"`
	Tags            []string `json:"tags"`
	LastModified    string   `json:"last_modified,omitempty"`
	Files           []File   `json:"files"`
	CompatibleFiles []File   `json:"compatible_files"`
}

type File struct {
	Filename    string `json:"filename"`
	SizeBytes   int64  `json:"size_bytes,omitempty"`
	Extension   string `json:"extension"`
	DownloadURL string `json:"download_url,omitempty"`
	SHA         string `json:"sha,omitempty"`
}

type DownloadRequest struct {
	RepoID   string `json:"repo_id"`
	Filename string `json:"filename"`
	Revision string `json:"revision,omitempty"`
}

type Download struct {
	ID              string     `json:"id"`
	RepoID          string     `json:"repo_id"`
	Filename        string     `json:"filename"`
	Revision        string     `json:"revision"`
	Status          string     `json:"status"`
	Error           string     `json:"error,omitempty"`
	ErrorCode       string     `json:"error_code,omitempty"`
	TargetPath      string     `json:"target_path"`
	ManifestPath    string     `json:"manifest_path,omitempty"`
	ResolvedURL     string     `json:"resolved_url"`
	ETag            string     `json:"etag,omitempty"`
	SizeBytes       int64      `json:"size_bytes,omitempty"`
	DownloadedBytes int64      `json:"downloaded_bytes"`
	StartedAt       time.Time  `json:"started_at"`
	CompletedAt     *time.Time `json:"completed_at,omitempty"`
}

func (d Download) Terminal() bool {
	switch d.Status {
	case DownloadStatusCompleted, DownloadStatusFailed, DownloadStatusCanceled:
		return true
	default:
		return false
	}
}

type Manifest struct {
	Source       string    `json:"source"`
	RepoID       string    `json:"repo_id"`
	Filename     string    `json:"filename"`
	Revision     string    `json:"revision"`
	ResolvedURL  string    `json:"resolved_url"`
	ETag         string    `json:"etag,omitempty"`
	License      string    `json:"license,omitempty"`
	SizeBytes    int64     `json:"size_bytes,omitempty"`
	SHA          string    `json:"sha,omitempty"`
	DownloadedAt time.Time `json:"downloaded_at"`
}

type downloadState struct {
	Download
	cancel context.CancelFunc
}

type hfModel struct {
	ID           string         `json:"id"`
	ModelID      string         `json:"modelId"`
	Author       string         `json:"author"`
	Private      bool           `json:"private"`
	Gated        interface{}    `json:"gated"`
	Tags         []string       `json:"tags"`
	Downloads    int64          `json:"downloads"`
	Likes        int64          `json:"likes"`
	LastModified string         `json:"lastModified"`
	Siblings     []hfSibling    `json:"siblings"`
	CardData     map[string]any `json:"cardData"`
}

type hfSibling struct {
	RFilename string `json:"rfilename"`
	Size      int64  `json:"size"`
	LFS       *struct {
		Size int64  `json:"size"`
		OID  string `json:"oid"`
	} `json:"lfs"`
}

func NewService(cfg Config) *Service {
	endpoint := strings.TrimRight(strings.TrimSpace(cfg.Endpoint), "/")
	if endpoint == "" {
		endpoint = "https://huggingface.co"
	}

	// Create separate clients for metadata vs streaming
	var metadataClient, streamingClient *http.Client
	if cfg.HTTPClient != nil {
		// Use provided client as base for metadata
		metadataClient = cfg.HTTPClient
		// Create streaming client without total timeout
		streamingClient = &http.Client{
			Transport: cfg.HTTPClient.Transport,
		}
	} else {
		// Create default clients
		// Metadata client with short timeout for API calls
		metadataClient = &http.Client{
			Timeout: 30 * time.Second,
		}
		// Streaming client without total timeout for long downloads
		streamingClient = &http.Client{}
	}

	extensions := cfg.CompatibleExtensions
	if len(extensions) == 0 {
		extensions = []string{".gguf", ".ggml", ".bin"}
	}
	normalizedExtensions := make(map[string]struct{}, len(extensions))
	for _, ext := range extensions {
		ext = strings.ToLower(strings.TrimSpace(ext))
		if ext == "" {
			continue
		}
		if !strings.HasPrefix(ext, ".") {
			ext = "." + ext
		}
		normalizedExtensions[ext] = struct{}{}
	}

	userAgent := strings.TrimSpace(cfg.UserAgent)
	if userAgent == "" {
		userAgent = "ai-engine-huggingface-hub/1.0"
	}

	return &Service{
		endpoint:             endpoint,
		modelsPath:           cfg.ModelsPath,
		maxDownloadBytes:     cfg.MaxDownloadBytes,
		compatibleExtensions: normalizedExtensions,
		metadataClient:       metadataClient,
		streamingClient:      streamingClient,
		userAgent:            userAgent,
		downloads:            make(map[string]*downloadState),
	}
}

func (s *Service) Search(ctx context.Context, req SearchRequest) (SearchResponse, error) {
	req.Query = strings.TrimSpace(req.Query)
	req.Author = strings.TrimSpace(req.Author)
	req.Sort = strings.TrimSpace(req.Sort)
	req.Cursor = strings.TrimSpace(req.Cursor)
	limit := req.Limit
	if limit <= 0 {
		limit = 20
	}
	if limit > 100 {
		limit = 100
	}

	requestURL := ""
	if req.Cursor != "" {
		decodedURL, err := s.decodeCursor(req.Cursor)
		if err != nil {
			return SearchResponse{}, err
		}
		requestURL = decodedURL
	} else {
		searchURL, err := url.Parse(s.url("api", "models"))
		if err != nil {
			return SearchResponse{}, err
		}
		q := searchURL.Query()
		q.Set("full", "true")
		q.Set("limit", strconv.Itoa(limit))
		if req.Query != "" {
			q.Set("search", req.Query)
		}
		if req.Author != "" {
			q.Set("author", req.Author)
		}
		if req.Sort != "" {
			q.Set("sort", req.Sort)
		}
		searchURL.RawQuery = q.Encode()
		requestURL = searchURL.String()
	}

	var raw []hfModel
	headers, err := s.getJSONWithHeaders(ctx, requestURL, &raw)
	if err != nil {
		return SearchResponse{}, err
	}

	models := make([]Model, 0, len(raw))
	for _, hf := range raw {
		model := s.modelFromHF(hf)
		if !req.IncludeIncompatible && !model.compatibleForV1() {
			continue
		}
		models = append(models, model)
	}

	nextCursor, hasMore := s.nextCursor(headers.Get("Link"))
	return SearchResponse{
		Models:         models,
		Query:          req.Query,
		Author:         req.Author,
		Sort:           req.Sort,
		Limit:          limit,
		CompatibleOnly: !req.IncludeIncompatible,
		NextCursor:     nextCursor,
		HasMore:        hasMore,
	}, nil
}

func (s *Service) Model(ctx context.Context, repoID string) (Model, error) {
	repoID = strings.TrimSpace(repoID)
	if err := validateRepoID(repoID); err != nil {
		return Model{}, err
	}

	var raw hfModel
	if err := s.getJSON(ctx, s.url(append([]string{"api", "models"}, splitPath(repoID)...)...), &raw); err != nil {
		return Model{}, err
	}
	model := s.modelFromHF(raw)
	if model.ID == "" {
		model.ID = repoID
	}
	return model, nil
}

func (s *Service) StartDownload(ctx context.Context, req DownloadRequest) (Download, error) {
	req.RepoID = strings.TrimSpace(req.RepoID)
	req.Filename = strings.TrimSpace(req.Filename)
	req.Revision = strings.TrimSpace(req.Revision)
	if req.Revision == "" {
		req.Revision = "main"
	}

	if err := validateRepoID(req.RepoID); err != nil {
		return Download{}, err
	}
	if err := validateRemoteFilename(req.Filename); err != nil {
		return Download{}, err
	}
	if err := validateRevision(req.Revision); err != nil {
		return Download{}, err
	}
	if !s.compatibleFilename(req.Filename) {
		return Download{}, NewError(
			ErrorCodeUnsupportedModelFormat,
			ErrUnsupportedModelFormat,
			"This backend currently supports GGUF/GGML/BIN local runtime artifacts.",
		)
	}
	if strings.TrimSpace(s.modelsPath) == "" {
		return Download{}, NewError(ErrorCodeInvalidRequest, ErrInvalidRequest, "runtime models path is required")
	}

	targetPath := filepath.Join(s.modelsPath, localFilename(req.RepoID, req.Filename))
	manifestPath := targetPath + ".hf.json"
	resolvedURL := s.resolveURL(req.RepoID, req.Revision, req.Filename)

	if info, err := os.Stat(targetPath); err == nil && !info.IsDir() {
		now := time.Now().UTC()
		manifest := Manifest{
			Source:       "huggingface",
			RepoID:       req.RepoID,
			Filename:     req.Filename,
			Revision:     req.Revision,
			ResolvedURL:  resolvedURL,
			SizeBytes:    info.Size(),
			DownloadedAt: now,
		}

		// Fetch model metadata to populate License, ETag and SHA
		ctxWithTimeout, ctxCancel := context.WithTimeout(ctx, 10*time.Second)
		defer ctxCancel()
		model, err := s.Model(ctxWithTimeout, req.RepoID)
		if err == nil {
			manifest.License = model.License
			// Find the matching file to get SHA and ETag
			for _, file := range model.CompatibleFiles {
				if file.Filename == req.Filename {
					manifest.SHA = file.SHA
					break
				}
			}
			// Fetch ETag via HEAD request
			etag, _, headErr := s.headDownload(ctxWithTimeout, resolvedURL)
			if headErr == nil {
				manifest.ETag = etag
			}
		}

		if _, err := os.Stat(manifestPath); os.IsNotExist(err) {
			if err := writeJSONFile(manifestPath, manifest); err != nil {
				return Download{}, NewError(ErrorCodeBackendUnavailable, err, "failed to write Hugging Face manifest")
			}
		}
		download := Download{
			ID:              uuid.NewString(),
			RepoID:          req.RepoID,
			Filename:        req.Filename,
			Revision:        req.Revision,
			Status:          DownloadStatusCompleted,
			TargetPath:      targetPath,
			ManifestPath:    manifestPath,
			ResolvedURL:     resolvedURL,
			SizeBytes:       info.Size(),
			DownloadedBytes: info.Size(),
			StartedAt:       now,
			CompletedAt:     &now,
		}
		s.storeDownload(download)
		return download, nil
	} else if err != nil && !os.IsNotExist(err) {
		return Download{}, NewError(ErrorCodeBackendUnavailable, err, "failed to inspect local model path")
	}

	s.mu.Lock()
	// Check for existing download with same target path
	for _, existing := range s.downloads {
		if existing.TargetPath == targetPath {
			if !existing.Terminal() {
				// Return in-progress download
				download := existing.Download
				s.mu.Unlock()
				return download, nil
			}
			// Return existing terminal download instead of creating a new one
			download := existing.Download
			s.mu.Unlock()
			return download, nil
		}
	}
	ctxDownload, cancel := context.WithCancel(context.Background())
	download := Download{
		ID:           uuid.NewString(),
		RepoID:       req.RepoID,
		Filename:     req.Filename,
		Revision:     req.Revision,
		Status:       DownloadStatusQueued,
		TargetPath:   targetPath,
		ManifestPath: manifestPath,
		ResolvedURL:  resolvedURL,
		StartedAt:    time.Now().UTC(),
	}
	s.downloads[download.ID] = &downloadState{Download: download, cancel: cancel}
	s.cleanupOldDownloads()
	s.mu.Unlock()

	go s.runDownload(ctxDownload, download.ID)
	return download, nil
}

func (s *Service) ListDownloads() []Download {
	s.mu.RLock()
	defer s.mu.RUnlock()

	downloads := make([]Download, 0, len(s.downloads))
	for _, state := range s.downloads {
		downloads = append(downloads, state.Download)
	}
	sort.Slice(downloads, func(i, j int) bool {
		return downloads[i].StartedAt.After(downloads[j].StartedAt)
	})
	return downloads
}

func (s *Service) GetDownload(id string) (Download, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	state, ok := s.downloads[id]
	if !ok {
		return Download{}, false
	}
	return state.Download, true
}

func (s *Service) CancelDownload(id string) (Download, error) {
	s.mu.Lock()
	state, ok := s.downloads[id]
	if !ok {
		s.mu.Unlock()
		return Download{}, NewError(ErrorCodeNotFound, ErrNotFound, "download not found")
	}
	if state.Status == DownloadStatusQueued || state.Status == DownloadStatusDownloading {
		if state.cancel != nil {
			state.cancel()
		}
		now := time.Now().UTC()
		state.Status = DownloadStatusCanceled
		state.CompletedAt = &now
		download := state.Download
		s.mu.Unlock()
		return download, nil
	}
	download := state.Download
	s.mu.Unlock()
	return download, nil
}

func (s *Service) runDownload(ctx context.Context, id string) {
	download, ok := s.GetDownload(id)
	if !ok {
		return
	}

	// Get the cancel function and ensure it's called when download completes
	s.mu.RLock()
	state, ok := s.downloads[id]
	s.mu.RUnlock()
	if !ok {
		return
	}
	if state.cancel != nil {
		defer state.cancel()
	}

	s.updateDownload(id, func(d *Download) {
		d.Status = DownloadStatusDownloading
	})

	model, err := s.Model(ctx, download.RepoID)
	if err != nil {
		s.failDownload(ctx, id, err)
		return
	}
	if model.Private || model.Gated {
		s.failDownload(ctx, id, NewError(ErrorCodeHuggingFaceTokenMissing, ErrHuggingFaceToken, "This Hugging Face model requires access approval or a token, which is not supported in v1."))
		return
	}
	var selected File
	for _, file := range model.CompatibleFiles {
		if file.Filename == download.Filename {
			selected = file
			break
		}
	}
	if selected.Filename == "" {
		s.failDownload(ctx, id, NewError(ErrorCodeUnsupportedModelFormat, ErrUnsupportedModelFormat, "This backend currently supports GGUF/GGML/BIN local runtime artifacts."))
		return
	}

	etag, sizeBytes, err := s.headDownload(ctx, download.ResolvedURL)
	if err != nil {
		s.failDownload(ctx, id, err)
		return
	}
	if sizeBytes <= 0 {
		sizeBytes = selected.SizeBytes
	}
	if s.maxDownloadBytes > 0 && sizeBytes > s.maxDownloadBytes {
		s.failDownload(ctx, id, NewError(ErrorCodeDownloadTooLarge, ErrDownloadTooLarge, "download exceeds configured maximum size"))
		return
	}

	s.updateDownload(id, func(d *Download) {
		d.ETag = etag
		d.SizeBytes = sizeBytes
	})

	if err := s.downloadFile(ctx, id, model.License, etag, selected.SHA, sizeBytes); err != nil {
		s.failDownload(ctx, id, err)
		return
	}
}

func (s *Service) headDownload(ctx context.Context, resolvedURL string) (string, int64, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodHead, resolvedURL, nil)
	if err != nil {
		return "", 0, err
	}
	s.prepareRequest(req)

	resp, err := s.metadataClient.Do(req)
	if err != nil {
		return "", 0, NewError(ErrorCodeBackendUnavailable, err, "failed to inspect Hugging Face model file")
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotFound {
		return "", 0, NewError(ErrorCodeNotFound, ErrNotFound, "model file not found")
	}
	if resp.StatusCode >= 400 {
		return "", 0, NewError(ErrorCodeBackendUnavailable, fmt.Errorf("huggingface head failed with status %d", resp.StatusCode), "failed to inspect Hugging Face model file")
	}

	size := resp.ContentLength
	if size <= 0 {
		if value := strings.TrimSpace(resp.Header.Get("Content-Length")); value != "" {
			if parsed, err := strconv.ParseInt(value, 10, 64); err == nil {
				size = parsed
			}
		}
	}
	return resp.Header.Get("ETag"), size, nil
}

func (s *Service) downloadFile(ctx context.Context, id string, license string, etag string, sha string, sizeBytes int64) error {
	download, ok := s.GetDownload(id)
	if !ok {
		return NewError(ErrorCodeNotFound, ErrNotFound, "download not found")
	}
	if err := os.MkdirAll(filepath.Dir(download.TargetPath), 0o755); err != nil {
		return NewError(ErrorCodeBackendUnavailable, err, "failed to create models directory")
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, download.ResolvedURL, nil)
	if err != nil {
		return err
	}
	s.prepareRequest(req)
	resp, err := s.streamingClient.Do(req)
	if err != nil {
		return NewError(ErrorCodeBackendUnavailable, err, "failed to download Hugging Face model file")
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotFound {
		return NewError(ErrorCodeNotFound, ErrNotFound, "model file not found")
	}
	if resp.StatusCode >= 400 {
		return NewError(ErrorCodeBackendUnavailable, fmt.Errorf("huggingface download failed with status %d", resp.StatusCode), "failed to download Hugging Face model file")
	}

	tmpPath := download.TargetPath + ".partial"
	file, err := os.Create(tmpPath)
	if err != nil {
		return NewError(ErrorCodeBackendUnavailable, err, "failed to create temporary model file")
	}

	var written int64
	hash := sha256.New()
	buf := make([]byte, 256*1024)
	for {
		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			w, writeErr := file.Write(buf[:n])
			if writeErr != nil {
				_ = file.Close()
				_ = os.Remove(tmpPath)
				return NewError(ErrorCodeBackendUnavailable, writeErr, "failed to write model file")
			}
			if w != n {
				_ = file.Close()
				_ = os.Remove(tmpPath)
				return NewError(ErrorCodeBackendUnavailable, io.ErrShortWrite, "failed to write model file")
			}
			hash.Write(buf[:n])
			written += int64(w)
			s.updateDownload(id, func(d *Download) {
				d.DownloadedBytes = written
			})
		}
		if readErr == nil {
			continue
		}
		if errors.Is(readErr, io.EOF) {
			break
		}
		// Handle context cancellation/timeout
		if errors.Is(readErr, context.Canceled) || errors.Is(readErr, context.DeadlineExceeded) {
			_ = file.Close()
			_ = os.Remove(tmpPath)
			return nil
		}
		_ = file.Close()
		_ = os.Remove(tmpPath)
		return NewError(ErrorCodeBackendUnavailable, readErr, "failed to download Hugging Face model file")
	}

	if err := file.Close(); err != nil {
		_ = os.Remove(tmpPath)
		return NewError(ErrorCodeBackendUnavailable, err, "failed to finalize temporary model file")
	}

	// Verify size if provided
	if sizeBytes > 0 && written != sizeBytes {
		_ = os.Remove(tmpPath)
		return NewError(ErrorCodeBackendUnavailable, fmt.Errorf("size mismatch: expected %d, got %d", sizeBytes, written), "download size verification failed")
	}

	// Verify SHA256 if provided
	if sha != "" {
		computedHash := hex.EncodeToString(hash.Sum(nil))
		// SHA might be prefixed with "sha256:" in some formats
		expectedSHA := strings.TrimPrefix(strings.ToLower(sha), "sha256:")
		if computedHash != expectedSHA {
			_ = os.Remove(tmpPath)
			return NewError(ErrorCodeBackendUnavailable, fmt.Errorf("sha256 mismatch: expected %s, got %s", expectedSHA, computedHash), "download integrity verification failed")
		}
	}

	if err := os.Rename(tmpPath, download.TargetPath); err != nil {
		_ = os.Remove(tmpPath)
		return NewError(ErrorCodeBackendUnavailable, err, "failed to finalize model file")
	}

	now := time.Now().UTC()
	manifest := Manifest{
		Source:       "huggingface",
		RepoID:       download.RepoID,
		Filename:     download.Filename,
		Revision:     download.Revision,
		ResolvedURL:  download.ResolvedURL,
		ETag:         etag,
		License:      license,
		SizeBytes:    written,
		SHA:          sha,
		DownloadedAt: now,
	}
	if sizeBytes > 0 {
		manifest.SizeBytes = sizeBytes
	}
	if err := writeJSONFile(download.ManifestPath, manifest); err != nil {
		return NewError(ErrorCodeBackendUnavailable, err, "failed to write Hugging Face manifest")
	}

	// Check-and-set: don't overwrite canceled state
	updated := s.updateDownloadConditional(id, func(d *Download) bool {
		// If already in a terminal canceled state, don't promote to completed
		if d.Status == DownloadStatusCanceled {
			return false
		}
		d.Status = DownloadStatusCompleted
		d.ETag = etag
		d.SizeBytes = manifest.SizeBytes
		d.DownloadedBytes = written
		d.CompletedAt = &now
		return true
	})

	// If update was rejected (canceled), clean up downloaded files
	if !updated {
		_ = os.Remove(download.TargetPath)
		_ = os.Remove(download.ManifestPath)
	}
	return nil
}

func (s *Service) failDownload(ctx context.Context, id string, err error) {
	if err == nil {
		return
	}
	status := DownloadStatusFailed
	if errors.Is(ctx.Err(), context.Canceled) {
		status = DownloadStatusCanceled
	}
	now := time.Now().UTC()
	s.updateDownload(id, func(d *Download) {
		d.Status = status
		d.Error = Message(err)
		d.ErrorCode = Code(err)
		d.CompletedAt = &now
	})
	if download, ok := s.GetDownload(id); ok {
		_ = os.Remove(download.TargetPath + ".partial")
	}
}

func (s *Service) storeDownload(download Download) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.downloads[download.ID] = &downloadState{Download: download}
	if download.Terminal() {
		s.cleanupOldDownloads()
	}
}

// cleanupOldDownloads removes old terminal downloads, keeping the 100 most recent.
// Must be called with s.mu held.
func (s *Service) cleanupOldDownloads() {
	const maxDownloads = 100

	if len(s.downloads) <= maxDownloads {
		return
	}

	// Collect all terminal downloads
	type downloadEntry struct {
		id        string
		startedAt time.Time
	}
	var terminals []downloadEntry
	for id, state := range s.downloads {
		if state.Terminal() {
			terminals = append(terminals, downloadEntry{id: id, startedAt: state.StartedAt})
		}
	}

	// If we don't have too many terminal downloads, nothing to clean
	activeCount := len(s.downloads) - len(terminals)
	if activeCount+maxDownloads >= len(s.downloads) {
		return
	}

	// Sort by StartedAt (oldest first)
	sort.Slice(terminals, func(i, j int) bool {
		return terminals[i].startedAt.Before(terminals[j].startedAt)
	})

	// Remove oldest terminal downloads to get down to max
	toRemove := len(s.downloads) - maxDownloads
	for i := 0; i < toRemove && i < len(terminals); i++ {
		delete(s.downloads, terminals[i].id)
	}
}

func (s *Service) updateDownload(id string, fn func(*Download)) Download {
	s.mu.Lock()
	defer s.mu.Unlock()
	state, ok := s.downloads[id]
	if !ok {
		return Download{}
	}
	fn(&state.Download)
	return state.Download
}

func (s *Service) updateDownloadConditional(id string, fn func(*Download) bool) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	state, ok := s.downloads[id]
	if !ok {
		return false
	}
	return fn(&state.Download)
}

func (s *Service) getJSON(ctx context.Context, requestURL string, out any) error {
	_, err := s.getJSONWithHeaders(ctx, requestURL, out)
	return err
}

func (s *Service) getJSONWithHeaders(ctx context.Context, requestURL string, out any) (http.Header, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, requestURL, nil)
	if err != nil {
		return nil, err
	}
	s.prepareRequest(req)
	resp, err := s.metadataClient.Do(req)
	if err != nil {
		return nil, NewError(ErrorCodeBackendUnavailable, err, "failed to contact Hugging Face Hub")
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotFound {
		return nil, NewError(ErrorCodeNotFound, ErrNotFound, "Hugging Face model not found")
	}
	if resp.StatusCode >= 400 {
		return nil, NewError(ErrorCodeBackendUnavailable, fmt.Errorf("huggingface API failed with status %d", resp.StatusCode), "failed to read Hugging Face Hub response")
	}
	if err := json.NewDecoder(resp.Body).Decode(out); err != nil {
		return nil, NewError(ErrorCodeBackendUnavailable, err, "failed to parse Hugging Face Hub response")
	}
	return resp.Header.Clone(), nil
}

func (s *Service) prepareRequest(req *http.Request) {
	req.Header.Set("Accept", "application/json")
	if s.userAgent != "" {
		req.Header.Set("User-Agent", s.userAgent)
	}
}

func (s *Service) modelFromHF(hf hfModel) Model {
	id := hf.ID
	if id == "" {
		id = hf.ModelID
	}
	gated := isGated(hf.Gated)
	model := Model{
		ID:              id,
		Author:          hf.Author,
		Private:         hf.Private,
		Gated:           gated,
		RequiresHFToken: hf.Private || gated,
		License:         licenseFromMetadata(hf.CardData, hf.Tags),
		Downloads:       hf.Downloads,
		Likes:           hf.Likes,
		Tags:            hf.Tags,
		LastModified:    hf.LastModified,
	}
	model.Files = make([]File, 0, len(hf.Siblings))
	model.CompatibleFiles = make([]File, 0, len(hf.Siblings))
	for _, sibling := range hf.Siblings {
		if sibling.RFilename == "" {
			continue
		}
		size := sibling.Size
		if size <= 0 && sibling.LFS != nil {
			size = sibling.LFS.Size
		}
		file := File{
			Filename:    sibling.RFilename,
			SizeBytes:   size,
			Extension:   strings.ToLower(filepath.Ext(sibling.RFilename)),
			DownloadURL: s.resolveURL(id, "main", sibling.RFilename),
		}
		if sibling.LFS != nil {
			file.SHA = sibling.LFS.OID
		}
		model.Files = append(model.Files, file)
		if s.compatibleFilename(file.Filename) && validateRemoteFilename(file.Filename) == nil {
			model.CompatibleFiles = append(model.CompatibleFiles, file)
		}
	}
	return model
}

func (m Model) compatibleForV1() bool {
	return !m.Private && !m.Gated && len(m.CompatibleFiles) > 0 && strings.TrimSpace(m.License) != ""
}

func (s *Service) compatibleFilename(filename string) bool {
	ext := strings.ToLower(filepath.Ext(filename))
	_, ok := s.compatibleExtensions[ext]
	return ok
}

func (s *Service) resolveURL(repoID string, revision string, filename string) string {
	parts := make([]string, 0, 8)
	parts = append(parts, splitPath(repoID)...)
	parts = append(parts, "resolve")
	parts = append(parts, splitPath(revision)...)
	parts = append(parts, splitPath(filename)...)
	return s.url(parts...)
}

func (s *Service) nextCursor(linkHeader string) (string, bool) {
	nextURL := nextLinkURL(linkHeader)
	if nextURL == "" {
		return "", false
	}
	resolved, err := s.resolveHubURL(nextURL)
	if err != nil {
		return "", false
	}
	if !s.matchesEndpoint(resolved) {
		return "", false
	}
	return base64.RawURLEncoding.EncodeToString([]byte(resolved.String())), true
}

func (s *Service) decodeCursor(cursor string) (string, error) {
	raw, err := base64.RawURLEncoding.DecodeString(cursor)
	if err != nil {
		return "", NewError(ErrorCodeInvalidRequest, ErrInvalidRequest, "cursor is invalid")
	}
	decoded, err := url.Parse(string(raw))
	if err != nil {
		return "", NewError(ErrorCodeInvalidRequest, ErrInvalidRequest, "cursor is invalid")
	}
	resolved, err := s.resolveHubURL(decoded.String())
	if err != nil || !s.matchesEndpoint(resolved) {
		return "", NewError(ErrorCodeInvalidRequest, ErrInvalidRequest, "cursor is invalid")
	}
	return resolved.String(), nil
}

func (s *Service) resolveHubURL(value string) (*url.URL, error) {
	base, err := url.Parse(s.endpoint)
	if err != nil {
		return nil, err
	}
	nextURL, err := url.Parse(strings.TrimSpace(value))
	if err != nil {
		return nil, err
	}
	return base.ResolveReference(nextURL), nil
}

func (s *Service) matchesEndpoint(candidate *url.URL) bool {
	if candidate == nil {
		return false
	}
	base, err := url.Parse(s.endpoint)
	if err != nil {
		return false
	}
	return strings.EqualFold(candidate.Scheme, base.Scheme) && strings.EqualFold(candidate.Host, base.Host)
}

func nextLinkURL(linkHeader string) string {
	for _, part := range strings.Split(linkHeader, ",") {
		sections := strings.Split(part, ";")
		if len(sections) < 2 {
			continue
		}
		rawURL := strings.TrimSpace(sections[0])
		if !strings.HasPrefix(rawURL, "<") || !strings.HasSuffix(rawURL, ">") {
			continue
		}
		for _, section := range sections[1:] {
			key, value, ok := strings.Cut(strings.TrimSpace(section), "=")
			if !ok {
				continue
			}
			if strings.EqualFold(strings.TrimSpace(key), "rel") && strings.EqualFold(strings.Trim(strings.TrimSpace(value), `"`), "next") {
				return strings.TrimSuffix(strings.TrimPrefix(rawURL, "<"), ">")
			}
		}
	}
	return ""
}

func (s *Service) url(parts ...string) string {
	base, err := url.Parse(s.endpoint)
	if err != nil {
		return s.endpoint
	}
	existing := splitPath(strings.Trim(base.EscapedPath(), "/"))
	all := make([]string, 0, len(existing)+len(parts))
	all = append(all, existing...)
	all = append(all, parts...)
	escaped := make([]string, 0, len(all))
	for _, part := range all {
		if part == "" {
			continue
		}
		escaped = append(escaped, url.PathEscape(part))
	}
	base.Path = "/" + strings.Join(escaped, "/")
	base.RawQuery = ""
	return base.String()
}

func validateRepoID(repoID string) error {
	if repoID == "" {
		return NewError(ErrorCodeInvalidRequest, ErrInvalidRequest, "repo_id is required")
	}
	if strings.Contains(repoID, "\\") || strings.HasPrefix(repoID, "/") || strings.HasSuffix(repoID, "/") {
		return NewError(ErrorCodeInvalidRequest, ErrInvalidRequest, "repo_id is invalid")
	}
	for _, part := range strings.Split(repoID, "/") {
		if part == "" || part == "." || part == ".." {
			return NewError(ErrorCodeInvalidRequest, ErrInvalidRequest, "repo_id is invalid")
		}
	}
	return nil
}

func validateRemoteFilename(filename string) error {
	if filename == "" {
		return NewError(ErrorCodeInvalidRequest, ErrInvalidRequest, "filename is required")
	}
	if strings.Contains(filename, "\\") || strings.HasPrefix(filename, "/") {
		return NewError(ErrorCodeInvalidRequest, ErrInvalidRequest, "filename is invalid")
	}
	for _, part := range strings.Split(filename, "/") {
		if part == "" || part == "." || part == ".." {
			return NewError(ErrorCodeInvalidRequest, ErrInvalidRequest, "filename is invalid")
		}
	}
	return nil
}

func validateRevision(revision string) error {
	if revision == "" {
		return NewError(ErrorCodeInvalidRequest, ErrInvalidRequest, "revision is required")
	}
	if strings.Contains(revision, "\\") || strings.HasPrefix(revision, "/") {
		return NewError(ErrorCodeInvalidRequest, ErrInvalidRequest, "revision is invalid")
	}
	for _, part := range strings.Split(revision, "/") {
		if part == "" || part == "." || part == ".." {
			return NewError(ErrorCodeInvalidRequest, ErrInvalidRequest, "revision is invalid")
		}
	}
	return nil
}

func localFilename(repoID string, filename string) string {
	return sanitizePathForFilename(repoID) + "__" + sanitizePathForFilename(filename)
}

func sanitizePathForFilename(value string) string {
	parts := splitPath(value)
	for i, part := range parts {
		parts[i] = sanitizeFilenamePart(part)
	}
	return strings.Join(parts, "__")
}

func sanitizeFilenamePart(value string) string {
	var b strings.Builder
	for _, r := range value {
		switch {
		case r >= 'a' && r <= 'z':
			b.WriteRune(r)
		case r >= 'A' && r <= 'Z':
			b.WriteRune(r)
		case r >= '0' && r <= '9':
			b.WriteRune(r)
		case r == '.', r == '-', r == '_':
			b.WriteRune(r)
		default:
			b.WriteRune('_')
		}
	}
	if b.Len() == 0 {
		return "file"
	}
	return b.String()
}

func splitPath(value string) []string {
	trimmed := strings.Trim(value, "/")
	if trimmed == "" {
		return nil
	}
	return strings.Split(trimmed, "/")
}

func isGated(value any) bool {
	switch gated := value.(type) {
	case bool:
		return gated
	case string:
		normalized := strings.ToLower(strings.TrimSpace(gated))
		return normalized != "" && normalized != "false" && normalized != "none"
	default:
		return false
	}
}

func licenseFromMetadata(cardData map[string]any, tags []string) string {
	if cardData != nil {
		if value, ok := cardData["license"]; ok {
			switch license := value.(type) {
			case string:
				if trimmed := strings.TrimSpace(license); trimmed != "" {
					return trimmed
				}
			case []any:
				values := make([]string, 0, len(license))
				for _, item := range license {
					if text, ok := item.(string); ok && strings.TrimSpace(text) != "" {
						values = append(values, strings.TrimSpace(text))
					}
				}
				if len(values) > 0 {
					return strings.Join(values, ",")
				}
			}
		}
	}
	for _, tag := range tags {
		tag = strings.TrimSpace(tag)
		if strings.HasPrefix(tag, "license:") {
			return strings.TrimPrefix(tag, "license:")
		}
	}
	return ""
}

func writeJSONFile(path string, value any) error {
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0o644)
}

func readJSONFile(path string, value any) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, value)
}
